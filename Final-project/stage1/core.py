"""Deterministic orchestration helpers for Stage 1.

This module intentionally avoids dataset-specific parsing. It samples rows,
renders prompts, executes generated extractors, validates JSONL, normalizes
candidate records, and builds Stage 2 text records.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_CANDIDATE_FIELDS = [
    "timestamp",
    "host",
    "user",
    "event_id",
    "event_type",
    "process_name",
    "process_path",
    "command_line",
    "parent_process_name",
    "parent_command_line",
    "process_id",
    "parent_process_id",
    "source_ip",
    "dest_ip",
    "dest_port",
    "raw_message",
    "source_line",
]

DEFAULT_RENDERED_SAMPLE_CAP = 25
DEFAULT_PLANNER_SAMPLE_SUMMARY_CAP = 12
DEFAULT_FIRST_RENDERED_ROWS = 5
DEFAULT_STRING_TRUNCATE_CHARS = 500
EVENT_ID_PATHS = [
    "EventID",
    "EventId",
    "event_id",
    "event.id",
    "event.code",
]

EVIDENCE_FIELDS = [
    "event_uid",
    "dataset",
    "scenario",
    "source_file",
    "source_line",
]

STAGE2_FIELDS = EVIDENCE_FIELDS + [
    "timestamp",
    "host",
    "user",
    "event_id",
    "event_type",
    "text_for_embedding",
]


@dataclass
class ValidationResult:
    ok: bool
    total_lines: int
    valid_records: int
    errors: list[dict[str, Any]]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_available_profiles() -> list[str]:
    profiles_dir = repo_root() / "stage1" / "profiles"
    return sorted(path.stem for path in profiles_dir.glob("*.json"))


def list_available_tasks() -> list[str]:
    tasks_dir = repo_root() / "stage1" / "tasks"
    return sorted(path.stem for path in tasks_dir.glob("*.json"))


def resolve_profile_path(dataset: str | None = None, profile_path: str | Path | None = None) -> Path:
    if profile_path:
        path = Path(profile_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset profile not found: {path}")
        return path
    if not dataset:
        available = ", ".join(list_available_profiles())
        raise ValueError(f"Provide --profile or --dataset. Available datasets: {available}")
    path = repo_root() / "stage1" / "profiles" / f"{dataset}.json"
    if not path.is_file():
        available = ", ".join(list_available_profiles())
        raise ValueError(f"Unknown dataset alias '{dataset}'. Available datasets: {available}")
    return path


def resolve_optional_profile_path(
    dataset: str | None = None,
    profile_path: str | Path | None = None,
) -> Path | None:
    if not dataset and not profile_path:
        return None
    return resolve_profile_path(dataset=dataset, profile_path=profile_path)


def resolve_task_path(task_name: str | None = None, task_path: str | Path | None = None) -> Path:
    if task_path:
        path = Path(task_path)
        if not path.is_file():
            raise FileNotFoundError(f"Task spec not found: {path}")
        return path
    if not task_name:
        available = ", ".join(list_available_tasks())
        raise ValueError(f"Provide --task or --task-name. Available tasks: {available}")
    path = repo_root() / "stage1" / "tasks" / f"{task_name}.json"
    if not path.is_file():
        available = ", ".join(list_available_tasks())
        raise ValueError(f"Unknown task alias '{task_name}'. Available tasks: {available}")
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, Any, str | None]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                yield line_no, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_no, None, str(exc)


def read_sample_lines(input_path: str | Path, limit: int = 500) -> list[str]:
    rows: list[str] = []
    with Path(input_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(line.rstrip("\n"))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def get_nested_value(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def extract_event_id(obj: dict[str, Any]) -> str | None:
    for path in EVENT_ID_PATHS:
        value = get_nested_value(obj, path)
        if value is not None and value != "":
            return str(value)
    return None


def event_id_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        event_id = extract_event_id(row)
        if event_id is None:
            continue
        counts[event_id] = counts.get(event_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def inspect_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    sample_limit: int = 500,
    profile_path: str | Path | None = None,
    dataset_label: str | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    input_file = Path(input_path)
    output.mkdir(parents=True, exist_ok=True)
    profile_file = Path(profile_path) if profile_path else None
    profile = load_json(profile_file) if profile_file else None
    sample_lines = read_sample_lines(input_file, sample_limit)
    sample_rows_path = output / "sample_rows.jsonl"
    with sample_rows_path.open("w", encoding="utf-8") as f:
        for row in sample_lines:
            f.write(row + "\n")

    parsed = []
    invalid_json = 0
    top_level_keys: dict[str, int] = {}
    nested_paths: dict[str, int] = {}
    for raw in sample_lines:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            invalid_json += 1
            continue
        if not isinstance(obj, dict):
            invalid_json += 1
            continue
        parsed.append(obj)
        for key in obj:
            top_level_keys[key] = top_level_keys.get(key, 0) + 1
        for path in flatten_keys(obj):
            nested_paths[path] = nested_paths.get(path, 0) + 1

    summary = {
        "input": str(input_file),
        "input_file_name": input_file.name,
        "input_suffix": input_file.suffix,
        "dataset_label": dataset_label,
        "sample_strategy": "first_non_empty_lines",
        "sample_policy": f"first_{sample_limit}_non_empty_lines",
        "sampled_rows": len(sample_lines),
        "valid_json_rows": len(parsed),
        "invalid_json_rows": invalid_json,
        "inferred_format": "ndjson" if parsed else "text",
        "top_level_keys": sorted(top_level_keys),
        "common_nested_paths": [
            key for key, _ in sorted(nested_paths.items(), key=lambda item: (-item[1], item[0]))[:50]
        ],
        "event_id_distribution": event_id_distribution(parsed),
        "selected_sample_strategy": None,
        "selected_sample_reason": None,
        "selected_sample_count": 0,
        "rendered_sample_cap": DEFAULT_RENDERED_SAMPLE_CAP,
    }
    if profile_file:
        summary["optional_profile_source"] = str(profile_file)
    summary_path = output / "inferred_format_summary.json"
    write_json(summary_path, summary)

    draft = {
        "debug_artifact": True,
        "required_input": False,
        "dataset_name": dataset_label or input_file.stem or "unknown_dataset",
        "format": summary["inferred_format"],
        "description": "Auto-generated draft from sampled rows. This is not required by the default pipeline.",
        "inferred_from": str(input_path),
        "sampled_rows": len(sample_lines),
        "valid_json_rows": len(parsed),
        "invalid_json_rows": invalid_json,
        "top_level_keys": summary["top_level_keys"],
        "inferred_common_paths": summary["common_nested_paths"],
    }
    if profile_file and profile:
        draft["optional_profile_source"] = str(profile_file)
        draft["optional_profile_hints"] = profile
    draft_path = output / "dataset_profile_draft.json"
    write_json(draft_path, draft)
    return {
        "sample_rows": sample_rows_path,
        "summary": summary_path,
        "profile_draft": draft_path,
    }


def flatten_keys(obj: Any, prefix: str = "") -> list[str]:
    if not isinstance(obj, dict):
        return []
    keys: list[str] = []
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.append(path)
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, path))
    return keys


def validate_dataset_profile(profile: dict[str, Any]) -> None:
    required = ["dataset_name", "format", "description"]
    missing = [field for field in required if not profile.get(field)]
    if missing:
        raise ValueError(f"dataset profile missing required fields: {', '.join(missing)}")


def validate_task_spec(task: dict[str, Any]) -> None:
    required = ["task_name", "goal", "target_event_types", "required_fields", "output_format"]
    missing = [field for field in required if not task.get(field)]
    if missing:
        raise ValueError(f"task spec missing required fields: {', '.join(missing)}")
    if task["output_format"] != "jsonl":
        raise ValueError("Stage 1 task output_format must be jsonl")


def validate_extraction_plan(plan: dict[str, Any]) -> None:
    required = ["dataset_summary", "available_event_types", "recommended_tasks", "not_recommended_tasks"]
    missing = [field for field in required if field not in plan]
    if missing:
        raise ValueError(f"extraction plan missing required fields: {', '.join(missing)}")
    if not isinstance(plan["dataset_summary"], str) or not plan["dataset_summary"].strip():
        raise ValueError("extraction plan dataset_summary must be a non-empty string")
    for field in ["available_event_types", "recommended_tasks", "not_recommended_tasks"]:
        if not isinstance(plan[field], list):
            raise ValueError(f"extraction plan {field} must be a list")
    for index, task in enumerate(plan["recommended_tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"recommended_tasks[{index}] must be an object")
        required_task = ["task_name", "priority", "reason", "target_event_types", "task_keywords", "required_fields"]
        missing_task = [field for field in required_task if field not in task]
        if missing_task:
            raise ValueError(f"recommended_tasks[{index}] missing fields: {', '.join(missing_task)}")
        if not isinstance(task["priority"], int):
            raise ValueError(f"recommended_tasks[{index}].priority must be an integer")
    if not plan["recommended_tasks"]:
        raise ValueError("extraction plan must include at least one recommended task")


TASK_FIELD_ALIASES = {
    "src_ip": "source_ip",
    "src_port": "source_port",
    "uid": "source_event_id",
    "request_id": "source_event_id",
    "record_id": "source_event_id",
}


def canonical_task_fields(fields: Iterable[str]) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for field in fields:
        value = TASK_FIELD_ALIASES.get(field, field)
        if value in seen:
            continue
        seen.add(value)
        canonical.append(value)
    return canonical


def build_dynamic_task_spec(plan: dict[str, Any]) -> dict[str, Any]:
    validate_extraction_plan(plan)
    task = max(plan["recommended_tasks"], key=lambda item: item["priority"])
    dynamic_task = {
        "task_name": task["task_name"],
        "goal": task.get("goal") or task.get("reason") or f"Extract {task['task_name']} events from security logs.",
        "target_event_types": task["target_event_types"],
        "task_keywords": task["task_keywords"],
        "required_fields": canonical_task_fields(task["required_fields"]),
        "output_format": "jsonl",
        "planner_reason": task.get("reason"),
        "planner_priority": task.get("priority"),
    }
    validate_task_spec(dynamic_task)
    return dynamic_task


def normalize_keyword(text: str) -> str:
    return text.replace("_", " ").replace("-", " ").lower()


def derive_task_keywords(task: dict[str, Any]) -> list[str]:
    raw_keywords: list[str] = []
    for value in task.get("task_keywords", []):
        if isinstance(value, str):
            raw_keywords.append(value)
    for key in ["task_name", "goal"]:
        value = task.get(key)
        if isinstance(value, str):
            raw_keywords.append(value)
    for key in ["target_event_types", "required_fields"]:
        for value in task.get(key, []):
            if isinstance(value, str):
                if key == "required_fields" and not any(
                    token in normalize_keyword(value)
                    for token in ["process", "command", "parent", "image", "credential", "network", "source", "dest"]
                ):
                    continue
                raw_keywords.append(value)

    keywords: list[str] = []
    seen: set[str] = set()
    for keyword in raw_keywords:
        normalized = normalize_keyword(keyword).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(normalized)
    return keywords


def flatten_text_values(obj: Any, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        values: list[str] = []
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            values.append(path)
            values.extend(flatten_text_values(value, path))
        return values
    if isinstance(obj, list):
        values = []
        for value in obj:
            values.extend(flatten_text_values(value, prefix))
        return values
    if obj is None:
        return []
    return [str(obj)]


def row_keyword_score(row: dict[str, Any], keywords: list[str]) -> int:
    return len(row_matched_keywords(row, keywords))


def event_id_keyword_value(keyword: str) -> str | None:
    compact = keyword.replace(" ", "")
    if not compact.startswith("eventid"):
        return None
    value = compact.removeprefix("eventid")
    return value if value.isdigit() else None


def row_matched_keywords(row: dict[str, Any], keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    haystack = normalize_keyword(" ".join(flatten_text_values(row)))
    matched: list[str] = []
    for keyword in keywords:
        if event_id_keyword_value(keyword) is not None:
            continue
        if keyword in haystack:
            matched.append(keyword)
    event_id = extract_event_id(row)
    if event_id:
        for keyword in keywords:
            if event_id_keyword_value(keyword) == event_id and keyword not in matched:
                matched.append(keyword)
    return matched


def truncate_value(value: Any, max_string_chars: int) -> Any:
    if isinstance(value, dict):
        return {key: truncate_value(nested, max_string_chars) for key, nested in value.items()}
    if isinstance(value, list):
        return [truncate_value(item, max_string_chars) for item in value]
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + "...TRUNCATED..."
    return value


def parse_sample_rows(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_no, obj, error in iter_jsonl(path):
        if error or not isinstance(obj, dict):
            continue
        rows.append((line_no, obj))
    return rows


def select_prompt_sample_rows(
    sample_rows_path: str | Path,
    task: dict[str, Any],
    cap: int = DEFAULT_RENDERED_SAMPLE_CAP,
    first_rows: int = DEFAULT_FIRST_RENDERED_ROWS,
    max_string_chars: int = DEFAULT_STRING_TRUNCATE_CHARS,
) -> tuple[str, dict[str, Any]]:
    rows = parse_sample_rows(sample_rows_path)
    keywords = derive_task_keywords(task)
    selected: list[tuple[int, dict[str, Any]]] = []
    selected_indexes: set[int] = set()
    selection_reasons: dict[int, list[str]] = {}
    reasons: list[str] = []

    def add_row(index: int, row: dict[str, Any], reason: str) -> None:
        if index in selected_indexes:
            row_reasons = selection_reasons.setdefault(index, [])
            if reason not in row_reasons:
                row_reasons.append(reason)
            if reason not in reasons:
                reasons.append(reason)
            return
        if len(selected) >= cap:
            return
        selected.append((index, row))
        selected_indexes.add(index)
        selection_reasons[index] = [reason]
        if reason not in reasons:
            reasons.append(reason)

    for index, row in rows[:first_rows]:
        add_row(index, row, f"first_{first_rows}_rows")

    seen_event_ids: set[str] = set()
    for index, row in rows:
        event_id = extract_event_id(row)
        if event_id is None or event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)
        add_row(index, row, "representative_event_ids")

    scored = [
        (row_keyword_score(row, keywords), index, row)
        for index, row in rows
        if index not in selected_indexes
    ]
    for score, index, row in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score <= 0:
            break
        add_row(index, row, "task_keyword_matches")

    for index, row in rows:
        add_row(index, row, "backfill_rows")
        if len(selected) >= cap:
            break

    selected.sort(key=lambda item: item[0])
    rendered_lines = [
        json.dumps(truncate_value(row, max_string_chars), ensure_ascii=False, sort_keys=True)
        for _, row in selected
    ]
    selected_row_metadata = [
        {
            "original_sample_index": index,
            "source_line": row.get("source_line"),
            "event_id": extract_event_id(row),
            "selection_reason": selection_reasons.get(index, []),
            "matched_keywords": row_matched_keywords(row, keywords),
        }
        for index, row in selected
    ]
    metadata = {
        "selected_sample_strategy": "first_rows+task_keyword_matches+representative_event_ids",
        "selected_sample_reason": "; ".join(reasons) if reasons else "no_valid_json_rows",
        "selected_sample_count": len(selected),
        "rendered_sample_cap": cap,
        "task_keywords": keywords,
        "selected_sample_rows": selected_row_metadata,
    }
    return "\n".join(rendered_lines), metadata


def compact_scalar(value: Any, max_chars: int = 120) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "...TRUNCATED..."
    return text


def find_first_scalar(row: dict[str, Any], keys: list[str]) -> str | None:
    lower_to_key = {str(key).lower(): key for key in row}
    for key in keys:
        actual = lower_to_key.get(key.lower())
        if actual is None:
            continue
        value = compact_scalar(row.get(actual))
        if value is not None:
            return value
    return None


def find_first_scalar_limited(row: dict[str, Any], keys: list[str], max_chars: int) -> str | None:
    lower_to_key = {str(key).lower(): key for key in row}
    for key in keys:
        actual = lower_to_key.get(key.lower())
        if actual is None:
            continue
        value = compact_scalar(row.get(actual), max_chars=max_chars)
        if value is not None:
            return value
    return None


def planner_event_hint(row: dict[str, Any]) -> str | None:
    event_id = extract_event_id(row)
    parts: list[str] = []
    if event_id:
        parts.append(f"event_id={event_id}")
    event_type = find_first_scalar(row, ["event_type", "event.type", "type", "log_type"])
    if event_type:
        parts.append(f"type={event_type}")
    uid = find_first_scalar(row, ["uid", "id", "event_uid"])
    if uid and not event_id:
        parts.append(f"uid={uid}")
    protocol = find_first_scalar(row, ["protocol", "proto"])
    if protocol:
        parts.append(f"protocol={protocol}")
    conn_state = find_first_scalar(row, ["conn_state", "state", "status"])
    if conn_state:
        parts.append(f"state={conn_state}")
    return ", ".join(parts) if parts else None


def planner_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key, value in row.items():
        normalized = normalize_keyword(str(key))
        if not any(token in normalized for token in ["label", "tag", "tactic", "technique", "category"]):
            continue
        scalar = compact_scalar(value, max_chars=80)
        if scalar is None:
            continue
        labels.append(f"{key}={scalar}")
    return labels


def planner_short_summary(row: dict[str, Any]) -> str:
    candidates = [
        ("timestamp", find_first_scalar(row, ["timestamp", "@timestamp", "datetime", "ts", "time"])),
        ("src", find_first_scalar(row, ["src_ip", "source_ip", "source.address", "SourceAddress"])),
        ("dest", find_first_scalar(row, ["dest_ip", "destination_ip", "destination.address", "DestAddress"])),
        ("dest_port", find_first_scalar(row, ["dest_port", "destination_port", "DestPort"])),
        ("host", find_first_scalar(row, ["host", "hostname", "computer_name"])),
        ("process", find_first_scalar(row, ["process_name", "process_path", "Image", "TargetImage"])),
        ("message", find_first_scalar_limited(row, ["message", "Message", "raw_message"], max_chars=80)),
    ]
    parts = [f"{name}={value}" for name, value in candidates if value]
    if not parts:
        for key in sorted(row)[:8]:
            value = compact_scalar(row.get(key), max_chars=60)
            if value:
                parts.append(f"{key}={value}")
            if len(parts) >= 5:
                break
    return " | ".join(parts) if parts else "row has no scalar summary fields"


def render_planner_sample_summaries(
    sample_rows_path: str | Path,
    cap: int = DEFAULT_PLANNER_SAMPLE_SUMMARY_CAP,
) -> tuple[str, dict[str, Any]]:
    planner_task = {
        "task_name": "plan_extraction_tasks",
        "goal": "Select the best supported extraction tasks for this dataset sample.",
        "target_event_types": [
            "process_activity",
            "network_activity",
            "network_connection",
            "registry_modification",
            "authentication",
            "file_activity",
            "dns_query",
            "powershell_activity",
            "credential_access",
            "defense_evasion",
        ],
        "required_fields": REQUIRED_CANDIDATE_FIELDS,
        "output_format": "jsonl",
    }
    rendered_rows, selection_metadata = select_prompt_sample_rows(
        sample_rows_path,
        planner_task,
        cap=cap,
        max_string_chars=DEFAULT_STRING_TRUNCATE_CHARS,
    )
    summaries: list[dict[str, Any]] = []
    for line in rendered_rows.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        summaries.append(
            {
                "source_line": row.get("source_line"),
                "fields_present": sorted(str(key) for key in row.keys())[:40],
                "event_hint": planner_event_hint(row),
                "short_summary": planner_short_summary(row),
                "labels": planner_labels(row),
            }
        )
    metadata = {
        "planner_sample_summary_strategy": selection_metadata["selected_sample_strategy"],
        "planner_sample_summary_reason": selection_metadata["selected_sample_reason"],
        "planner_sample_summary_count": len(summaries),
        "planner_sample_summary_cap": cap,
        "planner_selected_sample_rows": selection_metadata["selected_sample_rows"],
    }
    return json.dumps(summaries, ensure_ascii=False, indent=2), metadata


def render_prompt(
    template_path: str | Path,
    task_path: str | Path,
    sample_rows_path: str | Path,
    summary_path: str | Path,
    candidate_schema_path: str | Path,
    output_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    goal: str | None = None,
) -> str:
    task = load_json(task_path)
    validate_task_spec(task)
    if goal:
        task = dict(task)
        task["user_goal"] = goal
    if profile_path:
        profile = load_json(profile_path)
        validate_dataset_profile(profile)
        optional_hints = {
            "provided": True,
            "profile_source": str(profile_path),
            "profile": profile,
            "notes": "Optional user-provided hints. Samples and inferred context remain the source of truth.",
        }
    else:
        optional_hints = {
            "provided": False,
            "notes": "No user-provided profile. Use inferred context and sample rows as the source of truth.",
        }

    summary = load_json(summary_path)
    rendered_sample_rows, sample_metadata = select_prompt_sample_rows(sample_rows_path, task)
    summary.update(sample_metadata)
    write_json(summary_path, summary)

    template = Path(template_path).read_text(encoding="utf-8")
    rendered = template.format(
        optional_user_hints=json.dumps(optional_hints, ensure_ascii=False, indent=2),
        task_spec=json.dumps(task, ensure_ascii=False, indent=2),
        inferred_format_summary=json.dumps(summary, ensure_ascii=False, indent=2),
        candidate_event_schema=json.dumps(load_json(candidate_schema_path), ensure_ascii=False, indent=2),
        sample_rows=rendered_sample_rows,
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(rendered, encoding="utf-8")
    return rendered


def render_planner_prompt(
    template_path: str | Path,
    sample_rows_path: str | Path,
    summary_path: str | Path,
    candidate_schema_path: str | Path,
    extraction_plan_schema_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    summary = load_json(summary_path)
    rendered_sample_summaries, sample_metadata = render_planner_sample_summaries(sample_rows_path)
    summary.update(sample_metadata)
    write_json(summary_path, summary)

    template = Path(template_path).read_text(encoding="utf-8")
    rendered = template.format(
        inferred_format_summary=json.dumps(summary, ensure_ascii=False, indent=2),
        sample_summaries=rendered_sample_summaries,
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(rendered, encoding="utf-8")
    return rendered


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def extract_python_code(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
            code = payload.get("code")
            if isinstance(code, str):
                return code.strip() + "\n"
        except json.JSONDecodeError:
            pass
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip() + "\n"


def generated_code_has_cli_contract(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return "--input" in code and "--output" in code and "ArgumentParser" in code


def generate_extraction_code(
    rendered_prompt_path: str | Path,
    output_code_path: str | Path,
    model: str,
    api_key: str | None = None,
    vertex_ai: bool = False,
    project: str | None = None,
    location: str = "global",
) -> Path:
    """Generate Python extractor code using Stage 1's bundled Gemini client."""
    from dotenv import load_dotenv

    from stage1.gemini_client import build_gemini_client
    from stage1.logging_utils import Logger

    load_dotenv()
    args = argparse.Namespace(
        api_key=api_key,
        vertex_ai=vertex_ai,
        project=project,
        location=location,
    )
    log = Logger("INFO")
    client, types_mod = build_gemini_client(args, log)
    prompt = Path(rendered_prompt_path).read_text(encoding="utf-8")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_mod.GenerateContentConfig(
            system_instruction=(
                "Return only Python 3 source code. Do not include markdown fences or explanations."
            )
        ),
    )
    text = _extract_response_text(response)
    code = extract_python_code(text)
    if not generated_code_has_cli_contract(code):
        raise ValueError("generated code does not implement the required --input/--output CLI contract")
    Path(output_code_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_code_path).write_text(code, encoding="utf-8")
    return Path(output_code_path)


def generate_extraction_plan(
    planner_prompt_path: str | Path,
    output_plan_path: str | Path,
    model: str,
    api_key: str | None = None,
    vertex_ai: bool = False,
    project: str | None = None,
    location: str = "global",
) -> dict[str, Any]:
    from dotenv import load_dotenv

    from stage1.gemini_client import build_gemini_client
    from stage1.logging_utils import Logger

    load_dotenv()
    args = argparse.Namespace(
        api_key=api_key,
        vertex_ai=vertex_ai,
        project=project,
        location=location,
    )
    log = Logger("INFO")
    client, types_mod = build_gemini_client(args, log)
    prompt = Path(planner_prompt_path).read_text(encoding="utf-8")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types_mod.GenerateContentConfig(
            system_instruction="Return only one JSON object. Do not include markdown fences or explanations."
        ),
    )
    plan = extract_json_object(_extract_response_text(response))
    validate_extraction_plan(plan)
    write_json(output_plan_path, plan)
    return plan


def _extract_response_text(response: Any) -> str:
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
            executable = getattr(part, "executable_code", None)
            if executable and getattr(executable, "code", None):
                parts.append(executable.code)
    return "\n".join(parts).strip()


def execute_generated_code(
    generated_code_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(generated_code_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stdout_path).write_text(proc.stdout or "", encoding="utf-8")
    Path(stderr_path).write_text(proc.stderr or "", encoding="utf-8")

    output = Path(output_path)
    if proc.returncode != 0:
        raise RuntimeError(f"generated extractor failed with exit code {proc.returncode}")
    if not output.exists() or output.stat().st_size == 0:
        if stdout_looks_like_event_json(proc.stdout):
            raise RuntimeError(
                "generated extractor printed event-like JSON to stdout but did not write candidate_events.jsonl"
            )
        raise RuntimeError("generated extractor did not create a non-empty candidate_events.jsonl")
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_bytes": len(proc.stdout.encode("utf-8")),
        "stderr_bytes": len(proc.stderr.encode("utf-8")),
        "output_bytes": output.stat().st_size,
    }


def stdout_looks_like_event_json(stdout: str) -> bool:
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (
            "event_type" in obj or "timestamp" in obj or "source_line" in obj
        ):
            return True
    return False


def validate_candidate_events(path: str | Path, required_fields: list[str] | None = None) -> ValidationResult:
    required_fields = required_fields or REQUIRED_CANDIDATE_FIELDS
    total = 0
    valid = 0
    errors: list[dict[str, Any]] = []
    for line_no, obj, error in iter_jsonl(path):
        total += 1
        if error:
            errors.append({"line": line_no, "error": error})
            continue
        if not isinstance(obj, dict):
            errors.append({"line": line_no, "error": "record is not a JSON object"})
            continue
        missing = [field for field in required_fields if field not in obj]
        if missing:
            errors.append({"line": line_no, "error": f"missing fields: {', '.join(missing)}"})
            continue
        valid += 1
    return ValidationResult(ok=not errors, total_lines=total, valid_records=valid, errors=errors)


def make_event_uid(index: int, record: dict[str, Any], mode: str = "sequential") -> str:
    if mode == "hash":
        basis = json.dumps(record, ensure_ascii=False, sort_keys=True)
        return "E" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12].upper()
    return f"E{index:06d}"


def normalize_candidate_events(
    candidate_path: str | Path,
    output_path: str | Path,
    source_file: str | Path,
    dataset_label: str | None = None,
    scenario: str | None = None,
    uid_mode: str = "sequential",
    profile_path: str | Path | None = None,
) -> int:
    profile = load_json(profile_path) if profile_path else {}
    dataset = dataset_label or profile.get("dataset_name") or Path(source_file).stem or "unknown_dataset"
    scenario = scenario if scenario is not None else profile.get("scenario")

    def canonicalize_aliases(record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        if normalized.get("source_ip") in (None, "") and normalized.get("src_ip") not in (None, ""):
            normalized["source_ip"] = normalized.get("src_ip")
        if normalized.get("source_port") in (None, "") and normalized.get("src_port") not in (None, ""):
            normalized["source_port"] = normalized.get("src_port")

        source_event_id = normalized.get("source_event_id")
        for alias in ["uid", "request_id", "record_id"]:
            if source_event_id not in (None, ""):
                break
            value = normalized.get(alias)
            if value not in (None, ""):
                source_event_id = value
        if source_event_id not in (None, ""):
            normalized["source_event_id"] = source_event_id
            if normalized.get("event_id") == source_event_id:
                normalized["event_id"] = None
        if normalized.get("event_type") == "process_access":
            if normalized.get("source_process_path") in (None, "") and normalized.get("source_image") not in (None, ""):
                normalized["source_process_path"] = normalized.get("source_image")
            if normalized.get("target_process_path") in (None, "") and normalized.get("target_image") not in (None, ""):
                normalized["target_process_path"] = normalized.get("target_image")
            if normalized.get("source_thread_id") in (None, "") and normalized.get("SourceThreadId") not in (None, ""):
                normalized["source_thread_id"] = normalized.get("SourceThreadId")
            if normalized.get("call_trace") in (None, "") and normalized.get("CallTrace") not in (None, ""):
                normalized["call_trace"] = normalized.get("CallTrace")
        return normalized

    def rows() -> Iterable[dict[str, Any]]:
        index = 1
        for line_no, obj, error in iter_jsonl(candidate_path):
            if error or not isinstance(obj, dict):
                continue
            normalized = canonicalize_aliases(obj)
            for field in REQUIRED_CANDIDATE_FIELDS:
                normalized.setdefault(field, None)
            normalized["dataset"] = dataset
            normalized["scenario"] = scenario
            normalized["source_file"] = str(source_file)
            normalized["source_line"] = obj.get("source_line")
            normalized["event_uid"] = make_event_uid(index, normalized, uid_mode)
            normalized["text_for_embedding"] = event_text(normalized)
            yield normalized
            index += 1

    return write_jsonl(output_path, rows())


def build_stage2_input(normalized_path: str | Path, output_path: str | Path) -> int:
    def rows() -> Iterable[dict[str, Any]]:
        for _, obj, error in iter_jsonl(normalized_path):
            if error or not isinstance(obj, dict):
                continue
            text = obj.get("text_for_embedding") or event_text(obj)
            row = {field: obj.get(field) for field in STAGE2_FIELDS if field != "text_for_embedding"}
            row["text_for_embedding"] = text
            row["metadata"] = {
                "event_uid": obj.get("event_uid"),
                "dataset": obj.get("dataset"),
                "scenario": obj.get("scenario"),
                "source_file": obj.get("source_file"),
                "source_line": obj.get("source_line"),
                "timestamp": obj.get("timestamp"),
                "host": obj.get("host"),
                "user": obj.get("user"),
                "event_id": obj.get("event_id"),
                "event_type": obj.get("event_type"),
            }
            row["normalized_event"] = obj
            yield row

    return write_jsonl(output_path, rows())


def event_text(record: dict[str, Any]) -> str:
    if record.get("event_type") == "process_access":
        source_process = (
            record.get("source_process_name")
            or record.get("source_process_path")
            or record.get("SourceImage")
        )
        target_process = (
            record.get("target_process_name")
            or record.get("target_process_path")
            or record.get("TargetImage")
        )
        pieces = [
            f"At {record.get('timestamp')}",
            f"dataset={record.get('dataset')}",
            f"scenario={record.get('scenario')}",
            f"host={record.get('host')}",
            f"user={record.get('user')}",
            f"event_type={record.get('event_type')}",
            f"event_id={record.get('event_id')}",
            f"source process {source_process} accessed target process {target_process}",
            f"source_process_id={record.get('source_process_id')}",
            f"target_process_id={record.get('target_process_id')}",
            f"granted_access={record.get('granted_access')}",
            f"evidence={record.get('event_uid')}@{record.get('source_file')}:{record.get('source_line')}",
        ]
        return " | ".join(str(piece) for piece in pieces if piece is not None)

    pieces = [
        f"At {record.get('timestamp')}",
        f"dataset={record.get('dataset')}",
        f"scenario={record.get('scenario')}",
        f"host={record.get('host')}",
        f"user={record.get('user')}",
        f"event_type={record.get('event_type')}",
        f"event_id={record.get('event_id')}",
        f"source_event_id={record.get('source_event_id')}",
        f"process={record.get('process_name') or record.get('process_path')}",
        f"command={record.get('command_line')}",
        f"parent={record.get('parent_process_name')}",
        f"source={record.get('source_ip')}:{record.get('source_port')}",
        f"dest={record.get('dest_ip')}:{record.get('dest_port')}",
        f"protocol={record.get('protocol')}",
        f"conn_state={record.get('conn_state')}",
        f"tactic={record.get('mitre_attack_tactics')}",
        f"evidence={record.get('event_uid')}@{record.get('source_file')}:{record.get('source_line')}",
    ]
    return " | ".join(str(piece) for piece in pieces if piece is not None)


def run_stage1_pipeline(
    input_path: str | Path,
    run_id: str,
    task_path: str | Path | None = None,
    generated_code_source: str | Path | None = None,
    model: str = "gemini-3.1-flash-lite",
    sample_limit: int = 500,
    profile_path: str | Path | None = None,
    dataset_label: str | None = None,
    scenario: str | None = None,
    goal: str | None = None,
    auto_task: bool = False,
    planner_output: str | Path | None = None,
) -> dict[str, Any]:
    if auto_task and task_path:
        raise ValueError("Provide either an explicit task or auto_task, not both")
    if not auto_task and not task_path:
        raise ValueError("Provide a task_path or enable auto_task")

    root = repo_root()
    run_dir = root / "stage1" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    inspected = inspect_dataset(
        input_path,
        run_dir,
        sample_limit=sample_limit,
        profile_path=profile_path,
        dataset_label=dataset_label,
    )

    planner_prompt: Path | None = None
    extraction_plan_path: Path | None = None
    dynamic_task_path: Path | None = None
    extraction_plan: dict[str, Any] | None = None
    if auto_task:
        planner_prompt = run_dir / "planner_prompt.txt"
        render_planner_prompt(
            root / "stage1" / "prompts" / "extraction_planner_prompt.txt",
            inspected["sample_rows"],
            inspected["summary"],
            root / "stage1" / "schemas" / "candidate_event_schema.json",
            root / "stage1" / "schemas" / "extraction_plan_schema.json",
            planner_prompt,
        )
        extraction_plan_path = run_dir / "extraction_plan.json"
        if planner_output:
            extraction_plan = load_json(planner_output)
            validate_extraction_plan(extraction_plan)
            write_json(extraction_plan_path, extraction_plan)
        else:
            extraction_plan = generate_extraction_plan(planner_prompt, extraction_plan_path, model=model)
        dynamic_task_path = run_dir / "dynamic_task_spec.json"
        write_json(dynamic_task_path, build_dynamic_task_spec(extraction_plan))
        task_path = dynamic_task_path

    rendered_prompt = run_dir / "rendered_prompt.txt"
    render_prompt(
        root / "stage1" / "prompts" / "universal_extraction_prompt.txt",
        task_path,
        inspected["sample_rows"],
        inspected["summary"],
        root / "stage1" / "schemas" / "candidate_event_schema.json",
        rendered_prompt,
        profile_path=profile_path,
        goal=goal,
    )

    generated_code_path = run_dir / "generated_code.py"
    if generated_code_source:
        generated_code_path.write_text(Path(generated_code_source).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        generate_extraction_code(rendered_prompt, generated_code_path, model=model)

    candidate_path = run_dir / "candidate_events.jsonl"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    execution = execute_generated_code(
        generated_code_path,
        input_path,
        candidate_path,
        stdout_path,
        stderr_path,
    )
    validation = validate_candidate_events(candidate_path)
    if not validation.ok:
        write_json(run_dir / "candidate_validation_errors.json", validation.errors)
        raise RuntimeError("candidate_events.jsonl failed validation")

    normalized_path = run_dir / "normalized_logs.jsonl"
    normalized_count = normalize_candidate_events(
        candidate_path,
        normalized_path,
        input_path,
        dataset_label=dataset_label,
        scenario=scenario,
        profile_path=profile_path,
    )
    stage2_path = run_dir / "stage2_input.jsonl"
    stage2_count = build_stage2_input(normalized_path, stage2_path)

    metadata = {
        "run_id": run_id,
        "input": str(input_path),
        "profile": str(profile_path) if profile_path else None,
        "dataset_label": dataset_label,
        "scenario": scenario,
        "goal": goal,
        "task": str(task_path),
        "auto_task": auto_task,
        "planner_output": str(planner_output) if planner_output else None,
        "sample_limit": sample_limit,
        "generated_code_source": str(generated_code_source) if generated_code_source else "llm",
        "execution": execution,
        "candidate_validation": validation.__dict__,
        "normalized_count": normalized_count,
        "stage2_count": stage2_count,
        "artifacts": {
            "planner_prompt": str(planner_prompt) if planner_prompt else None,
            "extraction_plan": str(extraction_plan_path) if extraction_plan_path else None,
            "dynamic_task_spec": str(dynamic_task_path) if dynamic_task_path else None,
            "rendered_prompt": str(rendered_prompt),
            "generated_code": str(generated_code_path),
            "candidate_events": str(candidate_path),
            "normalized_logs": str(normalized_path),
            "stage2_input": str(stage2_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    }
    write_json(run_dir / "run_metadata.json", metadata)
    return metadata
