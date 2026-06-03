#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


DATASET_NAME = "OTRF Mordor"
SOURCE_JSONL = Path("data/mordor/mordor_real_source.jsonl")
SOURCE_JSON = Path("data/mordor/mordor_real_source.json")
SAMPLE_PATH = Path("data/samples/mordor_real_sample_500.jsonl")
RUN_ID = "mordor_real_profile_free_500_offline"
RUN_DIR = Path("stage1/runs") / RUN_ID
REPORT_PATH = RUN_DIR / "profile_free_validation_report.md"
BLOCKED_ROOT_DATASET = Path("empire_dcsync_dcerpc_drsuapi_DsGetNCChanges_2020-09-21185829.json")
REQUIRED_NORMALIZED_FIELDS = {
    "event_uid",
    "dataset",
    "scenario",
    "source_file",
    "source_line",
    "timestamp",
    "host",
    "user",
    "event_id",
    "event_type",
    "raw_message",
    "text_for_embedding",
}


def read_jsonl_count_and_sample(source: Path, sample_path: Path, limit: int) -> tuple[int, int]:
    count = 0
    sampled = 0
    with source.open("r", encoding="utf-8", errors="ignore") as fin, sample_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            count += 1
            if sampled < limit:
                fout.write(line.rstrip("\n") + "\n")
                sampled += 1
    return count, sampled


def read_json_array_count_and_sample(source: Path, sample_path: Path, limit: int) -> tuple[int, int]:
    with source.open("r", encoding="utf-8", errors="ignore") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{source} is JSON but not a JSON array")
    sampled = 0
    with sample_path.open("w", encoding="utf-8") as fout:
        for obj in payload[:limit]:
            fout.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            sampled += 1
    return len(payload), sampled


def infer_source_format(source: Path) -> str:
    with source.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            char = f.read(1)
            if not char:
                return "ndjson"
            if char.isspace():
                continue
            return "json_array" if char == "[" else "ndjson"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def choose_source(root: Path) -> Path:
    jsonl = root / SOURCE_JSONL
    json_file = root / SOURCE_JSON
    if jsonl.exists():
        return jsonl
    if json_file.exists():
        return json_file
    raise FileNotFoundError(
        "No safe Mordor source sample found. Expected data/mordor/mordor_real_source.jsonl "
        "or data/mordor/mordor_real_source.json. Refusing to read blocked root dataset "
        f"{BLOCKED_ROOT_DATASET}."
    )


def markdown_json(obj: Any, max_chars: int = 2000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n..."
    return text


def write_report(
    *,
    source: Path,
    source_format: str,
    source_count: int,
    sample_count: int,
    inspect_cmd: list[str],
    run_cmd: list[str],
    inspect_proc: subprocess.CompletedProcess[str],
    run_proc: subprocess.CompletedProcess[str],
    artifacts: dict[str, bool],
    candidate_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    prompt_markers: dict[str, bool],
    metadata: dict[str, Any],
    problems: list[str],
    source_url: str,
) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    conclusion = (
        "Yes, this dataset sample can be used for the Stage 1 MVP offline profile-free path."
        if not problems
        else "Not yet. Fix the problems below before treating this as a Stage 1 MVP dataset."
    )
    report = f"""# Mordor Profile-Free Stage 1 Validation

## A. Dataset Source
- dataset name: {DATASET_NAME}
- scenario name: {source.stem}
- URL: {source_url}
- local file: {source}
- source format: {source_format}
- size bytes: {source.stat().st_size}
- row/object count: {source_count}
- sample file: {SAMPLE_PATH}
- sample count: {sample_count}

## B. Profile-Free Status
- ran without `--profile`: {"--profile" not in run_cmd}
- ran without `--dataset`: {"--dataset" not in run_cmd}
- run metadata profile: {metadata.get("profile")}
- rendered prompt uses inferred context: {prompt_markers.get("INFERRED DATASET CONTEXT", False)}
- rendered prompt includes sample rows: {prompt_markers.get("SAMPLE ROWS", False)}
- optional hints marker present: {prompt_markers.get("OPTIONAL USER-PROVIDED HINTS", False)}

## C. Run Command
```powershell
{" ".join(run_cmd)}
```

## D. Artifacts Generated
{chr(10).join(f"- {name}: {exists}" for name, exists in artifacts.items())}

## E. Counts
- candidate_events count: {len(candidate_rows)}
- normalized_logs count: {len(normalized_rows)}
- stage2_input count: {len(stage2_rows)}

## F. First Normalized Event Example
```json
{markdown_json(normalized_rows[0] if normalized_rows else None)}
```

## G. First Stage2 Input Example
```json
{markdown_json(stage2_rows[0] if stage2_rows else None)}
```

## H. Problems Found
{chr(10).join(f"- {problem}" for problem in problems) if problems else "- none"}

## I. Conclusion
{conclusion}

## Inspect Command
```powershell
{" ".join(inspect_cmd)}
```

## Inspect Result
- return code: {inspect_proc.returncode}
- stderr: {inspect_proc.stderr.strip() or "none"}

## Run Result
- return code: {run_proc.returncode}
- stderr: {run_proc.stderr.strip() or "none"}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 1 profile-free flow on a safe Mordor sample.")
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--source-url", default="not provided")
    parser.add_argument("--keep-existing-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    (root / "data/mordor").mkdir(parents=True, exist_ok=True)
    (root / "data/samples").mkdir(parents=True, exist_ok=True)
    source = choose_source(root)
    source_format = infer_source_format(source)
    sample_path = root / SAMPLE_PATH

    if source_format == "json_array":
        source_count, sample_count = read_json_array_count_and_sample(source, sample_path, args.sample_limit)
    else:
        source_count, sample_count = read_jsonl_count_and_sample(source, sample_path, args.sample_limit)

    run_dir = root / RUN_DIR
    if run_dir.exists() and not args.keep_existing_run:
        shutil.rmtree(run_dir)

    inspect_cmd = [
        sys.executable,
        "stage1/scripts/inspect_dataset.py",
        "--input",
        str(SAMPLE_PATH),
        "--output-dir",
        str(RUN_DIR),
    ]
    run_cmd = [
        sys.executable,
        "stage1/scripts/run_stage1.py",
        "--input",
        str(SAMPLE_PATH),
        "--task-name",
        "extract_process_creation",
        "--run-id",
        RUN_ID,
        "--generated-code",
        "stage1/runs/smoke_generated_code.py",
    ]
    inspect_proc = run_command(inspect_cmd, root)
    run_proc = run_command(run_cmd, root)

    artifacts = {
        "sample_rows.jsonl": (run_dir / "sample_rows.jsonl").exists(),
        "inferred_format_summary.json": (run_dir / "inferred_format_summary.json").exists(),
        "dataset_profile_draft.json": (run_dir / "dataset_profile_draft.json").exists(),
        "rendered_prompt.txt": (run_dir / "rendered_prompt.txt").exists(),
        "generated_code.py": (run_dir / "generated_code.py").exists(),
        "candidate_events.jsonl": (run_dir / "candidate_events.jsonl").exists(),
        "normalized_logs.jsonl": (run_dir / "normalized_logs.jsonl").exists(),
        "stage2_input.jsonl": (run_dir / "stage2_input.jsonl").exists(),
        "stdout.txt": (run_dir / "stdout.txt").exists(),
        "stderr.txt": (run_dir / "stderr.txt").exists(),
        "run_metadata.json": (run_dir / "run_metadata.json").exists(),
    }
    candidate_rows = load_jsonl(run_dir / "candidate_events.jsonl")
    normalized_rows = load_jsonl(run_dir / "normalized_logs.jsonl")
    stage2_rows = load_jsonl(run_dir / "stage2_input.jsonl")
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    prompt_text = (run_dir / "rendered_prompt.txt").read_text(encoding="utf-8", errors="ignore") if (
        run_dir / "rendered_prompt.txt"
    ).exists() else ""
    prompt_markers = {
        marker: marker in prompt_text
        for marker in [
            "INFERRED DATASET CONTEXT",
            "OPTIONAL USER-PROVIDED HINTS",
            "SAMPLE ROWS",
            "EXTRACTION TASK",
        ]
    }

    problems: list[str] = []
    if inspect_proc.returncode != 0:
        problems.append("inspect_dataset.py failed")
    if run_proc.returncode != 0:
        problems.append("run_stage1.py failed")
    if metadata.get("profile") is not None:
        problems.append("run_metadata.json profile is not null")
    if "--profile" in run_cmd:
        problems.append("run command unexpectedly contains --profile")
    if "--dataset" in run_cmd:
        problems.append("run command unexpectedly contains --dataset")
    for marker, present in prompt_markers.items():
        if not present:
            problems.append(f"rendered_prompt.txt missing {marker}")
    if not candidate_rows:
        problems.append("candidate_events.jsonl is empty")
    if not normalized_rows:
        problems.append("normalized_logs.jsonl is empty")
    if not stage2_rows:
        problems.append("stage2_input.jsonl is empty")
    if normalized_rows:
        missing = REQUIRED_NORMALIZED_FIELDS - set(normalized_rows[0])
        if missing:
            problems.append(f"normalized_logs.jsonl missing fields: {sorted(missing)}")
    if stage2_rows:
        first_stage2 = stage2_rows[0]
        if "event_uid" not in first_stage2:
            problems.append("stage2_input.jsonl missing event_uid")
        if not isinstance(first_stage2.get("text_for_embedding"), str) or not first_stage2["text_for_embedding"].strip():
            problems.append("stage2_input.jsonl text_for_embedding is empty")
        if "metadata" not in first_stage2:
            problems.append("stage2_input.jsonl missing metadata")

    write_report(
        source=source,
        source_format=source_format,
        source_count=source_count,
        sample_count=sample_count,
        inspect_cmd=inspect_cmd,
        run_cmd=run_cmd,
        inspect_proc=inspect_proc,
        run_proc=run_proc,
        artifacts=artifacts,
        candidate_rows=candidate_rows,
        normalized_rows=normalized_rows,
        stage2_rows=stage2_rows,
        prompt_markers=prompt_markers,
        metadata=metadata,
        problems=problems,
        source_url=args.source_url,
    )
    print(REPORT_PATH)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
