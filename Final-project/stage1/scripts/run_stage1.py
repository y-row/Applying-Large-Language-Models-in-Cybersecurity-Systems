#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import resolve_optional_profile_path, resolve_task_path, run_stage1_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the universal Sieve-style Stage 1 pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", help="Legacy optional profile alias from stage1/profiles")
    parser.add_argument("--profile", help="Optional dataset profile JSON path used only as hints")
    parser.add_argument("--dataset-label", help="Optional dataset metadata label for normalized output")
    parser.add_argument("--scenario", help="Optional scenario metadata for normalized output")
    parser.add_argument("--goal", help="Optional natural language extraction goal to add to the task prompt")
    parser.add_argument("--task-name", help="Task alias from stage1/tasks, e.g. extract_process_creation")
    parser.add_argument("--task", help="Explicit task spec JSON path; overrides --task-name")
    parser.add_argument("--auto-task", action="store_true", help="Plan a dataset-supported dynamic task before code generation")
    parser.add_argument("--planner-output", help="Use an existing extraction_plan.json instead of calling the planner LLM")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument(
        "--generated-code",
        help="Use an existing extractor instead of calling the LLM; intended for tests and offline smoke runs.",
    )
    args = parser.parse_args()
    explicit_task_requested = bool(args.task_name or args.task)
    if args.auto_task == explicit_task_requested:
        parser.error("Provide either --auto-task or --task/--task-name, but not both.")
    profile_path = resolve_optional_profile_path(dataset=args.dataset, profile_path=args.profile)
    task_path = None if args.auto_task else resolve_task_path(task_name=args.task_name, task_path=args.task)
    metadata = run_stage1_pipeline(
        args.input,
        args.run_id,
        task_path=task_path,
        generated_code_source=args.generated_code,
        model=args.model,
        sample_limit=args.sample_limit,
        profile_path=profile_path,
        dataset_label=args.dataset_label,
        scenario=args.scenario,
        goal=args.goal,
        auto_task=args.auto_task,
        planner_output=args.planner_output,
    )
    print(metadata["artifacts"]["stage2_input"])


if __name__ == "__main__":
    main()
