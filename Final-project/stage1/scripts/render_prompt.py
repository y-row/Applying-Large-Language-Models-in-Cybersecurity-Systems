#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import render_prompt, repo_root


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Render the universal Stage 1 extraction prompt.")
    parser.add_argument("--profile", help="Optional dataset profile JSON path used only as hints")
    parser.add_argument("--goal", help="Optional natural language extraction goal to add to the task prompt")
    parser.add_argument("--task", required=True)
    parser.add_argument("--sample-rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--template", default=str(root / "stage1" / "prompts" / "universal_extraction_prompt.txt"))
    parser.add_argument("--candidate-schema", default=str(root / "stage1" / "schemas" / "candidate_event_schema.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render_prompt(
        args.template,
        args.task,
        args.sample_rows,
        args.summary,
        args.candidate_schema,
        args.output,
        profile_path=args.profile,
        goal=args.goal,
    )
    print(args.output)


if __name__ == "__main__":
    main()
