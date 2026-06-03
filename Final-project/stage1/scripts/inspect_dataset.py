#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import inspect_dataset, resolve_optional_profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a raw dataset and sample rows for Stage 1.")
    parser.add_argument("--input", required=True, help="Raw log input file")
    parser.add_argument("--dataset", help="Legacy optional profile alias from stage1/profiles")
    parser.add_argument("--profile", help="Optional dataset profile JSON path used only as hints")
    parser.add_argument("--dataset-label", help="Optional dataset metadata label for debug artifacts")
    parser.add_argument("--output-dir", help="Directory for sample_rows and summary artifacts")
    parser.add_argument("--sample-limit", type=int, default=500, help="Number of non-empty rows to sample")
    args = parser.parse_args()

    profile_path = resolve_optional_profile_path(dataset=args.dataset, profile_path=args.profile)
    output_dir = args.output_dir or Path(args.input).resolve().parent
    paths = inspect_dataset(
        args.input,
        output_dir,
        sample_limit=args.sample_limit,
        profile_path=profile_path,
        dataset_label=args.dataset_label,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
