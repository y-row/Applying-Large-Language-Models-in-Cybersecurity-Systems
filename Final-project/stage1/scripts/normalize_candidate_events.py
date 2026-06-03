#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import normalize_candidate_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize candidate events for Stage 2.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", help="Optional dataset profile JSON path used only for metadata hints")
    parser.add_argument("--dataset-label", help="Optional dataset metadata label")
    parser.add_argument("--scenario", help="Optional scenario metadata")
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--uid-mode", choices=("sequential", "hash"), default="sequential")
    args = parser.parse_args()
    count = normalize_candidate_events(
        args.input,
        args.output,
        args.source_file,
        dataset_label=args.dataset_label,
        scenario=args.scenario,
        uid_mode=args.uid_mode,
        profile_path=args.profile,
    )
    print(f"normalized_records={count}")


if __name__ == "__main__":
    main()
