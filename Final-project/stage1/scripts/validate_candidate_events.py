#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import validate_candidate_events, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidate_events.jsonl.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = validate_candidate_events(args.input)
    if args.report:
        write_json(args.report, result.__dict__)
    if not result.ok:
        for error in result.errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    print(f"valid_records={result.valid_records}")


if __name__ == "__main__":
    main()

