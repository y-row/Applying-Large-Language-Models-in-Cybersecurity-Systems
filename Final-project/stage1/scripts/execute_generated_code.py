#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import execute_generated_code, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute generated Stage 1 extractor code.")
    parser.add_argument("--code", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()
    result = execute_generated_code(args.code, args.input, args.output, args.stdout, args.stderr)
    if args.metadata:
        write_json(args.metadata, result)
    print(args.output)


if __name__ == "__main__":
    main()

