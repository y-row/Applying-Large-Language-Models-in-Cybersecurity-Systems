#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import build_stage2_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 embedding input JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = build_stage2_input(args.input, args.output)
    print(f"stage2_records={count}")


if __name__ == "__main__":
    main()

