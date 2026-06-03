#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from stage1.core import generate_extraction_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 1 extraction code via Gemini.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--api-key")
    parser.add_argument("--vertex-ai", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--location", default="global")
    args = parser.parse_args()
    generate_extraction_code(
        args.prompt,
        args.output,
        model=args.model,
        api_key=args.api_key,
        vertex_ai=args.vertex_ai,
        project=args.project,
        location=args.location,
    )
    print(args.output)


if __name__ == "__main__":
    main()

