"""Script import bootstrap for direct execution from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

