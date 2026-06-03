"""Lightweight logger for the standalone Stage 1 pipeline."""

from __future__ import annotations

import sys


class Logger:
    def __init__(self, level: str = "INFO"):
        self.level = level.upper()
        self.levels = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def _ok(self, want: str) -> bool:
        return self.levels.get(self.level, 20) <= self.levels.get(want, 20)

    def debug(self, msg: str) -> None:
        if self._ok("DEBUG"):
            print(f"[DEBUG] {msg}")

    def info(self, msg: str) -> None:
        if self._ok("INFO"):
            print(f"[INFO ] {msg}")

    def warn(self, msg: str) -> None:
        if self._ok("WARN"):
            print(f"[WARN ] {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        if self._ok("ERROR"):
            print(f"[ERROR] {msg}", file=sys.stderr)
