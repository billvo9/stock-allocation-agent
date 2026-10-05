#!/usr/bin/env python3
"""
PostToolUse(Edit|Write|MultiEdit) hook: lint the one edited Python file.

Runs `ruff check --no-fix --no-cache` on the edited .py file only. It never
formats, never fixes, and never writes a cache, so it cannot change files.
Formatting is checked by the full quality gates (/project-verify), because
code is often legitimately unformatted mid-change.

Lint findings exit 2 with a short summary on stderr, which Claude Code feeds
back to Claude. Everything else (non-Python file, file outside the project,
deleted file, ruff unavailable, ruff internal error) exits 0 silently.

Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_MAX_LINES = 20


def _find_ruff(project_dir: str) -> str | None:
    local = os.path.join(project_dir, ".venv", "bin", "ruff")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("ruff")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    if not isinstance(payload, dict):
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path")

    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or ""

    if not project_dir:
        return 0

    project = os.path.realpath(project_dir)
    target = os.path.realpath(file_path)

    inside = os.path.commonpath([project, target]) == project
    excluded = any(
        part in {".venv", ".git", "__pycache__"}
        for part in os.path.relpath(target, project).split(os.sep)
    )

    if not inside or excluded or not os.path.isfile(target):
        return 0

    ruff = _find_ruff(project)

    if ruff is None:
        return 0

    try:
        completed = subprocess.run(
            [ruff, "check", "--no-fix", "--no-cache", "--quiet", "--force-exclude", target],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0

    # 0 = clean, 1 = lint findings, 2 = ruff error (config, crash).
    if completed.returncode != 1:
        return 0

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    relative = os.path.relpath(target, project)

    print(f"ruff check found issues in {relative} (not auto-fixed):", file=sys.stderr)
    for line in lines[:_MAX_LINES]:
        print(line, file=sys.stderr)
    if len(lines) > _MAX_LINES:
        print(f"... {len(lines) - _MAX_LINES} more line(s)", file=sys.stderr)

    return 2


if __name__ == "__main__":
    sys.exit(main())
