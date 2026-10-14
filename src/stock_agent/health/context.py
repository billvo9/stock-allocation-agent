"""
Where a summary was produced: GitHub Actions run metadata (allowlisted, no
secrets) or the local machine, and the code state claimed by git.

GitHub Actions: only the variables in `GITHUB_VARIABLES` are read, each
checked against a strict pattern; a value that does not match is recorded as
null rather than copied. No token, secret or other environment variable is
ever read, so nothing secret can reach a summary. The caller passes the
environment mapping (tests pass a fake one).

Code state: `git rev-parse HEAD` and `git status --porcelain` through an
injected runner. Any modified, staged or untracked path makes the tree dirty
(an uncommitted new module runs too), except an untracked `.venv` entry: a
session worktree's symlinked virtual environment, which `.gitignore`'s
`.venv/` pattern does not match. Both values are the producer's claims, not
authenticated provenance.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

GITHUB_ACTIONS = "github_actions"
LOCAL = "local"

_ID = re.compile(r"[0-9]{1,20}")
_NAME = re.compile(r"[A-Za-z0-9 ._/()-]{1,100}")
_SHA = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}")
_SERVER = re.compile(r"https://[A-Za-z0-9.-]{1,100}")

# variable -> (run_context key, pattern)
GITHUB_VARIABLES: Mapping[str, tuple[str, re.Pattern[str]]] = {
    "GITHUB_WORKFLOW": ("workflow", _NAME),
    "GITHUB_RUN_ID": ("run_id", _ID),
    "GITHUB_RUN_NUMBER": ("run_number", _ID),
    "GITHUB_RUN_ATTEMPT": ("run_attempt", _ID),
    "GITHUB_EVENT_NAME": ("event", _NAME),
    "GITHUB_REF_NAME": ("ref", _NAME),
    "GITHUB_SHA": ("sha", _SHA),
    "GITHUB_REPOSITORY": ("repository", _REPOSITORY),
    "RUNNER_OS": ("runner_os", _NAME),
}


def in_github_actions(env: Mapping[str, str]) -> bool:
    return env.get("GITHUB_ACTIONS") == "true"


def run_context(env: Mapping[str, str]) -> dict[str, str | None]:
    """The summary's `run_context`: allowlisted Actions metadata, or just `local`."""

    if not in_github_actions(env):
        return {"environment": LOCAL}
    context: dict[str, str | None] = {"environment": GITHUB_ACTIONS}
    for variable, (key, pattern) in GITHUB_VARIABLES.items():
        value = env.get(variable)
        context[key] = value if value is not None and pattern.fullmatch(value) else None
    server = env.get("GITHUB_SERVER_URL", "")
    if _SERVER.fullmatch(server) and context["repository"] and context["run_id"]:
        context["run_url"] = f"{server}/{context['repository']}/actions/runs/{context['run_id']}"
    else:
        context["run_url"] = None
    return context


Runner = Callable[..., subprocess.CompletedProcess]
VENV_ENTRY = "?? .venv"


@dataclass(frozen=True)
class CodeState:
    sha: str | None
    dirty: bool | None


def code_state(root: Path, runner: Runner = subprocess.run) -> CodeState:
    """The checked-out commit and whether tracked files differ; None when git fails."""

    def git(*args: str) -> str | None:
        try:
            completed = runner(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout if completed.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    sha = head.strip() if head is not None and _SHA.fullmatch(head.strip()) else None
    status = git("status", "--porcelain")
    if status is None:
        return CodeState(sha=sha, dirty=None)
    changes = [line for line in status.splitlines() if line.strip() and line != VENV_ENTRY]
    return CodeState(sha=sha, dirty=bool(changes))
