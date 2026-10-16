"""
The one network step in the health center: recording the latest GitHub
Actions runs on main with the GitHub CLI, for the private local monitor only
and only when it is asked to (`scripts/health_check.py local --github`).

    gh run list --repo billvo9/stock-allocation-agent --branch main
                --workflow <ci.yml | health.yml> --limit 20 --json <RUN_FIELDS>

one call per workflow, so frequent CI runs never push the daily run out of
the list; the two lists are kept together as one JSON list.

The command reads public run metadata (workflow, branch, commit, status,
conclusion, time) with the owner's own `gh` sign-in; it sends nothing and
changes nothing. Its output is kept as private evidence (`ci_runs.json`) and
read by `verification.run_list_checks` (`ci.main_runs`, `ci.daily_health`).

Outcomes, never a guess:

    gh not installed, not signed in, offline   insufficient_evidence
    or timed out                               (`gh_unavailable`), with the reason
    the list was written                       judged by `verification`

The public CI run never calls this module, and the Portfolio Lab never makes
a network call: it reads the summary this run writes. The subprocess runner
and the clock are injected, so tests never run `gh`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from stock_agent.health.verification import RUN_FIELDS, RunList

REPOSITORY = "billvo9/stock-allocation-agent"
BRANCH = "main"
WORKFLOWS = ("ci.yml", "health.yml")
LIMIT = 20
TIMEOUT_SECONDS = 60
EVIDENCE_FILE = "ci_runs.json"

Runner = Callable[..., subprocess.CompletedProcess]


def command(workflow: str) -> list[str]:
    return [
        "gh",
        "run",
        "list",
        "--repo",
        REPOSITORY,
        "--branch",
        BRANCH,
        "--workflow",
        workflow,
        "--limit",
        str(LIMIT),
        "--json",
        ",".join(RUN_FIELDS),
    ]


def _first_line(text: str | None) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[0][:200] if lines else "no message"


def collect_runs(runner: Runner, *, now: Callable[[], datetime], evidence_dir: Path) -> RunList:
    """
    Run `gh run list` once per workflow. On success the combined list is
    written to `<evidence_dir>/ci_runs.json` and returned with the time it was
    taken; when gh cannot run or fails, nothing is written and the run list
    says why.
    """

    runs: list[object] = []
    for workflow in WORKFLOWS:
        argv = command(workflow)
        try:
            completed = runner(
                argv, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
            )
        except FileNotFoundError:
            return _unavailable(
                "The GitHub CLI (gh) is not installed, so no run list was recorded."
            )
        except (OSError, subprocess.SubprocessError) as error:
            return _unavailable(f"gh run list could not be run: {type(error).__name__}.")
        if completed.returncode != 0:
            return _unavailable(
                f"gh run list exited {completed.returncode} (not signed in, offline or "
                f"rate-limited): {_first_line(completed.stderr)}"
            )
        try:
            listed = json.loads(completed.stdout or "")
        except json.JSONDecodeError:
            listed = None
        if not isinstance(listed, list):
            # Keep what gh printed: the run-list checks report it as unreadable.
            text = completed.stdout or ""
            break
        runs.extend(listed)
    else:
        text = json.dumps(runs, indent=2)
    measured_at = now()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / EVIDENCE_FILE
    path.write_text(text, encoding="utf-8")
    return RunList(text, measured_at, str(path))


def _unavailable(detail: str) -> RunList:
    return RunList(None, None, None, reason="gh_unavailable", detail=detail)


__all__ = ["REPOSITORY", "collect_runs", "command"]
