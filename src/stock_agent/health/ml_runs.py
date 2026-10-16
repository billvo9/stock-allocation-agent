"""
The latest ML experiment run (local monitor only), from the experiment
runner's attempt ledger: `<ml-root>/ledger.jsonl` (default
`reports/ml/ledger.jsonl`), one line per attempt, failures included
(`forecasting.experiment.run_experiment`).

training.ml_runs
    The newest attempt by `started_at`: completed from a clean tree is
    healthy, failed is failed (`ml_run_failed`), completed from a tree with
    uncommitted changes is degraded (`dirty_tree`: it may not reproduce), and
    a run that does not record its tree state is insufficient_evidence
    (`tree_state_unknown`). No ledger is insufficient_evidence; a ledger with
    no readable line is unavailable; unreadable lines beside readable ones are
    degraded. In the private scope a run is healthy only when the registry
    shows every version it registered as non-synthetic: a synthetic version
    caps it at insufficient_evidence (`synthetic_input`), and so does a
    version the registry cannot show (`data_class_unknown`: the registry is
    missing or refused, or nothing was registered).

The value is the newest attempt's status; the detail counts every attempt.
Nothing is re-run or recomputed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.health.inputs import SYNTHETIC_FINDING, Finding, worst_finding
from stock_agent.health.model import Area, Check, HealthState

# The runner's attempt ledger (`model_diagnostics.artifacts.LEDGER_FILE`), named here
# because health does not import another track's modules.
LEDGER_FILE = "ledger.jsonl"
TITLE = "Latest ML experiment run"
STATUSES = frozenset({"completed", "failed"})
THRESHOLD = "the newest attempt completed from a clean tree"


@dataclass(frozen=True)
class Attempt:
    started_at: datetime
    experiment: str
    status: str
    run_id: str | None
    registered: tuple[str, ...]
    git_sha: str | None
    git_dirty: bool | None
    error: str | None


def _attempt(line: str) -> Attempt:
    record = json.loads(line)
    if not isinstance(record, dict):
        raise TypeError("not an object")
    status = record.get("status")
    if status not in STATUSES:
        raise ValueError(f"status {status!r}")
    started = pd.Timestamp(record["started_at"])
    if started.tzinfo is None:
        raise ValueError("started_at has no time zone")
    registered = record.get("registered") or []
    if not isinstance(registered, list) or not all(isinstance(r, str) for r in registered):
        raise ValueError("registered is not a list of references")
    experiment = record.get("experiment")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("experiment is missing")
    dirty = record.get("git_dirty")
    return Attempt(
        started_at=started.tz_convert("UTC").to_pydatetime().astimezone(UTC),
        experiment=experiment,
        status=status,
        run_id=record.get("run_id") if isinstance(record.get("run_id"), str) else None,
        registered=tuple(registered),
        git_sha=record.get("git_sha") if isinstance(record.get("git_sha"), str) else None,
        git_dirty=dirty if isinstance(dirty, bool) else None,
        error=str(record["error"])[:200] if record.get("error") else None,
    )


def _check(**fields) -> Check:
    return Check(
        check_id="training.ml_runs",
        area=Area.TRAINING,
        title=TITLE,
        threshold=THRESHOLD,
        **fields,
    )


def ml_runs_check(
    root: Path | None,
    *,
    data_classes: Mapping[str, str] | None = None,
    allow_synthetic: bool = False,
) -> Check:
    if root is None:
        return _check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="input_not_configured",
            measured_at=None,
            value=None,
            unit=None,
            summary="No ML experiment run was checked in this run.",
            detail="No --ml-root was given to this run.",
            evidence=None,
        )
    path = root / LEDGER_FILE
    evidence = str(path)
    if not path.is_file():
        return _check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="input_missing",
            measured_at=None,
            value=None,
            unit=None,
            summary="No ML experiment has run yet.",
            detail=f"{path} does not exist; the experiment runner writes it.",
            evidence=evidence,
        )
    attempts, unreadable, reading_error = [], 0, ""
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as error:
        lines, unreadable = [], -1
        reading_error = type(error).__name__
    for line in lines:
        try:
            attempts.append(_attempt(line))
        except (ValueError, KeyError, TypeError):
            unreadable += 1
    if not attempts:
        return _check(
            state=HealthState.UNAVAILABLE,
            reason="unrecognized_format",
            measured_at=None,
            value=None,
            unit=None,
            summary="The ML experiment ledger could not be read.",
            detail=f"{path}: "
            + (reading_error if unreadable < 0 else f"{unreadable} lines, none readable"),
            evidence=evidence,
        )
    latest = max(attempts, key=lambda attempt: attempt.started_at)
    completed = sum(attempt.status == "completed" for attempt in attempts)
    findings = []
    if latest.status == "failed":
        findings.append(
            Finding(
                HealthState.FAILED,
                "ml_run_failed",
                f"The latest ML experiment run ({latest.experiment}) failed.",
            )
        )
    elif latest.git_dirty:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "dirty_tree",
                f"The latest ML experiment run ({latest.experiment}) completed from a tree with "
                "uncommitted changes, so it may not reproduce.",
            )
        )
    elif latest.git_dirty is None:
        findings.append(
            Finding(
                HealthState.INSUFFICIENT_EVIDENCE,
                "tree_state_unknown",
                f"The latest ML experiment run ({latest.experiment}) does not record whether its "
                "tree was clean, so it is not judged.",
            )
        )
    if unreadable > 0:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "unreadable_inputs",
                f"{unreadable} ledger lines could not be read; the newest readable attempt was used.",
            )
        )
    if not allow_synthetic:
        classes = {(data_classes or {}).get(ref) for ref in latest.registered}
        if not latest.registered or None in classes:
            # Without the registry's data class, a synthetic run cannot be told apart.
            findings.append(
                Finding(
                    HealthState.INSUFFICIENT_EVIDENCE,
                    "data_class_unknown",
                    "The registry does not show the data class of this run's versions, so it "
                    "says nothing yet about real models.",
                )
            )
        elif "synthetic" in classes:
            findings.append(SYNTHETIC_FINDING)
    findings.append(
        Finding(
            HealthState.HEALTHY,
            None,
            f"The latest ML experiment run ({latest.experiment}) completed and registered "
            f"{len(latest.registered)} versions.",
        )
    )
    finding = worst_finding(findings)
    tree = {True: "dirty", False: "clean", None: "unknown"}[latest.git_dirty]
    code = latest.git_sha[:12] if latest.git_sha else "unknown"
    started = f"{latest.started_at:%Y-%m-%d %H:%M} UTC"
    failed = len(attempts) - completed
    detail = [
        f"latest attempt: {latest.experiment}, {latest.status}, started {started}",
        f"run {latest.run_id or 'none'}",
        f"code {code}, tree {tree}",
        f"registered: {', '.join(latest.registered) or 'none'}",
        f"attempts in the ledger: {len(attempts)} ({completed} completed, {failed} failed)",
    ]
    if latest.error:
        detail.append(f"error: {latest.error}")
    return _check(
        state=finding.state,
        reason=finding.reason,
        measured_at=latest.started_at,
        value=latest.status,
        unit=None,
        summary=finding.summary,
        detail="\n".join(detail),
        evidence=evidence,
    )


__all__ = ["TITLE", "ml_runs_check"]
