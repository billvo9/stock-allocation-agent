"""
Portfolio health from backtest summaries (artifact `backtest_summary`, agreed
format v1.0), read from `<inputs>/backtest_summary/` (the newest file by
`generated_at`). Producers: the T5a engine or the RL comparison harness.

Fields besides the envelope (`inputs`):

    evidence_status    text (for example "rules_illustration")
    policy             text: the allocation policy simulated
    transition_status  "approved" | "engineering_only_unapproved_transition"
    constraints        {max_turnover, max_cost_bps, max_weight, max_drawdown}, >= 0 or null
    measures           {turnover, costs_bps, max_weight, max_drawdown}, >= 0 or null
                       (a negative value, for example a drawdown written as -0.45,
                       is refused as unrecognized_format: the approved F6.5
                       drawdown is non-negative)
    units              optional {measure: unit text}

Each check compares one stored measure with its stored limit and takes its
worst finding: above the limit is failed with reason `constraint_breached`
(an informational constraint-violation alert, never an action); an
unapproved transition or unreadable input files are degraded; a measure the
producer did not report, or a limit it did not declare, is
insufficient_evidence; within the limit is healthy. Measures come from the
approved T5a formulas only and are never recomputed here (OD8). A backtest is
simulated, never a track record, and every check says so. In the private
scope a synthetic input is at most insufficient_evidence (`synthetic_input`).
"""

from __future__ import annotations

from pathlib import Path

from stock_agent.health.inputs import (
    TRANSITION_STATUSES,
    Artifact,
    Finding,
    FormatError,
    NotRead,
    choice,
    common_findings,
    load_latest,
    require,
    worst_finding,
)
from stock_agent.health.model import CONSTRAINT_BREACHED, Area, Better, Check, HealthState

ARTIFACT = "backtest_summary"

# check_id -> (title, measure key, limit key, plain name)
MEASURES = {
    "portfolio.turnover": ("Portfolio turnover", "turnover", "max_turnover", "turnover"),
    "portfolio.costs": ("Trading costs", "costs_bps", "max_cost_bps", "estimated costs"),
    "portfolio.concentration": (
        "Concentration (largest weight)",
        "max_weight",
        "max_weight",
        "the largest single weight",
    ),
    "portfolio.drawdown": ("Drawdown", "max_drawdown", "max_drawdown", "the maximum drawdown"),
}
SIMULATED = "Simulated backtest, not a track record."


def _parse(artifact: Artifact) -> tuple[str, dict, dict, dict]:
    document = artifact.document
    label = (
        f"run {artifact.run_id}, policy {require(document, 'policy', 'text')} "
        f"({require(document, 'evidence_status', 'text')}), transition "
        f"{choice(document, 'transition_status', TRANSITION_STATUSES)}"
    )
    constraints = require(document, "constraints", "object")
    measures = require(document, "measures", "object")
    units = document.get("units") or {}
    if not isinstance(units, dict):
        raise FormatError("units must be an object")
    for _, measure, limit, _ in MEASURES.values():
        require(measures, measure, "nonnegative", optional=True)
        require(constraints, limit, "nonnegative", optional=True)
        if measure in units:
            require(units, measure, "text")
    return label, constraints, measures, units


def _measure_finding(name: str, value, bound) -> Finding:
    if value is None:
        return Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "measure_not_reported",
            f"The backtest did not report {name}.",
        )
    if bound is None:
        return Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_limit_declared",
            f"The backtest declares no limit for {name}, so it cannot be judged.",
        )
    if value > bound:
        return Finding(
            HealthState.FAILED,
            CONSTRAINT_BREACHED,
            f"{name.capitalize()} ({value:g}) breaches its limit ({bound:g}). {SIMULATED}",
        )
    return Finding(
        HealthState.HEALTHY,
        None,
        f"{name.capitalize()} ({value:g}) is within its limit ({bound:g}).",
    )


def backtest_checks(directory: Path | None, *, allow_synthetic: bool = False) -> list[Check]:
    loaded = load_latest(directory, ARTIFACT)
    parsed = None
    if isinstance(loaded, Artifact):
        try:
            parsed = _parse(loaded)
        except FormatError as error:
            loaded = NotRead(
                HealthState.UNAVAILABLE, "unrecognized_format", f"{loaded.path}: {error}"
            )
    evidence = None if directory is None else str(directory)
    checks = []
    for check_id, (title, measure, limit, name) in MEASURES.items():
        common = {
            "check_id": check_id,
            "area": Area.PORTFOLIO,
            "title": title,
            "evidence": evidence,
        }
        if parsed is None:
            checks.append(
                Check(
                    state=loaded.state,
                    reason=loaded.reason,
                    measured_at=None,
                    value=None,
                    unit=None,
                    threshold=None,
                    summary="No backtest summary could be read, so this is not judged."
                    if loaded.state is HealthState.UNAVAILABLE
                    else "No backtest summary has been written yet.",
                    detail=loaded.detail,
                    **common,
                )
            )
            continue
        label, constraints, measures, units = parsed
        value, bound = measures[measure], constraints[limit]
        finding = worst_finding(
            [
                _measure_finding(name, value, bound),
                *common_findings(loaded, allow_synthetic=allow_synthetic),
            ]
        )
        checks.append(
            Check(
                state=finding.state,
                reason=finding.reason,
                measured_at=loaded.generated_at,
                value=value,
                unit=units.get(measure, "as defined by the producer"),
                threshold=None
                if bound is None
                else f"<= {bound} (the backtest's declared {limit})",
                better=Better.LOWER,
                tolerance=None,
                summary=finding.summary,
                detail=(
                    f"{label}; {measure} = {value}, {limit} = {bound}; "
                    f"{loaded.describe()}. {SIMULATED}"
                ),
                **common,
            )
        )
    return checks
