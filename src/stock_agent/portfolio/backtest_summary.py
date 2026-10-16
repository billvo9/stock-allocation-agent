"""
The `backtest_summary` health input (agreed format v1.0, 2026-10-14) for
one simulation of the approved T5a engine.

A summary is written to `<root>/backtest_summary/<run_id>.json` (locally
`reports/health_inputs/`, ignored by Git), strict JSON, write-once. The
Model Health Center reads it and compares each measure with its declared
limit; it recomputes nothing.

    artifact, schema_version "1.0", generated_at (UTC, microseconds, Z),
    code_sha (40-hex or null), synthetic, run_id, policy, evidence_status,
    transition_status "approved",
    constraints {max_weight, max_turnover, max_cost_bps, max_drawdown}
    measures    {turnover, costs_bps, max_weight, max_drawdown}
    units       {measure: definition}
    assumptions {the registered cost rates and multiplier, cash return,
                 lag, schedule}  (extra: ignored by the health reader)

Measures are computed from the engine's `StepResult`s only:

- `turnover`: the largest one-way turnover of any single execution, the
  build included (F2.4's one-way turnover, a full switch is 1), so it is
  comparable with D5's per-rebalance limit `max_turnover` (tau_max).
- `costs_bps`: the summed trading cost C of every execution, each a
  fraction of that execution's pre-trade wealth, in basis points; not
  annualized.
- `max_weight`: the largest non-cash weight after any trade or at any
  valued close, so drift above the cap shows (D5: drift is allowed and
  reported).
- `max_drawdown`: F6.5 (approved for group A on 2026-10-14) on the
  simulated pre-trade wealth path, which starts before the build's cost;
  null when any close in the path is unavailable (R5c: an incomplete
  history has no full-period maximum).

Limits: `max_weight` is D5's cap and `max_turnover` D5's tau_max (null when
none is registered). D5 registers no cost or drawdown limit, so those are
null unless a caller registers one.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stock_agent.portfolio.engine import CASH_RETURN_LABEL, SimulationResult
from stock_agent.portfolio.transition import Constraints

ARTIFACT = "backtest_summary"
SCHEMA_VERSION = "1.0"
TRANSITION_STATUS = "approved"
UNITS = {
    "turnover": (
        "largest one-way turnover of a single execution, the build included "
        "(fraction of pre-trade wealth; a full switch is 1)"
    ),
    "costs_bps": (
        "summed trading cost of every execution, each in basis points of its "
        "pre-trade wealth (not annualized)"
    ),
    "max_weight": "largest non-cash weight after a trade or at a valued close",
    "max_drawdown": "fraction of the running peak of simulated pre-trade wealth (F6.5)",
}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA = re.compile(r"[0-9a-f]{40}")


def max_drawdown(wealth: Sequence[float | None]) -> float | None:
    """
    F6.5: max over t of 1 - W_t / max_{s <= t} W_s, a non-negative fraction;
    None when any value is missing (R5c) or fewer than two values exist.
    """

    if len(wealth) < 2 or any(value is None for value in wealth):
        return None
    values = np.asarray(wealth, dtype=float)
    return float(np.max(1.0 - values / np.maximum.accumulate(values)))


def measures(result: SimulationResult, cash_index: int) -> dict[str, float | None]:
    """The four health measures of one simulation (see the module docstring)."""

    trades = [step for step in result.steps if step.turnover_one_way > 0]
    largest = 0.0
    for step in result.steps:
        for weights in (step.w_post_trade, step.w_next):
            risky = np.delete(np.asarray(weights), cash_index)
            largest = max(largest, float(risky.max()) if risky.size else 0.0)
    return {
        "turnover": max((step.turnover_one_way for step in trades), default=0.0),
        "costs_bps": float(sum(step.cost for step in result.steps) * 1e4),
        "max_weight": largest,
        "max_drawdown": max_drawdown(result.wealth),
    }


def _limit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite limit or None")
    return number


def summary_record(
    result: SimulationResult,
    *,
    constraints: Constraints,
    run_id: str,
    policy: str,
    evidence_status: str,
    synthetic: bool,
    generated_at: datetime,
    code_sha: str | None,
    assumptions: Mapping[str, object],
    max_cost_bps: float | None = None,
    max_drawdown_limit: float | None = None,
) -> dict[str, object]:
    """One `backtest_summary` document (agreed v1.0) for `result`."""

    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"run_id {run_id!r} must be a safe file name")
    for name, text in (("policy", policy), ("evidence_status", evidence_status)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{name} must be non-empty text")
    if code_sha is not None and not _SHA.fullmatch(code_sha):
        raise ValueError("code_sha must be a 40-hex git SHA or None")
    if generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    return {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "code_sha": code_sha,
        "synthetic": bool(synthetic),
        "run_id": run_id,
        "policy": policy,
        "evidence_status": evidence_status,
        "transition_status": TRANSITION_STATUS,
        "constraints": {
            "max_weight": constraints.w_max,
            "max_turnover": constraints.tau_max,
            "max_cost_bps": _limit(max_cost_bps, "max_cost_bps"),
            "max_drawdown": _limit(max_drawdown_limit, "max_drawdown"),
        },
        "measures": measures(result, constraints.cash_index),
        "units": dict(UNITS),
        "assumptions": {"cash_return": CASH_RETURN_LABEL, **assumptions},
        "status": result.status,
        "status_reason": result.reason,
    }


def write_summary(record: Mapping[str, object], root: Path) -> Path:
    """Write `record` to `<root>/backtest_summary/<run_id>.json`, once (strict JSON)."""

    path = Path(root) / ARTIFACT / f"{record['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path
