"""
Group D performance measures that need no new data, approved by the owner on
2026-10-26 ("Approve all M1–M9 (Recommended)"; `docs/research/group_d_decisions.md`,
the recommended options). Pure functions on dated wealth, the engine's steps and
its trades: no I/O, no clock, no app or UI import.

- M1 `cagr` (F6.1): G^(365.25 / D) - 1 on calendar days, G = W_end / W_0, W_0 the
  wealth before the build cost (so the build is inside G), D the calendar days
  from the build close to the last valued close. A span under 365 days is never
  annualized: its period return is returned, labelled partial.
- M2 `annual_volatility` (F6.2): sqrt(252) x sd(R_t), ddof 1, over per-session
  net returns. Over a path with a multi-session span, only the single-session
  returns are used and the result says "n of N sessions", N counting every
  return session (option (i)); the omitted spans can make it too low. sd <= ZERO_VARIANCE is `zero_variance`.
  The Sharpe ratio needs group C's cash series and is not here.
- M5 `stress_window` (F6.5 reused): a registered window's return and maximum
  drawdown on the wealth inside it. A window not wholly inside the path is
  `outside_window`, never shortened; a window edge without a value is
  `unavailable`.
- M6 `turnover_by_year` (F2.4, the approved reading): construction turnover is
  reported apart (it ends at the first execution that reaches the build's own
  target, or at the first execution of a different target, whichever comes
  first); the other executions' one-way turnover and traded notional are summed
  per calendar year, a part year labelled and never scaled; the headline is the
  mean of the full years' sums.
- M7 `participation` (F7.3): for each traded asset, q = |delta| x N x W- / MDV,
  flagged when q > p; delta the executed weight change, W- the pre-trade wealth
  per 1.0 of initial cash, N the registered notional, MDV the median of raw close
  x raw volume over the `window` sessions ending at the decision session d
  (never d + 1). Fewer than `window` such sessions is `unavailable`. The build is
  reported apart from the rebalances; there is no capacity figure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.portfolio.descriptive import MetricUnavailable, drawdown_summary
from stock_agent.portfolio.transition import TURNOVER_LIMIT, FeasibleTarget, StepResult

DAYS_PER_YEAR = 365.25  # M1
SESSIONS_PER_YEAR = 252  # M2
ZERO_VARIANCE = 1e-12  # M2: per-session sd at or below this is zero variance
OK = "ok"
UNAVAILABLE = "unavailable"
PARTIAL = "partial"
OUTSIDE_WINDOW = "outside_window"
ZERO = "zero_variance"
TOO_FEW = "insufficient_history"
EDGE_MISSING = "window_edge_value_unavailable"
NO_VOLUME = "volume_history_unavailable"


def cagr(initial: float, final: float, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """M1: the calendar-day CAGR, or the period return labelled partial under 365 days."""

    if not (math.isfinite(initial) and initial > 0 and math.isfinite(final) and final > 0):
        raise ValueError("wealth must be positive and finite")
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    growth = final / initial
    if days < 365:
        return {"value": growth - 1.0, "annualized": False, "label": PARTIAL, "days": days}
    return {"value": growth ** (DAYS_PER_YEAR / days) - 1.0, "annualized": True, "days": days}


def annual_volatility(returns: Sequence[float], spans: Sequence[int]) -> dict:
    """M2: sqrt(252) x sd of the single-session net returns (ddof 1)."""

    if len(returns) != len(spans):
        raise ValueError("returns and spans must align")
    single = np.array([r for r, s in zip(returns, spans, strict=True) if s == 1], dtype=float)
    total = int(sum(spans))  # every return session: a multi-session span counts each one
    detail = {"sessions": int(single.size), "of": total, "complete": single.size == total}
    if single.size < 2:
        return {"status": UNAVAILABLE, "reason": TOO_FEW, "value": None, **detail}
    sd = float(single.std(ddof=1))
    if sd <= ZERO_VARIANCE:
        return {"status": UNAVAILABLE, "reason": ZERO, "value": None, **detail}
    return {"status": OK, "reason": None, "value": math.sqrt(SESSIONS_PER_YEAR) * sd, **detail}


@dataclass(frozen=True)
class StressWindow:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def stress_window(wealth: pd.Series, window: StressWindow) -> dict:
    """M5: the window's return and F6.5 maximum drawdown on `wealth` (dated, NaN = no value)."""

    base = {"key": window.key, "label": window.label, "start": window.start, "end": window.end}
    if window.start < wealth.index[0] or window.end > wealth.index[-1]:
        return {
            **base,
            "status": UNAVAILABLE,
            "reason": OUTSIDE_WINDOW,
            "return": None,
            "drawdown": None,
        }
    if window.start not in wealth.index or window.end not in wealth.index:
        raise ValueError(f"window {window.key} must start and end on sessions")
    inside = wealth.loc[window.start : window.end]
    first, last = inside.iloc[0], inside.iloc[-1]
    if not (np.isfinite(first) and np.isfinite(last)):
        return {
            **base,
            "status": UNAVAILABLE,
            "reason": EDGE_MISSING,
            "return": None,
            "drawdown": None,
        }
    try:
        summary = drawdown_summary(inside)
    except MetricUnavailable:
        drawdown = None
    else:
        drawdown = {"value": summary.max_drawdown, "complete": summary.history_complete}
    return {
        **base,
        "status": OK,
        "reason": None,
        "return": float(last / first - 1.0),
        "drawdown": drawdown,
    }


@dataclass(frozen=True)
class Execution:
    """One executed target: its execution close (dated), its D5 record and its step."""

    session: pd.Timestamp
    feasibility: FeasibleTarget
    step: StepResult


def _construction(executions: Sequence[Execution]) -> int:
    """How many executions, from the build, are construction (M6 rule 1)."""

    build = executions[0].feasibility
    count = 1
    if TURNOVER_LIMIT not in build.binding:
        return count
    for execution in executions[1:]:
        record = execution.feasibility
        if not np.array_equal(record.requested, build.requested):
            return count  # a different target: a rebalance, construction is over
        count += 1
        if TURNOVER_LIMIT not in record.binding:
            return count  # it reached the build's own target
    return count


def turnover_by_year(
    executions: Sequence[Execution],
    *,
    first_session: pd.Timestamp,
    last_session: pd.Timestamp,
    year_bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict:
    """
    M6. `year_bounds` gives each calendar year's first and last session on the
    calendar; a year the path does not cover from its first to its last session
    is partial.
    """

    if not executions:
        return {"construction": None, "years": [], "annual_turnover": None, "annual_notional": None}
    built = _construction(executions)
    construction = executions[:built]
    years: dict[int, list[float]] = {}
    for execution in executions[built:]:
        sums = years.setdefault(execution.session.year, [0.0, 0.0])
        sums[0] += execution.step.turnover_one_way
        sums[1] += execution.step.traded_notional
    rows = []
    for year in range(first_session.year, last_session.year + 1):
        first, last = year_bounds[year]
        partial = first < first_session or last > last_session
        turnover, notional = years.get(year, [0.0, 0.0])
        rows.append(
            {
                "year": int(year),
                "turnover": float(turnover),
                "traded_notional": float(notional),
                "partial": bool(partial),
            }
        )
    full = [row for row in rows if not row["partial"]]
    return {
        "construction": {
            "executions": len(construction),
            "turnover": float(sum(e.step.turnover_one_way for e in construction)),
            "traded_notional": float(sum(e.step.traded_notional for e in construction)),
        },
        "years": rows,
        "annual_turnover": float(np.mean([r["turnover"] for r in full])) if full else None,
        "annual_notional": float(np.mean([r["traded_notional"] for r in full])) if full else None,
        "full_years": len(full),
    }


def median_dollar_volume(dollar_volume: np.ndarray, decision: int, window: int) -> np.ndarray:
    """MDV per asset over the `window` sessions ending at `decision` (NaN if any is missing)."""

    if decision + 1 < window:
        return np.full(dollar_volume.shape[1], np.nan)
    rows = dollar_volume[decision + 1 - window : decision + 1]
    complete = np.isfinite(rows).all(axis=0)
    values = np.median(np.where(np.isfinite(rows), rows, 0.0), axis=0)
    return np.where(complete, values, np.nan)


def participation(
    trades: Sequence[tuple[int, np.ndarray, float]],
    dollar_volume: np.ndarray,
    cash_index: int,
    *,
    notional: float,
    threshold: float,
    window: int,
) -> dict:
    """
    M7. `trades`: each execution as (decision session, executed weight change
    per asset, pre-trade wealth per 1.0 of initial cash), the build first.
    """

    def assess(trade) -> list[tuple[float | None, bool]]:
        decision, delta, wealth = trade
        mdv = median_dollar_volume(dollar_volume, decision, window)
        out = []
        for asset, change in enumerate(delta):
            if asset == cash_index or abs(change) == 0.0:
                continue
            if not (np.isfinite(mdv[asset]) and mdv[asset] > 0):
                out.append((None, False))
                continue
            q = float(abs(change) * notional * wealth / mdv[asset])
            out.append((q, bool(q > threshold)))
        return out

    def summary(found) -> dict:
        values = [q for q, _ in found if q is not None]
        return {
            "assessed": len(values),
            "unavailable": sum(q is None for q, _ in found),
            "flagged": sum(flag for _, flag in found),
            "max_q": max(values) if values else None,
        }

    if not trades:
        return {"build": None, "rebalances": None}
    rebalances = [pair for trade in trades[1:] for pair in assess(trade)]
    return {
        "build": summary(assess(trades[0])),
        "rebalances": {**summary(rebalances), "executions": len(trades) - 1},
    }
