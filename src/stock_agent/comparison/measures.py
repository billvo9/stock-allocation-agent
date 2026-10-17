"""
The approved measures of one evaluated episode, read from the environment's
ledger (every number comes from `portfolio_step` results; nothing is
re-derived here but sums and the approved descriptive drawdown).

Approved measures only (the owner's 2026-10-17 authorization; group D, with
CAGR, Sharpe and intervals, is not approved):

- Terminal wealth and total return. Wealth is the pre-trade mark-to-market
  value per 1.0 of initial cash; the evaluation starts at the common build
  close (D3: 100% cash, so wealth there is the initial cash) and ends at the
  last valued close. Total return = terminal / initial - 1.
- Drawdown and recovery: `portfolio.descriptive.drawdown_summary` (A6, F6.5)
  on the wealth at every session from the build close to the end. Sessions
  the environment does not value (a fund without a close) are missing, so
  such a history reports only the largest observed decline, labelled
  incomplete, never a full-period maximum drawdown. Synthetic sessions are
  integers; the function takes dates, so each session is passed as a
  placeholder date and mapped back. Only sessions are reported, never days.
- Turnover (F2.4): one-way turnover tau = 1/2 sum |delta_i| (cash included,
  a full switch is 1), split in two:
  - construction turnover: every execution from the build up to and
    including the first execution the turnover limit did not bind (the
    portfolio first reached its target; `portfolio.construction`), or every
    execution when it never did. Without a turnover limit it is the build
    alone;
  - rebalance turnover: every execution after construction, with the number
    of those executions and the turnover per rebalance.
  Not annualized: F2.4 annualizes per calendar year, and synthetic sessions
  carry no calendar dates, so the session count is reported instead. Traded
  notional sum_{i != cash} |delta_i| is split the same way.
- Costs (D2): the summed cost paid, as a fraction of initial wealth, and the
  summed per-execution cost C in basis points (the health measure).
- Constraint interventions (D5): targets changed by the cap or the turnover
  limit, and the one-way distance between the requested and executed
  targets. Fallback events: decisions the registered fallback took, with
  their reasons. Invalid targets: decisions whose target was not long-only
  and summing to 1 (each would have raised `InvalidTarget`).
- The build (D5, confirmed 2026-10-17): `portfolio.construction`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from stock_agent.portfolio.construction import build_report
from stock_agent.portfolio.descriptive import drawdown_summary
from stock_agent.portfolio.transition import CAP, TURNOVER_LIMIT
from stock_agent.rl.replay import EpisodeRecord

MAXIMUM_DRAWDOWN = "maximum_drawdown"
OBSERVED_DECLINE = "largest_observed_decline_incomplete_history"
_EPOCH = pd.Timestamp("1970-01-01")

UNITS = {
    "terminal_wealth": "pre-trade wealth at the last valued close per 1.0 of initial cash",
    "total_return": "terminal wealth / initial wealth - 1 (not annualized)",
    "drawdown": (
        "F6.5 on wealth at every session from the build close: the maximum drawdown when "
        "every session is valued, else the largest observed decline (incomplete history)"
    ),
    "turnover_one_way": "F2.4 one-way turnover summed over every execution, build included",
    "construction_turnover_one_way": (
        "F2.4 one-way turnover from the build until the portfolio first reaches its target "
        "(the build alone without a turnover limit; every execution if never reached)"
    ),
    "rebalance_turnover_one_way": "F2.4 one-way turnover of every execution after construction",
    "turnover_per_rebalance": (
        "rebalance turnover divided by the executions after construction (not annualized: "
        "synthetic sessions have no calendar dates)"
    ),
    "traded_notional": "sum of |weight change| over the funds, every execution, build included",
    "costs_paid": "summed trading cost as a fraction of initial wealth (D2 provisional rates)",
    "cost_sensitivity": (
        "terminal wealth and total return when the same decisions are replayed at 0x, 1x and "
        "2x the registered cost rates (D2)"
    ),
}


def _is_simplex(target) -> bool:
    return target is None or (
        bool(np.all(np.isfinite(target)))
        and bool(np.all(target >= 0))
        and abs(float(target.sum()) - 1.0) <= 1e-9
    )


def first_execution_index(record: EpisodeRecord) -> int:
    """The ledger index of the common build close: one session after the first decision."""

    return record.decisions[0].decision_index + 1


def wealth_by_session(record: EpisodeRecord) -> tuple[int, int, dict[int, float]]:
    """(start session, end session, pre-trade wealth at each valued close from the build)."""

    first = first_execution_index(record)
    rows = record.ledger[first:]
    wealth = {rows[0].session: rows[0].wealth_before}
    wealth.update({row.next_session: row.wealth_after for row in rows})
    return rows[0].session, rows[-1].next_session, wealth


def terminal_measures(record: EpisodeRecord) -> dict[str, float]:
    start, end, wealth = wealth_by_session(record)
    ratio = wealth[end] / wealth[start]
    return {"terminal_wealth": float(ratio), "total_return": float(ratio - 1.0)}


def drawdown_measures(record: EpisodeRecord) -> dict[str, object]:
    start, end, wealth = wealth_by_session(record)
    sessions = np.arange(start, end + 1)
    values = [wealth.get(int(s), np.nan) / wealth[start] for s in sessions]
    index = _EPOCH + pd.to_timedelta(sessions, unit="D")
    summary = drawdown_summary(pd.Series(values, index=index, dtype="float64"))

    def session(stamp) -> int | None:
        return None if stamp is None else int((stamp - _EPOCH).days)

    return {
        "measure": MAXIMUM_DRAWDOWN if summary.history_complete else OBSERVED_DECLINE,
        "value": summary.max_drawdown,
        "peak_session": session(summary.peak_date),
        "trough_session": session(summary.trough_date),
        "recovery_session": session(summary.recovery_date),
        "recovered": summary.recovered,
        "duration_sessions": summary.duration_sessions,
        "history_complete": summary.history_complete,
        "missing_sessions": summary.missing_sessions,
    }


def episode_measures(record: EpisodeRecord, cash_index: int) -> dict[str, object]:
    """Every approved measure of one episode (see the module docstring)."""

    first = first_execution_index(record)
    start, end, wealth = wealth_by_session(record)
    initial = wealth[start]
    rows = record.ledger[first:]
    targeted = [row for row in rows if row.feasibility is not None]
    build = build_report([(row.session, row.feasibility) for row in targeted], cash_index)
    complete = None if build is None else build.fully_built_session
    construction = [row for row in targeted if complete is None or row.session <= complete]
    after = [row for row in targeted if complete is not None and row.session > complete]
    turnover = sum(row.turnover_one_way for row in rows)
    notional = sum(row.traded_notional for row in rows)
    construction_turnover = sum(row.turnover_one_way for row in construction)
    rebalance_turnover = sum(row.turnover_one_way for row in after)
    span = end - start
    largest_weight = 0.0
    for row in rows:
        for weights in (row.w_post_trade, row.w_next):
            largest_weight = max(largest_weight, float(np.delete(weights, cash_index).max()))
    distances = [row.feasibility.distance_one_way for row in targeted]
    reasons = Counter(reason for d in record.decisions for reason in d.fallback_reasons)
    drawdown = drawdown_measures(record)
    return {
        **terminal_measures(record),
        "start_session": start,
        "end_session": end,
        "span_sessions": span,
        "drawdown": drawdown,
        "turnover_one_way": float(turnover),
        "construction_turnover_one_way": float(construction_turnover),
        "construction_executions": len(construction),
        "rebalance_turnover_one_way": float(rebalance_turnover),
        "rebalance_executions": len(after),
        "turnover_per_rebalance": float(rebalance_turnover / len(after)) if after else None,
        "traded_notional": float(notional),
        "construction_traded_notional": float(sum(row.traded_notional for row in construction)),
        "largest_execution_turnover": float(max((r.turnover_one_way for r in rows), default=0.0)),
        "costs_paid": float(sum(row.cost * row.wealth_before for row in rows) / initial),
        "costs_bps_summed": float(sum(row.cost for row in rows) * 1e4),
        "max_weight": largest_weight,
        "terminal_cash_weight": float(rows[-1].w_next[cash_index]),
        "decisions": len(record.decisions),
        "holds": sum(d.target is None for d in record.decisions),
        "trades": sum(row.turnover_one_way > 0 for row in rows),
        "interventions": {
            "targets": len(targeted),
            "cap_hits": sum(CAP in row.feasibility.binding for row in targeted),
            "turnover_limit_hits": sum(
                TURNOVER_LIMIT in row.feasibility.binding for row in targeted
            ),
            "distance_one_way_total": float(sum(distances)),
            "distance_one_way_max": float(max(distances, default=0.0)),
        },
        "fallback_events": sum(bool(d.fallback_reasons) for d in record.decisions),
        "fallback_reasons": dict(sorted(reasons.items())),
        "invalid_targets": sum(not _is_simplex(d.target) for d in record.decisions),
        "build": None
        if build is None
        else {**build.to_dict(), "delay_sessions": build.session - start},
    }
