"""
Every registered variant of a showcase configuration through the approved T5a
engine (D1-D8), and its approved measures.

For each universe, the stored bars become one F5.1 return table
(`showcase.returns`). Each strategy's rule decides at the sessions of its
`ScheduleSpec(every, offset, lag, first_decision)`, seeing only the returns
through each decision's close (`collect_targets`). The engine builds every
portfolio from cash at the one common build close (D3), executes each
decision at the next close (D1), charges the D2 provisional costs on every
trade, the build included, and replays the same decisions at 0x, 1x and 2x
the rates (`simulate_cost_sensitivity`). 1x is the headline.

Measures (approved only; group C and D are not):

- growth of $10,000: the engine's pre-trade wealth path from the build close
  (to the cent), None where a held asset has no close or after a stop;
- total return: final wealth / initial wealth - 1 (not annualized), at the
  window's last session; unavailable for a run that stopped earlier;
- maximum drawdown (A6, F6.5) with `portfolio.descriptive.drawdown_summary`:
  the full-period maximum when every session is valued, else only the
  largest observed decline, labelled incomplete (R5c);
- calendar-year returns (A7), partial years labelled, never annualized;
- turnover (F2.4, one way, a full switch is 1): the build's, the rebalances'
  (summed, with their count and the turnover per rebalance), not annualized;
- costs (D2): the summed cost paid in dollars and as a fraction of initial
  wealth, the summed per-execution cost in basis points, and the final wealth
  at 0x, 1x and 2x the rates;
- each holding's own growth of $10,000 over the same span (F5.4: no cost),
  its total return and maximum drawdown, as context.

Every variant is computed and reported in registered order; nothing is
ranked or selected. A variant that cannot run as registered refuses the whole
run (`ShowcaseRunRefused`): a rule that cannot decide, or a build that cannot
execute at the registered close (it would wait in the funding leg, whose
return is not approved).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from stock_agent.portfolio.descriptive import (
    UNAVAILABLE,
    MetricUnavailable,
    calendar_year_returns,
    drawdown_summary,
    growth_index,
)
from stock_agent.portfolio.engine import (
    COMPLETE,
    SimulationResult,
    collect_targets,
    simulate_cost_sensitivity,
)
from stock_agent.portfolio.schedule import ScheduleSpec
from stock_agent.portfolio.transition import CASH_ACCOUNT, Constraints, CostModel
from stock_agent.rl.policies import PolicyInputError
from stock_agent.showcase.config import ShowcaseConfig, Strategy, Universe
from stock_agent.showcase.returns import FUNDING_LEG, RETURN_LABEL, ReturnTable, return_table
from stock_agent.showcase.strategies import make_policy

ARTIFACT = "showcase_results"
SCHEMA_VERSION = 1
HEADLINE = 1.0
MAXIMUM_DRAWDOWN = "maximum_drawdown"
OBSERVED_DECLINE = "largest_observed_decline_incomplete_history"

LABELS = (
    "Exploratory: a private look at real history, not a test of any strategy.",
    "Hindsight-selected: every list was chosen in 2026, knowing how 2003-2024 turned out.",
    (
        "Survivorship-biased: companies that failed, merged or were renamed are not in the lists, "
        "and the stock list's spin-off screen used 2003-2024 corporate history."
    ),
    "2003-2024 only: 2025 on is the sealed holdout and is never used.",
    "Not evidence of strategy skill, and not investment advice.",
    "Vendor-reported dividends, not issuer-verified; non-cash distributions are not included.",
    "Fund prices are market closes, not net asset values.",
    "Before taxes. Trading costs are the provisional D2 rates, with 0x and 2x shown.",
    "Fully invested at all times; there is no cash holding.",
)


class ShowcaseRunRefused(RuntimeError):
    """A variant that cannot run as registered; nothing is reported for the run."""


@dataclass(frozen=True)
class UniverseData:
    """A universe's return table and the lineage of its bars."""

    universe: Universe
    table: ReturnTable
    lineage: Mapping[str, list[dict]]
    revised_sessions: Mapping[str, int]


def _stamp(session: pd.Timestamp) -> str:
    return pd.Timestamp(session).strftime("%Y-%m-%d")


def _number(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def universe_data(
    universe: Universe, sessions: pd.DatetimeIndex, load: Callable[[str], object]
) -> UniverseData:
    """
    The universe's return table from `load(ticker)` (a `StoredBars`; see
    `showcase.store.window_bars`), with each ticker's batches as lineage.
    """

    bars, lineage, revised = {}, {}, {}
    for ticker in universe.tickers:
        stored = load(ticker)
        bars[ticker] = stored.bars.drop(columns=["batch_id"])
        lineage[ticker] = [
            {
                "batch_id": batch.batch_id,
                "source": batch.source,
                "retrieved_at": batch.retrieved_at,
                "first_session": batch.first_session,
                "last_session": batch.last_session,
                "content_sha256": batch.content_sha256,
            }
            for batch in stored.batches
        ]
        revised[ticker] = len(stored.revised_sessions)
    return UniverseData(universe, return_table(sessions, bars), lineage, revised)


def _wealth_series(table: ReturnTable, result: SimulationResult) -> pd.Series:
    index = table.sessions[list(result.sessions)]
    values = [np.nan if w is None else float(w) for w in result.wealth]
    return pd.Series(values, index=index, dtype="float64")


def _drawdown(growth: pd.Series) -> dict[str, object]:
    try:
        summary = drawdown_summary(growth)
    except MetricUnavailable as error:
        return {"measure": None, "value": None, "reason": error.reason}
    return {
        "measure": MAXIMUM_DRAWDOWN if summary.history_complete else OBSERVED_DECLINE,
        "value": summary.max_drawdown,
        "peak": _stamp(summary.peak_date),
        "trough": _stamp(summary.trough_date),
        "recovery": None if summary.recovery_date is None else _stamp(summary.recovery_date),
        "recovered": summary.recovered,
        "duration_sessions": summary.duration_sessions,
        "history_complete": summary.history_complete,
        "missing_sessions": summary.missing_sessions,
    }


def _calendar_years(growth: pd.Series, year_ends: pd.DatetimeIndex) -> list[dict]:
    frame = calendar_year_returns(growth, year_ends)
    return [
        {
            "year": int(row.year),
            "total_return": _number(row.total_return),
            "partial": bool(row.partial),
            "status": row.status,
            "reason": row.reason if isinstance(row.reason, str) else None,
        }
        for row in frame.itertuples()
    ]


def year_end_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last session of each calendar year of `sessions`."""

    frame = pd.Series(sessions, index=sessions)
    return pd.DatetimeIndex(frame.groupby(sessions.year).max().to_numpy())


def _activity(result: SimulationResult, initial: float) -> dict[str, object]:
    """Turnover (F2.4) and costs (D2) from the engine's steps; each step starts at a valued close."""

    wealth_at = dict(zip(result.sessions, result.wealth, strict=True))
    steps = list(zip(result.steps, result.spans, strict=True))
    paid = sum(step.cost * wealth_at[start] for step, (start, _) in steps)
    by_session = {start: step for step, (start, _) in steps}
    trades = [t.session for t in result.trades]
    turnovers = [by_session[s].turnover_one_way for s in trades if s in by_session]
    build = turnovers[0] if turnovers else 0.0
    rebalances = turnovers[1:]
    reasons = Counter(e.reason for e in result.executions if e.reason is not None)
    return {
        "turnover": {
            "build_one_way": build,
            "rebalance_one_way": float(sum(rebalances)),
            "rebalances": len(rebalances),
            "per_rebalance_one_way": float(sum(rebalances) / len(rebalances))
            if rebalances
            else None,
            "total_one_way": float(sum(step.turnover_one_way for step in result.steps)),
        },
        "costs": {
            "paid_dollars": float(paid),
            "paid_fraction_of_initial": float(paid / initial),
            "summed_bps": float(sum(step.cost for step in result.steps) * 1e4),
        },
        "executions": {
            "decisions": len(result.executions),
            "trades": len(trades),
            "by_reason": dict(sorted(reasons.items())),
        },
    }


def _stop_detail(table: ReturnTable, result: SimulationResult) -> dict[str, str] | None:
    """
    For a stopped run, why each holding's return at the session it stopped on is
    unavailable (the engine reports every such stop as `distribution_in_gap`).
    """

    if result.status == COMPLETE:
        return None
    stop = result.sessions[-1] + 1
    if stop >= len(table.sessions):
        return None
    detail = {}
    for ticker, frame in table.frames.items():
        row = frame.iloc[stop]
        if row["close_observed"] and row["status"] == UNAVAILABLE:
            detail[ticker] = str(row["reason"])
    return {"session": _stamp(table.sessions[stop]), "reasons": detail}


def run_strategy(
    config: ShowcaseConfig, data: UniverseData, strategy: Strategy
) -> dict[str, object]:
    """One registered variant: its simulations at 0x, 1x and 2x, and its measures."""

    table = data.table
    n_assets = len(table.assets)
    schedule = ScheduleSpec(
        every=strategy.every,
        offset=config.offset,
        lag=config.execution_lag,
        first_decision=config.first_decision,
    )
    policy = make_policy(
        strategy.rule, n_assets, table.cash_index, window=config.inverse_volatility_window
    )
    try:
        targets = collect_targets(policy, table.returns, schedule)
    except PolicyInputError as error:
        raise ShowcaseRunRefused(f"{strategy.key}: the rule cannot decide ({error})") from None
    classes = {**config.asset_classes(), FUNDING_LEG: CASH_ACCOUNT}
    costs = CostModel.provisional({asset: classes[asset] for asset in table.assets})
    results = simulate_cost_sensitivity(
        table.returns,
        targets,
        assets=table.assets,
        cost_model=costs,
        lag=config.execution_lag,
        constraints=Constraints(cash_index=table.cash_index),
        initial_wealth=config.initial_wealth,
        has_close=table.has_close,
    )
    planned = config.first_decision + config.execution_lag
    for result in results.values():
        # D3: the build executes at the registered close. A deferred build would sit in the
        # funding leg until it executes, and that leg's return is not approved.
        if not result.trades or result.trades[0].session != planned:
            raise ShowcaseRunRefused(
                f"{strategy.key}: the build could not execute at the registered close "
                f"{_stamp(table.sessions[planned])} (a holding has no close there)"
            )
    headline = results[HEADLINE]
    growth = _wealth_series(table, headline)
    # A stopped run has no total return over the window: its last value is not the window's end.
    final = headline.final_wealth if headline.status == COMPLETE else None
    sensitivity = {}
    for multiplier, result in results.items():
        end = result.final_wealth if result.status == COMPLETE else None
        sensitivity[f"{multiplier:g}x"] = {
            "final_wealth": _number(end),
            "total_return": None if end is None else end / config.initial_wealth - 1.0,
            "status": result.status,
        }
    return {
        "key": strategy.key,
        "label": strategy.label,
        "stop_detail": _stop_detail(table, headline),
        "description": strategy.description,
        "rule": strategy.rule,
        "every": strategy.every,
        "status": headline.status,
        "status_reason": headline.reason,
        "start": _stamp(growth.index[0]),
        "end": _stamp(growth.index[-1]),
        "final_wealth": _number(final),
        "total_return": None if final is None else final / config.initial_wealth - 1.0,
        "drawdown": _drawdown(growth),
        "calendar_years": _calendar_years(growth, year_end_sessions(table.sessions)),
        "cost_sensitivity": sensitivity,
        **_activity(headline, config.initial_wealth),
        "dates": [_stamp(day) for day in growth.index],
        "wealth": [None if not math.isfinite(v) else round(float(v), 2) for v in growth],
        "complete": headline.status == COMPLETE,
    }


def components(
    config: ShowcaseConfig, data: UniverseData, start: pd.Timestamp
) -> list[dict[str, object]]:
    """Each holding's own growth of $10,000 from `start` (F5.4: no cost), as context."""

    rows = []
    for member in data.universe.members:
        frame = data.table.frames[member.ticker].set_index("date").loc[start:]
        growth = growth_index(
            frame["total_return"], frame["sessions_spanned"], config.initial_wealth
        )
        end = float(growth.iloc[-1])
        rows.append(
            {
                "ticker": member.ticker,
                "sector": member.sector,
                "kind": member.kind,
                "total_return": None
                if not math.isfinite(end)
                else end / config.initial_wealth - 1.0,
                "drawdown": _drawdown(growth),
                "missing_sessions": data.table.missing_sessions[member.ticker],
                "unavailable_returns": data.table.unavailable[member.ticker],
                "revised_sessions": data.revised_sessions[member.ticker],
            }
        )
    return rows


def run_showcase(
    config: ShowcaseConfig,
    load: Callable[[str], object],
    *,
    sessions: pd.DatetimeIndex,
    code_sha: str | None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Every registered variant, in registered order, as one strict-JSON-ready record."""

    universes = []
    for universe in config.universes:
        data = universe_data(universe, sessions, load)
        strategies = [run_strategy(config, data, strategy) for strategy in config.strategies]
        build = pd.Timestamp(strategies[0]["start"], tz="UTC")
        if any(s["start"] != strategies[0]["start"] for s in strategies):
            raise ShowcaseRunRefused("every variant of a universe must build at the same close")
        dates = max((s.pop("dates") for s in strategies), key=len)
        for strategy in strategies:  # a stopped run's path ends early: no value after it
            strategy["wealth"] += [None] * (len(dates) - len(strategy["wealth"]))
        universes.append(
            {
                "key": universe.key,
                "label": universe.label,
                "selection": universe.selection,
                "members": [
                    {"ticker": m.ticker, "kind": m.kind, "sector": m.sector}
                    for m in universe.members
                ],
                "build": _stamp(build),
                "dates": dates,
                "components": components(config, data, build),
                "strategies": strategies,
                "lineage": dict(data.lineage),
            }
        )
    when = (generated_at or datetime.now(UTC)).astimezone(UTC)
    return {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "generated_at": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "code_sha": code_sha,
        "config": {"name": config.name, "sha256": config.sha256()},
        "window": {"start": config.start.isoformat(), "end": config.end.isoformat()},
        "variants_run": len(config.variants()),
        "simulations_run": len(config.variants()) * len(config.cost_multipliers),
        "cost_multipliers": list(config.cost_multipliers),
        "cost_classes": dict(config.cost_classes),
        "initial_wealth": config.initial_wealth,
        "return_label": RETURN_LABEL,
        "labels": list(LABELS),
        "universes": universes,
    }
