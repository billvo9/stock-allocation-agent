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

Measures (approved only: OD8 groups A and B, and the group D measures the
configuration registers; group C and the rest of group D are not built yet):

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
  its total return and maximum drawdown, as context;
- with a registered `offset_family` (F7.2, T5a D7): each rebalancing rule at
  all k offsets of its cadence at 1x costs, reported as the registered
  offset's total return and the max - min spread across the k offsets. One variant with a k-member family: `variants_run` does
  not count the offsets, and the registered offset alone is the headline.

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
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.portfolio import performance
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
    simulate,
    simulate_cost_sensitivity,
)
from stock_agent.portfolio.schedule import ScheduleSpec, offset_family
from stock_agent.portfolio.transition import CASH_ACCOUNT, DUST, Constraints, CostModel
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
    dollar_volume: np.ndarray | None = None  # raw close x raw volume, the funding leg NaN (M7)


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
    table = return_table(sessions, bars)
    volume = np.full(table.returns.shape, np.nan)
    for column, ticker in enumerate(universe.tickers):
        frame = bars[ticker].set_index(pd.DatetimeIndex(bars[ticker]["session"]))
        dollars = (frame["close"] * frame["volume"]).reindex(sessions)
        volume[:, column] = dollars.to_numpy(dtype="float64")
    return UniverseData(universe, table, lineage, revised, volume)


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


def _targets(config: ShowcaseConfig, table: ReturnTable, strategy: Strategy, offset: int):
    schedule = ScheduleSpec(
        every=strategy.every,
        offset=offset,
        lag=config.execution_lag,
        first_decision=config.first_decision,
    )
    policy = make_policy(
        strategy.rule, len(table.assets), table.cash_index, window=config.inverse_volatility_window
    )
    try:
        return schedule, collect_targets(policy, table.returns, schedule)
    except PolicyInputError as error:
        raise ShowcaseRunRefused(f"{strategy.key}: the rule cannot decide ({error})") from None


def _check_build(config: ShowcaseConfig, table: ReturnTable, strategy: Strategy, result) -> None:
    # D3: the build executes at the registered close. A deferred build would sit in the
    # funding leg until it executes, and that leg's return is not approved.
    planned = config.first_decision + config.execution_lag
    if not result.trades or result.trades[0].session != planned:
        raise ShowcaseRunRefused(
            f"{strategy.key}: the build could not execute at the registered close "
            f"{_stamp(table.sessions[planned])} (a holding has no close there)"
        )


def _spread(values: list[float | None], registered: float | None) -> dict[str, object]:
    """F7.2's report: the registered offset's value and the max - min spread (no min or max)."""

    known = [v for v in values if v is not None]
    return {
        "registered": registered,
        "spread": max(known) - min(known) if len(known) == len(values) and known else None,
    }


def _offset_family(
    config: ShowcaseConfig,
    table: ReturnTable,
    strategy: Strategy,
    costs: CostModel,
    schedule: ScheduleSpec,
    headline: SimulationResult,
) -> dict[str, object] | None:
    """
    F7.2: the rule at every one of its k offsets, at the headline (1x) costs: the
    registered offset's total return and the max - min spread of the total
    return across the k offsets, nothing more (which offset was highest or
    lowest is never recorded). One variant with a k-member sensitivity family;
    no per-offset interval, and the family never chooses the headline (the
    registered offset's run is the headline run itself). The spread is
    unavailable when any offset stopped early.
    """

    if not config.offset_family:
        return None
    if strategy.rule == "buy_and_hold":
        return {"applies": False, "k": strategy.every, "reason": "no_rebalances_after_the_build"}
    returns, stopped = [], 0
    registered = None
    for spec in offset_family(schedule):
        if spec.offset == schedule.offset:
            result = headline
        else:
            _, targets = _targets(config, table, strategy, spec.offset)
            result = simulate(
                table.returns,
                targets,
                lag=config.execution_lag,
                cost_rate=costs.rates_for(table.assets),
                constraints=Constraints(cash_index=table.cash_index),
                initial_wealth=config.initial_wealth,
                has_close=table.has_close,
            )
            _check_build(config, table, strategy, result)
        complete = result.status == COMPLETE
        stopped += not complete
        total = result.final_wealth / config.initial_wealth - 1.0 if complete else None
        returns.append(total)
        if spec.offset == schedule.offset:
            registered = total
    return {
        "applies": True,
        "k": strategy.every,
        "registered_offset": schedule.offset,
        "total_return": _spread(returns, registered),
        "stopped_offsets": stopped,
    }


def _year_bounds(sessions: pd.DatetimeIndex) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Each calendar year's first and last NYSE session (the calendar's, not the window's)."""

    first, last = nyse.COVERAGE.first.date(), nyse.COVERAGE.last.date()
    bounds = {}
    for year in sorted(set(sessions.year)):
        days = nyse.sessions(max(date(year, 1, 1), first), min(date(year, 12, 31), last))
        bounds[int(year)] = (days[0], days[-1])
    return bounds


def _group_d(
    config: ShowcaseConfig, data: UniverseData, result: SimulationResult, growth: pd.Series
) -> dict[str, object] | None:
    """
    The group D measures of the headline run (approved 2026-10-26; `portfolio.performance`):
    M1 CAGR, M2 volatility, M5 the registered stress windows, M6 turnover per calendar year
    and M7 the participation flag. None unless the configuration registers `measures`.
    """

    measures = config.measures
    if measures is None:
        return None
    table = data.table
    initial = config.initial_wealth
    complete = result.status == COMPLETE
    valued = growth.dropna()
    if not complete:
        cagr = {"value": None, "annualized": False, "reason": "stopped", "days": None}
    else:  # M1: to the last valued close, named when it is not the window's last session
        cagr = {
            **performance.cagr(initial, float(valued.iloc[-1]), growth.index[0], valued.index[-1]),
            "end": _stamp(valued.index[-1]),
            "reason": None,
        }
    volatility = performance.annual_volatility(
        [step.net_return for step in result.steps], [end - start for start, end in result.spans]
    )
    windows = [
        performance.stress_window(
            growth,
            performance.StressWindow(
                w.key, w.label, pd.Timestamp(w.start, tz="UTC"), pd.Timestamp(w.end, tz="UTC")
            ),
        )
        for w in measures.stress_windows
    ]
    for window in windows:
        ended_early = not complete and window["end"] > growth.index[-1]
        if window["reason"] == performance.OUTSIDE_WINDOW and ended_early:
            window["reason"] = "run_stopped_before_window_end"  # inside the registered span
        window["start"], window["end"] = _stamp(window["start"]), _stamp(window["end"])
    by_session = {start: step for step, (start, _) in zip(result.steps, result.spans, strict=True)}
    executions = [
        performance.Execution(table.sessions[t.session], t.feasibility, by_session[t.session])
        for t in result.trades
        if t.session in by_session
    ]
    turnover = performance.turnover_by_year(
        executions,
        first_session=growth.index[0],
        last_session=growth.index[-1],
        year_bounds=_year_bounds(table.sessions),
    )
    wealth_at = dict(zip(result.sessions, result.wealth, strict=True))
    pre = np.zeros(len(table.assets))
    pre[table.cash_index] = 1.0
    trades, previous = [], {}
    for step, (start, end) in zip(result.steps, result.spans, strict=True):
        previous[start] = pre
        pre = np.asarray(step.w_next)
    for trade in result.trades:
        step = by_session.get(trade.session)
        if step is None:
            continue
        delta = np.asarray(step.w_post_trade) - previous[trade.session]
        delta[np.abs(delta) < DUST] = 0.0  # the engine's own no-trade threshold
        trades.append((trade.decision, delta, wealth_at[trade.session] / initial))
    flag = performance.participation(
        trades,
        data.dollar_volume,
        table.cash_index,
        notional=measures.notional,
        threshold=measures.threshold,
        window=measures.volume_window,
    )
    return {
        "cagr": cagr,
        "volatility": volatility,
        "stress_windows": windows,
        "turnover_by_year": {
            **turnover,
            "years": [{**row, "year": int(row["year"])} for row in turnover["years"]],
        },
        "participation": {
            **flag,
            "notional": measures.notional,
            "threshold": measures.threshold,
            "window": measures.volume_window,
        },
    }


def run_strategy(
    config: ShowcaseConfig, data: UniverseData, strategy: Strategy
) -> dict[str, object]:
    """One registered variant: its simulations at 0x, 1x and 2x, and its measures."""

    table = data.table
    schedule, targets = _targets(config, table, strategy, config.offset)
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
    for result in results.values():
        _check_build(config, table, strategy, result)
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
        "offset_family": _offset_family(config, table, strategy, costs, schedule, headline),
        "group_d": _group_d(config, data, headline, growth),
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
        "offset_family": config.offset_family,
        "group_d": config.measures is not None,
        "offset_family_simulations": sum(
            s["offset_family"]["k"] - 1
            for u in universes
            for s in u["strategies"]
            if s["offset_family"] and s["offset_family"]["applies"]
        ),
        "simulations_run": len(config.variants()) * len(config.cost_multipliers),
        "cost_multipliers": list(config.cost_multipliers),
        "cost_classes": dict(config.cost_classes),
        "initial_wealth": config.initial_wealth,
        "return_label": RETURN_LABEL,
        "labels": list(LABELS),
        "universes": universes,
    }
