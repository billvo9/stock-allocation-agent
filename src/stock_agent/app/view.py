"""
Pure view logic for the Explore page: periods and their clamping to
coverage, data gaps, the calls to the metrics provider, and display
formatting. No Streamlit, no Plotly, and no financial formula: every metric
comes from the provider (`provider.DescriptiveMetrics`).

Point in time. A view is built only from records inside the selected
period, so nothing after the period's end reaches the provider or the page.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stock_agent.app import labels
from stock_agent.app.history import UNAVAILABLE, FundHistory
from stock_agent.app.provider import (
    CALENDAR_COLUMNS,
    GROWTH_START_VALUE,
    DescriptiveMetrics,
    DrawdownSummary,
    MetricUnavailable,
)

MINUS = "−"

FULL = "full"
LAST_10 = "last_10"
CRISIS_2008 = "crisis_2008"
YEAR_2022 = "year_2022"
CUSTOM = "custom"
PRESET_LABELS = {
    FULL: "Full history",
    LAST_10: "Last 10 years",
    CRISIS_2008: "2008 crisis (2007-2009)",
    YEAR_2022: "2022",
    CUSTOM: "Custom years",
}


# --- formatting ---


def money(value: float) -> str:
    """Whole US dollars with thousands separators: $12,345 or −$1,234."""

    text = f"${abs(value):,.0f}"
    return f"{MINUS}{text}" if value < 0 and text != "$0" else text


def percent(fraction: float, *, signed: bool = False) -> str:
    """One decimal: 5.0%, or +5.0% / −0.0% when signed. The sign is never rounded away."""

    fraction = fraction + 0.0  # -0.0 becomes 0.0
    text = f"{fraction:+.1%}" if signed else f"{fraction:.1%}"
    return text.replace("-", MINUS)


def day(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


# --- periods ---


@dataclass(frozen=True)
class Period:
    """The dates shown: the requested period clamped to the shared coverage."""

    start: pd.Timestamp
    end: pd.Timestamp
    label: str
    clamped: bool


def shared_coverage(histories: Sequence[FundHistory]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The dates every fund covers: the latest coverage start to the earliest coverage end."""

    if not histories or any(h.coverage_start is None for h in histories):
        return None
    start = max(h.coverage_start for h in histories)
    end = min(h.coverage_end for h in histories)
    return (start, end) if start <= end else None


def preset_years(preset: str, coverage: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[int, int]:
    """Calendar years a preset asks for; the clamp to coverage happens in `clamp_period`."""

    first, last = coverage[0].year, coverage[1].year
    years = {
        FULL: (first, last),
        LAST_10: (last - 9, last),
        CRISIS_2008: (2007, 2009),
        YEAR_2022: (2022, 2022),
    }
    if preset not in years:
        raise ValueError(f"no fixed years for preset {preset!r}")
    return years[preset]


def clamp_period(
    start_year: int,
    end_year: int,
    coverage: tuple[pd.Timestamp, pd.Timestamp],
    label: str,
) -> Period | None:
    """
    The calendar years [start_year, end_year] cut to `coverage`; None when
    they do not overlap it. `clamped` says whether the cut removed any date.
    """

    if start_year > end_year:
        raise ValueError("start_year is after end_year")
    requested_start = pd.Timestamp(year=start_year, month=1, day=1, tz="UTC")
    requested_end = pd.Timestamp(year=end_year, month=12, day=31, tz="UTC")
    start = max(requested_start, coverage[0])
    end = min(requested_end, coverage[1])
    if start > end:
        return None
    clamped = start != requested_start or end != requested_end
    return Period(start=start, end=end, label=label, clamped=clamped)


def preset_period(preset: str, coverage: tuple[pd.Timestamp, pd.Timestamp]) -> Period | None:
    if preset == FULL:
        return Period(coverage[0], coverage[1], PRESET_LABELS[FULL], clamped=False)
    start_year, end_year = preset_years(preset, coverage)
    return clamp_period(start_year, end_year, coverage, PRESET_LABELS[preset])


# --- gaps ---


@dataclass(frozen=True)
class Gap:
    """A run of consecutive unavailable records with one reason."""

    first: pd.Timestamp
    last: pd.Timestamp
    sessions: int
    reason: str


def gaps(history: FundHistory) -> tuple[Gap, ...]:
    records = history.records
    unavailable = records["status"].eq(UNAVAILABLE)
    if not unavailable.any():
        return ()
    key = records["status"].astype(str) + "|" + records["reason"].fillna("").astype(str)
    runs = key.ne(key.shift()).cumsum()
    found = []
    for _, run in records[unavailable].groupby(runs[unavailable], sort=True):
        found.append(
            Gap(run["date"].iloc[0], run["date"].iloc[-1], len(run), str(run["reason"].iloc[0]))
        )
    return tuple(found)


def unavailable_reasons(history: FundHistory) -> tuple[str, ...]:
    reasons = history.records.loc[history.records["status"].eq(UNAVAILABLE), "reason"]
    return tuple(sorted({str(reason) for reason in reasons}))


# --- the provider calls ---


@dataclass(frozen=True)
class Unavailable:
    """A metric that is not shown, with a machine-readable reason and words."""

    reason: str
    detail: str


@dataclass(frozen=True, eq=False)
class FundView:
    history: FundHistory  # restricted to the period
    gaps: tuple[Gap, ...]
    growth: pd.Series | Unavailable
    drawdown: DrawdownSummary | Unavailable
    calendar: pd.DataFrame | Unavailable


@dataclass(frozen=True, eq=False)
class ExploreView:
    period: Period
    start: pd.Timestamp | Unavailable  # t0, shared by every fund
    funds: tuple[FundView, ...]


def _attempt(call: Callable[..., Any], *args: Any) -> Any:
    try:
        return call(*args)
    except MetricUnavailable as missing:
        return Unavailable(missing.reason, missing.detail)


def returns_from(history: FundHistory, start: pd.Timestamp) -> pd.Series:
    """Daily total returns from `start` on, indexed by session date; gaps stay NaN."""

    records = history.records[history.records["date"].ge(start)]
    return pd.Series(
        records["daily_total_return"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(records["date"]),
        name=history.fund_id,
    )


def _checked_growth(growth: Any, returns: pd.Series) -> pd.Series | Unavailable:
    if isinstance(growth, Unavailable):
        return growth
    if not isinstance(growth, pd.Series) or not growth.index.equals(returns.index):
        raise ValueError("growth_index must return a series on the input's session dates")
    if growth.iloc[0] != GROWTH_START_VALUE:
        raise ValueError("growth_index must start at the start value")
    return growth


def _checked_calendar(table: Any) -> pd.DataFrame | Unavailable:
    if isinstance(table, Unavailable):
        return table
    if not isinstance(table, pd.DataFrame) or not set(CALENDAR_COLUMNS) <= set(table.columns):
        raise ValueError(f"calendar_year_returns must return the columns {CALENDAR_COLUMNS}")
    if table["year"].duplicated().any():
        raise ValueError("calendar_year_returns must return one row per year")
    return table


def build_view(
    histories: Sequence[FundHistory], period: Period, metrics: DescriptiveMetrics
) -> ExploreView:
    """
    Every metric for `histories` (funds with available data) over `period`.

    The shared start t0 comes from the provider's `common_start`; the page
    never picks it. A metric whose input is unavailable is unavailable for
    the same reason.
    """

    windowed = tuple(history.between(period.start, period.end) for history in histories)
    start = _attempt(metrics.common_start, windowed, period.start)
    if start is None:
        start = Unavailable(labels.NO_COMMON_START_REASON, labels.NO_COMMON_START)
    elif not isinstance(start, Unavailable) and not period.start <= start <= period.end:
        raise ValueError("common_start returned a session outside the period")
    funds = []
    for history in windowed:
        if isinstance(start, Unavailable):
            growth = start
        else:
            returns = returns_from(history, start)
            growth = _checked_growth(
                _attempt(metrics.growth_index, returns, GROWTH_START_VALUE), returns
            )
        if isinstance(growth, Unavailable):
            drawdown, calendar = growth, growth
        else:
            drawdown = _attempt(metrics.drawdown_summary, growth)
            calendar = _checked_calendar(_attempt(metrics.calendar_year_returns, growth))
        funds.append(FundView(history, gaps(history), growth, drawdown, calendar))
    return ExploreView(period=period, start=start, funds=tuple(funds))


# --- tables for the page ---


def last_value(growth: pd.Series) -> tuple[pd.Timestamp, float] | None:
    """The last available value of a growth series and its date."""

    available = growth.dropna()
    return None if available.empty else (available.index[-1], float(available.iloc[-1]))


def year_end_table(funds: Sequence[tuple[str, pd.Series]]) -> pd.DataFrame:
    """
    The accessible table behind the growth chart: each series at its start
    and at the last session of every calendar year, as dollar text.
    """

    points: set[pd.Timestamp] = set()
    for _, growth in funds:
        if growth.empty:
            continue
        index = growth.index
        points.add(index[0])
        points.update(pd.Series(index, index=index).groupby(index.year).max())
    dates = sorted(points)
    table = {"Session date": [day(date) for date in dates]}
    for name, growth in funds:
        values = growth.reindex(dates)
        table[name] = ["—" if pd.isna(value) else money(value) for value in values]
    return pd.DataFrame(table)


def calendar_table(funds: Sequence[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Calendar-year returns as text, one row per year and one column per fund.
    A losing year says so in words; a partial year is labelled. Both flags
    come from the provider, never from the rounded text.
    """

    years = sorted({int(year) for _, table in funds for year in table["year"]})
    rows: dict[str, list[object]] = {"Year": years}
    for name, table in funds:
        by_year = table.set_index(table["year"].astype(int))
        cells = []
        for year in years:
            if year not in by_year.index:
                cells.append("—")
                continue
            row = by_year.loc[year]
            parts = [percent(float(row["total_return"]), signed=True)]
            if bool(row["losing"]):
                parts.append("lost money")
            if bool(row["partial"]):
                parts.append("partial year")
            cells.append(" · ".join(parts))
        rows[name] = cells
    return pd.DataFrame(rows)


def losing_summary(name: str, table: pd.DataFrame) -> str:
    """'Lost money in 2008 and 2022: 2 of 18 calendar years shown (1 partial).'"""

    losing = sorted(int(year) for year in table.loc[table["losing"].astype(bool), "year"])
    shown = len(table)
    partial = int(table["partial"].astype(bool).sum())
    partial_text = f" ({partial} partial)" if partial else ""
    if not losing:
        return f"{name}: no calendar year shown lost money (0 of {shown}{partial_text})."
    years = ", ".join(str(year) for year in losing)
    return (
        f"{name}: lost money in {years}: {len(losing)} of {shown} calendar years shown"
        f"{partial_text}."
    )
