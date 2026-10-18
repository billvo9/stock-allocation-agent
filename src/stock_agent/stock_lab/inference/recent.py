"""
The isolated descriptive reader for the Lab page (decision A6): one stock's
recent observed prices from the inference store, for display only. It never
imports the writer or the frozen-model reader, opens the store read-only,
and creates nothing.

What it gives (`RecentPrices`)
    state      "observed", or why nothing is shown: "no_store" (no inference
               store yet), "no_data" (nothing stored for this stock) or
               "unreadable" (the file is refused; `reason` says why)
    sessions   up to `count` (default RECENT_SESSIONS, at most 252) of the
               latest observed sessions, oldest first, each with its raw close,
               the split-consistent close on the last shown session's share
               basis (for a chart), the price change from the previous session
               and plain notes: gap, ex-dividend, split, large move, revised.
               A session the provider did not serve stays a gap, and no change
               is computed across it.
    lineage    the newest retrieval time, the source, the batch ids, the counts
               of revised sessions and gaps, the data class

What it never gives
    No category (Increase, Stable or Decrease at a band), feature, streak,
    probability or score is computed on protected-period prices: those are
    the models' inputs and targets, and A6 only shows what was observed. The
    page shows these prices apart from the research history and never as
    research evidence.

Current forecasts and completed outcomes
    Given the stock's latest forecast (its as-of session and its five target
    sessions, as the forecast record names them), each shown session that is
    one of those targets is marked `forecast_target`: it is that forecast's
    outcome, scored only in the forecast history, never here. Later sessions
    are not marked: no forecast covers them.

Which rows
    The vintage rule (`schema`): per session, the latest retrieval that covers
    it. Only the latest contiguous stretch is shown: two neighbouring sessions
    belong to one stretch when one stored request covers both, so a span that
    was never requested is never bridged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.stock_lab.inference.schema import (
    INFERENCE_SESSIONS,
    InferenceBatch,
    InferenceRefused,
    bars_of,
    batches,
    data_class_of,
    default_path,
    read_only,
    select_vintage,
    session_day,
)
from stock_agent.stock_lab.prices import session_returns

RECENT_SESSIONS = 60
OBSERVED = "observed"
NO_STORE = "no_store"
NO_DATA = "no_data"
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class RecentSession:
    session: str  # YYYY-MM-DD
    close: float | None  # the raw close as served; None for a gap
    level: float | None  # the close on the last shown session's share basis
    change: float | None  # the price change from the previous session
    notes: tuple[str, ...]
    forecast_target: bool  # a target session of the latest forecast: its outcome


@dataclass(frozen=True)
class RecentPrices:
    ticker: str
    state: str
    reason: str | None
    sessions: tuple[RecentSession, ...]
    first_session: str | None
    last_session: str | None
    last_close: float | None
    retrieved_at: str | None
    source: str | None
    batch_ids: tuple[str, ...]
    revised: int
    gaps: int
    data_class: str | None
    forecast_as_of: str | None
    forecast_targets: tuple[str, ...]


def _empty(ticker: str, state: str, reason: str | None = None) -> RecentPrices:
    return RecentPrices(
        ticker, state, reason, (), None, None, None, None, None, (), 0, 0, None, None, ()
    )


def _stretch_start(sessions: pd.Series, stored: Sequence[InferenceBatch], count: int) -> int:
    """
    The first row of the latest contiguous stretch (module docstring), looking
    back no further than `count` rows plus the one before them (for the first
    shown change).
    """

    days = pd.DatetimeIndex(sessions)
    ranges = [
        (pd.Timestamp(batch.request_start, tz="UTC"), pd.Timestamp(batch.request_end, tz="UTC"))
        for batch in stored
    ]
    start = len(days) - 1
    floor = max(len(days) - count - 1, 0)
    while start > floor and any(
        first <= days[start - 1] and days[start] <= last for first, last in ranges
    ):
        start -= 1
    return start


def _forecast(as_of: str | None, targets: Sequence[str]) -> tuple[str | None, tuple[str, ...]]:
    """The latest forecast's as-of session and its target sessions, checked (dates, in order)."""

    if as_of is None:
        if targets:
            raise ValueError("forecast targets need the forecast's as-of session")
        return None, ()
    try:
        days = [session_day(day, "forecast session") for day in (as_of, *targets)]
    except InferenceRefused as error:
        raise ValueError(error.detail) from None
    if not targets or any(later <= earlier for earlier, later in pairwise(days)):
        raise ValueError("forecast targets must follow the as-of session, in order")
    texts = tuple(day.strftime("%Y-%m-%d") for day in days)
    return texts[0], texts[1:]


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def read_recent(
    ticker: str,
    *,
    path: Path | None = None,
    count: int = RECENT_SESSIONS,
    forecast_as_of: str | None = None,
    forecast_targets: Sequence[str] = (),
) -> RecentPrices:
    """The stock's recent observed prices; see the module docstring."""

    if type(count) is not int or not 1 <= count <= INFERENCE_SESSIONS:
        raise ValueError(f"count must be an int from 1 to {INFERENCE_SESSIONS}")
    as_of, targets = _forecast(forecast_as_of, forecast_targets)
    try:
        with read_only(path or default_path()) as connection:
            if connection is None:
                return _empty(ticker, NO_STORE)
            stored = batches(connection, ticker)
            if not stored:
                return _empty(ticker, NO_DATA)
            frame = bars_of(connection, stored)
        data_class = data_class_of(stored)
    except InferenceRefused as error:
        return _empty(ticker, UNREADABLE, error.reason.value)

    vintage = select_vintage(stored, frame)
    bars = vintage.bars.reset_index(drop=True)
    bars = bars.iloc[_stretch_start(bars["session"], stored, count) :].reset_index(drop=True)
    returns = session_returns(bars)  # no band: no category is ever computed here
    basis = float(np.prod(bars["split_factor"].to_numpy(dtype="float64")))
    shown = slice(max(len(bars) - count, 0), len(bars))
    revised = set(vintage.revised)
    rows = []
    for index in range(shown.start, shown.stop):
        day = bars["session"].iloc[index]
        text = day.strftime("%Y-%m-%d")
        flags = returns.iloc[index]
        notes = [
            note
            for note, present in (
                ("gap", pd.isna(bars["close"].iloc[index])),
                ("ex-dividend", bool(flags["ex_dividend"])),
                ("split", bool(flags["split_applied"])),
                ("large move", bool(flags["large_move"])),
                ("revised", text in revised),
            )
            if present
        ]
        rows.append(
            RecentSession(
                session=text,
                close=_number(bars["close"].iloc[index]),
                level=_number(flags["level"] / basis),
                change=_number(flags["price_return"]),
                notes=tuple(notes),
                forecast_target=text in targets,
            )
        )
    used = {bars["batch_id"].iloc[index] for index in range(shown.start, shown.stop)}
    lineage = [batch for batch in vintage.used if batch.batch_id in used]
    newest = lineage[-1]
    return RecentPrices(
        ticker=ticker,
        state=OBSERVED,
        reason=None,
        sessions=tuple(rows),
        first_session=rows[0].session,
        last_session=rows[-1].session,
        last_close=rows[-1].close,
        retrieved_at=newest.retrieved_at,
        source=newest.source,
        batch_ids=tuple(batch.batch_id for batch in lineage),
        revised=sum("revised" in row.notes for row in rows),
        gaps=sum("gap" in row.notes for row in rows),
        data_class=data_class,
        forecast_as_of=as_of,
        forecast_targets=targets,
    )
