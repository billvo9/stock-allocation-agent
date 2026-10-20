"""
The outcome reader (decision A4's scoring, inside decision A2's isolation):
what one recorded forecast is scored on, and when each stock's newest close
was stored. It never imports the writer, opens the store read-only and
computes no return, category or score (scoring.score_forecast does that).

Contract
    read_outcomes(store_path, ticker, *, as_of_session, target_sessions,
                  known_at, scored_by, sessions, calendar_id) -> Outcomes

    as_of_session, target_sessions
                    the forecast's, as its record names them: the target
                    sessions must be exactly the calendar sessions after the
                    as-of session
    known_at        when the forecast was known: the later of its created_at
                    and the time the research store recorded it
    scored_by       the scoring job's decision time: only batches retrieved
                    AND recorded by then are read, so a re-run with the same
                    value reads the same outcomes; it cannot be before
                    known_at

    Only the calendar sessions from the as-of session (the first return's
    base) through the last target session are returned, never an earlier or
    a later one:
        bars          prices.BAR_COLUMNS, one row per span session some
                      eligible batch covers (the vintage rule: per session,
                      the latest retrieval); a session no batch covers yet is
                      absent, so its targets stay pending
        batch_of      session (YYYY-MM-DD) -> the batch that supplied it
        seen_before   every span session that a batch retrieved by known_at
                      already covered: a target needing one is never scored
                      (it was available to the forecast, or could have been)
        revised       span sessions whose values differ between batches

    latest_close(store_path, ticker, *, retrieved_by, calendar_id)
        the newest session with a close in the batches retrieved and recorded
        by `retrieved_by` (the vintage rule), with its retrieval time; None
        when nothing is stored. For freshness only.

Refusals (`InferenceRefused`): invalid_request (targets that do not follow the
as-of session on the calendar; scored_by before known_at), no_store,
not_inference_store, location, calendar_mismatch, integrity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.inference.schema import (
    InferenceBatch,
    InferenceRefused,
    Refusal,
    Sessions,
    bars_of,
    batches,
    calendar,
    eligible_batches,
    read_only,
    select_vintage,
    session_day,
    utc_time,
)
from stock_agent.stock_lab.prices import BAR_COLUMNS

LATEST_SPAN = pd.Timedelta(days=21)  # the newest close lies within three weeks of the newest row


@dataclass(frozen=True, eq=False)
class Outcomes:
    ticker: str
    span: pd.DatetimeIndex
    bars: pd.DataFrame
    batch_of: dict[str, str]
    seen_before: frozenset[str]
    revised: tuple[str, ...]
    batches: tuple[InferenceBatch, ...]


@dataclass(frozen=True)
class LatestClose:
    ticker: str
    session: str
    retrieved_at: str
    batch_id: str


def _span(sessions: Sessions, as_of: pd.Timestamp, targets: Sequence[str]) -> pd.DatetimeIndex:
    days = [session_day(day, "target session") for day in targets]
    if not days or days[0] <= as_of:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the targets must follow the as-of session")
    found = calendar(sessions, as_of, days[-1])
    if list(found) != [as_of, *days]:
        raise InferenceRefused(
            Refusal.INVALID_REQUEST,
            "the target sessions are not the calendar sessions after the as-of session",
        )
    return found


def read_outcomes(
    store_path: str | Path,
    ticker: str,
    *,
    as_of_session: object,
    target_sessions: Sequence[str],
    known_at: pd.Timestamp,
    scored_by: pd.Timestamp,
    sessions: Sessions,
    calendar_id: str,
) -> Outcomes:
    """One forecast's outcomes; the contract is in the module docstring."""

    known = utc_time(known_at, "known_at")
    decision = utc_time(scored_by, "scored_by")
    if decision < known:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "scored_by is before the forecast existed")
    as_of = session_day(as_of_session, "as_of_session")
    span = _span(sessions, as_of, target_sessions)
    first, last = span[0].strftime("%Y-%m-%d"), span[-1].strftime("%Y-%m-%d")
    with read_only(Path(store_path)) as connection:
        if connection is None:
            raise InferenceRefused(Refusal.NO_STORE, "no inference store exists yet")
        stored = batches(connection, ticker)
        eligible = [
            batch
            for batch in eligible_batches(stored, decision, calendar_id)
            if batch.request_start <= last and batch.request_end >= first
        ]
        frame = bars_of(connection, eligible)
    frame = frame[frame["session"].isin(span)]
    seen = {
        day.strftime("%Y-%m-%d")
        for batch in eligible
        if pd.Timestamp(batch.retrieved_at) <= known
        for day in span
        if batch.covers(day)
    }
    if frame.empty:
        bars = pd.DataFrame(columns=list(BAR_COLUMNS))
        return Outcomes(ticker, span, bars, {}, frozenset(seen), (), ())
    vintage = select_vintage(eligible, frame)
    chosen = vintage.bars.reset_index(drop=True)
    batch_of = {
        row.session.strftime("%Y-%m-%d"): row.batch_id for row in chosen.itertuples(index=False)
    }
    return Outcomes(
        ticker=ticker,
        span=span,
        bars=chosen[list(BAR_COLUMNS)],
        batch_of=batch_of,
        seen_before=frozenset(seen),
        revised=vintage.revised,
        batches=vintage.used,
    )


def latest_close(
    store_path: str | Path, ticker: str, *, retrieved_by: pd.Timestamp, calendar_id: str
) -> LatestClose | None:
    """The newest stored close of `ticker` by `retrieved_by` (module docstring)."""

    decision = utc_time(retrieved_by, "retrieved_by")
    with read_only(Path(store_path)) as connection:
        if connection is None:
            return None
        eligible = eligible_batches(batches(connection, ticker), decision, calendar_id)
        if not eligible:
            return None
        newest = max(batch.request_end for batch in eligible)
        floor = (pd.Timestamp(newest) - LATEST_SPAN).strftime("%Y-%m-%d")
        recent = [batch for batch in eligible if batch.request_end >= floor]
        frame = bars_of(connection, recent)
    vintage = select_vintage(recent, frame[frame["session"] >= pd.Timestamp(floor, tz="UTC")])
    closed = vintage.bars[vintage.bars["close"].notna()]
    if closed.empty:
        return None
    row = closed.iloc[-1]
    batch = next(b for b in vintage.used if b.batch_id == row["batch_id"])
    return LatestClose(
        ticker, row["session"].strftime("%Y-%m-%d"), batch.retrieved_at, batch.batch_id
    )
