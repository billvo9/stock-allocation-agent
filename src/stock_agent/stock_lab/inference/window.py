"""
The frozen-model reader (decision A2): a previously frozen model's inputs,
and nothing else. It never imports the writer and opens the store read-only.

Contract
    load_inference_window(store_path, ticker, as_of_session,
                          lookback=INFERENCE_SESSIONS, *, retrieved_by,
                          sessions, calendar_id) -> InferenceWindow

    store_path      the inference store's file (`schema.DB_NAME`; there is no
                    default: the caller names the store it reads)
    ticker          one stock, as stored
    as_of_session   the session whose close the forecast uses: a YYYY-MM-DD
                    string, a date, or a naive or UTC midnight
    lookback        exactly INFERENCE_SESSIONS (252); any other value is
                    refused (`wrong_lookback`)
    retrieved_by    required, timezone-aware: the forecast's decision time.
                    Only batches retrieved AND recorded at or before it are
                    read, so a re-run with the same value reproduces the
                    inputs, and a batch that arrived later is never seen
    sessions        the exchange calendar, `sessions(start, end)` (the
                    owner-run script passes the NYSE rules calendar)
    calendar_id     that calendar's id; every batch read must carry it

    The window is the 252 calendar sessions ending at the as-of session,
    both ends included: 252 closes, 251 session returns, enough for the
    longest feature window (`features.LONGEST_WINDOW`).

Refusals (`InferenceRefused`, with a `schema.Refusal` reason)
    invalid_request          a malformed argument, or an as-of that is not a
                             session of the calendar
    wrong_lookback           a lookback other than 252
    before_protected_period  the window would start before 2025-01-01: older
                             sessions are development data, never read here
    close_not_available      retrieved_by is earlier than the as-of close's
                             availability (20:00 America/New_York)
    after_next_open          retrieved_by is at or after the next session's
                             open (09:30 America/New_York): by then the as-of
                             close is history, and a "decision" would be a
                             retrospective forecast scored on outcomes that
                             already exist.
    no_store, not_inference_store, location, calendar_mismatch, integrity
                             (a batch whose rows break its count, hash or
                             request range, or that was retrieved before its
                             last close was available; mixed data classes)
    short_window             a session of the window that no eligible batch
                             covers (never padded, never shortened)
    as_of_close_missing      the as-of session was served without a close

What is returned (`InferenceWindow`)
    bars              prices.BAR_COLUMNS plus batch_id, exactly one row per
                      window session, in order. A session the provider did
                      not serve is a row without prices, listed in `gaps`;
                      gaps are never filled.
    sessions, as_of_session, retrieved_by, lookback, calendar_id, data_class
    batches           the batches that supplied a session (lineage)
    revised_sessions  sessions whose values differ between eligible batches

What the reader cannot know
    It reads no clock, so `retrieved_by` is the caller's claim. A caller that
    passed past decision times would rebuild past windows (only from what was
    stored before each one, never from a later vintage). Two other rules keep
    that from becoming a backtest: only reviewed modules may import this
    reader (the forecast job, at A4, must take `retrieved_by` from its clock),
    and the forecast record refuses a `created_at` away from the store's
    clock, so no forecast can be recorded retrospectively.

Point in time
    Every row is dated within [sessions[0], as_of_session]; later rows of an
    eligible batch are never returned. Values are as the provider served
    them at retrieval, not a point-in-time vintage of the provider. Nothing
    here reads a clock or the network, adjusts a close or computes a return.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.inference.schema import (
    INFERENCE_SESSIONS,
    PROTECTED_START,
    InferenceBatch,
    InferenceRefused,
    Refusal,
    Sessions,
    bars_of,
    batches,
    calendar,
    close_available_at,
    data_class_of,
    opens_at,
    read_only,
    select_vintage,
    session_day,
    utc_time,
)

CALENDAR_SPAN = pd.Timedelta(days=INFERENCE_SESSIONS * 2)  # enough calendar days for the window
NEXT_SPAN = pd.Timedelta(days=14)  # the next session after an as-of lies within two weeks


@dataclass(frozen=True, eq=False)
class InferenceWindow:
    """A frozen model's inputs; every field is described in the module docstring."""

    ticker: str
    as_of_session: pd.Timestamp
    retrieved_by: pd.Timestamp
    lookback: int
    sessions: pd.DatetimeIndex
    bars: pd.DataFrame
    gaps: tuple[str, ...]
    revised_sessions: tuple[str, ...]
    batches: tuple[InferenceBatch, ...]
    data_class: str
    calendar_id: str


def _window_sessions(sessions: Sessions, as_of: pd.Timestamp, lookback: int) -> pd.DatetimeIndex:
    found = calendar(sessions, as_of - CALENDAR_SPAN, as_of)
    if len(found) == 0 or found[-1] != as_of:
        raise InferenceRefused(
            Refusal.INVALID_REQUEST, f"{as_of.date()} is not a session of the calendar"
        )
    if len(found) < lookback:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the calendar is too short for the window")
    return found[-lookback:]


def _next_session(sessions: Sessions, as_of: pd.Timestamp) -> pd.Timestamp:
    upcoming = calendar(sessions, as_of + pd.Timedelta(days=1), as_of + NEXT_SPAN)
    if len(upcoming) == 0:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the calendar has no next session")
    return upcoming[0]


def _eligible(
    stored: tuple[InferenceBatch, ...], decision: pd.Timestamp, calendar_id: str
) -> list[InferenceBatch]:
    """Batches retrieved and recorded by the decision time, each re-checked."""

    eligible = []
    for batch in stored:
        if (
            pd.Timestamp(batch.recorded_at) > decision
            or pd.Timestamp(batch.retrieved_at) > decision
        ):
            continue
        if batch.calendar_id != calendar_id:
            raise InferenceRefused(
                Refusal.CALENDAR_MISMATCH,
                f"batch {batch.batch_id} follows {batch.calendar_id}, not {calendar_id}",
            )
        end = session_day(batch.request_end, "request_end")
        if pd.Timestamp(batch.retrieved_at) < close_available_at(end):
            raise InferenceRefused(
                Refusal.INTEGRITY,
                f"batch {batch.batch_id} claims a retrieval before its last close was available",
            )
        if pd.Timestamp(batch.retrieved_at) > pd.Timestamp(batch.recorded_at):
            raise InferenceRefused(
                Refusal.INTEGRITY, f"batch {batch.batch_id} was recorded before it was retrieved"
            )
        eligible.append(batch)
    return eligible


def load_inference_window(
    store_path: str | Path,
    ticker: str,
    as_of_session: object,
    lookback: int = INFERENCE_SESSIONS,
    *,
    retrieved_by: pd.Timestamp,
    sessions: Sessions,
    calendar_id: str,
) -> InferenceWindow:
    """The inputs for a forecast as of `as_of_session`; the contract is in the module docstring."""

    if not isinstance(ticker, str) or not ticker or ticker != ticker.strip().upper():
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"not a stored ticker: {ticker!r}")
    if type(lookback) is not int:
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"lookback {lookback!r} is not an int")
    if lookback != INFERENCE_SESSIONS:
        raise InferenceRefused(
            Refusal.WRONG_LOOKBACK,
            f"the frozen model reads exactly {INFERENCE_SESSIONS} sessions, not {lookback}",
        )
    if not isinstance(calendar_id, str) or not calendar_id:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "calendar_id must be named")
    decision = utc_time(retrieved_by, "retrieved_by")
    as_of = session_day(as_of_session, "as_of_session")
    window = _window_sessions(sessions, as_of, lookback)
    if window[0].date() < PROTECTED_START:
        raise InferenceRefused(
            Refusal.BEFORE_PROTECTED_PERIOD,
            f"a {lookback}-session window ending {as_of.date()} starts {window[0].date()}, "
            "inside development data",
        )
    if decision < close_available_at(as_of):
        raise InferenceRefused(
            Refusal.CLOSE_NOT_AVAILABLE,
            f"the {as_of.date()} close is not available until 20:00 America/New_York",
        )
    following = _next_session(sessions, as_of)
    if decision >= opens_at(following):
        raise InferenceRefused(
            Refusal.AFTER_NEXT_OPEN,
            f"the decision is after the {following.date()} open: a forecast as of "
            f"{as_of.date()} can no longer be made, only reconstructed",
        )

    with read_only(Path(store_path)) as connection:
        if connection is None:
            raise InferenceRefused(Refusal.NO_STORE, "no inference store exists yet")
        eligible = _eligible(batches(connection, ticker), decision, calendar_id)
        overlapping = [
            batch
            for batch in eligible
            if batch.request_start <= window[-1].strftime("%Y-%m-%d")
            and batch.request_end >= window[0].strftime("%Y-%m-%d")
        ]
        frame = bars_of(connection, overlapping)

    frame = frame[frame["session"].isin(window)]
    vintage = select_vintage(overlapping, frame)
    covered = pd.DatetimeIndex(vintage.bars["session"])
    missing = window[~window.isin(covered)]
    if len(missing):
        raise InferenceRefused(
            Refusal.SHORT_WINDOW,
            f"{len(missing)} of {lookback} sessions have no batch retrieved by the decision time "
            f"(first {missing[0].date()})",
        )
    bars = vintage.bars.reset_index(drop=True)
    if pd.isna(bars["close"].iloc[-1]):
        raise InferenceRefused(
            Refusal.AS_OF_CLOSE_MISSING, f"the {as_of.date()} session was served without a close"
        )
    gaps = tuple(day.strftime("%Y-%m-%d") for day in bars.loc[bars["close"].isna(), "session"])
    return InferenceWindow(
        ticker=ticker,
        as_of_session=as_of,
        retrieved_by=decision,
        lookback=lookback,
        sessions=window,
        bars=bars,
        gaps=gaps,
        revised_sessions=vintage.revised,
        batches=vintage.used,
        data_class=data_class_of(vintage.used),
        calendar_id=calendar_id,
    )
