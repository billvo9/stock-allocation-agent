"""
The prospective window: which sessions the prospective store may hold.

Owner decisions of 2026-10-14 (`docs/PROJECT_STATE.md`, "prospective
forecasting track"): H_end is 2026-10-14; the sealed holdout runs from
2025-01-01 to H_end and is kept for one historical evaluation; a separate
prospective store may use the last 63 sessions before H_end only as inputs to
frozen models, never to train, tune or evaluate. Lead implementation choice
(more conservative than authorized): the first collection requests only the
63-session lookback before H_end and later sessions.

Window
    `WINDOW_START` is the session `LOOKBACK_SESSIONS` sessions before `H_END`
    on the rules calendar (`calendars/nyse_rules.py`): 2026-07-16. The span
    [WINDOW_START, H_END] holds 64 sessions, H_END and the 63 before it, so the
    63 daily returns that end at H_END use exactly the lookback. Every session
    from WINDOW_START on may be requested once it is complete. Nothing earlier
    may be: [2025-01-01, WINDOW_START) is the rest of the sealed period
    (`sealed_holdout`), and anything before 2025-01-01 belongs to the research
    store (`before_window`).

Complete sessions (a request rule, not an availability timestamp)
    Under the OD9 descriptive convention, a session's close is assumed
    available at 18:00 America/New_York on its date. `last_complete_session(now)`
    is the latest session whose 18:00 America/New_York is at or before `now`,
    and a request may end no later than it (`incomplete_session`). In the
    owner's words, this is "not proof of final vendor publication time and must
    not be used as evidence of historical trading executability or forecasting
    validity". No availability timestamp is stored for any row.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.schema import (
    FundDataRefused,
    RefusalReason,
    check_id,
    refuse,
    utc_date,
    utc_timestamp,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

H_END = pd.Timestamp("2026-10-14", tz="UTC").as_unit("us")
LOOKBACK_SESSIONS = 63
WINDOW_START = nyse_rules.session_offset(H_END, -LOOKBACK_SESSIONS)

CLOSE_TIMEZONE = "America/New_York"
CLOSE_HOUR = 18
REQUEST_CUTOFF_RULE = (
    "a session may be requested only after 18:00 America/New_York on its date (the OD9 "
    "descriptive close convention, assumed; not proof of vendor publication time)"
)


class ProspectiveRefusal(StrEnum):
    SEALED_HOLDOUT = "sealed_holdout"
    BEFORE_WINDOW = "before_window"
    INCOMPLETE_SESSION = "incomplete_session"
    NOT_PROSPECTIVE_STORE = "not_prospective_store"
    FOREIGN_ENTRY = "foreign_entry"


class ProspectiveRefused(FundDataRefused):
    """
    A prospective-store refusal. `reason` is a `ProspectiveRefusal`; F1's
    `RefusalReason` is unchanged. Callers catch `FundDataRefused` for both.
    """

    def __init__(self, reason: ProspectiveRefusal, detail: str) -> None:
        super().__init__(reason, detail)


def close_convention_at(session: pd.Timestamp) -> pd.Timestamp:
    """18:00 America/New_York on the session's date, in UTC (the assumed convention)."""

    local = pd.Timestamp(f"{session:%Y-%m-%d} {CLOSE_HOUR:02d}:00", tz=CLOSE_TIMEZONE)
    return local.tz_convert("UTC").as_unit("us")


def last_complete_session(now: pd.Timestamp) -> pd.Timestamp:
    """
    The latest session whose 18:00 America/New_York is at or before `now` (a
    UTC timestamp). Refuses a `now` past the rules calendar's coverage: the
    calendar must be extended first.
    """

    moment = utc_timestamp("now", now)
    local = moment.tz_convert(CLOSE_TIMEZONE)
    day = pd.Timestamp(local.date().isoformat(), tz="UTC").as_unit("us")
    if local.hour < CLOSE_HOUR:
        day -= pd.Timedelta(days=1)
    coverage = nyse_rules.COVERAGE
    if day > coverage.last:
        raise refuse(
            RefusalReason.OUTSIDE_COVERAGE,
            f"{moment} is past the calendar {nyse_rules.CALENDAR_ID}; extend it first",
        )
    if day < WINDOW_START:
        raise ProspectiveRefused(
            ProspectiveRefusal.INCOMPLETE_SESSION,
            f"no session of the prospective window is complete at {moment}",
        )
    return nyse_rules.sessions(WINDOW_START, day)[-1]


def check_window(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    [start, end] (session dates, both inclusive) must lie inside the window:
    start on or after WINDOW_START, start <= end, end inside the calendar.
    """

    for label, value in (("start", start), ("end", end)):
        if value < MODEL_HOLDOUT_START:
            raise ProspectiveRefused(
                ProspectiveRefusal.BEFORE_WINDOW,
                f"{label} {value.date()} is before the holdout start; that period belongs to "
                "the research store",
            )
        if value < WINDOW_START:
            raise ProspectiveRefused(
                ProspectiveRefusal.SEALED_HOLDOUT,
                f"{label} {value.date()} is in the sealed holdout before the prospective "
                f"window ({WINDOW_START.date()})",
            )
    if start > end:
        raise refuse(RefusalReason.INVALID_REQUEST, f"start {start.date()} is after end")
    if end > nyse_rules.COVERAGE.last:
        raise refuse(
            RefusalReason.OUTSIDE_COVERAGE,
            f"end {end.date()} is past the calendar {nyse_rules.CALENDAR_ID}",
        )


def check_request(
    fund_ids: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, now: pd.Timestamp
) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    """
    A prospective request for [start, end], both inclusive session dates
    (00:00 UTC), checked before any provider is contacted: valid unique fund
    ids, the window, and an end no later than `last_complete_session(now)`.
    """

    if isinstance(fund_ids, str) or not isinstance(fund_ids, Sequence) or not fund_ids:
        raise refuse(RefusalReason.INVALID_REQUEST, "fund_ids must be a non-empty sequence")
    try:
        ids = tuple(check_id("fund_id", fund_id) for fund_id in fund_ids)
    except FundDataRefused as error:
        raise refuse(RefusalReason.INVALID_REQUEST, error.detail) from error
    if len(set(ids)) != len(ids):
        raise refuse(RefusalReason.INVALID_REQUEST, f"fund_ids repeat: {list(ids)}")
    first, last = utc_date("start", start), utc_date("end", end)
    check_window(first, last)
    complete = last_complete_session(now)
    if last > complete:
        raise ProspectiveRefused(
            ProspectiveRefusal.INCOMPLETE_SESSION,
            f"end {last.date()} is after the last complete session {complete.date()} at {now} "
            f"({REQUEST_CUTOFF_RULE})",
        )
    return ids, first, last


__all__ = [
    "CLOSE_HOUR",
    "CLOSE_TIMEZONE",
    "H_END",
    "LOOKBACK_SESSIONS",
    "REQUEST_CUTOFF_RULE",
    "WINDOW_START",
    "ProspectiveRefusal",
    "ProspectiveRefused",
    "check_request",
    "check_window",
    "close_convention_at",
    "last_complete_session",
]
