"""
Which session a premarket briefing is for (standard library only; the
exchange calendar is passed in, so this module never imports one).

The target is 08:30 America/New_York on an NYSE session morning, an hour
before the 09:30 open. The time zone is the IANA zone, never a fixed offset,
so 08:30 is 13:30 UTC in winter and 12:30 UTC in summer.

States, from the moment a briefing is prepared (New York time):

    premarket    a session day before 09:30: the briefing is for that
                 session, and it is the only state in which a scheduled
                 delivery may go out
    after_open   a session day from 09:30 until the close is available
                 (20:00, the daily job's convention): the briefing is for
                 that session but late; it is a preview, never delivered
    advance      a session day from 20:00: the briefing is for the next
                 session, prepared the evening before (a preview)
    closed_day   a weekend or market holiday: the briefing is for the next
                 session; scheduled runs skip these days unless configured

The briefing's data is the close of the prior session (the last session
before the one it is for). The calendar days in between that were not
sessions are listed, so a Monday briefing can say plainly that Friday's
end-of-day data holds nothing from the weekend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
TARGET_TIME = time(8, 30)
OPEN_TIME = time(9, 30)
CLOSE_AVAILABLE = time(20, 0)  # the daily job's availability convention (stock_lab_daily)
LOOKAROUND = 10  # calendar days searched on each side for sessions

PREMARKET = "premarket"
AFTER_OPEN = "after_open"
ADVANCE = "advance"
CLOSED_DAY = "closed_day"
STATE_WORDS = {
    PREMARKET: "Before the open",
    AFTER_OPEN: "After the open: a late preview, not a premarket briefing",
    ADVANCE: "The evening before: a preview, not a premarket briefing",
    CLOSED_DAY: "On a day the market is closed: a preview, not a premarket briefing",
}

# The exchange calendar: every session date in [start, end] (dates or timestamps).
Sessions = Callable[[date, date], Iterable]


class TimingRefused(ValueError):
    """A time or calendar the briefing cannot plan from."""


@dataclass(frozen=True)
class Timing:
    generated_at: datetime  # timezone-aware
    market_date: date  # the session the briefing is for
    prior_session: date  # the session whose close is the briefing's data
    scheduled_at: datetime  # 08:30 New York on the market date
    state: str
    closed_days: tuple[date, ...]  # non-session days between the prior close and the market date

    @property
    def deliverable(self) -> bool:
        """Only a premarket briefing may be delivered on schedule."""

        return self.state == PREMARKET


def scheduled_at(market_date: date) -> datetime:
    """08:30 New York time on `market_date`, timezone-aware."""

    return datetime.combine(market_date, TARGET_TIME, tzinfo=NEW_YORK)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    converted = getattr(value, "date", None)  # pandas Timestamp
    if callable(converted):
        return converted()
    raise TimingRefused(f"the calendar returned {value!r}, not a date")


def _sessions(calendar: Sessions, start: date, end: date) -> list[date]:
    return sorted({_as_date(day) for day in calendar(start, end)})


def briefing_timing(now: datetime, calendar: Sessions) -> Timing:
    """The briefing's session, its prior session and its state at `now` (module docstring)."""

    if now.tzinfo is None:
        raise TimingRefused("the preparation time must be timezone-aware")
    local = now.astimezone(NEW_YORK)
    today = local.date()
    days = _sessions(
        calendar, today - timedelta(days=LOOKAROUND), today + timedelta(days=LOOKAROUND)
    )
    coming = [day for day in days if day > today]
    if today in days and local.time() < OPEN_TIME:
        market, state = today, PREMARKET
    elif today in days and local.time() < CLOSE_AVAILABLE:
        market, state = today, AFTER_OPEN
    elif coming:
        market, state = coming[0], ADVANCE if today in days else CLOSED_DAY
    else:
        raise TimingRefused(f"the calendar has no session within {LOOKAROUND} days after {today}")
    earlier = [day for day in days if day < market]
    if not earlier:
        raise TimingRefused(f"the calendar has no session before {market}")
    prior = earlier[-1]
    closed = tuple(prior + timedelta(days=k) for k in range(1, (market - prior).days))
    return Timing(
        generated_at=now,
        market_date=market,
        prior_session=prior,
        scheduled_at=scheduled_at(market),
        state=state,
        closed_days=closed,
    )


def long_date(day: date) -> str:
    """'Friday 16 October 2026'."""

    return f"{day:%A} {day.day} {day:%B %Y}"


def _short(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B}"


def gap_note(plan: Timing) -> str | None:
    """
    Plain words for the days between the prior close and the market date when
    the market was closed, or None when there were none.
    """

    if not plan.closed_days:
        return None
    holidays = [day for day in plan.closed_days if day.weekday() < 5]
    weekend = [day for day in plan.closed_days if day.weekday() >= 5]
    kinds = []
    if weekend:
        kinds.append("the weekend")
    if holidays:
        kinds.append("a market holiday" if len(holidays) == 1 else "market holidays")
    days = [_short(day) for day in plan.closed_days]
    listed = days[0] if len(days) == 1 else ", ".join(days[:-1]) + " and " + days[-1]
    return (
        f"The latest data is the close of {long_date(plan.prior_session)}. The market was closed "
        f"on {listed} ({' and '.join(kinds)}), so these numbers contain no information from "
        "those days: nothing that happened since that close is reflected."
    )
