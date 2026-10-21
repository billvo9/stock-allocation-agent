"""
The morning check: is the prior NYSE session's evening recorded while it still
can be? (The Stock Lab only; the exchange calendar is passed in, so this module
never imports one.)

A session's evening forecast can be recorded from its close's availability
(20:00 New York) until the next session opens (09:30 New York); after that it
is a permanent gap (stock_lab.job, "Missed evenings"). The evening agent's
run.sh notifies at once when a step fails, but a run that never starts (the Mac
shut down or logged out at 20:30, the agent not loaded, run.sh missing) cannot.
The morning check runs separately at a set time before the open (08:00 New York
by default, scripts/stock_lab_schedule.py --morning-check) and catches each of
those while there is still time to record the evening.

What it reads
    - Which evening is still open, from the briefing's timing
      (stock_lab.briefing.timing): on a session morning before 09:30, the
      prior session's.
    - Which evenings are recorded: stock_lab.job.recorded_evenings, the same
      set evening_record reads for Model Health's `evening_missed`, through
      the store's read methods only. Nothing is written.

Outcomes (status: the notice shown, stock_lab.notify)
    recorded      the prior session has a real forecast: none
    not_recorded  before the 09:30 open of a session day, the prior session
                  has no real forecast yet: evening_not_recorded (it can still
                  be recorded). An empty or missing store counts as not
                  recorded: a check that cannot see the record is no quiet
                  morning.
    missed        after the open of a session day (a late check, from a Mac
                  that woke late), the evening before that day has no real
                  forecast: evening_missed, a permanent gap. From 20:00 the
                  check is still of that evening, never of tonight's, whose run
                  is still to come.
    not_due       a weekend or market holiday: none, and nothing is read, so
                  these days raise no false alarm. Nothing is lost: the next
                  session morning checks the same evening while it can still
                  be recorded.
    failed        the calendar or the record could not be read: check_failed
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime

from stock_agent.stock_lab.briefing.timing import (
    ADVANCE,
    AFTER_OPEN,
    NEW_YORK,
    OPEN_TIME,
    PREMARKET,
    Sessions,
    briefing_timing,
)
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.job import recorded_evenings
from stock_agent.stock_lab.store import StoreRefused

RECORDED = "recorded"
NOT_RECORDED = "not_recorded"
MISSED = "missed"
NOT_DUE = "not_due"
FAILED = "failed"
NOTICE_FOR = {
    RECORDED: None,
    NOT_RECORDED: "evening_not_recorded",
    MISSED: "evening_missed",
    NOT_DUE: None,
    FAILED: "check_failed",
}


@dataclass(frozen=True)
class MorningCheck:
    status: str
    reason: str | None  # machine-readable: the timing state or the failure
    session: date | None  # the prior session whose evening was checked
    detail: str  # one plain sentence for the log: dates only, never a value

    @property
    def notice(self) -> str | None:
        """The stock_lab.notify notice to show, or None."""

        return NOTICE_FOR[self.status]


def morning_check(store, *, now: datetime, calendar: Sessions) -> MorningCheck:
    """
    The outcome at `now` (timezone-aware) for `store` (its forecast_symbols and forecasts
    reads): see the module docstring.
    """

    try:
        plan = briefing_timing(now, calendar)
        if plan.state == ADVANCE:
            # From 20:00 on a session day tonight's run is still to come, so the evening
            # checked is the one before today, as at today's open (then after_open).
            today = now.astimezone(NEW_YORK).date()
            plan = briefing_timing(datetime.combine(today, OPEN_TIME, NEW_YORK), calendar)
    except ValueError as error:  # TimingRefused, or the calendar's CalendarRefused
        return MorningCheck(
            FAILED, "calendar_refused", None, f"the check could not run: the calendar ({error})"
        )
    if plan.state not in (PREMARKET, AFTER_OPEN):
        return MorningCheck(
            NOT_DUE, plan.state, None, "the market is closed today; nothing to check"
        )
    prior = plan.prior_session
    try:
        recorded = prior.isoformat() in recorded_evenings(store)
    except (ForecastRefused, StoreRefused) as error:
        return MorningCheck(
            FAILED, "record_unreadable", prior, f"the check could not run: the record ({error})"
        )
    except (sqlite3.Error, OSError) as error:
        return MorningCheck(
            FAILED,
            "record_unreadable",
            prior,
            f"the check could not run: the record ({type(error).__name__})",
        )
    if recorded:
        return MorningCheck(RECORDED, None, prior, f"the evening of {prior} is recorded")
    if plan.state == PREMARKET:
        return MorningCheck(
            NOT_RECORDED,
            None,
            prior,
            f"the evening of {prior} is not recorded yet; it can still be recorded before the "
            f"09:30 open of {plan.market_date}",
        )
    return MorningCheck(
        MISSED,
        None,
        prior,
        f"the evening of {prior} was not recorded before the 09:30 open of {plan.market_date}: "
        "a permanent gap",
    )
