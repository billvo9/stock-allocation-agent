"""
The Stock Lab's evenings: which as-of sessions hold a real forecast, and which
due evenings were missed. Standard library, pandas and the Stock Lab only; the
exchange calendar is passed in (`sessions(start, end)`), so this module never
imports a calendar of its own.

Moved unchanged from `stock_lab.job` (owner decision of 2026-10-23), which
re-exports these names, so the evening job, the morning check, Model Health and
the Lab app's History read one rule. `evenings_from` is the rule itself, on a
set of recorded evenings; the Lab app's History passes it the evenings known at
its own preparation time.

Missed evenings. A session's evening forecast can be recorded from its close's
availability (20:00 New York) until the next session opens (09:30 New York);
after that the window reader refuses it, so nothing ever fills it in. Every
session from the first real forecast's as-of session whose next open has
passed is due; a due session with no real forecast (of any stock) is missed.
`evening_record` lists them, and how many of the last RECENT_EVENINGS due
evenings were missed, so Model Health can flag a recent miss without staying
flagged for every later day.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import pandas as pd

from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.forecast import ForecastRefused, session_open
from stock_agent.stock_lab.store import StockLabStore, StoreRefused

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]
RECENT_EVENINGS = 10  # due evenings (two trading weeks) within which a miss stays flagged


def recorded_evenings(store: StockLabStore) -> set[str]:
    """
    The as-of sessions (YYYY-MM-DD) with a real forecast of any stock: the evenings recorded.
    ForecastRefused or StoreRefused when a stored forecast no longer verifies. The morning
    check (stock_lab.morning) reads the same set.
    """

    return {
        record["as_of_session"]
        for symbol in store.forecast_symbols(DataClass.REAL)
        for record in store.forecasts(symbol)
        if record["data_class"] == DataClass.REAL.value
    }


def evening_record(
    store: StockLabStore,
    *,
    now: pd.Timestamp,
    expected: pd.Timestamp,
    sessions: Sessions,
    recent: int = RECENT_EVENINGS,
) -> dict | None:
    """
    The prospective record's evenings (module docstring, "Missed evenings"): {since, due,
    recorded, missed: [session], recent, recent_missed}. None before the first real forecast,
    or when a stored forecast no longer verifies (the ledger check reports that).
    """

    try:
        made = recorded_evenings(store)
    except (ForecastRefused, StoreRefused):
        return None
    return evenings_from(made, now=now, expected=expected, sessions=sessions, recent=recent)


def evenings_from(
    made: Collection[str],
    *,
    now: pd.Timestamp,
    expected: pd.Timestamp,
    sessions: Sessions,
    recent: int = RECENT_EVENINGS,
) -> dict | None:
    """`evening_record` for the recorded evenings `made` (YYYY-MM-DD); None when there are none."""

    if not made:
        return None
    since = min(made)
    days = [day.strftime("%Y-%m-%d") for day in sessions(pd.Timestamp(since, tz="UTC"), expected)]
    try:  # the session after `expected`, whose open closes `expected`'s evening
        after = sessions(expected + pd.Timedelta(days=1), expected + pd.Timedelta(days=10))
    except ValueError:  # beyond the calendar's coverage: `expected` is not judged yet
        after = ()
    opens = [*days[1:], *(day.strftime("%Y-%m-%d") for day in after[:1])]
    due = [
        day for day, following in zip(days, opens, strict=False) if session_open(following) <= now
    ]
    missed = [day for day in due if day not in made]
    return {
        "since": since,
        "due": len(due),
        "recorded": len(due) - len(missed),
        "missed": missed,
        "recent": recent,
        "recent_missed": len([day for day in due[-recent:] if day not in made]),
    }
