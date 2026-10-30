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
after that the live window reader refuses it. Every session from the first
real live forecast's as-of session whose next open has passed is due; a due
session with no real live forecast (of any stock) is missed. `evening_record`
lists them, and how many of the last RECENT_EVENINGS due evenings were missed,
so Model Health can flag a recent miss without staying flagged for every later
day.

Reconstructions (owner decision of 2026-10-30). Every (stock, evening) without
a live forecast may be reconstructed after the fact, per stock: a missed
evening's stocks, and a partly recorded evening's missing ones
(stock_lab.evening.reconstruct_missed; `missing_forecasts` lists them). A
reconstruction never makes an evening recorded: a missed evening stays missed
on time, and `evening_record` notes every due evening that holds one
(`reconstructed`, present only when some do).
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import date, datetime, time

import pandas as pd

from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.forecast import NEW_YORK, ForecastRefused, session_open
from stock_agent.stock_lab.store import StockLabStore, StoreRefused

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]
RECENT_EVENINGS = 10  # due evenings (two trading weeks) within which a miss stays flagged
EVENING_OPENS = time(20, 0)  # New York: a session's close counts as final (the inference store's)


def recorded_evenings(store: StockLabStore) -> set[str]:
    """
    The as-of sessions (YYYY-MM-DD) with a real live forecast of any stock: the evenings
    recorded (a reconstruction is not one). ForecastRefused or StoreRefused when a stored
    forecast no longer verifies. The morning check (stock_lab.morning) reads the same set.
    """

    return {
        record["as_of_session"]
        for symbol in store.forecast_symbols(DataClass.REAL)
        for record in store.forecasts(symbol)
        if record["data_class"] == DataClass.REAL.value
    }


def reconstructed_evenings(store: StockLabStore) -> set[str]:
    """The as-of sessions with a reconstructed forecast of any stock (module docstring)."""

    return {
        record["as_of_session"]
        for symbol in store.forecast_symbols(DataClass.REAL)
        for record in store.forecasts(symbol, reconstructed=True)
    }


def evening_opens(day: str) -> pd.Timestamp:
    """When `day`'s evening began: its close counts as final, 20:00 New York (in UTC)."""

    opened = datetime.combine(date.fromisoformat(day), EVENING_OPENS, tzinfo=NEW_YORK)
    return pd.Timestamp(opened).tz_convert("UTC")


def missing_forecasts(
    store: StockLabStore,
    *,
    stocks: Collection[str],
    now: pd.Timestamp,
    expected: pd.Timestamp,
    sessions: Sessions,
) -> list[tuple[str, str]]:
    """
    Every (session, stock) a reconstruction would fill, in order: each due evening since the
    first live forecast, for each of `stocks` (the model's) that was watched when that evening
    began (evening_opens: the append-only watchlist, never today's) and has no real forecast
    of it, live or reconstructed. Per stock (owner, 2026-10-30): a partly recorded evening's
    missing stocks too. Empty before the first live forecast. ForecastRefused or StoreRefused
    as above.
    """

    made = recorded_evenings(store)
    if not made:
        return []
    gaps = []
    for day in _due(min(made), now=now, expected=expected, sessions=sessions):
        for ticker in (t for t in store.watched_at(evening_opens(day)) if t in set(stocks)):
            held = {
                record["as_of_session"]
                for record in (
                    *store.forecasts(ticker),
                    *store.forecasts(ticker, reconstructed=True),
                )
                if record["data_class"] == DataClass.REAL.value
            }
            if day not in held:
                gaps.append((day, ticker))
    return sorted(gaps)


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
        rebuilt = reconstructed_evenings(store)
    except (ForecastRefused, StoreRefused):
        return None
    found = evenings_from(made, now=now, expected=expected, sessions=sessions, recent=recent)
    if found is None:
        return None
    due = _due(found["since"], now=now, expected=expected, sessions=sessions)
    noted = [day for day in due if day in rebuilt]
    if noted:  # a missed one stays missed on time; each was reconstructed after the fact
        found["reconstructed"] = noted
    return found


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
    due = _due(since, now=now, expected=expected, sessions=sessions)
    missed = [day for day in due if day not in made]
    return {
        "since": since,
        "due": len(due),
        "recorded": len(due) - len(missed),
        "missed": missed,
        "recent": recent,
        "recent_missed": len([day for day in due[-recent:] if day not in made]),
    }


def _due(since: str, *, now: pd.Timestamp, expected: pd.Timestamp, sessions: Sessions) -> list[str]:
    """The sessions from `since` through `expected` whose evening closed by `now` (next open)."""

    days = [day.strftime("%Y-%m-%d") for day in sessions(pd.Timestamp(since, tz="UTC"), expected)]
    try:  # the session after `expected`, whose open closes `expected`'s evening
        after = sessions(expected + pd.Timedelta(days=1), expected + pd.Timedelta(days=10))
    except ValueError:  # beyond the calendar's coverage: `expected` is not judged yet
        after = ()
    opens = [*days[1:], *(day.strftime("%Y-%m-%d") for day in after[:1])]
    return [
        day for day, following in zip(days, opens, strict=False) if session_open(following) <= now
    ]
