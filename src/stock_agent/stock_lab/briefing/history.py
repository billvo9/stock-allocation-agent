"""
The Stock Lab's forecast history as known at a time (read-only; pandas and the
Stock Lab only): each recorded evening's next-session outlook per stock, its
stored result, and the evenings that were missed. The Lab app's History reads
it (owner decision of 2026-10-23).

As known when it is prepared (the briefing's own rule, `report._known`)
    A forecast counts only once it was created and recorded (its ledger entry)
    by then; a recalculation replaces the forecast it supersedes only once it
    existed by then, and the history shows the live forecast; a score counts
    only once it was written by then. A failed ledger check, or a record that
    fails its check when read, withholds the whole history: nothing is shown
    that can't be vouched for.

Missed evenings
    The evening job's rule (`stock_lab.evenings.evenings_from`), applied to the
    evenings known by then: from the first evening with a live forecast of this
    data class, every session whose next open has passed is due, and a due
    evening with no live forecast of any stock is missed. A missed evening stays
    missed on time.

Reconstructions (owner decision of 2026-10-30)
    A stock's forecast reconstructed after the fact, as known by then, is its
    entry on an evening with no live forecast of it (Entry.reconstructed, with
    its created_at: when it was made): on a missed evening, and beside a partly
    recorded evening's live ones. `reconstructed` lists the evenings that hold
    one. They never make an evening recorded and never enter the live track;
    the view labels them.

Nothing is recomputed: probabilities, outcomes and moves are the stored ones,
and each outlook's day is the target session its record names. The exchange
calendar is passed in (`sessions(start, end)`); it names the day each evening's
outlook was for, which the view compares with the record's.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from stock_agent.stock_lab.briefing.report import ReadOnlyRecords, _known
from stock_agent.stock_lab.briefing.timing import NEW_YORK
from stock_agent.stock_lab.contracts import CODE_CATEGORIES, DataClass
from stock_agent.stock_lab.evenings import Sessions, evenings_from
from stock_agent.stock_lab.forecast import ForecastRefused, is_reconstructed
from stock_agent.stock_lab.store import StoreRefused, stamp

SCORED, UNSCORABLE, PENDING = "scored", "unscorable", "pending"


@dataclass(frozen=True)
class Outcome:
    status: str  # scored, unscorable or pending (no score written by then)
    category: str | None  # increase, stable or decrease, as stored
    move: float | None  # the stored realized return of that session
    reason: str | None  # why a target could not be scored


@dataclass(frozen=True)
class Entry:
    symbol: str
    recorded: bool  # False: this evening recorded no forecast for this stock
    created_at: str | None
    session: str | None  # the target session the record names (its first)
    increase: float | None  # the next session's probabilities, exactly as stored
    stable: float | None
    decrease: float | None
    band: float | None
    refusal: str | None  # the model's own reason for giving no probability
    outcome: Outcome | None
    reconstructed: bool = False  # made after the fact for a missed evening (kept apart)


@dataclass(frozen=True)
class Evening:
    as_of: str  # the close the outlook was made from
    session: str | None  # the calendar's next session: the day the outlook is for
    missed: bool
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class History:
    generated_at: str
    data_class: str
    problem: str | None  # set: nothing is shown
    since: str | None
    due: int
    recorded: int
    missed: tuple[str, ...]
    evenings: tuple[Evening, ...]  # every recorded or missed evening, newest first
    listed: tuple[str, ...]  # the stocks each evening lists, in order
    reconstructed: tuple[str, ...] = ()  # evenings with a reconstruction known by then


def forecast_history(
    records: ReadOnlyRecords,
    *,
    now: datetime,
    sessions: Sessions,
    data_class: DataClass | str,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
) -> History:
    """The history as known at `now` (timezone-aware); see the module docstring."""

    kind = DataClass(data_class).value
    moment = pd.Timestamp(now).tz_convert("UTC")
    cutoff = stamp(moment)

    def empty(problem: str | None, listed: Sequence[str] = ()) -> History:
        return History(cutoff, kind, problem, None, 0, 0, (), (), tuple(listed))

    try:
        _, problem = records.verify_ledger()
        if problem is not None:
            return empty("ledger_unverified")
        chosen = records.watched() if tickers is None else tickers
        listed = [symbol for symbol in chosen if symbol not in set(excluded)]
        symbols = {*records.forecast_symbols(DataClass(kind)), *listed}
    except (ForecastRefused, StoreRefused, ValueError):
        return empty("record_unreadable")
    known, rebuilt = {}, {}
    for symbol in sorted(symbols):
        found = _known(records, symbol, cutoff, kind)
        again = _known(records, symbol, cutoff, kind, reconstructed=True)
        if found.problem is not None or again.problem is not None:
            return empty("record_check_failed")
        known[symbol], rebuilt[symbol] = found, again
    made = {record["as_of_session"] for found in known.values() for record in found.forecasts}
    today = pd.Timestamp(moment.tz_convert(NEW_YORK).date(), tz="UTC")
    rule = evenings_from(made, now=moment, expected=today, sessions=sessions)
    if rule is None:
        return empty(None, listed)
    missed = set(rule["missed"])
    days = sorted(made | missed, reverse=True)
    evenings = tuple(_evening(day, day in missed, listed, known, rebuilt, sessions) for day in days)
    held = {f["as_of_session"] for found in rebuilt.values() for f in found.live}
    noted = tuple(day for day in sorted(days) if day in held)
    return History(
        generated_at=cutoff,
        data_class=kind,
        problem=None,
        since=rule["since"],
        due=rule["due"],
        recorded=rule["recorded"],
        missed=tuple(rule["missed"]),
        evenings=evenings,
        listed=tuple(listed),
        reconstructed=noted,
    )


def _next_session(day: str, sessions: Sessions) -> str | None:
    start = pd.Timestamp(day, tz="UTC")
    try:
        found = sessions(start + pd.Timedelta(days=1), start + pd.Timedelta(days=10))
    except ValueError:  # beyond the calendar's coverage
        return None
    return pd.Timestamp(found[0]).strftime("%Y-%m-%d") if len(found) else None


def _evening(day: str, missed: bool, listed, known, rebuilt, sessions: Sessions) -> Evening:
    """One evening: each stock's live forecast, else its reconstruction (module docstring)."""

    session = _next_session(day, sessions)
    entries = []
    for symbol in listed:
        record, scores = None, ()
        for track in (known, rebuilt):  # the live record first; a reconstruction only without
            found = track.get(symbol)
            live = [] if found is None else [f for f in found.live if f["as_of_session"] == day]
            record = max(live, key=lambda f: f["created_at"], default=None)
            if record is not None:
                scores = found.scores
                break
        entries.append(_entry(symbol, record, scores))
    if missed and not any(entry.recorded for entry in entries):
        return Evening(day, session, True, ())
    return Evening(day, session, missed, tuple(entries))


def _entry(symbol: str, record, scores) -> Entry:
    if record is None:
        return Entry(symbol, False, None, None, None, None, None, None, None, None)
    created, session = record["created_at"], record["target_sessions"][0]
    refusal = record["refusal"]
    if refusal is not None:
        reason = refusal.get("detail") or refusal.get("reason")
        return Entry(symbol, True, created, session, None, None, None, None, reason, None,
                     is_reconstructed(record))  # fmt: skip
    first = record["daily"][0]
    found = [
        s for s in scores if s["forecast_id"] == record["forecast_id"] and str(s["target"]) == "1"
    ]
    score = max(found, key=lambda s: s["scored_at"], default=None)
    if score is None:
        outcome = Outcome(PENDING, None, None, None)
    elif score["status"] == SCORED:
        category = CODE_CATEGORIES[int(score["outcome"])].value
        move = score["realized_return"]
        outcome = Outcome(SCORED, category, None if move is None else float(move), None)
    else:
        outcome = Outcome(UNSCORABLE, None, None, score.get("reason"))
    return Entry(
        symbol=symbol,
        recorded=True,
        created_at=created,
        session=session,
        increase=first["p_increase"],
        stable=first["p_stable"],
        decrease=first["p_decrease"],
        band=float(record["band"]),
        refusal=None,
        outcome=outcome,
        reconstructed=is_reconstructed(record),
    )


__all__ = ["Entry", "Evening", "History", "Outcome", "forecast_history"]
