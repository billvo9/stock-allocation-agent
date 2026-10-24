"""
The Lab app's History view model (docs/product/lab_app_design.md, "History in
detail"): each evening's next-session outlook per stock and what happened, as a
timeline, newest first; JSON-ready, with display-ready words and the raw recorded
values.

Where each part comes from
    The Stock Lab's forecast history as known at the request time
    (`stock_lab.briefing.history`, owner decision of 2026-10-23): only records,
    recalculations and scores that existed by then; a missed evening by the
    evening job's own rule, shown as missed and never filled in. The chances are
    the recorded ones (rounded for display only), and the result is the stored
    score: its category and the session's move as the evening job wrote it.
    Nothing is computed from prices here (decision A6).

Colours
    The result's move is a price direction, so it is marked up, flat or down; an
    outlook (what the model said) is never coloured.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from datetime import date, datetime, tzinfo

import pandas as pd

from stock_agent.app.web import home, words
from stock_agent.stock_lab.briefing.history import Entry, Evening, History, forecast_history
from stock_agent.stock_lab.briefing.reliability import MIN_SCORED, MIN_SESSIONS
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.briefing.timing import CLOSE_AVAILABLE
from stock_agent.stock_lab.contracts import DataClass

SCHEMA = "lab_app_history"
VERSION = 1
SHOWN = 40  # evenings on the page, newest first; the counts cover every one
DIRECTIONS = {"increase": "up", "stable": "flat", "decrease": "down"}


def build_history(
    records: ReadOnlyRecords,
    *,
    now: datetime,
    calendar: Callable,
    data_class: DataClass | str,
    local_zone: tzinfo | None,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
) -> dict:
    """History at `now` (timezone-aware), read from the store as it was then."""

    found = forecast_history(
        records,
        now=now,
        sessions=calendar,
        data_class=data_class,
        tickers=tickers,
        excluded=excluded,
    )
    return history_view(found, now=now, local_zone=local_zone)


def history_view(found: History, *, now: datetime, local_zone: tzinfo | None) -> dict:
    """The History view from the forecast history read at `now`."""

    shown = found.evenings[:SHOWN]
    problem = None if found.problem is None else _problem(found.problem)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "data_class": found.data_class,
        "generated_at": pd.Timestamp(now).tz_convert("UTC").isoformat(),
        "title": words.HISTORY_TITLE,
        "lede": words.HISTORY_LEDE,
        "checked": words.checked(now, local_zone),
        "problem": problem,
        "counts": None if problem or not found.evenings else _counts(found),
        "empty": words.HISTORY_EMPTY if problem is None and not found.evenings else None,
        "filters": [] if problem or not found.evenings else ["All", *found.listed],
        "note": {
            "words": words.HISTORY_NOTE,
            "button": words.JUDGE,
            "sheet": words.judge_sheet(MIN_SESSIONS, MIN_SCORED, words.HISTORY_JUDGE_DETAIL),
        },
        "evenings": [_evening(e, now, local_zone) for e in shown],
        "more": words.history_more(SHOWN) if len(found.evenings) > SHOWN else None,
        "demo": words.DEMO if found.data_class == DataClass.SYNTHETIC.value else None,
    }


def _problem(reason: str) -> dict:
    text = words.HISTORY_PROBLEMS.get(reason, words.HISTORY_PROBLEMS["record_unreadable"])
    return {
        "title": words.HISTORY_PROBLEM_TITLE,
        "words": text,
        "technical": words.TECHNICAL.format(detail=reason),
    }


def _counts(found: History) -> str:
    recorded = [e for e in found.evenings if not e.missed]
    scored = [e for e in recorded if any(_scored(x) for x in e.entries)]
    return words.history_counts(
        found.since, recorded=len(recorded), scored=len(scored), missed=len(found.missed)
    )


def _scored(item: Entry) -> bool:
    return item.outcome is not None and item.outcome.status == "scored"


def _evening(evening: Evening, now: datetime, zone: tzinfo | None) -> dict:
    day = evening.session
    title = words.short_day(day) if day else words.short_day(evening.as_of)
    if evening.missed:
        return {
            "as_of": evening.as_of,
            "session": day,
            "state": "missed",
            "title": title,
            "sub": words.missed_evening(evening.as_of),
            "missed_words": words.MISSED_STAYS,
            "items": [],
        }
    items = [_item(x, day, now) for x in evening.entries]
    made = max((x.created_at for x in evening.entries if x.created_at), default=None)
    waiting = any(item["state"] == "pending" for item in items)
    return {
        "as_of": evening.as_of,
        "session": day,
        "state": "pending" if waiting else "recorded",
        "title": title,
        "sub": words.recorded_evening(
            None if made is None else words.when(made, now=now, local_zone=zone),
            waiting=waiting,
        ),
        "missed_words": None,
        "items": items,
    }


def _item(x: Entry, day: str | None, now: datetime) -> dict:
    if not x.recorded:
        return _quiet(x.symbol, "absent", words.NO_OUTLOOK_RECORDED)
    if x.refusal is not None:
        return _quiet(x.symbol, "refused", words.REFUSED_CHIP, x.refusal)
    if day is None or x.session != day:
        return _quiet(x.symbol, "withheld", words.HISTORY_WITHHELD, f"recorded for {x.session}")
    up, down = words.percent(x.increase), words.percent(x.decrease)
    said = words.history_said(up, down)
    recorded = {"increase": x.increase, "stable": x.stable, "decrease": x.decrease}
    outcome = x.outcome
    if outcome.status == "scored":
        chance = {"increase": x.increase, "stable": x.stable, "decrease": x.decrease}
        word = words.CATEGORY_SHORT[outcome.category]
        moved = home.finite(outcome.move)
        move = words.signed_percent(moved)
        gave = words.history_gave(word, words.percent(chance[outcome.category]))
        result = {
            "words": f"{word} {move}",
            "direction": DIRECTIONS[outcome.category],
            "gave": gave,
        }
        label = f"{x.symbol}: {said}. {words.long_weekday(day)}: {word}, {move}. {gave}."
        state, result_words = "scored", None
        recorded = {**recorded, "category": outcome.category, "move": moved}
    elif outcome.status == "unscorable":
        result, result_words, state = None, words.NO_RESULT, "unscorable"
        label = f"{x.symbol}: {said}. {words.NO_RESULT}."
    else:
        result, state = None, "pending"
        result_words = _waiting(day, now)
        label = f"{x.symbol}: {said}. {result_words}."
    reason = outcome.reason if state == "unscorable" and outcome.reason else None
    return {
        "ticker": x.symbol,
        "state": state,
        "said": {"lead": words.SAID, "up": up, "down": down},
        "said_words": None,
        "result": result,
        "result_words": result_words,
        "recorded": recorded,
        "label": label,
        "technical": None if reason is None else words.TECHNICAL.format(detail=reason),
    }


def _quiet(symbol: str, state: str, text: str, detail: str | None = None) -> dict:
    """An item with no chances to show: no outlook, the model's refusal, or withheld."""

    return {
        "ticker": symbol,
        "state": state,
        "said": None,
        "said_words": text,
        "result": None,
        "result_words": None,
        "recorded": None,
        "label": f"{symbol}: {text}.",
        "technical": None if detail is None else words.TECHNICAL.format(detail=detail),
    }


def _waiting(day: str, now: datetime) -> str:
    """'After Thu's close' until the evening update after that close; 'No result yet' after."""

    close = pd.Timestamp.combine(date.fromisoformat(day), CLOSE_AVAILABLE).tz_localize(
        words.NEW_YORK
    )
    return words.after_close(day) if pd.Timestamp(now) < close else words.NO_RESULT_YET


__all__ = ["SCHEMA", "VERSION", "build_history", "history_view"]
