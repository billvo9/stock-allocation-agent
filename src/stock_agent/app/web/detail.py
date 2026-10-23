"""
The Lab app's Stock detail view model (docs/product/lab_app_design.md, "Stock
detail in detail"): one stock's screen as JSON-ready data, with display-ready
words and the raw recorded or served values.

Where each part comes from
    The outlook, the five-session total, each day on its own, how reliable the
    model has been and the context come from the premarket briefing's report
    (`build_briefing`), prepared at the request time for this one stock: it
    reads the Stock Lab's store read-only, counts only the records that
    existed by then and never recomputes a probability.
    - The next session and the five-session total are shown exactly as
      recorded (rounded for display only), each with the band its record
      names. A stock without a fresh outlook gets Home's honest empty state.
    - How reliable: the development record's bucket for the recorded chance
      of Up (the stored evaluation pools every stock it tested; its 90%
      Jeffreys range counts the dates), and the live record's progress
      toward Model Health's minimum (settled sessions and scored targets,
      every stock together).
    - The context chips come only from what the evening job recorded with
      the forecast (its `signals`): the current run, and the 20-session
      change and the volatility regime of its window. A context the briefing
      computed from stored prices is never shown here.
    The chart is the injected recent-prices reader's last CHART_SESSIONS
    sessions, exactly as it gives them (decision A6): each session's
    split-consistent level for the line, and its raw close, its change and
    its notes for the readout. Nothing is computed from the prices: no change
    over a period, no high or low, no average.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import datetime, tzinfo
from urllib.parse import urljoin

import pandas as pd

from stock_agent.app.web import home, words
from stock_agent.stock_lab.briefing.reliability import MIN_CLUSTERS, MIN_SCORED, MIN_SESSIONS, OK
from stock_agent.stock_lab.briefing.report import (
    Briefing,
    Probabilities,
    ReadOnlyRecords,
    StockBriefing,
    build_briefing,
)
from stock_agent.stock_lab.briefing.timing import briefing_timing
from stock_agent.stock_lab.contracts import DataClass, StableBand

SCHEMA = "lab_app_stock"
VERSION = 1
CHART_SESSIONS = 252  # a year of sessions: the most the recent-prices reader serves
RECORDED = "window"  # a briefing context recorded with the forecast, never computed by it
CATEGORIES = ("increase", "stable", "decrease")


def build_detail(
    records: ReadOnlyRecords,
    *,
    symbol: str,
    now: datetime,
    calendar: Callable,
    calendar_id: str,
    data_class: DataClass | str,
    recent: Callable[..., object],
    local_zone: tzinfo | None,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
    names: Mapping[str, str] | None = None,
    research_url: str | None = None,
) -> dict | None:
    """
    `symbol`'s screen at `now` (timezone-aware), or None when it is not on the
    watchlist Home shows (`tickers`, or the store's, minus `excluded`); nothing
    else is read for a stock that is not listed. A store whose watchlist cannot
    be read refuses (ValueError), so the server answers with its plain error
    rather than "not on the watchlist".
    `recent(symbol, count=, retrieved_by=)` is the recent-prices reader and
    `calendar` the exchange calendar, both passed in by the launcher.
    """

    listed = records.watched() if tickers is None else tickers
    if symbol in set(excluded) or symbol not in listed:
        return None
    kind = DataClass(data_class).value
    briefing, problem = None, None
    try:
        briefing = build_briefing(
            records,
            timing=briefing_timing(now, calendar),
            calendar=calendar,
            calendar_id=calendar_id,
            data_class=kind,
            tickers=[symbol],
        )
    except ValueError as error:  # the store, the timing or the calendar refused
        problem = type(error).__name__
    prices = recent(symbol, count=CHART_SESSIONS, retrieved_by=pd.Timestamp(now))
    names = words.NAMES if names is None else names
    return detail_view(
        briefing,
        prices,
        symbol=symbol,
        now=now,
        local_zone=local_zone,
        data_class=kind,
        problem=problem,
        name=names.get(symbol),
        research_url=research_url,
    )


def detail_view(
    briefing: Briefing | None,
    prices: object,
    *,
    symbol: str,
    now: datetime,
    local_zone: tzinfo | None,
    data_class: str,
    problem: str | None = None,
    name: str | None = None,
    research_url: str | None = None,
) -> dict:
    """The screen from a prepared briefing (None when it could not be prepared)."""

    stock = None
    if briefing is not None:
        stock = next((s for s in briefing.stocks if s.symbol == symbol), None)
    look = home.outlook(symbol, stock, briefing, now, local_zone)
    if briefing is None:
        look["sheet"]["paragraphs"][-1] = words.TECHNICAL.format(
            detail=f"the record could not be opened ({problem})."
        )
    fresh = look["state"] == "fresh"
    lately = _lately(symbol, stock)
    chart = _chart(symbol, prices)
    if chart["available"]:
        momentum = any(chip["key"] == "momentum" for chip in lately["chips"])
        chart["why"]["sheet"] = words.no_total(lately=momentum)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "data_class": data_class,
        "generated_at": pd.Timestamp(now).tz_convert("UTC").isoformat(),
        "ticker": symbol,
        "name": name,
        "title": f"{symbol} · {name}" if name else symbol,
        "price": home.price_now(symbol, prices),
        "chart": chart,
        "outlook": _outlook(symbol, stock, look, now, local_zone),
        "five": _five(symbol, stock) if fresh else None,
        "reliable": _reliable(briefing, stock, fresh),
        "lately": lately,
        "learning": _learning(symbol, research_url),
        "demo": words.DEMO if data_class == DataClass.SYNTHETIC.value else None,
    }


# --- the chart: the reader's sessions as it gave them ---


def _chart(symbol: str, found) -> dict:
    empty = {"available": False, "label": None, "points": [], "latest": None, "ranges": []}
    missing = home.prices_words(symbol, found)
    if missing is not None:
        return {**empty, "words": missing}
    rows = list(found.sessions)
    served = [i for i, row in enumerate(rows) if home.finite(row.close) is not None]
    if not served:
        return {**empty, "words": words.NO_CLOSE}
    latest = served[-1]
    points = []
    for index, row in enumerate(rows):
        close, change = home.finite(row.close), home.finite(row.change)
        notes = [words.NOTE_WORDS.get(note, note) for note in row.notes]
        close_words, change_words = words.money(close), words.signed_percent(change)
        points.append(
            {
                "session": row.session,
                "level": home.finite(row.level),
                "close_words": close_words,
                "change_words": change_words,
                "direction": home.direction(change),
                "day": words.short_day(row.session),
                "when": words.point_when(row.session, notes, latest=index == latest),
                "label": words.point_label(
                    row.session, close_words, change_words, notes, latest=index == latest
                ),
            }
        )
    ranges = []
    for key, label, moves, title in words.RANGES:
        first = max(0, min(len(rows) - 1 - moves, latest))  # always from the latest close back
        note = words.range_note(
            rows[first].session, rows[-1].session, shown=len(rows) - first, asked=moves + 1
        )
        ranges.append({"key": key, "label": label, "name": title, "first": first, "note": note})
    return {
        "available": True,
        "words": None,
        "label": words.chart_label(symbol),
        "points": points,
        "latest": latest,
        "ranges": ranges,
        "default": words.DEFAULT_RANGE,
        "drag": words.DRAG,
        "why": {"words": words.NO_TOTAL_BUTTON, "sheet": words.no_total(lately=False)},
    }


# --- the outlook and the five-session total, exactly as recorded ---


def _bar(p: Probabilities) -> dict:
    """Recorded chances for the Up / Flat / Down bar, and their words."""

    values = (p.increase, p.stable, p.decrease)
    return {
        "recorded": dict(zip(CATEGORIES, values, strict=True)),
        "label": words.bar_label(*values),
        "legend": [[key, words.percent(v)] for key, v in zip(words.LEGEND, values, strict=True)],
    }


def _band(value: float) -> str:
    return StableBand(value).words()[1:]


def _outlook(symbol: str, stock: StockBriefing | None, look: dict, now, zone) -> dict:
    if look["state"] != "fresh":
        sheet = look["sheet"]
        return {
            "state": look["state"],
            "level": look["level"],
            "title": sheet["title"],
            "words": sheet["paragraphs"][0],
            "details": words.DETAILS,
            "sheet": sheet,
        }
    first, lineage = stock.next_session, stock.lineage
    made = words.when(lineage.created_at, now=now, local_zone=zone)
    return {
        "state": "fresh",
        "level": "plain",
        "title": words.outlook_title(first.session),
        "sub": words.outlook_sub(first.session, made),
        "pill": {"words": words.EXPERIMENTAL_PILL, "sheet": words.EXPERIMENTAL_SHEET},
        "bar": _bar(first.probabilities),
        "definition": words.outlook_definition(symbol, lineage.as_of_session, _band(lineage.band)),
        "notes": [words.NEWS, words.WHOLE_NUMBERS],
    }


def _five(symbol: str, stock: StockBriefing) -> dict:
    lineage = stock.lineage
    days = [
        {
            "day": words.day_and_date(day.session),
            "bar": _bar(day.probabilities),
            "up": words.percent(day.probabilities.increase),
            "down": words.percent(day.probabilities.decrease),
            "label": words.day_label(
                day.session,
                day.probabilities.increase,
                day.probabilities.stable,
                day.probabilities.decrease,
            ),
        }
        for day in stock.five_sessions
    ]
    total = stock.cumulative
    return {
        "title": words.FIVE_TITLE,
        "sub": None
        if total is None
        else words.five_sub(symbol, stock.cumulative_end, lineage.as_of_session),
        "bar": None if total is None else _bar(total),
        "words": words.NO_TOTAL_RECORDED if total is None else None,
        "definition": None
        if total is None
        else words.five_definition(_band(lineage.cumulative_band)),
        "toggle": list(words.EACH_DAY),
        "days": days,
        "days_note": words.each_day_note(_band(lineage.band)),
    }


# --- how reliable: the development record and the live record's progress ---


def _reliable(briefing: Briefing | None, stock: StockBriefing | None, fresh: bool) -> dict:
    if briefing is None:
        return {
            "title": words.RELIABLE_TITLE,
            "development": {"state": "unreadable", "words": words.RELIABLE_UNREADABLE},
            "live": None,
        }
    return {
        "title": words.RELIABLE_TITLE,
        "development": _development(briefing, stock, fresh),
        "live": _live(briefing),
    }


def _development(briefing: Briefing, stock: StockBriefing | None, fresh: bool) -> dict:
    if not fresh:
        return {"state": "later", "words": words.DEVELOPMENT_LATER}
    track = {found.category: found for found in stock.track}
    up = track.get("increase")
    if up is None:
        return {"state": "none", "words": words.DEVELOPMENT_NONE}
    if up.status != OK:
        text = words.development_short(up.bin_low, up.bin_high, up.n, up.dates, MIN_CLUSTERS)
        return {"state": "insufficient", "flag": words.NOT_ENOUGH, "words": text}
    synthetic = briefing.data_class == DataClass.SYNTHETIC.value
    found = briefing.reliability.development
    if found is not None and found.run_id == stock.lineage.evaluation_run_id:
        scope = words.development_scope(list(found.stocks), found.outcome_end, synthetic)
    else:  # the stored evaluation behind this outlook, though the briefing names another
        scope = words.development_scope([], None, synthetic)
    rows = []
    for category in CATEGORIES:
        bucket = track.get(category)
        if bucket is None:
            continue
        label, _ = words.CATEGORY_WORDS[category]
        happened = (
            words.happened(bucket.observed, bucket.low, bucket.high)
            if bucket.status == OK
            else words.NOT_ENOUGH
        )
        rows.append([f"{label} {words.about(bucket.bin_low, bucket.bin_high)}", happened])
    paragraphs = words.development_sheet_paragraphs(
        scope, up.bin_low, up.bin_high, up.observed, up.n, up.dates, synthetic
    )
    technical = words.TECHNICAL.format(detail=briefing.reliability.development_words)
    return {
        "state": "ok",
        "headline": words.development_headline(up.bin_low, up.bin_high, up.observed),
        "meta": words.development_meta(scope, up.n, up.dates, up.low, up.high),
        "button": words.READ_THIS,
        "sheet": {
            "title": words.READ_THIS_TITLE,
            "columns": list(words.READ_THIS_COLUMNS),
            "rows": rows,
            "paragraphs": [*paragraphs, technical],
        },
    }


def _live(briefing: Briefing) -> dict:
    found = briefing.reliability.prospective
    sessions, needed = found.settled_sessions, MIN_SESSIONS
    return {
        "state": found.status,
        "flag": None if found.status == OK else words.NOT_ENOUGH,
        "words": words.live_words(
            found.status,
            sessions,
            found.scored_targets,
            sessions_needed=needed,
            targets_needed=MIN_SCORED,
        ),
        "scope": words.LIVE_SCOPE,
        "meter": {
            "value": min(sessions, needed),
            "max": needed,
            "label": words.meter_label(sessions, needed),
        },
        "button": words.JUDGE,
        "sheet": words.judge_sheet(needed, MIN_SCORED, found.words),
    }


# --- lately: only what the evening job recorded with the forecast ---


def _lately(symbol: str, stock: StockBriefing | None) -> dict:
    nothing = {"title": words.LATELY_TITLE, "sub": None, "chips": [], "words": words.NOTHING_LATELY}
    if stock is None or stock.described_as_of is None or stock.lineage is None:
        return nothing
    nothing["words"] = words.NOTHING_SHOWN  # a record exists, but nothing of it can be shown
    as_of = stock.described_as_of
    chips = []
    if stock.streak.available:
        chips.append(
            {
                "key": "run",
                "words": words.run_chip(stock.streak.direction, stock.streak.length),
                "sheet": words.run_sheet(symbol, as_of, _band(stock.lineage.band)),
            }
        )
    context = stock.context
    if context.available and context.regime_basis == RECORDED:
        if context.ret_20 is not None:
            chips.append(
                {
                    "key": "momentum",
                    "words": words.momentum_chip(context.ret_20),
                    "sheet": words.momentum_sheet(symbol, as_of),
                }
            )
        if context.regime in words.REGIME_WORDS:
            chips.append(
                {
                    "key": "regime",
                    "words": words.REGIME_WORDS[context.regime],
                    "sheet": words.regime_sheet(symbol, as_of),
                }
            )
    if not chips:
        return nothing
    return {
        "title": words.LATELY_TITLE,
        "sub": words.lately_sub(as_of),
        "chips": chips,
        "words": None,
    }


# --- for learning: the research page ---


def _learning(symbol: str, research_url: str | None) -> dict:
    url = None
    if research_url:
        url = urljoin(research_url.rstrip("/") + "/", words.RESEARCH_PAGE)
    return {
        "title": words.LEARNING_TITLE,
        "url": url,
        "words": words.research_link(symbol),
        "sub": words.research_sub(symbol) if url else words.RESEARCH_ELSEWHERE,
        "elsewhere": words.RESEARCH_ELSEWHERE,
    }
