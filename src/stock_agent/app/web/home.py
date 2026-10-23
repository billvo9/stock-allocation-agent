"""
The Lab app's Home view model (docs/product/lab_app_design.md, "Home in
detail"): JSON-ready data with display-ready words and the raw recorded values.

Where each part comes from
    The status line and the outlooks come from the premarket briefing's report
    (`build_briefing`), prepared at the request time: it reads the Stock Lab's
    store read-only, counts only the records that existed by then and never
    recomputes a probability. A fresh outlook is shown exactly as recorded
    (rounded for display only); every other state is named, with its reason.
    The price, its session and that session's change come from the injected
    recent-prices reader, exactly as it gives them (decision A6): nothing is
    computed from the prices, so there is no dollar change and no change over
    a period. The one-month line is the reader's split-consistent levels.

The status line, first match
    bad      the record cannot be read; the ledger check failed; a stock's
             record failed its check
    neutral  real outlooks have not started (not activated); no run yet
    warn     the latest run, after the last success, was not ok; an evening
             is missed (a stock is stale and not pending); an outlook is
             withheld
    ok       "Updated last night, 8:31 PM"

Pending: from 20:00 New York on a session day (the briefing's "advance"
state), a stock one session behind is waiting for tonight's update, until a
run for tonight's session has finished.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import datetime, tzinfo

import pandas as pd

from stock_agent.app.web import words
from stock_agent.stock_lab.briefing.report import (
    FRESH,
    STALE,
    Briefing,
    ReadOnlyRecords,
    StockBriefing,
    build_briefing,
)
from stock_agent.stock_lab.briefing.timing import (
    ADVANCE,
    CLOSE_AVAILABLE,
    briefing_timing,
)
from stock_agent.stock_lab.contracts import DataClass, StableBand

SCHEMA = "lab_app_home"
VERSION = 1
SPARK_SESSIONS = 22  # the one-month line: 21 sessions of moves and the session before
OBSERVED = "observed"
UNVERIFIED = ("record_check_failed", "ledger_unverified")
WITHHELD = ("as_of_not_prior_session", "target_mismatch")


def build_home(
    records: ReadOnlyRecords,
    *,
    now: datetime,
    calendar: Callable,
    calendar_id: str,
    data_class: DataClass | str,
    recent: Callable[..., object],
    local_zone: tzinfo | None,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
    names: Mapping[str, str] | None = None,
) -> dict:
    """
    Home at `now` (timezone-aware). `recent(symbol, count=, retrieved_by=)` is the
    recent-prices reader, passed in by the launcher; `calendar` is the exchange
    calendar the briefing plans with.
    """

    kind = DataClass(data_class).value
    briefing, problem = None, None
    try:
        briefing = build_briefing(
            records,
            timing=briefing_timing(now, calendar),
            calendar=calendar,
            calendar_id=calendar_id,
            data_class=kind,
            tickers=tickers,
            excluded=excluded,
        )
        listed = tuple(stock.symbol for stock in briefing.stocks)
    except ValueError as error:  # the store, the timing or the calendar refused
        problem = type(error).__name__
        listed = _listed(records, tickers, excluded)
    moment = pd.Timestamp(now)
    prices = {
        symbol: recent(symbol, count=SPARK_SESSIONS, retrieved_by=moment) for symbol in listed
    }
    return home_view(
        briefing,
        prices,
        now=now,
        local_zone=local_zone,
        tickers=listed,
        data_class=kind,
        problem=problem,
        names=names,
    )


def _listed(records, tickers, excluded) -> tuple[str, ...]:
    if tickers is None:
        try:
            tickers = records.watched()
        except ValueError:
            tickers = ()
    return tuple(t for t in tickers if t not in set(excluded))


def home_view(
    briefing: Briefing | None,
    prices: Mapping[str, object],
    *,
    now: datetime,
    local_zone: tzinfo | None,
    tickers: Sequence[str],
    data_class: str,
    problem: str | None = None,
    names: Mapping[str, str] | None = None,
) -> dict:
    """
    The Home view from a prepared briefing (None when it could not be prepared).
    `names` are the stocks' display names (default: the watchlist's company names).
    """

    names = words.NAMES if names is None else names
    by_symbol = {} if briefing is None else {s.symbol: s for s in briefing.stocks}
    stocks = []
    for symbol in tickers:
        stock = by_symbol.get(symbol)
        stocks.append(
            {
                "ticker": symbol,
                "name": names.get(symbol),
                "price": _price(symbol, prices.get(symbol)),
                "outlook": _outlook(symbol, stock, briefing, now, local_zone),
            }
        )
    sessions = [s["price"]["session"] for s in stocks if s["price"]["available"]]
    for stock in stocks:  # a close older than the others names its own day
        price = stock["price"]
        price["older"] = price["available"] and price["session"] != max(sessions)
    bands = [
        StableBand(s.lineage.band).words()[1:]
        for s in by_symbol.values()
        if s.status == FRESH and s.lineage is not None
    ]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "data_class": data_class,
        "generated_at": pd.Timestamp(now).tz_convert("UTC").isoformat(),
        "status": _status(briefing, problem, now, local_zone),
        "prices_note": words.prices_note(max(sessions)) if sessions else None,
        "stocks": stocks,
        "footer": words.FOOTER,
        "explain": {
            "title": words.EXPLAIN_TITLE,
            "paragraphs": words.explain(next(iter(bands), None)),
        },
        "demo": words.DEMO if data_class == DataClass.SYNTHETIC.value else None,
    }


# --- the status line ---


def _status(briefing: Briefing | None, problem: str | None, now, zone) -> dict:
    if briefing is None:
        return _line(
            "bad",
            words.RECORD_UNREADABLE,
            [f"The Stock Lab's record could not be opened ({problem})."],
        )
    update = briefing.latest_update

    def when(stamp: str) -> str:
        return words.when(stamp, now=now, local_zone=zone)

    details = [
        f"Last successful update: {when(update.last_success_at)}."
        if update.last_success_at
        else words.NO_SUCCESS
    ]
    if update.last_run_at:
        ran = words.RUN_WORDS.get(update.last_status, update.last_status)
        details.append(f"Latest run: {ran}, {when(update.last_run_at)}.")
    if briefing.gap_note:
        details.append(briefing.gap_note)
    stocks = briefing.stocks
    if not briefing.ledger.verified:
        return _line("bad", words.LEDGER_FAILED, [*details, f"Check: {briefing.ledger.problem}"])
    if any(s.reason == "record_check_failed" for s in stocks):
        return _line("bad", words.RECORD_FAILED, details)
    if update.activation == "not_authorized":
        return _line("neutral", words.NOT_STARTED, details)
    if update.last_run_at is None:
        return _line("neutral", words.NO_RUN, details)
    if update.last_status != "ok" and (
        update.last_success_at is None or update.last_run_at > update.last_success_at
    ):
        template = words.RUN_PROBLEM.get(update.last_status, words.RUN_PROBLEM_OTHER)
        text = template.format(when=when(update.last_run_at), status=update.last_status)
        return _line("warn", text, details)
    late = [s for s in stocks if s.status == STALE and not _pending(s, briefing)]
    if late:
        prior = pd.Timestamp(briefing.prior_session).date()
        evening = pd.Timestamp.combine(prior, CLOSE_AVAILABLE).tz_localize(words.NEW_YORK)
        ran = pd.Timestamp(update.last_run_at) >= evening
        text = words.missed(
            words.evening_name(prior, now), ran=ran, missing=len(late), total=len(stocks)
        )
        return _line("warn", text, details)
    withheld = [s for s in stocks if s.reason in WITHHELD]
    if withheld:
        return _line("warn", words.withheld(len(withheld)), details)
    if update.last_success_at is None:
        return _line("neutral", words.NO_SUCCESS, details)
    return _line("ok", words.updated(when(update.last_success_at)), details)


def _line(level: str, text: str, details: list[str]) -> dict:
    return {"level": level, "words": text, "details": details}


def _pending(stock: StockBriefing, briefing: Briefing) -> bool:
    return (
        briefing.state == ADVANCE
        and stock.status == STALE
        and stock.sessions_behind == 1
        and briefing.latest_update.expected_session != briefing.prior_session
    )


# --- the outlook chip ---


def _outlook(symbol, stock: StockBriefing | None, briefing, now, zone) -> dict:
    if stock is not None and stock.status == FRESH and stock.next_session is not None:
        return _fresh(symbol, stock, now, zone)
    market = None if briefing is None else briefing.market_date
    day = None if market is None else words.weekday(market)
    if stock is None:
        state = "unverified"
    elif stock.status == STALE:
        state = "pending" if _pending(stock, briefing) else "stale"
    elif stock.reason in UNVERIFIED:
        state = "unverified"
    elif stock.reason == "refused":
        state = "refused"
    elif stock.reason in WITHHELD:
        state = "withheld"
    else:
        state = "none"
    level, chip, title, plain = _EMPTY[state]
    detail = stock.words if stock is not None else "The forecast record could not be opened."
    return {
        "state": state,
        "level": level,
        "chip": None,
        "words": chip.format(day=day),
        "day": day,
        "for_session": market,
        "recorded": None,
        "sheet": {
            "title": title.format(day=None if market is None else words.long_weekday(market)),
            "rows": [],
            "paragraphs": [plain.format(ticker=symbol), f"Details: {detail}"],
        },
    }


# state: (level, chip words, sheet title, plain explanation)
_EMPTY = {
    "pending": (
        "neutral",
        words.PENDING_CHIP,
        "{day}'s outlook comes tonight",
        (
            "Each weekday evening, after the close, the Lab records the next day's chances. "
            "Tonight's update hasn't recorded {ticker}'s yet."
        ),
    ),
    "stale": (
        "warn",
        words.STALE_CHIP,
        "No outlook for {day}",
        (
            "The latest update recorded nothing new for {ticker}, so there is no outlook for this "
            "day. An older outlook is never shown in its place."
        ),
    ),
    "none": (
        "neutral",
        words.NONE_CHIP,
        "No outlook yet",
        "No outlook is recorded for {ticker}. Nothing is shown in its place.",
    ),
    "refused": (
        "neutral",
        words.REFUSED_CHIP,
        "The model gave no outlook",
        "The model recorded that it could not give {ticker}'s chances for this day, and why.",
    ),
    "unverified": (
        "bad",
        words.UNVERIFIED_CHIP,
        "Outlook can't be verified",
        (
            "A record didn't pass its integrity check or couldn't be opened, so the app shows "
            "nothing rather than a number it can't vouch for. The app only reads the record and "
            "never changes it."
        ),
    ),
    "withheld": (
        "warn",
        words.WITHHELD_CHIP,
        "Outlook withheld",
        "{ticker}'s recorded outlook doesn't match the exchange calendar, so it isn't shown.",
    ),
}


def _fresh(symbol: str, stock: StockBriefing, now, zone) -> dict:
    first = stock.next_session
    p = first.probabilities
    band = StableBand(stock.lineage.band).words()[1:]
    lineage = stock.lineage
    made = words.when(lineage.created_at, now=now, local_zone=zone)
    return {
        "state": "fresh",
        "level": "plain",
        "chip": {"up": words.percent(p.increase), "down": words.percent(p.decrease)},
        "words": None,
        "day": words.weekday(first.session),
        "for_session": first.session,
        "recorded": {"increase": p.increase, "stable": p.stable, "decrease": p.decrease},
        "sheet": {
            "title": f"{symbol}: {words.long_weekday(first.session)}'s outlook",
            "rows": [
                [f"Up more than {band}", words.percent(p.increase)],
                [f"Flat, within {band}", words.percent(p.stable)],
                [f"Down more than {band}", words.percent(p.decrease)],
            ],
            "paragraphs": [
                (
                    f"The chances recorded for {words.short_day(first.session)}, compared with "
                    f"{words.long_weekday(lineage.as_of_session)}'s close."
                ),
                f"Recorded {made}, after the close. It knows nothing about news since then.",
                words.WHOLE_NUMBERS,
                words.EXPERIMENTAL,
            ],
        },
    }


# --- the price, as the reader gave it ---


def _number(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _price(symbol: str, found) -> dict:
    empty = {
        "available": False,
        "close": None,
        "close_words": words.DASH,
        "session": None,
        "session_words": None,
        "change": None,
        "change_words": words.DASH,
        "direction": "none",
        "line": [],
    }
    state = getattr(found, "state", None)
    if state != OBSERVED:
        text = words.NO_PRICES.get(state, words.PRICES_UNREADABLE).format(ticker=symbol)
        return {**empty, "words": text}
    served = [row for row in found.sessions if _number(row.close) is not None]
    if not served:
        return {**empty, "words": words.NO_CLOSE}
    latest = served[-1]
    close, change = _number(latest.close), _number(latest.change)
    if change is None:
        direction = "none"
    else:
        direction = "up" if change > 0 else "down" if change < 0 else "flat"
    return {
        "available": True,
        "close": close,
        "close_words": words.money(close),
        "session": latest.session,
        "session_words": words.short_day(latest.session),
        "change": change,
        "change_words": words.signed_percent(change),
        "direction": direction,
        "line": [_number(row.level) for row in found.sessions],
        "words": None,
    }
