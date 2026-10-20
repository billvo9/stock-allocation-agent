"""
The Premarket Briefing page (Streamlit): the latest briefing the briefing
script wrote on this computer, real or synthetic demo (one kind at a time,
never mixed), laid out in the Lab's design, with what the duplicate-send log
recorded for its market date.

Read-only. The page reads the written JSON copies and the log through
`stock_lab.briefing.stored` and never loads `delivery` (the writers): it
prepares no briefing, runs no job, writes no file and sends nothing. Every
number and sentence comes from the written briefing, in the words of its HTML
and text renderings (`stock_lab.briefing.render`); the page computes nothing.
Text read from a file is shown literally (`prose.literal`): no markdown in it
is rendered. Every category is written in words beside its marker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import stock_agent.stock_lab.briefing.render as rendering
from stock_agent.app import briefing_wording as words
from stock_agent.app import wording
from stock_agent.app.prose import bullets, literal, say, section
from stock_agent.stock_lab.briefing.report import (
    FRESH,
    STALE,
    Briefing,
    StockBriefing,
    generated_local,
)
from stock_agent.stock_lab.briefing.report import new_york_time as when
from stock_agent.stock_lab.briefing.stored import (
    FAILED,
    IN_PROGRESS,
    NOT_RECORDED,
    REFUSED,
    SENT,
    UNREADABLE,
    DeliveryStatus,
    Listing,
    StoredBriefing,
)
from stock_agent.stock_lab.briefing.timing import NEW_YORK, PREMARKET, long_date
from stock_agent.ui import components as ui

PREFIX = "briefing-"
STATUS_LEVELS = {FRESH: "VALID", STALE: "WARNING", "unavailable": "UNAVAILABLE"}
CATEGORIES = rendering.CATEGORIES
DASH = rendering.DASH
# For a preview, the one delivery-log state worth saying, across channels.
PREVIEW_ORDER = (SENT, IN_PROGRESS, FAILED, NOT_RECORDED, UNREADABLE)


def new_york_today() -> date:
    return datetime.now(NEW_YORK).date()


def category(name: str) -> str:
    """A category in words beside its marker: '▲ Increase'."""

    return f"{rendering.MARKERS[name]} {rendering.WORDS[name]}"


def shown(path: Path) -> str:
    """A path with the home folder as ~."""

    home, text = str(Path.home()), str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text


def _sentence(text: str | None) -> str:
    text = (text or "").strip()
    return text[:1].upper() + text[1:]


def _plural(forms: tuple[str, str], count: int, **values) -> str:
    return (forms[0] if count == 1 else forms[1]).format(count=count, **values)


def _day(market_date: str) -> str:
    return long_date(date.fromisoformat(market_date))


def option_label(entry: StoredBriefing) -> str:
    """'Tuesday 20 October 2026 · premarket briefing, prepared 08:30 New York'."""

    day = _day(entry.market_date)
    if entry.briefing is None:
        return words.UNREADABLE_OPTION.format(day=day, name=entry.name)
    briefing = entry.briefing
    local = generated_local(briefing)
    prepared = (
        f"{local:%H:%M} New York"
        if local.date().isoformat() == briefing.market_date
        else f"{local:%a} {local.day} {local:%b}, {local:%H:%M} New York"
    )
    kind = words.PREMARKET_OPTION if briefing.state == PREMARKET else words.PREVIEW_OPTION
    return words.OPTION.format(day=day, kind=kind, prepared=prepared)


def _change_words(stock: StockBriefing) -> str:
    """'Increase +2 · Stable −1 · Decrease −1' (the header names the points)."""

    change = stock.change
    if not change.comparable or change.next_session is None:
        return words.NOT_COMPARABLE
    moves = [rendering.points(getattr(change.next_session, c)) for c in CATEGORIES]
    if all(move == "0 pts" for move in moves):
        return words.NO_CHANGE
    return " · ".join(
        f"{rendering.WORDS[c]} {move.removesuffix(' pts')}"
        for c, move in zip(CATEGORIES, moves, strict=True)
    )


def glance_table(briefing: Briefing) -> pd.DataFrame:
    """At a glance: one row per stock, the stored probabilities or the reason there are none."""

    stock_column, status_column, change_column = words.GLANCE_COLUMNS
    rows = []
    for stock in briefing.stocks:
        cells = rendering.glance(stock)
        row = {stock_column: stock.symbol}
        if isinstance(cells, str):
            row[status_column] = cells
            row.update({category(c): DASH for c in CATEGORIES})
            row[change_column] = DASH
        else:
            row[status_column] = rendering.STATUS_WORDS[stock.status]
            row.update({category(c): cell for c, cell in zip(CATEGORIES, cells[:3], strict=True)})
            row[change_column] = _change_words(stock)
        rows.append(row)
    return pd.DataFrame(rows).set_index(stock_column)


def five_table(stock: StockBriefing) -> pd.DataFrame:
    session_column, ahead_column = words.FIVE_COLUMNS
    return pd.DataFrame(
        [
            {
                session_column: s.session,
                ahead_column: s.horizon,
                **{category(c): rendering.pct(getattr(s.probabilities, c)) for c in CATEGORIES},
            }
            for s in stock.five_sessions
        ]
    ).set_index(session_column)


def _labelled(rows: Sequence[tuple[str, str]], column: str) -> pd.DataFrame:
    labels, texts = zip(*rows, strict=True)
    return pd.DataFrame({column: texts}, index=pd.Index(labels, name=words.ITEM_COLUMN))


def detail_table(stock: StockBriefing) -> pd.DataFrame:
    rows = rendering.detail_rows(stock)
    if stock.next_session is not None and stock.lineage is not None and stock.cumulative:
        rows.insert(0, (words.TOTAL_LABEL, rendering.total_words(stock)))
    return _labelled(rows, words.DETAIL_COLUMN)


def facts_table(briefing: Briefing) -> pd.DataFrame:
    return _labelled(rendering.facts(briefing), words.FACT_COLUMN)


def _bucket_table(buckets: Sequence) -> pd.DataFrame:
    frame = pd.DataFrame(rendering.bucket_rows(buckets), columns=list(rendering.BUCKET_COLUMNS))
    return frame.set_index(rendering.BUCKET_COLUMNS[0])


def _channel(status: DeliveryStatus) -> str:
    name = status.channel or ""
    return words.CHANNELS.get(name, words.CHANNEL.format(channel=name))


def _premarket_item(status: DeliveryStatus) -> ui.Item:
    at = when(status.at)
    if status.status == UNREADABLE:
        text = f"{_sentence(status.detail)}."
        return ui.Item("WARNING", words.LOG_UNREADABLE_WORD, words.LOG_UNREADABLE_TITLE, text)
    if status.status == SENT:
        text = words.DELIVERED_PREVIEW if status.channel == "preview" else words.DELIVERED_OTHER
        return ui.Item(
            "VALID",
            words.DELIVERED,
            words.DELIVERED_TITLE.format(channel=_channel(status)),
            text.format(at=at, attempts=status.attempts),
        )
    if status.status == FAILED:
        text = words.FAILED_TEXT.format(
            detail=_sentence(status.detail), at=at, attempts=status.attempts
        )
        return ui.Item("WARNING", words.FAILED_WORD, words.FAILED_TITLE, text)
    if status.status == IN_PROGRESS:
        text = words.UNFINISHED_TEXT.format(at=at)
        return ui.Item("WARNING", words.UNFINISHED_WORD, words.UNFINISHED_TITLE, text)
    return ui.Item(
        "WARNING", words.NOT_RECORDED_WORD, words.NOT_RECORDED_TITLE, words.NOT_RECORDED_TEXT
    )


def delivery_items(briefing: Briefing, statuses: Sequence[DeliveryStatus]) -> list[ui.Item]:
    """
    What the duplicate-send log recorded for the briefing's market date, in
    words: one row per channel for a premarket briefing; one row for a
    preview, which is never a delivery, naming that date's delivery state.
    """

    if briefing.state == PREMARKET:
        return [_premarket_item(status) for status in statuses]
    found = min(statuses, key=lambda s: PREVIEW_ORDER.index(s.status))
    tail = words.PREVIEW_STATES[found.status].format(at=when(found.at))
    text = words.PREVIEW_TEXT + tail
    return [ui.Item("NOTICE", words.PREVIEW_WORD, words.PREVIEW_TITLE, text)]


def _track(stock: StockBriefing, name: str) -> str | None:
    """The development track record of the same range, as a sentence."""

    text = rendering.track_text(stock, name)
    return None if text is None else literal(_sentence(text) + ".")


def _stock(stock: StockBriefing) -> None:
    status = rendering.STATUS_WORDS[stock.status]
    with st.expander(literal(f"{stock.symbol} · {status}")):
        ui.pill(STATUS_LEVELS[stock.status], status)
        say(literal(stock.words))
        if stock.next_session is not None and stock.lineage is not None:
            band = rendering.band(stock.lineage.band)
            say(
                literal(words.NEXT_SESSION.format(session=stock.next_session.session, band=band)),
                "caption",
            )
            p = stock.next_session.probabilities
            ui.stat_row(
                [
                    ui.Stat(category(c), rendering.pct(getattr(p, c)), caption=_track(stock, c))
                    for c in CATEGORIES
                ]
            )
            say(f"**{words.CHANGE}** {literal(rendering.change_text(stock))}")
            say(literal(words.FIVE_CAPTION.format(band=band)), "caption")
            st.table(five_table(stock))
        st.table(detail_table(stock))


def _reliability(briefing: Briefing) -> None:
    part = briefing.reliability
    section(words.RELIABILITY_TITLE, words.RELIABILITY_CAPTION)
    say(f"#### {words.DEVELOPMENT_TITLE}")
    for line in rendering.development_lines(briefing):
        say(literal(line))
    found = part.development
    if found is not None and found.buckets:
        table = words.DEVELOPMENT_TABLE.format(
            estimator=found.estimator, method=rendering.DEVELOPMENT_METHOD
        )
        say(literal(table), "caption")
        st.table(_bucket_table(found.buckets))
    prospective = part.prospective
    say(f"#### {words.PROSPECTIVE_TITLE}")
    say(literal(prospective.words))
    for group in prospective.groups:
        if group.target == "1":
            table = words.PROSPECTIVE_TABLE.format(
                label=rendering.group_label(group), method=rendering.PROSPECTIVE_METHOD
            )
            say(literal(table), "caption")
            st.table(_bucket_table(group.buckets))
    for entry in prospective.rolling:
        say(literal(rendering.rolling_text(entry)))
    say(f"**{words.COMPARED}** {literal(prospective.comparison_words)}")
    if prospective.comparison_caveat:  # only when the persistence flag is on, as written
        say(literal(prospective.comparison_caveat))
    for note in (*part.notes, rendering.INTERVAL_NOTE):
        say(literal(note), "caption")


def _briefing(
    entry: StoredBriefing,
    statuses: Sequence[DeliveryStatus],
    directory: str,
    today: date,
) -> None:
    briefing = entry.briefing
    synthetic = briefing.data_class == "synthetic"
    premarket = briefing.state == PREMARKET
    ui.pills(
        [
            ("WARNING", words.SYNTHETIC_BADGE) if synthetic else ("NOTICE", words.REAL_BADGE),
            ("VALID", words.PREMARKET_BADGE) if premarket else ("WARNING", words.PREVIEW_BADGE),
        ]
    )
    if synthetic:
        say(words.SYNTHETIC_NOTICE, "warning")
    dateline = words.DATELINE.format(
        market=rendering.market(briefing),
        prepared=rendering.prepared(briefing),
        state=briefing.state_words,
    )
    say(literal(dateline))
    if not synthetic and briefing.market_date < today.isoformat():
        market = rendering.market(briefing)
        say(words.NOT_TODAY.format(market=market, today=long_date(today)), "warning")
    if briefing.warnings:
        section(words.NOTICES_TITLE)
        for warning in briefing.warnings:
            level = "blocking" if rendering.blocking(briefing, warning) else "notice"
            say(literal(warning), level)
    section(words.GLANCE_TITLE, words.GLANCE_CAPTION)
    if briefing.stocks:
        st.table(glance_table(briefing))
    else:
        say(words.EMPTY_WATCHLIST)
    section(words.STATUS_TITLE)
    st.table(facts_table(briefing))
    section(words.DELIVERY_TITLE, words.DELIVERY_CAPTION)
    ui.status_list(delivery_items(briefing, statuses))
    if briefing.stocks:
        section(words.STOCKS_TITLE, words.STOCKS_CAPTION)
        for stock in briefing.stocks:
            _stock(stock)
    _reliability(briefing)
    section(words.LIMITATIONS_TITLE)
    bullets(literal(line) for line in briefing.limitations)
    source = words.SOURCE.format(
        name=entry.name,
        directory=directory,
        schema=briefing.schema,
        version=briefing.schema_version,
    )
    say(literal(source), "caption")


def _choose(listings: dict[str, Listing]) -> tuple[str, StoredBriefing | None]:
    kinds = list(words.KINDS)
    default = next((kind for kind in kinds if listings[kind].briefings), kinds[0])
    with ui.card("briefing-controls"):
        say(f"**{words.CONTROLS_TITLE}**")
        left, right = st.columns([2, 3])
        kind = left.radio(
            words.KIND_LABEL,
            kinds,
            index=kinds.index(default),
            format_func=words.KINDS.get,
            horizontal=True,
            help=words.KIND_HELP,
            key=f"{PREFIX}kind",
        )
        options = {entry.name: entry for entry in listings[kind].briefings}
        if not options:
            return kind, None
        name = right.selectbox(
            words.CHOOSE_LABEL,
            list(options),
            format_func=lambda key: option_label(options[key]),
            help=words.CHOOSE_HELP,
            key=f"{PREFIX}choice-{kind}",
        )
    return kind, options[name]


def _reason(text: str | None) -> None:
    """A reason given by the reader, shown literally."""

    say(literal(f"{_sentence(text)}."), "caption")


def render(
    list_briefings: Callable[[str], Listing],
    delivery_status: Callable[[str, str], Sequence[DeliveryStatus]],
    directory: Callable[[], Path],
    today: Callable[[], date],
) -> None:
    ui.hero(words.TITLE, eyebrow=words.EYEBROW, lede=words.LEDE, key="briefing")
    ui.pills([("WARNING", words.EXPERIMENTAL_BADGE), ("NOTICE", words.READ_ONLY_BADGE)])
    folder = literal(shown(directory()))
    listings = {kind: list_briefings(kind) for kind in words.KINDS}
    kind, chosen = _choose(listings)
    listing = listings[kind]
    unreadable = [entry for entry in listing.briefings if entry.briefing is None]
    if listing.state == REFUSED:
        with ui.state_card("briefing-refused"):
            ui.pill("WARNING", words.REFUSED_TITLE)
            say(words.REFUSED.format(directory=folder))
            _reason(listing.reason)
    elif chosen is None:
        with ui.state_card("briefing-none"):
            ui.pill("FUTURE", words.NONE_TITLE)
            say(words.NONE[kind].format(directory=folder))
            st.code(words.COMMANDS[kind], language="bash")
    elif chosen.briefing is None:
        with ui.state_card("briefing-unreadable"):
            ui.pill("WARNING", words.UNREADABLE_TITLE)
            say(words.UNREADABLE.format(name=literal(chosen.name)))
            _reason(chosen.problem)
    else:
        newer = [entry for entry in unreadable if entry.market_date >= chosen.market_date]
        if newer:
            say(
                literal(
                    words.NEWER_UNREADABLE.format(
                        day=_day(newer[0].market_date),
                        reason=newer[0].problem,
                        shown=_day(chosen.market_date),
                    )
                ),
                "warning",
            )
        statuses = delivery_status(chosen.market_date, chosen.data_class)
        _briefing(chosen, statuses, folder, today())
    others = [entry for entry in unreadable if entry is not chosen]
    if others:
        names = literal(", ".join(entry.name for entry in others))
        say(_plural(words.OTHER_UNREADABLE, len(others), names=names), "caption")
    if listing.older:
        say(_plural(words.OLDER, listing.older), "caption")
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
