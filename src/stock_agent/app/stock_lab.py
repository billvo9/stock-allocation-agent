"""
The Stock Probability page (Streamlit): the saved watchlist, one stock's
history in detail, and the model-diagnostics placeholder.

The watchlist is stored in the private SQLite store (`stock_lab.store`), so
it survives restarts; the page writes only watchlist events there. Prices
are read from the same store (written by the owner-run fetch script, which
the page never imports) or come from the synthetic demo generator, always
labelled. Every number is computed by `stock_lab.view` from stored history;
the page lays it out and computes nothing. No probability is shown yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
import streamlit as st

from stock_agent.app import persist, stock_figures, wording
from stock_agent.app import stock_wording as words
from stock_agent.app.prose import bullets, say, section
from stock_agent.stock_lab.contracts import DEFAULT_STABLE_BAND, StableBand
from stock_agent.stock_lab.runs import RunRule
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import MAX_TICKERS_PER_PASTE, parse_tickers
from stock_agent.stock_lab.synthetic import demo_spec, is_demo
from stock_agent.stock_lab.view import (
    TYPICAL_LABEL,
    demo_rows,
    feature_table,
    headline,
    price_table,
    real_bars_shown,
    recent_sessions,
    recent_table,
    run_length_table,
    search,
    stock_detail,
    summary_table,
    threshold_table,
    watch_rows,
)
from stock_agent.ui import components as ui

# The saved-state key must not start with PREFIX, or each save would copy the last one.
SAVED = "stock_lab_saved"
FLASH = "stock_lab_flash"
PREFIX = "stock-lab-"
KEPT = (f"{PREFIX}search", f"{PREFIX}stock", f"{PREFIX}band", f"{PREFIX}rule")
BANDS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)


def _band_words(tau: float) -> str:
    return StableBand(tau).words()


def _add(store: StockLabStore, text: str) -> None:
    try:
        parsed = parse_tickers(text)
    except ValueError:
        say(words.TOO_MANY.format(limit=MAX_TICKERS_PER_PASTE), "warning")
        return
    results = store.add(parsed)
    added = [result.ticker for result in results if result.added]
    already = [result.ticker for result in results if not result.added]
    if added:
        say(words.ADDED.format(tickers=", ".join(added)), "success")
    if already:
        say(words.ALREADY.format(tickers=", ".join(already)), "notice")
    for entry in parsed:
        if entry.rewritten:
            say(
                words.REWRITTEN.format(entered=entry.entered.strip(), ticker=entry.ticker),
                "caption",
            )
    refused = [entry for entry in parsed if not entry.ok]
    if refused:
        say(f"**{words.REFUSED_TITLE}**")
        st.table(
            pd.DataFrame(
                {
                    words.REFUSED_COLUMNS[0]: [entry.entered for entry in refused],
                    words.REFUSED_COLUMNS[1]: [entry.detail for entry in refused],
                }
            ).set_index(words.REFUSED_COLUMNS[0])
        )


def _watchlist(store: StockLabStore, demos: Mapping[str, pd.DataFrame], band: StableBand) -> None:
    section(words.WATCHLIST_TITLE, words.WATCHLIST_SCOPE)
    flash = st.session_state.pop(FLASH, None)
    if flash:
        say(flash, "success")
    with st.form(f"{PREFIX}add-form", clear_on_submit=True, border=False):
        text = st.text_area(
            words.ADD_LABEL, placeholder=words.ADD_PLACEHOLDER, help=words.ADD_HELP, height=90
        )
        submitted = st.form_submit_button(words.ADD_BUTTON)
    if submitted and text.strip():
        _add(store, text)

    rows = watch_rows(store, band=band)
    if not rows:
        with ui.state_card("stock-lab-empty"):
            ui.pill("FUTURE", words.EMPTY_TITLE)
            say(words.EMPTY)
    else:
        query = st.text_input(
            words.SEARCH_LABEL, placeholder=words.SEARCH_PLACEHOLDER, key=f"{PREFIX}search"
        )
        shown = search(rows, query)
        if shown:
            st.dataframe(
                summary_table(shown, band=band),
                hide_index=True,
                column_config={"Stock": st.column_config.TextColumn(pinned=True)},
                key=f"{PREFIX}watchlist",
            )
            say(words.WATCHLIST_BAND_NOTE.format(band=band.words()), "caption")
        else:
            st.text(words.NO_MATCH.format(query=query.strip()))
        if any(row.classification.status.value == "unverified" for row in rows):
            say(words.NOT_CHECKED_NOTE, "caption")
        removing = st.multiselect(
            words.REMOVE_LABEL,
            [row.ticker for row in rows],
            help=words.REMOVE_HELP,
            key=f"{PREFIX}remove",
        )
        if st.button(words.REMOVE_BUTTON, disabled=not removing, key=f"{PREFIX}remove-button"):
            removed = store.remove(removing)
            if removed:
                st.session_state[FLASH] = words.REMOVED.format(tickers=", ".join(removed))
                st.session_state.pop(f"{PREFIX}remove", None)
                st.rerun()
    with st.expander(words.STATUS_GUIDE_TITLE):
        bullets(f"**{name}.** {meaning}" for name, meaning in words.STATUS_GUIDE)

    section(words.DEMO_TITLE, words.DEMO_SCOPE)
    ui.pills([("WARNING", words.SYNTHETIC_BADGE)])
    st.dataframe(
        summary_table(demo_rows(demos, band=band), band=band),
        hide_index=True,
        column_config={"Stock": st.column_config.TextColumn(pinned=True)},
        key=f"{PREFIX}demo-table",
    )


def _choices(store: StockLabStore, demos: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """Selectable stocks: id -> label. Saved stocks with prices first, then the demo stocks."""

    choices = {ticker: ticker for ticker in store.watched() if store.batches(ticker)}
    choices.update({stock_id: demo_spec(stock_id).name for stock_id in demos})
    return choices


def _detail(store: StockLabStore, demos: Mapping[str, pd.DataFrame]) -> None:
    section(words.DETAIL_TITLE)
    choices = _choices(store, demos)
    if not choices:
        say(words.DETAIL_EMPTY)
        return
    if st.session_state.get(f"{PREFIX}stock") not in choices:
        st.session_state.pop(f"{PREFIX}stock", None)  # a removed stock is no longer offered
    left, middle, right = st.columns([2, 2, 2])
    stock_id = left.selectbox(
        words.CHOOSE_LABEL,
        list(choices),
        format_func=choices.get,
        help=words.CHOOSE_HELP,
        key=f"{PREFIX}stock",
    )
    tau = middle.select_slider(
        words.BAND_LABEL,
        options=BANDS,
        value=persist.value(SAVED, f"{PREFIX}band", DEFAULT_STABLE_BAND),
        format_func=_band_words,
        help=words.BAND_HELP,
        key=f"{PREFIX}band",
    )
    policy = right.radio(
        words.RULE_LABEL,
        list(words.RULE_OPTIONS),
        format_func=words.RULE_OPTIONS.get,
        help=words.RULE_HELP,
        key=f"{PREFIX}rule",
    )
    band, rule = StableBand(tau), RunRule(policy)
    limitations: list[str] = []
    synthetic_data = is_demo(stock_id)
    if synthetic_data:
        bars = demos[stock_id]
        ui.pills([("WARNING", words.SYNTHETIC_BADGE), ("NOTICE", words.DESCRIPTIVE_BADGE)])
    else:
        bars = real_bars_shown(store.bars(stock_id))
        limitations.append(words.REAL_CUTOFF)
        ui.pills([("NOTICE", words.DESCRIPTIVE_BADGE)])
    if bars is None or bars.empty:
        say(words.DETAIL_EMPTY)
        return
    detail = stock_detail(
        stock_id,
        bars,
        name=choices[stock_id],
        synthetic_data=synthetic_data,
        band=band,
        rule=rule,
        limitations=limitations,
    )
    cards = headline(detail)
    ui.stat_row([ui.Stat(c.label, c.value, caption=c.caption, help=c.help) for c in cards])
    say(words.SHOWN_THROUGH.format(day=detail.shown_through) + " " + rule.words(), "caption")
    ui.chart(
        stock_figures.price_figure(detail.frame, synthetic=synthetic_data),
        key=f"{PREFIX}price-chart",
    )
    with st.expander(words.AS_TABLE.format(chart=words.PRICE_TITLE)):
        st.table(price_table(detail))
    ui.chart(
        stock_figures.recent_figure(recent_sessions(detail), band, synthetic=synthetic_data),
        key=f"{PREFIX}recent-chart",
    )
    with st.expander(words.AS_TABLE.format(chart=words.RECENT_TABLE_TITLE)):
        st.table(recent_table(detail))

    section(words.BAND_MEANING_TITLE)
    typical = next(c for c in cards if c.label == TYPICAL_LABEL)
    stable = next(c for c in cards if c.label.startswith("Stable sessions"))
    say(words.BAND_MEANING.format(typical=typical.value, band=band.words(), stable=stable.value))
    st.table(threshold_table(detail).set_index("Stable band"))

    section(words.RUNS_TITLE, words.RUNS_CAPTION)
    st.table(run_length_table(detail).set_index("Direction"))

    with st.expander(words.FEATURES_TITLE):
        say(words.FEATURES_CAPTION, "caption")
        st.table(feature_table(detail).set_index("Measure"))
    with st.expander(words.METHOD_TITLE):
        say(words.METHOD)
    if detail.limitations:
        section(words.LIMITATIONS_TITLE)
        bullets(detail.limitations)


def _diagnostics() -> None:
    section(words.DIAGNOSTICS_TITLE)
    with ui.state_card("stock-lab-diagnostics"):
        ui.pill("FUTURE", words.DIAGNOSTICS_BADGE)
        say(f"**{words.DIAGNOSTICS_EMPTY_TITLE}**")
        say(words.DIAGNOSTICS_EMPTY)
        say(words.DIAGNOSTICS_EXPECTATIONS, "caption")


def render(
    open_store: Callable[[], StockLabStore],
    demo_histories: Callable[[], Mapping[str, pd.DataFrame]],
) -> None:
    persist.restore(SAVED, skip=(f"{PREFIX}band",))
    ui.hero(words.TITLE, eyebrow=words.EYEBROW, lede=words.LEDE, key="stock-lab")
    ui.pills([("WARNING", words.EXPERIMENTAL_BADGE)])
    say(words.NOT_YET, "notice")
    store = open_store()
    demos = demo_histories()
    watchlist_tab, detail_tab, diagnostics_tab = st.tabs(list(words.TABS))
    with watchlist_tab:
        _watchlist(store, demos, StableBand(DEFAULT_STABLE_BAND))
    with detail_tab:
        _detail(store, demos)
    with diagnostics_tab:
        _diagnostics()
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
    persist.save(SAVED, KEPT)
