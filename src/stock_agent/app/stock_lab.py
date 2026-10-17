"""
The Stock Probability page (Streamlit): the saved watchlist, one stock's
history, its probabilities for the next five sessions where a stored forecast
exists, its reversals and failed breakouts, and the model diagnostics.

The watchlist is stored in the private SQLite store (`stock_lab.store`), so
it survives restarts; the page writes only watchlist events there. Prices
are read from the same store (written by the owner-run fetch script, which
the page never imports) or come from the synthetic demo generator, always
labelled. Descriptive numbers are computed by `stock_lab.view` and
`stock_lab.forecast_view` from stored history; probabilities and test
results come only from stored, checked records (the fitted models are never
imported here); the page lays them out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from stock_agent.app import persist, stock_figures, wording
from stock_agent.app import stock_wording as words
from stock_agent.app.prose import bullets, say, section
from stock_agent.stock_lab.contracts import (
    DEFAULT_STABLE_BAND,
    DEVELOPMENT_END,
    DataClass,
    StableBand,
)
from stock_agent.stock_lab.features import build_features, latest_row
from stock_agent.stock_lab.forecast_view import (
    band_table,
    calibration_status,
    comparison_table,
    cumulative_cards,
    current_streak_sentence,
    duration_summary,
    five_session_frame,
    five_session_table,
    fold_table,
    history_table,
    model_words,
    move_table,
    next_session_cards,
    plain_sentence,
    regime_table,
    reliability_frame,
    scored_summary,
    scores_table,
    selection_sentence,
    size_design_words,
    size_table,
    staleness,
    streak_profile,
    streak_table,
    trap_view,
)
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
KEPT = (
    f"{PREFIX}search",
    f"{PREFIX}stock",
    f"{PREFIX}band",
    f"{PREFIX}rule",
    f"{PREFIX}rev-direction",
    f"{PREFIX}rev-k",
    f"{PREFIX}move-x",
    f"{PREFIX}move-sign",
)
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


@dataclass(frozen=True, eq=False)
class Selected:
    """The stock chosen above the tabs, its shown bars and the controls' values."""

    stock_id: str
    name: str
    synthetic: bool
    bars: pd.DataFrame | None
    band: StableBand
    rule: RunRule
    outcome_end: pd.Timestamp | None  # real data: outcomes stop at the development end


def _controls(store: StockLabStore, demos: Mapping[str, pd.DataFrame]) -> Selected | None:
    choices = _choices(store, demos)
    if not choices:
        return None
    if st.session_state.get(f"{PREFIX}stock") not in choices:
        st.session_state.pop(f"{PREFIX}stock", None)  # a removed stock is no longer offered
    with ui.card("stock-lab-controls"):
        say(f"**{words.CONTROLS_TITLE}**")
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
    synthetic_data = is_demo(stock_id)
    if synthetic_data:
        bars, outcome_end = demos[stock_id], None
    else:
        bars = real_bars_shown(store.bars(stock_id))
        outcome_end = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    return Selected(
        stock_id,
        choices[stock_id],
        synthetic_data,
        bars,
        StableBand(tau),
        RunRule(policy),
        outcome_end,
    )


def _badges(selected: Selected, *extra: tuple[str, str]) -> None:
    badges = [("WARNING", words.SYNTHETIC_BADGE)] if selected.synthetic else []
    ui.pills([*badges, *extra])


def _detail(selected: Selected | None) -> None:
    section(words.DETAIL_TITLE)
    if selected is None or selected.bars is None or selected.bars.empty:
        say(words.DETAIL_EMPTY)
        return
    band, rule = selected.band, selected.rule
    _badges(selected, ("NOTICE", words.DESCRIPTIVE_BADGE))
    limitations = [] if selected.synthetic else [words.REAL_CUTOFF]
    detail = stock_detail(
        selected.stock_id,
        selected.bars,
        name=selected.name,
        synthetic_data=selected.synthetic,
        band=band,
        rule=rule,
        limitations=limitations,
    )
    synthetic_data = selected.synthetic
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


def _cards(cards) -> None:
    ui.stat_row(
        [
            ui.Stat(
                card.label,
                card.value,
                caption=" · ".join(filter(None, (card.status, card.caption))),
            )
            for card in cards
        ]
    )


def _forecast(store: StockLabStore, selected: Selected | None) -> None:
    section(words.FORECAST_TITLE)
    if selected is None:
        say(words.DETAIL_EMPTY)
        return
    record = store.latest_forecast(selected.stock_id)
    if record is None:
        with ui.state_card("stock-lab-no-forecast"):
            if selected.synthetic:
                ui.pill("FUTURE", words.FORECAST_NONE_DEMO_TITLE)
                say(words.FORECAST_NONE_DEMO)
                st.code(words.DEMO_COMMAND, language="bash")
            else:
                ui.pill("FUTURE", words.FORECAST_NONE_REAL_TITLE)
                say(words.FORECAST_NONE_REAL)
        return
    evaluation = store.evaluation(record["evaluation_run_id"])  # the one it was made with
    _badges(
        selected,
        ("WARNING", words.FORECAST_BADGE),
        ("NOTICE", words.DEVELOPMENT_BADGE),
    )
    if record["refusal"] is not None:
        say(words.FORECAST_REFUSED.format(detail=record["refusal"]["detail"]), "warning")
        return
    stale = staleness(record, store.latest_job_run())
    if stale:
        say(stale, "warning")
    record_band = StableBand(record["band"])
    say(words.FORECAST_BAND.format(band=record_band.words()), "caption")
    say(plain_sentence(record))
    section(words.NEXT_TITLE)
    _cards(next_session_cards(record, evaluation))
    say(words.DAILY_NOTE, "caption")

    section(words.FIVE_TITLE)
    frame = five_session_frame(record, evaluation)
    columns = st.columns(3)
    for column, category in zip(columns, ("increase", "stable", "decrease"), strict=True):
        with column:
            ui.chart(
                stock_figures.five_session_figure(frame, category, synthetic=selected.synthetic),
                key=f"{PREFIX}five-{category}",
            )
    with st.expander(words.AS_TABLE.format(chart=words.FIVE_TABLE)):
        st.table(five_session_table(record))
    say(words.TRACK_NOTE, "caption")

    totals = cumulative_cards(record)
    if totals is not None:
        section(words.CUMULATIVE_TITLE)
        _cards(totals)
        say(
            words.CUMULATIVE_NOTE.format(band=StableBand(record["cumulative_band"]).words()),
            "caption",
        )
    say(
        words.RECORDED.format(
            as_of=record["as_of_session"],
            created=record["created_at"][:16].replace("T", " ") + " UTC",
            estimator=model_words(record["estimator"]),
            version=record["model_version"],
            features=record["feature_version"],
            evaluation=record["evaluation_run_id"],
        ),
        "caption",
    )
    history = store.forecasts(selected.stock_id)
    earlier = len(history) - 1
    if earlier:
        say(words.SUPERSEDED.format(count=earlier), "caption")
    bullets(record["limitations"])
    _record_history(store, selected, history)


def _record_history(store: StockLabStore, selected: Selected, history) -> None:
    section(words.HISTORY_TITLE)
    job = store.latest_job_run()
    if job is not None and job.get("data_class") == history[-1]["data_class"]:
        say(
            words.JOB_STATUS.format(
                status=job["status"],
                finished=job["finished_at"][:16].replace("T", " ") + " UTC",
                expected=job["expected_session"],
            ),
            "caption",
        )
    scored = store.scores(selected.stock_id)
    summary = scored_summary(scored)
    say(
        words.HISTORY_NOTE.format(
            summary=words.HISTORY_NONE
            if summary is None
            else words.HISTORY_SUMMARY.format(**summary)
        )
    )
    table = history_table(history, scored)
    if len(table):
        st.table(table)
    if selected.synthetic and summary is None:
        say(words.REPLAY_NOTE, "caption")
        st.code(words.REPLAY_COMMAND, language="bash")


def _reversals(selected: Selected | None) -> None:
    section(words.REVERSALS_TITLE)
    if selected is None or selected.bars is None or selected.bars.empty:
        say(words.DETAIL_EMPTY)
        return
    _badges(selected, ("NOTICE", words.DESCRIPTIVE_BADGE))
    say(words.REVERSALS_LEDE)
    frame = build_features(selected.bars, band=selected.band, rule=selected.rule)
    last = latest_row(frame)
    current = last["run_direction"] if last is not None else None
    length = int(last["run_length"]) if last is not None else 0
    left, right = st.columns(2)
    direction = left.radio(
        words.DIRECTION_LABEL,
        list(words.DIRECTION_OPTIONS),
        format_func=words.DIRECTION_OPTIONS.get,
        index=1 if current == "increase" else 0,
        horizontal=True,
        key=f"{PREFIX}rev-direction",
    )
    default_k = min(max(length, 1), 7) if current == direction else 3
    k = right.slider(
        words.LENGTH_LABEL, 1, 7, default_k, help=words.LENGTH_HELP, key=f"{PREFIX}rev-k"
    )
    profile = streak_profile(frame, direction, selected.band, outcome_end=selected.outcome_end)
    sentence = current_streak_sentence(profile, direction, k)
    if sentence:
        say(sentence)
    marked = length if current == direction else None
    ui.chart(
        stock_figures.streak_figure(profile, direction, marked, synthetic=selected.synthetic),
        key=f"{PREFIX}streak-chart",
    )
    with st.expander(words.AS_TABLE.format(chart=words.STREAK_TABLE)):
        st.table(streak_table(profile, direction))
    section(words.REGIME_TITLE, words.REGIME_NOTE)
    st.table(regime_table(frame, k, direction, selected.band))
    durations = duration_summary(frame, k, direction, selected.rule)
    with st.expander(words.DURATION_TITLE):
        median = durations["median"]
        say(
            words.DURATION_NOTE.format(
                median=words.INSUFFICIENT if median is None else f"{median:.0f} session(s)",
                runs=durations["runs"],
                censored=durations["censored"],
            )
        )
        if len(durations["table"]):
            st.table(durations["table"])

    section(words.MOVE_TITLE)
    left, right = st.columns(2)
    threshold = left.select_slider(
        words.MOVE_LABEL,
        options=(0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08),
        value=0.02,
        format_func=lambda x: f"{x:.1%}".replace(".0%", "%"),
        key=f"{PREFIX}move-x",
    )
    sign = right.radio(
        words.MOVE_SIGN_LABEL,
        list(words.MOVE_SIGN),
        format_func=words.MOVE_SIGN.get,
        horizontal=True,
        key=f"{PREFIX}move-sign",
    )
    definition, table, events, pending = move_table(
        frame, threshold, int(sign), selected.band, outcome_end=selected.outcome_end
    )
    st.table(table)
    say(
        words.MOVE_NOTE.format(definition=definition.capitalize(), events=events, pending=pending),
        "caption",
    )


def _traps(selected: Selected | None) -> None:
    section(words.TRAPS_TITLE)
    if selected is None or selected.bars is None or selected.bars.empty:
        say(words.DETAIL_EMPTY)
        return
    _badges(selected, ("NOTICE", words.DESCRIPTIVE_BADGE))
    say(words.TRAPS_LEDE)
    frame = build_features(selected.bars, band=selected.band, rule=selected.rule)
    bull = trap_view(frame, outcome_end=selected.outcome_end)
    rule = bull.rule
    values = {
        "decline": f"{rule.decline:.0%}",
        "high": rule.high_window,
        "rebound": f"{rule.rebound:.0%}",
        "window": rule.breakout_window,
        "failure": rule.failure_window,
        "margin": f"{rule.failure_margin:.0%}",
    }
    ui.status_list(
        [
            ui.Item(level, word, title, text.format(**values))
            for level, word, title, text in words.TRAP_STEPS
        ]
    )
    if bull.latest is None:
        say(words.TRAP_NONE)
    else:
        say(words.TRAP_LATEST.format(session=bull.latest["session"], status=bull.latest["status"]))
        if bull.latest["status_code"] == "unresolved":
            say(words.TRAP_NOW.format(failure=rule.failure_window), "notice")
    say(
        words.TRAP_RATES.format(
            rate=bull.failure_rate,
            n=bull.setups_resolved,
            base=bull.base_rate,
            lift=f"{bull.lift} as often" if bull.lift != "—" else "no comparison",
        )
    )
    st.table(bull.by_regime)
    with st.expander(words.TRAP_DETAILS):
        say(
            words.TRAP_DETAILS_TEXT.format(
                precision=bull.precision, recall=bull.recall, false_alarm=bull.false_alarm_rate
            )
        )
    bear = trap_view(frame, kind="bear", outcome_end=selected.outcome_end)
    with st.expander(words.BEAR_TITLE):
        say(
            words.BEAR_TEXT.format(
                rate=bear.failure_rate, n=bear.setups_resolved, base=bear.base_rate
            )
        )
        st.table(bear.by_regime)


def _diagnostics(store: StockLabStore, selected: Selected | None) -> None:
    section(words.DIAGNOSTICS_TITLE)
    data_class = DataClass.SYNTHETIC if selected is None or selected.synthetic else DataClass.REAL
    evaluation = store.latest_evaluation(data_class)
    if evaluation is None:
        with ui.state_card("stock-lab-diagnostics"):
            ui.pill("FUTURE", words.DIAGNOSTICS_BADGE)
            say(f"**{words.DIAGNOSTICS_EMPTY_TITLE}**")
            say(words.DIAGNOSTICS_NONE)
            st.code(words.DEMO_COMMAND, language="bash")
            say(words.DIAGNOSTICS_EXPECTATIONS, "caption")
        return
    synthetic_data = evaluation["data_class"] == DataClass.SYNTHETIC.value
    if synthetic_data:
        ui.pills([("WARNING", words.SYNTHETIC_BADGE), ("WARNING", words.FORECAST_BADGE)])
        say(words.DIAGNOSTICS_SYNTHETIC, "notice")
    say(selection_sentence(evaluation))
    say(words.DIAGNOSTICS_EXPECTATIONS, "caption")
    section(words.SCORES_TITLE, words.SCORES_NOTE)
    st.table(scores_table(evaluation))
    section(words.COMPARISON_TITLE, words.COMPARISON_NOTE)
    comparisons = comparison_table(evaluation)
    if len(comparisons):
        st.table(comparisons)
    chosen = evaluation["selection"]["selected"]
    frame = reliability_frame(evaluation, [chosen, "B1"])
    for code, category in ((2, "Increase"), (1, "Stable"), (0, "Decrease")):
        status = calibration_status(frame, chosen, code)
        say(f"**{category}: {status}**")
        left, right = st.columns([3, 2])
        with left:
            ui.chart(
                stock_figures.reliability_figure(
                    frame, chosen, "B1", code, synthetic=synthetic_data
                ),
                key=f"{PREFIX}reliability-{code}",
            )
        with right:
            ui.chart(
                stock_figures.forecast_count_figure(frame, chosen, code, synthetic=synthetic_data),
                key=f"{PREFIX}counts-{code}",
            )
    with st.expander(words.FOLDS_TITLE):
        say(words.FOLDS_NOTE, "caption")
        st.table(fold_table(evaluation))
        say(words.CANDIDATES.format(family=", ".join(evaluation["candidate_family"])), "caption")
    design = size_design_words(evaluation)
    if design is not None:
        with st.expander(words.SIZE_TITLE):
            say(words.SIZE_NOTE.format(**design), "caption")
            for placebo, note in (
                ("other_vs_other", words.SIZE_SYMMETRIC),
                ("other_vs_own", words.SIZE_OWN),
            ):
                say(note)
                st.table(size_table(evaluation, placebo))
    bands = band_table(evaluation)
    if len(bands):
        with st.expander(words.BAND_TITLE):
            say(words.BAND_NOTE, "caption")
            st.table(bands)


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
    selected = _controls(store, demos)
    tabs = st.tabs(list(words.TABS))
    with tabs[0]:
        _watchlist(store, demos, StableBand(DEFAULT_STABLE_BAND))
    with tabs[1]:
        _forecast(store, selected)
    with tabs[2]:
        _detail(selected)
    with tabs[3]:
        _reversals(selected)
    with tabs[4]:
        _traps(selected)
    with tabs[5]:
        _diagnostics(store, selected)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
    persist.save(SAVED, KEPT)
