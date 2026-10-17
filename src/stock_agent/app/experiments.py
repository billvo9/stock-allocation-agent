"""
Experiments: the prospective forecasts recorded so far, the registered
synthetic strategy comparison, and an honest empty state for everything else.

"Prospective forecasts (unvalidated)" lays out what the private forecast
ledger records (forecast_records.py): each forecast's point, its 80% band,
the historical average beside it, its as-of session and labels, and its
outcome only once scored. It computes nothing. Without a ledger it says "No
forecast recorded yet"; a ledger that fails its integrity check shows
nothing but that fact.

"Synthetic strategy comparison (engineering evaluation)" lays out the newest
artifact of the registered comparison protocol (comparison.py): its labels,
facts, results per constraint set and scenario, every learned seed, the
build, and each strategy's registry stage. Without one it says so; a file
outside the protocol is withheld with its reason. The page also points to
the separate research diagnostics dashboard, which reads saved
model-diagnostics runs.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from stock_agent.app import comparison, forecast_records, wording
from stock_agent.app.comparison import Comparisons
from stock_agent.app.forecast_records import ForecastRecords, LedgerUnreadable
from stock_agent.app.prose import say, section
from stock_agent.ui import components as ui


def _empty_experiments() -> None:
    with ui.state_card("experiments-empty"):
        ui.pill("FUTURE", wording.EXPERIMENTS_EMPTY_BADGE)
        say(f"**{wording.EXPERIMENTS_EMPTY_TITLE}**")
        say(wording.EXPERIMENTS_EMPTY)
        say(wording.EXPERIMENTS_LATER, "caption")


def _forecast(record, records: ForecastRecords) -> None:
    with ui.card(f"prospective-{record['forecast_id']}"):
        badges = [("WARNING", wording.PROSPECTIVE_BADGE)]
        if record["data"]["data_class"] == "synthetic":
            badges.append(("WARNING", wording.PROSPECTIVE_SYNTHETIC))
        ui.pills(badges)
        lag = record["rows"][0]["entry_lag"] if record["rows"] else 1
        say(
            wording.PROSPECTIVE_RECORDED.format(
                as_of=record["as_of_session"], created_at=record["created_at"], lag=lag
            ),
            "caption",
        )
        say("; ".join(record["labels"]))
        refused = [refusal["asset"] for refusal in record.get("refusals", ())]
        if refused:
            say(wording.PROSPECTIVE_REFUSED.format(assets=", ".join(refused)), "caption")
        st.dataframe(
            forecast_records.forecast_table(record, records.scores),
            hide_index=True,
            key=f"prospective-table-{record['forecast_id']}",
        )


def _prospective(load_records: Callable[[], ForecastRecords | None] | None) -> bool:
    """The panel; True when it showed at least one forecast."""

    section(wording.PROSPECTIVE_TITLE)
    say(wording.PROSPECTIVE_INTRO)
    try:
        records = None if load_records is None else load_records()
    except LedgerUnreadable as error:
        say(wording.PROSPECTIVE_BROKEN.format(reason=error), "warning")
        return False
    if records is None:
        with ui.state_card("prospective-empty"):
            ui.pill("FUTURE", wording.PROSPECTIVE_EMPTY_BADGE)
            say(f"**{wording.PROSPECTIVE_EMPTY_TITLE}**")
            say(wording.PROSPECTIVE_EMPTY)
            say(wording.PROSPECTIVE_LATER, "caption")
        return False
    say(wording.PROSPECTIVE_UNITS, "caption")
    say(wording.PROSPECTIVE_BAND, "caption")
    for record in records.forecasts:
        _forecast(record, records)
    section(wording.PROSPECTIVE_SCORES_TITLE)
    if records.cumulative:
        st.dataframe(
            forecast_records.cumulative_table(records.cumulative),
            hide_index=True,
            key="prospective-cumulative",
        )
        for caveat in dict.fromkeys(row["caveat"] for row in records.cumulative):
            say(caveat, "caption")
    else:
        say(wording.PROSPECTIVE_NO_SCORES)
    return True


def _empty_comparison() -> None:
    with ui.state_card("comparison-empty"):
        ui.pill("FUTURE", wording.COMPARISON_EMPTY_BADGE)
        say(f"**{wording.COMPARISON_EMPTY_TITLE}**")
        say(wording.COMPARISON_EMPTY)


def _comparison(load: Callable[[], Comparisons | None] | None) -> bool:
    """The registered synthetic comparison; True when one was shown."""

    section(wording.COMPARISON_TITLE)
    found = None if load is None else load()
    if found is None:
        _empty_comparison()
        return False
    for name, reason in found.withheld:
        say(wording.COMPARISON_WITHHELD.format(file=name, reason=reason), "warning")
    document = found.shown
    if document is None:
        return False
    with ui.card("comparison"):
        ui.pills(
            [
                ("WARNING", wording.COMPARISON_SCOPE_BADGE),
                ("WARNING", wording.COMPARISON_EVALUATION_BADGE),
            ]
        )
        say(wording.COMPARISON_INTRO)
        say(wording.COMPARISON_NO_CLAIM)
        for fact in comparison.facts(document, found.looks):
            say(fact, "caption")
        say(wording.COMPARISON_NOT_REPORTED, "caption")
    sets = document["constraint_sets"]
    names = [comparison.constraint_set_label(s) for s in sets]
    chosen = st.selectbox(wording.COMPARISON_SET_LABEL, names, key="comparison-set")
    constraint_set = sets[names.index(chosen)]["name"]
    scenario = st.selectbox(
        wording.COMPARISON_SCENARIO_LABEL,
        comparison.scenarios_of(document),
        key="comparison-scenario",
    )
    st.dataframe(
        comparison.results_table(document, constraint_set, scenario),
        hide_index=True,
        key="comparison-results",
    )
    for note in (
        wording.COMPARISON_UNITS,
        wording.COMPARISON_DRAWDOWN_NOTE,
        wording.COMPARISON_TURNOVER_NOTE,
        wording.COMPARISON_COSTS_NOTE,
    ):
        say(note, "caption")
    section(wording.COMPARISON_SEEDS_TITLE)
    say(wording.COMPARISON_SEEDS_NOTE)
    st.dataframe(
        comparison.seed_table(document, constraint_set, scenario),
        hide_index=True,
        key="comparison-seeds",
    )
    section(wording.COMPARISON_BUILD_TITLE)
    say(wording.COMPARISON_BUILD_NOTE)
    st.dataframe(
        comparison.build_table(document, constraint_set), hide_index=True, key="comparison-build"
    )
    section(wording.COMPARISON_STRATEGIES_TITLE)
    st.dataframe(comparison.strategy_table(document), hide_index=True, key="comparison-strategies")
    return True


def render(
    load_records: Callable[[], ForecastRecords | None] | None = None,
    load_comparisons: Callable[[], Comparisons | None] | None = None,
) -> None:
    ui.hero(
        wording.EXPERIMENTS_TITLE,
        eyebrow=wording.EXPERIMENTS_EYEBROW,
        lede=wording.EXPERIMENTS_LEDE,
        key="experiments",
    )
    forecasts = _prospective(load_records)
    compared = _comparison(load_comparisons)
    if not (forecasts or compared):
        _empty_experiments()
    with ui.card("experiments-dashboard"):
        section(wording.DASHBOARD_TITLE)
        say(wording.DASHBOARD_TEXT)
        st.code(wording.DASHBOARD_COMMAND, language="bash")
        st.link_button(wording.DASHBOARD_LINK, wording.DASHBOARD_URL)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
