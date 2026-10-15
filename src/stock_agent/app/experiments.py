"""
Experiments: the prospective forecasts recorded so far, and an honest empty
state for everything else.

"Prospective forecasts (unvalidated)" lays out what the private forecast
ledger records (forecast_records.py): each forecast's point, its 80% band,
the historical average beside it, its as-of session and labels, and its
outcome only once scored. It computes nothing. Without a ledger it says "No
forecast recorded yet"; a ledger that fails its integrity check shows
nothing but that fact. The page also points to the separate research
diagnostics dashboard, which reads saved model-diagnostics runs.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from stock_agent.app import forecast_records, wording
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


def render(load_records: Callable[[], ForecastRecords | None] | None = None) -> None:
    ui.hero(
        wording.EXPERIMENTS_TITLE,
        eyebrow=wording.EXPERIMENTS_EYEBROW,
        lede=wording.EXPERIMENTS_LEDE,
        key="experiments",
    )
    if not _prospective(load_records):
        _empty_experiments()
    with ui.card("experiments-dashboard"):
        section(wording.DASHBOARD_TITLE)
        say(wording.DASHBOARD_TEXT)
        st.code(wording.DASHBOARD_COMMAND, language="bash")
        st.link_button(wording.DASHBOARD_LINK, wording.DASHBOARD_URL)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
