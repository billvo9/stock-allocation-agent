"""
Experiments: an honest empty state until machine-learning or reinforcement-
learning runs produce artifacts this lab can show. It reads nothing and
estimates nothing; it points to the separate research diagnostics dashboard,
which reads saved model-diagnostics runs.
"""

from __future__ import annotations

import streamlit as st

from stock_agent.app import wording
from stock_agent.app.prose import say, section
from stock_agent.ui import components as ui


def render() -> None:
    ui.hero(
        wording.EXPERIMENTS_TITLE,
        eyebrow=wording.EXPERIMENTS_EYEBROW,
        lede=wording.EXPERIMENTS_LEDE,
        key="experiments",
    )
    with ui.state_card("experiments-empty"):
        ui.pill("FUTURE", wording.EXPERIMENTS_EMPTY_BADGE)
        say(f"**{wording.EXPERIMENTS_EMPTY_TITLE}**")
        say(wording.EXPERIMENTS_EMPTY)
        say(wording.EXPERIMENTS_LATER, "caption")
    with ui.card("experiments-dashboard"):
        section(wording.DASHBOARD_TITLE)
        say(wording.DASHBOARD_TEXT)
        st.code(wording.DASHBOARD_COMMAND, language="bash")
        st.link_button(wording.DASHBOARD_LINK, wording.DASHBOARD_URL)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
