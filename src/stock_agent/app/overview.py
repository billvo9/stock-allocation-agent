"""
The Overview: what is happening in the lab and what needs attention, built
only from facts that exist: the loaded funds and their records
(`status`), their synthetic labels, and the pending capabilities recorded
in `wording.PENDING` with their date. Plain-language summary and next
actions first; the exact counts sit in an expander.

Layout only; every surface comes from the shared components.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import streamlit as st

from stock_agent.app import labels, status, wording
from stock_agent.app.history import FundHistory
from stock_agent.app.prose import bullets, say, section
from stock_agent.ui import components as ui


def render(
    load_histories: Callable[[], Sequence[FundHistory]],
    *,
    links: Mapping[str, object] | None = None,
) -> None:
    """`links` maps NEXT_LINKS keys to pages; without it no link is drawn."""

    ui.hero(
        wording.OVERVIEW_TITLE,
        eyebrow=wording.OVERVIEW_EYEBROW,
        lede=wording.OVERVIEW_LEDE,
        key="overview",
    )
    current = status.data_status(tuple(load_histories()))
    badges = [("NOTICE", labels.EVIDENCE_BADGE)]
    if current.synthetic:
        badges.append(("WARNING", labels.SYNTHETIC_BADGE))
    badges.append(("FUTURE", "No forecasts"))
    ui.pills(badges)

    summary, actions = st.columns([3, 2], gap="large")
    with summary, ui.card("overview-happening"):
        section(wording.HAPPENING_TITLE)
        bullets(status.happening(current))
    with actions, ui.card("overview-next"):
        section(wording.NEXT_TITLE)
        if links:
            ui.page_links(
                [(links[key], label) for key, label in wording.NEXT_LINKS.items() if key in links],
                key="overview",
            )

    with ui.card("overview-attention"):
        section(wording.ATTENTION_TITLE)
        items = status.attention(current)
        if items:
            ui.status_list([ui.Item(i.level, i.word, i.title, i.text) for i in items])
        else:
            say(wording.NOTHING_NEEDS_ATTENTION)

    with ui.card("overview-pending"):
        section(wording.PENDING_TITLE, f"As recorded on {wording.PENDING_AS_OF}.")
        ui.status_list(
            [ui.Item("FUTURE", word, title, text) for word, title, text, _ in wording.PENDING]
        )

    with st.expander(wording.DETAILS):
        st.dataframe(status.measurements(current), hide_index=True)

    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
