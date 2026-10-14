"""
Evidence: where every number in the lab comes from, what it assumes, and
why none of it is ready for investment decisions. Built from metadata that
exists: each fund history's source, coverage, vintage and limitations, the
synthetic generator's version and seed, the approved calculations, and the
installed software versions. It reads no file and computes no metric.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from importlib import metadata

import pandas as pd
import streamlit as st

from stock_agent.app import labels, status, synthetic, view, wording
from stock_agent.app.history import FundHistory
from stock_agent.app.prose import bullets, say, section
from stock_agent.ui import components as ui

PACKAGES = ("streamlit", "plotly", "pandas", "numpy")


def software_versions() -> pd.DataFrame:
    rows = [{"Software": "Python", "Version": platform.python_version()}]
    for package in PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"Software": package, "Version": version})
    return pd.DataFrame(rows)


def _reasons(current: status.DataStatus) -> list[ui.Item]:
    found = [ui.Item(*reason) for reason in wording.NOT_READY]
    if current.synthetic:
        found.insert(0, ui.Item(*wording.SYNTHETIC_REASON))
    return found


def render(load_histories: Callable[[], Sequence[FundHistory]]) -> None:
    ui.hero(
        wording.EVIDENCE_TITLE,
        eyebrow=wording.EVIDENCE_EYEBROW,
        lede=wording.EVIDENCE_LEDE,
        key="evidence",
    )
    histories = tuple(load_histories())
    current = status.data_status(histories)
    badges = [("NOTICE", labels.EVIDENCE_BADGE)]
    if current.synthetic:
        badges.append(("WARNING", labels.SYNTHETIC_BADGE))
    ui.pills(badges)
    if current.synthetic:
        say(labels.SYNTHETIC_NOTICE, "warning")

    with ui.card("evidence-not-ready"):
        section(wording.NOT_READY_TITLE)
        ui.status_list(_reasons(current))

    section(wording.DATA_TITLE, wording.DATA_SCOPE)
    st.dataframe(view.sources_table(histories), hide_index=True)
    codes = sorted({code for history in histories for code in history.limitations})
    if codes:
        say("**Limitations**")
        bullets(labels.LIMITATION_TEXT.get(code, code) for code in codes)

    left, right = st.columns(2, gap="large")
    with left, ui.card("evidence-calculations"):
        section(wording.CALCULATIONS_TITLE)
        bullets(wording.CALCULATIONS)
        say(wording.CALCULATIONS_SOURCE, "caption")
        say(wording.BUILDER_STEP, "caption")
    with right, ui.card("evidence-models"):
        section(wording.MODELS_TITLE)
        say(wording.MODELS)
        section(wording.PUBLICATION_TITLE)
        if current.all_synthetic:
            say(
                wording.PUBLICATION.format(
                    version=synthetic.GENERATOR_VERSION,
                    seed=synthetic.SEED,
                    start=synthetic.CALENDAR_START,
                    end=synthetic.CALENDAR_END,
                )
            )
            say(wording.SEALED.format(end=synthetic.CALENDAR_END), "caption")
        else:
            say("Each fund's source, retrieval and vintage are listed under Data versions.")

    section(wording.ASSUMPTIONS_TITLE)
    bullets(
        (
            *labels.HONEST_LABELS,
            labels.FRACTIONAL_SHARES,
            labels.DAILY_CLOSES,
        )
    )

    with st.expander(wording.SOFTWARE_TITLE):
        say(wording.SOFTWARE_SCOPE, "caption")
        st.dataframe(software_versions(), hide_index=True)

    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
