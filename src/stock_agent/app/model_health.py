"""
Model Health: a destination whose content the Model Health Center supplies.
Until it does, the page says plainly that nothing is collected yet, and
never estimates or fills in a check.

Integration (one call): pass `render` a loader that returns a
`HealthReport`, or None while nothing is collected. The page lays out
whatever the report states, in its order, with every status in words; it
computes nothing. A report is a contract: construction refuses an unknown
status level or a check without a name, word or summary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from stock_agent.app import wording
from stock_agent.app.prose import say, section
from stock_agent.ui import components as ui
from stock_agent.ui import theme


@dataclass(frozen=True)
class HealthCheck:
    """
    One collected check. `level` is a status level of the design system
    (VALID, WARNING, INVALID/BLOCKED, NOTICE or FUTURE); `word` is that status
    in words ("Fresh", "Drift", "Not checked"); `summary` is one plain
    sentence; `detail` and `measured_at` are the exact measurement and when
    it was taken, shown in an expander.
    """

    name: str
    level: str
    word: str
    summary: str
    detail: str = ""
    measured_at: str = ""

    def __post_init__(self) -> None:
        if self.level not in theme.STATUS:
            raise ValueError(f"level must be one of {tuple(theme.STATUS)}, got {self.level!r}")
        if not (self.name and self.word and self.summary):
            raise ValueError("a check needs a name, a status word and a summary")


@dataclass(frozen=True)
class HealthReport:
    """What the Model Health Center collected, when, and from where."""

    collected_at: str
    source: str
    checks: tuple[HealthCheck, ...]

    def __post_init__(self) -> None:
        if not (self.collected_at and self.source):
            raise ValueError("a report needs its collection time and source")
        if not isinstance(self.checks, tuple):
            raise TypeError("checks must be a tuple of HealthCheck")


def render(load_report: Callable[[], HealthReport | None] | None = None) -> None:
    ui.hero(
        wording.HEALTH_TITLE,
        eyebrow=wording.HEALTH_EYEBROW,
        lede=wording.HEALTH_LEDE,
        key="health",
    )
    report = None if load_report is None else load_report()
    if report is None or not report.checks:
        with ui.state_card("health-empty"):
            ui.pill("FUTURE", wording.HEALTH_EMPTY_BADGE)
            say(f"**{wording.HEALTH_EMPTY_TITLE}**")
            say(wording.HEALTH_EMPTY)
            say(wording.HEALTH_LATER, "caption")
    else:
        say(
            wording.HEALTH_COLLECTED.format(collected_at=report.collected_at, source=report.source),
            "caption",
        )
        with ui.card("health-checks"):
            ui.status_list(
                [
                    ui.Item(check.level, check.word, check.name, check.summary)
                    for check in report.checks
                ]
            )
        with st.expander(wording.DETAILS):
            rows = [
                {
                    "Check": check.name,
                    "Status": check.word,
                    "Measured": check.measured_at or "—",
                    "Detail": check.detail or "—",
                }
                for check in report.checks
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)
