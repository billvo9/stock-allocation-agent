"""
Model Health's Stock Lab checks as the Lab app reads them, and the Health tab's
badge (docs/product/lab_app_design.md, "Health in detail").

The checks come from Model Health's latest local summary, read with the Model
Health page's own reader (`app.health_summary`): only the Stock Lab section, and
only from a summary that was read and was written by the request time. A check
is flagged when Model Health marks it a warning or worse.

The badge (owner decision of 2026-10-26, "Keep the green headline with the
badge"): amber "!" or red "✕" when Home's status line is amber or red, as before;
otherwise "!" whenever Model Health flags a Stock Lab check, even under a green
headline; otherwise none. Home and Health both use `tab_badge`, so the tab and
the screen never disagree.
"""

from __future__ import annotations

from datetime import datetime

from stock_agent.app.health_summary import Loaded
from stock_agent.app.web import words

STOCK_LAB_SECTION = "stock_lab"
# Model Health's levels (health.view.CONDITION_LEVELS) as the page's marks.
MARKS = {
    "VALID": "ok",
    "NOTICE": "note",
    "WARNING": "warn",
    "UNAVAILABLE": "warn",
    "INVALID/BLOCKED": "bad",
}
FLAGGED = ("warn", "bad")
BADGE_MARKS = {"warn": "!", "bad": "✕"}


def usable(loaded: Loaded | None, now: datetime) -> bool:
    """Whether the summary was read and was written by `now` (a later one wasn't known then)."""

    return (
        loaded is not None
        and loaded.view is not None
        and loaded.generated_at is not None
        and loaded.generated_at <= now
    )


def stock_lab_checks(loaded: Loaded | None, now: datetime) -> list[dict]:
    """The summary's Stock Lab checks, worst first; none when the summary isn't usable."""

    if not usable(loaded, now):
        return []
    return [
        {
            "check_id": check.check_id,
            "title": check.title,
            "mark": MARKS.get(check.level, "warn"),
            "state_word": check.state_word,
            "words": check.summary,
        }
        for section in loaded.sections
        if section.key == STOCK_LAB_SECTION
        for check in section.checks
    ]


def flagged(items: list[dict]) -> int:
    """How many of these checks Model Health marks a warning or worse."""

    return sum(item["mark"] in FLAGGED for item in items)


def tab_badge(level: str, flags: int) -> dict | None:
    """The Health tab's badge for Home's status level and the flagged checks (module docstring)."""

    if level in BADGE_MARKS:
        return {"level": level, "mark": BADGE_MARKS[level], "words": words.HEALTH_BADGES[level]}
    if flags:
        return {"level": "warn", "mark": BADGE_MARKS["warn"], "words": words.flagged_badge(flags)}
    return None


__all__ = ["flagged", "stock_lab_checks", "tab_badge", "usable"]
