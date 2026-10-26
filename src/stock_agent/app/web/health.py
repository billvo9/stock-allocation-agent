"""
The Lab app's Health view model (docs/product/lab_app_design.md, "Health in
detail"): one indicator with one sentence, and the details behind it; JSON-ready,
with display-ready words.

Where each part comes from
    The headline is Home's status line (`home.status_line`), prepared the same way
    at the request time, so Home and Health never disagree. Below it, the live
    record's progress toward Model Health's minimum (the briefing's prospective
    record, as on a stock's screen).
    The details: the evening updates (the status line's own detail lines), then
    Model Health's Stock Lab checks from its latest local summary, read with the
    Model Health page's own reader (`app.health_summary`) and shown with that
    summary's time. A missing, refused or stale summary (more than 26 hours old)
    is named as such, and a summary written after the request time is not used.
    The app runs no check and changes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from datetime import datetime, tzinfo

import pandas as pd

from stock_agent.app.health_summary import Loaded
from stock_agent.app.web import checks, home, words
from stock_agent.app.web.detail import research_page
from stock_agent.stock_lab.briefing.reliability import MIN_SCORED, MIN_SESSIONS
from stock_agent.stock_lab.briefing.report import Briefing, ReadOnlyRecords
from stock_agent.stock_lab.contracts import DataClass

SCHEMA = "lab_app_health"
VERSION = 1


def build_health(
    records: ReadOnlyRecords,
    *,
    now: datetime,
    calendar: Callable,
    calendar_id: str,
    data_class: DataClass | str,
    local_zone: tzinfo | None,
    summary: Callable[[], Loaded] | None,
    research_url: str | None,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
) -> dict:
    """
    Health at `now` (timezone-aware). `summary()` reads Model Health's latest local
    summary (None: there is none to read, as in the synthetic demo).
    """

    kind = DataClass(data_class).value
    found = home.prepare(
        records,
        now=now,
        calendar=calendar,
        calendar_id=calendar_id,
        data_class=kind,
        tickers=tickers,
        excluded=excluded,
    )
    status = home.status_line(
        found.briefing, found.problem, now, local_zone, found.previous_session
    )
    return health_view(
        status,
        found.briefing,
        None if summary is None else summary(),
        now=now,
        local_zone=local_zone,
        data_class=kind,
        research_url=research_url,
    )


def health_view(
    status: dict,
    briefing: Briefing | None,
    loaded: Loaded | None,
    *,
    now: datetime,
    local_zone: tzinfo | None,
    data_class: str,
    research_url: str | None,
) -> dict:
    """The Health view from Home's status line, the briefing and Model Health's summary."""

    level = status["level"]
    found = _checks(loaded, now, local_zone, data_class)
    flagged = checks.flagged(found["items"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "data_class": data_class,
        "generated_at": pd.Timestamp(now).tz_convert("UTC").isoformat(),
        "level": level,
        "title": words.HEALTH_TITLES[level],
        "line": status["words"],
        "aside": _aside(level, briefing),
        "flagged": words.flagged(flagged) if flagged else None,
        "badge": checks.tab_badge(level, flagged),  # the tab's, as Home sends it
        "checked": words.checked(now, local_zone),
        "toggle": list(words.SHOW_DETAILS),
        "updates": {"title": words.UPDATES_DETAILS_TITLE, "lines": status["details"]},
        "checks": found,
        "link": None
        if not research_url
        else {
            "url": research_page(research_url, words.MODEL_HEALTH_PAGE),
            "words": words.MODEL_HEALTH_LINK,
            "sub": words.MODEL_HEALTH_SUB,
        },
        "elsewhere": words.RESEARCH_ELSEWHERE,
        "demo": words.DEMO if data_class == DataClass.SYNTHETIC.value else None,
    }


def _aside(level: str, briefing: Briefing | None) -> str | None:
    if level == "bad":
        return words.HEALTH_BAD_ASIDE
    if briefing is None:
        return None
    found = briefing.reliability.prospective
    return words.live_words(
        found.status,
        found.settled_sessions,
        found.scored_targets,
        sessions_needed=MIN_SESSIONS,
        targets_needed=MIN_SCORED,
    )


def _checks(loaded: Loaded | None, now: datetime, zone: tzinfo | None, data_class: str) -> dict:
    """Model Health's Stock Lab checks from its summary, or why none are shown."""

    def note(text: str, technical: str | None = None) -> dict:
        detail = None if technical is None else words.TECHNICAL.format(detail=technical)
        return {"title": words.CHECKS_TITLE, "note": text, "technical": detail, "items": []}

    if loaded is None:
        demo = data_class == DataClass.SYNTHETIC.value
        return note(words.SUMMARY_DEMO if demo else words.SUMMARY_MISSING)
    if loaded.refused_reason is not None:
        return note(words.SUMMARY_REFUSED, loaded.refused_reason)
    if loaded.view is None or loaded.generated_at is None:
        return note(words.SUMMARY_MISSING)
    if not checks.usable(loaded, now):  # written after this view's time: not known then
        return note(words.SUMMARY_LATER)
    made = words.when(loaded.generated_at.isoformat(), now=now, local_zone=zone)
    stale = loaded.view.age_warning is not None
    items = checks.stock_lab_checks(loaded, now)
    text = words.summary_from(made, stale=stale)
    if not items:
        text = f"{text} {words.SUMMARY_NO_CHECKS}"
    return {"title": words.CHECKS_TITLE, "note": text, "technical": None, "items": items}


__all__ = ["SCHEMA", "VERSION", "build_health", "health_view"]
