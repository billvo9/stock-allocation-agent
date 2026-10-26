"""
The Lab app's Health view model (docs/product/lab_app_design.md, "Health in
detail"): the headline is Home's status line at the same moment, so the two never
disagree; the details are the evening updates and Model Health's Stock Lab checks
from its latest local summary, with that summary's time, and a missing, refused,
stale or later summary named as such. Fixtures only: a Stock Lab store on a
simulated clock (real-class records use the made-up ticker SYNTH) and summaries
written under tmp_path. No network, no private store.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, ny
from test_app_destinations import scan_for_advice

from stock_agent.app.health_summary import load_summary
from stock_agent.app.web import health, home, words
from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.health.context import CodeState
from stock_agent.health.model import Check, HealthState, Scope
from stock_agent.health.summary import build_summary, write_summary
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords

NEW_YORK = ZoneInfo("America/New_York")
T = REAL_TEST_TICKER
TUESDAY, WEDNESDAY, THURSDAY = "2026-10-20", "2026-10-21", "2026-10-22"
KEY = "k" * 43


def at(day: str, hhmm: str) -> datetime:
    return ny(day, hhmm).tz_convert(NEW_YORK).to_pydatetime()


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def evening(lab, as_of, *, status="ok"):
    record = lab.forecast(T, as_of, created_at=ny(as_of, "20:31"))
    lab.job(at=ny(as_of, "20:32"), expected=as_of, status=status, activation=None)
    return record


def no_prices(symbol, *, count, retrieved_by):
    return SimpleNamespace(state="no_store", reason=None, sessions=(), last_session=None)


def checked(found: dict) -> dict:
    scan_for_advice(json.dumps(found, allow_nan=False))
    assert found["title"] == words.HEALTH_TITLES[found["level"]]
    return found


def view(lab, now, *, summary=None, research_url=None, data_class="real", records=None):
    return checked(
        health.build_health(
            records or ReadOnlyRecords(lab.path),
            now=now,
            calendar=nyse_sessions,
            calendar_id="test_nyse_rules",
            data_class=data_class,
            local_zone=NEW_YORK,
            summary=summary,
            research_url=research_url,
            tickers=[T] if data_class == "real" else ["DEMO_CALM"],
        )
    )


def status_on_home(lab, now) -> dict:
    return home.build_home(
        ReadOnlyRecords(lab.path),
        now=now,
        calendar=nyse_sessions,
        calendar_id="test_nyse_rules",
        data_class="real",
        recent=no_prices,
        local_zone=NEW_YORK,
        tickers=[T],
    )["status"]


def a_check(check_id: str, state: HealthState = HealthState.HEALTHY, **fields) -> Check:
    measured = state is not HealthState.INSUFFICIENT_EVIDENCE
    defaults = {
        "title": f"Planted {check_id}",
        "measured_at": ny(WEDNESDAY, "07:30").to_pydatetime() if measured else None,
        "value": 1 if measured else None,
        "unit": "planted units" if measured else None,
        "threshold": "planted threshold",
        "summary": f"Planted sentence for {check_id}.",
        "detail": "planted detail",
        "evidence": "fixture:lab",
        "reason": None if state is HealthState.HEALTHY else "planted_reason",
    }
    return Check(
        check_id=check_id, area=check_id.split(".")[0], state=state, **{**defaults, **fields}
    )


def summary_file(path, checks, *, made):
    write_summary(
        path,
        build_summary(
            checks,
            generated_at=made.to_pydatetime(),
            code=CodeState("c" * 40, False),
            scope=Scope.PRIVATE,
            run_context={"environment": "local"},
        ),
    )
    return lambda now: load_summary(path, now=now)


def reading(read, now):
    return lambda: read(now)


CHECKS = [
    a_check("data.stock_lab_freshness"),
    a_check("forecast.stock_lab_drift", HealthState.DEGRADED, title="Inputs look normal"),
    a_check("forecast.stock_lab_scores", HealthState.INSUFFICIENT_EVIDENCE),
    a_check("ci.tests"),  # outside the Stock Lab: only on the full Model Health page
]


# --- the headline: Home's status line ---


def same_as_home(found: dict, on_home: dict) -> None:
    """Health's headline and update lines are Home's status line at the same moment."""

    assert (found["level"], found["line"], found["updates"]["lines"]) == (
        on_home["level"],
        on_home["words"],
        on_home["details"],
    )


def test_all_good_is_homes_green_status_line(lab):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "07:40")
    found = view(lab, now)
    same_as_home(found, status_on_home(lab, now))
    assert found["level"] == "ok" and found["line"] == "Updated last night, 8:32 PM"
    assert found["updates"]["title"] == words.UPDATES_DETAILS_TITLE
    assert found["aside"].startswith("Live record: no scored days yet.")
    assert found["toggle"] == ["Show details", "Hide details"]


def test_a_missed_evening_needs_a_look_on_both_screens(lab):
    evening(lab, TUESDAY)
    now = at(THURSDAY, "07:40")  # Wednesday's update recorded nothing
    found = view(lab, now)
    same_as_home(found, status_on_home(lab, now))
    assert (found["level"], found["title"]) == ("warn", "Needs a look")


def test_a_failed_ledger_check_is_something_wrong_on_both_screens(lab, monkeypatch):
    evening(lab, TUESDAY)
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    now = at(WEDNESDAY, "07:40")
    found = view(lab, now)
    same_as_home(found, status_on_home(lab, now))
    assert found["level"] == "bad" and found["aside"] == words.HEALTH_BAD_ASIDE


def test_before_any_run_nothing_has_started(lab):
    now = at(WEDNESDAY, "07:40")
    found = view(lab, now)
    same_as_home(found, status_on_home(lab, now))
    assert (found["level"], found["title"]) == ("neutral", "Not started yet")


# --- the details: Model Health's Stock Lab checks ---


def test_the_stock_lab_checks_come_from_model_healths_summary(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    checks = view(lab, now, summary=reading(read, now))["checks"]
    assert checks["title"] == words.CHECKS_TITLE
    assert checks["note"] == "From Model Health's summary, this morning, 7:30 AM."
    assert [(c["check_id"], c["mark"], c["state_word"]) for c in checks["items"]] == [
        ("forecast.stock_lab_drift", "warn", "Degraded"),
        ("forecast.stock_lab_scores", "note", "Insufficient evidence"),
        ("data.stock_lab_freshness", "ok", "Healthy"),
    ]  # worst first, as Model Health orders them; nothing outside the Stock Lab
    drift = checks["items"][0]
    assert (drift["title"], drift["words"]) == (
        "Inputs look normal",
        "Planted sentence for forecast.stock_lab_drift.",
    )


def test_a_flagged_check_is_named_under_the_headline(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    found = view(lab, now, summary=reading(read, now))
    assert found["level"] == "ok"  # the headline stays the live status line
    assert found["flagged"] == "Model Health flags 1 check: see the details."
    calm = summary_file(tmp_path / "calm.json", CHECKS[:1], made=ny(WEDNESDAY, "07:30"))
    assert view(lab, now, summary=reading(calm, now))["flagged"] is None


def test_a_summary_more_than_a_day_old_is_named_stale(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    made = ny(WEDNESDAY, "10:00") - timedelta(hours=27)
    read = summary_file(tmp_path / "latest.json", CHECKS, made=made)
    note = view(lab, now, summary=reading(read, now))["checks"]["note"]
    assert note == (
        "From Model Health's latest summary, yesterday, 7:00 AM: more than a day old, so these "
        "checks may be out of date."
    )


def test_a_missing_summary_says_so_and_shows_no_check(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    checks = view(lab, now, summary=lambda: load_summary(tmp_path / "none.json", now=now))["checks"]
    assert (checks["note"], checks["items"], checks["technical"]) == (
        words.SUMMARY_MISSING,
        [],
        None,
    )


def test_a_summary_the_reader_refuses_shows_no_check(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    bad = tmp_path / "latest.json"
    bad.write_text("{not json", encoding="utf-8")
    checks = view(lab, now, summary=lambda: load_summary(bad, now=now))["checks"]
    assert checks["note"] == words.SUMMARY_REFUSED and checks["items"] == []
    assert checks["technical"].startswith("Technical detail: ")


def test_a_summary_written_after_the_request_time_is_not_used(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "07:29")
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    found = view(lab, now, summary=reading(read, now))
    assert found["checks"]["note"] == words.SUMMARY_LATER
    assert found["checks"]["items"] == [] and found["flagged"] is None
    exactly = at(WEDNESDAY, "07:30")  # written at the request time: known then
    checks = view(lab, exactly, summary=reading(read, exactly))["checks"]
    assert checks["note"] == "From Model Health's summary, this morning, 7:30 AM."
    assert len(checks["items"]) == 3


def test_a_summary_without_stock_lab_checks_says_so(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    read = summary_file(
        tmp_path / "latest.json", [a_check("ci.tests")], made=ny(WEDNESDAY, "07:30")
    )
    checks = view(lab, now, summary=reading(read, now))["checks"]
    assert checks["items"] == []
    assert checks["note"].endswith(words.SUMMARY_NO_CHECKS)


# --- the link and the demo ---


def test_the_model_health_link_keeps_this_launchs_research_key(lab):
    found = view(lab, at(WEDNESDAY, "10:00"), research_url=f"http://127.0.0.1:8502/?key={KEY}")
    assert found["link"] == {
        "url": f"http://127.0.0.1:8502/model-health?key={KEY}",
        "words": words.MODEL_HEALTH_LINK,
        "sub": words.MODEL_HEALTH_SUB,
    }
    alone = view(lab, at(WEDNESDAY, "10:00"))
    assert alone["link"] is None and alone["elsewhere"] == words.RESEARCH_ELSEWHERE


def test_the_demo_has_no_model_health_checks(lab):
    lab.forecast("DEMO_CALM", TUESDAY, created_at=ny(TUESDAY, "20:31"))
    lab.job(at=ny(TUESDAY, "20:32"), expected=TUESDAY, status="ok", activation=None,
            data_class="synthetic")  # fmt: skip
    found = view(lab, at(WEDNESDAY, "07:40"), data_class="synthetic")
    assert found["demo"] == words.DEMO
    assert found["checks"]["note"] == words.SUMMARY_DEMO and found["checks"]["items"] == []


# --- the Health tab's badge (owner decision of 2026-10-26: "Keep the green headline with
# the badge"): amber or red as Home's status line says, and "!" whenever Model Health flags
# a Stock Lab check, even under a green headline ---

FLAGGED_BADGE = {"level": "warn", "mark": "!", "words": "1 check flagged"}


def home_with(lab, now, summary=None) -> dict:
    return home.build_home(
        ReadOnlyRecords(lab.path),
        now=now,
        calendar=nyse_sessions,
        calendar_id="test_nyse_rules",
        data_class="real",
        recent=no_prices,
        local_zone=NEW_YORK,
        tickers=[T],
        summary=summary,
    )


def both(lab, now, summary=None) -> tuple[dict, dict]:
    """Home's badge and Health's at the same moment: always the same."""

    on_home = home_with(lab, now, summary)
    on_health = view(lab, now, summary=summary)
    assert on_home["badge"] == on_health["badge"]
    return on_home, on_health


def test_a_green_headline_with_a_flagged_check_shows_the_badge(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    on_home, on_health = both(lab, now, reading(read, now))
    assert on_home["status"]["level"] == "ok" and on_health["level"] == "ok"  # still green
    assert on_home["badge"] == FLAGGED_BADGE
    assert on_health["flagged"] == "Model Health flags 1 check: see the details."


def test_two_flagged_checks_are_counted(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    two = [*CHECKS, a_check("data.stock_lab_tickers", HealthState.FAILED)]
    read = summary_file(tmp_path / "latest.json", two, made=ny(WEDNESDAY, "07:30"))
    on_home, _ = both(lab, now, reading(read, now))
    assert on_home["badge"] == {**FLAGGED_BADGE, "words": "2 checks flagged"}


def test_a_green_headline_with_no_flagged_check_shows_no_badge(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    calm = [
        a_check("data.stock_lab_freshness"),
        a_check("forecast.stock_lab_scores", HealthState.INSUFFICIENT_EVIDENCE),  # too early
        a_check("ci.tests", HealthState.FAILED),  # outside the Stock Lab
    ]
    read = summary_file(tmp_path / "latest.json", calm, made=ny(WEDNESDAY, "07:30"))
    on_home, on_health = both(lab, now, reading(read, now))
    assert on_home["status"]["level"] == "ok" and on_home["badge"] is None
    assert on_health["flagged"] is None
    assert both(lab, now)[0]["badge"] is None  # no summary to read at all


@pytest.mark.parametrize(
    ("kind", "note"),
    [
        ("missing", words.SUMMARY_MISSING),
        ("refused", words.SUMMARY_REFUSED),
        ("later", words.SUMMARY_LATER),
    ],
)
def test_a_summary_that_cannot_be_used_flags_nothing(lab, tmp_path, kind, note):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "07:29")
    path = tmp_path / "latest.json"
    if kind == "refused":
        path.write_text("{not json", encoding="utf-8")
    elif kind == "later":  # written a minute after the request time: not known then
        summary_file(path, CHECKS, made=ny(WEDNESDAY, "07:30"))
    on_home, on_health = both(lab, now, lambda: load_summary(path, now=now))
    assert on_home["badge"] is None and on_health["checks"]["note"] == note
    if kind == "later":  # written at the request time itself: known then, and it flags
        exactly = at(WEDNESDAY, "07:30")
        assert both(lab, exactly, lambda: load_summary(path, now=exactly))[0]["badge"] == (
            FLAGGED_BADGE
        )


@pytest.mark.parametrize(
    ("state", "flags"),
    [
        (HealthState.HEALTHY, False),
        (HealthState.INSUFFICIENT_EVIDENCE, False),
        (HealthState.DEGRADED, True),
        (HealthState.UNAVAILABLE, True),
        (HealthState.FAILED, True),
    ],
)
def test_a_warning_or_worse_flags_and_nothing_milder_does(lab, tmp_path, state, flags):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    one = [a_check("forecast.stock_lab_drift", state)]
    read = summary_file(tmp_path / "latest.json", one, made=ny(WEDNESDAY, "07:30"))
    on_home, _ = both(lab, now, reading(read, now))
    assert on_home["badge"] == (FLAGGED_BADGE if flags else None)


def test_a_stale_summarys_flagged_check_still_shows_the_badge(lab, tmp_path):
    # Model Health's latest word, named stale in the details, still flags the check.
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    made = ny(WEDNESDAY, "10:00") - timedelta(hours=27)
    read = summary_file(tmp_path / "latest.json", CHECKS, made=made)
    on_home, on_health = both(lab, now, reading(read, now))
    assert on_home["badge"] == FLAGGED_BADGE
    assert "more than a day old" in on_health["checks"]["note"]


def test_amber_and_red_headlines_show_their_own_badge(lab, tmp_path, monkeypatch):
    evening(lab, TUESDAY)
    now = at(THURSDAY, "07:40")  # Wednesday's update recorded nothing: amber
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(THURSDAY, "07:30"))
    on_home, _ = both(lab, now, reading(read, now))
    assert on_home["status"]["level"] == "warn"
    assert on_home["badge"] == {"level": "warn", "mark": "!", "words": "needs a look"}
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    on_home, _ = both(lab, now, reading(read, now))
    assert on_home["status"]["level"] == "bad"
    assert on_home["badge"] == {"level": "bad", "mark": "✕", "words": "something is wrong"}
    assert both(lab, now)[0]["badge"]["level"] == "bad"  # with no summary, as today


def test_before_anything_has_started_a_flagged_check_still_shows_the_badge(lab, tmp_path):
    now = at(WEDNESDAY, "10:00")
    read = summary_file(tmp_path / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    on_home, _ = both(lab, now, reading(read, now))
    assert on_home["status"]["level"] == "neutral" and on_home["badge"] == FLAGGED_BADGE
    assert both(lab, now)[0]["badge"] is None


# --- a summary the reader can't open never takes Home (or Health) down ---


def broken_folder(tmp_path):
    """A summary inside a folder this account may not open (as a privacy-protected folder)."""

    folder = tmp_path / "closed"
    folder.mkdir()
    summary_file(folder / "latest.json", CHECKS, made=ny(WEDNESDAY, "07:30"))
    folder.chmod(0)
    return folder / "latest.json"


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens any folder")
def test_a_summary_in_a_folder_that_cannot_be_opened_is_refused(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    path = broken_folder(tmp_path)
    try:
        read = lambda: load_summary(path, now=now)
        on_home, on_health = both(lab, now, read)
        assert on_home["status"]["level"] == "ok" and on_home["badge"] is None
        assert on_health["checks"]["note"] == words.SUMMARY_REFUSED
    finally:
        path.parent.chmod(0o700)


def test_a_summary_too_deeply_nested_to_read_is_refused(lab, tmp_path):
    evening(lab, TUESDAY)
    now = at(WEDNESDAY, "10:00")
    path = tmp_path / "latest.json"
    path.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")
    on_home, on_health = both(lab, now, lambda: load_summary(path, now=now))
    assert on_home["badge"] is None
    assert on_health["checks"]["note"] == words.SUMMARY_REFUSED


def test_home_keeps_its_screen_whatever_the_summary_reader_does(lab):
    evening(lab, TUESDAY)

    def failing():
        raise RuntimeError("an unexpected reader failure")

    found = home_with(lab, at(WEDNESDAY, "10:00"), failing)
    assert found["status"]["level"] == "ok" and found["badge"] is None
