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
