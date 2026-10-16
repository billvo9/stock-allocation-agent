"""
Prospective forecast ledger health: chain integrity, the kept head copy and
the sessions without a record, in every state. A missing ledger, backup or
calendar is never healthy; a broken chain or a ledger that lost its kept
head fails; a truncated tail raises a deterioration alert.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.calendars import nyse_rules
from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.forecasting.prospective import ledger as job_ledger
from stock_agent.health.alerts import derive_alerts
from stock_agent.health.context import CodeState
from stock_agent.health.ledger import RulesCalendar, ledger_checks
from stock_agent.health.model import AlertKind, HealthState, Scope
from stock_agent.health.summary import build_summary

CALENDAR = RulesCalendar(nyse_rules.sessions, nyse_rules.CALENDAR_ID)
# Friday 2026-10-16 after its close, so the deadlines of 10-14 and 10-15 have passed
# (09:30 New York on the next weekday) and 10-16's has not (Monday 10-19).
NOW = datetime(2026, 10, 16, 22, tzinfo=UTC)
INSUFFICIENT = HealthState.INSUFFICIENT_EVIDENCE


def _append(path: Path, kind: str, as_of: str | None = None, data_class: str = "real"):
    payload = {"record": kind}
    if as_of is not None:
        payload |= {"as_of_session": as_of, "data": {"data_class": data_class}}
    return job_ledger.append_entry(
        path, kind, payload, recorded_at=pd.Timestamp("2026-10-15T23:00:00Z")
    )


def _ledger(tmp_path: Path, sessions=("2026-10-14", "2026-10-15"), data_class="real") -> Path:
    path = tmp_path / "ledger.jsonl"
    for as_of in sessions:
        _append(path, "forecast", as_of, data_class)
    return path


def _by_id(checks):
    return {check.check_id: check for check in checks}


def _kept(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "kept_head.txt"
    path.write_text(value + "\n", encoding="utf-8")
    return path


def _checks(path, *, backup=None, calendar=CALENDAR, now=NOW):
    return _by_id(ledger_checks(path, backup=backup, calendar=calendar, now=now))


def test_no_ledger_path_is_insufficient_evidence_everywhere():
    checks = _checks(None)
    assert list(checks) == ["forecast.ledger", "forecast.ledger_backup", "forecast.ledger_sessions"]
    for check in checks.values():
        assert (check.state, check.value) == (INSUFFICIENT, None)
    assert checks["forecast.ledger"].reason == "input_not_configured"
    assert checks["forecast.ledger_backup"].reason == "backup_not_configured"


def test_a_ledger_not_written_yet_is_insufficient_evidence(tmp_path):
    checks = _checks(tmp_path / "ledger.jsonl", backup=_kept(tmp_path, "a" * 64))
    assert {c.state for c in checks.values()} == {INSUFFICIENT}
    assert checks["forecast.ledger"].reason == "input_missing"
    assert "activation decision" in checks["forecast.ledger"].detail


def test_an_empty_ledger_has_no_entry_and_no_forecast(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("", encoding="utf-8")
    checks = _checks(path)
    assert (checks["forecast.ledger"].state, checks["forecast.ledger"].value) == (INSUFFICIENT, 0)
    assert checks["forecast.ledger"].reason == "no_entries"
    assert checks["forecast.ledger_sessions"].reason == "no_forecast_recorded"


def test_a_complete_real_ledger_with_its_kept_head_is_healthy(tmp_path):
    path = _ledger(tmp_path)
    head = job_ledger.head(job_ledger.read_ledger(path))
    checks = _checks(path, backup=_kept(tmp_path, head))
    assert {c.state for c in checks.values()} == {HealthState.HEALTHY}
    integrity = checks["forecast.ledger"]
    assert integrity.value == 2 and integrity.unit == "ledger entries"
    assert f"head hash {head}" in integrity.detail
    assert "first 2026-10-14, last 2026-10-15" in integrity.detail
    assert checks["forecast.ledger_backup"].value == 0
    sessions = checks["forecast.ledger_sessions"]
    assert sessions.value == 0 and "sessions due: 2 (2026-10-14 to 2026-10-15)" in sessions.detail


def test_a_session_without_a_record_is_degraded_and_listed(tmp_path):
    path = _ledger(tmp_path, ("2026-10-14", "2026-10-16"))
    monday = datetime(2026, 10, 19, 14, tzinfo=UTC)  # after 10-16's deadline (09:30 New York)
    sessions = _checks(path, now=monday)["forecast.ledger_sessions"]
    assert (sessions.state, sessions.reason, sessions.value) == (
        HealthState.DEGRADED,
        "sessions_without_record",
        1,
    )
    assert "sessions without a record: 2026-10-15" in sessions.detail
    assert "No cadence is pre-registered" in sessions.summary


def test_a_refusal_record_counts_as_a_record(tmp_path):
    path = _ledger(tmp_path, ("2026-10-14",))
    _append(path, "refusal", "2026-10-15")
    _append(path, "score")
    checks = _checks(path)
    assert checks["forecast.ledger_sessions"].state is HealthState.HEALTHY
    assert "1 forecasts, 1 scores, 1 refusals" in checks["forecast.ledger"].detail


def test_a_synthetic_ledger_is_never_healthy_in_the_private_scope(tmp_path):
    path = _ledger(tmp_path, data_class="synthetic")
    head = job_ledger.head(job_ledger.read_ledger(path))
    checks = _checks(path, backup=_kept(tmp_path, head))
    assert {(c.state, c.reason) for c in checks.values()} == {(INSUFFICIENT, "synthetic_input")}
    allowed = ledger_checks(
        path, backup=_kept(tmp_path, head), calendar=CALENDAR, now=NOW, allow_synthetic=True
    )
    assert {c.state for c in allowed} == {HealthState.HEALTHY}


@pytest.mark.parametrize(
    "tamper",
    ["edit", "remove_middle", "reorder", "incomplete_tail"],
)
def test_a_broken_chain_fails_and_nothing_else_is_judged(tmp_path, tamper):
    path = _ledger(tmp_path, ("2026-10-14", "2026-10-15", "2026-10-16"))
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if tamper == "edit":
        lines[1] = lines[1].replace("2026-10-15", "2026-10-13")
    elif tamper == "remove_middle":
        del lines[1]
    elif tamper == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines[-1] = lines[-1].rstrip("\n")
    path.write_text("".join(lines), encoding="utf-8")
    head = job_ledger.head(job_ledger.read_ledger(_ledger(tmp_path / "copy")))
    checks = _checks(path, backup=_kept(tmp_path, head))
    integrity = checks["forecast.ledger"]
    assert (integrity.state, integrity.reason, integrity.value) == (
        HealthState.FAILED,
        "ledger_refused",
        None,
    )
    assert "integrity check" in integrity.summary
    for check_id in ("forecast.ledger_backup", "forecast.ledger_sessions"):
        assert (checks[check_id].state, checks[check_id].reason) == (
            HealthState.UNAVAILABLE,
            "ledger_refused",
        )


def test_a_symlinked_ledger_is_refused(tmp_path):
    target = _ledger(tmp_path / "real")
    link = tmp_path / "ledger.jsonl"
    link.symlink_to(target)
    assert _checks(link)["forecast.ledger"].state is HealthState.FAILED


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files")
def test_an_unreadable_ledger_is_unavailable(tmp_path):
    path = _ledger(tmp_path)
    path.chmod(0)
    try:
        integrity = _checks(path)["forecast.ledger"]
    finally:
        path.chmod(0o600)
    assert (integrity.state, integrity.reason) == (HealthState.UNAVAILABLE, "ledger_unreadable")


def test_a_truncated_tail_loses_the_kept_head_and_raises_a_deterioration(tmp_path):
    path = _ledger(tmp_path, ("2026-10-14", "2026-10-15", "2026-10-16"))
    head = job_ledger.head(job_ledger.read_ledger(path))
    backup = _kept(tmp_path, head)
    before = ledger_checks(path, backup=backup, calendar=CALENDAR, now=NOW)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:2]), encoding="utf-8")  # the chain still verifies
    after = _by_id(ledger_checks(path, backup=backup, calendar=CALENDAR, now=NOW))
    assert after["forecast.ledger"].state is HealthState.HEALTHY  # the chain alone cannot tell
    assert (after["forecast.ledger_backup"].state, after["forecast.ledger_backup"].reason) == (
        HealthState.FAILED,
        "head_not_in_ledger",
    )
    previous = build_summary(
        before,
        generated_at=NOW,
        code=CodeState(None, None),
        scope=Scope.PRIVATE,
        run_context={"environment": "local"},
    )
    alerts = derive_alerts(list(after.values()), raised_at=NOW, previous=previous)
    kinds = {(alert.kind, alert.check_id) for alert in alerts}
    assert (AlertKind.DETERIORATION, "forecast.ledger") in kinds  # 3 entries, then 2
    assert (AlertKind.FAILURE, "forecast.ledger_backup") in kinds


def test_a_kept_head_behind_the_ledger_is_degraded(tmp_path):
    path = _ledger(tmp_path, ("2026-10-14",))
    first = job_ledger.head(job_ledger.read_ledger(path))
    _append(path, "forecast", "2026-10-15")
    backup = _checks(path, backup=_kept(tmp_path, first))["forecast.ledger_backup"]
    assert (backup.state, backup.reason, backup.value) == (
        HealthState.DEGRADED,
        "backup_behind",
        1,
    )


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "backup_unreadable"),
        ("not a hash", "backup_unrecognized"),
        ("", "backup_unrecognized"),
    ],
)
def test_an_unusable_kept_head_is_unavailable(tmp_path, content, reason):
    path = _ledger(tmp_path)
    if content is None:
        backup = tmp_path / "kept_dir"
        backup.mkdir()
    else:
        backup = tmp_path / "kept.txt"
        backup.write_text(content, encoding="utf-8")
    check = _checks(path, backup=backup)["forecast.ledger_backup"]
    assert (check.state, check.reason, check.value) == (HealthState.UNAVAILABLE, reason, None)


def test_without_a_calendar_sessions_are_not_counted(tmp_path):
    check = _checks(_ledger(tmp_path), calendar=None)["forecast.ledger_sessions"]
    assert (check.state, check.reason, check.value) == (
        INSUFFICIENT,
        "calendar_not_configured",
        None,
    )


def test_a_calendar_that_refuses_the_range_is_unavailable(tmp_path):
    def refusing(start, end):
        raise CalendarRefused("outside_coverage", "dates after the rules calendar coverage")

    check = _checks(_ledger(tmp_path), calendar=RulesCalendar(refusing, "test"))[
        "forecast.ledger_sessions"
    ]
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "calendar_refused")


def test_the_health_reader_agrees_with_the_jobs_head(tmp_path):
    path = _ledger(tmp_path)
    entries = job_ledger.read_ledger(path)
    integrity = _checks(path)["forecast.ledger"]
    assert job_ledger.head(entries) in integrity.detail
    assert integrity.measured_at == NOW
