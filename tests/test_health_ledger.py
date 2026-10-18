"""
Prospective forecast ledger health: chain integrity, the kept head copy, the
sessions without a record and the registered trial's conditions, in every
state. A missing ledger, backup or calendar is never healthy; a broken chain
or a ledger that lost its kept head fails; a truncated tail raises a
deterioration alert; a section 5 condition during the trial fails, and
health never writes to the ledger.
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
CHAIN_CHECKS = ("forecast.ledger", "forecast.ledger_backup", "forecast.ledger_sessions")
# The planted records below carry no forecast rows, so the scoring check cannot read them.
NO_ROWS = (HealthState.UNAVAILABLE, "unrecognized_rows")


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
    assert list(checks) == [*CHAIN_CHECKS, "forecast.ledger_scores", "forecast.ledger_trial"]
    for check in checks.values():
        assert (check.state, check.value) == (INSUFFICIENT, None)
    assert checks["forecast.ledger"].reason == "input_not_configured"
    assert checks["forecast.ledger_backup"].reason == "backup_not_configured"


def test_a_ledger_not_written_yet_is_insufficient_evidence(tmp_path):
    # The genesis hash is what the job reports for a ledger with no entry.
    checks = _checks(tmp_path / "ledger.jsonl", backup=_kept(tmp_path, job_ledger.GENESIS))
    assert {c.state for c in checks.values()} == {INSUFFICIENT}
    assert checks["forecast.ledger"].reason == "input_missing"
    assert "activation decision" in checks["forecast.ledger"].detail


def test_a_deleted_ledger_with_a_kept_head_fails(tmp_path):
    path = _ledger(tmp_path)
    head = job_ledger.head(job_ledger.read_ledger(path))
    backup = _kept(tmp_path, head)
    path.unlink()  # the whole file gone: the chain alone would say "not written yet"
    checks = _checks(path, backup=backup)
    assert (checks["forecast.ledger"].state, checks["forecast.ledger"].reason) == (
        HealthState.FAILED,
        "ledger_deleted",
    )
    assert "deleted or moved" in checks["forecast.ledger"].summary
    assert (checks["forecast.ledger_backup"].state, checks["forecast.ledger_backup"].reason) == (
        HealthState.FAILED,
        "head_not_in_ledger",
    )
    sessions = checks["forecast.ledger_sessions"]
    assert (sessions.state, sessions.reason) == (HealthState.UNAVAILABLE, "ledger_deleted")
    trial = checks["forecast.ledger_trial"]
    assert (trial.state, trial.reason) == (HealthState.FAILED, "kept_head_failed")
    assert all(check.state is not HealthState.HEALTHY for check in checks.values())


def test_a_genesis_kept_head_is_the_head_of_an_empty_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("", encoding="utf-8")
    genesis = _kept(tmp_path, job_ledger.GENESIS)
    backup = _checks(path, backup=genesis)["forecast.ledger_backup"]
    assert (backup.state, backup.value) == (HealthState.HEALTHY, 0)
    # Once records exist, a genesis copy is simply behind, never a deletion.
    _append(path, "forecast", "2026-10-14")
    _append(path, "forecast", "2026-10-15")
    behind = _checks(path, backup=genesis)["forecast.ledger_backup"]
    assert (behind.state, behind.reason, behind.value) == (HealthState.DEGRADED, "backup_behind", 2)


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
    assert {checks[check_id].state for check_id in CHAIN_CHECKS} == {HealthState.HEALTHY}
    scores = checks["forecast.ledger_scores"]
    assert (scores.state, scores.reason) == NO_ROWS
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
    assert {(checks[c].state, checks[c].reason) for c in CHAIN_CHECKS} == {
        (INSUFFICIENT, "synthetic_input")
    }
    assert (checks["forecast.ledger_scores"].state, checks["forecast.ledger_scores"].reason) == (
        NO_ROWS
    )
    allowed = _by_id(
        ledger_checks(
            path, backup=_kept(tmp_path, head), calendar=CALENDAR, now=NOW, allow_synthetic=True
        )
    )
    assert {allowed[check_id].state for check_id in CHAIN_CHECKS} == {HealthState.HEALTHY}


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


# --- matured forecasts and their scores (forecast.ledger_scores) ---

RECORDED = pd.Timestamp("2026-10-15T23:00:00Z")


def _row(asset="VTI", horizon=1, target="total_log_return", status="ok", entry_lag=1) -> dict:
    return {
        "asset": asset,
        "horizon": horizon,
        "target": target,
        "entry_lag": entry_lag,
        "status": status,
    }


def _forecast(path: Path, as_of: str, rows: list[dict], data_class: str = "real"):
    payload = {"as_of_session": as_of, "data": {"data_class": data_class}, "rows": rows}
    return job_ledger.append_entry(path, "forecast", payload, recorded_at=RECORDED)


def _score(path: Path, entry, row: dict) -> None:
    score = {
        "forecast_entry_sha256": entry.entry_sha256,
        "asset": row["asset"],
        "horizon": row["horizon"],
        "target": row["target"],
    }
    job_ledger.append_entry(path, "score", {"scores": [score]}, recorded_at=RECORDED)


def _scores(path, **kwargs):
    return _checks(path, **kwargs)["forecast.ledger_scores"]


# 2026-10-12 + entry lag 1 + horizon 1 exits on 10-14, due at 09:30 New York on 10-15
# (before NOW); 2026-10-14 exits on 10-16, due on Monday 10-19 (after NOW).
DUE, NOT_DUE = "2026-10-12", "2026-10-14"


def test_a_matured_row_without_a_score_is_degraded_and_listed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _forecast(path, DUE, [_row(), _row(target="log_realized_volatility", status="refused")])
    _forecast(path, NOT_DUE, [_row()])
    check = _scores(path)
    assert (check.state, check.reason, check.value) == (HealthState.DEGRADED, "scores_overdue", 1)
    assert (
        "oldest exit session 2026-10-14" in check.summary and "Nothing is filled" in check.summary
    )
    assert "2026-10-12 VTI h1 total_log_return (exit 2026-10-14)" in check.detail
    assert "scored: 0; not due yet: 1; due without a score: 1" in check.detail  # refused row: none


def test_every_matured_row_scored_is_healthy(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = _forecast(path, DUE, [_row()])
    _forecast(path, NOT_DUE, [_row()])
    _score(path, entry, _row())
    check = _scores(path)
    assert (check.state, check.reason, check.value) == (HealthState.HEALTHY, None, 0)
    assert check.measured_at == NOW and "1 scored, 1 not due yet" in check.summary


def test_no_matured_row_is_insufficient_evidence_never_healthy(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _forecast(path, NOT_DUE, [_row(), _row(asset="AGG")])
    check = _scores(path)
    assert (check.state, check.reason, check.value) == (INSUFFICIENT, "none_matured", 0)


@pytest.mark.parametrize(
    "setup, reason",
    [
        (lambda path: _forecast(path, DUE, [_row(status="refused")]), "no_forecast_recorded"),
        (lambda path: _append(path, "refusal", DUE), "no_forecast_recorded"),
    ],
)
def test_a_ledger_without_a_made_row_is_insufficient_evidence(tmp_path, setup, reason):
    path = tmp_path / "ledger.jsonl"
    setup(path)
    check = _scores(path)
    assert (check.state, check.reason, check.value) == (INSUFFICIENT, reason, None)


def test_without_a_calendar_matured_rows_are_not_counted(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _forecast(path, DUE, [_row()])
    check = _scores(path, calendar=None)
    assert (check.state, check.reason, check.value) == (
        INSUFFICIENT,
        "calendar_not_configured",
        None,
    )


def test_a_row_off_the_calendar_or_unreadable_is_unavailable(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _forecast(path, "2026-10-10", [_row()])  # a Saturday: not a session
    _forecast(path, DUE, [{"asset": "VTI"}])  # a row without its identity fields
    check = _scores(path)
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "unrecognized_rows")
    assert "rows not placed on the calendar: 1; unreadable rows: 1" in check.detail


def test_a_synthetic_scored_ledger_is_never_healthy_in_the_private_scope(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = _forecast(path, DUE, [_row()], data_class="synthetic")
    _score(path, entry, _row())
    check = _scores(path)
    assert (check.state, check.reason) == (INSUFFICIENT, "synthetic_input")
    allowed = _by_id(
        ledger_checks(path, backup=None, calendar=CALENDAR, now=NOW, allow_synthetic=True)
    )
    assert allowed["forecast.ledger_scores"].state is HealthState.HEALTHY


def test_a_broken_chain_leaves_scoring_unavailable(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _forecast(path, DUE, [_row()])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0].replace('"VTI"', '"AGG"') + "\n", encoding="utf-8")
    check = _scores(path)
    assert (check.state, check.value) == (HealthState.UNAVAILABLE, None)


# --- the registered trial (forecast.ledger_trial) ---

FUNDS = ("VTI", "AGG", "SHV")
DAYS = [f"{d:%Y-%m-%d}" for d in nyse_rules.sessions("2026-10-14", "2026-12-31")]


def _evening(session: str) -> pd.Timestamp:
    return pd.Timestamp(f"{session} 19:00").tz_localize("America/New_York").tz_convert("UTC")


def _after_deadline(session: str) -> datetime:
    following = DAYS[DAYS.index(session) + 1]  # a weekday session here: its 09:30 has passed
    stamp = pd.Timestamp(f"{following} 10:00").tz_localize("America/New_York")
    return stamp.tz_convert("UTC").to_pydatetime()


def _record(path: Path, session: str, funds=FUNDS, kind="forecast"):
    payload = {"as_of_session": session, "data": {"data_class": "real"}, "refusals": []}
    if kind == "forecast":
        payload["rows"] = [
            {"asset": f, "horizon": 5, "target": "total_log_return", "entry_lag": 1, "status": "ok"}
            for f in funds
        ]
    return job_ledger.append_entry(path, kind, payload, recorded_at=_evening(session))


@pytest.fixture
def registered(prospective_frozen, tmp_path):
    from forecast_prospective_helpers import register

    path = tmp_path / "registered.jsonl"
    register(prospective_frozen.models(), path)
    return path


def _trial(path, tmp_path, now, *, kept=True):
    backup = _kept(tmp_path, job_ledger.head(job_ledger.read_ledger(path))) if kept else None
    return _checks(path, backup=backup, now=now)["forecast.ledger_trial"]


def test_without_a_registration_the_trial_is_not_judged(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    check = _checks(empty)["forecast.ledger_trial"]
    assert (check.state, check.reason) == (INSUFFICIENT, "not_registered")


def test_records_without_a_registration_fail(tmp_path):
    check = _checks(_ledger(tmp_path))["forecast.ledger_trial"]
    assert (check.state, check.reason) == (HealthState.FAILED, "records_without_registration")


def test_a_registration_awaits_session_1(registered, tmp_path):
    check = _trial(registered, tmp_path, NOW)
    assert (check.state, check.reason, check.value) == (INSUFFICIENT, "awaiting_session_1", 0)
    assert "registered 2026-10-14T14:00:00+00:00" in check.detail


def test_a_complete_trial_so_far_is_healthy_and_names_its_sessions(registered, tmp_path):
    for session in DAYS[:3]:
        _record(registered, session)
    before = registered.read_bytes()
    check = _trial(registered, tmp_path, _after_deadline(DAYS[2]))
    assert (check.state, check.value, check.unit) == (
        HealthState.HEALTHY,
        0,
        "section 5 conditions",
    )
    assert "session 1 2026-10-14" in check.detail
    assert "latest due session 3 (trial)" in check.detail
    assert "trial: sessions 1-20, to 2026-11-10" in check.detail
    assert "confirmatory phase from session 21, 2026-11-11" in check.detail
    assert registered.read_bytes() == before  # health never writes to the ledger


def test_a_missed_trial_session_is_a_void_condition_reported_not_decided(registered, tmp_path):
    _record(registered, DAYS[0])
    _record(registered, DAYS[2])
    check = _trial(registered, tmp_path, _after_deadline(DAYS[2]))
    assert (check.state, check.reason, check.value) == (
        HealthState.FAILED,
        "trial_void_condition",
        1,
    )
    assert f"condition=session_not_recorded phase=trial session={DAYS[1]} number=2" in check.detail
    assert "owner" in check.summary and "changes nothing" in check.summary


def test_a_refused_session_and_a_data_gap_are_listed(registered, tmp_path):
    _record(registered, DAYS[0])
    _record(registered, DAYS[1], funds=("AGG", "SHV"))
    _record(registered, DAYS[2], kind="refusal")
    _record(registered, DAYS[3], funds=("AGG", "SHV"))
    check = _trial(registered, tmp_path, _after_deadline(DAYS[3]))
    assert check.state is HealthState.FAILED
    assert f"condition=session_refused phase=trial session={DAYS[2]}" in check.detail
    assert f"condition=data_gap phase=trial session={DAYS[3]} number=4 asset=VTI" in check.detail


def test_a_condition_after_the_trial_counts_against_c3_not_as_a_void(registered, tmp_path):
    for session in DAYS[:21]:
        _record(registered, session)
    _record(registered, DAYS[22])  # DAYS[21], session 22, missed
    check = _trial(registered, tmp_path, _after_deadline(DAYS[22]))
    assert (check.state, check.reason) == (HealthState.DEGRADED, "condition_after_trial")
    assert "phase=confirmatory" in check.detail and "C3" in check.summary


def test_a_lost_or_stale_kept_head_is_a_trial_condition(registered, tmp_path):
    _record(registered, DAYS[0])
    lost = _checks(registered, backup=_kept(tmp_path, "a" * 64), now=_after_deadline(DAYS[0]))
    assert "condition=kept_head_failed" in lost["forecast.ledger_trial"].detail
    stale_head = job_ledger.read_ledger(registered)[0].entry_sha256
    stale = _checks(registered, backup=_kept(tmp_path, stale_head), now=_after_deadline(DAYS[0]))
    assert "condition=kept_head_unresolved" in stale["forecast.ledger_trial"].detail
    assert stale["forecast.ledger_trial"].state is HealthState.FAILED


def test_without_a_kept_head_the_trial_is_never_healthy(registered, tmp_path):
    _record(registered, DAYS[0])
    check = _trial(registered, tmp_path, _after_deadline(DAYS[0]), kept=False)
    assert (check.state, check.reason) == (INSUFFICIENT, "kept_head_not_judged")


def test_a_broken_chain_is_a_trial_condition(registered, tmp_path):
    _record(registered, DAYS[0])
    lines = registered.read_text(encoding="utf-8").splitlines(keepends=True)
    registered.write_text(lines[1], encoding="utf-8")  # the registration line removed
    check = _checks(registered, now=_after_deadline(DAYS[0]))["forecast.ledger_trial"]
    assert (check.state, check.reason) == (HealthState.FAILED, "chain_break")
    assert "condition=chain_break" in check.detail


def test_registered_sessions_are_expected_from_session_1(registered, tmp_path):
    _record(registered, DAYS[0])
    _record(registered, DAYS[2])
    sessions = _checks(registered, now=_after_deadline(DAYS[2]))["forecast.ledger_sessions"]
    assert (sessions.state, sessions.value) == (HealthState.DEGRADED, 1)
    assert "session 1" in sessions.threshold and "section 3" in sessions.summary
    assert "No cadence is pre-registered" not in sessions.summary
    integrity = _checks(registered, now=NOW)["forecast.ledger"]
    assert "1 registrations, 2 forecasts" in integrity.detail


def test_a_synthetic_registered_trial_is_insufficient_evidence(registered, tmp_path):
    payload = {
        "as_of_session": DAYS[0],
        "data": {"data_class": "synthetic"},
        "rows": [{"asset": "VTI", "status": "ok"}],
    }
    job_ledger.append_entry(registered, "forecast", payload, recorded_at=_evening(DAYS[0]))
    check = _trial(registered, tmp_path, _after_deadline(DAYS[0]))
    assert (check.state, check.reason) == (INSUFFICIENT, "synthetic_input")


@pytest.mark.parametrize(
    ("missed", "state", "reason"),
    [
        (19, HealthState.FAILED, "trial_void_condition"),  # session 20
        (20, HealthState.DEGRADED, "condition_after_trial"),  # session 21
    ],
)
def test_session_20_closes_the_trial(registered, tmp_path, missed, state, reason):
    for index in range(23):
        if index != missed:
            _record(registered, DAYS[index])
    check = _trial(registered, tmp_path, _after_deadline(DAYS[22]))
    assert (check.state, check.reason, check.value) == (state, reason, 1)


def test_a_second_registration_entry_fails_the_trial(registered, tmp_path):
    first = job_ledger.read_ledger(registered)[0]
    job_ledger.append_entry(
        registered, "registration", dict(first.payload), recorded_at=_evening(DAYS[0])
    )
    check = _checks(registered, now=NOW)["forecast.ledger_trial"]
    assert (check.state, check.reason) == (HealthState.FAILED, "registration_not_unique")


def test_without_a_calendar_the_trial_is_not_counted(registered, tmp_path):
    _record(registered, DAYS[0])
    check = _checks(registered, calendar=None, now=NOW)["forecast.ledger_trial"]
    assert (check.state, check.reason) == (INSUFFICIENT, "calendar_not_configured")


def test_a_synthetic_trial_with_a_void_condition_still_fails(registered, tmp_path):
    for session in (DAYS[0], DAYS[2]):  # DAYS[1] missed
        payload = {
            "as_of_session": session,
            "data": {"data_class": "synthetic"},
            "rows": [{"asset": fund, "status": "ok"} for fund in FUNDS],
        }
        job_ledger.append_entry(registered, "forecast", payload, recorded_at=_evening(session))
    check = _trial(registered, tmp_path, _after_deadline(DAYS[2]))
    assert (check.state, check.reason) == (HealthState.FAILED, "trial_void_condition")


def test_a_calendar_refusing_the_trial_range_is_unavailable(registered, tmp_path):
    _record(registered, DAYS[0])

    def refusing(start, end):
        raise CalendarRefused("outside_coverage", "dates after the rules calendar coverage")

    calendar = RulesCalendar(refusing, "test")
    check = _checks(registered, calendar=calendar, now=NOW)["forecast.ledger_trial"]
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "calendar_refused")
