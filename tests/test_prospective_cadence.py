"""
The registered cadence of a prospective ledger
(docs/research/prospective_registration.md, sections 3 to 6): session 1 is
the first record after the registration, sessions 1-20 are the trial and the
confirmatory phase starts at session 21. Section 5's conditions are reported
as structured items (kind, phase, session, number, fund); nothing here decides
or changes a record.
"""

from __future__ import annotations

import pandas as pd
import pytest
from forecast_prospective_helpers import REGISTERED, register

from stock_agent.calendars import nyse_rules
from stock_agent.forecasting.prospective import cadence, ledger, registration_record, window
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.forecasting.prospective.refusal import ForecastRefused

FUNDS = ("VTI", "AGG", "SHV")
DAYS = [f"{d:%Y-%m-%d}" for d in nyse_rules.sessions("2026-10-14", "2026-12-31")]


def _calendar(start, end):
    return nyse_rules.sessions(start, end)


def _evening(session: str) -> pd.Timestamp:
    return window.local_instant(session, "19:00", "America/New_York")


def _deadline(session: str) -> pd.Timestamp:
    return window.local_instant(window.next_weekday(session), "09:30", "America/New_York")


def _forecast(path, session, funds=FUNDS, recorded_at=None):
    payload = {
        "as_of_session": session,
        "data": {"data_class": "real"},
        "rows": [
            {"asset": fund, "horizon": 5, "target": "total_log_return", "status": "ok"}
            for fund in funds
        ],
        "refusals": [{"asset": fund} for fund in FUNDS if fund not in funds],
    }
    when = _evening(session) if recorded_at is None else recorded_at
    return ledger.append_entry(path, "forecast", payload, recorded_at=when)


def _refusal(path, session):
    payload = {"as_of_session": session, "data": {"data_class": "real"}, "refusals": []}
    return ledger.append_entry(path, "refusal", payload, recorded_at=_evening(session))


@pytest.fixture
def path(prospective_frozen, tmp_path):
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)  # 2026-10-14 14:00 UTC
    return path


def _report(path, now, kept=None):
    entries = ledger.read_ledger(path)
    head = ledger.head(entries) if kept == "head" else kept
    return cadence.cadence(entries, calendar=_calendar, now=pd.Timestamp(now), kept=head)


def _kinds(report):
    return [(c.condition, c.phase, c.session, c.number, c.asset) for c in report.conditions]


def test_the_conventions_are_the_registered_configs():
    prospective = load_config(REGISTERED).prospective
    assert cadence.CLOSE_AVAILABLE_LOCAL == prospective.close_available_local
    assert cadence.DEADLINE_LOCAL == prospective.record_deadline_local
    assert cadence.TIMEZONE == prospective.timezone
    assert cadence.TRIAL_SESSIONS == 20


def test_before_session_1_nothing_is_due(path):
    report = _report(path, "2026-10-20T12:00:00Z", kept="head")
    assert report.first_session is None and report.number is None
    assert report.conditions == ()
    assert report.registration.entry.sequence == 0


def test_session_1_the_trial_and_the_confirmatory_phase(path):
    for session in DAYS[:21]:
        _forecast(path, session)
    report = _report(path, _deadline(DAYS[20]) + pd.Timedelta(minutes=1), kept="head")
    assert report.first_session == DAYS[0] == "2026-10-14"
    assert report.trial_last_session == DAYS[19] == "2026-11-10"
    assert report.confirmatory_first_session == DAYS[20] == "2026-11-11"
    assert (report.number, report.phase) == (21, "confirmatory")
    assert report.conditions == ()
    early = _report(path, _deadline(DAYS[4]) + pd.Timedelta(minutes=1), kept="head")
    assert (early.number, early.phase) == (5, "trial")


def test_a_missed_session_is_a_trial_condition_and_never_filled(path):
    _forecast(path, DAYS[0])
    _forecast(path, DAYS[2])  # DAYS[1] missed
    report = _report(path, _deadline(DAYS[2]) + pd.Timedelta(minutes=1), kept="head")
    assert _kinds(report) == [("session_not_recorded", "trial", DAYS[1], 2, None)]
    assert report.trial_conditions == report.conditions
    assert len(ledger.read_ledger(path)) == 3  # nothing was written


def test_a_session_not_yet_past_its_deadline_is_not_missing(path):
    _forecast(path, DAYS[0])
    report = _report(path, _deadline(DAYS[1]) - pd.Timedelta(minutes=1), kept="head")
    assert report.conditions == ()


def test_a_refused_or_late_record_is_a_condition(path):
    _forecast(path, DAYS[0])
    _refusal(path, DAYS[1])
    _forecast(path, DAYS[2], recorded_at=_deadline(DAYS[2]) + pd.Timedelta(seconds=1))
    report = _report(path, _deadline(DAYS[3]), kept="head")
    assert ("session_refused", "trial", DAYS[1], 2, None) in _kinds(report)
    assert ("record_outside_window", "trial", DAYS[2], 3, None) in _kinds(report)
    assert ("session_not_recorded", "trial", DAYS[3], 4, None) in _kinds(report)


def test_a_record_whose_close_is_not_after_the_registration_is_a_condition(
    prospective_frozen, tmp_path
):
    path = tmp_path / "late.jsonl"
    register(
        prospective_frozen.models(),
        path,
        registered_at=pd.Timestamp("2026-10-14T22:30:00Z"),
        recorded_at=pd.Timestamp("2026-10-14T22:45:00Z"),
    )
    _forecast(path, DAYS[0])  # written outside the job, which refuses it
    report = _report(path, _deadline(DAYS[0]), kept="head")
    assert ("record_before_registration", "trial", DAYS[0], 1, None) in _kinds(report)


@pytest.mark.parametrize(("missed", "gap"), [(2, False), (3, True)])
def test_a_fund_without_rows_for_more_than_2_sessions_is_a_data_gap(path, missed, gap):
    _forecast(path, DAYS[0])
    for session in DAYS[1 : 1 + missed]:
        _forecast(path, session, funds=("AGG", "SHV"))  # VTI refused alone
    _forecast(path, DAYS[1 + missed])
    report = _report(path, _deadline(DAYS[1 + missed]), kept="head")
    expected = [("data_gap", "trial", DAYS[3], 4, "VTI")] if gap else []
    assert _kinds(report) == expected
    if gap:
        assert f"{DAYS[1]} to {DAYS[3]}" in report.conditions[0].detail


def test_missed_and_refused_sessions_count_toward_a_data_gap(path):
    _forecast(path, DAYS[0])
    _forecast(path, DAYS[1], funds=("AGG", "SHV"))
    _refusal(path, DAYS[2])
    _forecast(path, DAYS[4])  # DAYS[3] missed: VTI, AGG and SHV have no rows on 2-4 or 3
    report = _report(path, _deadline(DAYS[4]), kept="head")
    gaps = [c for c in report.conditions if c.condition == "data_gap"]
    assert [(g.asset, g.session, g.number) for g in gaps] == [("VTI", DAYS[3], 4)]


def test_after_the_trial_a_condition_is_reported_for_c3_not_as_a_void(path):
    for session in DAYS[:22]:
        if session != DAYS[21]:
            _forecast(path, session)
    _forecast(path, DAYS[22])
    report = _report(path, _deadline(DAYS[22]), kept="head")
    assert _kinds(report) == [("session_not_recorded", "confirmatory", DAYS[21], 22, None)]
    assert report.trial_conditions == () and report.later_conditions == report.conditions


def test_a_kept_head_no_longer_in_the_ledger_is_failed(path):
    _forecast(path, DAYS[0])
    report = _report(path, _deadline(DAYS[0]), kept="a" * 64)
    assert [c.condition for c in report.conditions] == ["kept_head_failed"]
    assert report.conditions[0].phase == "trial"


def test_a_stale_kept_head_is_a_condition_only_after_the_next_deadline(path):
    registration_head = ledger.head(ledger.read_ledger(path))
    _forecast(path, DAYS[0])  # recorded 19:00 New York on 10-14; next deadline 09:30 on 10-15
    before = _report(path, _deadline(DAYS[0]) - pd.Timedelta(minutes=1), kept=registration_head)
    assert before.conditions == ()
    after = _report(path, _deadline(DAYS[0]), kept=registration_head)
    assert [c.condition for c in after.conditions] == ["kept_head_unresolved"]


def test_an_unregistered_ledger_is_refused(tmp_path):
    path = tmp_path / "plain.jsonl"
    _forecast(path, DAYS[0])
    with pytest.raises(ForecastRefused) as refused:
        _report(path, _deadline(DAYS[0]))
    assert refused.value.reason == "not_registered"


def test_conditions_are_structured(path):
    _forecast(path, DAYS[0])
    report = _report(path, _deadline(DAYS[1]), kept="head")
    (condition,) = report.conditions
    assert condition.to_json() == {
        "condition": "session_not_recorded",
        "phase": "trial",
        "session": DAYS[1],
        "number": 2,
        "asset": None,
        "detail": condition.detail,
    }
    assert condition.line().startswith(
        f"condition=session_not_recorded phase=trial session={DAYS[1]} number=2"
    )
    assert registration_record.KIND == "registration"


# --- boundaries and phases ---------------------------------------------------------------


@pytest.mark.parametrize(("missed", "phase"), [(19, "trial"), (20, "confirmatory")])
def test_session_20_is_the_last_trial_session_and_21_the_first_confirmatory(path, missed, phase):
    for index in range(23):
        if index != missed:
            _forecast(path, DAYS[index])
    report = _report(path, _deadline(DAYS[22]), kept="head")
    assert _kinds(report) == [("session_not_recorded", phase, DAYS[missed], missed + 1, None)]
    assert cadence.phase_of(20) == "trial" and cadence.phase_of(21) == "confirmatory"


@pytest.mark.parametrize(
    ("local", "outside"),
    [("17:59:59", True), ("18:00:00", False)],  # the window opens at 18:00 New York
)
def test_the_recording_window_opens_at_18_00(path, local, outside):
    when = window.local_instant(DAYS[0], local, "America/New_York")
    _forecast(path, DAYS[0], recorded_at=when)
    report = _report(path, _deadline(DAYS[0]), kept="head")
    kinds = [c.condition for c in report.conditions]
    assert ("record_outside_window" in kinds) is outside


@pytest.mark.parametrize(("offset", "outside"), [(-1, False), (0, True)])
def test_the_recording_window_closes_at_the_deadline(path, offset, outside):
    _forecast(path, DAYS[0], recorded_at=_deadline(DAYS[0]) + pd.Timedelta(seconds=offset))
    report = _report(path, _deadline(DAYS[1]), kept="head")
    kinds = [c.condition for c in report.conditions]
    assert ("record_outside_window" in kinds) is outside


def test_a_record_on_a_date_that_is_not_a_session_is_reported(path):
    _forecast(path, DAYS[0])
    saturday = "2026-10-17"
    payload = {"as_of_session": saturday, "rows": [], "refusals": []}
    ledger.append_entry(path, "forecast", payload, recorded_at=_evening(saturday))
    report = _report(path, _deadline(DAYS[3]), kept="head")
    off = [c for c in report.conditions if c.condition == "record_off_calendar"]
    assert [(c.session, c.number) for c in off] == [(saturday, None)]


def test_a_data_gap_starting_in_the_trial_is_a_trial_condition(path):
    for index in range(18):
        _forecast(path, DAYS[index])
    for index in (18, 19, 20):  # sessions 19, 20 and 21
        _forecast(path, DAYS[index], funds=("AGG", "SHV"))
    _forecast(path, DAYS[21])
    report = _report(path, _deadline(DAYS[21]), kept="head")
    assert _kinds(report) == [("data_gap", "trial", DAYS[20], 21, "VTI")]


def test_a_long_or_open_gap_is_reported_once(path):
    _forecast(path, DAYS[0])
    for index in range(1, 6):  # five sessions without VTI, still open at `now`
        _forecast(path, DAYS[index], funds=("AGG", "SHV"))
    report = _report(path, _deadline(DAYS[5]), kept="head")
    (gap,) = report.conditions
    assert (gap.condition, gap.session, gap.number) == ("data_gap", DAYS[3], 4)
    assert f"5 consecutive expected sessions, {DAYS[1]} to {DAYS[5]}" in gap.detail


def test_after_the_trial_refused_late_and_gap_conditions_are_confirmatory(path):
    for index in range(21):
        _forecast(path, DAYS[index])
    _refusal(path, DAYS[21])
    _forecast(path, DAYS[22], funds=("AGG", "SHV"), recorded_at=_deadline(DAYS[22]))
    for index in (23, 24):  # VTI: no rows on sessions 22-25
        _forecast(path, DAYS[index], funds=("AGG", "SHV"))
    report = _report(path, _deadline(DAYS[24]), kept="head")
    assert {c.phase for c in report.conditions} == {"confirmatory"}
    assert {c.condition for c in report.conditions} == {
        "session_refused",
        "record_outside_window",
        "data_gap",
    }


def test_a_kept_head_lapse_takes_the_phase_of_the_deadline_it_missed(path):
    for index in range(19):  # sessions 1-19, each copied
        _forecast(path, DAYS[index])
    stale = ledger.head(ledger.read_ledger(path))
    _forecast(path, DAYS[19])  # session 20's record is never copied; its deadline is in the trial
    later = _report(path, _deadline(DAYS[22]), kept=stale)  # first judged at session 23
    (lapse,) = [c for c in later.conditions if c.condition == "kept_head_unresolved"]
    assert lapse.phase == "trial"
    assert later.phase == "confirmatory"  # the latest due session is past the trial


def test_a_kept_head_lapse_after_the_trial_is_confirmatory(path):
    for index in range(22):
        _forecast(path, DAYS[index])
    stale = ledger.head(ledger.read_ledger(path))
    _forecast(path, DAYS[22])  # session 23's record not copied
    report = _report(path, _deadline(DAYS[23]), kept=stale)
    (lapse,) = [c for c in report.conditions if c.condition == "kept_head_unresolved"]
    assert lapse.phase == "confirmatory"
    lost = _report(path, _deadline(DAYS[23]), kept="b" * 64)
    (failed,) = [c for c in lost.conditions if c.condition == "kept_head_failed"]
    assert failed.phase == "confirmatory"
