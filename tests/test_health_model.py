"""
The health vocabulary: state precedence, the never-healthy-without-a-measurement
rule, the summary's JSON round trip and every refusal of the strict reader.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from stock_agent.health import model
from stock_agent.health.model import (
    PRECEDENCE,
    Alert,
    AlertKind,
    Area,
    Check,
    HealthState,
    HealthSummary,
    Scope,
    SummaryRefusal,
    SummaryRefused,
    read_summary,
    worst,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def make_check(check_id="ci.tests", state=HealthState.HEALTHY, **fields) -> Check:
    defaults = {
        "area": Area(check_id.split(".")[0]) if "." in check_id else Area.CI,
        "title": "Test suite",
        "measured_at": NOW,
        "value": 1845,
        "unit": "tests passed",
        "threshold": "0 failed",
        "summary": "All tests passed.",
        "detail": "pass 1: 1845 passed",
        "evidence": "evidence/verify.log",
        "reason": None if state is HealthState.HEALTHY else "some_reason",
    }
    return Check(check_id=check_id, state=state, **{**defaults, **fields})


def make_summary(*checks: Check, alerts=()) -> HealthSummary:
    return HealthSummary(
        generated_at=NOW,
        code_sha="a" * 40,
        code_dirty=False,
        scope=Scope.SYNTHETIC,
        run_context={"environment": "local"},
        checks=checks,
        alerts=alerts,
    )


def test_precedence_is_documented_worst_first():
    assert [state.value for state in PRECEDENCE] == [
        "failed",
        "unavailable",
        "degraded",
        "insufficient_evidence",
        "healthy",
    ]
    assert set(PRECEDENCE) == set(HealthState)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([HealthState.HEALTHY, HealthState.FAILED], HealthState.FAILED),
        ([HealthState.DEGRADED, HealthState.UNAVAILABLE], HealthState.UNAVAILABLE),
        ([HealthState.HEALTHY, HealthState.DEGRADED], HealthState.DEGRADED),
        (
            [HealthState.HEALTHY, HealthState.INSUFFICIENT_EVIDENCE],
            HealthState.INSUFFICIENT_EVIDENCE,
        ),
        ([HealthState.HEALTHY, HealthState.HEALTHY], HealthState.HEALTHY),
        ([], HealthState.INSUFFICIENT_EVIDENCE),
    ],
)
def test_the_overall_state_is_the_worst_present(states, expected):
    assert worst(states) is expected


def test_a_summary_without_checks_is_never_healthy():
    assert make_summary().overall_state is HealthState.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("missing", ["measured_at", "value"])
def test_a_check_is_never_healthy_without_a_measurement(missing):
    with pytest.raises(SummaryRefused) as refused:
        make_check(**{missing: None})
    assert refused.value.reason is SummaryRefusal.POLICY_VIOLATION


def test_an_unmeasured_check_can_report_insufficient_evidence():
    check = make_check(
        state=HealthState.INSUFFICIENT_EVIDENCE,
        measured_at=None,
        value=None,
        reason="not_collected",
    )
    assert check.state is HealthState.INSUFFICIENT_EVIDENCE


def test_a_state_other_than_healthy_needs_a_reason():
    with pytest.raises(SummaryRefused, match="without a reason"):
        make_check(state=HealthState.DEGRADED, reason=None)


@pytest.mark.parametrize(
    "fields",
    [
        {"check_id": "Tests"},
        {"check_id": "deps.tests", "area": Area.CI},  # the prefix must name the area
        {"value": float("nan")},
        {"value": True},
        {"summary": "  "},
        {"detail": "bad\x00text"},
        {"reason": "Not A Code"},
        {"runbook": "README.md#ci-tests"},
        {"measured_at": datetime(2026, 10, 14, 12, tzinfo=timezone(timedelta(hours=2)))},
    ],
)
def test_invalid_check_fields_are_refused(fields):
    with pytest.raises(SummaryRefused):
        make_check(**fields)


def test_the_runbook_anchor_is_derived_from_the_check_id():
    check = make_check("deps.pip_check", title="pip check")
    assert check.runbook == "docs/operations/runbook.md#deps-pip-check"


def test_alerts_are_informational_only():
    with pytest.raises(SummaryRefused) as refused:
        Alert(
            kind=AlertKind.FAILURE,
            check_id="ci.tests",
            state=HealthState.FAILED,
            message="x",
            raised_at=NOW,
            automatic_action="rollback",
        )
    assert refused.value.reason is SummaryRefusal.POLICY_VIOLATION


def test_run_context_accepts_only_allowlisted_keys():
    with pytest.raises(SummaryRefused, match="run_context"):
        HealthSummary(
            generated_at=NOW,
            code_sha=None,
            code_dirty=None,
            scope=Scope.SYNTHETIC,
            run_context={"environment": "github_actions", "GITHUB_TOKEN": "ghs_x"},
            checks=(),
        )


def test_duplicate_check_ids_are_refused():
    with pytest.raises(SummaryRefused) as refused:
        make_summary(make_check(), make_check())
    assert refused.value.reason is SummaryRefusal.DUPLICATE_CHECK


def _round_trip_document():
    failing = make_check("deps.pins", HealthState.FAILED, title="Pins", reason="pin_mismatch")
    alert = Alert(AlertKind.FAILURE, "deps.pins", HealthState.FAILED, "Pins failed.", NOW)
    return make_summary(make_check(), failing, alerts=(alert,)).to_json_dict()


def test_a_summary_round_trips_through_json(tmp_path):
    document = _round_trip_document()
    path = tmp_path / "summary.json"
    path.write_text(model.dumps(document), encoding="utf-8")
    summary = read_summary(path)
    assert summary.to_json_dict() == document
    assert document["overall_state"] == "failed"
    assert document["state_counts"]["failed"] == 1
    assert document["alert_policy"] == "informational_only"
    assert document["schema_version"] == "1.0"
    assert document["generated_at"] == "2026-10-14T12:00:00.000000Z"
    assert set(document["checks"][0]) == model.CHECK_FIELDS


def _refusal(tmp_path, text: str) -> SummaryRefusal:
    path = tmp_path / "summary.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SummaryRefused) as refused:
        read_summary(path)
    return refused.value.reason


def _mutated(change) -> str:
    document = _round_trip_document()
    change(document)
    return json.dumps(document)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda d: d.update(format="other"), SummaryRefusal.NOT_A_SUMMARY),
        (lambda d: d.update(schema_version="2.0"), SummaryRefusal.UNSUPPORTED_VERSION),
        (lambda d: d.update(extra=1), SummaryRefusal.FIELD_SET),
        (lambda d: d.pop("alerts"), SummaryRefusal.FIELD_SET),
        (lambda d: d.update(overall_state="healthy"), SummaryRefusal.STATE_MISMATCH),
        (lambda d: d["state_counts"].update(failed=0), SummaryRefusal.STATE_MISMATCH),
        (lambda d: d.update(alert_policy="act"), SummaryRefusal.POLICY_VIOLATION),
        (
            lambda d: d["alerts"][0].update(automatic_action="trade"),
            SummaryRefusal.POLICY_VIOLATION,
        ),
        (lambda d: d["alerts"][0].update(alert_id="other"), SummaryRefusal.INVALID_FIELD),
        (
            lambda d: d.update(precedence=list(reversed(d["precedence"]))),
            SummaryRefusal.INVALID_FIELD,
        ),
        (lambda d: d["checks"][0].update(state="green"), SummaryRefusal.INVALID_FIELD),
        (lambda d: d["checks"][0].update(measured_at=None), SummaryRefusal.POLICY_VIOLATION),
        (lambda d: d["checks"][0].pop("evidence"), SummaryRefusal.FIELD_SET),
        (lambda d: d.update(generated_at="2026-10-14 12:00"), SummaryRefusal.INVALID_FIELD),
        (lambda d: d.update(code_sha="abc"), SummaryRefusal.INVALID_FIELD),
    ],
)
def test_the_reader_refuses_any_defect(tmp_path, change, reason):
    assert _refusal(tmp_path, _mutated(change)) is reason


@pytest.mark.parametrize(
    "text",
    [
        '{"format": NaN}',
        '{"format": "a", "format": "b"}',
        "not json",
    ],
)
def test_the_reader_refuses_non_strict_json(tmp_path, text):
    assert _refusal(tmp_path, text) is SummaryRefusal.INVALID_JSON


def test_a_missing_file_is_refused_as_not_found(tmp_path):
    with pytest.raises(SummaryRefused) as refused:
        read_summary(tmp_path / "absent.json")
    assert refused.value.reason is SummaryRefusal.NOT_FOUND


def test_not_applicable_entries_never_count_and_never_repeat_a_check():
    item = model.NotApplicable(
        "forecast.coverage", Area.FORECAST, "Coverage", "point_forecast", "point forecasts only"
    )
    summary = HealthSummary(
        generated_at=NOW,
        code_sha=None,
        code_dirty=None,
        scope=Scope.PRIVATE,
        run_context={"environment": "local"},
        checks=(make_check(),),
        not_applicable=(item,),
    )
    assert summary.overall_state is HealthState.HEALTHY
    assert summary.to_json_dict()["not_applicable"][0]["reason"] == "point_forecast"
    assert HealthSummary.from_json_dict(summary.to_json_dict()) == summary
    with pytest.raises(SummaryRefused) as refused:
        HealthSummary(
            generated_at=NOW,
            code_sha=None,
            code_dirty=None,
            scope=Scope.PRIVATE,
            run_context={"environment": "local"},
            checks=(make_check("forecast.coverage", title="Coverage"),),
            not_applicable=(item,),
        )
    assert refused.value.reason is SummaryRefusal.DUPLICATE_CHECK


# --- operator conditions --------------------------------------------------------------


def test_conditions_follow_the_state_and_split_stale_and_not_activated():
    from stock_agent.health.model import CONDITION_PRECEDENCE, Condition, condition

    assert condition(HealthState.DEGRADED, "stale_data") is Condition.STALE
    assert condition(HealthState.DEGRADED, "low_coverage") is Condition.DEGRADED
    assert condition(HealthState.INSUFFICIENT_EVIDENCE, "not_authorized") is Condition.NOT_ACTIVATED
    assert condition(HealthState.INSUFFICIENT_EVIDENCE, "input_missing") is Condition.NO_EVIDENCE
    # A stale or not-authorized reason never softens a failure or an unavailable measurement.
    assert condition(HealthState.FAILED, "stale_data") is Condition.FAILED
    assert condition(HealthState.UNAVAILABLE, "not_authorized") is Condition.UNAVAILABLE
    assert condition(HealthState.HEALTHY, None) is Condition.HEALTHY
    # Every condition of a worse state ranks worse, so worst-of keeps the state precedence.
    state_of = {
        Condition.STALE: HealthState.DEGRADED,
        Condition.NO_EVIDENCE: HealthState.INSUFFICIENT_EVIDENCE,
        Condition.NOT_ACTIVATED: HealthState.INSUFFICIENT_EVIDENCE,
    }
    ranks = [
        PRECEDENCE.index(state_of.get(item) or HealthState(item.value))
        for item in CONDITION_PRECEDENCE
    ]
    assert ranks == sorted(ranks) and list(CONDITION_PRECEDENCE) == list(Condition)


def test_every_stale_reason_the_checks_emit_is_a_stale_condition():
    import re
    from pathlib import Path

    from stock_agent.health.model import NOT_ACTIVATED_REASONS, STALE_REASONS

    package = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "health"
    emitted = set()
    for path in package.glob("*.py"):
        emitted |= set(re.findall(r'"(stale_[a-z_]+)"', path.read_text(encoding="utf-8")))
    assert emitted and emitted <= STALE_REASONS
    assert "daily_run_overdue" in STALE_REASONS and NOT_ACTIVATED_REASONS == {"not_authorized"}
