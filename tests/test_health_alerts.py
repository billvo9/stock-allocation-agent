"""
Informational alerts: one per failed, unavailable or degraded check, a
constraint violation for a breached stored limit, deterioration against the
previous summary only beyond the check's tolerance, and no alert while a
producer has not delivered yet.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_agent.health.alerts import derive_alerts
from stock_agent.health.context import CodeState
from stock_agent.health.model import (
    AUTOMATIC_ACTION,
    CONSTRAINT_BREACHED,
    AlertKind,
    Area,
    Better,
    Check,
    HealthState,
    Scope,
)
from stock_agent.health.summary import build_summary

NOW = datetime(2026, 10, 14, 12, tzinfo=UTC)
BEFORE = NOW - timedelta(days=1)


def check(check_id="data.coverage", state=HealthState.HEALTHY, value=1.0, **fields) -> Check:
    base = {
        "area": Area(check_id.split(".")[0]),
        "title": "Session coverage",
        "measured_at": NOW,
        "unit": "fraction",
        "threshold": ">= 0.995",
        "summary": "coverage",
        "detail": "detail",
        "evidence": "store",
        "better": Better.HIGHER,
        "tolerance": 0.0005,
        "reason": None if state is HealthState.HEALTHY else "low_coverage",
    }
    return Check(check_id=check_id, state=state, value=value, **{**base, **fields})


def previous(*checks: Check):
    return build_summary(
        checks,
        generated_at=BEFORE,
        code=CodeState(None, None),
        scope=Scope.PRIVATE,
        run_context={"environment": "local"},
    )


def kinds(alerts):
    return [(alert.kind, alert.check_id) for alert in alerts]


def test_each_bad_state_raises_its_alert_and_nothing_acts():
    checks = [
        check("ci.tests", HealthState.FAILED, 3, reason="tests_failed"),
        check("data.freshness", HealthState.UNAVAILABLE, None, reason="no_prices"),
        check("data.coverage", HealthState.DEGRADED, 0.98),
        check(
            "training.health",
            HealthState.INSUFFICIENT_EVIDENCE,
            None,
            measured_at=None,
            reason="input_missing",
        ),
        check("data.missing_sessions", HealthState.HEALTHY, 0),
    ]
    alerts = derive_alerts(checks, raised_at=NOW)
    assert kinds(alerts) == [
        (AlertKind.FAILURE, "ci.tests"),
        (AlertKind.UNAVAILABLE_DATA, "data.freshness"),
        (AlertKind.DEGRADED, "data.coverage"),
    ]
    assert all(alert.automatic_action == AUTOMATIC_ACTION == "none" for alert in alerts)


def test_a_breached_limit_is_a_constraint_violation():
    breach = check("portfolio.concentration", HealthState.FAILED, 0.55, reason=CONSTRAINT_BREACHED)
    assert kinds(derive_alerts([breach], raised_at=NOW)) == [
        (AlertKind.CONSTRAINT_VIOLATION, "portfolio.concentration")
    ]


def test_deterioration_beyond_tolerance_within_the_healthy_band_alerts():
    alerts = derive_alerts([check(value=0.998)], raised_at=NOW, previous=previous(check(value=1.0)))
    assert kinds(alerts) == [(AlertKind.DETERIORATION, "data.coverage")]
    assert "from 1.0 to 0.998" in alerts[0].message


def test_movement_within_tolerance_or_in_the_better_direction_does_not_alert():
    within = derive_alerts([check(value=0.9996)], raised_at=NOW, previous=previous(check()))
    better = derive_alerts([check()], raised_at=NOW, previous=previous(check(value=0.999)))
    assert within == better == ()


def test_lower_is_better_values_deteriorate_upwards():
    before = check("data.missing_sessions", value=0, better=Better.LOWER, tolerance=0)
    now = check("data.missing_sessions", value=1, better=Better.LOWER, tolerance=0)
    assert kinds(derive_alerts([now], raised_at=NOW, previous=previous(before))) == [
        (AlertKind.DETERIORATION, "data.missing_sessions")
    ]


def test_a_changed_unit_is_not_compared():
    before = check(value=4, unit="of 4 gates passed", tolerance=0)
    now = check(value=3, unit="of 3 gates passed", tolerance=0)
    assert derive_alerts([now], raised_at=NOW, previous=previous(before)) == ()


def test_a_state_alert_says_whether_it_is_new():
    degraded = check(state=HealthState.DEGRADED, value=0.98)
    new = derive_alerts([degraded], raised_at=NOW, previous=previous(check()))
    again = derive_alerts([degraded], raised_at=NOW, previous=previous(degraded))
    assert "new: healthy in the previous summary" in new[0].message
    assert "also degraded in the previous summary" in again[0].message
    # A worse state is reported once, as the state alert, not again as deterioration.
    assert kinds(new) == [(AlertKind.DEGRADED, "data.coverage")]
