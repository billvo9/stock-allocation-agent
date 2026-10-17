"""
Data-quality monitors on synthetic fund stores: the latest permitted session
(capped at the lockbox, never today), freshness, coverage, missing sessions
and cross-source discrepancies, each with its thresholds; and the monitors
never modify the store. Plus the smoke check that runs them on planted
defects.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from stock_agent.health import data_quality, smoke
from stock_agent.health.data_quality import data_quality_checks, latest_permitted_session
from stock_agent.health.model import HealthState

AS_OF = smoke.SMOKE_AS_OF
NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def run(store, as_of=AS_OF) -> dict:
    checks = data_quality_checks(store, as_of=as_of, measured_at=NOW, evidence="store")
    return {check.check_id: check for check in checks}


def states(checks: dict) -> dict:
    return {key: (check.state, check.value) for key, check in checks.items()}


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        ("2025-01-02T12:00:00", "2024-12-31"),  # capped at the lockbox
        ("2030-06-01T00:00:00", "2024-12-31"),  # never today
        ("2024-07-05T10:00:00", "2024-07-03"),  # before the as-of date; July 4 closed
        ("2024-07-08T00:00:00", "2024-07-05"),
        ("2003-01-02T23:00:00", None),  # nothing complete before coverage starts
    ],
)
def test_latest_permitted_session(as_of, expected):
    result = latest_permitted_session(utc(as_of))
    assert result == (None if expected is None else utc(expected))


def test_a_clean_two_source_store_is_healthy():
    assert states(run(smoke.synthetic_store())) == {
        "data.freshness": (HealthState.HEALTHY, 0),
        "data.coverage": (HealthState.HEALTHY, 1.0),
        "data.missing_sessions": (HealthState.HEALTHY, 0),
        "data.vendor_discrepancy": (HealthState.HEALTHY, 0),
    }


def test_planted_defects_are_each_reported():
    checks = run(smoke.synthetic_store(drop_tail=3, bumped=2))
    assert states(checks) == {
        "data.freshness": (HealthState.DEGRADED, 3),
        "data.coverage": (HealthState.DEGRADED, round(1 - 3 / 192, 6)),
        "data.missing_sessions": (HealthState.DEGRADED, 3),
        "data.vendor_discrepancy": (HealthState.DEGRADED, 2),
    }
    assert "SYN_CASH last close 2024-12-26 (3 behind)" in checks["data.freshness"].detail
    assert "SYN_CASH 2024-12-27" in checks["data.missing_sessions"].detail
    assert "SYN_EQ 2024-10-01 unadjusted 1.000%" in checks["data.vendor_discrepancy"].detail


def test_a_long_gap_is_a_broken_feed():
    checks = run(smoke.synthetic_store(drop_tail=6))
    assert (checks["data.missing_sessions"].state, checks["data.missing_sessions"].reason) == (
        HealthState.FAILED,
        "missing_run",
    )
    assert checks["data.coverage"].state is HealthState.DEGRADED  # 1 - 6/192 = 96.9 %


def test_coverage_below_the_floor_fails():
    checks = run(smoke.synthetic_store(drop_tail=20))
    assert (checks["data.coverage"].state, checks["data.coverage"].value) == (
        HealthState.FAILED,
        round(1 - 20 / 192, 6),
    )


def test_one_source_is_insufficient_evidence_for_discrepancies():
    store = smoke.synthetic_store()
    single = type(store)(
        batches=store.batches,
        prices=store.prices[store.prices["source_id"] != smoke.SHADOW_SOURCE],
        distributions=store.distributions,
        splits=store.splits,
        missing_sessions=store.missing_sessions,
    )
    check = run(single)["data.vendor_discrepancy"]
    assert (check.state, check.reason, check.value) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "single_source",
        None,
    )


def _without(store, fund_id):
    return type(store)(
        batches=store.batches,
        prices=store.prices[store.prices["fund_id"] != fund_id],
        distributions=store.distributions,
        splits=store.splits,
        missing_sessions=store.missing_sessions,
    )


def test_a_requested_fund_without_any_close_is_degraded():
    checks = run(_without(smoke.synthetic_store(), "SYN_BOND"))
    assert (checks["data.freshness"].state, checks["data.freshness"].reason) == (
        HealthState.DEGRADED,
        "fund_without_closes",
    )
    assert checks["data.missing_sessions"].state is HealthState.FAILED


def test_an_empty_store_is_unavailable_not_healthy():
    store = smoke.synthetic_store()
    empty = type(store)(
        batches=store.batches,
        prices=store.prices.iloc[:0],
        distributions=store.distributions,
        splits=store.splits,
        missing_sessions=store.missing_sessions,
    )
    checks = run(empty)
    assert checks["data.freshness"].state is HealthState.UNAVAILABLE
    assert checks["data.coverage"].state is HealthState.FAILED


def test_a_batch_retrieved_after_the_as_of_time_is_not_known_then():
    store = smoke.synthetic_store(bumped=2)
    between = smoke.FUND_RETRIEVED_AT + pd.Timedelta(minutes=30)  # before the second source
    checks = run(store, as_of=between)
    assert (checks["data.vendor_discrepancy"].state, checks["data.vendor_discrepancy"].reason) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "single_source",
    )
    assert (
        "1 batches retrieved after the as-of time were excluded" in checks["data.coverage"].detail
    )
    before_any = run(store, as_of=smoke.FUND_RETRIEVED_AT - pd.Timedelta(minutes=1))
    assert before_any["data.coverage"].state is HealthState.INSUFFICIENT_EVIDENCE
    assert before_any["data.freshness"].state is HealthState.UNAVAILABLE


def test_the_monitors_never_modify_the_store():
    store = smoke.synthetic_store(drop_tail=3, bumped=2)
    before = {name: getattr(store, name).copy() for name in ("batches", "prices")}
    run(store)
    for name, frame in before.items():
        pd.testing.assert_frame_equal(getattr(store, name), frame)


def test_not_collected_checks_cover_all_four():
    checks = data_quality.not_collected_checks(
        reason="input_not_configured", state=HealthState.INSUFFICIENT_EVIDENCE, detail="none"
    )
    assert [check.check_id for check in checks] == list(data_quality.CHECK_IDS)
    assert all(check.value is None and check.measured_at is None for check in checks)


def test_the_smoke_checks_pass_on_the_current_monitors():
    checks = {check.check_id: check for check in smoke.smoke_checks(now=lambda: NOW)}
    assert (checks["smoke.data_monitors"].state, checks["smoke.data_monitors"].value) == (
        HealthState.HEALTHY,
        8,
    )
    assert (checks["smoke.input_readers"].state, checks["smoke.input_readers"].value) == (
        HealthState.HEALTHY,
        11,
    )


def test_a_monitor_that_stops_detecting_fails_the_smoke_check(monkeypatch):
    monkeypatch.setattr(data_quality, "DISCREPANCY_TOLERANCE", 0.5)
    check = smoke.smoke_checks(now=lambda: NOW)[0]
    assert (check.state, check.reason, check.value) == (
        HealthState.FAILED,
        "monitor_misreported",
        7,
    )
    assert "[WRONG] planted data.vendor_discrepancy" in check.detail
