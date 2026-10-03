from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_reconciliation import (
    EdgarReconciliationError,
    FiscalFlowObservation,
    is_valid_prior_fiscal_observation,
    reconcile_fiscal_flow,
)


def _observation(
    *,
    symbol: str = "MU",
    metric_name: str = "operating_cash_flow",
    fiscal_year: int = 2026,
    fiscal_quarter: int = 3,
    period_end: str = "2026-05-28",
    available_at: str = "2026-06-24 22:59:46",
    accession_number: str = "Q3",
    direct_value: float | None = None,
    ytd_value: float | None = None,
    fy_value: float | None = None,
) -> FiscalFlowObservation:
    return FiscalFlowObservation(
        symbol=symbol,
        metric_name=metric_name,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        period_end=pd.Timestamp(
            period_end,
            tz="UTC",
        ),
        available_at=pd.Timestamp(
            available_at,
            tz="UTC",
        ),
        accession_number=accession_number,
        direct_value=direct_value,
        ytd_value=ytd_value,
        fy_value=fy_value,
    )


def test_direct_quarter_value_wins():
    current = _observation(
        direct_value=25.0,
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "direct_quarter"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q3"
    assert result.prior_accession_number is None


def test_q1_uses_current_ytd():
    current = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
    )

    assert result.value == pytest.approx(20.0)
    assert result.method == "q1_ytd"
    assert result.reason == "ok"


def test_q2_derives_from_prior_q1_ytd():
    prior = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    current = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "ytd_minus_prior_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q2"
    assert result.prior_accession_number == "Q1"


def test_q3_derives_from_prior_q2_ytd():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "ytd_minus_prior_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q3"
    assert result.prior_accession_number == "Q2"


def test_q4_derives_from_fy_minus_q3_ytd():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=3,
        period_end="2025-05-29",
        available_at="2025-06-25 22:50:42",
        accession_number="FY2025-Q3",
        ytd_value=72.0,
    )

    current = _observation(
        fiscal_year=2025,
        fiscal_quarter=4,
        period_end="2025-08-28",
        available_at="2025-10-03 18:42:25",
        accession_number="FY2025-10K",
        fy_value=100.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(28.0)
    assert result.method == "fy_minus_q3_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "FY2025-10K"
    assert result.prior_accession_number == "FY2025-Q3"


def test_prior_from_different_fiscal_year_is_rejected():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=2,
        period_end="2025-02-27",
        available_at="2025-03-20 23:20:23",
        accession_number="FY2025-Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_year=2026,
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="FY2026-Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "different_fiscal_year"


def test_wrong_prior_fiscal_quarter_is_rejected():
    prior = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "wrong_prior_fiscal_quarter"


def test_future_prior_observation_is_rejected():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-07-01 12:00:00",
        accession_number="Q2-AMENDMENT",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "prior_available_after_current"


def test_prior_period_end_must_be_before_current_period_end():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-06-30",
        available_at="2026-06-20 12:00:00",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "prior_period_not_before_current"


def test_missing_prior_ytd_returns_missing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=None,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "missing_prior_ytd"


def test_missing_current_ytd_returns_missing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=None,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "missing_current_ytd"


def test_missing_fy_for_q4_returns_missing():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=3,
        period_end="2025-05-29",
        available_at="2025-06-25 22:50:42",
        accession_number="Q3",
        ytd_value=72.0,
    )

    current = _observation(
        fiscal_year=2025,
        fiscal_quarter=4,
        period_end="2025-08-28",
        available_at="2025-10-03 18:42:25",
        accession_number="10K",
        fy_value=None,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "missing_current_fy"


def test_different_symbol_is_rejected():
    prior = _observation(
        symbol="NVDA",
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="NVDA-Q2",
        ytd_value=45.0,
    )

    current = _observation(
        symbol="MU",
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="MU-Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "different_symbol"


def test_different_metric_is_rejected():
    prior = _observation(
        metric_name="revenue",
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        metric_name="operating_cash_flow",
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "different_metric"


def test_same_accession_cannot_be_used_as_prior_filing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="SAME",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="SAME",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "same_accession_number"


def test_valid_prior_helper_returns_true_for_same_fiscal_sequence():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    assert is_valid_prior_fiscal_observation(
        current=current,
        prior=prior,
    )


def test_invalid_fiscal_quarter_raises():
    current = _observation(
        fiscal_quarter=5,
    )

    with pytest.raises(
        EdgarReconciliationError,
        match="fiscal_quarter must be between 1 and 4",
    ):
        reconcile_fiscal_flow(
            current=current,
        )
