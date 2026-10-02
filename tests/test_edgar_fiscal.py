import pytest

from stock_agent.data.fundamentals.edgar_fiscal import (
    EdgarFiscalIdentityError,
    infer_edgar_fiscal_identity,
)


def test_infer_q3_fiscal_identity():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2026-02-27",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
        },
        {
            "type": "duration",
            "start_date": "2025-08-29",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "YTD9",
        },
    ]

    result = infer_edgar_fiscal_identity(
        form="10-Q",
        period_end="2026-05-28",
        reporting_periods=reporting_periods,
    )

    assert result.fiscal_year == 2026
    assert result.fiscal_period == "Q3"
    assert result.fiscal_quarter == 3


def test_infer_10k_as_fourth_fiscal_quarter():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2024-08-30",
            "end_date": "2025-08-28",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
        },
        {
            "type": "duration",
            "start_date": "2025-05-30",
            "end_date": "2025-08-28",
            "fiscal_year": 2025,
            "fiscal_period": "Q4",
        },
    ]

    result = infer_edgar_fiscal_identity(
        form="10-K",
        period_end="2025-08-28",
        reporting_periods=reporting_periods,
    )

    assert result.fiscal_year == 2025
    assert result.fiscal_period == "FY"
    assert result.fiscal_quarter == 4


def test_ignores_reporting_periods_for_other_period_end_dates():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2025-11-28",
            "end_date": "2026-02-26",
            "fiscal_year": 2026,
            "fiscal_period": "Q2",
        },
        {
            "type": "duration",
            "start_date": "2026-02-27",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
        },
    ]

    result = infer_edgar_fiscal_identity(
        form="10-Q",
        period_end="2026-05-28",
        reporting_periods=reporting_periods,
    )

    assert result.fiscal_year == 2026
    assert result.fiscal_period == "Q3"
    assert result.fiscal_quarter == 3


def test_duplicate_identical_fiscal_metadata_is_safe():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2026-02-27",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
        },
        {
            "type": "duration",
            "start_date": "2026-02-27",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
        },
    ]

    result = infer_edgar_fiscal_identity(
        form="10-Q/A",
        period_end="2026-05-28",
        reporting_periods=reporting_periods,
    )

    assert result.fiscal_year == 2026
    assert result.fiscal_period == "Q3"
    assert result.fiscal_quarter == 3


def test_conflicting_matching_fiscal_metadata_raises():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2025-11-28",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q2",
        },
        {
            "type": "duration",
            "start_date": "2026-02-27",
            "end_date": "2026-05-28",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
        },
    ]

    with pytest.raises(
        EdgarFiscalIdentityError,
        match="Conflicting EDGAR fiscal identities",
    ):
        infer_edgar_fiscal_identity(
            form="10-Q",
            period_end="2026-05-28",
            reporting_periods=reporting_periods,
        )


def test_missing_matching_fiscal_metadata_raises():
    reporting_periods = [
        {
            "type": "duration",
            "start_date": "2025-11-28",
            "end_date": "2026-02-26",
            "fiscal_year": 2026,
            "fiscal_period": "Q2",
        }
    ]

    with pytest.raises(
        EdgarFiscalIdentityError,
        match="Unable to determine EDGAR fiscal identity",
    ):
        infer_edgar_fiscal_identity(
            form="10-Q",
            period_end="2026-05-28",
            reporting_periods=reporting_periods,
        )
