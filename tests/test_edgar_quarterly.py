from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_quarterly import (
    EdgarQuarterlyBuildError,
    build_edgar_quarter,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)

PERIOD_END = "2026-05-28"


def _income_statement(
    *,
    include_gross_profit: bool = True,
) -> pd.DataFrame:
    rows = [
        {
            "concept": ("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"),
            "standard_concept": "Revenue",
            "dimension": False,
            "2026-05-28 (Q3)": 100.0,
            "2026-05-28 (YTD)": 270.0,
            "2026-02-26 (YTD)": 170.0,
        },
        {
            "concept": "us-gaap_OperatingIncomeLoss",
            "standard_concept": "OperatingIncomeLoss",
            "dimension": False,
            "2026-05-28 (Q3)": 30.0,
            "2026-05-28 (YTD)": 75.0,
            "2026-02-26 (YTD)": 45.0,
        },
        {
            "concept": "us-gaap_NetIncomeLoss",
            "standard_concept": "NetIncome",
            "dimension": False,
            "2026-05-28 (Q3)": 20.0,
            "2026-05-28 (YTD)": 50.0,
            "2026-02-26 (YTD)": 30.0,
        },
    ]

    if include_gross_profit:
        rows.insert(
            1,
            {
                "concept": "us-gaap_GrossProfit",
                "standard_concept": "GrossProfit",
                "dimension": False,
                "2026-05-28 (Q3)": 40.0,
                "2026-05-28 (YTD)": 105.0,
                "2026-02-26 (YTD)": 65.0,
            },
        )

    return pd.DataFrame(rows)


def _cash_flow_statement() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "concept": ("us-gaap_NetCashProvidedByUsedInOperatingActivities"),
                "standard_concept": ("NetCashFromOperatingActivities"),
                "dimension": False,
                # No direct Q3 value on purpose.
                # The assembler must derive:
                #
                # 70 - 45 = 25
                "2026-05-28 (YTD)": 70.0,
                "2026-02-26 (YTD)": 45.0,
            }
        ]
    )


def _build(
    *,
    income_statement: pd.DataFrame | None = None,
    cash_flow_statement: pd.DataFrame | None = None,
    accepted_at: str | None = "2026-06-25T21:15:00Z",
    fiscal_quarter: int = 3,
) -> pd.DataFrame:
    if income_statement is None:
        income_statement = _income_statement()

    if cash_flow_statement is None:
        cash_flow_statement = _cash_flow_statement()

    return build_edgar_quarter(
        symbol="MU",
        provider_symbol="MU",
        period_end=PERIOD_END,
        fiscal_quarter=fiscal_quarter,
        filing_date="2026-06-25",
        accepted_at=accepted_at,
        retrieved_at="2026-09-26T17:00:00Z",
        sec_form_type="10-Q",
        sec_accession_number="0000723125-26-000015",
        income_statement=income_statement,
        cash_flow_statement=cash_flow_statement,
    )


def test_build_edgar_quarter_returns_canonical_columns():
    result = _build()

    assert result.columns.tolist() == QUARTERLY_FUNDAMENTAL_COLUMNS

    assert len(result) == 1


def test_build_edgar_quarter_maps_core_income_metrics():
    result = _build()

    row = result.iloc[0]

    assert row["revenue"] == pytest.approx(100.0)
    assert row["gross_profit"] == pytest.approx(40.0)
    assert row["operating_income"] == pytest.approx(30.0)
    assert row["net_income"] == pytest.approx(20.0)


def test_build_edgar_quarter_derives_cash_flow_from_ytd():
    result = _build()

    row = result.iloc[0]

    # Q3 standalone operating cash flow:
    #
    # 70 YTD - 45 previous YTD = 25
    assert row["operating_cash_flow"] == pytest.approx(25.0)


def test_build_edgar_quarter_preserves_missing_optional_metric():
    result = _build(
        income_statement=_income_statement(
            include_gross_profit=False,
        )
    )

    row = result.iloc[0]

    assert row["revenue"] == pytest.approx(100.0)
    assert pd.isna(row["gross_profit"])
    assert row["net_income"] == pytest.approx(20.0)


def test_missing_cash_flow_statement_does_not_crash():
    result = build_edgar_quarter(
        symbol="MU",
        provider_symbol="MU",
        period_end=PERIOD_END,
        fiscal_quarter=3,
        filing_date="2026-06-25",
        accepted_at="2026-06-25T21:15:00Z",
        retrieved_at="2026-09-26T17:00:00Z",
        sec_form_type="10-Q",
        sec_accession_number="0000723125-26-000015",
        income_statement=_income_statement(),
        cash_flow_statement=None,
    )

    row = result.iloc[0]

    assert row["revenue"] == pytest.approx(100.0)
    assert pd.isna(row["operating_cash_flow"])


def test_build_edgar_quarter_uses_acceptance_timestamp():
    result = _build()

    row = result.iloc[0]

    assert row["available_at"] == pd.Timestamp("2026-06-25T21:15:00Z")

    assert row["availability_source"] == "sec_acceptance_datetime"


def test_build_edgar_quarter_uses_conservative_filing_fallback():
    result = _build(
        accepted_at=None,
    )

    row = result.iloc[0]

    assert row["available_at"] == pd.Timestamp("2026-06-26T00:00:00Z")

    assert row["availability_source"] == "sec_filing_date_plus_1d"


def test_build_edgar_quarter_preserves_provenance():
    result = _build()

    row = result.iloc[0]

    assert row["symbol"] == "MU"
    assert row["provider_symbol"] == "MU"
    assert row["sec_form_type"] == "10-Q"

    assert row["sec_accession_number"] == "0000723125-26-000015"

    assert row["source"] == "edgar"
    assert row["currency"] == "USD"


def test_build_edgar_quarter_rejects_invalid_fiscal_quarter():
    with pytest.raises(
        EdgarQuarterlyBuildError,
        match="fiscal_quarter",
    ):
        _build(
            fiscal_quarter=5,
        )


def test_build_edgar_quarter_rejects_ambiguous_concept():
    income = _income_statement()

    duplicate_revenue = income.iloc[[0]].copy()

    duplicate_revenue["concept"] = "us-gaap_SalesRevenueNet"

    income = pd.concat(
        [
            income,
            duplicate_revenue,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        EdgarQuarterlyBuildError,
        match="Ambiguous EDGAR concept resolution",
    ):
        _build(
            income_statement=income,
        )


def test_build_edgar_quarter_rejects_completely_missing_statements():
    with pytest.raises(
        EdgarQuarterlyBuildError,
        match="No usable EDGAR statements",
    ):
        build_edgar_quarter(
            symbol="MU",
            provider_symbol="MU",
            period_end=PERIOD_END,
            fiscal_quarter=3,
            filing_date="2026-06-25",
            accepted_at="2026-06-25T21:15:00Z",
            retrieved_at="2026-09-26T17:00:00Z",
            sec_form_type="10-Q",
            sec_accession_number=("0000723125-26-000015"),
            income_statement=None,
            cash_flow_statement=None,
        )
