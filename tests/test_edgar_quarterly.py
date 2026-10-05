from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_quarterly import (
    FLOW_METRIC_STATEMENTS,
    EdgarQuarterlyBuildError,
    build_edgar_quarter,
    extract_fiscal_flow_observations,
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


def test_build_edgar_quarter_defers_cross_filing_ytd_derivation():
    result = _build()

    row = result.iloc[0]

    # Q2/Q3 standalone flows that require a previous filing
    # are deliberately deferred to the history layer.
    assert pd.isna(row["operating_cash_flow"])


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


# ---------------------------------------------------------------------------
# Single-filing fiscal-flow observation extraction (E1-E5)
# ---------------------------------------------------------------------------

AVAILABLE_AT = pd.Timestamp("2026-06-25 21:15", tz="UTC")


def _extract(
    *,
    income_statement: pd.DataFrame | None = None,
    cash_flow_statement: pd.DataFrame | None = None,
    period_end: str = PERIOD_END,
    fiscal_year: int = 2026,
    fiscal_quarter: int = 3,
):
    observations = extract_fiscal_flow_observations(
        symbol="MU",
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        period_end=period_end,
        available_at=AVAILABLE_AT,
        accession_number="ACC",
        income_statement=income_statement,
        cash_flow_statement=cash_flow_statement,
    )

    return {observation.metric_name: observation for observation in observations}


def test_extraction_emits_one_observation_per_flow_metric():
    observations = _extract(
        income_statement=_income_statement(),
        cash_flow_statement=_cash_flow_statement(),
    )

    assert sorted(observations) == sorted(FLOW_METRIC_STATEMENTS)

    for observation in observations.values():
        assert observation.symbol == "MU"
        assert observation.fiscal_year == 2026
        assert observation.fiscal_quarter == 3
        assert observation.period_end == pd.Timestamp(PERIOD_END, tz="UTC")
        assert observation.available_at == AVAILABLE_AT
        assert observation.accession_number == "ACC"


def test_e1_quarter_filing_with_direct_and_ytd_carries_both():
    revenue = _extract(income_statement=_income_statement())["revenue"]

    assert revenue.direct_value == pytest.approx(100.0)
    assert revenue.ytd_value == pytest.approx(270.0)
    assert revenue.fy_value is None


def test_e2_ytd_only_quarter_filing_has_no_direct_value():
    cash_flow = _extract(cash_flow_statement=_cash_flow_statement())["operating_cash_flow"]

    assert cash_flow.direct_value is None
    assert cash_flow.ytd_value == pytest.approx(70.0)
    assert cash_flow.fy_value is None


def test_extraction_never_uses_another_period_column():
    # The statement reports only the PRIOR quarter's YTD (2026-02-26).
    # The 2026-05-28 observation must stay empty rather than pick it up.
    cash_flow_statement = pd.DataFrame(
        [
            {
                "concept": "us-gaap_NetCashProvidedByUsedInOperatingActivities",
                "standard_concept": "NetCashFromOperatingActivities",
                "dimension": False,
                "2026-02-26 (YTD)": 45.0,
                "2026-02-26 (Q2)": 25.0,
            }
        ]
    )

    cash_flow = _extract(cash_flow_statement=cash_flow_statement)["operating_cash_flow"]

    assert cash_flow.direct_value is None
    assert cash_flow.ytd_value is None
    assert cash_flow.fy_value is None


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_extraction_rejects_non_finite_values(bad_value):
    cash_flow_statement = _cash_flow_statement()
    cash_flow_statement["2026-05-28 (YTD)"] = bad_value

    with pytest.raises(EdgarQuarterlyBuildError, match="non-finite"):
        _extract(cash_flow_statement=cash_flow_statement)


def test_e3_annual_filing_with_fy_only():
    income = pd.DataFrame(
        [
            {
                "concept": "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
                "standard_concept": "Revenue",
                "dimension": False,
                "2026-08-27 (FY)": 370.0,
            }
        ]
    )

    revenue = _extract(
        income_statement=income,
        period_end="2026-08-27",
        fiscal_quarter=4,
    )["revenue"]

    assert revenue.fiscal_quarter == 4
    assert revenue.direct_value is None
    assert revenue.ytd_value is None
    assert revenue.fy_value == pytest.approx(370.0)


def test_e4_missing_concept_or_statement_yields_empty_observation():
    observations = _extract(
        income_statement=_income_statement(include_gross_profit=False),
        cash_flow_statement=None,
    )

    for metric_name in ("gross_profit", "operating_cash_flow"):
        observation = observations[metric_name]

        assert observation.direct_value is None
        assert observation.ytd_value is None
        assert observation.fy_value is None


def test_e5_ambiguous_concept_raises():
    income = _income_statement()

    duplicate_revenue = income.iloc[[0]].copy()
    duplicate_revenue["concept"] = "us-gaap_SalesRevenueNet"

    income = pd.concat(
        [income, duplicate_revenue],
        ignore_index=True,
    )

    with pytest.raises(
        EdgarQuarterlyBuildError,
        match="Ambiguous EDGAR concept resolution",
    ):
        _extract(income_statement=income)


# ---------------------------------------------------------------------------
# Availability under the SEC 5:30 p.m. ET filing-date cutoff (T1-T8)
#
#   accepted_at  = SEC acceptance time
#   filing_date  = SEC-assigned filing date
#   available_at = earliest time the point-in-time pipeline permits use
#
# Filings that SEC moves to a later filing date get available_at = 06:00 ET
# on that filing date: the SEC-assigned next-business-day filing time, used
# as a modeling proxy for availability (not a measured dissemination time).
# ---------------------------------------------------------------------------


def _availability(
    *,
    accepted_at: str,
    filing_date: str,
    form: str = "10-Q",
) -> pd.Series:
    row = build_edgar_quarter(
        symbol="MU",
        provider_symbol="MU",
        period_end="2024-11-28",
        fiscal_quarter=1,
        filing_date=filing_date,
        accepted_at=accepted_at,
        retrieved_at="2027-01-01T00:00:00Z",
        sec_form_type=form,
        sec_accession_number="ACC",
        income_statement=_income_statement(),
        cash_flow_statement=_cash_flow_statement(),
    ).iloc[0]

    assert pd.Timestamp(row["available_at"]).tzinfo is not None
    assert pd.Timestamp(row["available_at"]).utcoffset() == pd.Timedelta(0)

    return row


@pytest.mark.parametrize(
    ("case", "accepted_at", "filing_date", "form", "expected_available_at", "expected_source"),
    [
        pytest.param(
            "after-hours, winter (real MU Q1 FY2025, 18:52 EST)",
            "2024-12-18T23:52:13Z",
            "2024-12-19",
            "10-Q",
            "2024-12-19T11:00:00Z",
            "sec_deferred_filing_date_6am",
            id="T1",
        ),
        pytest.param(
            "after-hours, summer (real MU Q3 FY2025, 18:50 EDT)",
            "2025-06-25T22:50:42Z",
            "2025-06-26",
            "10-Q",
            "2025-06-26T10:00:00Z",
            "sec_deferred_filing_date_6am",
            id="T2",
        ),
        pytest.param(
            "same-day 10-K (real MU FY2025, 14:42 EDT)",
            "2025-10-03T18:42:25Z",
            "2025-10-03",
            "10-K",
            "2025-10-03T18:42:25Z",
            "sec_acceptance_datetime",
            id="T3",
        ),
        pytest.param(
            "accepted 17:31 EDT but SEC kept the same filing date",
            "2025-03-20T21:31:00Z",
            "2025-03-20",
            "10-Q",
            "2025-03-20T21:31:00Z",
            "sec_acceptance_datetime",
            id="T4",
        ),
        pytest.param(
            "Friday 19:00 EDT -> Monday filing date",
            "2025-03-21T23:00:00Z",
            "2025-03-24",
            "10-Q",
            "2025-03-24T10:00:00Z",
            "sec_deferred_filing_date_6am",
            id="T5",
        ),
        pytest.param(
            "Dec 24 18:00 EST -> Dec 26 filing date (Christmas holiday)",
            "2025-12-24T23:00:00Z",
            "2025-12-26",
            "10-Q",
            "2025-12-26T11:00:00Z",
            "sec_deferred_filing_date_6am",
            id="T6",
        ),
        pytest.param(
            "10-Q/A after hours is deferred like a 10-Q",
            "2024-12-18T23:52:13Z",
            "2024-12-19",
            "10-Q/A",
            "2024-12-19T11:00:00Z",
            "sec_deferred_filing_date_6am",
            id="T7",
        ),
        pytest.param(
            "inconsistent metadata: filing date before acceptance date -> later time",
            "2025-03-21T15:00:00Z",
            "2025-03-20",
            "10-Q",
            "2025-03-21T15:00:00Z",
            "sec_acceptance_datetime",
            id="T8",
        ),
    ],
)
def test_availability_follows_sec_filing_date_cutoff(
    case,
    accepted_at,
    filing_date,
    form,
    expected_available_at,
    expected_source,
):
    row = _availability(
        accepted_at=accepted_at,
        filing_date=filing_date,
        form=form,
    )

    assert row["available_at"] == pd.Timestamp(expected_available_at), case
    assert row["availability_source"] == expected_source, case

    # available_at is never earlier than the SEC acceptance time.
    assert row["available_at"] >= pd.Timestamp(accepted_at), case


@pytest.mark.parametrize(
    ("filing_date", "expected_available_at"),
    [
        # Spring forward: 06:00 EDT on the transition date.
        pytest.param("2025-03-09", "2025-03-09T10:00:00Z", id="dst-spring-forward"),
        # Fall back: 06:00 EST on the transition date (not 05:00).
        pytest.param("2025-11-02", "2025-11-02T11:00:00Z", id="dst-fall-back"),
    ],
)
def test_deferred_availability_is_06_00_wall_clock_on_dst_dates(
    filing_date,
    expected_available_at,
):
    accepted_at = (
        pd.Timestamp(filing_date, tz="America/New_York") - pd.Timedelta(hours=5)
    ).tz_convert("UTC")

    row = _availability(
        accepted_at=accepted_at.isoformat(),
        filing_date=filing_date,
    )

    assert row["availability_source"] == "sec_deferred_filing_date_6am"
    assert row["available_at"] == pd.Timestamp(expected_available_at)

    local = pd.Timestamp(row["available_at"]).tz_convert("America/New_York")
    assert (local.hour, local.minute, local.second) == (6, 0, 0)
