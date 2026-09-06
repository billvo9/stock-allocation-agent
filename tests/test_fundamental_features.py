from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.features import (
    build_quarterly_fundamental_features,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)


def _fundamental_row(
    *,
    symbol: str,
    period_end: str,
    revenue: float | None = 100.0,
    gross_profit: float | None = 40.0,
    operating_income: float | None = 20.0,
    net_income: float | None = 15.0,
    diluted_eps: float | None = 1.0,
    total_assets: float | None = 500.0,
    total_debt: float | None = 100.0,
    free_cash_flow: float | None = 10.0,
) -> dict:
    period_timestamp = pd.Timestamp(
        period_end,
        tz="UTC",
    )

    filing_date = period_timestamp + pd.Timedelta(days=25)

    available_at = filing_date + pd.Timedelta(days=1)

    return {
        "symbol": symbol,
        "provider_symbol": symbol,
        "period_end": period_timestamp,
        "available_at": available_at,
        "retrieved_at": pd.Timestamp("2027-12-31T12:00:00Z"),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "diluted_eps": diluted_eps,
        "diluted_average_shares": 100.0,
        "total_assets": total_assets,
        "total_debt": total_debt,
        "cash_and_cash_equivalents": 50.0,
        "inventory": 30.0,
        "stockholders_equity": 300.0,
        "operating_cash_flow": 25.0,
        "capital_expenditure": -15.0,
        "free_cash_flow": free_cash_flow,
        "filing_date": filing_date,
        "sec_form_type": "10-Q",
        "sec_accession_number": None,
        "availability_source": ("sec_filing_date_plus_1d"),
        "currency": "USD",
        "source": "yfinance",
    }


def _frame(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


def test_builds_margin_features():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=1000.0,
                gross_profit=400.0,
                operating_income=250.0,
                free_cash_flow=100.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    row = result.iloc[0]

    assert row["gross_margin"] == pytest.approx(0.40)

    assert row["operating_margin"] == pytest.approx(0.25)

    assert row["fcf_margin"] == pytest.approx(0.10)


def test_builds_debt_to_assets():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                total_debt=100.0,
                total_assets=500.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    assert result.loc[
        0,
        "debt_to_assets",
    ] == pytest.approx(0.20)


def test_revenue_yoy_matches_same_period_last_year():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-09-30",
                revenue=115.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-12-31",
                revenue=118.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z")].iloc[0]

    assert current["revenue_growth_yoy"] == pytest.approx(0.20)


def test_missing_prior_quarter_does_not_use_wrong_quarter():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            # 2025-09-30 intentionally missing.
            _fundamental_row(
                symbol="MU",
                period_end="2025-12-31",
                revenue=130.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-06-30",
                revenue=140.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-09-30",
                revenue=150.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-09-30T00:00:00Z")].iloc[0]

    assert pd.isna(current["revenue_growth_yoy"])


def test_two_missing_prior_quarters_remain_missing():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2025-06-30",
                revenue=110.0,
            ),
            # 2025 Q3 and Q4 intentionally missing.
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-06-30",
                revenue=130.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-09-30",
                revenue=140.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-12-31",
                revenue=150.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    q3 = result[result["period_end"] == pd.Timestamp("2026-09-30T00:00:00Z")].iloc[0]

    q4 = result[result["period_end"] == pd.Timestamp("2026-12-31T00:00:00Z")].iloc[0]

    assert pd.isna(q3["revenue_growth_yoy"])

    assert pd.isna(q4["revenue_growth_yoy"])


def test_yoy_growth_is_independent_by_symbol():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="NVDA",
                period_end="2025-03-31",
                revenue=200.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=120.0,
            ),
            _fundamental_row(
                symbol="NVDA",
                period_end="2026-03-31",
                revenue=300.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    mu = result[
        (result["symbol"] == "MU") & (result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z"))
    ].iloc[0]

    nvda = result[
        (result["symbol"] == "NVDA")
        & (result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z"))
    ].iloc[0]

    assert mu["revenue_growth_yoy"] == pytest.approx(0.20)

    assert nvda["revenue_growth_yoy"] == pytest.approx(0.50)


def test_zero_denominators_return_nan():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                revenue=0.0,
                gross_profit=10.0,
                operating_income=5.0,
                free_cash_flow=2.0,
                total_debt=50.0,
                total_assets=0.0,
            )
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    row = result.iloc[0]

    assert pd.isna(row["gross_margin"])

    assert pd.isna(row["operating_margin"])

    assert pd.isna(row["fcf_margin"])

    assert pd.isna(row["debt_to_assets"])


def test_negative_to_positive_eps_is_positive_growth():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                diluted_eps=-1.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-03-31",
                diluted_eps=1.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-03-31T00:00:00Z")].iloc[0]

    assert current["eps_growth_yoy"] == pytest.approx(2.0)


def test_yoy_allows_small_fiscal_period_shift():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-04-10",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-04-10T00:00:00Z")].iloc[0]

    assert current["revenue_growth_yoy"] == pytest.approx(0.20)


def test_yoy_rejects_large_fiscal_period_shift():
    fundamentals = _frame(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-03-31",
                revenue=100.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-04-25",
                revenue=120.0,
            ),
        ]
    )

    result = build_quarterly_fundamental_features(fundamentals)

    current = result[result["period_end"] == pd.Timestamp("2026-04-25T00:00:00Z")].iloc[0]

    assert pd.isna(current["revenue_growth_yoy"])
