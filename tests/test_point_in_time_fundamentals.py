from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.point_in_time import (
    align_quarterly_fundamentals_asof,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)


def _fundamental_row(
    symbol: str,
    period_end: str,
    available_at: str,
    revenue: float,
    retrieved_at: str = "2026-09-04T12:00:00Z",
) -> pd.DataFrame:
    available_timestamp = pd.Timestamp(available_at)

    filing_date = available_timestamp - pd.Timedelta(days=1)

    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "provider_symbol": symbol,
                "period_end": pd.Timestamp(period_end),
                "available_at": (available_timestamp),
                "retrieved_at": pd.Timestamp(retrieved_at),
                "revenue": revenue,
                "gross_profit": 400.0,
                "operating_income": 250.0,
                "net_income": 200.0,
                "diluted_eps": 2.0,
                "diluted_average_shares": 100.0,
                "total_assets": 5000.0,
                "total_debt": 1000.0,
                "cash_and_cash_equivalents": 500.0,
                "inventory": 300.0,
                "stockholders_equity": 3000.0,
                "operating_cash_flow": 350.0,
                "capital_expenditure": -150.0,
                "free_cash_flow": 200.0,
                "filing_date": filing_date,
                "sec_form_type": "10-Q",
                "sec_accession_number": None,
                "availability_source": ("sec_filing_date_plus_1d"),
                "currency": "USD",
                "source": "yfinance",
            }
        ],
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


def _market_frame(
    dates: list[str],
    symbol: str = "MU",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates,
                utc=True,
            ),
            "symbol": symbol,
        }
    )


def test_future_fundamentals_are_not_used():
    market = _market_frame(
        [
            "2026-06-26",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == False
    )

    assert pd.isna(
        result.loc[
            0,
            "revenue",
        ]
    )


def test_report_is_available_exactly_on_available_at():
    market = _market_frame(
        [
            "2026-06-27",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == True
    )

    assert result.loc[
        0,
        "revenue",
    ] == pytest.approx(1000.0)


def test_latest_prior_report_is_carried_forward():
    market = _market_frame(
        [
            "2026-06-26",
            "2026-07-15",
        ]
    )

    q1 = _fundamental_row(
        symbol="MU",
        period_end="2026-02-28",
        available_at="2026-03-20",
        revenue=800.0,
    )

    q2 = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    fundamentals = pd.concat(
        [
            q1,
            q2,
        ],
        ignore_index=True,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert result.loc[
        0,
        "revenue",
    ] == pytest.approx(800.0)

    assert result.loc[
        1,
        "revenue",
    ] == pytest.approx(1000.0)


def test_asset_with_no_prior_report_does_not_crash():
    market = _market_frame(
        [
            "2026-01-15",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-02-28",
        available_at="2026-03-20",
        revenue=800.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == False
    )

    assert pd.isna(
        result.loc[
            0,
            "fundamental_age_days",
        ]
    )


def test_multiple_symbols_align_independently():
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-06-27",
                    "2026-06-27",
                ],
                utc=True,
            ),
            "symbol": [
                "MU",
                "NVDA",
            ],
        }
    )

    mu = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    nvda = _fundamental_row(
        symbol="NVDA",
        period_end="2026-04-30",
        available_at="2026-05-22",
        revenue=2000.0,
    )

    fundamentals = pd.concat(
        [
            mu,
            nvda,
        ],
        ignore_index=True,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    mu_result = result[result["symbol"] == "MU"].iloc[0]

    nvda_result = result[result["symbol"] == "NVDA"].iloc[0]

    assert mu_result["revenue"] == pytest.approx(1000.0)

    assert nvda_result["revenue"] == pytest.approx(2000.0)


def test_duplicate_symbol_availability_raises():
    first = _fundamental_row(
        symbol="MU",
        period_end="2026-02-28",
        available_at="2026-06-27",
        revenue=800.0,
    )

    second = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
        retrieved_at="2026-09-05T12:00:00Z",
    )

    fundamentals = pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )

    market = _market_frame(
        [
            "2026-06-27",
        ]
    )

    with pytest.raises(
        ValueError,
        match="available_at",
    ):
        align_quarterly_fundamentals_asof(
            market_frame=market,
            fundamentals=fundamentals,
        )
