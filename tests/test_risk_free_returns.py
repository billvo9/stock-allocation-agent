import pandas as pd
import pytest

from stock_agent.data.rates.transform import (
    prepare_risk_free_returns,
)


def _make_rates(
    dates: list[str],
    yields: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "rate_id": "USD_TREASURY_3M",
            "provider_series_id": "DGS3MO",
            "currency": "USD",
            "tenor": "3M",
            "annual_yield": yields,
            "quote_convention": ("investment_basis"),
            "source": "FRED",
        }
    )


def test_risk_free_return_one_day():
    rates = _make_rates(
        dates=[
            "2026-01-02",
        ],
        yields=[
            0.0365,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-03",
        ]
    )

    result = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id="USD_TREASURY_3M",
    )

    assert len(result) == 1

    assert result.iloc[0] == pytest.approx(0.0001)

    assert result.index.equals(
        pd.to_datetime(
            [
                "2026-01-03",
            ]
        )
    )


def test_risk_free_return_weekend_accrual():
    rates = _make_rates(
        dates=[
            "2026-01-02",
        ],
        yields=[
            0.0365,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
        ]
    )

    result = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id="USD_TREASURY_3M",
    )

    assert result.iloc[0] == pytest.approx(0.0003)


def test_risk_free_return_uses_prior_not_future_rate():
    rates = _make_rates(
        dates=[
            "2026-01-02",
            "2026-01-05",
        ],
        yields=[
            0.0365,
            0.99,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
        ]
    )

    result = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id="USD_TREASURY_3M",
    )

    # Friday -> Monday is three calendar days.
    #
    # The Monday 99% rate must NOT be used.
    #
    # 0.0365 * 3 / 365 = 0.0003
    assert result.iloc[0] == pytest.approx(0.0003)


def test_risk_free_return_uses_latest_prior_rate_when_start_missing():
    rates = _make_rates(
        dates=[
            "2026-01-01",
        ],
        yields=[
            0.0365,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-03",
        ]
    )

    result = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id="USD_TREASURY_3M",
    )

    # No Jan 2 observation exists.
    # Jan 1 is the latest known observation.
    assert result.iloc[0] == pytest.approx(0.0001)


def test_risk_free_return_rejects_missing_prior_rate():
    rates = _make_rates(
        dates=[
            "2026-01-05",
        ],
        yields=[
            0.0365,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-03",
        ]
    )

    with pytest.raises(
        ValueError,
        match="No risk-free rate available",
    ):
        prepare_risk_free_returns(
            rates=rates,
            portfolio_dates=portfolio_dates,
            rate_id="USD_TREASURY_3M",
        )


def test_risk_free_return_uses_366_days_in_leap_year():
    rates = _make_rates(
        dates=[
            "2024-02-28",
        ],
        yields=[
            0.0366,
        ],
    )

    portfolio_dates = pd.to_datetime(
        [
            "2024-02-28",
            "2024-03-01",
        ]
    )

    result = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id="USD_TREASURY_3M",
    )

    # Two calendar days:
    # Feb 28 -> Feb 29
    # Feb 29 -> Mar 1
    #
    # 0.0366 * 2 / 366 = 0.0002
    assert result.iloc[0] == pytest.approx(0.0002)
