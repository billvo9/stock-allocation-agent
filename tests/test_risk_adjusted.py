from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.evaluation.results import (
    BacktestHistory,
)
from stock_agent.evaluation.risk_adjusted import (
    prepare_aligned_risk_free_returns,
)


def _make_rates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "rate_id": [
                "USD_TREASURY_3M",
                "USD_TREASURY_3M",
            ],
            "provider_series_id": [
                "DGS3MO",
                "DGS3MO",
            ],
            "currency": [
                "USD",
                "USD",
            ],
            "tenor": [
                "3M",
                "3M",
            ],
            "annual_yield": [
                0.0365,
                0.0365,
            ],
            "quote_convention": [
                "investment_basis",
                "investment_basis",
            ],
            "source": [
                "FRED",
                "FRED",
            ],
        }
    )


def test_prepare_aligned_risk_free_returns_matches_history():
    history = BacktestHistory(
        wealth=pd.Series(
            data=[
                100_000.0,
                101_000.0,
                102_000.0,
            ],
            index=pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            name="wealth",
            dtype=float,
        ),
        portfolio_returns=pd.Series(
            data=[
                0.01,
                0.00990099,
            ],
            index=pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            name="portfolio_return",
            dtype=float,
        ),
    )

    result = prepare_aligned_risk_free_returns(
        history=history,
        rates=_make_rates(),
        rate_id="USD_TREASURY_3M",
    )

    assert result.index.equals(history.portfolio_returns.index)

    assert len(result) == 2

    # Jan 2 -> Jan 3 = 1 calendar day
    assert result.iloc[0] == pytest.approx(0.0365 / 365.0)

    # Jan 3 -> Jan 6 = 3 calendar days
    assert result.iloc[1] == pytest.approx(0.0365 * 3.0 / 365.0)


def test_prepare_aligned_risk_free_returns_rejects_misaligned_history():
    history = BacktestHistory(
        wealth=pd.Series(
            data=[
                100_000.0,
                101_000.0,
                102_000.0,
            ],
            index=pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            name="wealth",
            dtype=float,
        ),
        portfolio_returns=pd.Series(
            data=[
                0.01,
                0.00990099,
            ],
            index=pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-07",
                ]
            ),
            name="portfolio_return",
            dtype=float,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="misaligned",
    ):
        prepare_aligned_risk_free_returns(
            history=history,
            rates=_make_rates(),
            rate_id="USD_TREASURY_3M",
        )
