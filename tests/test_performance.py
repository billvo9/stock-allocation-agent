from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.evaluation.baseline import (
    BaselineResult,
)
from stock_agent.evaluation.performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from stock_agent.evaluation.results import (
    BacktestHistory,
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


def test_calculate_performance_metrics_returns_metrics():
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
                -0.005,
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

    result = BaselineResult(
        initial_wealth=100_000.0,
        final_wealth=102_000.0,
        cumulative_return=0.02,
        max_drawdown=0.005,
        total_turnover=1.0,
        history=history,
    )

    metrics = calculate_performance_metrics(
        result=result,
        rates=_make_rates(),
        rate_id="USD_TREASURY_3M",
    )

    assert isinstance(
        metrics,
        PerformanceMetrics,
    )

    assert metrics.cagr == pytest.approx(metrics.cagr)
    assert metrics.annualized_volatility > 0
    assert metrics.sharpe == pytest.approx(metrics.sharpe)
    assert metrics.sortino == pytest.approx(metrics.sortino)
    assert metrics.calmar == pytest.approx(metrics.calmar)


def test_calculate_performance_metrics_requires_history():
    result = BaselineResult(
        initial_wealth=100_000.0,
        final_wealth=101_000.0,
        cumulative_return=0.01,
        max_drawdown=0.0,
        total_turnover=0.0,
        history=None,
    )

    with pytest.raises(
        ValueError,
        match="does not contain history",
    ):
        calculate_performance_metrics(
            result=result,
            rates=_make_rates(),
            rate_id="USD_TREASURY_3M",
        )
