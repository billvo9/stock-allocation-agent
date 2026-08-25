import pandas as pd
import pytest

from stock_agent.evaluation.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def test_calculate_cagr():
    returns = pd.Series(
        [0.10, 0.10],
        dtype=float,
    )

    result = calculate_cagr(
        returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(0.21)


def test_calculate_annualized_volatility():
    returns = pd.Series(
        [0.01, -0.01],
        dtype=float,
    )

    result = calculate_annualized_volatility(
        returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(0.02)


def test_calculate_sharpe_ratio():
    returns = pd.Series(
        [0.01, 0.03],
        dtype=float,
    )

    result = calculate_sharpe_ratio(returns, periods_per_year=2, annual_risk_free_rate=0.0)

    assert result == pytest.approx(2.0)


def test_calculate_sortino_ratio():
    returns = pd.Series(
        [0.02, -0.01],
        dtype=float,
    )

    result = calculate_sortino_ratio(returns, periods_per_year=2, annual_risk_free_rate=0.0)

    assert result == pytest.approx(1.0)


def test_calculate_calmar_ratio():
    returns = pd.Series(
        [0.20, -0.10, 0.20],
        dtype=float,
    )

    result = calculate_calmar_ratio(
        returns,
        periods_per_year=3,
    )

    assert result == pytest.approx(2.96)
