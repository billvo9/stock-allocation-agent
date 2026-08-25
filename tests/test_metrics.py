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

    risk_free_returns = pd.Series(
        [0.0, 0.0],
        index=returns.index,
        dtype=float,
    )

    result = calculate_sharpe_ratio(
        returns=returns,
        risk_free_returns=risk_free_returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(2.0)


def test_sharpe_uses_risk_free_returns():
    returns = pd.Series(
        [0.02, 0.04],
        dtype=float,
    )

    risk_free_returns = pd.Series(
        [0.01, 0.01],
        index=returns.index,
        dtype=float,
    )

    result = calculate_sharpe_ratio(
        returns=returns,
        risk_free_returns=risk_free_returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(2)


def test_sharpe_rejects_mismatched_risk_free_dates():
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
            ]
        ),
        dtype=float,
    )

    risk_free_returns = pd.Series(
        [0.001, 0.001],
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-06",
            ]
        ),
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="matching dates",
    ):
        calculate_sharpe_ratio(
            returns=returns,
            risk_free_returns=risk_free_returns,
        )


def test_calculate_sortino_ratio():
    returns = pd.Series(
        [0.02, -0.01],
        dtype=float,
    )

    risk_free_returns = pd.Series(
        [0.0, 0.0],
        index=returns.index,
        dtype=float,
    )

    result = calculate_sortino_ratio(
        returns=returns,
        risk_free_returns=risk_free_returns,
        periods_per_year=2,
    )

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


def test_calmar_includes_drawdown_from_initial_wealth():
    returns = pd.Series(
        [-0.10, 0.00],
        dtype=float,
    )

    result = calculate_calmar_ratio(
        returns,
        periods_per_year=2,
    )

    assert result == pytest.approx(-1.0)
