import numpy as np
import pandas as pd
import pytest

from stock_agent.evaluation.baseline import (
    drift_weights,
    run_equal_weight_baseline,
    update_peak_and_drawdown,
)

weights = np.array([0.50, 0.30, 0.20])
returns = np.array([0.10, 0.00, -0.10])

def test_drift_weights():
    result = drift_weights(weights, returns)
    assert np.isclose(result.sum(), 1.0)

def test_update_peak_and_drawdown_NEWPEAK():
    new_peak, drawdown = update_peak_and_drawdown(105_000, 100_000)

    assert new_peak == pytest.approx(105_000)
    assert drawdown == pytest.approx(0)

def test_update_peak_and_drawdown_DRAWNDOWN():
    new_peak, drawdown = update_peak_and_drawdown(90_000, 100_000)

    assert new_peak == pytest.approx(100_000)
    assert drawdown == pytest.approx(0.10)

def test_equal_weight_baseline_zero_returns():
    dates = pd.date_range(
    "2026-01-01",
    periods=6,
    freq="D",
    )

    rows = []

    for date in dates:
        for symbol in ["MU", "SNDK", "MRVL"]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": 0.0,
                }
            )

    frame = pd.DataFrame(rows)

    result = run_equal_weight_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(100_000)
    assert result.cumulative_return == pytest.approx(0)
    assert result.max_drawdown == pytest.approx(0)
    assert result.total_turnover == pytest.approx(0)

def test_equal_weight_baseline_equal_positive_returns():
    dates = pd.date_range(
    "2026-01-01",
    periods=2,
    freq="D",
    )

    rows = []

    for date in dates:
        for symbol in ["MU", "SNDK", "MRVL"]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": 0.01,
                }
            )

    frame = pd.DataFrame(rows)

    result = run_equal_weight_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(102_010)
    assert result.cumulative_return == pytest.approx(0.0201)
    assert result.max_drawdown == pytest.approx(0)
    assert result.total_turnover == pytest.approx(0)

 