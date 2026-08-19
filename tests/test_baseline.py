import numpy as np
import pandas as pd
import pytest

from stock_agent.evaluation.baseline import (
    drift_weights,
    inverse_volatility_weights,
    prepare_equal_weight_targets,
    prepare_feature_matrix,
    prepare_inverse_volatility_targets,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_rebalanced_backtest,
    update_peak_and_drawdown,
    weights_to_array,
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

    assert result.final_wealth == pytest.approx(101_000)
    assert result.cumulative_return == pytest.approx(0.01)
    assert result.max_drawdown == pytest.approx(0)
    assert result.total_turnover == pytest.approx(0)


def test_equal_weight_baseline_does_not_apply_first_date_return():
    dates = pd.date_range(
        "2026-01-01",
        periods=2,
        freq="D",
    )

    rows = []

    for date, daily_return in zip(
        dates,
        [0.50, 0.0],
        strict=True,
    ):
        for symbol in ["MU", "SNDK", "MRVL"]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": daily_return,
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


def test_equal_weight_baseline_requires_two_dates():
    dates = pd.date_range(
        "2026-01-01",
        periods=1,
        freq="D",
    )

    rows = []

    for date, daily_return in zip(
        dates,
        [0.50],
        strict=True,
    ):
        for symbol in ["MU", "SNDK", "MRVL"]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": daily_return,
                }
            )

    frame = pd.DataFrame(rows)

    with pytest.raises(
        ValueError,
        match="at least two dates",
    ):
        run_equal_weight_baseline(
            frame=frame,
            symbols=["MU", "SNDK", "MRVL"],
            initial_wealth=100_000,
            rebalance_every=5,
            transaction_cost_rate=0.001,
        )


def test_prepare_feature_matrix_orders_symbols():

    dates = pd.date_range(
        "2026-01-01",
        periods=2,
        freq="D",
    )

    rows = []

    for date in dates:
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "MRVL",
                    "volatility_20d": 0.05,
                },
                {
                    "date": date,
                    "symbol": "MU",
                    "volatility_20d": 0.02,
                },
                {
                    "date": date,
                    "symbol": "SNDK",
                    "volatility_20d": 0.04,
                },
            ]
        )

    frame = pd.DataFrame(rows)

    result = prepare_feature_matrix(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        column="volatility_20d",
    )

    assert list(result.columns) == ["MU", "SNDK", "MRVL"]

    assert result.loc[dates[0], "MU"] == pytest.approx(0.02)
    assert result.loc[dates[0], "SNDK"] == pytest.approx(0.04)
    assert result.loc[dates[0], "MRVL"] == pytest.approx(0.05)


def test_feature_matrices_align_returns_and_volatility():

    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "MRVL",
                "daily_return": 0.03,
                "volatility_20d": 0.05,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "SNDK",
                "daily_return": 0.02,
                "volatility_20d": 0.04,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "MU",
                "daily_return": 0.01,
                "volatility_20d": 0.02,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "MRVL",
                "daily_return": -0.01,
                "volatility_20d": 0.05,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "MU",
                "daily_return": 0.005,
                "volatility_20d": 0.02,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "SNDK",
                "daily_return": -0.02,
                "volatility_20d": 0.04,
            },
        ]
    )

    returns = prepare_feature_matrix(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        column="daily_return",
    )

    volatilities = prepare_feature_matrix(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        column="volatility_20d",
    )

    assert list(returns.columns) == ["MU", "SNDK", "MRVL"]
    assert list(volatilities.columns) == ["MU", "SNDK", "MRVL"]

    assert returns.index.equals(volatilities.index)

    date = pd.Timestamp("2026-01-02")

    assert returns.loc[date, "MU"] == pytest.approx(0.01)
    assert volatilities.loc[date, "MU"] == pytest.approx(0.02)


def test_weights_to_array_preserves_symbol_order():
    weights = {
        "MRVL": 0.20,
        "MU": 0.50,
        "SNDK": 0.30,
    }

    symbols = ["MU", "SNDK", "MRVL"]

    result = weights_to_array(weights, symbols)

    assert np.allclose(
        result,
        [0.50, 0.30, 0.20],
    )


def test_weights_to_array_rejects_missing_symbol():
    weights = {
        "MU": 0.50,
        "MRVL": 0.50,
    }

    symbols = ["MU", "SNDK", "MRVL"]

    with pytest.raises(
        ValueError,
        match="SNDK",
    ):
        weights_to_array(weights, symbols)


def test_inverse_volatility_weights_from_feature_row():
    result = inverse_volatility_weights(
        pd.Series(
            {
                "MRVL": 0.05,
                "MU": 0.02,
                "SNDK": 0.04,
            }
        ),
        ["MU", "SNDK", "MRVL"],
    )

    assert np.allclose(result, [50 / 95, 25 / 95, 20 / 95])


def test_inverse_volatility_weights_rejects_missing_symbol():
    volatility_row = pd.Series(
        {
            "MU": 0.02,
            "SNDK": 0.04,
            "MUU": 0.04,
        }
    )

    symbols = ["MU", "SNDK", "MRVL"]

    with pytest.raises(
        ValueError,
        match="MRVL",
    ):
        inverse_volatility_weights(volatility_row, symbols)


def test_prepare_inverse_volatility_targets():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "MU",
                "volatility_20d": 0.02,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "SNDK",
                "volatility_20d": 0.04,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "symbol": "MRVL",
                "volatility_20d": 0.05,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "MU",
                "volatility_20d": 0.05,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "SNDK",
                "volatility_20d": 0.02,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "symbol": "MRVL",
                "volatility_20d": 0.04,
            },
        ]
    )

    result = prepare_inverse_volatility_targets(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
    )

    assert list(result.columns) == ["MU", "SNDK", "MRVL"]

    first_date = pd.Timestamp("2026-01-01")
    second_date = pd.Timestamp("2026-01-02")

    assert result.loc[first_date, "MU"] == pytest.approx(50 / 95)
    assert result.loc[first_date, "SNDK"] == pytest.approx(25 / 95)
    assert result.loc[first_date, "MRVL"] == pytest.approx(20 / 95)

    row_sum = float(
        result.loc[
            [first_date],
            ["MU", "SNDK", "MRVL"],
        ]
        .to_numpy(dtype=float)
        .sum()
    )

    assert row_sum == pytest.approx(1.0)

    assert result.loc[second_date, "MU"] != pytest.approx(result.loc[first_date, "MU"])


def test_run_rebalanced_backtest_applies_previous_date_weights():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.00, 0.00, 0.00],
            [0.10, 0.00, -0.10],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],
            # Deliberately very different.
            # The Jan 2 return should NOT use this row.
            [0.00, 1.00, 0.00],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.0,
    )

    assert result.final_wealth == pytest.approx(103_000)
    assert result.cumulative_return == pytest.approx(0.03)
    assert result.max_drawdown == pytest.approx(0.0)
    assert result.total_turnover == pytest.approx(0.0)


def test_run_rebalanced_backtest_applies_rebalance_and_transaction_cost():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],  # initialize
            [0.20, 0.30, 0.50],  # target used before Jan 3 return
            [0.10, 0.20, 0.70],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(99_940)
    assert result.cumulative_return == pytest.approx(-0.0006)
    assert result.max_drawdown == pytest.approx(0.0006)
    assert result.total_turnover == pytest.approx(0.6)


def test_run_inverse_volatility_baseline_zero_returns():
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    rows = []

    for date in dates:
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "MU",
                    "daily_return": 0.0,
                    "volatility_20d": 0.02,
                },
                {
                    "date": date,
                    "symbol": "SNDK",
                    "daily_return": 0.0,
                    "volatility_20d": 0.04,
                },
                {
                    "date": date,
                    "symbol": "MRVL",
                    "daily_return": 0.0,
                    "volatility_20d": 0.05,
                },
            ]
        )

    frame = pd.DataFrame(rows)

    result = run_inverse_volatility_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(100_000)
    assert result.cumulative_return == pytest.approx(0.0)
    assert result.max_drawdown == pytest.approx(0.0)
    assert result.total_turnover == pytest.approx(0.0)


def test_run_inverse_volatility_baseline_rebalances_when_volatility_changes():
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    rows = []

    # Date 1
    rows.extend(
        [
            {
                "date": dates[0],
                "symbol": "MU",
                "daily_return": 0.0,
                "volatility_20d": 0.02,
            },
            {
                "date": dates[0],
                "symbol": "SNDK",
                "daily_return": 0.0,
                "volatility_20d": 0.04,
            },
            {
                "date": dates[0],
                "symbol": "MRVL",
                "daily_return": 0.0,
                "volatility_20d": 0.05,
            },
        ]
    )

    # Date 2 — substantially different volatility profile
    rows.extend(
        [
            {
                "date": dates[1],
                "symbol": "MU",
                "daily_return": 0.0,
                "volatility_20d": 0.05,
            },
            {
                "date": dates[1],
                "symbol": "SNDK",
                "daily_return": 0.0,
                "volatility_20d": 0.02,
            },
            {
                "date": dates[1],
                "symbol": "MRVL",
                "daily_return": 0.0,
                "volatility_20d": 0.04,
            },
        ]
    )

    # Date 3 — valid values; these are not the target used for the Jan 3 return
    rows.extend(
        [
            {
                "date": dates[2],
                "symbol": "MU",
                "daily_return": 0.0,
                "volatility_20d": 0.03,
            },
            {
                "date": dates[2],
                "symbol": "SNDK",
                "daily_return": 0.0,
                "volatility_20d": 0.03,
            },
            {
                "date": dates[2],
                "symbol": "MRVL",
                "daily_return": 0.0,
                "volatility_20d": 0.03,
            },
        ]
    )

    frame = pd.DataFrame(rows)

    result = run_inverse_volatility_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    assert result.total_turnover > 0
    assert result.final_wealth < 100_000
    assert result.cumulative_return < 0


def test_prepare_equal_weight_targets():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, 0.02, -0.01],
            [0.03, -0.02, 0.01],
            [0.00, 0.01, 0.02],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = prepare_equal_weight_targets(
        returns=returns,
        symbols=["MU", "SNDK", "MRVL"],
    )

    assert list(result.columns) == ["MU", "SNDK", "MRVL"]

    assert result.index.equals(returns.index)

    assert np.allclose(
        result.iloc[0].to_numpy(dtype=float),
        [1 / 3, 1 / 3, 1 / 3],
    )

    assert np.allclose(
        result.iloc[-1].to_numpy(dtype=float),
        [1 / 3, 1 / 3, 1 / 3],
    )

    assert np.allclose(
        result.sum(axis=1).to_numpy(dtype=float),
        [1.0, 1.0, 1.0],
    )


def test_equal_weight_targets_are_constant_across_dates():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, 0.02, -0.01],
            [0.03, -0.02, 0.01],
            [0.00, 0.01, 0.02],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = prepare_equal_weight_targets(
        returns=returns,
        symbols=["MU", "SNDK", "MRVL"],
    )

    assert np.allclose(
        result.iloc[0].to_numpy(dtype=float),
        result.iloc[-1].to_numpy(dtype=float),
    )

    assert np.allclose(
        result.iloc[0].to_numpy(dtype=float),
        [1 / 3, 1 / 3, 1 / 3],
    )


def test_equal_weight_baseline_forwards_non_default_initial_wealth():
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
                    "daily_return": 0.10,
                }
            )

    frame = pd.DataFrame(rows)

    result = run_equal_weight_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=50_000,
        rebalance_every=2,
        transaction_cost_rate=0.0,
    )

    assert result.initial_wealth == pytest.approx(50_000)
    assert result.final_wealth == pytest.approx(55_000)


def test_run_rebalanced_backtest_respects_first_rebalance_step():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],
            [0.20, 0.30, 0.50],
            [0.10, 0.20, 0.70],
            [0.40, 0.40, 0.20],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
        first_rebalance_step=3,
    )

    assert result.total_turnover == pytest.approx(1.0)
    assert result.final_wealth == pytest.approx(99_900)
    assert result.cumulative_return == pytest.approx(-0.001)
    assert result.max_drawdown == pytest.approx(0.001)
