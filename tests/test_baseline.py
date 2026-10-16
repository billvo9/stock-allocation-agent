import numpy as np
import pandas as pd
import pytest

from stock_agent.evaluation.baseline import (
    CASH_SYMBOL,
    ScheduleMismatchError,
    add_cash_returns,
    backtest_inputs,
    drift_weights,
    inverse_volatility_weights,
    prepare_equal_weight_targets,
    prepare_feature_matrix,
    prepare_inverse_volatility_targets,
    prepare_momentum_targets,
    require_common_schedule,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_momentum_baseline,
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
    # OD8.14 rewrite: the build from cash is a trade (D3), so it costs 10 bp and turns over 1.0.
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

    assert result.final_wealth == pytest.approx(99_900)
    assert result.final_wealth != pytest.approx(100_000)  # the old free build
    assert result.cumulative_return == pytest.approx(-0.001)
    assert result.max_drawdown == pytest.approx(0.001)
    assert result.total_turnover == pytest.approx(1.0)


def test_equal_weight_baseline_equal_positive_returns():
    # OD8.14 rewrite: decided at date 0, built at the close of date 1 (D1, D3), so only
    # date 2's return is earned, net of the build cost: 100,000 x 0.999 x 1.01.
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
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

    assert result.final_wealth == pytest.approx(100_899)
    assert result.cumulative_return == pytest.approx(0.00899)
    assert result.max_drawdown == pytest.approx(0)
    assert result.total_turnover == pytest.approx(1.0)


def test_equal_weight_baseline_does_not_apply_returns_before_the_build_executes():
    # OD8.14 rewrite: neither the decision date's return nor the execution date's is earned.
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    rows = []

    for date, daily_return in zip(
        dates,
        [0.50, 0.50, 0.0],
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

    assert result.final_wealth == pytest.approx(99_900)
    assert result.cumulative_return == pytest.approx(-0.001)
    assert result.max_drawdown == pytest.approx(0.001)
    assert result.total_turnover == pytest.approx(1.0)


def test_equal_weight_baseline_requires_three_dates():
    # OD8.14 rewrite: a decision, its execution close and one earned return (D1).
    dates = pd.date_range(
        "2026-01-01",
        periods=2,
        freq="D",
    )

    rows = []

    for date, daily_return in zip(
        dates,
        [0.50, 0.50],
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
        match="at least three dates",
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


def test_run_rebalanced_backtest_applies_the_build_target_from_the_next_close():
    # OD8.14 rewrite: the target decided at Jan 1 trades at the Jan 2 close and earns Jan 3.
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
            [0.50, 0.50, 0.50],  # into the execution close: never earned
            [0.10, 0.00, -0.10],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],
            # Deliberately very different, and not a decision date (rebalance_every=5).
            [0.00, 1.00, 0.00],
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
    assert result.total_turnover == pytest.approx(1.0)


def test_run_rebalanced_backtest_applies_rebalance_and_transaction_cost():
    # OD8.14 rewrite: build 10 bp on 1.0; the rebalance decided Jan 3 trades at the Jan 4
    # close: 10 bp on traded notional 0.6 (one-way turnover 0.3), multiplicative.
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
        ]
    )

    returns = pd.DataFrame(
        data=[[0.00, 0.00, 0.00]] * 5,
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],  # the build
            [0.90, 0.05, 0.05],  # not a decision date
            [0.20, 0.30, 0.50],  # the rebalance
            [0.10, 0.20, 0.70],
            [0.10, 0.20, 0.70],  # decided on the last date: never executed
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

    assert result.final_wealth == pytest.approx(100_000 * 0.999 * 0.9994)
    assert result.cumulative_return == pytest.approx(0.999 * 0.9994 - 1)
    assert result.max_drawdown == pytest.approx(1 - 0.999 * 0.9994)
    assert result.total_turnover == pytest.approx(1.3)


def test_run_inverse_volatility_baseline_zero_returns():
    # OD8.14 rewrite: the build from cash costs 10 bp and turns over 1.0 (D3).
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

    assert result.final_wealth == pytest.approx(99_900)
    assert result.cumulative_return == pytest.approx(-0.001)
    assert result.max_drawdown == pytest.approx(0.001)
    assert result.total_turnover == pytest.approx(1.0)


def test_run_inverse_volatility_baseline_rebalances_when_volatility_changes():
    # OD8.14 rewrite: with next-close execution the rebalance decided on date 3 trades at
    # the close of date 4, so five dates are needed for it to execute (three sufficed only
    # at the same close). The build alone would turn over exactly 1.0.
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    volatilities = [
        (0.02, 0.04, 0.05),
        (0.02, 0.04, 0.05),
        (0.05, 0.02, 0.04),  # the rebalance decision: a different profile
        (0.03, 0.03, 0.03),
        (0.03, 0.03, 0.03),
    ]

    rows = []

    for date, profile in zip(dates, volatilities, strict=True):
        for symbol, volatility in zip(["MU", "SNDK", "MRVL"], profile, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": 0.0,
                    "volatility_20d": volatility,
                }
            )

    frame = pd.DataFrame(rows)

    result = run_inverse_volatility_baseline(
        frame=frame,
        symbols=["MU", "SNDK", "MRVL"],
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    assert result.total_turnover > 1.0
    assert result.final_wealth < 99_900
    assert result.cumulative_return < -0.001


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
    # OD8.14 rewrite: three dates, so the build (decided at date 0) earns date 2's return.
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
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


def test_run_rebalanced_backtest_respects_the_registered_offset():
    # OD8.14 rewrite (D7): the per-strategy first_rebalance_step is replaced by one registered
    # offset. With every=2 and offset=1 the decisions are dates 0 (build), 1, 3 and 5; date 2's
    # very different target is never used, and date 5's has no following session.
    dates = pd.date_range("2026-01-01", periods=7, freq="D")

    returns = pd.DataFrame(
        data=[[0.00, 0.00, 0.00]] * 7,
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.30, 0.20],
            [0.20, 0.30, 0.50],
            [0.90, 0.05, 0.05],
            [0.40, 0.40, 0.20],
            [0.40, 0.40, 0.20],
            [0.40, 0.40, 0.20],
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
        rebalance_offset=1,
    )

    executed = [e.execution for e in result.simulation.executions if e.execution is not None]
    assert executed == [1, 2, 4]
    assert result.total_turnover == pytest.approx(1.6)
    assert result.final_wealth == pytest.approx(100_000 * 0.999 * 0.9994**2)
    assert result.max_drawdown == pytest.approx(1 - 0.999 * 0.9994**2)


def test_add_cash_returns_adds_zero_return_cash():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, -0.02, 0.03],
            [0.02, 0.01, -0.01],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = add_cash_returns(returns)

    assert list(result.columns) == [
        "MU",
        "SNDK",
        "MRVL",
        "CASH",
    ]

    # verify CASH is exactly 0 on every date
    assert np.allclose(
        result["CASH"].to_numpy(),
        [0.0, 0.0],
    )


def test_add_cash_returns_adds_nonzero_return_cash():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, -0.02, 0.03],
            [0.02, 0.01, -0.01],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = add_cash_returns(
        returns,
        cash_return=0.0001,
    )

    assert np.allclose(
        result["CASH"].to_numpy(),
        [0.0001, 0.0001],
    )


def test_add_cash_returns_does_not_mutate_input():

    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, -0.02, 0.03],
            [0.02, 0.01, -0.01],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL"],
    )

    result = add_cash_returns(returns)

    assert "CASH" not in returns.columns
    assert "CASH" in result.columns


def test_add_cash_returns_rejects_existing_cash_column():

    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.01, -0.02, 0.03, 0.0],
            [0.02, 0.01, -0.01, 0.0],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL", "CASH"],
    )

    with pytest.raises(
        ValueError,
        match="CASH",
    ):
        add_cash_returns(returns)


def test_rebalanced_backtest_supports_full_cash_position():
    # OD8.14 rewrite: three dates (decision, execution, one earned return); building an
    # all-cash target from cash trades nothing.

    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.00, 0.00, 0.00, 0.00],
            [-0.20, -0.30, -0.15, 0.00],
            [-0.20, -0.30, -0.15, 0.00],
        ],
        columns=["MU", "SNDK", "MRVL", "CASH"],
        index=dates,
    )

    target_weights = pd.DataFrame(
        data=[
            [0.00, 0.00, 0.00, 1.00],
            [0.00, 0.00, 0.00, 1.00],
            [0.00, 0.00, 0.00, 1.00],
        ],
        index=dates,
        columns=["MU", "SNDK", "MRVL", "CASH"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(100_000)
    assert result.cumulative_return == pytest.approx(0.0)
    assert result.max_drawdown == pytest.approx(0.0)
    assert result.total_turnover == pytest.approx(0.0)


def test_rebalanced_backtest_charges_only_the_risky_leg_when_moving_to_cash():
    # OD8.14 rewrite (spec T2.1): selling everything into cash costs 10 bp, not 20 bp; the
    # cash account is the free funding leg. Build (10 bp) then sale (10 bp), multiplicative.
    dates = pd.date_range("2026-01-01", periods=5, freq="D")

    returns = pd.DataFrame(
        data=[[0.00, 0.00, 0.00, 0.00]] * 5,
        index=dates,
        columns=["MU", "SNDK", "MRVL", "CASH"],
    )

    target_weights = pd.DataFrame(
        data=[[0.50, 0.30, 0.20, 0.00]] + [[0.00, 0.00, 0.00, 1.00]] * 4,
        index=dates,
        columns=["MU", "SNDK", "MRVL", "CASH"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    sale = result.simulation.steps[2]
    assert sale.cost == pytest.approx(0.001)  # charging the cash leg too would give 0.002
    assert result.total_turnover == pytest.approx(2.0)
    assert result.final_wealth == pytest.approx(99_800.10)
    assert result.final_wealth != pytest.approx(99_700.20)
    assert result.cumulative_return == pytest.approx(0.999**2 - 1)
    assert result.max_drawdown == pytest.approx(1 - 0.999**2)


def make_momentum_test_frame(
    dates: pd.DatetimeIndex,
    symbols: list[str],
    returns_by_symbol: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []

    for symbol in symbols:
        for date, daily_return in zip(
            dates,
            returns_by_symbol[symbol],
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "daily_return": daily_return,
                }
            )

    return pd.DataFrame(rows)


def test_prepare_momentum_targets_removes_warmup_dates():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [0.02, -0.01, 0.03, 0.01, -0.02],
            "SNDK": [0.01, 0.02, -0.01, 0.04, 0.01],
            "MRVL": [-0.01, 0.03, 0.02, -0.02, 0.03],
        },
    )

    result = prepare_momentum_targets(
        frame=frame,
        symbols=symbols,
        lookback=3,
    )

    assert len(result) == 3
    assert result.index[0] == dates[2]


def test_prepare_momentum_targets_contains_cash_and_sums_to_one():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [0.03, 0.01, -0.02, 0.04, 0.02],
            "SNDK": [-0.02, 0.03, 0.01, -0.01, 0.05],
            "MRVL": [0.01, -0.01, 0.02, 0.03, -0.02],
        },
    )

    result = prepare_momentum_targets(
        frame=frame,
        symbols=symbols,
        lookback=3,
    )

    assert list(result.columns) == ["MU", "SNDK", "MRVL", "CASH"]

    assert np.allclose(
        result.sum(axis=1).to_numpy(),
        1.0,
    )


def test_prepare_momentum_targets_uses_cash_when_all_momentum_negative():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [-0.01, -0.02, -0.01],
            "SNDK": [-0.03, -0.01, -0.02],
            "MRVL": [-0.02, -0.02, -0.01],
        },
    )

    result = prepare_momentum_targets(
        frame=frame,
        symbols=symbols,
        lookback=3,
    )

    first_target = result.iloc[0]

    assert first_target["CASH"] == pytest.approx(1.0)
    assert first_target["MU"] == pytest.approx(0.0)
    assert first_target["SNDK"] == pytest.approx(0.0)
    assert first_target["MRVL"] == pytest.approx(0.0)


def test_run_momentum_baseline_uses_cash_when_all_momentum_negative():
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [-0.01, -0.02, -0.01, -0.03, -0.02],
            "SNDK": [-0.02, -0.01, -0.03, -0.02, -0.01],
            "MRVL": [-0.01, -0.03, -0.02, -0.01, -0.02],
        },
    )

    result = run_momentum_baseline(
        frame=frame,
        symbols=symbols,
        lookback=3,
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    assert result.final_wealth == pytest.approx(100_000)
    assert result.cumulative_return == pytest.approx(0.0)
    assert result.max_drawdown == pytest.approx(0.0)
    assert result.total_turnover == pytest.approx(0.0)


def test_run_momentum_baseline_applies_positive_momentum_exposure():
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [0.02, 0.02, 0.02, 0.03, 0.04],
            "SNDK": [0.01, 0.01, 0.01, 0.00, 0.00],
            "MRVL": [-0.01, -0.01, -0.01, -0.02, -0.02],
        },
    )

    result = run_momentum_baseline(
        frame=frame,
        symbols=symbols,
        lookback=3,
        initial_wealth=100_000,
        rebalance_every=2,
        transaction_cost_rate=0.0,
    )

    assert result.final_wealth > 100_000
    assert result.cumulative_return > 0


def test_run_momentum_baseline_refuses_a_cash_rate_before_group_c():
    # A D4 (a) consequence, not an OD8.14 rewrite: uninvested cash earns 0% (synthetic
    # engineering only), and group C cash returns (D4 (b)) are not approved, so a nonzero
    # cash return is refused rather than silently used.
    dates = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="D",
    )

    symbols = ["MU", "SNDK", "MRVL"]

    frame = make_momentum_test_frame(
        dates=dates,
        symbols=symbols,
        returns_by_symbol={
            "MU": [-0.01, -0.01, -0.01, -0.01, -0.01],
            "SNDK": [-0.02, -0.02, -0.02, -0.02, -0.02],
            "MRVL": [-0.03, -0.03, -0.03, -0.03, -0.03],
        },
    )

    with pytest.raises(ValueError, match="group C cash returns are not approved"):
        run_momentum_baseline(
            frame=frame,
            symbols=symbols,
            lookback=3,
            initial_wealth=100_000,
            rebalance_every=2,
            transaction_cost_rate=0.0,
            cash_return=0.001,
        )


def test_rebalanced_backtest_records_return_and_wealth_history():
    # OD8.14 rewrite: wealth starts at the build's execution close (Jan 2), and the first
    # realized return is dated Jan 3; Jan 2's +50% is never earned.
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
            [0.00, 0.00],
            [0.50, 0.50],
            [0.01, 0.01],
            [0.02, 0.02],
        ],
        index=dates,
        columns=["A", "B"],
    )

    target_weights = pd.DataFrame(
        data=[[0.50, 0.50]] * 4,
        index=dates,
        columns=["A", "B"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        transaction_cost_rate=0.0,
        rebalance_every=5,
    )

    assert result.history is not None

    assert list(result.history.portfolio_returns.index) == [
        dates[2],
        dates[3],
    ]

    assert result.history.portfolio_returns.iloc[0] == pytest.approx(0.01)
    assert result.history.portfolio_returns.iloc[1] == pytest.approx(0.02)

    assert result.history.wealth.iloc[-1] == pytest.approx(result.final_wealth)

    assert list(result.history.wealth.index) == list(dates[1:])

    assert result.history.wealth.iloc[0] == pytest.approx(100_000)

    assert result.history.wealth.iloc[-1] == pytest.approx(103_020)

    assert result.history.wealth.iloc[-1] == pytest.approx(result.final_wealth)


def test_backtest_history_returns_include_transaction_costs():
    dates = pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
    )

    returns = pd.DataFrame(
        data=[
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
        ],
        index=dates,
        columns=["A", "B"],
    )

    target_weights = pd.DataFrame(
        data=[
            [0.50, 0.50],  # initial portfolio
            [1.00, 0.00],  # target used at the rebalance
            [1.00, 0.00],
        ],
        index=dates,
        columns=["A", "B"],
    )

    result = run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=100_000,
        transaction_cost_rate=0.001,
        rebalance_every=2,
    )

    assert result.history is not None

    assert result.history.portfolio_returns.iloc[-1] == pytest.approx(-0.001)

    assert result.total_turnover == pytest.approx(1.0)

    assert result.final_wealth == pytest.approx(99_900)

    assert result.history.wealth.iloc[-1] == pytest.approx(99_900)


def test_the_added_cash_account_earns_zero_and_is_never_charged():
    # D4 (a): uninvested cash earns 0% (synthetic engineering only); D2: the cash leg is free.
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame(0.01, index=dates, columns=["A", "B"])
    targets = pd.DataFrame(0.5, index=dates, columns=["A", "B"])

    inputs = backtest_inputs(
        returns,
        targets,
        rebalance_every=2,
        transaction_cost_rate=0.001,
    )

    assert inputs.columns == ("A", "B", CASH_SYMBOL)
    assert (inputs.returns[:, 2] == 0.0).all()
    assert list(inputs.cost_rate) == [0.001, 0.001, 0.0]
    assert all(target[2] == 0.0 for target in inputs.targets.values())


def _common_frame():
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    rng = np.random.RandomState(4)
    returns = {symbol: list(rng.normal(0.0, 0.01, 12)) for symbol in ["MU", "SNDK", "MRVL"]}
    return dates, make_momentum_test_frame(dates, ["MU", "SNDK", "MRVL"], returns)


def test_compared_runs_must_share_one_build_date():
    # D3 and D7: on one frame, equal weight would build at the first date and momentum only
    # after its look-back; the guard refuses that comparison.
    dates, frame = _common_frame()
    symbols = ["MU", "SNDK", "MRVL"]
    equal = run_equal_weight_baseline(frame, symbols, rebalance_every=2, transaction_cost_rate=0.0)
    momentum = run_momentum_baseline(
        frame, symbols, lookback=3, rebalance_every=2, transaction_cost_rate=0.0
    )
    assert equal.build_date == dates[0] and momentum.build_date == dates[2]

    with pytest.raises(ScheduleMismatchError, match="different dates"):
        require_common_schedule({"equal_weight": equal, "momentum": momentum})


def test_a_common_build_date_aligns_every_run_and_counts_offsets_from_it():
    dates, frame = _common_frame()
    symbols = ["MU", "SNDK", "MRVL"]
    build = dates[3]
    shared = {"rebalance_every": 2, "transaction_cost_rate": 0.001, "build_date": build}
    runs = {
        "equal_weight": run_equal_weight_baseline(frame, symbols, **shared),
        "momentum": run_momentum_baseline(frame, symbols, lookback=3, **shared),
    }
    require_common_schedule(runs)
    for run in runs.values():
        assert run.build_date == build
        assert run.history.wealth.index[0] == dates[4]
        first = run.schedule.first_decision  # each run's own row of the common build date
        offsets = [e.decision - first for e in run.simulation.executions]
        assert offsets == [0, 2, 4, 6, 8]  # the build, then every 2 dates counted from it


def test_compared_portfolios_must_share_one_rebalance_schedule():
    dates, frame = _common_frame()
    symbols = ["MU", "SNDK", "MRVL"]
    shared = {"transaction_cost_rate": 0.0, "build_date": dates[3]}
    runs = {
        "every_2": run_equal_weight_baseline(frame, symbols, rebalance_every=2, **shared),
        "every_3": run_equal_weight_baseline(frame, symbols, rebalance_every=3, **shared),
    }

    with pytest.raises(ScheduleMismatchError, match="different schedules"):
        require_common_schedule(runs)


def test_a_build_date_outside_the_inputs_is_refused():
    dates, frame = _common_frame()

    with pytest.raises(ValueError, match="build date"):
        run_equal_weight_baseline(
            frame, ["MU", "SNDK", "MRVL"], build_date=dates[0] - pd.Timedelta(days=1)
        )


def test_run_rebalanced_backtest_refuses_a_nonzero_cash_column():
    # D4 (a): a cash column that earns anything but 0% is group C, which is not approved.
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01], CASH_SYMBOL: [0.0, 0.0, 0.0001, 0.0]})
    returns.index = dates
    targets = pd.DataFrame({"A": [0.5] * 4, CASH_SYMBOL: [0.5] * 4}, index=dates)

    with pytest.raises(ValueError, match="group C cash returns are not approved"):
        run_rebalanced_backtest(returns, targets, rebalance_every=2, transaction_cost_rate=0.0)
