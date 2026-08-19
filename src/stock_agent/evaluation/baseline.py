from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.agents.equal_weight import equal_weight
from stock_agent.agents.inverse_volatility import inverse_volatility
from stock_agent.environment.portfolio_math import (
    calculate_drawdown,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    update_wealth,
)


def prepare_feature_matrix(
    frame: pd.DataFrame,
    symbols: list[str],
    column: str,
) -> pd.DataFrame:
    """Convert a long-format feature column into a date x symbol matrix."""

    filtered = frame[frame["symbol"].isin(symbols)][["date", "symbol", column]]

    matrix = filtered.pivot(
        index="date",
        columns="symbol",
        values=column,
    ).sort_index()

    missing_symbols = set(symbols) - set(matrix.columns)

    if missing_symbols:
        raise ValueError(f"missing required symbols: {sorted(missing_symbols)}")

    matrix = matrix[symbols]

    if matrix.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    return matrix


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    """
    Update portfolio weights after asset returns occur.
    """
    if weights.shape != asset_returns.shape:
        raise ValueError("Weights and asset returns must have the same shape.")

    post_return_values = weights * (1.0 + asset_returns)

    portfolio_total = np.sum(post_return_values)

    if portfolio_total <= 0:
        raise ValueError("Portfolio value after asset returns must be positive.")

    return post_return_values / portfolio_total


@dataclass(frozen=True)
class BaselineResult:
    initial_wealth: float
    final_wealth: float
    cumulative_return: float
    max_drawdown: float
    total_turnover: float


def update_peak_and_drawdown(
    wealth: float,
    peak_wealth: float,
) -> tuple[float, float]:
    """
    Update running peak wealth and calculate current drawdown.
    """
    new_peak = max(peak_wealth, wealth)
    drawdown = 0.0

    if peak_wealth > wealth:
        drawdown = calculate_drawdown(wealth, peak_wealth)

    return (new_peak, drawdown)


def weights_to_array(
    weights: dict[str, float],
    symbols: list[str],
) -> np.ndarray:
    missing_symbols = set(symbols) - set(weights)

    if missing_symbols:
        raise ValueError(f"Missing weights for symbols: {sorted(missing_symbols)}")

    return np.array(
        [weights[symbol] for symbol in symbols],
        dtype=float,
    )


def inverse_volatility_weights(
    volatility_row: pd.Series,
    symbols: list[str],
) -> np.ndarray:
    missing_symbols = set(symbols) - set(volatility_row.index)

    if missing_symbols:
        raise ValueError(f"Missing volatility values for symbols: {sorted(missing_symbols)}")

    volatility_map = {symbol: float(volatility_row[symbol]) for symbol in symbols}

    inverse_volatility_dict = inverse_volatility(volatility_map)

    return weights_to_array(
        inverse_volatility_dict,
        symbols,
    )


def prepare_inverse_volatility_targets(
    frame: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    volatility_matrix = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="volatility_20d",
    )

    target_rows = []

    for _, volatility_row in volatility_matrix.iterrows():
        weights = inverse_volatility_weights(volatility_row, symbols)
        target_rows.append(weights)

    targets = pd.DataFrame(
        data=target_rows,
        index=volatility_matrix.index,
        columns=symbols,
    )

    return targets


def run_rebalanced_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
    first_rebalance_step: int | None = None,
) -> BaselineResult:
    if initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive.")

    if rebalance_every <= 0:
        raise ValueError("Rebalance interval must be positive.")

    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")

    if not returns.index.equals(target_weights.index):
        raise ValueError("Returns and target weights must have matching dates.")

    if list(returns.columns) != list(target_weights.columns):
        raise ValueError("Returns and target weights must have matching symbols.")

    if len(returns) < 2:
        raise ValueError("Backtest requires at least two dates.")

    if first_rebalance_step is None:
        first_rebalance_step = rebalance_every

    if first_rebalance_step <= 0:
        raise ValueError("First rebalance step must be positive.")

    current_weights = target_weights.iloc[0].to_numpy(dtype=float)

    wealth = initial_wealth
    peak_wealth = initial_wealth
    max_drawdown = 0.0
    total_turnover = 0.0

    for step in range(1, len(returns)):
        transaction_cost = 0.0

        should_rebalance = (
            step >= first_rebalance_step and (step - first_rebalance_step) % rebalance_every == 0
        )

        if should_rebalance:
            rebalance_target_weights = target_weights.iloc[step - 1].to_numpy(dtype=float)

            turnover = calculate_turnover(
                current_weights=current_weights,
                target_weights=rebalance_target_weights,
            )

            transaction_cost = calculate_transaction_cost(
                turnover=turnover,
                transaction_cost_rate=transaction_cost_rate,
            )

            total_turnover += turnover

            current_weights = rebalance_target_weights.copy()

        asset_returns = returns.iloc[step].to_numpy(dtype=float)

        portfolio_return = calculate_portfolio_return(
            asset_weights=current_weights,
            asset_returns=asset_returns,
        )

        wealth = update_wealth(
            current_wealth=wealth,
            portfolio_return=portfolio_return,
            transaction_cost=transaction_cost,
        )

        current_weights = drift_weights(
            current_weights,
            asset_returns,
        )

        peak_wealth, current_drawdown = update_peak_and_drawdown(
            wealth=wealth,
            peak_wealth=peak_wealth,
        )

        max_drawdown = max(
            max_drawdown,
            current_drawdown,
        )

    cumulative_return = wealth / initial_wealth - 1.0

    return BaselineResult(
        initial_wealth=initial_wealth,
        final_wealth=wealth,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        total_turnover=total_turnover,
    )


def run_inverse_volatility_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
) -> BaselineResult:
    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    target_weights = prepare_inverse_volatility_targets(
        frame=frame,
        symbols=symbols,
    )

    # Your part:
    return run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        first_rebalance_step=rebalance_every,
    )


def prepare_equal_weight_targets(
    returns: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    target_map = equal_weight(symbols)

    target_array = weights_to_array(
        target_map,
        symbols,
    )

    # 1. repeat target_array once for every date in returns
    # 2. build a DataFrame
    # 3. use returns.index as the index
    # 4. use symbols as the columns
    # 5. return the DataFrame
    return pd.DataFrame(
        data=np.tile(target_array, (len(returns), 1)), index=returns.index, columns=symbols
    )


def run_equal_weight_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
) -> BaselineResult:
    returns = prepare_feature_matrix(
        frame=frame,
        symbols=symbols,
        column="daily_return",
    )

    target_weights = prepare_equal_weight_targets(
        returns=returns,
        symbols=symbols,
    )

    return run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=rebalance_every,
        transaction_cost_rate=transaction_cost_rate,
        first_rebalance_step=rebalance_every + 1,
    )
