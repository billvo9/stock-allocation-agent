from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.agents.equal_weight import equal_weight
from stock_agent.environment.portfolio_math import (
    calculate_drawdown,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    update_wealth,
)


def prepare_return_matrix(
    frame: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Convert long-format feature data into date x symbol returns."""

    filtered = frame[frame["symbol"].isin(symbols)][["date", "symbol", "daily_return"]]

    returns = filtered.pivot(
        index="date",
        columns="symbol",
        values="daily_return",
    ).sort_index()

    missing_symbols = set(symbols) - set(returns.columns)

    if missing_symbols:
        raise ValueError(f"missing required symbols: {sorted(missing_symbols)}")

    returns = returns[symbols]

    if returns.isna().any().any():
        raise ValueError("Return matrix contains missing values.")

    return returns


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


def run_equal_weight_baseline(
    frame: pd.DataFrame,
    symbols: list[str],
    initial_wealth: float = 100_000.0,
    rebalance_every: int = 5,
    transaction_cost_rate: float = 0.001,
) -> BaselineResult:
    """Backtest a periodically rebalanced equal-weight strategy.

    The first available date initializes the portfolio. Its daily return is not
    applied because that return represents a period before the portfolio exists.
    """
    if initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive.")

    if rebalance_every <= 0:
        raise ValueError("Rebalance interval must be positive.")

    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")

    returns = prepare_return_matrix(
        frame=frame,
        symbols=symbols,
    )

    if returns.empty:
        raise ValueError("Return matrix cannot be empty.")

    if len(returns) < 2:
        raise ValueError("Return matrix must contain at least two dates.")

    target_map = equal_weight(symbols)

    target_weights = np.array(
        [target_map[symbol] for symbol in symbols],
        dtype=float,
    )

    current_weights = target_weights.copy()

    wealth = initial_wealth
    peak_wealth = initial_wealth

    max_drawdown = 0.0
    total_turnover = 0.0

    return_matrix = returns.iloc[1:].to_numpy(dtype=float)

    for step, asset_returns in enumerate(return_matrix):
        transaction_cost = 0.0

        # Rebalance at the beginning of every rebalance interval.
        # Step 0 is excluded because the portfolio already starts
        # at the target equal weights.
        if step > 0 and step % rebalance_every == 0:
            turnover = calculate_turnover(
                current_weights=current_weights,
                target_weights=target_weights,
            )

            transaction_cost = calculate_transaction_cost(
                turnover=turnover,
                transaction_cost_rate=transaction_cost_rate,
            )

            total_turnover += turnover

            current_weights = target_weights.copy()

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
            weights=current_weights,
            asset_returns=asset_returns,
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
