from __future__ import annotations

import numpy as np


def validate_weights(weights: np.ndarray) -> None:
    """Validate a long-only portfolio weight vector."""

    if weights.ndim != 1:
        raise ValueError(
            f"Portfolio weights must be one-dimensional."
            f"Received shape {weights.shape}"
        )

    if np.any(weights < 0):
        raise ValueError(
            "Portfolio weights cannot be negative in the V1 long-only environment."
        )

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"Portfolio weights must sum to 1."
            f"Received {weights.sum():.8f}."
        )

def calculate_turnover(current_weights: np.ndarray,
                       target_weights: np.ndarray,
) -> float:
    """Calculate portfolio turnover when moving from current to target weights."""
    if current_weights.shape != target_weights.shape:
        raise ValueError(
            "current_weights and target_weights must have the same shape."
        )

    validate_weights(current_weights)
    validate_weights(target_weights)

    weight_difference = current_weights - target_weights

    return float(np.sum(np.abs(weight_difference)))


def calculate_portfolio_return(
    asset_weights: np.ndarray,
    asset_returns: np.ndarray,
) -> float:
    """Calculate one-period portfolio return."""

    if len(asset_weights) != len(asset_returns):
        raise ValueError(
            "Asset weights and asset returns must have equal length."
        )

    return float(np.dot(asset_weights, asset_returns))

def calculate_transaction_cost(
        turnover: float,
        transaction_cost_rate: float,
) -> float: 
    """Calculate transaction cost as a fraction of portfolio wealth."""
    if turnover < 0:
        raise ValueError("Turnover cannot be negative")

    if transaction_cost_rate < 0:
        raise ValueError("Transaction cost rate cannot be negative.")

    return turnover * transaction_cost_rate

def calculate_drawdown(
        wealth: float,
        peak_wealth: float,
) -> float:
    """Return drawdown as a positive fraction."""

    if wealth <= 0:
        raise ValueError("Wealth mus be positive.")

    if peak_wealth <= 0:
        raise ValueError("Peak wealth must be positive.")

    if wealth > peak_wealth:
        return 0.0

    return 1.0 - wealth / peak_wealth

def update_wealth(
    current_wealth: float,
    portfolio_return: float,
    transaction_cost: float,
) -> float:
    """Update portfolio wealth after return and trading costs."""

    if current_wealth <= 0:
        raise ValueError("Current wealth must be positive.")

    if transaction_cost < 0:
        raise ValueError("Transaction cost cannot be negative.")

    net_return = portfolio_return - transaction_cost
    new_wealth = current_wealth * (1.0 + net_return)

    if new_wealth <= 0:
        raise ValueError(
            "Portfolio wealth became non-positive after applying return and costs."
        )

    return float(new_wealth)