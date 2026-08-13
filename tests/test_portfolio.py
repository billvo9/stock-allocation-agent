import numpy as np
import pytest

from stock_agent.environment.portfolio_math import (
    calculate_drawdown,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    validate_weights,
)


def test_valid_weights():
    weights = np.array([0.30, 0.25, 0.25, 0.20])

    validate_weights(weights)

def test_weights_must_sum_to_one():
    weights = np.array([0.30, 0.25, 0,25, 0.10])

    with pytest.raises(ValueError, match="must sum to 1"):
        validate_weights(weights)

def test_calculate_turnover():
    current = np.array([0.40, 0.30, 0.20, 0.10])
    target = np.array([0.30, 0.30, 0.30, 0.10])

    result = calculate_turnover(current, target)

    assert np.isclose(result, 0.20)

def test_portfolio_return():
    weights = np.array([0.50, 0.30, 0.20])
    returns = np.array([0.02, -0.01, 0.01])

    result = calculate_portfolio_return(
        weights,
        returns,
    )

    assert np.isclose(result, 0.009)


def test_transaction_cost():
    result = calculate_transaction_cost(
        turnover=0.20,
        transaction_cost_rate=0.001,
    )

    assert np.isclose(result, 0.0002)

def test_drawdown():
    result = calculate_drawdown(
        wealth=90_000,
        peak_wealth=100_000,
    )
    
    assert np.isclose(result, 0.10)