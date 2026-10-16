import math

import numpy as np
import pytest

from stock_agent.environment.reward import calculate_reward, log_growth_reward
from stock_agent.portfolio.transition import portfolio_step


def test_reward_without_drawdown_penalty():
    # D8 (approved 2026-10-15): wealth is already net of costs, so no cost is subtracted again.
    result = calculate_reward(
        previous_wealth=100_000,
        current_wealth=102_000,
        drawdown=0.05,
        drawdown_threshold=0.10,
        risk_penalty=2.0,
    )

    expected = math.log(102_000 / 100_000)

    assert result == pytest.approx(expected)


def test_reward_with_drawdown_penalty():
    result = calculate_reward(
        previous_wealth=100_000,
        current_wealth=98_000,
        drawdown=0.15,
        drawdown_threshold=0.10,
        risk_penalty=2.0,
    )

    expected = math.log(98_000 / 100_000) - 2.0 * (0.05**2)

    assert result == pytest.approx(expected)


def test_negative_drawdown_raises_error():
    with pytest.raises(ValueError, match="drawdown"):
        calculate_reward(
            previous_wealth=100_000,
            current_wealth=102_000,
            drawdown=-0.05,
            drawdown_threshold=0.10,
            risk_penalty=2.0,
        )


def test_the_old_path_no_longer_counts_costs_twice():
    # The rebalance example: costs are inside the wealth, so without a drawdown penalty the
    # old path equals the D8 (a) reward, log(1 + R_net) = 0.0037118964.
    step = portfolio_step(
        np.array([0.55, 0.25, 0.20]),
        np.array([0.40, 0.40, 0.20]),
        np.array([0.02, -0.01, 0.0001]),
        np.array([0.001, 0.001, 0.0]),
        2,
    )
    old_path = calculate_reward(
        previous_wealth=100_000,
        current_wealth=100_000 * (1 + step.net_return),
        drawdown=0.0,
        drawdown_threshold=0.10,
        risk_penalty=2.0,
    )

    assert old_path == pytest.approx(log_growth_reward(step), abs=1e-15)
    assert old_path == pytest.approx(0.0037118964, abs=1e-10)
