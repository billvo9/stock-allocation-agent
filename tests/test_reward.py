import math

import pytest

from stock_agent.environment.reward import calculate_reward


def test_reward_without_drawdown_penalty():
    result = calculate_reward(
        previous_wealth=100_000,
        current_wealth=102_000,
        turnover=0.20,
        transaction_cost_rate=0.001,
        drawdown=0.05,
        drawdown_threshold=0.10,
        risk_penalty=2.0,
    )

    expected = math.log(102_000 / 100_000) - (0.001 * 0.20)

    assert result == pytest.approx(expected)


def test_reward_with_drawdown_penalty():
    result = calculate_reward(
        previous_wealth=100_000,
        current_wealth=98_000,
        turnover=0.10,
        transaction_cost_rate=0.001,
        drawdown=0.15,
        drawdown_threshold=0.10,
        risk_penalty=2.0,
    )

    expected = (
        math.log(98_000 / 100_000)
        - (0.001 * 0.10)
        - 2.0 * (0.05**2)
    )

    assert result == pytest.approx(expected)

def test_negative_turnover_raises_error():
    with pytest.raises(ValueError, match="Turnover"):
        calculate_reward(
            previous_wealth=100_000,
            current_wealth=102_000,
            turnover=-0.10,
            transaction_cost_rate=0.001,
            drawdown=0.05,
            drawdown_threshold=0.10,
            risk_penalty=2.0,
        )

    

