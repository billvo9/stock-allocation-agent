from __future__ import annotations

import math


def calculate_reward(
    previous_wealth: float,
    current_wealth: float,
    turnover: float,
    transaction_cost_rate: float,
    drawdown: float,
    drawdown_threshold: float,
    risk_penalty: float,
) -> float:
    """
    Calculate the risk-adjusted portfolio reward.
    """

    if previous_wealth <= 0:
        raise ValueError("previous_wealth must larger than 0")

    if current_wealth <= 0:
        raise ValueError("current_wealth must larger than 0")

    if turnover < 0:
        raise ValueError("Turnover must larger than or equal to 0")

    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate must larger than or equal to 0")

    if drawdown < 0:
        raise ValueError("drawdown must larger than or equal to 0")

    if drawdown_threshold < 0:
        raise ValueError("drawdown_threshold must larger than or equal to 0")

    if risk_penalty < 0:
        raise ValueError("risk_penalty must larger than or equal to 0")

    log_return = math.log(current_wealth / previous_wealth)
    transaction_penalty = transaction_cost_rate * turnover
    excess_drawdown = max(0.0, drawdown - drawdown_threshold)
    drawdown_penalty = risk_penalty

    reward = log_return - transaction_penalty - drawdown_penalty * (excess_drawdown) ** 2

    return reward
