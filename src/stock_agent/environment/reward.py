from __future__ import annotations

import math

from stock_agent.portfolio.transition import PENDING_APPROVAL, StepResult


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

    Known defect (spec section 8): the wealth passed in is already net of
    costs, so subtracting `transaction_cost_rate * turnover` again counts
    costs twice. Replaced by `log_growth_reward` once D8 is approved.
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


def log_growth_reward(step: StepResult) -> float:
    """
    The RL reward contract proposed in D8 (a): log(1 + step.net_return).
    Costs are already inside `net_return`, so there is no separate cost
    term. PENDING the owner's approval of OD8 group B.
    """

    raise NotImplementedError(PENDING_APPROVAL)
