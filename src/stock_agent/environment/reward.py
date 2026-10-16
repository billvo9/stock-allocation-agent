from __future__ import annotations

import math

from stock_agent.portfolio.transition import StepResult


def calculate_reward(
    previous_wealth: float,
    current_wealth: float,
    drawdown: float,
    drawdown_threshold: float,
    risk_penalty: float,
) -> float:
    """
    The log wealth ratio less a squared excess-drawdown penalty.

    Wealth here is already net of trading costs (`portfolio_step`), so no
    cost term is subtracted: the D8 decision (approved 2026-10-15) forbids
    counting already incurred costs twice. This penalised form is NOT the
    approved reward: D8 (a), `log_growth_reward`, is, and D8 (b) and (c)
    are not approved. Nothing in production calls this function.
    """

    if previous_wealth <= 0:
        raise ValueError("previous_wealth must larger than 0")

    if current_wealth <= 0:
        raise ValueError("current_wealth must larger than 0")

    if drawdown < 0:
        raise ValueError("drawdown must larger than or equal to 0")

    if drawdown_threshold < 0:
        raise ValueError("drawdown_threshold must larger than or equal to 0")

    if risk_penalty < 0:
        raise ValueError("risk_penalty must larger than or equal to 0")

    log_return = math.log(current_wealth / previous_wealth)
    excess_drawdown = max(0.0, drawdown - drawdown_threshold)

    return log_return - risk_penalty * excess_drawdown**2


def log_growth_reward(step: StepResult) -> float:
    """
    The RL reward, D8 (a) (approved 2026-10-15): log(1 + step.net_return).
    Costs are already inside `net_return`, so there is no separate cost
    term.
    """

    return math.log1p(step.net_return)
