"""
The RL reward, D8 (a) (approved by the owner on 2026-10-15).

`calculate_reward`, a penalised reward that subtracted trading costs a
second time, was removed: D8 approves only log(1 + R_net), and its options
(b) and (c) are not approved.
"""

from __future__ import annotations

import math

from stock_agent.portfolio.transition import StepResult


def log_growth_reward(step: StepResult) -> float:
    """
    The RL reward, D8 (a): log(1 + step.net_return). Costs are already
    inside `net_return`, so there is no separate cost term.
    """

    return math.log1p(step.net_return)
