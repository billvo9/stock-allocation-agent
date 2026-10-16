"""
The RL reward: decision D8 (a), approved by the owner on 2026-10-15.

    reward = log(1 + R_net)

R_net comes from the injected transition and already contains the trading
cost, so there is no separate cost term (subtracting the cost again would
count it twice). Any other reward (drawdown or variance penalties) is an
owner-approved variant registered before training, never a silent change;
D8 (b) and (c) are not approved. The same formula is
`stock_agent.environment.reward.log_growth_reward`, which this package may
not import (tests check that the two agree).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stock_agent.rl.transition import StepResult


@dataclass(frozen=True)
class RewardInfo:
    name: str
    version: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "status": self.status}


REWARD_INFO = RewardInfo(name="log_net_growth", version="1.0.0", status="approved")


def log_growth_reward(result: StepResult) -> float:
    return math.log1p(result.net_return)
