"""The proposed D8 (a) reward: log(1 + R_net), with costs counted once (spec T8.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.rl import reference_transition as ref
from stock_agent.rl import reward


def test_the_reward_is_log_one_plus_net_return_with_no_second_cost_term():
    result = ref.proposed_portfolio_step(
        np.array([0.55, 0.25, 0.20]),
        np.array([0.40, 0.40, 0.20]),
        np.array([0.02, -0.01, 0.0001]),
        np.array([1e-3, 1e-3, 0.0]),
        2,
    )
    value = reward.log_growth_reward(result)
    assert value == pytest.approx(0.0037118964, abs=1e-10)
    double_counted = math.log1p(result.net_return) - result.cost
    assert abs(value - double_counted) > 1e-4


def test_the_reward_is_labelled_pending_approval():
    assert reward.REWARD_INFO.name == "log_net_growth"
    assert reward.REWARD_INFO.status == "proposed_d8a_pending_owner_approval"
