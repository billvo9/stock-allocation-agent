"""The approved D8 (a) reward: log(1 + R_net), with costs counted once (spec T8.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.environment import reward as engine_reward
from stock_agent.rl import reward
from stock_agent.rl import transition as tr


def test_the_reward_is_log_one_plus_net_return_with_no_second_cost_term():
    result = tr.portfolio_step(
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


def test_the_reward_is_labelled_approved():
    assert reward.REWARD_INFO.name == "log_net_growth"
    assert reward.REWARD_INFO.status == "approved"


def test_the_rl_reward_and_the_engine_reward_are_one_formula():
    rng = np.random.RandomState(8)
    for _ in range(200):
        w_pre, target = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        step = tr.portfolio_step(
            w_pre, target, rng.uniform(-0.2, 0.2, 3), np.array([1e-3, 5e-4, 0.0]), 2
        )
        assert reward.log_growth_reward(step) == engine_reward.log_growth_reward(step)
