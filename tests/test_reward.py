"""
The D8 (a) reward and the inputs it rests on.

`calculate_reward` was removed (D8 (a) is the only approved reward), and its
tests with it. Its input checks live on the approved path: a negative or
out-of-range turnover is refused by `StepResult`, `check_step_result` and
`Constraints`, and a drawdown needs positive wealth.
"""

import dataclasses
import math

import numpy as np
import pytest

from stock_agent.environment import reward
from stock_agent.portfolio.backtest_summary import max_drawdown
from stock_agent.portfolio.transition import Constraints, StepResult, portfolio_step
from stock_agent.rl import transition as rl_transition

REBALANCE = (
    np.array([0.55, 0.25, 0.20]),
    np.array([0.40, 0.40, 0.20]),
    np.array([0.02, -0.01, 0.0001]),
    np.array([0.001, 0.001, 0.0]),
    2,
)


def test_the_reward_is_log_growth_with_costs_counted_once():
    step = portfolio_step(*REBALANCE)

    assert reward.log_growth_reward(step) == pytest.approx(0.0037118964, abs=1e-10)
    assert reward.log_growth_reward(step) == pytest.approx(math.log(1 + step.net_return))
    assert abs(reward.log_growth_reward(step) - 0.0034118964) > 1e-4  # cost subtracted twice


def test_the_unapproved_penalised_reward_is_gone():
    assert not hasattr(reward, "calculate_reward")


def test_a_negative_turnover_is_refused_on_the_approved_path():
    step = portfolio_step(*REBALANCE)

    with pytest.raises(ValueError, match="non-negative"):
        dataclasses.replace(step, turnover_one_way=-0.10)

    with pytest.raises(ValueError, match="non-negative"):
        dataclasses.replace(step, traded_notional=-0.10)


def test_a_negative_turnover_from_an_injected_transition_is_refused():
    step = portfolio_step(*REBALANCE)
    # Bypass StepResult's own check, as a defective injected transition might.
    broken = object.__new__(StepResult)
    for field in dataclasses.fields(StepResult):
        object.__setattr__(broken, field.name, getattr(step, field.name))
    object.__setattr__(broken, "turnover_one_way", -0.10)

    with pytest.raises(rl_transition.TransitionContractError, match="non-negative"):
        rl_transition.check_step_result(broken, n_assets=3)


@pytest.mark.parametrize("tau_max", [-0.10, 0.0, 1.5])
def test_an_out_of_range_turnover_limit_is_refused(tau_max):
    with pytest.raises(ValueError, match="tau_max"):
        Constraints(2, tau_max=tau_max)


@pytest.mark.parametrize("wealth", [[100.0, -5.0], [100.0, 0.0], [100.0, float("inf")]])
def test_a_drawdown_needs_positive_finite_wealth(wealth):
    with pytest.raises(ValueError, match="positive"):
        max_drawdown(wealth)
