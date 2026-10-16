"""
Replay equivalence of the RL environment against Track A's independent oracle.

`tests/reference/t5a_reference.py` is a loop-based shares-and-cash ledger of
the approved D1-D8 rules that imports nothing from `stock_agent`. Driving
the RL environment (with the approved `portfolio_step`) through Track A's
replay harness, and matching the oracle's wealth path within 1e-12 on random
paths with missing closes, cross-checks the environment's timing, gap and
cost scheduling and the transition's arithmetic against an independent
implementation.

Scenario scope (the RL rules): decisions only on complete sessions, every
fund held once built (strictly positive targets), and the session after the
build decision complete.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference.replay import ReplayScenario, oracle_runner, replay
from t5a_runners import rl_env_runner

from stock_agent.rl import transition as tr


def random_scenario(seed: int, sessions: int = 80, gap_probability: float = 0.08):
    rng = np.random.RandomState(seed)
    n, cash = 4, 3
    returns = np.zeros((sessions, n))
    returns[:, :cash] = np.expm1(rng.normal(0.0003, 0.015, size=(sessions, cash)))
    returns[:, cash] = 0.0001
    missing = rng.random_sample((sessions, cash)) < gap_probability
    missing[:3] = False
    closes = np.ones((sessions, n), dtype=bool)
    closes[:, :cash] = ~missing
    observed = np.full((sessions, n), np.nan)
    for i in range(n):
        growth = 1.0
        for t in range(1, sessions):
            growth *= 1.0 + returns[t, i]
            if closes[t, i]:
                observed[t, i] = growth - 1.0
                growth = 1.0
    complete = [t for t in range(1, sessions) if closes[t].all()]
    decisions = {}
    for position, t in enumerate(complete[:-2]):
        if position == 0 or rng.rand() < 0.5:
            decisions[t] = list(rng.dirichlet(np.ones(n)))
        else:
            decisions[t] = None
    return ReplayScenario(
        returns=observed,
        targets=decisions,
        cash_index=cash,
        cost_rate=(5e-4, 5e-4, 2e-4, 0.0),
        lag=1,
        has_close=closes,
    )


@pytest.mark.parametrize("seed", range(8))
def test_the_rl_environment_replays_the_oracle_wealth_path(seed):
    (report,) = replay(
        {"oracle": oracle_runner, "rl_env": rl_env_runner},
        random_scenario(seed),
        reference="oracle",
    ).values()
    assert report.equivalent, report.detail


def test_the_scenarios_exercise_gaps_and_holds():
    scenario = random_scenario(0)
    assert (~scenario.has_close).any()
    assert any(target is None for target in scenario.targets.values())
    path = oracle_runner(scenario)
    assert any(value is None for value in path.wealth)


def test_a_planted_cost_defect_is_detected():
    def double_cost(w_pre, target, r_next, cost_rate, cash_index):
        return tr.portfolio_step(w_pre, target, r_next, 2 * cost_rate, cash_index)

    (report,) = replay(
        {
            "oracle": oracle_runner,
            "rl_env": lambda scenario: rl_env_runner(scenario, transition=double_cost),
        },
        random_scenario(1),
        reference="oracle",
    ).values()
    assert not report.equivalent
