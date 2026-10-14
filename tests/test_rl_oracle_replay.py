"""
Replay equivalence of the RL environment against Track A's independent oracle.

`tests/reference/t5a_reference.py` is a loop-based shares-and-cash ledger of
the recommended D1-D8 rules that imports nothing from `stock_agent`. Driving
the RL environment (with the engineering stand-in transition) through
Track A's replay harness, and matching the oracle's wealth path within 1e-12
on random paths with missing closes, cross-checks the environment's timing,
gap and cost scheduling and the stand-in's arithmetic against an independent
implementation.

Scenario scope (the RL rules): decisions only on complete sessions, every
fund held once built (strictly positive targets), and the session after the
build decision complete.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference.replay import ReplayPath, ReplayScenario, oracle_runner, replay

from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import replay as rl_replay
from stock_agent.rl import scenarios as sc


class SessionScript:
    """The target the scenario decided at each session (None: hold)."""

    name = "session_script"

    def __init__(self, targets):
        self.targets = targets

    def reset(self):
        pass

    def decide(self, observation):
        target = self.targets.get(observation.decision_session)
        return None if target is None else np.asarray(target, dtype=np.float64)


def rl_env_runner(scenario: ReplayScenario, transition=ref.proposed_portfolio_step) -> ReplayPath:
    returns = np.array(scenario.returns, dtype=np.float64)
    sessions, n = returns.shape
    closes = np.isfinite(returns) if scenario.has_close is None else np.array(scenario.has_close)
    closes[0] = True
    observed = np.where(closes, returns, np.nan)
    observed[0] = 0.0
    names = tuple(f"asset_{i}" for i in range(n))
    data = sc.ScenarioData(
        scenario_id="oracle_replay",
        generator_version="replay",
        seed=0,
        returns=observed,
        has_close=closes,
        obs_lag=np.zeros((sessions, n), dtype=np.int64),
        cost_multiplier=np.ones(sessions),
        regime=np.array(["replay"] * sessions, dtype=object),
        latent_returns=np.nan_to_num(observed),
        asset_names=names,
        cash_index=scenario.cash_index,
        base_cost_rates=tuple(float(rate) for rate in scenario.cost_rate),
    )
    spec = ob.FeatureSpec(return_windows=(1,), vol_windows=(2,), include_cost_regime=False)
    builder = ob.ObservationBuilder(
        spec,
        ob.FeatureNormalizer.identity(spec.market_names(names, scenario.cash_index)),
        names,
        scenario.cash_index,
    )
    env = rl_env.AllocationEnv(
        data,
        start=min(scenario.targets),
        stop=sessions,
        transition=transition,
        transition_info=ref.TRANSITION_INFO,
        builder=builder,
    )
    rl_replay.rollout(env, SessionScript(scenario.targets))
    valued = dict(zip(env.complete.tolist(), env.wealth_path.tolist(), strict=False))
    execution = int(env.complete[env.first_decision_index + 1])
    scale = scenario.initial_wealth / valued[execution]
    path = tuple(range(execution, sessions))
    return ReplayPath(path, tuple(valued[t] * scale if t in valued else None for t in path))


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
        return ref.proposed_portfolio_step(w_pre, target, r_next, 2 * cost_rate, cash_index)

    (report,) = replay(
        {
            "oracle": oracle_runner,
            "rl_env": lambda scenario: rl_env_runner(scenario, transition=double_cost),
        },
        random_scenario(1),
        reference="oracle",
    ).values()
    assert not report.equivalent
