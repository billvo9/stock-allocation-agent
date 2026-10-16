"""
Replay adapters for the production simulators (test helpers, not tests).

Each adapter puts one simulator behind the `EpisodeRunner` seam of
`tests/reference/replay.py`: it takes a `ReplayScenario` (returns, the
target decided at each session, lag, cost rates, constraints) and returns
the pre-trade wealth at each valued close. The reference oracle's adapter,
`oracle_runner`, lives with the oracle.

- `engine_runner`: `portfolio.engine.simulate`, the backtest engine.
- `rl_env_runner`: `rl.env.AllocationEnv` driven by a policy that replays the
  scenario's targets through the environment's own step API.

Both run the approved `portfolio_step` unless a mutant is injected.
"""

from __future__ import annotations

import numpy as np
from reference.replay import ReplayPath, ReplayScenario

from stock_agent.portfolio import engine
from stock_agent.portfolio.transition import Constraints, portfolio_step
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import replay as rl_replay
from stock_agent.rl import scenarios as sc
from stock_agent.rl import transition as tr


def engine_runner(scenario: ReplayScenario, transition=portfolio_step) -> ReplayPath:
    """The backtest engine behind the replay harness's `EpisodeRunner` seam."""

    result = engine.simulate(
        np.asarray(scenario.returns, dtype=float),
        {
            d: None if target is None else np.asarray(target, dtype=float)
            for d, target in scenario.targets.items()
        },
        lag=scenario.lag,
        cost_rate=np.asarray(scenario.cost_rate, dtype=float),
        constraints=Constraints(scenario.cash_index, scenario.w_max, scenario.tau_max),
        initial_wealth=scenario.initial_wealth,
        has_close=scenario.has_close,
        transition=transition,
    )
    return ReplayPath(tuple(result.sessions), tuple(result.wealth), result.status, result.reason)


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


def rl_env_runner(
    scenario: ReplayScenario, transition=portfolio_step, info=tr.APPROVED_TRANSITION
) -> ReplayPath:
    """
    The RL environment behind the `EpisodeRunner` seam. Scope (the RL rules):
    decisions only on complete sessions, every fund held once built, and the
    session after the build decision complete.
    """

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
        transition_info=info,
        builder=builder,
        config=rl_env.EnvConfig(max_weight=scenario.w_max, max_turnover=scenario.tau_max),
    )
    rl_replay.rollout(env, SessionScript(scenario.targets))
    valued = dict(zip(env.complete.tolist(), env.wealth_path.tolist(), strict=False))
    execution = int(env.complete[env.first_decision_index + 1])
    scale = scenario.initial_wealth / valued[execution]
    path = tuple(range(execution, sessions))
    return ReplayPath(path, tuple(valued[t] * scale if t in valued else None for t in path))
