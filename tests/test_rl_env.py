"""
The allocation environment: timing (D1, L = 1), gaps (D6), the injected
transition, and exact replay.

The replay-equivalence test re-implements the schedule as a plain loop over
the same transition; any policy run through the environment must reproduce
it exactly, so no policy can gain an execution or cost advantage from a
different API.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import scenarios as sc
from stock_agent.rl import transition as tr

STEP = ref.proposed_portfolio_step


def tiny(returns_a, *, rate=0.0):
    sessions = len(returns_a)
    returns = np.column_stack([returns_a, np.zeros(sessions), np.zeros(sessions)])
    return sc.ScenarioData(
        scenario_id="tiny",
        generator_version="test",
        seed=0,
        returns=returns,
        has_close=np.ones((sessions, 3), dtype=bool),
        obs_lag=np.zeros((sessions, 3), dtype=np.int64),
        cost_multiplier=np.ones(sessions),
        regime=np.array(["calm"] * sessions, dtype=object),
        latent_returns=returns.copy(),
        asset_names=("A", "B", "cash"),
        cash_index=2,
        base_cost_rates=(rate, rate, 0.0),
    )


def builder_for(data, spec=None):
    spec = spec or ob.FeatureSpec(return_windows=(1,), vol_windows=(2,), include_cost_regime=False)
    names = spec.market_names(data.asset_names, data.cash_index)
    return ob.ObservationBuilder(
        spec, ob.FeatureNormalizer.identity(names), data.asset_names, data.cash_index
    )


def make_env(data, start=0, stop=None, transition=STEP, **config):
    return rl_env.AllocationEnv(
        data,
        start=start,
        stop=data.sessions if stop is None else stop,
        transition=transition,
        transition_info=ref.TRANSITION_INFO,
        builder=builder_for(data, ob.FeatureSpec() if start >= 60 else None),
        config=rl_env.EnvConfig(**config),
    )


def run(env, targets):
    """Reset, then feed targets in order (None = hold); return observations and infos."""

    obs, info = env.reset()
    observations, infos = [obs], [info]
    for target in targets:
        obs, _, terminated, truncated, info = env.step(target)
        observations.append(obs)
        infos.append(info)
        if terminated or truncated:
            break
    return observations, infos


def test_a_decision_at_d_executes_at_d_plus_1_and_first_earns_d_plus_2_t1_1():
    env = make_env(tiny([0.0, 0.10, -0.05, 0.0]))
    _, info = env.reset()
    assert info["decision_session"] == 0
    env.step(np.array([1.0, 0.0, 0.0]))
    assert env.wealth_path[2] == pytest.approx(0.95, abs=1e-15)
    assert abs(env.wealth_path[2] - 1.045) > 0.09


def test_the_label_tie_enters_at_d1_and_exits_at_d3_t1_2():
    env = make_env(tiny([0.0, 0.10, 0.10, -0.10, 0.0]))
    run(env, [np.array([1.0, 0.0, 0.0]), None, np.array([0.0, 0.0, 1.0])])
    assert env.wealth_path[-1] - 1.0 == pytest.approx(-0.01, abs=1e-12)


def test_the_build_from_cash_pays_trading_costs_d3():
    env = make_env(tiny([0.0, 0.0, 0.0, 0.0], rate=1e-3))
    run(env, [np.array([0.5, 0.5, 0.0]), None])
    assert env.wealth_path[-1] == pytest.approx(0.999, abs=1e-15)
    assert env.ledger[1].turnover_one_way == pytest.approx(1.0)


def test_holding_cash_never_trades():
    env = make_env(tiny([0.0, 0.05, -0.02, 0.03, 0.01]))
    run(env, [None, None, None])
    assert all(row.cost == 0.0 for row in env.ledger)
    np.testing.assert_array_equal(env.wealth_path, np.ones(5))


def test_rewards_telescope_to_log_wealth_since_the_first_execution_close():
    data = sc.generate_scenario("crash", seed=4)
    env = make_env(data, 60, 400)
    rng = np.random.RandomState(1)
    env.reset()
    total, done = 0.0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(rng.dirichlet(np.ones(4)))
        total += reward
        done = terminated or truncated
    first_execution = env.first_decision_index + 1
    expected = math.log(env.wealth_path[-1] / env.wealth_path[first_execution])
    assert total == pytest.approx(expected, rel=1e-12)


def test_the_observation_carries_post_trade_weights_and_drawdown():
    env = make_env(tiny([0.0, 0.10, -0.20, 0.05, 0.0, 0.0]))
    observations, _ = run(env, [np.array([1.0, 0.0, 0.0]), None, None])
    np.testing.assert_array_equal(observations[1].weights, [1.0, 0.0, 0.0])
    wealth = env.wealth_path
    assert observations[3].drawdown == pytest.approx(1.0 - wealth[3] / wealth[:4].max())


def _random_targets(seed, count, n=4):
    rng = np.random.RandomState(seed)
    return [None if rng.rand() < 0.2 else rng.dirichlet(np.ones(n)) for _ in range(count)]


def test_an_observation_at_d_is_unchanged_by_every_return_after_d():
    data = sc.generate_scenario("calm", seed=4)
    targets = _random_targets(2, 400)
    base_obs, base_infos = run(make_env(data, 60, 360), targets)
    j = 50
    d = base_infos[j]["decision_session"]
    future = np.array(data.returns)
    future[d + 1 :, :3] += 0.3
    changed = dataclasses.replace(data, returns=future)
    new_obs, new_infos = run(make_env(changed, 60, 360), targets)
    assert new_infos[j]["decision_session"] == d
    np.testing.assert_array_equal(new_obs[j].vector, base_obs[j].vector)
    assert not np.array_equal(new_obs[j + 1].vector, base_obs[j + 1].vector)


def test_decisions_happen_only_on_complete_sessions_and_spans_cover_gaps_d6():
    data = sc.generate_scenario("missing_closes", seed=1)
    env = make_env(data, 60, 700)
    _, infos = run(env, _random_targets(3, 1000))
    for info in infos:
        assert data.is_complete(info["decision_session"])
    spanning = [row for row in env.ledger if row.next_session - row.session > 1]
    assert spanning
    row = spanning[0]
    span = data.span_returns(row.session, row.next_session)
    expected = row.wealth_before * (1 - row.cost) * (1 + float(row.w_post_trade @ span))
    assert row.wealth_after == pytest.approx(expected, rel=1e-14)


def test_costs_use_the_rate_at_the_execution_session():
    data = sc.generate_scenario("high_costs", seed=0)
    env = make_env(data, 60, 700)
    run(env, [np.full(4, 0.25)] * 1000)
    for row in env.ledger:
        np.testing.assert_array_equal(row.cost_rates, data.cost_rates_at(row.session))
    boundary = [r for r in env.ledger if r.decision_session is not None and r.cost_rates[0] > 1e-3]
    first = boundary[0]
    assert data.cost_multiplier[first.decision_session] == 1.0
    assert data.cost_multiplier[first.session] == 5.0


def test_rebalancing_every_k_sessions_holds_in_between_d7():
    data = sc.generate_scenario("calm", seed=0)
    env = make_env(data, 60, 300, rebalance_every=5, rebalance_offset=2)
    _, infos = run(env, [np.full(4, 0.25)] * 100)
    assert infos[-1]["terminal"] and not any(info["terminal"] for info in infos[:-1])
    indices = [info["decision_index"] for info in infos[:-1]]
    assert indices[0] == 2
    assert all(b - a == 5 for a, b in itertools.pairwise(indices))
    traded = [m for m, row in enumerate(env.ledger) if row.target is not None]
    assert traded == [i + 1 for i in indices]
    assert all(row.target is None for row in env.ledger[:3])


def plain_loop(data, start, stop, transition, targets):
    """The schedule written out by hand: decision j executes at complete session j + 1."""

    complete = data.complete_sessions(start, stop)
    weights = np.zeros(data.n_assets)
    weights[data.cash_index] = 1.0
    wealth = [1.0]
    for m in range(len(complete) - 1):
        target = targets[m - 1] if 1 <= m <= len(targets) else None
        result = transition(
            weights,
            target,
            data.span_returns(int(complete[m]), int(complete[m + 1])),
            data.cost_rates_at(int(complete[m])),
            data.cash_index,
        )
        wealth.append(wealth[-1] * (1.0 + result.net_return))
        weights = result.w_next
    return np.array(wealth)


@pytest.mark.parametrize("scenario", ["calm", "missing_closes", "high_costs"])
def test_replay_through_the_env_equals_a_plain_loop_on_the_same_transition(scenario):
    data = sc.generate_scenario(scenario, seed=6)
    env = make_env(data, 60, 500)
    complete = data.complete_sessions(60, 500)
    targets = _random_targets(9, len(complete) - 2)
    run(env, targets)
    np.testing.assert_array_equal(env.wealth_path, plain_loop(data, 60, 500, STEP, targets))


def test_identical_inputs_replay_to_an_identical_ledger():
    data = sc.generate_scenario("mixed", seed=2)
    targets = _random_targets(4, 500)
    first, second = make_env(data, 60, 500), make_env(data, 60, 500)
    run(first, targets)
    run(second, targets)
    assert len(first.ledger) == len(second.ledger)
    for a, b in zip(first.ledger, second.ledger, strict=True):
        assert a.session == b.session and a.cost == b.cost and a.net_return == b.net_return
        np.testing.assert_array_equal(a.w_next, b.w_next)


def test_a_transition_that_breaks_the_contract_stops_the_episode():
    def broken(w_pre, target, r_next, cost_rate, cash_index):
        result = STEP(w_pre, target, r_next, cost_rate, cash_index)
        return dataclasses.replace(result, w_next=result.w_next * 1.1)

    env = make_env(tiny([0.0, 0.01, 0.02, 0.0]), transition=broken)
    with pytest.raises(tr.TransitionContractError):
        env.reset()


def test_stepping_after_the_window_ends_is_refused():
    env = make_env(tiny([0.0, 0.01, 0.02, 0.0]))
    run(env, [None, None])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(None)


def test_a_window_with_fewer_than_three_complete_sessions_is_refused():
    with pytest.raises(ValueError, match="three complete sessions"):
        make_env(tiny([0.0, 0.01]))


def test_the_info_records_the_transition_and_reward_identity():
    env = make_env(tiny([0.0, 0.01, 0.02, 0.0]))
    _, info = env.reset()
    assert info["transition"]["status"] == tr.ENGINEERING_ONLY
    assert info["reward"]["name"] == "log_net_growth"
