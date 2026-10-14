"""Rule-based baselines and the rollout record, all on the identical environment."""

from __future__ import annotations

import numpy as np
import pytest

from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import optimizer as opt
from stock_agent.rl import policies as pol
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import replay
from stock_agent.rl import scenarios as sc

SPEC = ob.FeatureSpec()


def make_env(data, start=60, stop=360, **config):
    names = SPEC.market_names(data.asset_names, data.cash_index)
    builder = ob.ObservationBuilder(
        SPEC, ob.FeatureNormalizer.identity(names), data.asset_names, data.cash_index
    )
    return rl_env.AllocationEnv(
        data,
        start=start,
        stop=stop,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        builder=builder,
        config=rl_env.EnvConfig(**config),
    )


@pytest.fixture(scope="module")
def calm():
    return sc.generate_scenario("calm", seed=8)


@pytest.fixture(scope="module")
def first_obs(calm):
    obs, _ = make_env(calm).reset()
    return obs


def test_buy_and_hold_trades_once_then_holds(calm):
    record = replay.rollout(make_env(calm), pol.buy_and_hold())
    targets = [d.target for d in record.decisions]
    np.testing.assert_array_equal(targets[0], pol.FIXED_BALANCED_WEIGHTS)
    assert all(t is None for t in targets[1:])
    assert sum(row.cost > 0 for row in record.ledger) == 1


def test_buy_and_hold_rearms_on_reset(calm):
    policy = pol.buy_and_hold()
    replay.rollout(make_env(calm), policy)
    again = replay.rollout(make_env(calm), policy)
    assert again.decisions[0].target is not None


def test_constant_mixes_rebalance_to_their_registered_weights(first_obs):
    np.testing.assert_allclose(pol.equal_weight().decide(first_obs), [1 / 3, 1 / 3, 1 / 3, 0.0])
    np.testing.assert_array_equal(pol.fixed_balanced().decide(first_obs), [0.6, 0.3, 0.1, 0.0])


def test_inverse_volatility_weights_by_hand(first_obs):
    window = first_obs.snapshot.returns[-60:, :3]
    inverse = 1.0 / window.std(axis=0, ddof=1)
    expected = np.append(inverse / inverse.sum(), 0.0)
    np.testing.assert_allclose(pol.InverseVolatility().decide(first_obs), expected, rtol=1e-12)


def test_inverse_volatility_refuses_too_little_history(calm):
    builder = make_env(calm).builder
    early = builder.build(ob.snapshot_at(calm, 10), np.eye(4)[3], 0.0)
    with pytest.raises(pol.PolicyInputError):
        pol.InverseVolatility().decide(early)


def test_mean_variance_solves_on_the_same_point_in_time_window(first_obs):
    policy = pol.MeanVariance(risk_aversion=5.0, window=60)
    window = first_obs.snapshot.returns[-60:]
    mu = window.mean(axis=0)
    cov = np.zeros((4, 4))
    cov[:3, :3] = np.cov(window[:, :3], rowvar=False, ddof=1)
    cov[:3, :3] += policy.ridge * np.trace(cov[:3, :3]) / 3 * np.eye(3)
    expected = opt.solve_mean_variance(mu, cov, 5.0).weights
    np.testing.assert_allclose(policy.decide(first_obs), expected, atol=1e-12)


def test_mean_variance_shrinks_means_toward_zero_when_registered(first_obs):
    plain = pol.MeanVariance(risk_aversion=5.0).decide(first_obs)
    shrunk = pol.MeanVariance(risk_aversion=5.0, mean_shrinkage=1.0).decide(first_obs)
    assert not np.allclose(plain, shrunk)
    assert shrunk[3] == pytest.approx(1.0)  # no expected premium: hold riskless cash


@pytest.mark.parametrize(
    "factory",
    [
        pol.buy_and_hold,
        pol.equal_weight,
        pol.fixed_balanced,
        pol.InverseVolatility,
        pol.MeanVariance,
    ],
)
def test_every_rule_runs_on_the_identical_environment_and_replays_exactly(calm, factory):
    record = replay.rollout(make_env(calm), factory())
    scripted = replay.rollout(make_env(calm), replay.ScriptedPolicy(record.targets()))
    np.testing.assert_array_equal(record.wealth_path, scripted.wealth_path)
    assert record.identity == scripted.identity
    for target in record.targets():
        if target is not None:
            assert np.all(target >= 0) and target.sum() == pytest.approx(1.0, abs=1e-12)


def test_the_episode_record_summarizes_engineering_facts(calm):
    record = replay.rollout(make_env(calm), pol.equal_weight())
    summary = record.summary()
    assert summary["log_growth"] == pytest.approx(record.total_reward, rel=1e-12)
    first_execution = record.ledger[record.decisions[0].decision_index + 1]
    assert summary["terminal_wealth"] == pytest.approx(
        record.wealth_path[-1] / first_execution.wealth_before, rel=1e-12
    )
    assert summary["n_decisions"] == len(record.decisions)
    assert 0.0 <= summary["max_drawdown"] < 1.0
    assert summary["total_cost"] > 0
    assert summary["transition_status"] == "engineering_only_unapproved_transition"


def test_a_scripted_policy_refuses_to_run_past_its_script(calm):
    with pytest.raises(ValueError, match="script"):
        replay.rollout(make_env(calm), replay.ScriptedPolicy([None]))
