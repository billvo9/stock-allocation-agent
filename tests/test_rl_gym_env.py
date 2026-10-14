"""
The gymnasium wrapper: logits in, the same core environment underneath.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stock_agent.rl import (
    actions,
    gym_env,
    splits,
)
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import scenarios as sc

SPEC = ob.FeatureSpec()
TRAIN = splits.FoldSpec("train", ("mixed", "stale_observations"), (0, 1), 60, 600)


@pytest.fixture(scope="module")
def builder():
    norm = ob.fit_normalizer(TRAIN, SPEC)
    return ob.ObservationBuilder(SPEC, norm, sc.ASSET_NAMES, sc.CASH_INDEX)


def make(builder, episode_sessions=120, **config):
    return gym_env.AllocationGymEnv(
        gym_env.EpisodeSampler(TRAIN, episode_sessions),
        builder=builder,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        env_config=rl_env.EnvConfig(**config),
    )


def test_the_env_passes_the_gymnasium_checker(builder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(make(builder), skip_render_check=True)


def test_spaces_are_bounded_fund_logits_and_bounded_observations(builder):
    env = make(builder)
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == -actions.ACTION_BOUND)
    assert np.all(env.action_space.high == actions.ACTION_BOUND)
    assert env.observation_space.shape == (builder.dimension,)
    assert env.observation_space.dtype == np.float32
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)


def test_reset_with_a_seed_replays_the_same_episode(builder):
    first, second = make(builder), make(builder)
    obs_a, info_a = first.reset(seed=11)
    obs_b, info_b = second.reset(seed=11)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a["episode"] == info_b["episode"]
    rng = np.random.RandomState(0)
    for _ in range(10):
        logits = rng.normal(size=3).astype(np.float32)
        step_a = first.step(logits)
        step_b = second.step(logits)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]


def test_episodes_are_sampled_inside_the_train_fold_window(builder):
    env = make(builder, episode_sessions=100)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        scenario_id, scenario_seed, start, stop = info["episode"]
        assert (scenario_id, scenario_seed) in TRAIN.episodes()
        assert TRAIN.start <= start and stop <= TRAIN.stop and stop - start == 100


def test_a_step_applies_the_fund_logit_map_through_the_core_env(builder):
    env = make(builder)
    env.reset(seed=5)
    logits = np.array([0.5, -0.3, 12.0], dtype=np.float32)
    env.step(logits)
    executed = env.core.ledger[env.core.first_decision_index + 1]
    expected = actions.logits_to_weights(np.array([0.5, -0.3, 10.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(executed.target, expected, rtol=1e-6)


def test_the_reward_is_the_core_env_reward(builder):
    env = make(builder)
    env.reset(seed=5)
    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    executed = env.core.ledger[env.core.first_decision_index + 1 :]
    assert reward == pytest.approx(sum(row.reward for row in executed), abs=1e-12)


def test_non_finite_features_are_zero_imputed_for_the_network_and_counted(builder):
    data = sc.generate_scenario("calm", 0)
    early = builder.build(ob.snapshot_at(data, 10), np.eye(4)[3], 0.0)
    vector, imputed = gym_env.network_input(early)
    assert imputed > 0
    assert np.all(np.isfinite(vector)) and vector.dtype == np.float32


def test_a_fixed_episode_can_be_requested_for_evaluation(builder):
    env = make(builder)
    _, info = env.reset(options={"episode": ("calm", 1, 100, 300)})
    assert info["episode"] == ("calm", 1, 100, 300)


def test_the_sampler_refuses_windows_longer_than_the_fold():
    with pytest.raises(ValueError, match="episode_sessions"):
        gym_env.EpisodeSampler(TRAIN, TRAIN.stop - TRAIN.start + 1)
