"""
PPO machinery (stable-baselines3): engineering tests on synthetic data.

A completed training run here is an engineering result, never evidence that
the policy is a good investment.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from stock_agent.rl import actions, gym_env, ppo, replay, safety, splits
from stock_agent.rl import config as cfg
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import scenarios as sc

TINY_PLAN = {
    "train": {
        "role": "train",
        "scenario_ids": ["mixed"],
        "seeds": [0, 1],
        "start": 60,
        "stop": 300,
    },
    "dev": {
        "role": "dev",
        "scenario_ids": ["calm", "crash"],
        "seeds": [10],
        "start": 360,
        "stop": 480,
    },
    "test": {"role": "test", "scenario_ids": ["calm"], "seeds": [20], "start": 540, "stop": 660},
    "embargo": 60,
}


def tiny_config(**training):
    return cfg.run_config_from_dict(
        {
            "schema_version": 1,
            "name": "tiny",
            "split_plan": TINY_PLAN,
            "env": {"rebalance_every": 5},
            "training": {
                "seeds": [0],
                "total_timesteps": 256,
                "n_envs": 2,
                "episode_sessions": 60,
                "eval_every": 128,
                **training,
            },
            "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 2, "net_arch": [16, 16]},
            "cem": {"iterations": 1, "population": 2},
        }
    )


def builder_for(config, loader=sc.generate_scenario):
    norm = ob.fit_normalizer(config.plan.train, config.features, loader=loader)
    return ob.ObservationBuilder(config.features, norm, sc.ASSET_NAMES, sc.CASH_INDEX)


def train(tmp_path, config, seed=0, name="run", loader=sc.generate_scenario, builder=None):
    return ppo.train_ppo_seed(
        config,
        seed,
        builder=builder or builder_for(config, loader),
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        out_dir=tmp_path / name,
        run_id="test-run",
        code={"git_sha": "abc", "git_dirty": False},
        created_at="2026-10-14T00:00:00Z",
        loader=loader,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    config = tiny_config()
    return config, train(tmp_path_factory.mktemp("ppo"), config)


def _state(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def test_clipped_samples_keep_their_raw_logits_and_likelihoods():
    """
    With a wide policy (log std 3, a standard deviation near 20) many sampled
    fund logits fall outside the +/-10 action box. SB3 must store the raw
    sample and its log probability while the environment receives the
    clipped logits; re-scoring the stored raw actions must reproduce the
    stored log probabilities. That is what keeps the PPO ratio valid under
    the clip.
    """

    config = tiny_config()
    builder = builder_for(config)
    env = gym_env.AllocationGymEnv(
        gym_env.EpisodeSampler(config.plan.train, 60),
        builder=builder,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        env_config=config.env,
    )
    received = []
    step = env.step

    def recording_step(action):
        received.append(np.array(action, dtype=np.float64))
        return step(action)

    env.step = recording_step
    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=0,
        device="cpu",
        n_steps=64,
        batch_size=32,
        policy_kwargs={"log_std_init": 3.0},
    )
    _, callback = model._setup_learn(64, callback=None)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=64)
    buffer = model.rollout_buffer
    stored = buffer.actions.reshape(-1, 3)
    beyond = np.abs(stored) > actions.ACTION_BOUND
    assert beyond.sum() >= 10
    np.testing.assert_allclose(
        np.vstack(received),
        np.clip(stored, -actions.ACTION_BOUND, actions.ACTION_BOUND),
        rtol=1e-6,
    )
    observations = torch.as_tensor(buffer.observations.reshape(64, -1))
    with torch.no_grad():
        _, rescored, _ = model.policy.evaluate_actions(observations, torch.as_tensor(stored))
        mean = model.policy.get_distribution(observations).distribution.mean.numpy()
    np.testing.assert_allclose(rescored.numpy(), buffer.log_probs.reshape(-1), atol=1e-5)
    row = int(np.flatnonzero(beyond.any(axis=1))[0])
    ours = actions.gaussian_log_prob(stored[row], mean[row], model.policy.log_std.detach().numpy())
    assert ours == pytest.approx(float(buffer.log_probs.reshape(-1)[row]), abs=1e-4)


def test_training_records_health_and_a_dev_selected_checkpoint(trained):
    _, run = trained
    assert run.health["status"] in ("healthy", "warning")
    assert run.health["updates"] == 256 // (64 * 2)
    selection = run.metadata["selection"]
    assert [e["timesteps"] for e in selection["evaluations"]] == [0, 128, 256]
    assert selection["checkpoints_compared"] == 3
    assert (run.directory / ppo.MODEL_FILE).exists()
    json.dumps(run.health, allow_nan=False)


def test_the_checkpoint_records_what_replay_needs(trained):
    config, run = trained
    meta = json.loads((run.directory / ppo.METADATA_FILE).read_text())
    assert meta["label"] == "engineering_only_unapproved_transition"
    assert meta["config_sha256"] == config.sha256()
    assert meta["code"] == {"git_sha": "abc", "git_dirty": False}
    assert meta["scenario_generator_version"] == sc.GENERATOR_VERSION
    assert meta["normalizer"]["fitted_on"]["role"] == "train"
    assert set(meta["libraries"]) == {"python", "numpy", "torch", "stable_baselines3", "gymnasium"}
    assert meta["reward"]["status"] == "proposed_d8a_pending_owner_approval"


def test_a_checkpoint_loads_with_its_frozen_normalizer_and_acts_identically(trained):
    config, run = trained
    loaded = ppo.load_ppo_checkpoint(run.directory)
    assert loaded.builder.normalizer == builder_for(config).normalizer
    again = ppo.load_ppo_checkpoint(run.directory)
    data = sc.generate_scenario("calm", 10)
    obs = loaded.builder.build(ob.snapshot_at(data, 400), np.eye(4)[3], 0.0)
    np.testing.assert_array_equal(loaded.policy.decide(obs), again.policy.decide(obs))


def _copy_checkpoint(source, target, **metadata_changes):
    target.mkdir()
    (target / ppo.MODEL_FILE).write_bytes((source / ppo.MODEL_FILE).read_bytes())
    meta = json.loads((source / ppo.METADATA_FILE).read_text())
    meta.update(metadata_changes)
    (target / ppo.METADATA_FILE).write_text(json.dumps(meta))
    return target


def test_a_checkpoint_from_another_generator_version_is_refused(tmp_path, trained):
    _, run = trained
    stale = _copy_checkpoint(run.directory, tmp_path / "stale", scenario_generator_version="1.0.0")
    with pytest.raises(ValueError, match="generator"):
        ppo.load_ppo_checkpoint(stale)


def test_a_checkpoint_from_another_config_is_refused(trained):
    _, run = trained
    with pytest.raises(ValueError, match="config"):
        ppo.load_ppo_checkpoint(run.directory, expected_config_sha256="0" * 64)


def test_a_checkpoint_from_other_code_warns_loudly_and_records_it(trained):
    config, run = trained
    with pytest.warns(RuntimeWarning, match="code"):
        loaded = ppo.load_ppo_checkpoint(
            run.directory, expected_config_sha256=config.sha256(), expected_code_sha="f" * 40
        )
    assert loaded.provenance == {
        "trained_code_sha": "abc",
        "current_code_sha": "f" * 40,
        "code_sha_matches": False,
        "config_sha256_matches": True,
        "scenario_generator_version": sc.GENERATOR_VERSION,
    }


def test_a_tampered_checkpoint_is_refused(tmp_path, trained):
    _, run = trained
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in (ppo.METADATA_FILE, ppo.MODEL_FILE):
        (copy / name).write_bytes((run.directory / name).read_bytes())
    model = bytearray((copy / ppo.MODEL_FILE).read_bytes())
    model[-10] ^= 0xFF
    (copy / ppo.MODEL_FILE).write_bytes(bytes(model))
    with pytest.raises(ValueError, match="hash"):
        ppo.load_ppo_checkpoint(copy)


def test_the_same_seed_replays_to_identical_parameters_and_another_seed_does_not(tmp_path):
    config = tiny_config()
    builder = builder_for(config)
    first = train(tmp_path, config, seed=3, name="a", builder=builder)
    second = train(tmp_path, config, seed=3, name="b", builder=builder)
    other = train(tmp_path, config, seed=4, name="c", builder=builder)
    load = lambda run: PPO.load(run.directory / ppo.MODEL_FILE, device="cpu")
    a, b, c = _state(load(first)), _state(load(second)), _state(load(other))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)
    assert first.health["dev_selection"] == second.health["dev_selection"]


def test_training_samples_train_episodes_and_selects_on_dev_only(tmp_path, monkeypatch):
    roles = []
    original_sampler = ppo.EpisodeSampler
    original_envs = ppo.fold_envs

    def sampler(fold, *args, **kwargs):
        roles.append(("sampler", fold.role))
        return original_sampler(fold, *args, **kwargs)

    def envs(fold, **kwargs):
        roles.append(("selection", fold.role))
        return original_envs(fold, **kwargs)

    monkeypatch.setattr(ppo, "EpisodeSampler", sampler)
    monkeypatch.setattr(ppo, "fold_envs", envs)
    train(tmp_path, tiny_config(total_timesteps=128))
    assert sorted(roles) == [("sampler", "train"), ("selection", "dev")]


def _toy(scenario_id, seed):
    """Equity earns +0.4% every session; everything else 0. Engineering toy only."""

    sessions = 400
    returns = np.zeros((sessions, 4))
    returns[:, 0] = 0.004
    return sc.ScenarioData(
        scenario_id=scenario_id,
        generator_version=sc.GENERATOR_VERSION,
        seed=seed,
        returns=returns,
        has_close=np.ones((sessions, 4), dtype=bool),
        obs_lag=np.zeros((sessions, 4), dtype=np.int64),
        cost_multiplier=np.ones(sessions),
        regime=np.array(["toy"] * sessions, dtype=object),
        latent_returns=returns.copy(),
    )


def test_the_ppo_machinery_learns_an_obvious_toy_allocation(tmp_path):
    config = dataclasses.replace(
        tiny_config(total_timesteps=16384, n_envs=4, eval_every=16384),
        ppo=cfg.PPOHyperparameters(
            n_steps=128, batch_size=128, n_epochs=5, learning_rate=3e-3, net_arch=(16,)
        ),
    )
    run = train(tmp_path, config, loader=_toy)
    model = PPO.load(run.directory / "best_dev.zip", device="cpu")
    builder = builder_for(config, loader=_toy)
    obs = builder.build(ob.snapshot_at(_toy("calm", 0), 200), np.full(4, 0.25), 0.0)
    weights = ppo.PPOPolicy(model, sc.CASH_INDEX).decide(obs)
    assert weights[0] > 0.6
    assert run.metadata["selection"]["selected_timesteps"] > 0


@pytest.mark.parametrize(
    "scenario",
    ["crash", "high_correlation", "extreme_volatility", "stale_observations", "high_costs"],
)
def test_a_trained_policy_survives_every_stress_scenario(trained, scenario):
    config, run = trained
    loaded = ppo.load_ppo_checkpoint(run.directory)
    data = sc.generate_scenario(scenario, 101)
    env = rl_env.AllocationEnv(
        data,
        start=960,
        stop=1200,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        builder=loaded.builder,
        config=config.env,
    )
    raw = replay.rollout(env, loaded.policy)
    assert np.all(np.isfinite(raw.wealth_path)) and np.all(raw.wealth_path > 0)
    for decision in raw.decisions:
        assert decision.target is not None and decision.target.sum() == pytest.approx(1.0)
    guarded = replay.rollout(env, safety.GuardedPolicy(loaded.policy, gate=config.gate))
    if scenario == "stale_observations":
        assert any("stale_inputs" in d.fallback_reasons for d in guarded.decisions)


def test_an_unhealthy_model_never_acts_through_the_guard(trained):
    config, run = trained
    loaded = ppo.load_ppo_checkpoint(run.directory)
    guarded = safety.GuardedPolicy(loaded.policy, gate=config.gate, model_healthy=False)
    data = sc.generate_scenario("calm", 10)
    obs = loaded.builder.build(ob.snapshot_at(data, 400), np.eye(4)[3], 0.0)
    guarded.decide(obs)
    assert guarded.last_reasons == ("model_unhealthy",)


def test_ppo_training_refuses_a_normalizer_not_fitted_on_the_train_fold(tmp_path):
    config = tiny_config()
    names = config.features.market_names(sc.ASSET_NAMES, sc.CASH_INDEX)
    identity = ob.ObservationBuilder(
        config.features, ob.FeatureNormalizer.identity(names), sc.ASSET_NAMES, sc.CASH_INDEX
    )
    with pytest.raises(splits.LeakageError, match="identity"):
        train(tmp_path, config, builder=identity)


def test_the_training_fold_must_be_long_enough_for_an_episode():
    with pytest.raises(ValueError, match="episode_sessions"):
        gym_env.EpisodeSampler(splits.FoldSpec("train", ("calm",), (0,), 60, 100), 60 + 41)
