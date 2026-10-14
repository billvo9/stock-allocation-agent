"""
PPO training on the synthetic allocation environment (stable-baselines3).

One call to `train_ppo_seed` trains one seed:

1. Seeds Python, NumPy and torch, and pins torch to one CPU thread, so a
   (config, seed, code version) triple replays to identical parameters.
2. Trains SB3's PPO with a diagonal-Gaussian policy over logits on
   `n_envs` copies of `AllocationGymEnv` that sample episodes from the
   TRAIN fold only.
3. Evaluates the deterministic policy (the Gaussian mean) on every DEV-fold
   episode before training, every `eval_every` timesteps and at the end, and
   keeps the checkpoint with the best mean dev log growth. The test fold is
   never built here.
4. Records per-update diagnostics (approximate KL, clip fraction, entropy,
   value loss, explained variance, action standard deviation, parameter
   magnitudes) and per-rollout environment rewards for `rl.health`.
5. Writes `checkpoint.json` beside the model: frozen normalization
   statistics, feature spec, configuration and its hash, code version,
   library versions, scenario generator version, transition and reward
   identity, and the model file's SHA-256.

Checkpoints are loaded only through `load_ppo_checkpoint`, which verifies
the recorded hash. SB3 model files contain pickled objects: load only files
this pipeline wrote. The hash proves integrity, not trusted provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from stock_agent.rl.actions import fund_logits_to_weights
from stock_agent.rl.config import RunConfig
from stock_agent.rl.env import fold_envs
from stock_agent.rl.gym_env import AllocationGymEnv, EpisodeSampler, network_input
from stock_agent.rl.health import seed_health
from stock_agent.rl.observation import (
    FeatureNormalizer,
    FeatureSpec,
    Observation,
    ObservationBuilder,
    ScenarioLoader,
    require_fitted_on,
)
from stock_agent.rl.replay import rollout
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.scenarios import GENERATOR_VERSION, generate_scenario
from stock_agent.rl.transition import PortfolioStep, TransitionInfo

CHECKPOINT_SCHEMA = "rl_ppo_checkpoint"
CHECKPOINT_SCHEMA_VERSION = "1.0.0"
MODEL_FILE = "best_dev.zip"
METADATA_FILE = "checkpoint.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable_baselines3": package_version("stable_baselines3"),
        "gymnasium": package_version("gymnasium"),
    }


class PPOPolicy:
    """A trained PPO model as an allocation policy: the Gaussian mean, mapped to weights."""

    def __init__(self, model: BaseAlgorithm, cash_index: int, name: str = "ppo") -> None:
        self.model = model
        self.cash_index = cash_index
        self.name = name

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> np.ndarray:
        vector, _ = network_input(observation)
        logits, _ = self.model.predict(vector, deterministic=True)
        return fund_logits_to_weights(np.asarray(logits), self.cash_index)


class RecordingPPO(PPO):
    """SB3 PPO that keeps each update's diagnostics for the health report."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.update_records: list[dict[str, float]] = []

    def train(self) -> None:
        super().train()
        values = self.logger.name_to_value
        params = torch.cat([p.detach().flatten() for p in self.policy.parameters()])
        finite = torch.isfinite(params)
        self.update_records.append(
            {
                "approx_kl": float(values["train/approx_kl"]),
                "clip_fraction": float(values["train/clip_fraction"]),
                "entropy": -float(values["train/entropy_loss"]),
                "value_loss": float(values["train/value_loss"]),
                "explained_variance": float(values["train/explained_variance"]),
                "action_std": float(values.get("train/std", math.nan)),
                "param_max_abs": float(params[finite].abs().max()) if finite.any() else math.nan,
                "param_non_finite": int((~finite).sum()),
            }
        )


class TrainingMonitor(BaseCallback):
    """Environment rewards per rollout, and dev-fold checkpoint selection."""

    def __init__(
        self, evaluate: Callable[[BaseAlgorithm], float], eval_every: int, save_path: Path
    ):
        super().__init__()
        self.evaluate = evaluate
        self.eval_every = eval_every
        self.save_path = Path(save_path)
        self.rollouts: list[dict] = []
        self.evaluations: list[dict] = []
        self._best = -math.inf
        self._next_eval = eval_every
        self._current: dict | None = None

    def _evaluate(self) -> None:
        score = float(self.evaluate(self.model))
        finite = math.isfinite(score)
        self.evaluations.append(
            {"timesteps": int(self.num_timesteps), "score": score if finite else None}
        )
        if (finite and score > self._best) or not self.save_path.exists():
            self._best = score if finite else self._best
            self.model.save(self.save_path)

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_rollout_start(self) -> None:
        self._current = {
            "rewards": [],
            "non_finite_rewards": 0,
            "imputed_features": 0,
            "features": 0,
        }

    def _on_step(self) -> bool:
        rewards = np.array(self.locals["rewards"], dtype=np.float64)
        finite = np.isfinite(rewards)
        self._current["rewards"].extend(float(r) for r in rewards[finite])
        self._current["non_finite_rewards"] += int((~finite).sum())
        width = int(np.prod(self.model.observation_space.shape))
        for info in self.locals["infos"]:
            self._current["imputed_features"] += int(info.get("imputed_features", 0))
            self._current["features"] += width
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            self._next_eval += self.eval_every
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts.append(self._current)

    def _on_training_end(self) -> None:
        if not self.evaluations or self.evaluations[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


def mean_log_growth(envs, policy) -> float:
    return float(np.mean([rollout(env, policy).total_reward for env in envs]))


@dataclass(frozen=True)
class SeedRun:
    seed: int
    directory: Path
    health: dict
    metadata: dict


def _ppo_kwargs(config: RunConfig) -> dict[str, object]:
    ppo = config.ppo
    arch = list(ppo.net_arch)
    return {
        "learning_rate": ppo.learning_rate,
        "n_steps": ppo.n_steps,
        "batch_size": ppo.batch_size,
        "n_epochs": ppo.n_epochs,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.gae_lambda,
        "clip_range": ppo.clip_range,
        "ent_coef": ppo.ent_coef,
        "vf_coef": ppo.vf_coef,
        "max_grad_norm": ppo.max_grad_norm,
        "target_kl": ppo.target_kl,
        "policy_kwargs": {
            "net_arch": {"pi": arch, "vf": arch},
            "log_std_init": ppo.log_std_init,
            "activation_fn": torch.nn.Tanh,
        },
    }


def train_ppo_seed(
    config: RunConfig,
    seed: int,
    *,
    builder: ObservationBuilder,
    transition: PortfolioStep,
    transition_info: TransitionInfo,
    out_dir: Path,
    run_id: str,
    code: dict,
    created_at: str,
    loader: ScenarioLoader = generate_scenario,
) -> SeedRun:
    require_fitted_on(builder.normalizer, config.plan.train)
    config.plan.check_lookback(builder.spec.lookback)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    set_random_seed(seed)
    plan = config.plan
    sampler = EpisodeSampler(plan.train, config.training.episode_sessions, loader)

    def make_env() -> AllocationGymEnv:
        return AllocationGymEnv(
            sampler,
            builder=builder,
            transition=transition,
            transition_info=transition_info,
            env_config=config.env,
        )

    vec_env = DummyVecEnv([make_env for _ in range(config.training.n_envs)])
    model = RecordingPPO(
        "MlpPolicy", vec_env, seed=seed, device="cpu", verbose=0, **_ppo_kwargs(config)
    )
    dev_envs = fold_envs(
        plan.dev,
        transition=transition,
        transition_info=transition_info,
        builder=builder,
        config=config.env,
        loader=loader,
    )
    monitor = TrainingMonitor(
        lambda m: mean_log_growth(dev_envs, PPOPolicy(m, builder.cash_index)),
        config.training.eval_every,
        out_dir / MODEL_FILE,
    )
    model.learn(total_timesteps=config.training.total_timesteps, callback=monitor)
    health = seed_health(
        seed,
        updates=model.update_records,
        rollouts=monitor.rollouts,
        evaluations=monitor.evaluations,
        thresholds=config.health,
    )
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "seed": int(seed),
        "created_at": created_at,
        "label": "approved_transition" if transition_info.approved else transition_info.status,
        "transition": transition_info.to_dict(),
        "reward": REWARD_INFO.to_dict(),
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "code": dict(code),
        "libraries": library_versions(),
        "scenario_generator_version": GENERATOR_VERSION,
        "feature_spec": builder.spec.to_dict(),
        "normalizer": builder.normalizer.to_dict(),
        "asset_names": list(builder.asset_names),
        "cash_index": builder.cash_index,
        "model": {"file": MODEL_FILE, "sha256": file_sha256(out_dir / MODEL_FILE)},
        "selection": health["dev_selection"],
        "timesteps": int(model.num_timesteps),
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    vec_env.close()
    return SeedRun(seed=int(seed), directory=out_dir, health=health, metadata=metadata)


@dataclass(frozen=True)
class LoadedCheckpoint:
    policy: PPOPolicy
    metadata: dict
    builder: ObservationBuilder


def load_ppo_checkpoint(directory: Path) -> LoadedCheckpoint:
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    if (metadata.get("schema"), metadata.get("schema_version")) != (
        CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA_VERSION,
    ):
        raise ValueError("Unsupported checkpoint schema.")
    model_path = directory / metadata["model"]["file"]
    if file_sha256(model_path) != metadata["model"]["sha256"]:
        raise ValueError("The checkpoint's model file does not match its recorded hash.")
    builder = ObservationBuilder(
        FeatureSpec.from_dict(metadata["feature_spec"]),
        FeatureNormalizer.from_dict(metadata["normalizer"]),
        tuple(metadata["asset_names"]),
        int(metadata["cash_index"]),
    )
    model = PPO.load(model_path, device="cpu")
    policy = PPOPolicy(model, builder.cash_index, name=f"ppo_seed_{metadata['seed']}")
    return LoadedCheckpoint(policy=policy, metadata=metadata, builder=builder)
