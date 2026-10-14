"""
Run configuration for RL training and evaluation, loaded from YAML.

Every setting that can change a result lives in one validated, frozen
`RunConfig`: the fold plan, features, environment schedule, PPO
hyperparameters, seeds, the CEM baseline, the health gate and the training-
health thresholds. Unknown keys are errors (a typo must never silently fall
back to a default), and `sha256()` identifies the configuration in every
artifact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from stock_agent.rl.env import EnvConfig
from stock_agent.rl.learned import CEMConfig
from stock_agent.rl.observation import FeatureSpec
from stock_agent.rl.safety import HealthGate
from stock_agent.rl.splits import SplitPlan, default_split_plan

CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PPOHyperparameters:
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5
    target_kl: float | None = None

    def __post_init__(self) -> None:
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be positive.")
        if self.n_steps < 2 or self.batch_size < 2 or self.n_epochs < 1:
            raise ValueError("n_steps and batch_size must be at least 2, n_epochs at least 1.")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1].")
        if not self.clip_range > 0 or self.max_grad_norm <= 0:
            raise ValueError("clip_range and max_grad_norm must be positive.")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("Loss coefficients must be non-negative.")
        if not self.net_arch or any(width < 1 for width in self.net_arch):
            raise ValueError("net_arch needs positive layer widths.")
        if self.target_kl is not None and not self.target_kl > 0:
            raise ValueError("target_kl must be positive or null.")


@dataclass(frozen=True)
class TrainingSettings:
    seeds: tuple[int, ...]
    total_timesteps: int
    n_envs: int
    episode_sessions: int
    eval_every: int

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Training needs one or more distinct seeds.")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be non-negative.")
        if self.total_timesteps < 1 or self.n_envs < 1 or self.eval_every < 1:
            raise ValueError("total_timesteps, n_envs and eval_every must be positive.")
        if self.episode_sessions < 3:
            raise ValueError("episode_sessions must be at least three sessions.")


@dataclass(frozen=True)
class HealthThresholds:
    max_param_abs: float = 1e4
    kl_warning: float = 0.1
    clip_fraction_warning: float = 0.5
    min_action_std: float = 0.01


@dataclass(frozen=True)
class RunConfig:
    name: str
    plan: SplitPlan
    features: FeatureSpec
    env: EnvConfig
    training: TrainingSettings
    ppo: PPOHyperparameters
    cem: CEMConfig
    gate: HealthGate
    health: HealthThresholds

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "split_plan": self.plan.to_dict(),
            "features": self.features.to_dict(),
            "env": _plain(self.env),
            "training": _plain(self.training),
            "ppo": _plain(self.ppo),
            "cem": _plain(self.cem),
            "gate": _plain(self.gate),
            "health": _plain(self.health),
        }

    def sha256(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(instance) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in dataclasses.asdict(instance).items()
    }


def _build(cls, record: dict | None, section: str):
    record = dict(record or {})
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(record) - set(fields))
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {unknown}")
    for key, value in record.items():
        if isinstance(value, list):
            record[key] = tuple(value)
    return cls(**record)


SECTIONS = {
    "schema_version",
    "name",
    "split_plan",
    "features",
    "env",
    "training",
    "ppo",
    "cem",
    "gate",
    "health",
}


def run_config_from_dict(record: dict) -> RunConfig:
    unknown = sorted(set(record) - SECTIONS)
    if unknown:
        raise ValueError(f"Unknown top-level keys: {unknown}")
    if record.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {record.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION}."
        )
    plan_record = record.get("split_plan", "default")
    plan = default_split_plan() if plan_record == "default" else SplitPlan.from_dict(plan_record)
    features = record.get("features") or {}
    feature_spec = FeatureSpec.from_dict({**FeatureSpec().to_dict(), **features})
    unknown_features = sorted(set(features) - set(FeatureSpec().to_dict()))
    if unknown_features:
        raise ValueError(f"Unknown keys in features: {unknown_features}")
    plan.check_lookback(feature_spec.lookback)
    return RunConfig(
        name=str(record["name"]),
        plan=plan,
        features=feature_spec,
        env=_build(EnvConfig, record.get("env"), "env"),
        training=_build(TrainingSettings, record["training"], "training"),
        ppo=_build(PPOHyperparameters, record.get("ppo"), "ppo"),
        cem=_build(CEMConfig, record.get("cem"), "cem"),
        gate=_build(HealthGate, record.get("gate"), "gate"),
        health=_build(HealthThresholds, record.get("health"), "health"),
    )


def load_run_config(path: str | Path) -> RunConfig:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("A run configuration must be a YAML mapping.")
    return run_config_from_dict(record)
