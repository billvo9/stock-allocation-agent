"""
Test helpers for the registered comparison (not tests).

- `tiny_record` / `tiny_config`: the registered configuration with a tiny fold
  plan and training budget, so a full train-and-evaluate run takes seconds.
- `FixedClock`: a deterministic clock (one microsecond per call).
- `FakeKind`: a light learned kind behind the runner's `LearnedKind` seam.
  Each seed is a fixed mix chosen by the seed, so seeds differ on purpose;
  "training" reads the dev fold, as a real selection does, and writes a
  checkpoint file whose hash the selection freezes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from stock_agent.comparison.protocol import comparison_config_from_dict
from stock_agent.rl.policies import ConstantMix

REPO = Path(__file__).resolve().parents[1]
REGISTERED = REPO / "configs" / "comparison" / "ppo_comparison_v1.yaml"
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
        "stop": 420,
    },
    "test": {
        "role": "test",
        "scenario_ids": ["calm", "missing_closes"],
        "seeds": [20, 21],
        "start": 540,
        "stop": 600,
    },
    "embargo": 60,
}
CODE = {"git_sha": "ab" * 20, "git_dirty": False}


def registered_record() -> dict:
    return yaml.safe_load(REGISTERED.read_text(encoding="utf-8"))


def tiny_record() -> dict:
    record = copy.deepcopy(registered_record())
    record["rl"]["split_plan"] = copy.deepcopy(TINY_PLAN)
    record["rl"]["training"] = {
        "seeds": [0, 1, 2, 3, 4],
        "total_timesteps": 128,
        "n_envs": 2,
        "episode_sessions": 60,
        "eval_every": 64,
    }
    record["rl"]["ppo"].update({"n_steps": 32, "batch_size": 32, "n_epochs": 1})
    record["rl"]["cem"] = {"iterations": 1, "population": 2}
    return record


def tiny_config(record: dict | None = None):
    return comparison_config_from_dict(record or tiny_record())


def small_config(*strategies: str):
    """The tiny configuration restricted to some strategies (faster; same protocol)."""

    record = tiny_record()
    record["strategies"] = list(strategies)
    if "linear_softmax_cem" not in strategies:
        record["cem_seeds"] = []
    return comparison_config_from_dict(record)


class FixedClock:
    def __init__(self, start: datetime = datetime(2026, 10, 17, 12, tzinfo=UTC)) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(microseconds=1)
        return self.now


def fake_weights(seed: int) -> list[float]:
    k = seed % 5
    return [0.2 + 0.1 * k, 0.3, 0.5 - 0.1 * k, 0.0]


class FakeKind:
    def __init__(self, name: str) -> None:
        self.name = name

    def train(self, run_config, seed, *, builder, run_dir, subpath, loader, context) -> dict:
        for scenario, dev_seed in run_config.plan.dev.episodes():
            loader(scenario, dev_seed)  # a real selection reads the dev fold
        path = Path(run_dir) / f"{subpath}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "weights": fake_weights(seed),
            "normalizer": builder.normalizer.to_dict(),
            "env": run_config.env.to_dict(),
        }
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        return {
            "checkpoint": {
                "path": f"{subpath}.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "selection": {"metric": "fake dev selection", "checkpoints_compared": 2},
            "variants_evaluated": 3,
            "healthy": True,
        }

    def load(self, member, *, run_dir, builder, run_config):
        path = Path(run_dir) / member["checkpoint"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["checkpoint"]["sha256"]:
            from stock_agent.comparison.fairness import SelectionError

            raise SelectionError("not the frozen checkpoint")
        weights = json.loads(path.read_text(encoding="utf-8"))["weights"]
        return lambda: ConstantMix(f"fake_{self.name}_{member['seed']}", weights)


FAKE_KINDS = {"ppo": FakeKind("ppo"), "cem": FakeKind("cem")}
