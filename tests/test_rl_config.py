"""Versioned, validated YAML run configurations for RL training and evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stock_agent.rl import config as cfg
from stock_agent.rl import splits

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "rl"


@pytest.mark.parametrize("name", ["ppo_smoke.yaml", "ppo_dev.yaml"])
def test_the_shipped_configs_load_and_validate(name):
    run = cfg.load_run_config(CONFIG_DIR / name)
    assert run.plan == splits.default_split_plan()
    assert len(run.training.seeds) >= 3
    assert run.training.total_timesteps > 0


def test_the_smoke_config_is_small_enough_for_a_few_cpu_minutes():
    run = cfg.load_run_config(CONFIG_DIR / "ppo_smoke.yaml")
    assert run.training.total_timesteps * len(run.training.seeds) <= 100_000


def _write(tmp_path, record):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    return path


def _smoke_record():
    return yaml.safe_load((CONFIG_DIR / "ppo_smoke.yaml").read_text(encoding="utf-8"))


def test_unknown_keys_are_rejected(tmp_path):
    record = _smoke_record()
    record["ppo"]["learning_rat"] = 0.1
    with pytest.raises(ValueError, match="learning_rat"):
        cfg.load_run_config(_write(tmp_path, record))


def test_an_unsupported_schema_version_is_rejected(tmp_path):
    record = _smoke_record()
    record["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        cfg.load_run_config(_write(tmp_path, record))


def test_an_explicit_split_plan_is_validated(tmp_path):
    record = _smoke_record()
    plan = splits.default_split_plan().to_dict()
    plan["dev"]["seeds"] = plan["train"]["seeds"][:1]
    record["split_plan"] = plan
    with pytest.raises(splits.LeakageError):
        cfg.load_run_config(_write(tmp_path, record))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("ppo", "gamma", 1.5),
        ("ppo", "clip_range", 0.0),
        ("ppo", "batch_size", 0),
        ("training", "seeds", []),
        ("training", "seeds", [1, 1]),
        ("training", "episode_sessions", 2),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, key, value):
    record = _smoke_record()
    record[section][key] = value
    with pytest.raises(ValueError):
        cfg.load_run_config(_write(tmp_path, record))


def test_the_config_hash_is_stable_and_sensitive(tmp_path):
    first = cfg.load_run_config(CONFIG_DIR / "ppo_smoke.yaml")
    second = cfg.load_run_config(CONFIG_DIR / "ppo_smoke.yaml")
    assert first.sha256() == second.sha256()
    record = _smoke_record()
    record["ppo"]["learning_rate"] = 1e-3
    assert cfg.load_run_config(_write(tmp_path, record)).sha256() != first.sha256()


def test_the_config_round_trips_through_its_record(tmp_path):
    run = cfg.load_run_config(CONFIG_DIR / "ppo_smoke.yaml")
    assert cfg.run_config_from_dict(run.to_dict()) == run


def test_a_config_whose_embargo_misses_the_lookback_is_rejected(tmp_path):
    record = _smoke_record()
    record["features"] = {"return_windows": [5, 20, 120], "vol_windows": [20, 60]}
    with pytest.raises(splits.LeakageError, match="lookback"):
        cfg.load_run_config(_write(tmp_path, record))
