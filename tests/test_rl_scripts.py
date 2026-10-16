"""
The train and evaluate scripts end to end on a tiny configuration.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {
    "schema_version": 1,
    "name": "tiny_script",
    "split_plan": {
        "train": {
            "role": "train",
            "scenario_ids": ["mixed"],
            "seeds": [0],
            "start": 60,
            "stop": 240,
        },
        "dev": {"role": "dev", "scenario_ids": ["calm"], "seeds": [10], "start": 300, "stop": 400},
        "test": {
            "role": "test",
            "scenario_ids": ["calm", "stale_observations"],
            "seeds": [20],
            "start": 460,
            "stop": 600,
        },
        "embargo": 60,
    },
    "env": {"rebalance_every": 5},
    "training": {
        "seeds": [0, 1],
        "total_timesteps": 128,
        "n_envs": 2,
        "episode_sessions": 60,
        "eval_every": 128,
    },
    "ppo": {"n_steps": 32, "batch_size": 32, "n_epochs": 1, "net_arch": [8]},
    "cem": {"iterations": 1, "population": 2},
}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("rl_runs")
    config = root / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    train = _load("train_ppo")
    code = train.main(
        [
            "--config",
            str(config),
            "--output-root",
            str(root / "out"),
            "--health-inputs-root",
            str(root / "health_inputs"),
        ],
        clock=lambda: datetime(2026, 10, 14, 12, 0, 0, 123456, tzinfo=UTC),
        code_version=lambda: {"git_sha": SHA, "git_dirty": False},
    )
    assert code == 0
    (directory,) = (root / "out").iterdir()
    return directory


SHA = "0123456789abcdef0123456789abcdef01234567"


def test_training_writes_the_agreed_health_input(run_dir):
    path = run_dir.parent.parent / "health_inputs" / "rl_training_health" / f"{run_dir.name}.json"
    record = json.loads(path.read_text())
    assert record["artifact"] == "rl_training_health" and record["schema_version"] == "1.0"
    assert record["generated_at"] == "2026-10-14T12:00:00.123456Z"
    assert record["code_sha"] == SHA and record["synthetic"] is True
    assert record["git_dirty"] is False
    assert record["status"] == "completed" and record["seeds"] == 2
    assert record["transition_status"] == "approved"
    assert record["scenario_version"] == "1.1.0"
    assert len(record["checks"]) == 8


def test_the_cem_baseline_records_its_provenance(run_dir):
    cem = json.loads((run_dir / "cem_baseline.json").read_text())
    assert cem["run_id"] == run_dir.name
    assert cem["normalizer"]["fitted_on"]["role"] == "train"
    assert cem["split_plan"]["train"]["seeds"] == [0]
    assert {e["role"] for e in cem["scenarios"]} == {"train", "dev"}


def test_training_writes_every_artifact_with_its_label(run_dir):
    assert run_dir.name.startswith("20261014T120000Z-")
    for name in ("run.json", "training_health.json", "cem_baseline.json"):
        record = json.loads((run_dir / name).read_text())
        assert record["label"] == "approved_transition"
    for seed in (0, 1):
        assert (run_dir / f"seed_{seed}" / "best_dev.zip").exists()
        assert (run_dir / f"seed_{seed}" / "checkpoint.json").exists()
    health = json.loads((run_dir / "training_health.json").read_text())
    assert [s["seed"] for s in health["seeds"]] == [0, 1]
    assert health["seed_spread"]["n"] == 2


def test_the_run_record_carries_reproduction_facts(run_dir):
    run = json.loads((run_dir / "run.json").read_text())
    assert run["code"] == {"git_sha": SHA, "git_dirty": False}
    assert run["config"]["name"] == "tiny_script"
    assert run["normalizer_fitted_on"]["role"] == "train"
    assert set(run["libraries"]) >= {"torch", "stable_baselines3", "gymnasium"}


def _evaluate(run_dir, *extra, sha=SHA):
    evaluate = _load("evaluate_rl")
    return evaluate.main(
        ["--run-dir", str(run_dir), *extra],
        clock=lambda: datetime(2026, 10, 14, 13, 0, tzinfo=UTC),
        code_version=lambda: {"git_sha": sha, "git_dirty": False},
    )


@pytest.fixture(scope="module")
def evaluated(run_dir):
    assert _evaluate(run_dir) == 0
    return run_dir


def test_evaluation_writes_an_engineering_report_and_withholds_tables(evaluated):
    # The transition is approved, but tables stay withheld until the registered comparison.
    report = json.loads((evaluated / "evaluation_test_1.json").read_text())
    assert report["fold_role"] == "test" and report["fairness"] == "passed"
    assert report["fold_evaluation_number"] == 1
    assert report["comparison_tables"].startswith("withheld")
    assert not list(evaluated.glob("comparison_*.json"))
    assert report["provenance"]["code_sha_matches"] is True
    methods = report["methods"]
    for name in (
        "buy_and_hold",
        "mean_variance",
        "linear_softmax_cem",
        "ppo_seed_0",
        "ppo_seed_1",
        "guarded:ppo_seed_0",
    ):
        assert methods[name]["all_finite"] and methods[name]["episodes"] == 2
    assert "ppo_with_forecasts" in report["unavailable"]
    assert methods["guarded:ppo_seed_0"]["fallback_reasons"].get("stale_inputs", 0) > 0


def test_every_test_fold_evaluation_is_counted_and_never_overwritten(evaluated):
    first = (evaluated / "evaluation_test_1.json").read_bytes()
    assert _evaluate(evaluated) == 0
    assert (evaluated / "evaluation_test_1.json").read_bytes() == first
    second = json.loads((evaluated / "evaluation_test_2.json").read_text())
    assert second["fold_evaluation_number"] == 2
    ledger = [
        json.loads(line) for line in (evaluated / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [(e["fold"], e["number"]) for e in ledger] == [("test", 1), ("test", 2)]
    assert ledger[0]["file"] == "evaluation_test_1.json"


def test_evaluating_with_other_code_is_recorded(evaluated):
    with pytest.warns(RuntimeWarning, match="code"):
        assert _evaluate(evaluated, "--fold", "dev", sha="e" * 40) == 0
    report = json.loads((evaluated / "evaluation_dev_1.json").read_text())
    assert report["provenance"]["code_sha_matches"] is False


def test_a_second_run_never_overwrites_the_first(run_dir, tmp_path):
    train = _load("train_ppo")
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    with pytest.raises(FileExistsError):
        train.main(
            [
                "--config",
                str(config),
                "--output-root",
                str(run_dir.parent),
                "--health-inputs-root",
                str(tmp_path / "health_inputs"),
            ],
            clock=lambda: datetime(2026, 10, 14, 12, 0, 0, 123456, tzinfo=UTC),
            code_version=lambda: {"git_sha": SHA, "git_dirty": False},
        )
