"""The versioned training-health JSON consumed by the Model Health Center."""

from __future__ import annotations

import json
import math

import pytest

from stock_agent.rl import health
from stock_agent.rl.config import HealthThresholds
from stock_agent.rl.reference_transition import TRANSITION_INFO
from stock_agent.rl.reward import REWARD_INFO

THRESHOLDS = HealthThresholds()


def _updates(n=3, **override):
    rows = []
    for i in range(n):
        row = {
            "approx_kl": 0.01 + 0.001 * i,
            "clip_fraction": 0.1,
            "entropy": 5.0 - 0.1 * i,
            "value_loss": 1e-4 / (i + 1),
            "explained_variance": 0.1 * i,
            "action_std": 0.6 - 0.01 * i,
            "param_max_abs": 2.0,
            "param_non_finite": 0,
        }
        row.update(override)
        rows.append(row)
    return rows


def _rollouts(n=3, rewards=(0.001, -0.002, 0.0005)):
    return [
        {"rewards": list(rewards), "non_finite_rewards": 0, "imputed_features": 1, "features": 100}
        for _ in range(n)
    ]


def _evaluations():
    return [{"timesteps": 0, "score": -0.01}, {"timesteps": 100, "score": 0.02}]


def _seed(seed=0, updates=None, rollouts=None):
    return health.seed_health(
        seed,
        updates=_updates() if updates is None else updates,
        rollouts=_rollouts() if rollouts is None else rollouts,
        evaluations=_evaluations(),
        thresholds=THRESHOLDS,
    )


def test_a_clean_seed_is_healthy_with_every_required_metric():
    record = _seed()
    assert record["status"] == "healthy" and record["reasons"] == [] and record["warnings"] == []
    for key in (
        "reward",
        "approx_kl",
        "clip_fraction",
        "entropy",
        "value_loss",
        "explained_variance",
        "action_std",
        "parameters",
        "dev_selection",
        "imputed_feature_share",
    ):
        assert key in record
    assert record["reward"]["n"] == 9
    assert record["dev_selection"]["selected_timesteps"] == 100
    assert record["dev_selection"]["checkpoints_compared"] == 2
    assert record["imputed_feature_share"] == pytest.approx(0.01)


def test_a_non_finite_reward_makes_the_seed_unhealthy():
    rollouts = _rollouts()
    rollouts[1]["non_finite_rewards"] = 2
    assert _seed(rollouts=rollouts)["reasons"] == ["non_finite_reward"]


def test_a_non_finite_loss_makes_the_seed_unhealthy():
    record = _seed(updates=_updates(value_loss=math.nan))
    assert record["status"] == "unhealthy" and "non_finite_loss" in record["reasons"]


def test_exploding_or_non_finite_parameters_make_the_seed_unhealthy():
    assert "parameter_explosion" in _seed(updates=_updates(param_max_abs=1e6))["reasons"]
    assert "non_finite_parameters" in _seed(updates=_updates(param_non_finite=3))["reasons"]


def test_kl_spikes_high_clipping_and_collapsed_exploration_are_warnings():
    record = _seed(updates=_updates(approx_kl=0.5, clip_fraction=0.9, action_std=0.001))
    assert record["status"] == "warning"
    assert record["warnings"] == ["approx_kl_spike", "high_clip_fraction", "action_std_collapse"]


def test_no_updates_is_unhealthy():
    assert _seed(updates=[])["reasons"] == ["no_updates"]


def test_the_report_is_versioned_labelled_and_reports_the_seed_spread():
    seeds = [_seed(0), _seed(1), _seed(2, updates=_updates(param_max_abs=1e9))]
    report = health.training_health_report(
        run_id="run-1",
        config_sha256="abc",
        created_at="2026-10-14T00:00:00Z",
        transition=TRANSITION_INFO,
        reward=REWARD_INFO,
        thresholds=THRESHOLDS,
        seeds=seeds,
    )
    assert report["schema"] == "rl_training_health"
    assert report["schema_version"] == health.HEALTH_SCHEMA_VERSION
    assert report["label"] == "engineering_only_unapproved_transition"
    assert report["status"] == "unhealthy"
    spread = report["seed_spread"]
    assert spread["n"] == 3 and spread["values"] == [0.02, 0.02, 0.02]
    assert spread["std"] == 0.0
    json.dumps(report, allow_nan=False)


def test_the_seed_spread_never_reports_only_the_best_seed():
    seeds = []
    for seed, best in enumerate([0.01, 0.03, -0.02]):
        record = _seed(seed)
        record["dev_selection"]["selected_score"] = best
        seeds.append(record)
    spread = health.seed_spread(seeds)
    assert spread["values"] == [0.01, 0.03, -0.02]
    assert spread["min"] == -0.02 and spread["max"] == 0.03 and spread["median"] == 0.01


def test_model_health_reads_the_status_of_one_seed():
    report = {"seeds": [{"seed": 0, "status": "healthy"}, {"seed": 1, "status": "unhealthy"}]}
    assert health.model_is_healthy(report, 0)
    assert not health.model_is_healthy(report, 1)
    assert not health.model_is_healthy(report, 7)


# --- the agreed Model Health Center input (health-input-formats-agreed.md, v1.0) ---

GENERATED_AT = "2026-10-14T12:00:00.123456Z"
CHECK_IDS = (
    "nan_or_inf",
    "exploding_values",
    "approx_kl",
    "clip_fraction",
    "entropy_collapse",
    "value_loss",
    "reward_trend",
    "seed_spread",
)


def _report(seeds):
    return health.training_health_report(
        run_id="run-1",
        config_sha256="abc",
        created_at="2026-10-14T12:00:00Z",
        transition=TRANSITION_INFO,
        reward=REWARD_INFO,
        thresholds=THRESHOLDS,
        seeds=seeds,
    )


def _input(seeds, **kwargs):
    return health.health_input_record(
        _report(seeds),
        generated_at=GENERATED_AT,
        code_sha="a" * 40,
        scenario_version="1.1.0",
        thresholds=THRESHOLDS,
        **kwargs,
    )


def test_the_health_input_uses_the_agreed_envelope():
    record = _input([_seed(0), _seed(1), _seed(2)])
    assert record["artifact"] == "rl_training_health"
    assert record["schema_version"] == "1.0"
    assert record["generated_at"] == GENERATED_AT
    assert record["code_sha"] == "a" * 40
    assert record["synthetic"] is True
    assert record["run_id"] == "run-1"
    assert record["algorithm"] == "ppo"
    assert record["status"] == "completed"
    assert record["transition_status"] == "engineering_only_unapproved_transition"
    assert record["seeds"] == 3
    assert record["scenario_version"] == "1.1.0"
    assert [check["check_id"] for check in record["checks"]] == list(CHECK_IDS)
    for check in record["checks"]:
        assert set(check) == {"check_id", "status", "value", "unit", "threshold", "detail"}
        assert check["status"] in ("pass", "warn", "fail", "not_run")
    json.dumps(record, allow_nan=False)


def test_clean_seeds_pass_the_hard_checks():
    checks = {c["check_id"]: c for c in _input([_seed(0), _seed(1), _seed(2)])["checks"]}
    for check_id in ("nan_or_inf", "exploding_values", "approx_kl", "clip_fraction"):
        assert checks[check_id]["status"] == "pass", check_id
    assert checks["entropy_collapse"]["status"] == "pass"
    assert checks["nan_or_inf"]["value"] == 0


def test_non_finite_values_and_explosions_fail():
    rollouts = _rollouts()
    rollouts[0]["non_finite_rewards"] = 2
    checks = {
        c["check_id"]: c
        for c in _input(
            [_seed(0, rollouts=rollouts), _seed(1, updates=_updates(param_max_abs=1e9))]
        )["checks"]
    }
    assert checks["nan_or_inf"]["status"] == "fail" and checks["nan_or_inf"]["value"] == 2
    assert checks["exploding_values"]["status"] == "fail"
    assert checks["exploding_values"]["threshold"] == THRESHOLDS.max_param_abs


def test_kl_clip_and_exploration_problems_warn():
    seeds = [_seed(0, updates=_updates(approx_kl=0.5, clip_fraction=0.9, action_std=0.001))]
    checks = {c["check_id"]: c for c in _input(seeds)["checks"]}
    assert checks["approx_kl"]["status"] == "warn" and checks["approx_kl"]["value"] == 0.5
    assert checks["clip_fraction"]["status"] == "warn"
    assert checks["entropy_collapse"]["status"] == "warn"


def test_a_falling_dev_objective_warns_on_reward_trend():
    record = _seed(0)
    record["dev_selection"]["evaluations"] = [
        {"timesteps": 0, "score": 0.03},
        {"timesteps": 100, "score": 0.01},
    ]
    checks = {c["check_id"]: c for c in _input([record])["checks"]}
    assert checks["reward_trend"]["status"] == "warn"
    assert checks["reward_trend"]["value"] == pytest.approx(-0.02)


def test_a_growing_value_loss_warns():
    updates = _updates()
    updates[-1]["value_loss"] = 1.0
    checks = {c["check_id"]: c for c in _input([_seed(0, updates=updates)])["checks"]}
    assert checks["value_loss"]["status"] == "warn"


def test_seed_spread_needs_three_seeds_to_be_judged():
    one = {c["check_id"]: c for c in _input([_seed(0)])["checks"]}
    assert one["seed_spread"]["status"] == "warn"
    three = {c["check_id"]: c for c in _input([_seed(0), _seed(1), _seed(2)])["checks"]}
    assert three["seed_spread"]["status"] == "pass"
    assert three["seed_spread"]["value"] == 0.0


def test_a_failed_run_reports_its_checks_as_not_run():
    record = health.failed_health_input_record(
        run_id="run-2",
        generated_at=GENERATED_AT,
        code_sha=None,
        scenario_version="1.1.0",
        transition=TRANSITION_INFO,
        seeds=3,
        detail="RuntimeError: boom",
    )
    assert record["status"] == "failed" and record["code_sha"] is None
    assert {c["status"] for c in record["checks"]} == {"not_run"}
    json.dumps(record, allow_nan=False)


def test_health_inputs_are_written_once_under_their_artifact_folder(tmp_path):
    record = _input([_seed(0)])
    path = health.write_health_input(record, tmp_path)
    assert path == tmp_path / "rl_training_health" / "run-1.json"
    assert json.loads(path.read_text()) == record
    with pytest.raises(FileExistsError):
        health.write_health_input(record, tmp_path)


def test_an_invalid_code_sha_is_refused():
    with pytest.raises(ValueError, match="code_sha"):
        health.health_input_record(
            _report([_seed(0)]),
            generated_at=GENERATED_AT,
            code_sha="abc",
            scenario_version="1.1.0",
            thresholds=THRESHOLDS,
        )


def test_a_missing_dev_score_is_unhealthy_and_fails_nan_or_inf_without_crashing():
    record = health.seed_health(
        0,
        updates=_updates(),
        rollouts=_rollouts(),
        evaluations=[
            {"timesteps": 0, "score": 0.01},
            {"timesteps": 100, "score": None},
            {"timesteps": 200, "score": 0.02},
        ],
        thresholds=THRESHOLDS,
    )
    assert "non_finite_dev_score" in record["reasons"]
    assert record["dev_selection"]["selected_timesteps"] == 200
    assert record["dev_selection"]["non_finite_scores"] == 1
    checks = {c["check_id"]: c for c in _input([record])["checks"]}
    assert checks["nan_or_inf"]["status"] == "fail" and checks["nan_or_inf"]["value"] == 1


def test_no_finite_dev_score_selects_nothing():
    record = health.seed_health(
        0,
        updates=_updates(),
        rollouts=_rollouts(),
        evaluations=[{"timesteps": 0, "score": None}],
        thresholds=THRESHOLDS,
    )
    assert record["dev_selection"]["selected_timesteps"] is None
    assert record["dev_selection"]["selected_score"] is None
    json.dumps(_input([record]), allow_nan=False)


def test_the_envelope_records_whether_the_tree_was_dirty():
    assert _input([_seed(0)], git_dirty=True)["git_dirty"] is True
    assert _input([_seed(0)])["git_dirty"] is None


def test_the_seed_spread_detail_never_lists_dev_scores():
    seeds = []
    for seed, best in enumerate([0.0123, 0.0456, -0.0789]):
        record = _seed(seed)
        record["dev_selection"]["selected_score"] = best
        seeds.append(record)
    check = next(c for c in _input(seeds)["checks"] if c["check_id"] == "seed_spread")
    for score in ("0.0123", "0.0456", "0.0789"):
        assert score not in check["detail"]
