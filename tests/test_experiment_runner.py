"""The experiment runner: configs, recorded provenance, outputs, determinism, planted signals."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stock_agent.forecasting import artifacts as fa
from stock_agent.forecasting.config import ConfigError, load_config, parse_config
from stock_agent.forecasting.experiment import run_experiment
from stock_agent.model_diagnostics.artifacts import read_ledger, read_run
from stock_agent.registry import open_registry

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "ml"
PROVENANCE = {
    "code": {"git_sha": "0" * 40, "git_dirty": False},
    "environment": {"python": "3.12"},
}
SMALL = {
    "schema_version": "1.0",
    "experiment": "test_planted",
    "track": "equity_cross_sectional",
    "description": "small planted-signal run for tests",
    "data": {
        "source": "synthetic_equity_panel",
        "params": {
            "n_assets": 16,
            "n_sessions": 780,
            "start": "2012-01-03",
            "signal": "linear",
            "signal_strength": 0.001,
            "late_listings": 2,
            "delistings": 0,
            "seed": 3,
        },
    },
    "universe": {"id": "test_universe", "min_history_sessions": 60},
    "features": {"families": ["relative_technical", "risk_liquidity", "fundamental_quality"]},
    "label": {"horizon": 10, "entry_lag": 1, "benchmark": "equal_weight"},
    "folds": {
        "start": "2013-09-02",
        "block_sessions": 42,
        "inner_block_sessions": 42,
        "inner_windows": 2,
        "inner_min_train_sessions": 126,
    },
    "models": {"candidates": ["ridge"], "include_ols": True, "increment_reference": "ridge"},
    "evaluation": {
        "seed": 0,
        "permutation_draws": 1,
        "permutation_models": ["ridge"],
        "bootstrap_reps": 19,
        "calibration_reps": 20,
        "calibration_bootstrap_reps": 9,
    },
    "registry": {"model_prefix": "test_xs"},
}


def config_with(**changes):
    raw = copy.deepcopy(SMALL)
    for path, value in changes.items():
        target = raw
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return parse_config(raw)


def run(config, root, minute=0):
    return run_experiment(
        config,
        output_root=root,
        provenance=PROVENANCE,
        started_at=datetime(2026, 10, 14, 12, minute, tzinfo=UTC),
        clock=lambda: datetime(2026, 10, 14, 13, minute, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    root = tmp_path_factory.mktemp("planted")
    config = config_with()
    first = run(config, root, minute=0)
    second = run(config, root, minute=1)
    return root, config, first, second


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    root = tmp_path_factory.mktemp("null")
    return run(config_with(data__params__signal="none", experiment="test_null"), root)


# --- configs ---


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_parses(path):
    config = load_config(path)
    assert config.data_class == "synthetic"
    assert config.sha256 == load_config(path).sha256


def test_the_config_hash_ignores_key_order_and_formatting(tmp_path):
    reordered = dict(reversed(list(copy.deepcopy(SMALL).items())))
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(reordered, sort_keys=True))
    assert load_config(path).sha256 == parse_config(SMALL).sha256
    assert config_with(evaluation__seed=1).sha256 != parse_config(SMALL).sha256


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("data__source", "yfinance_prices", "Only synthetic sources"),
        ("label__entry_lag", 0, "entry_lag"),
        ("models__candidates", ["ridge", "xgboost"], "candidates"),
        ("models__increment_reference", "lasso", "increment_reference"),
        ("evaluation__surprise", 1, "unknown keys"),
        ("folds__start", "2013-09-02T12:00", "date"),
        ("universe__id", "Has Spaces", "universe.id"),
    ],
)
def test_malformed_or_real_data_configs_are_refused(path, value, match):
    with pytest.raises(ConfigError, match=match):
        config_with(**{path: value})


def test_synthetic_data_must_end_before_the_lockbox(tmp_path):
    late = config_with(data__params__start="2022-06-01")
    with pytest.raises(ValueError, match="lockbox"):
        run(late, tmp_path)
    assert read_ledger(tmp_path)[-1]["status"] == "failed"


def test_unknown_permutation_models_fail_and_are_ledgered(tmp_path):
    with pytest.raises(ConfigError, match="permutation_models"):
        run(config_with(evaluation__permutation_models=["nope"]), tmp_path)
    line = read_ledger(tmp_path)[-1]
    assert line["status"] == "failed" and "permutation_models" in line["error"]


# --- outputs and provenance ---


def test_the_record_states_what_reproduces_the_run(planted):
    _, config, result, _ = planted
    record = result.record
    assert record["code"] == PROVENANCE["code"]
    assert record["config_sha256"] == config.sha256
    assert set(record["data"]["fingerprints"]) == {
        "prices_sha256",
        "fundamentals_sha256",
        "universe_sha256",
        "features_sha256",
        "labeled_frame_sha256",
    }
    assert record["seeds"]["data_seed"] == 3 and record["seeds"]["evaluation_seed"] == 0
    assert record["label"]["label_spec"]["horizon"] == 10
    assert record["label"]["label_spec"]["entry_lag"] == 1
    assert record["folds"]["n_folds"] >= 3 and len(record["folds"]["fold_table_sha256"]) == 64
    assert record["universe"]["point_in_time"] is True
    assert record["lockbox_evidence"]["holdout_values_used"] is False


def test_planted_persistence_trips_only_the_canary_check(planted):
    # A forecastable planted drift is persistent (constant within a quarter), so the
    # memorizer's latest matured label legitimately predicts the next: T2's canary
    # check fires on planted data. Every other leakage check holds; the null run
    # below shows the canary itself is quiet without a signal.
    _, _, result, _ = planted
    checks = result.diagnostics.tables["checks"]
    failed = checks[(checks["severity"] == "leakage") & checks["passed"].eq(False).fillna(False)]
    assert set(failed["check"]) <= {"canary_shows_no_skill_through_harness"}
    unsafe = checks[checks["check"] == "canary_detects_unpurged_leakage"]
    assert unsafe["passed"].all()


def test_outputs_are_versioned_and_verifiable(planted):
    root, _, result, _ = planted
    stored = json.loads((result.directory / "record.json").read_text())
    assert stored["run_id"] == result.run_id
    _, tables = read_run(root / "runs" / result.run_id)  # a T2 contract run (dashboard-readable)
    assert set(tables["predictions"]["model"]) >= {"ridge", "ols", "zero", "memorizer"}
    forecast = result.record["forecasts"]["ridge"]
    artifact = fa.read_forecast_artifact(root / forecast["path"])
    assert artifact.metadata.model_id == "test_xs.ridge" and artifact.metadata.model_version == 1
    ridge = tables["predictions"].query("model == 'ridge'").sort_values(["date", "symbol"])
    np.testing.assert_array_equal(artifact.frame["prediction"], ridge["prediction"])
    metrics = pd.read_parquet(result.directory / "metrics.parquet")
    assert {"rank_ic_coverage", "coverage_weighted_rank_ic", "holm_adjusted_p"} <= set(
        metrics["metric"]
    )
    calibration = metrics.query("metric == 'interval_calibration'")
    assert (calibration["reason"] == "point_forecast").all()
    missingness = pd.read_parquet(result.directory / "missingness.parquet")
    assert set(missingness["scope"]) == {"panel", "training", "evaluation"}


def test_runs_register_candidates_with_computed_stages(planted):
    root, _, first, second = planted
    reader = open_registry(root / "registry")
    entry = reader.entry("test_xs.ridge@v1")
    assessment = reader.assessment(entry.ref)
    reasons = {c.reason for c in assessment.blockers}
    assert assessment.stage == "engineering_only"
    assert "data_class_engineering_only" in reasons
    assert not reasons & {"artifacts_unverified", "evidence_not_reproducible"}
    assert reader.entry("test_xs.ridge@v2").parent == entry.ref
    assert [str(e.ref) for e in second.entries] == ["test_xs.ridge@v2"]
    assert first.record["multiple_testing"]["prior_variants_in_scope"] == 0
    assert second.record["multiple_testing"]["prior_variants_in_scope"] == 1
    comparison = reader.compare("test_xs.ridge@v1", "test_xs.ridge@v2")
    assert comparison.same_training_data and comparison.same_folds
    assert (comparison.table["b_minus_a"].fillna(0.0) == 0.0).all()


def test_identical_configs_give_identical_results_under_new_run_ids(planted):
    root, _, first, second = planted
    assert first.run_id != second.run_id
    assert first.record["outputs"] == second.record["outputs"]
    lines = read_ledger(root)
    assert [line["status"] for line in lines] == ["completed", "completed"]
    assert lines[0]["results_sha256"] == lines[1]["results_sha256"]


# --- planted signal and null data ---


def _summary(result, model="ridge"):
    return result.record["summary"][model]


def test_the_pipeline_recovers_a_planted_signal(planted):
    _, _, result, _ = planted
    ridge = _summary(result)
    assert ridge["rank_ic_coverage"] == 1.0
    assert ridge["coverage_weighted_rank_ic"] > 0.1
    assert ridge["coverage_weighted_rank_ic_t"] > 3.0
    assert ridge["oos_r2_vs_zero"] > 0.0
    assert ridge["holm_adjusted_p"] < 0.01


def test_the_pipeline_finds_nothing_in_null_data(null):
    for model in ("ridge", "ols"):
        summary = _summary(null, model)
        assert abs(summary["coverage_weighted_rank_ic_t"]) < 3.0
        assert summary["oos_r2_vs_zero"] < 0.01
    assert _summary(null)["holm_adjusted_p"] > 0.05
    assert null.record["checks_summary"]["leakage_checks_failed"] == 0
