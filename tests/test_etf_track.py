"""The ETF return and risk track: synthetic series, per-asset models, honest metrics."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from ml_helpers import PROVENANCE

from stock_agent.features.invariant_labels import assert_labels_prefix_invariant
from stock_agent.forecasting import artifacts as fa
from stock_agent.forecasting import etf_data
from stock_agent.forecasting.config import ConfigError, parse_config
from stock_agent.forecasting.experiment import run_experiment
from stock_agent.model_validation.checks import assert_prefix_stable
from stock_agent.registry import open_registry

ETF = {
    "schema_version": "1.0",
    "experiment": "test_etf",
    "track": "etf_return_risk",
    "description": "small ETF run for tests",
    "data": {
        "source": "synthetic_etf_panel",
        "params": {
            "n_sessions": 900,
            "start": "2012-01-02",
            "return_signal": "trend",
            "signal_strength": 0.3,
            "volatility": "garch",
            "seed": 4,
        },
    },
    "universe": {"id": "test_etf_6", "min_history_sessions": 61},
    "features": {"families": ["relative_technical", "risk_liquidity"]},
    "label": {"horizon": 10, "entry_lag": 1, "benchmark": None},
    "folds": {
        "start": "2013-12-02",
        "block_sessions": 42,
        "inner_block_sessions": 42,
        "inner_windows": 2,
        "inner_min_train_sessions": 126,
    },
    "models": {"candidates": ["ridge"], "include_ols": True, "increment_reference": "ridge"},
    "evaluation": {"seed": 0, "permutation_draws": 0, "bootstrap_reps": 49},
    "registry": {"model_prefix": "test_etf"},
}
SPEC = etf_data.SyntheticEtfSpec(n_sessions=600, seed=2)


def etf_config(**changes):
    raw = copy.deepcopy(ETF)
    for path, value in changes.items():
        target = raw
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return parse_config(raw)


def run_etf(config, root):
    return run_experiment(
        config,
        output_root=root,
        provenance=PROVENANCE,
        started_at=datetime(2026, 10, 14, 12, tzinfo=UTC),
        clock=lambda: datetime(2026, 10, 14, 13, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    root = tmp_path_factory.mktemp("etf_planted")
    return root, run_etf(etf_config(), root)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    root = tmp_path_factory.mktemp("etf_null")
    config = etf_config(
        experiment="test_etf_null",
        data__params__return_signal="none",
        data__params__volatility="constant",
    )
    return run_etf(config, root)


# --- data ---


def test_the_generator_is_deterministic_heterogeneous_and_before_the_lockbox():
    first, second = etf_data.synthetic_etf_panel(SPEC), etf_data.synthetic_etf_panel(SPEC)
    pd.testing.assert_frame_equal(first.prices, second.prices)
    truth = first.truth
    vols = truth.groupby("symbol")["conditional_vol"].mean()
    assert vols["BOND_SHORT"] < vols["BOND_AGG"] < vols["EQ_US"]
    drift_moves = truth.groupby("symbol")["drift"].std()
    trend = [name for name, profile in etf_data.ASSETS.items() if profile["trend"]]
    assert (drift_moves[trend] > 0).all() and (drift_moves.drop(trend) == 0).all()
    flat = etf_data.synthetic_etf_panel(replace(SPEC, volatility="constant", return_signal="none"))
    assert (flat.truth.groupby("symbol")["conditional_vol"].std() < 1e-15).all()
    assert first.prices["date"].max() < pd.Timestamp("2025-01-01", tz="UTC")
    with pytest.raises(ValueError, match="lockbox"):
        etf_data.synthetic_etf_panel(replace(SPEC, start="2023-06-01"))


def test_features_are_prefix_stable():
    panel = etf_data.synthetic_etf_panel(SPEC)
    dates = panel.prices["date"].drop_duplicates().sort_values()
    assert_prefix_stable(
        etf_data.etf_features,
        panel.prices,
        cutoffs=dates.iloc[[120, 300, 520]],
        value_columns=list(etf_data.CONTRACTS_BY_NAME),
    )


@pytest.mark.parametrize("target", etf_data.TARGETS)
def test_both_targets_are_prefix_invariant(target):
    panel = etf_data.synthetic_etf_panel(SPEC)
    dates = panel.prices["date"].drop_duplicates().sort_values()

    def build(prices):
        return etf_data.etf_labels(prices, horizon=10, entry_lag=1, target=target)

    assert_labels_prefix_invariant(build, panel.prices, cutoffs=dates.iloc[[100, 300, 590]])


def test_the_volatility_label_covers_exactly_the_returns_after_entry():
    panel = etf_data.synthetic_etf_panel(SPEC)
    labels = etf_data.etf_labels(
        panel.prices, horizon=10, entry_lag=1, target="log_realized_volatility"
    )
    prices = panel.prices.query("symbol == 'EQ_US'").reset_index(drop=True)
    row = labels.query("symbol == 'EQ_US'").reset_index(drop=True).iloc[50]
    log_returns = np.log(prices["close"] / prices["close"].shift(1))
    window = log_returns.iloc[52:62]  # entry at row 51; returns 52..61 end at the exit row 61
    assert row["target_start_date"] == prices["date"].iloc[51]
    assert row["target_end_date"] == prices["date"].iloc[61]
    assert row["target_return"] == pytest.approx(0.5 * np.log((window**2).mean()))


# --- configs ---


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("label__benchmark", "equal_weight", "benchmark must be null"),
        ("features__families", ["fundamental_quality"], "no feature family"),
        ("evaluation__permutation_models", ["ridge"], "permutation"),
    ],
)
def test_etf_configs_refuse_what_the_track_cannot_do(tmp_path, path, value, match):
    with pytest.raises(ConfigError, match=match):
        run_etf(etf_config(**{path: value}), tmp_path)


def test_a_source_must_serve_its_track():
    with pytest.raises(ConfigError, match="does not serve"):
        etf_config(track="equity_cross_sectional")


# --- the track ---


def test_every_asset_gets_its_own_models_and_selection(planted):
    _, result = planted
    selection = result.tables["selection"]
    assert set(selection["asset"]) == set(etf_data.ASSETS)
    assert set(selection["target"]) == set(etf_data.TARGETS)
    chosen = selection[selection["chosen"]].groupby(["target", "asset"])["penalty"].apply(tuple)
    assert chosen.nunique() > 1  # assets are tuned separately, not forced to one choice
    predictions = result.tables["predictions"]
    assert not predictions.duplicated(["target", "model", "date", "symbol"]).any()


def test_every_leakage_check_holds_and_entries_are_engineering_only(planted):
    root, result = planted
    summary = result.record["checks_summary"]
    assert summary["leakage_checks_failed"] == 0 and summary["instrument_checks_failed"] == 0
    assert summary["families"]["leakage"]["total"] > 0
    reader = open_registry(root / "registry")
    for ref in ("test_etf.ret.ridge@v1", "test_etf.vol.ridge@v1"):
        assessment = reader.assessment(ref)
        reasons = {c.reason for c in assessment.blockers}
        assert assessment.stage == "engineering_only"
        assert "data_class_engineering_only" in reasons
        assert not reasons & {"artifacts_unverified", "evidence_not_reproducible"}


def test_forecast_artifacts_carry_return_intervals_and_volatility_points(planted):
    root, result = planted
    returns = fa.read_forecast_artifact(root / result.record["forecasts"]["ret.ridge"]["path"])
    volatility = fa.read_forecast_artifact(root / result.record["forecasts"]["vol.ridge"]["path"])
    assert returns.metadata.forecast_kind == "interval"
    assert (returns.frame["interval_level"] == 0.8).all()
    assert volatility.metadata.forecast_kind == "point"
    assert set(returns.frame["asset"]) == set(etf_data.ASSETS)
    assert len(returns.frame) == len(volatility.frame)


def test_clark_west_is_reported_only_for_the_nested_pair(planted):
    _, result = planted
    metrics = result.tables["metrics"]
    rows = metrics[metrics["metric"].str.startswith("clark_west")]
    assert set(rows["model"]) == {"ols"}
    assert set(rows["target"]) == {"forward_return"}
    assert set(rows["metric"]) == {"clark_west_vs_historical_mean"}


def test_results_are_reported_per_asset_and_pooled(planted):
    _, result = planted
    metrics = result.tables["metrics"]
    r2 = metrics.query("metric == 'oos_r2_vs_historical_mean' and model == 'ridge'")
    assert set(r2["asset"]) == {*etf_data.ASSETS, "__pooled__"}


def test_planted_volatility_and_trend_are_recovered(planted):
    _, result = planted
    volatility = result.record["summary"]["vol.ridge"]
    assert volatility["oos_r2_vs_historical_mean"] > 0.05
    assert volatility["mse_improvement_vs_historical_mean_t"] > 2.0
    assert volatility["qlike_improvement_vs_persistence_t"] > 2.0
    returns = result.record["summary"]["ret.ridge"]
    assert returns["oos_r2_vs_historical_mean"] > 0.0
    assert returns["mse_improvement_vs_historical_mean_t"] > 2.0


def test_null_series_show_nothing_and_intervals_are_calibrated(null):
    for key in ("ret.ridge", "vol.ridge"):
        summary = null.record["summary"][key]
        assert summary["oos_r2_vs_historical_mean"] < 0.01
        assert summary["mse_improvement_vs_historical_mean_t"] < 2.0
    returns = null.record["summary"]["ret.ridge"]
    for level in (50, 80, 90):
        assert abs(returns[f"interval_coverage_{level}"] - level / 100) < 0.05
    assert null.record["checks_summary"]["leakage_checks_failed"] == 0
