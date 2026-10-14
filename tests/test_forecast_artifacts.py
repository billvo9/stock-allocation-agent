"""Out-of-fold forecast artifacts: contract, write-once storage, as-of selection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.forecasting import artifacts as fa
from stock_agent.model_diagnostics.controls import PooledMeanForecast
from stock_agent.model_validation.audit import file_sha256
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows
from stock_agent.model_validation.harness import run_walk_forward
from stock_agent.registry import ArtifactRef, Registry

METADATA = fa.ForecastMetadata(
    model_id="equity_xs.ridge",
    model_version=3,
    variant_id="abc123",
    run_id="20261014T000000Z-aaaaaaaaaaaa",
    track="equity_cross_sectional",
    target="forward_return",
    horizon=5,
    label_id="forward_simple_return:adjusted_close:horizon=5:entry_lag=1",
    forecast_kind="point",
    data_class="synthetic",
    knowledge_rule=fa.SYNTHETIC_END_OF_UTC_DATE,
    config_sha256="c" * 64,
    code={"git_sha": "d" * 40, "git_dirty": False},
)


@pytest.fixture(scope="module")
def harness_output(panel_factory):
    labeled = panel_factory(sessions=160, horizon=5)
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(
        sessions, start=sessions.iloc[80], end=sessions.iloc[150], block_sessions=20
    )
    folds = make_expanding_folds(labeled, windows, holdout_start=sessions.iloc[150])
    result = run_walk_forward(
        labeled, folds, feature_columns=["feature_a"], make_estimator=PooledMeanForecast
    )
    return labeled, folds, result.predictions


@pytest.fixture
def table(harness_output) -> pd.DataFrame:
    labeled, folds, predictions = harness_output
    return fa.forecast_frame(predictions, labeled, folds, metadata=METADATA)


def test_the_table_has_one_out_of_fold_row_per_date_and_asset(harness_output, table):
    labeled, folds, predictions = harness_output
    assert len(table) == len(predictions)
    assert not table.duplicated(["date", "asset"]).any()
    for fold in folds:
        rows = table[table["fold_id"] == fold.fold_id]
        train = labeled.iloc[fold.train_rows]
        assert (rows["train_label_end_max"] == train["target_end_date"].max()).all()
        assert (rows["train_label_end_max"] < fold.test_start).all()
        assert (rows["train_end"] == train["date"].max()).all()
        assert ((rows["date"] >= fold.test_start) & (rows["date"] < fold.test_end)).all()


def test_available_at_is_the_end_of_the_synthetic_utc_date(table):
    expected = table["date"] + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    assert (table["available_at"] == expected).all()
    assert str(table["available_at"].iloc[0]).endswith("23:59:59.999999+00:00")


def test_write_then_read_round_trips_and_returns_the_manifest_hash(table, tmp_path):
    directory, manifest_sha = fa.write_forecast_artifact(table, METADATA, tmp_path / "forecasts")
    artifact = fa.read_forecast_artifact(directory)
    pd.testing.assert_frame_equal(artifact.frame, table)
    assert artifact.metadata == METADATA
    assert artifact.manifest_sha256 == manifest_sha
    assert directory.name.startswith("equity_xs.ridge-v0003-")
    with pytest.raises(FileExistsError):
        fa.write_forecast_artifact(table, METADATA, tmp_path / "forecasts")


def test_the_registry_verifies_a_forecast_artifact_by_its_manifest(table, tmp_path):
    directory, manifest_sha = fa.write_forecast_artifact(table, METADATA, tmp_path / "forecasts")
    reference = ArtifactRef("forecast_artifact", f"forecasts/{directory.name}", manifest_sha)
    registry = Registry(tmp_path / "registry", tmp_path)
    entry = registry.register(
        model_id="equity_xs.ridge",
        research_scope="equity_xs:synthetic:h5",
        config={},
        training_data={"data_class": "synthetic"},
        metrics={},
        artifacts=[reference],
        evidence={},
        code={},
    )
    assert registry.artifacts_verified(entry)
    (directory / fa.DATA_FILE).write_bytes(b"tampered")
    assert not registry.artifacts_verified(entry)
    assert registry.stage(entry.ref) == "engineering_only"


@pytest.mark.parametrize(
    "tamper",
    ["data", "data_and_file_hash", "metadata", "rename"],
)
def test_tampered_artifacts_are_refused(table, tmp_path, tamper):
    directory, _ = fa.write_forecast_artifact(table, METADATA, tmp_path)
    if tamper.startswith("data"):
        frame = pd.read_parquet(directory / fa.DATA_FILE)
        frame.loc[0, "prediction"] += 1.0
        frame.to_parquet(directory / fa.DATA_FILE, index=False)
        if tamper == "data_and_file_hash":  # an editor who also rewrites the file hash
            manifest = json.loads((directory / fa.MANIFEST_FILE).read_text())
            manifest["files"][fa.DATA_FILE]["file_sha256"] = file_sha256(directory / fa.DATA_FILE)
            (directory / fa.MANIFEST_FILE).write_text(json.dumps(manifest))
    elif tamper == "metadata":
        payload = json.loads((directory / fa.METADATA_FILE).read_text())
        payload["model_version"] = 4
        (directory / fa.METADATA_FILE).write_text(json.dumps(payload))
    else:
        directory = directory.rename(directory.parent / "another-name")
    with pytest.raises(fa.ForecastArtifactError):
        fa.read_forecast_artifact(directory)


def _violates(table: pd.DataFrame, change, match: str, metadata=METADATA) -> None:
    broken = table.copy()
    change(broken)
    with pytest.raises(fa.ForecastArtifactError, match=match):
        fa.validate_forecasts(fa.conform(broken), metadata)


def test_leaky_or_malformed_rows_are_refused(table):
    day = pd.Timedelta(days=1)

    def duplicate(frame):
        frame.loc[1, ["date", "asset"]] = frame.loc[0, ["date", "asset"]].to_numpy()

    def label_reaches_test(frame):
        frame.loc[0, "train_label_end_max"] = frame.loc[0, "test_start"]

    def train_overlaps_test(frame):
        frame.loc[0, "train_end"] = frame.loc[0, "test_start"]

    def outside_window(frame):
        frame.loc[0, "date"] = frame.loc[0, "test_start"] - day

    def early_availability(frame):
        frame.loc[0, "available_at"] = frame.loc[0, "date"] + pd.Timedelta(hours=21)

    def other_version(frame):
        frame.loc[0, "model_version"] = 2

    def non_finite(frame):
        frame.loc[0, "prediction"] = np.nan

    def intraday(frame):
        frame.loc[0, "date"] = frame.loc[0, "date"] + pd.Timedelta(hours=1)

    _violates(table, duplicate, "one row per")
    _violates(table, label_reaches_test, "Not out-of-fold")
    _violates(table, train_overlaps_test, "precede its test window")
    _violates(table, outside_window, "inside its fold's test window")
    _violates(table, early_availability, "before its inputs are known")
    _violates(table, other_version, "model_version")
    _violates(table, non_finite, "finite")
    _violates(table, intraday, "00:00 UTC")


def test_folds_must_keep_one_disjoint_test_window(table):
    def widen(frame):
        first = frame["fold_id"] == frame["fold_id"].min()
        frame.loc[first.idxmax(), "test_end_exclusive"] += pd.Timedelta(days=3)

    def overlap(frame):
        second = frame["fold_id"] == sorted(frame["fold_id"].unique())[1]
        frame.loc[second, "test_start"] -= pd.Timedelta(days=5)
        frame.loc[second, "train_end"] -= pd.Timedelta(days=5)
        frame.loc[second, "train_label_end_max"] -= pd.Timedelta(days=30)

    _violates(table, widen, "same test window")
    _violates(table, overlap, "disjoint")


def test_lockbox_dates_and_real_data_are_refused(table):
    def into_lockbox(frame):
        frame.loc[0, "available_at"] = pd.Timestamp("2025-01-01", tz="UTC")

    _violates(table, into_lockbox, "lockbox")
    with pytest.raises(fa.ForecastArtifactError, match="OD9 pending"):
        replace(METADATA, data_class="research_panel")
    with pytest.raises(fa.ForecastArtifactError, match="lockbox"):
        replace(METADATA, lockbox_start="2026-01-01T00:00:00+00:00")
    with pytest.raises(fa.ForecastArtifactError, match="approved rule"):
        replace(METADATA, knowledge_rule="close_at_16h")


def test_interval_columns_follow_the_forecast_kind(table, harness_output):
    def add_interval(frame):
        frame.loc[0, ["interval_low", "interval_high", "interval_level"]] = [-1.0, 1.0, 0.8]

    _violates(table, add_interval, "Point forecasts")
    labeled, folds, predictions = harness_output
    interval = replace(METADATA, forecast_kind="interval")
    bounds = predictions[["fold_id", "date", "symbol"]].assign(
        interval_low=predictions["prediction"] - 0.1,
        interval_high=predictions["prediction"] + 0.1,
        interval_level=0.8,
    )
    frame = fa.forecast_frame(predictions, labeled, folds, metadata=interval, intervals=bounds)
    assert (frame["interval_level"] == 0.8).all()

    def reversed_bounds(frame):
        frame.loc[0, "interval_low"] = frame.loc[0, "interval_high"] + 1.0

    _violates(frame, reversed_bounds, "low <= high", metadata=interval)


def test_asof_selection_uses_only_available_forecasts(table):
    first_date = table["date"].min()
    known = first_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    assert fa.forecasts_asof(table, known - pd.Timedelta(microseconds=1)).empty
    at_known = fa.forecasts_asof(table, known)
    assert (at_known["date"] == first_date).all()
    assert len(at_known) == table.loc[table["date"] == first_date, "asset"].nunique()

    decision = table["date"].sort_values().unique()[10] + pd.Timedelta(hours=12)
    chosen = fa.forecasts_asof(table, decision)
    assert (chosen["available_at"] <= decision).all()
    for row in chosen.itertuples(index=False):
        usable = table[(table["asset"] == row.asset) & (table["available_at"] <= decision)]
        assert row.date == usable["date"].max()
    assert (chosen["date"] < table["date"].max()).all()
    with pytest.raises(fa.ForecastArtifactError, match="timezone-aware"):
        fa.forecasts_asof(table, "2021-06-01")


def test_metadata_is_canonical_json(tmp_path: Path, table):
    directory, _ = fa.write_forecast_artifact(table, METADATA, tmp_path)
    payload = json.loads((directory / fa.METADATA_FILE).read_text())
    assert payload["knowledge_rule"] == "synthetic_end_of_utc_date"
    assert payload["forecast_schema_version"] == fa.FORECAST_SCHEMA_VERSION
    manifest = json.loads((directory / fa.MANIFEST_FILE).read_text())
    assert set(manifest["files"]) == {fa.DATA_FILE, fa.METADATA_FILE}
