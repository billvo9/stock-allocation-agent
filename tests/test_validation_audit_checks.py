"""
Feature contract, prefix stability of the production feature builders,
canonical fingerprints, fold metadata, and the run manifest.
"""

from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from stock_agent.data.macro.schema import MACRO_COLUMNS
from stock_agent.features.build import load_feature_query
from stock_agent.features.macro import build_macro_features
from stock_agent.features.training import MODEL_LABEL_SPEC
from stock_agent.model_validation.audit import (
    describe_folds,
    frame_content_sha256,
    frame_keys_sha256,
    frame_rows_sha256,
    run_manifest,
)
from stock_agent.model_validation.checks import (
    LeakageError,
    assert_prefix_stable,
    validate_feature_columns,
)
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows

PRICES_PATH_IN_SQL = "data/raw/prices.parquet"


def _ts(value):
    return pd.Timestamp(value, tz="UTC")


# --- feature contract ---


@pytest.fixture
def contract_frame():
    return pd.DataFrame(
        {
            "date": [_ts("2024-01-02")],
            "symbol": ["A"],
            "momentum_20d": [0.1],
            "is_large": [True],
            "target_return": [0.02],
            "target_end_date": [_ts("2024-02-01")],
            "adjusted_close": [101.0],
            "close": [100.0],
            "volume": [1e6],
            "label_excess": [0.0],
            "fwd_return_5d": [0.0],
            "future_vol": [0.0],
            "report_date": [_ts("2024-01-01")],
            "edgar_available_at": [_ts("2024-01-01")],
            "sector": ["semis"],
            "lag": [pd.Timedelta(days=1)],
        }
    )


def test_feature_contract_accepts_numeric_and_bool_features(contract_frame):
    assert validate_feature_columns(contract_frame, ["momentum_20d", "is_large"]) == [
        "momentum_20d",
        "is_large",
    ]


@pytest.mark.parametrize(
    "column",
    [
        "date",
        "symbol",
        "target_return",
        "target_end_date",
        "adjusted_close",
        "close",
        "volume",
        "label_excess",
        "fwd_return_5d",
        "future_vol",
        "report_date",
        "edgar_available_at",
        "sector",
        "lag",
        "not_a_column",
    ],
)
def test_feature_contract_rejects_labels_keys_timestamps_levels_and_non_numeric(
    contract_frame, column
):
    with pytest.raises(ValueError, match="Invalid feature columns"):
        validate_feature_columns(contract_frame, ["momentum_20d", column])


@pytest.mark.parametrize("columns", [[], ["momentum_20d", "momentum_20d"]])
def test_feature_contract_rejects_empty_or_duplicate_lists(contract_frame, columns):
    with pytest.raises(ValueError):
        validate_feature_columns(contract_frame, columns)


# --- prefix stability ---


def _raw_prices(seed=0, sessions=90):
    rng = np.random.RandomState(seed)
    calendar = pd.bdate_range("2023-01-02", periods=sessions)  # naive, like prices.parquet
    frames = [
        pd.DataFrame(
            {
                "date": calendar,
                "symbol": symbol,
                "adjusted_close": 50.0 * np.exp(np.cumsum(0.02 * rng.randn(sessions))),
            }
        )
        for symbol in ("A", "B", "C")
    ]
    return pd.concat(frames, ignore_index=True)


def _production_market_features(tmp_path):
    query = load_feature_query()
    assert PRICES_PATH_IN_SQL in query

    def build(raw):
        path = tmp_path / f"prices_{len(raw)}.parquet"
        raw.to_parquet(path, index=False)
        return duckdb.sql(query.replace(PRICES_PATH_IN_SQL, str(path))).df()

    return build


def test_production_market_feature_sql_is_prefix_stable(tmp_path):
    raw = _raw_prices()
    cutoffs = raw["date"].drop_duplicates().sort_values().iloc[[25, 50, 80]]

    assert_prefix_stable(
        _production_market_features(tmp_path),
        raw,
        cutoffs=cutoffs,
        value_columns=["adjusted_close", "daily_return", "momentum_20d", "volatility_20d"],
        rtol=1e-12,
        atol=1e-15,
    )


def _with_feature(feature):
    def build(raw):
        out = raw.sort_values(["symbol", "date"]).copy()
        out["feature"] = feature(out)
        return out

    return build


@pytest.mark.parametrize(
    "feature",
    [
        lambda f: f.groupby("symbol")["adjusted_close"].shift(-1),  # peeks one session ahead
        lambda f: f.groupby("symbol")["adjusted_close"].bfill(),  # back-fill from the future
        lambda f: (f["adjusted_close"] - f["adjusted_close"].mean()) / f["adjusted_close"].std(),
        lambda f: f.groupby("symbol")["adjusted_close"].transform("max"),  # full-history max
    ],
    ids=["negative_shift", "backfill", "full_sample_zscore", "full_history_max"],
)
def test_prefix_check_catches_features_that_use_the_future(feature):
    raw = _raw_prices()
    # A gap just before a cutoff: bfill fills it from the session after the
    # cutoff. A prefix check only detects leaks that cross one of its cutoffs.
    raw.loc[9, "adjusted_close"] = np.nan
    cutoffs = raw["date"].drop_duplicates().sort_values().iloc[[10, 40, 70]]

    with pytest.raises(LeakageError, match="feature"):
        assert_prefix_stable(
            _with_feature(feature), raw, cutoffs=cutoffs, value_columns=["feature"]
        )


def test_prefix_check_accepts_backward_looking_features():
    raw = _raw_prices()
    cutoffs = raw["date"].drop_duplicates().sort_values().iloc[[10, 40, 70]]

    assert_prefix_stable(
        _with_feature(lambda f: f.groupby("symbol")["adjusted_close"].pct_change(5)),
        raw,
        cutoffs=cutoffs,
        value_columns=["feature"],
    )


def _m2_vintages():
    rows = []
    for index, observation in enumerate(pd.date_range("2018-01-01", "2020-12-01", freq="MS")):
        value = 14_000.0 * (1.005**index)
        for lag_days, revision in [(40, 1.0), (75, 1.002)]:
            released = observation + pd.Timedelta(days=lag_days)
            rows.append(
                {
                    "observation_date": observation,
                    "available_at": released,
                    "vintage_date": released,
                    "series_id": "us_m2_money_supply",
                    "provider_series_id": "M2SL",
                    "value": value * revision,
                    "frequency": "monthly",
                    "units": "billions_usd",
                    "source": "FRED",
                }
            )
    frame = pd.DataFrame(rows, columns=MACRO_COLUMNS)
    for column in ("observation_date", "available_at", "vintage_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def test_production_macro_features_are_prefix_stable_by_availability():
    decisions = pd.bdate_range("2019-03-01", "2020-12-31", tz="UTC")

    def build(vintages):
        return build_macro_features(decision_dates=decisions, macro_vintages=vintages)

    assert_prefix_stable(
        build,
        _m2_vintages(),
        cutoffs=["2019-06-03", "2019-11-15", "2020-07-01"],
        raw_time_column="available_at",
        key_columns=("date",),
    )


# --- fingerprints ---


GOLDEN = pd.DataFrame(
    {
        "date": [_ts("2024-01-03"), _ts("2024-01-02"), _ts("2024-01-02")],
        "symbol": ["A", "B", "A"],
        "x": [1.5, -0.0, np.nan],
        "flag": [True, False, True],
    }
)


def test_fingerprints_match_pinned_values():
    # Pinned so an accidental change to the hashing scheme is noticed. The
    # byte layout is canonical, so these do not depend on pandas/numpy versions.
    assert (
        frame_keys_sha256(GOLDEN)
        == "3a50a370b57b0f3dcfa2e10040e8d5368e11cd853fdb5fb3ad2c4df33ec82c19"
    )
    assert (
        frame_content_sha256(GOLDEN, ["x", "flag"])
        == "dc422c5d29ec9e6704fcda831c3be8baf83c2a5a3586ac2f723ef2c5aba486b4"
    )
    assert (
        frame_rows_sha256(GOLDEN)
        == "6622e4070ca4784be6e9c965bbd660e362e5b6af1c2de3d5ef77cb8b4083a7b6"
    )


def test_key_and_content_fingerprints_ignore_row_order_but_row_binding_does_not():
    shuffled = GOLDEN.iloc[[2, 0, 1]].reset_index(drop=True)

    assert frame_keys_sha256(shuffled) == frame_keys_sha256(GOLDEN)
    assert frame_content_sha256(shuffled, ["x", "flag"]) == frame_content_sha256(
        GOLDEN, ["x", "flag"]
    )
    assert frame_rows_sha256(shuffled) != frame_rows_sha256(GOLDEN)


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.assign(x=[np.nextafter(1.5, 2.0), -0.0, np.nan]),
        lambda f: f.assign(x=[1.5, -0.0, 0.0]),
        lambda f: f.assign(flag=[True, True, True]),
        lambda f: f.assign(symbol=["A", "C", "A"]),
        lambda f: f.assign(date=[_ts("2024-01-03"), _ts("2024-01-02"), _ts("2024-01-04")]),
    ],
    ids=["one_ulp", "nan_to_zero", "bool", "symbol", "date"],
)
def test_content_fingerprint_changes_with_any_single_cell(change):
    assert frame_content_sha256(change(GOLDEN), ["x", "flag"]) != frame_content_sha256(
        GOLDEN, ["x", "flag"]
    )


@pytest.mark.parametrize(
    "variant",
    [
        lambda f: f.assign(date=f["date"].dt.as_unit("us")),
        lambda f: f.assign(date=f["date"].dt.tz_convert("America/New_York")),
        lambda f: f.assign(symbol=f["symbol"].astype(object)),
        lambda f: f.assign(x=[1.5, 0.0, np.nan]),  # -0.0 and 0.0 hash alike
        lambda f: f.assign(x=[1.5, -0.0, np.copysign(np.nan, -1.0)]),  # NaN sign
    ],
    ids=["datetime_unit", "timezone", "object_strings", "signed_zero", "nan_sign"],
)
def test_fingerprints_canonicalize_representation(variant):
    assert frame_content_sha256(variant(GOLDEN), ["x", "flag"]) == frame_content_sha256(
        GOLDEN, ["x", "flag"]
    )


# --- fold metadata and run manifest ---


@pytest.fixture
def folded(labeled_panel):
    panel = labeled_panel(sessions=150, seed=2)
    panel["available_at"] = panel["date"] - pd.Timedelta(days=1)
    sessions = panel["date"].drop_duplicates().sort_values()
    windows = make_test_windows(
        sessions, start=sessions.iloc[60], end=sessions.iloc[140], block_sessions=20
    )
    folds = make_expanding_folds(panel, windows, holdout_start=sessions.iloc[140])
    return panel, folds


def _describe(panel, folds):
    return describe_folds(
        panel,
        folds,
        feature_columns=["feature_a", "feature_b"],
        availability_columns=["available_at"],
    )


def test_fold_metadata_is_deterministic_and_json_safe(folded):
    panel, folds = folded
    first, second = _describe(panel, folds), _describe(panel, folds)

    pd.testing.assert_frame_equal(first, second)
    records = json.loads(first.to_json(orient="records"))
    assert records == json.loads(json.dumps(first.to_dict(orient="records")))


def test_fold_metadata_records_cutoffs_and_availability(folded):
    panel, folds = folded
    meta = _describe(panel, folds)

    for row, fold in zip(meta.itertuples(), folds, strict=True):
        assert row.train_max_target_end_date < row.test_start
        assert row.train_last_date < row.test_start
        assert row.score_available_at >= row.test_last_date
        assert row.embargo_sessions == 0
        train = panel.iloc[fold.train_rows]
        assert row.train_max_available_at == train["available_at"].max().isoformat()
        assert sum(json.loads(row.train_rows_by_symbol).values()) == row.n_train


def test_training_set_can_be_reconstructed_from_metadata_alone(folded):
    panel, folds = folded
    meta = _describe(panel, folds)

    for row in meta.itertuples():
        cutoff = pd.Timestamp(row.knowledge_cutoff)
        rebuilt = panel[panel["target_return"].notna() & (panel["target_end_date"] < cutoff)]
        assert frame_keys_sha256(rebuilt) == row.train_keys_sha256
        content = [
            "feature_a",
            "feature_b",
            "target_return",
            "target_start_date",
            "target_end_date",
            "available_at",
        ]
        assert frame_content_sha256(rebuilt, content) == row.train_content_sha256


def test_changing_a_test_value_leaves_training_fingerprints_unchanged(folded):
    panel, folds = folded
    base = _describe(panel, folds)

    changed = panel.copy()
    changed.loc[folds[-1].test_rows[0], "target_return"] += 1.0
    moved = _describe(changed, folds)

    assert (moved["train_content_sha256"].iloc[-1]) == base["train_content_sha256"].iloc[-1]
    assert moved["test_content_sha256"].iloc[-1] != base["test_content_sha256"].iloc[-1]
    assert moved["frame_content_sha256"].iloc[0] != base["frame_content_sha256"].iloc[0]


def test_run_manifest_records_label_timing_inputs_and_versions(tmp_path):
    data = tmp_path / "prices.parquet"
    data.write_bytes(b"example bytes")

    manifest = run_manifest(
        label_spec=MODEL_LABEL_SPEC,
        holdout_start=_ts("2025-01-01"),
        universe=["NVDA", "MU"],
        excluded_symbols=["SP500"],
        feature_columns=["momentum_20d"],
        input_files={"prices": data},
        git_sha="abc123",
    )

    assert manifest["label_spec"] == {
        "horizon": 20,
        "entry_lag": 1,
        "price_column": "adjusted_close",
    }
    assert manifest["holdout_start"] == "2025-01-01T00:00:00+00:00"
    assert manifest["universe"] == ["MU", "NVDA"]
    assert manifest["input_files_sha256"]["prices"] == hashlib.sha256(b"example bytes").hexdigest()
    assert set(manifest["versions"]) == {"python", "numpy", "pandas"}
    json.dumps(manifest)


def test_fold_metadata_refuses_folds_built_for_another_row_order(folded):
    panel, folds = folded
    reordered = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(LeakageError, match="different frame"):
        describe_folds(reordered, folds, feature_columns=["feature_a", "feature_b"])
