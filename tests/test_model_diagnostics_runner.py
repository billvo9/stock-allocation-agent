"""
End-to-end development runs on a small synthetic panel: the lockbox gate,
truncation before labeling, the prediction-then-scoring order, determinism,
label-free prediction hashes, and the guarantee that diagnostics never
feed back into predictions.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec
from stock_agent.model_diagnostics import contract, runner, scoring
from stock_agent.model_diagnostics.controls import FeatureScore, add_control_columns
from stock_agent.model_diagnostics.record import sha256_json
from stock_agent.model_validation.checks import LeakageError
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows
from stock_agent.model_validation.harness import run_walk_forward

UNIVERSE = ("A", "B", "C", "D")
SPEC = LabelSpec(horizon=10, entry_lag=1)
FEATURES = ("feature_a", "momentum_20d")
CONFIG = runner.DiagnosticsConfig(
    hac_lag=5,
    block_length=21,
    bootstrap_reps=29,
    acf_max_lag=5,
    permutation_draws=2,
    permutation_block_length=21,
    stale_lags=(60,),
    calibration_reps=40,
    calibration_bootstrap_reps=9,
)
STARTED = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
PROVENANCE = {"code": {"git_sha": "abc", "git_dirty": False}, "environment": {"python": "3.12"}}


def _raw_panel(factory, *, late_symbol=False):
    labeled = factory(symbols=UNIVERSE, sessions=330, horizon=SPEC.horizon, seed=5)
    raw = labeled.drop(columns=["target_return", "target_start_date", "target_end_date"])
    raw["momentum_20d"] = raw.groupby("symbol")["adjusted_close"].pct_change(20)
    raw = raw.dropna(subset=["momentum_20d"])
    if late_symbol:
        late = raw[raw["symbol"] == "A"].tail(20).assign(symbol="Z")
        raw = pd.concat([raw, late])
    return raw.sort_values(["date", "symbol"]).reset_index(drop=True)


def _lockbox(raw):
    return raw["date"].drop_duplicates().sort_values().iloc[280]


def _folds(labeled, lockbox):
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(sessions, start=sessions.iloc[100], end=lockbox, block_sessions=40)
    return make_expanding_folds(labeled, windows, holdout_start=lockbox, lockbox_start=lockbox)


def _run(labeled, folds, lockbox, **overrides):
    options = {
        "universe": UNIVERSE,
        "label_spec": SPEC,
        "diagnostic_features": FEATURES,
        "provenance": PROVENANCE,
        "started_at": STARTED,
        "models": runner.default_models(),
        "config": CONFIG,
        "lockbox_start": lockbox,
        **overrides,
    }
    return runner.run_diagnostics(labeled, folds, **options)


@pytest.fixture(scope="module")
def development(panel_factory):
    raw = _raw_panel(panel_factory)
    lockbox = _lockbox(raw)
    labeled, _ = runner.prepare_development_frame(
        raw, symbols=UNIVERSE, label_spec=SPEC, holdout_start=lockbox
    )
    folds = _folds(labeled, lockbox)
    return raw, labeled, folds, lockbox, _run(labeled, folds, lockbox)


# --- truncation and the lockbox gate ---


def test_prepare_truncates_before_labeling_and_excludes_symbols_without_history(panel_factory):
    raw = _raw_panel(panel_factory, late_symbol=False)
    lockbox = _lockbox(raw)
    late = raw[raw["date"] >= lockbox].assign(symbol="SNDK")
    with_late = pd.concat([raw, late]).sort_values(["date", "symbol"]).reset_index(drop=True)
    labeled, facts = runner.prepare_development_frame(
        with_late, symbols=[*UNIVERSE, "SNDK"], label_spec=SPEC, holdout_start=lockbox
    )
    assert facts["excluded_symbols"] == {"SNDK": "no_pre_lockbox_rows"}
    assert labeled["date"].max() < lockbox
    assert facts["rows_unlabeled_after_truncation"] == len(UNIVERSE) * (
        SPEC.horizon + SPEC.entry_lag
    )
    full = SPEC.apply(raw).sort_values(["date", "symbol"]).reset_index(drop=True)
    full = full[full["date"] < lockbox].reset_index(drop=True)
    pd.testing.assert_frame_equal(labeled[list(FEATURES)], full[list(FEATURES)])
    both = labeled["target_return"].notna()
    np.testing.assert_array_equal(
        labeled.loc[both, "target_return"], full.loc[both, "target_return"]
    )
    # Labels built before truncation reach into the lockbox; these are now absent.
    assert (full.loc[~both, "target_end_date"] >= lockbox).all()


def test_frames_with_lockbox_rows_or_lockbox_labels_are_refused(development):
    raw, _, _, lockbox, _ = development
    untruncated = SPEC.apply(raw).sort_values(["date", "symbol"]).reset_index(drop=True)
    with pytest.raises(LeakageError):
        _run(untruncated, _folds(untruncated, lockbox), lockbox)
    labeled_first = untruncated[untruncated["date"] < lockbox].reset_index(drop=True)
    folds_first = _folds(labeled_first, lockbox)
    assert any(len(fold.held_out_rows) for fold in folds_first)
    with pytest.raises(LeakageError, match="truncate the panel before labeling"):
        _run(labeled_first, folds_first, lockbox)
    without_held_out = [
        dataclasses.replace(fold, held_out_rows=fold.held_out_rows[:0]) for fold in folds_first
    ]
    with pytest.raises(LeakageError, match="label after truncating"):
        _run(labeled_first, without_held_out, lockbox)


def test_unlabeled_lockbox_rows_are_refused_even_outside_every_fold(development):
    # Architect review: a frame must not even hold lockbox rows, labeled or
    # not, because whole-frame code (placebos, drift) could read them.
    raw, labeled, _, lockbox, _ = development
    tail = (
        raw[raw["date"] >= lockbox]
        .head(8)
        .assign(target_return=np.nan, target_start_date=pd.NaT, target_end_date=pd.NaT)
    )
    extended = pd.concat([labeled, tail], ignore_index=True)
    for column in ("target_start_date", "target_end_date"):
        extended[column] = pd.to_datetime(extended[column], utc=True)
    folds = _folds(extended, lockbox)
    assert not any(len(fold.held_out_rows) for fold in folds)
    with pytest.raises(LeakageError, match="rows dated in the lockbox"):
        _run(extended, folds, lockbox)


def test_holdout_values_cannot_change_any_development_output(development):
    raw, _, _, lockbox, clean = development
    poisoned = raw.copy()
    after = poisoned["date"] >= lockbox
    poisoned.loc[after, ["adjusted_close", "feature_a", "feature_b", "momentum_20d"]] = 1e9
    labeled, _ = runner.prepare_development_frame(
        poisoned, symbols=UNIVERSE, label_spec=SPEC, holdout_start=lockbox
    )
    run = _run(labeled, _folds(labeled, lockbox), lockbox)
    assert run.record["outputs"] == clean.record["outputs"]
    assert run.record["lockbox_evidence"]["holdout_values_used"] is False


def test_lockbox_evidence_is_derived_from_the_data(development):
    *_, lockbox, run = development
    evidence = run.record["lockbox_evidence"]
    assert evidence["holdout_values_used"] is False
    assert evidence["held_out_rows_total"] == 0
    assert pd.Timestamp(evidence["max_target_end_date"]) < lockbox


# --- stage order and completeness ---

EVENTS: list[str] = []


class LoggingScore(FeatureScore):
    def predict(self, X):
        EVENTS.append("predict")
        return super().predict(X)


def test_scorers_receive_labels_only_after_every_prediction_is_complete(development, monkeypatch):
    # Every prediction path is logged: main models, permutation draws, the
    # stale control (all through the harness), the unsafe canary reference,
    # and per-date predict calls of a spy model.
    _, labeled, folds, lockbox, _ = development
    for name, module in (
        ("score_model", scoring),
        ("run_walk_forward", runner),
        ("unpurged_reference_predictions", runner),
    ):
        original = getattr(module, name)

        def logged(*args, _original=original, _name=name, **kwargs):
            EVENTS.append("score" if _name == "score_model" else "predict")
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, logged)
    EVENTS.clear()
    spy = runner.ModelSpec("spy", "control", LoggingScore, ("feature_a",), output_kind="score")
    claimed = {**PROVENANCE, "truncation": {"holdout_values_used": True}}  # ignored
    run = _run(labeled, folds, lockbox, models=[*runner.default_models(), spy], provenance=claimed)
    n_harness = len(runner.default_models()) + 1 + 2 * CONFIG.permutation_draws + 1
    assert EVENTS.count("predict") >= n_harness + 1  # + unsafe reference + spy batches
    assert max(i for i, e in enumerate(EVENTS) if e == "predict") < EVENTS.index("score")
    assert run.record["lockbox_evidence"]["holdout_values_used"] is False


def test_runs_refuse_predictions_that_omit_a_scored_row(development, monkeypatch):
    _, labeled, folds, lockbox, _ = development

    def dropping(*args, **kwargs):
        result = run_walk_forward(*args, **kwargs)
        worst = (result.predictions["target_return"] - result.predictions["prediction"]).abs()
        trimmed = result.predictions.drop(index=worst.idxmax()).reset_index(drop=True)
        return dataclasses.replace(result, predictions=trimmed)

    monkeypatch.setattr(runner, "run_walk_forward", dropping)
    with pytest.raises(ValueError, match="exactly the folds' test rows"):
        _run(labeled, folds, lockbox)


def test_every_model_is_scored_on_the_same_rows(development):
    _, _, folds, _, run = development
    predictions = run.tables["predictions"]
    expected = sum(len(fold.test_rows) for fold in folds)
    keys = predictions.groupby("model").apply(
        lambda frame: tuple(map(tuple, frame[["fold_id", "date", "symbol"]].astype(str).to_numpy()))
    )
    assert keys.nunique() == 1
    assert (predictions.groupby("model").size() == expected).all()
    assert set(predictions["role"]) == {"null", "control", "canary", "canary_unsafe_reference"}


# --- reproducibility and record ---


def test_identical_inputs_give_an_identical_run_under_a_new_run_id(development):
    _, labeled, folds, lockbox, run = development
    assert run.run_id == f"20261009T120000Z-{run.record['spec_sha256'][:12]}"
    again = _run(labeled, folds, lockbox, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    assert again.run_id == f"20261010T000000Z-{run.record['spec_sha256'][:12]}"
    assert again.record["spec"] == run.record["spec"]
    assert again.record["outputs"] == run.record["outputs"]
    for name, table in run.tables.items():
        pd.testing.assert_frame_equal(table, again.tables[name])


def test_the_spec_hash_covers_code_state_and_settings(development):
    *_, run = development
    spec = run.record["spec"]
    assert sha256_json(spec) == run.record["spec_sha256"]
    for change in (
        {"code": {"git_sha": "def", "git_dirty": True}},
        {"config": {**spec["config"], "hac_lag": 40}},
        {"environment": {"python": "3.13"}},
    ):
        assert sha256_json({**spec, **change}) != run.record["spec_sha256"]


def test_record_lists_what_the_owner_asked_to_reproduce(development):
    _, _, folds, _, run = development
    spec = run.record["spec"]
    assert (
        spec["label"]["label_id"] == "forward_simple_return:adjusted_close:horizon=10:entry_lag=1"
    )
    assert spec["label"]["target_column"] == "target_return"
    assert spec["universe"]["symbol_codes"] == {"A": 0, "B": 1, "C": 2, "D": 3}
    assert spec["features"]["diagnostic_features"] == list(FEATURES)
    assert spec["folds"]["n_folds"] == len(folds)
    assert spec["config"]["hac_lag"] == 5 and spec["config"]["block_length"] == 21
    assert {m["name"] for m in spec["models"]} == {m.name for m in runner.default_models()}
    assert len(spec["registered_variants"]) == len(spec["models"])
    for name in ("zero", "momentum_20d"):
        assert run.record["outputs"]["predictions_sha256_label_free"][name]
    assert run.record["inference_calibration"]["rates"].keys() == {
        "iid",
        "momentum",
        "block_constant",
    }


def test_prediction_hashes_ignore_labels(development):
    _, labeled, folds, lockbox, _ = development
    models = [m for m in runner.default_models() if m.name in ("zero", "momentum_20d")]
    first = _run(labeled, folds, lockbox, models=models)
    flipped = labeled.assign(target_return=-labeled["target_return"])
    second = _run(flipped, folds, lockbox, models=models)
    hashes = "predictions_sha256_label_free"
    assert first.record["outputs"][hashes] == second.record["outputs"][hashes]
    assert first.record["outputs"]["labels_sha256"] != second.record["outputs"]["labels_sha256"]


def test_diagnostics_never_feed_back_into_predictions(development):
    _, labeled, folds, lockbox, run = development
    narrower = _run(labeled, folds, lockbox, diagnostic_features=("feature_b",))
    pd.testing.assert_frame_equal(run.tables["predictions"], narrower.tables["predictions"])
    assert narrower.record["spec"]["models"] == run.record["spec"]["models"]
    frame = add_control_columns(labeled, UNIVERSE)
    direct = run_walk_forward(
        frame,
        folds,
        feature_columns=["momentum_20d"],
        make_estimator=FeatureScore,
        lockbox_start=lockbox,
    ).predictions
    saved = run.tables["predictions"].query("model == 'momentum_20d'")
    np.testing.assert_array_equal(saved["prediction"], direct["prediction"])


def test_candidate_models_may_not_use_control_columns():
    with pytest.raises(ValueError, match="control_"):
        runner.ModelSpec("ridge", "candidate", FeatureScore, ("control_symbol_code",))
    with pytest.raises(ValueError, match="role"):
        runner.ModelSpec("x", "winner", FeatureScore, ("feature_a",))


# --- tables and checks ---


def test_every_table_matches_the_saved_output_contract(development):
    *_, run = development
    assert set(run.tables) == set(contract.TABLES)
    for name, table in run.tables.items():
        contract.validate(name, table)
    assert set(run.tables["metrics"]["scope"]) == {"evaluation"}
    assert set(run.tables["feature_stats"]["scope"]) == {"training", "evaluation"}


def test_checks_report_constant_nulls_and_a_closed_lockbox(development):
    *_, run = development
    lockbox = run.tables["checks"].query("check == 'lockbox_closed'")
    assert lockbox["passed"].tolist() == [True]
    undefined = run.tables["checks"].query("check == 'null_rank_ic_undefined'")
    assert set(undefined["model"]) == {"zero", "pooled_mean"}
    complete = run.tables["checks"].query("check == 'predictions_complete'")
    assert complete["passed"].all()
    summary = run.record["checks_summary"]
    assert summary["leakage_checks_failed"] == 0


def test_null_comparisons_and_timeliness_controls_are_recorded(development):
    *_, run = development
    draws = run.tables["null_draws"]
    permutation = draws.query("null_kind == 'block_permutation'")
    assert set(permutation["model"]) == {"per_symbol_mean", "momentum_20d"}
    assert permutation.groupby("model")["draw"].nunique().tolist() == [2, 2]
    assert permutation["seed"].nunique() == 1
    assert len(draws.query("null_kind == 'static_ordering'")) == 24
    metrics = set(run.tables["metrics"]["metric"])
    assert {"mean_rank_ic_vs_block_permutation", "rank_ic_minus_stale_60"} <= metrics


def test_saved_null_predictions_equal_purged_training_oracles(development):
    _, labeled, folds, _, run = development
    saved = run.tables["predictions"]
    for fold in folds:
        train = labeled.iloc[fold.train_rows]
        assert (train["target_end_date"] < fold.test_start).all()
        pooled = saved.query("model == 'pooled_mean' and fold_id == @fold.fold_id")
        np.testing.assert_allclose(pooled["prediction"], train["target_return"].mean())
        by_symbol = train.groupby("symbol")["target_return"].mean()
        per_symbol = saved.query("model == 'per_symbol_mean' and fold_id == @fold.fold_id")
        np.testing.assert_allclose(per_symbol["prediction"], per_symbol["symbol"].map(by_symbol))


def test_stale_comparison_is_model_minus_stale(development):
    *_, run = development
    metrics = run.tables["metrics"].query("model == 'momentum_20d' and fold_id.isna()")

    def value(metric, method):
        row = metrics.query("metric == @metric and inference_method == @method")
        return float(row["estimate"].iloc[0])

    difference = value("rank_ic_minus_stale_60", "newey_west")
    model = value("mean_rank_ic", "newey_west")
    stale = value("mean_rank_ic_stale_60", "point")
    assert difference == pytest.approx(model - stale)


def test_a_stale_lag_longer_than_the_history_is_unavailable_not_dropped(development):
    _, labeled, folds, lockbox, _ = development
    models = [m for m in runner.default_models() if m.name == "momentum_20d"]
    config = dataclasses.replace(CONFIG, stale_lags=(5_000,), permutation_models=())
    run = _run(labeled, folds, lockbox, models=models, config=config)
    rows = run.tables["metrics"].query("metric == 'rank_ic_minus_stale_5000'")
    assert set(rows["reason"]) == {"insufficient_history"}


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (
            lambda f: [dataclasses.replace(f[0], mode="holdout"), *f[1:]],
            LeakageError,
            "development",
        ),
        (
            lambda f: [
                dataclasses.replace(f[0], holdout_start=f[0].holdout_start + pd.Timedelta("1D")),
                *f[1:],
            ],
            LeakageError,
            "after the lockbox",
        ),
    ],
    ids=["holdout_mode_fold", "holdout_start_after_lockbox"],
)
def test_development_runs_refuse_holdout_folds(development, change, error, match):
    _, labeled, folds, lockbox, _ = development
    with pytest.raises(error, match=match):
        _run(labeled, change(list(folds)), lockbox)


def test_run_inputs_are_validated(development):
    _, labeled, folds, lockbox, _ = development
    zero = runner.default_models()[0]
    with pytest.raises(ValueError, match="unique"):
        _run(labeled, folds, lockbox, models=[zero, zero])
    with pytest.raises(ValueError, match="provenance"):
        _run(labeled, folds, lockbox, provenance={"code": {}})
    with pytest.raises(ValueError, match="control columns"):
        _run(add_control_columns(labeled, UNIVERSE), folds, lockbox)
    reordered = labeled.iloc[::-1].reset_index(drop=True)
    with pytest.raises(LeakageError, match="different frame"):
        _run(reordered, folds, lockbox)


def test_the_canary_positive_control_fires_end_to_end(panel_factory):
    # With purging removed the memorizer copies labels that overlap the test
    # window; the run's rank-IC checks must see it, while the purged
    # memorizer stays quiet. Short windows (15 sessions, horizon 10) put the
    # overlap in a large share of each window.
    names = tuple(f"S{i:02d}" for i in range(12))
    raw = panel_factory(symbols=names, sessions=260, horizon=10, seed=1)
    raw = raw.drop(columns=["target_return", "target_start_date", "target_end_date"])
    lockbox = raw["date"].drop_duplicates().sort_values().iloc[255]
    labeled, _ = runner.prepare_development_frame(
        raw, symbols=names, label_spec=SPEC, holdout_start=lockbox
    )
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(sessions, start=sessions.iloc[60], end=lockbox, block_sessions=15)
    folds = make_expanding_folds(labeled, windows, holdout_start=lockbox, lockbox_start=lockbox)
    memorizer = [m for m in runner.default_models() if m.name == "memorizer"]
    config = dataclasses.replace(
        CONFIG,
        block_length=15,
        permutation_models=(),
        stale_models=(),
        calibration_signals=("iid",),
    )
    run = _run(
        labeled,
        folds,
        lockbox,
        universe=names,
        models=memorizer,
        config=config,
        diagnostic_features=("feature_a",),
    )
    checks = run.tables["checks"].set_index("check")
    assert bool(checks.loc["canary_detects_unpurged_leakage", "passed"])
    assert bool(checks.loc["canary_shows_no_skill_through_harness", "passed"])


# --- reporting contract (schema 1.1) ---


def _check_rows(rows):
    frame = pd.DataFrame(rows, columns=["check", "severity", "passed"])
    frame["passed"] = frame["passed"].astype("boolean")
    return frame


def test_the_checks_summary_counts_undetermined_checks_in_every_family():
    # Regression: the 1.0 summary had no place for checks without a stored
    # result, so an undetermined leakage check was invisible in it.
    checks = _check_rows(
        [
            ("lockbox_closed", "leakage", True),
            ("canary", "leakage", False),
            ("predictions_complete", "leakage", pd.NA),
            ("size", "instrument", pd.NA),
            ("control_apparent_skill", "research_warning", True),
            ("control_apparent_skill", "research_warning", False),
            ("null_rank_ic_undefined", "research_warning", pd.NA),
            ("fpr", "info", False),
        ]
    )
    summary = runner.summarize_checks(checks)
    assert summary["leakage_checks_failed"] == 1  # legacy keys keep their 1.0 meaning
    assert summary["instrument_checks_failed"] == 0
    assert summary["research_warnings"] == 1
    counts = {
        family: {key: value for key, value in entry.items() if key != "passed_means"}
        for family, entry in summary["families"].items()
    }
    assert counts == {
        "info": {"total": 1, "passed": 0, "failed": 1, "undetermined": 0},
        "instrument": {"total": 1, "passed": 0, "failed": 0, "undetermined": 1},
        "leakage": {"total": 3, "passed": 1, "failed": 1, "undetermined": 1},
        "research_warning": {"total": 3, "passed": 1, "failed": 1, "undetermined": 1},
    }
    assert summary["families"]["research_warning"]["passed_means"] == "warning active"
    assert summary["families"]["leakage"]["passed_means"] == "check held"


def test_family_counts_always_add_up_to_the_family_total():
    rng = np.random.default_rng(3)
    severities = np.array(["leakage", "instrument", "research_warning", "info"])
    for _ in range(200):
        size = int(rng.integers(0, 25))
        passed = pd.array(rng.choice([True, False, None], size=size), dtype="boolean")
        checks = pd.DataFrame(
            {
                "check": [f"c{i}" for i in range(size)],
                "severity": rng.choice(severities, size=size),
                "passed": passed,
            }
        )
        families = runner.summarize_checks(checks)["families"]
        assert set(families) == set(checks["severity"])
        for family, entry in families.items():
            assert entry["passed"] + entry["failed"] + entry["undetermined"] == entry["total"]
            assert entry["total"] == int((checks["severity"] == family).sum())
        assert sum(entry["total"] for entry in families.values()) == size


def test_the_record_states_its_decision_method_and_candidate_eligibility(development):
    *_, run = development
    spec = run.record["spec"]
    assert spec["output_schema_version"] == contract.OUTPUT_SCHEMA_VERSION == "1.1"
    assert spec["decision_method"] == runner.DECISION_METHOD == "fold_block_t"
    for model in spec["models"]:
        assert model["eligible_as_candidate"] is (model["role"] == contract.CANDIDATE_ROLE)
    (reference,) = spec["reference_predictors"]
    canary = next(m for m in spec["models"] if m["name"] == CONFIG.canary_model)
    assert reference["role"] == contract.UNSAFE_REFERENCE_ROLE
    assert reference["eligible_as_candidate"] is False
    assert reference["output_kind"] == canary["output_kind"]
    assert reference["reference_for"] == canary["name"]
    assert reference["purged"] is False and reference["through_harness"] is False
    stored = run.tables["predictions"]
    assert set(stored.loc[stored["role"] == reference["role"], "model"]) == {reference["name"]}
    summary = run.record["checks_summary"]["families"]
    checks = run.tables["checks"]
    for family, entry in summary.items():
        rows = checks[checks["severity"] == family]["passed"]
        assert entry["total"] == len(rows)
        assert entry["undetermined"] == int(rows.isna().sum())
        assert entry["passed"] + entry["failed"] + entry["undetermined"] == entry["total"]


def test_eligibility_is_not_part_of_the_variant_identity():
    spec = next(m for m in runner.default_models() if m.role == "null")
    description = spec.describe()
    without = {k: v for k, v in description.items() if k != "eligible_as_candidate"}
    assert runner.variant_id(without) == description["variant_id"]


def test_curve_intervals_record_the_method_level_and_lag_they_used(development):
    *_, run = development
    curves = run.tables["curves"]
    with_interval = curves["curve"].isin(
        [
            "rank_position_realized",
            "rank_position_realized_minus_date_mean",
            "pooled_decile_realized",
        ]
    )
    recorded = curves[with_interval]
    assert len(recorded)
    assert (recorded["ci_method"] == scoring.CURVE_CI_METHOD).all()
    assert (recorded["ci_level"] == CONFIG.ci_level).all()
    assert (recorded["hac_lag"] == CONFIG.hac_lag).all()
    others = curves[~with_interval]
    assert others["ci_method"].isna().all() and others["hac_lag"].isna().all()
    assert others["ci_level"].isna().all()
    assert others["ci_low"].isna().all()  # no interval is ever stored without its method
