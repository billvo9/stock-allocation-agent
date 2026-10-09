"""
Null models, controls and the canary run through the unchanged harness.

Oracles recompute each fitted value from the fold's training rows by hand;
negative tests poison every row outside them (purge band, test rows) and
require bit-identical predictions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics import controls as c
from stock_agent.model_diagnostics.canary import unpurged_reference_predictions
from stock_agent.model_diagnostics.runner import default_models
from stock_agent.model_validation.checks import LeakageError, assert_prefix_stable
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows
from stock_agent.model_validation.harness import run_walk_forward

UNIVERSE = ("A", "B", "C", "D")
IDENTITY = [c.CONTROL_SYMBOL_CODE]
KEYS = [c.CONTROL_SYMBOL_CODE, c.CONTROL_SESSION_INDEX]


def _setup(labeled_panel, **kwargs):
    options = {"symbols": UNIVERSE, "sessions": 260, "horizon": 10, "seed": 0, **kwargs}
    panel = c.add_control_columns(labeled_panel(**options), UNIVERSE)
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    holdout = sessions.iloc[-1] + pd.offsets.BDay(1)
    windows = make_test_windows(sessions, start=sessions.iloc[80], end=holdout, block_sessions=40)
    folds = make_expanding_folds(panel, windows, holdout_start=holdout, lockbox_start=holdout)
    return panel, folds, holdout


def _run(panel, folds, holdout, features, make):
    return run_walk_forward(
        panel, folds, feature_columns=features, make_estimator=make, lockbox_start=holdout
    )


def _poison_outside_training(panel, fold):
    poisoned = panel.copy()
    outside = np.setdiff1d(np.arange(len(panel)), fold.train_rows)
    labels = poisoned["target_return"].to_numpy(copy=True)
    labels[outside] = np.where(np.isfinite(labels[outside]), 10.0, labels[outside])
    poisoned["target_return"] = labels
    assert not np.array_equal(poisoned["target_return"], panel["target_return"], equal_nan=True)
    return poisoned


# --- control columns ---


def test_control_columns_use_the_sorted_universe_and_dense_sessions(labeled_panel):
    raw = labeled_panel(symbols=("B", "A"), sessions=30, horizon=3)
    frame = c.add_control_columns(raw, ["B", "A", "C"])
    assert c.symbol_codes(["B", "A", "C"]) == {"A": 0, "B": 1, "C": 2}
    np.testing.assert_array_equal(frame[c.CONTROL_SYMBOL_CODE], raw["symbol"].map({"A": 0, "B": 1}))
    sessions = np.unique(frame["date"])
    assert frame[c.CONTROL_SESSION_INDEX].tolist() == [
        int(np.searchsorted(sessions, d)) for d in frame["date"]
    ]
    assert list(raw.columns) == list(frame.columns[: len(raw.columns)])  # input unchanged


def test_control_columns_are_prefix_stable_and_label_blind(labeled_panel):
    raw = labeled_panel(symbols=UNIVERSE, sessions=60, horizon=5, drop_fraction=0.1)
    cutoffs = [raw["date"].iloc[len(raw) // 3], raw["date"].iloc[2 * len(raw) // 3]]
    assert_prefix_stable(
        lambda frame: c.add_control_columns(frame, UNIVERSE),
        raw,
        cutoffs=cutoffs,
        value_columns=list(c.CONTROL_COLUMNS),
    )
    relabeled = raw.assign(target_return=raw["target_return"] * -3.0)
    pd.testing.assert_frame_equal(
        c.add_control_columns(raw, UNIVERSE)[list(c.CONTROL_COLUMNS)],
        c.add_control_columns(relabeled, UNIVERSE)[list(c.CONTROL_COLUMNS)],
    )


def test_control_columns_reject_unknown_symbols_and_duplicates(labeled_panel):
    raw = labeled_panel(symbols=("A", "Z"), sessions=20, horizon=3)
    with pytest.raises(ValueError, match="outside the universe"):
        c.add_control_columns(raw, ["A"])
    with pytest.raises(ValueError, match="already has control"):
        c.add_control_columns(c.add_control_columns(raw, ["A", "Z"]), ["A", "Z"])


# --- nulls and controls through the harness ---


def test_zero_forecast_is_exactly_zero(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    result = _run(panel, folds, holdout, IDENTITY, c.ZeroForecast)
    assert (result.predictions["prediction"] == 0.0).all()


def test_pooled_mean_equals_the_purged_training_mean(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    result = _run(panel, folds, holdout, IDENTITY, c.PooledMeanForecast)
    for fold in folds:
        oracle = panel.loc[fold.train_rows, "target_return"].mean()
        scored = result.predictions[result.predictions["fold_id"] == fold.fold_id]
        np.testing.assert_allclose(scored["prediction"], oracle)


def test_per_symbol_mean_matches_a_purged_oracle(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    result = _run(panel, folds, holdout, IDENTITY, lambda: c.PerSymbolMeanForecast(min_history=5))
    for fold in folds:
        purged = (panel["target_end_date"] < fold.test_start) & panel["target_return"].notna()
        oracle = panel[purged].groupby("symbol")["target_return"].mean()
        scored = result.predictions[result.predictions["fold_id"] == fold.fold_id]
        np.testing.assert_allclose(scored["prediction"], scored["symbol"].map(oracle))


@pytest.mark.parametrize(
    ("features", "make"),
    [
        (IDENTITY, c.PooledMeanForecast),
        (IDENTITY, lambda: c.PerSymbolMeanForecast(min_history=5)),
        (KEYS, c.NearestKeyMemorizer),
    ],
    ids=["pooled_mean", "per_symbol_mean", "memorizer"],
)
def test_fitted_state_ignores_every_label_outside_the_training_rows(labeled_panel, features, make):
    panel, folds, holdout = _setup(labeled_panel)
    clean = _run(panel, folds, holdout, features, make).predictions
    for fold in folds:
        # Labels in the purge band, the test window and later all become +10.
        poisoned = _poison_outside_training(panel, fold)
        scored = _run(poisoned, [fold], holdout, features, make).predictions
        np.testing.assert_array_equal(
            scored["prediction"],
            clean.loc[clean["fold_id"] == fold.fold_id, "prediction"],
        )


@pytest.mark.parametrize("spec", default_models(score_feature="feature_a"), ids=lambda s: s.name)
def test_every_null_and_control_refuses_an_unpurged_fold(labeled_panel, spec):
    panel, folds, holdout = _setup(labeled_panel)
    fold = folds[-1]
    leaky = dataclasses.replace(
        fold,
        train_rows=np.flatnonzero(
            panel["target_return"].notna() & (panel["date"] < fold.test_start)
        ),
    )

    def make():
        params = dict(spec.params)
        if spec.seeded:
            params["seed"] = 0
        return spec.estimator(**params)

    with pytest.raises(LeakageError):
        _run(panel, [leaky], holdout, list(spec.feature_columns), make)


def test_unseen_or_short_history_symbols_fall_back_to_the_pooled_mean(labeled_panel):
    # D lists late: its first training labels mature after the first folds start.
    panel, folds, holdout = _setup(labeled_panel, listing={"D": (150, 260)})
    result = _run(panel, folds, holdout, IDENTITY, lambda: c.PerSymbolMeanForecast(min_history=30))
    first = result.estimators[0]
    assert 3 not in first.symbol_means_  # D has no usable history yet
    assert first.fallback_predictions_ == int(
        (result.predictions.query("fold_id == 0")["symbol"] == "D").sum()
    )
    d_rows = result.predictions.query("fold_id == 0 and symbol == 'D'")
    np.testing.assert_allclose(d_rows["prediction"], first.pooled_mean_)


def test_per_symbol_mean_rejects_non_integer_codes():
    with pytest.raises(ValueError, match="finite integers"):
        c.PerSymbolMeanForecast().fit(np.array([[0.5], [1.0]]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="positive integer"):
        c.PerSymbolMeanForecast(min_history=0)


def test_feature_score_is_the_signed_feature_and_ignores_labels(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    plus = _run(panel, folds, holdout, ["feature_a"], lambda: c.FeatureScore(sign=1)).predictions
    relabeled = panel.assign(target_return=panel["target_return"] * 7.0)
    again = _run(relabeled, folds, holdout, ["feature_a"], c.FeatureScore).predictions
    rows = np.concatenate([fold.test_rows for fold in folds])
    np.testing.assert_array_equal(plus["prediction"], panel.loc[rows, "feature_a"])
    np.testing.assert_array_equal(plus["prediction"], again["prediction"])
    minus = _run(panel, folds, holdout, ["feature_a"], lambda: c.FeatureScore(sign=-1))
    np.testing.assert_array_equal(minus.predictions["prediction"], -plus["prediction"])
    with pytest.raises(ValueError, match="sign"):
        c.FeatureScore(sign=2)


def test_a_missing_score_feature_is_refused_not_filled(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    holed = panel.copy()
    holed.loc[folds[0].test_rows[0], "feature_a"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(holed, folds, holdout, ["feature_a"], c.FeatureScore)


def test_random_noise_is_seeded_per_fold(labeled_panel):
    panel, folds, holdout = _setup(labeled_panel)
    seeds = iter(range(100))
    first = _run(panel, folds, holdout, IDENTITY, lambda: c.RandomNoiseScore(seed=next(seeds)))
    seeds = iter(range(100))
    again = _run(panel, folds, holdout, IDENTITY, lambda: c.RandomNoiseScore(seed=next(seeds)))
    np.testing.assert_array_equal(first.predictions["prediction"], again.predictions["prediction"])
    by_fold = first.predictions.groupby("fold_id")["prediction"].first()
    assert by_fold.nunique() == len(folds)
    with pytest.raises(ValueError):
        c.RandomNoiseScore(seed=-1)


# --- canary ---


def test_memorizer_copies_the_nearest_label_of_the_same_symbol():
    X = np.array([[0, 1], [0, 5], [1, 2], [1, 9]], dtype=float)
    y = np.array([0.1, 0.5, 0.2, 0.9])
    model = c.NearestKeyMemorizer().fit(X, y)
    test = np.array([[0, 3], [0, 100], [1, 6], [1, 5], [2, 1]], dtype=float)
    # (0,3) ties between sessions 1 and 5 -> earlier; (1,6) is nearer 9 than 2;
    # symbol 2 is unseen -> pooled training mean
    np.testing.assert_allclose(model.predict(test), [0.1, 0.5, 0.9, 0.2, y.mean()])
    assert model.fallback_predictions_ == 1


def test_memorizer_has_no_skill_purged_and_spurious_skill_unpurged(labeled_panel):
    symbols = tuple(f"S{i:02d}" for i in range(30))
    panel = c.add_control_columns(
        labeled_panel(symbols=symbols, sessions=260, horizon=10, seed=1), symbols
    )
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    holdout = sessions.iloc[-1] + pd.offsets.BDay(1)
    windows = make_test_windows(
        sessions, start=sessions.iloc[60], end=sessions.iloc[240], block_sessions=5
    )
    folds = make_expanding_folds(panel, windows, holdout_start=holdout, lockbox_start=holdout)

    safe = _run(panel, folds, holdout, KEYS, c.NearestKeyMemorizer).predictions
    unsafe = unpurged_reference_predictions(
        panel,
        folds,
        feature_columns=KEYS,
        make_estimator=c.NearestKeyMemorizer,
        lockbox_start=holdout,
    )
    pd.testing.assert_frame_equal(
        safe.drop(columns="prediction"), unsafe.drop(columns="prediction")
    )
    assert abs(np.corrcoef(safe["prediction"], safe["target_return"])[0, 1]) < 0.15
    assert np.corrcoef(unsafe["prediction"], unsafe["target_return"])[0, 1] > 0.5


def test_unsafe_reference_still_respects_the_lockbox(labeled_panel):
    panel, folds, _ = _setup(labeled_panel)
    early_lockbox = folds[-1].test_start  # labels before this window end after it
    with pytest.raises(LeakageError, match="lockbox"):
        unpurged_reference_predictions(
            panel,
            folds[-1:],
            feature_columns=KEYS,
            make_estimator=c.NearestKeyMemorizer,
            lockbox_start=early_lockbox,
        )


def test_symbols_with_fewer_rows_than_min_history_use_the_pooled_mean():
    codes = np.r_[np.zeros(5), np.ones(50)][:, None]
    labels = np.r_[np.full(5, 1.0), np.full(50, 0.0)]
    model = c.PerSymbolMeanForecast(min_history=10).fit(codes, labels)
    np.testing.assert_allclose(model.predict(np.array([[0.0], [1.0]])), [labels.mean(), 0.0])
    assert model.fallback_predictions_ == 1
    assert model.training_counts_ == {0: 5, 1: 50}
