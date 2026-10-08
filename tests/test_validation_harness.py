"""
Train-only preprocessing, the walk-forward harness, and the leakage canary.

The canary is a pure memorizer on independent random walks: it has no real
predictive power, so any out-of-sample skill it shows comes from leakage. It
must score near zero through the harness (purged folds) and score strongly
when purging is removed, which the harness itself refuses to run.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec
from stock_agent.model_validation.checks import LeakageError
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows
from stock_agent.model_validation.harness import run_walk_forward
from stock_agent.model_validation.preprocessing import MedianImputer, Pipeline, Standardizer

FEATURES = ["feature_a", "feature_b"]


class SpyEstimator:
    """Records exactly what it is fitted on; predicts the training mean."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _folds(panel, sessions_between=(40, 100), block=20, holdout_index=110):
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    windows = make_test_windows(
        sessions,
        start=sessions.iloc[sessions_between[0]],
        end=sessions.iloc[sessions_between[1]],
        block_sessions=block,
    )
    return make_expanding_folds(panel, windows, holdout_start=sessions.iloc[holdout_index])


# --- preprocessing ---


def test_median_imputer_fills_with_training_medians_only():
    train = np.array([[1.0, np.nan], [3.0, 10.0], [100.0, 30.0]])
    imputer = MedianImputer().fit(train)

    np.testing.assert_allclose(imputer.medians_, [3.0, 20.0])
    filled = imputer.transform(np.array([[np.nan, np.nan], [-5.0, 1.0]]))
    np.testing.assert_allclose(filled, [[3.0, 20.0], [-5.0, 1.0]])


def test_standardizer_parameters_come_from_training_rows_only():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    scaler = Standardizer().fit(train)

    np.testing.assert_allclose(scaler.mean_, [2.0, 5.0])
    np.testing.assert_allclose(scaler.scale_, [1.0, 1.0])  # zero variance -> 1.0
    shifted = scaler.transform(np.array([[1000.0, 6.0]]))  # a test-period level shift
    np.testing.assert_allclose(shifted, [[998.0, 1.0]])
    np.testing.assert_allclose(scaler.mean_, [2.0, 5.0])


def test_transforms_reject_unusable_training_data():
    with pytest.raises(ValueError, match="no observed"):
        MedianImputer().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
    with pytest.raises(ValueError, match="impute first"):
        Standardizer().fit(np.array([[1.0], [np.nan]]))
    with pytest.raises(RuntimeError, match="fitted before"):
        Standardizer().transform(np.ones((2, 1)))


def test_standardizer_treats_near_constant_columns_as_constant():
    # np.std of a column of 0.1s is ~1e-17, not 0; dividing by it would turn
    # test values into ~1e15.
    scaler = Standardizer().fit(np.full((3, 1), 0.1))
    assert scaler.scale_[0] == 1.0
    assert abs(scaler.transform(np.array([[0.2]]))[0, 0] - 0.1) < 1e-12


@pytest.mark.parametrize("make", [MedianImputer, Standardizer])
def test_transforms_are_fit_once(make):
    transform = make().fit(np.array([[1.0], [2.0]]))
    with pytest.raises(RuntimeError, match="already fitted"):
        transform.fit(np.array([[5.0], [6.0]]))


def test_pipeline_fits_transforms_then_estimator_on_transformed_rows():
    spy = SpyEstimator()
    pipeline = Pipeline([MedianImputer(), Standardizer()], spy)
    pipeline.fit(np.array([[1.0], [np.nan], [3.0]]), np.array([0.1, 0.2, 0.3]))

    ((X_fit, _),) = spy.fit_calls
    np.testing.assert_allclose(X_fit[:, 0], [-1.224744871391589, 0.0, 1.224744871391589])
    with pytest.raises(RuntimeError, match="already fitted"):
        pipeline.fit(np.ones((2, 1)), np.ones(2))


# --- harness ---


def test_estimator_sees_exactly_the_training_rows_and_declared_features(labeled_panel):
    panel = labeled_panel()
    folds = _folds(panel)

    result = run_walk_forward(panel, folds, feature_columns=FEATURES, make_estimator=SpyEstimator)

    for fold, spy in zip(folds, result.estimators, strict=True):
        ((X_fit, y_fit),) = spy.fit_calls
        np.testing.assert_array_equal(X_fit, panel.loc[fold.train_rows, FEATURES].to_numpy())
        np.testing.assert_array_equal(y_fit, panel.loc[fold.train_rows, "target_return"])


def test_prediction_frame_is_keyed_and_carries_label_timing(labeled_panel):
    panel = labeled_panel()
    folds = _folds(panel)

    result = run_walk_forward(panel, folds, feature_columns=FEATURES, make_estimator=SpyEstimator)

    assert list(result.predictions.columns) == [
        "fold_id",
        "date",
        "symbol",
        "target_start_date",
        "target_end_date",
        "target_return",
        "prediction",
    ]
    assert len(result.predictions) == sum(len(fold.test_rows) for fold in folds)


def test_factory_must_build_a_fresh_estimator_per_fold(labeled_panel):
    panel = labeled_panel()
    shared = SpyEstimator()
    with pytest.raises(ValueError, match="fresh"):
        run_walk_forward(
            panel, _folds(panel), feature_columns=FEATURES, make_estimator=lambda: shared
        )


def test_a_transform_shared_across_folds_fails_loudly(labeled_panel):
    panel = labeled_panel()
    scaler = Standardizer()
    # Caught before fitting by the harness's component check; the transforms'
    # own fit-once guard is the second layer (test_transforms_are_fit_once).
    with pytest.raises(ValueError, match="shares"):
        run_walk_forward(
            panel,
            _folds(panel),
            feature_columns=FEATURES,
            make_estimator=lambda: Pipeline([MedianImputer(), scaler], SpyEstimator()),
        )


def test_fold_results_do_not_depend_on_earlier_folds(labeled_panel):
    panel = labeled_panel()
    folds = _folds(panel)

    def make():
        return Pipeline([MedianImputer(), Standardizer()], SpyEstimator())

    together = run_walk_forward(panel, folds, feature_columns=FEATURES, make_estimator=make)
    alone = run_walk_forward(panel, folds[-1:], feature_columns=FEATURES, make_estimator=make)

    last = together.predictions[together.predictions["fold_id"] == folds[-1].fold_id]
    np.testing.assert_array_equal(last["prediction"], alone.predictions["prediction"])


@pytest.mark.parametrize("seed", range(3))
def test_fitted_state_ignores_every_value_outside_the_training_rows(labeled_panel, seed):
    panel = labeled_panel(seed=seed)
    folds = _folds(panel)
    rng = np.random.RandomState(seed)

    def make():
        return Pipeline([MedianImputer(), Standardizer()], SpyEstimator())

    for fold in folds:
        base = run_walk_forward(panel, [fold], feature_columns=FEATURES, make_estimator=make)

        perturbed = panel.copy()
        outside = np.setdiff1d(np.arange(len(panel)), fold.train_rows)
        perturbed.loc[outside, FEATURES] = rng.randn(len(outside), len(FEATURES)) * 1e3
        labeled = perturbed["target_return"].notna().to_numpy()
        changed = outside[labeled[outside]]
        perturbed.loc[changed, "target_return"] = rng.randn(len(changed))
        moved = run_walk_forward(perturbed, [fold], feature_columns=FEATURES, make_estimator=make)

        (before,), (after,) = base.estimators, moved.estimators
        np.testing.assert_array_equal(before.transforms[0].medians_, after.transforms[0].medians_)
        np.testing.assert_array_equal(before.transforms[1].mean_, after.transforms[1].mean_)
        np.testing.assert_array_equal(before.transforms[1].scale_, after.transforms[1].scale_)
        assert before.estimator.mean_ == after.estimator.mean_


def test_predictions_do_not_depend_on_test_labels(labeled_panel):
    panel = labeled_panel()
    folds = _folds(panel)

    def make():
        return Pipeline([Standardizer()], SpyEstimator())

    # Later folds legitimately train on earlier folds' matured test rows, so
    # perturb one fold's own test labels at a time.
    for fold in folds:
        own = panel.copy()
        own.loc[fold.test_rows, "target_return"] = 99.0
        base = run_walk_forward(panel, [fold], feature_columns=FEATURES, make_estimator=make)
        moved = run_walk_forward(own, [fold], feature_columns=FEATURES, make_estimator=make)
        np.testing.assert_array_equal(
            base.predictions["prediction"], moved.predictions["prediction"]
        )


class LeastSquares:
    """Test-only linear estimator whose predictions depend on each row's features."""

    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.coef_


@pytest.mark.parametrize("seed", range(3))
def test_each_prediction_depends_only_on_training_rows_and_its_own_features(labeled_panel, seed):
    # A transform that computes statistics over the test batch (test-set
    # mean, test-set median, batch normalization) breaks this invariant.
    panel = labeled_panel(seed=seed)
    rng = np.random.RandomState(seed)

    def make():
        return Pipeline([MedianImputer(), Standardizer()], LeastSquares())

    for fold in _folds(panel):
        first, others = fold.test_rows[0], fold.test_rows[1:]
        panel_with_gap = panel.copy()
        panel_with_gap.loc[first, "feature_a"] = np.nan  # exercises the imputer too
        base = run_walk_forward(
            panel_with_gap, [fold], feature_columns=FEATURES, make_estimator=make
        )

        moved_panel = panel_with_gap.copy()
        moved_panel.loc[others, FEATURES] = rng.randn(len(others), len(FEATURES)) * 50.0 + 25.0
        moved = run_walk_forward(moved_panel, [fold], feature_columns=FEATURES, make_estimator=make)

        np.testing.assert_allclose(
            base.predictions["prediction"].iloc[0],
            moved.predictions["prediction"].iloc[0],
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda panel: panel.sample(frac=1.0, random_state=0).reset_index(drop=True),
        lambda panel: panel.drop(index=0).reset_index(drop=True),
    ],
    ids=["same_keys_reordered", "different_keys"],
)
def test_harness_refuses_folds_built_for_another_frame_or_row_order(labeled_panel, change):
    panel = labeled_panel()
    folds = _folds(panel)
    with pytest.raises(LeakageError, match="different frame"):
        run_walk_forward(
            change(panel), folds, feature_columns=FEATURES, make_estimator=SpyEstimator
        )


def test_harness_enforces_the_feature_contract(labeled_panel):
    panel = labeled_panel()
    with pytest.raises(ValueError, match="label, key"):
        run_walk_forward(
            panel, _folds(panel), feature_columns=["target_return"], make_estimator=SpyEstimator
        )


@pytest.mark.parametrize(
    "predict",
    [lambda X: np.zeros(len(X) + 1), lambda X: np.full(len(X), np.nan)],
)
def test_harness_rejects_malformed_predictions(labeled_panel, predict):
    class Broken(SpyEstimator):
        def predict(self, X):
            return predict(X)

    panel = labeled_panel()
    with pytest.raises(ValueError, match="predictions"):
        run_walk_forward(panel, _folds(panel), feature_columns=FEATURES, make_estimator=Broken)


# --- leakage canary ---


class NearestKeyMemorizer:
    """Predicts the label of the nearest training row of the same symbol."""

    def fit(self, X, y):
        keys = X[:, 0].astype(np.int64)
        order = np.argsort(keys, kind="stable")
        self.keys_, self.labels_ = keys[order], y[order]
        return self

    def predict(self, X):
        keys = X[:, 0].astype(np.int64)
        lo = np.searchsorted(self.keys_, (keys // 1000) * 1000)  # first key of the symbol
        hi = np.searchsorted(self.keys_, (keys // 1000 + 1) * 1000) - 1  # last key of it
        above = np.clip(np.searchsorted(self.keys_, keys), lo, hi)
        below = np.clip(above - 1, lo, hi)
        use_below = np.abs(self.keys_[below] - keys) <= np.abs(self.keys_[above] - keys)
        return self.labels_[np.where(use_below, below, above)]  # ties go to the earlier key


def _canary_panel(seed, symbols=40, sessions=260, horizon=10):
    rng = np.random.RandomState(seed)
    calendar = pd.bdate_range("2022-01-03", periods=sessions, tz="UTC")
    frames = []
    for code in range(symbols):
        prices = 100.0 * np.exp(np.cumsum(0.01 * rng.randn(sessions)))
        frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "symbol": f"S{code:02d}",
                    "adjusted_close": prices,
                    "canary_key": code * 1000 + np.arange(sessions),
                }
            )
        )
    labeled = LabelSpec(horizon=horizon, entry_lag=1).apply(pd.concat(frames))
    labeled = labeled.sort_values(["date", "symbol"]).reset_index(drop=True)
    windows = make_test_windows(calendar, start=calendar[60], end=calendar[250], block_sessions=5)
    folds = make_expanding_folds(labeled, windows, holdout_start=calendar[-1] + pd.offsets.BDay(1))
    return labeled, folds


@pytest.mark.parametrize("seed", range(5))
def test_memorizer_has_no_skill_through_purged_folds(seed):
    labeled, folds = _canary_panel(seed)

    result = run_walk_forward(
        labeled, folds, feature_columns=["canary_key"], make_estimator=NearestKeyMemorizer
    )

    scored = result.predictions
    correlation = np.corrcoef(scored["prediction"], scored["target_return"])[0, 1]
    assert abs(correlation) < 0.15


@pytest.mark.parametrize("seed", range(5))
def test_memorizer_scores_spuriously_without_the_purge_and_the_harness_refuses(seed):
    labeled, folds = _canary_panel(seed)
    dates = labeled["date"]
    labels = labeled["target_return"].to_numpy()
    keys = labeled[["canary_key"]].to_numpy(dtype=np.float64)

    predictions, actual = [], []
    for fold in folds:
        unpurged = np.flatnonzero(labeled["target_return"].notna() & (dates < fold.test_start))
        model = NearestKeyMemorizer().fit(keys[unpurged], labels[unpurged])
        predictions.append(model.predict(keys[fold.test_rows]))
        actual.append(labels[fold.test_rows])
    correlation = np.corrcoef(np.concatenate(predictions), np.concatenate(actual))[0, 1]
    assert correlation > 0.5  # overlapping labels leak the test returns

    leaky = dataclasses.replace(folds[0], train_rows=np.flatnonzero(dates < folds[0].test_start))
    with pytest.raises(LeakageError):
        run_walk_forward(
            labeled, [leaky], feature_columns=["canary_key"], make_estimator=NearestKeyMemorizer
        )


class DateBatchRecorder(SpyEstimator):
    """Records how many distinct dates each predict call receives."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def predict(self, X):
        self.batches.append(np.unique(X[:, -1]))
        return super().predict(X)


def test_predict_receives_one_date_at_a_time(labeled_panel):
    # A whole-window call would let a batch-aware estimator read later dates'
    # features, and momentum_20d at t+21 equals the label at t.
    panel = labeled_panel()
    panel["session_code"] = panel["date"].rank(method="dense")
    folds = _folds(panel)

    result = run_walk_forward(
        panel, folds, feature_columns=[*FEATURES, "session_code"], make_estimator=DateBatchRecorder
    )

    for fold, recorder in zip(folds, result.estimators, strict=True):
        assert all(len(batch) == 1 for batch in recorder.batches)
        assert len(recorder.batches) == panel.loc[fold.test_rows, "date"].nunique()


def test_a_shared_inner_estimator_fails_even_inside_a_fresh_pipeline(labeled_panel):
    panel = labeled_panel()
    shared = SpyEstimator()
    with pytest.raises(ValueError, match="shares"):
        run_walk_forward(
            panel,
            _folds(panel),
            feature_columns=FEATURES,
            make_estimator=lambda: Pipeline([MedianImputer()], shared),
        )


@pytest.mark.parametrize(
    "select, message",
    [
        (lambda folds: [], "At least one fold"),
        (lambda folds: [folds[0], dataclasses.replace(folds[1], fold_id=0)], "unique"),
        (lambda folds: [folds[0], folds[0]], "unique"),
    ],
)
def test_fold_lists_must_be_non_empty_and_distinct(labeled_panel, select, message):
    panel = labeled_panel()
    with pytest.raises(ValueError, match=message):
        run_walk_forward(
            panel, select(_folds(panel)), feature_columns=FEATURES, make_estimator=SpyEstimator
        )


def test_rows_scored_in_two_folds_are_rejected(labeled_panel):
    panel = labeled_panel()
    first, second = _folds(panel)[:2]
    overlapping = dataclasses.replace(second, fold_id=99, test_rows=first.test_rows)
    with pytest.raises(ValueError, match="more than one fold"):
        run_walk_forward(
            panel, [first, overlapping], feature_columns=FEATURES, make_estimator=SpyEstimator
        )


def test_relabeling_the_same_keys_invalidates_the_folds(labeled_panel):
    short = labeled_panel(horizon=5)
    folds = _folds(short)
    long = labeled_panel(horizon=20)  # identical keys and row order, longer labels
    with pytest.raises(LeakageError):
        run_walk_forward(long, folds, feature_columns=FEATURES, make_estimator=SpyEstimator)


def test_harness_enforces_the_lockbox(labeled_panel):
    panel = labeled_panel()
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    folds = make_expanding_folds(
        panel,
        [(sessions.iloc[60], sessions.iloc[80])],
        holdout_start=sessions.iloc[100],
        lockbox_start=sessions.iloc[100],
    )
    with pytest.raises(LeakageError, match="lockbox"):
        run_walk_forward(
            panel,
            folds,
            feature_columns=FEATURES,
            make_estimator=SpyEstimator,
            lockbox_start=sessions.iloc[90],
        )
