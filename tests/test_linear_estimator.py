"""
The real linear core inside the nested pipeline, under normal and
adversarial conditions: every fold-specific value comes from that fold's
eligible training rows only, penalty scales come from each fit's own rows,
the harness's one-date-at-a-time prediction equals whole-window prediction,
and selection behaves sensibly on planted signal and on pure noise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from stock_agent.features.training import LabelSpec
from stock_agent.model_diagnostics import linear, nested
from stock_agent.model_validation.harness import run_walk_forward
from stock_agent.model_validation.preprocessing import MedianImputer, Standardizer

FEATURES = ["feature_a", "feature_b"]
SPEC = LabelSpec(5, 1, benchmark="equal_weight")


def _specs(design):
    return {s.name: s for s in nested.linear_models(SPEC, FEATURES, design)}


def _run(frame, folds, spec, design):
    inner = {fold.fold_id: nested.build_inner_folds(frame, fold, design) for fold in folds}
    return nested.run_nested(
        frame,
        folds,
        spec,
        inner,
        selector=linear.select_hyperparameters,
        lockbox_start=folds[0].holdout_start,
    )


@pytest.fixture(scope="module")
def sandbox(linear_run_root):
    _, _, labeled, folds, design = linear_run_root
    return labeled.reset_index(drop=True), folds, design


@pytest.mark.parametrize("name", ["ridge", "lasso", "elastic_net"])
def test_fold_values_depend_only_on_that_fold_s_eligible_training_rows(sandbox, name):
    frame, folds, design = sandbox
    spec = _specs(design)[name]
    baseline = _run(frame, folds, spec, design)
    target = folds[1]
    outside = np.setdiff1d(np.arange(len(frame)), target.train_rows)
    rng = np.random.RandomState(4)
    shocked = frame.copy()
    # Adversarial: the fold's test rows, purged rows and every later row get
    # extreme features and labels; a leak would move this fold's fit.
    for column in FEATURES:
        shocked.loc[outside, column] = rng.randn(len(outside)) * 50.0
    shocked.loc[outside, "target_return"] = rng.randn(len(outside)) * 5.0
    perturbed = _run(shocked, folds, spec, design)

    def fold_rows(result, table):
        rows = pd.DataFrame(getattr(result, table))
        return rows[rows["fold_id"] == target.fold_id].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        fold_rows(baseline, "coefficient_rows"),
        fold_rows(perturbed, "coefficient_rows"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        fold_rows(baseline, "selection_rows"),
        fold_rows(perturbed, "selection_rows"),
        check_exact=True,
    )
    # The shock does reach later folds (the test can see a change when one exists).
    later = pd.DataFrame(perturbed.coefficient_rows)
    original = pd.DataFrame(baseline.coefficient_rows)
    assert not np.allclose(
        later[later["fold_id"] == folds[-1].fold_id]["coefficient"],
        original[original["fold_id"] == folds[-1].fold_id]["coefficient"],
    )


@pytest.mark.parametrize(("method", "l1_ratio"), [("lasso", None), ("elastic_net", 0.5)])
def test_inner_penalty_scales_come_from_each_inner_fit_s_rows(sandbox, method, l1_ratio):
    frame, folds, design = sandbox
    inner = nested.build_inner_folds(frame, folds[-1], design)
    point = {"penalty": 0.1, **({"l1_ratio": l1_ratio} if l1_ratio else {})}
    result = run_walk_forward(
        frame,
        inner,
        feature_columns=FEATURES,
        make_estimator=lambda: linear.LinearForecast(method, **point),
        lockbox_start=folds[0].holdout_start,
    )
    alphas = []
    for fold, estimator in zip(inner, result.estimators, strict=True):
        rows = frame.iloc[fold.train_rows]
        filled = MedianImputer().fit(rows[FEATURES]).transform(rows[FEATURES])
        standardized = Standardizer().fit(filled).transform(filled)
        expected = 0.1 * linear.lambda_max(
            standardized, rows["target_return"].to_numpy(), l1_ratio or 1.0
        )
        assert estimator.alpha_ == pytest.approx(expected, rel=1e-12)
        alphas.append(estimator.alpha_)
    outer = linear.LinearForecast(method, **point).fit(
        frame.iloc[folds[-1].train_rows][FEATURES].to_numpy(),
        frame.iloc[folds[-1].train_rows]["target_return"].to_numpy(),
    )
    assert all(alpha != pytest.approx(outer.alpha_, rel=1e-9) for alpha in alphas)


@pytest.mark.parametrize("name", ["ridge", "elastic_net"])
def test_the_stored_tables_record_exactly_what_was_scored_and_fitted(sandbox, name):
    frame, folds, design = sandbox
    spec = _specs(design)[name]
    grid = spec.params["selection"]["grid"]
    base = nested.estimator_kwargs(spec.params)
    result = _run(frame, folds, spec, design)
    selection = pd.DataFrame(result.selection_rows)
    coefficients = pd.DataFrame(result.coefficient_rows)
    for fold, estimator in zip(folds, result.estimators, strict=True):
        inner = nested.build_inner_folds(frame, fold, design)
        outputs = []
        for point in grid:  # independent re-scoring of every grid point
            with threadpool_limits(limits=1):
                outputs.append(
                    run_walk_forward(
                        frame,
                        inner,
                        feature_columns=FEATURES,
                        make_estimator=lambda point=point: spec.estimator(**{**base, **point}),
                        lockbox_start=folds[0].holdout_start,
                    ).predictions
                )
        expected = linear.select_hyperparameters(grid, outputs)
        rows = selection[selection["fold_id"] == fold.fold_id].reset_index(drop=True)
        assert rows["penalty"].tolist() == [point["penalty"] for point in grid]
        assert rows["inner_score"].tolist() == list(expected.scores)
        assert rows["n_inner_dates"].tolist() == list(expected.n_dates)
        assert rows["chosen"].tolist() == [k == expected.chosen for k in range(len(grid))]
        chosen = rows[rows["chosen"]].iloc[0]
        assert estimator.penalty == chosen["penalty"]
        if name == "elastic_net":
            assert estimator.l1_ratio == chosen["l1_ratio"]
        fitted = coefficients[coefficients["fold_id"] == fold.fold_id].set_index("feature")
        for column, values in (
            ("coefficient", estimator.coef_),
            ("median", estimator.medians_),
            ("mean", estimator.means_),
            ("scale", estimator.scales_),
        ):
            np.testing.assert_array_equal(fitted.loc[FEATURES, column].to_numpy(), values)
        assert fitted.loc["__intercept__", "coefficient"] == estimator.intercept_
        assert (fitted["alpha"] == estimator.alpha_).all()


def test_scores_further_apart_than_the_tie_tolerance_are_not_tied():
    grid = [{"penalty": 0.1}, {"penalty": 1.0}]
    frame = pd.DataFrame({"date": [0, 0], "target_return": [0.0, 0.0], "prediction": [0.0, 0.0]})

    def scored(error):  # one date, mean squared error = error
        return frame.assign(prediction=[error**0.5, -(error**0.5)])

    # Scores below 1: the tolerance is absolute (1e-12); a larger gap is not a tie.
    apart = linear.select_hyperparameters(grid, [scored(1e-3), scored(1e-3 + 3e-12)])
    assert apart.chosen == 0
    within = linear.select_hyperparameters(grid, [scored(1e-3), scored(1e-3 + 1e-13)])
    assert within.chosen == 1  # tied: the larger penalty wins


class _FakeSolver:
    """Stands in for sklearn's coordinate descent to drive one convergence signal at a time."""

    def __init__(self, *, warn: bool, at_max_iter: bool, other: bool = False):
        self.warn, self.at_max_iter, self.other = warn, at_max_iter, other

    def __call__(self, **options):  # called like the sklearn class
        self.max_iter = options["max_iter"]
        return self

    def fit(self, X, y):
        if self.warn:
            warnings.warn("stalled", ConvergenceWarning, stacklevel=2)
        if self.other:
            warnings.warn("unrelated", RuntimeWarning, stacklevel=2)
        self.n_iter_ = self.max_iter if self.at_max_iter else 1
        self.coef_, self.intercept_ = np.zeros(X.shape[1]), 0.0
        return self


@pytest.mark.parametrize(
    ("warn", "at_max_iter", "converged"),
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_either_convergence_signal_alone_marks_the_fit(
    sandbox, monkeypatch, warn, at_max_iter, converged
):
    frame, folds, _ = sandbox
    rows = frame.iloc[folds[-1].train_rows]
    monkeypatch.setattr(linear, "Lasso", _FakeSolver(warn=warn, at_max_iter=at_max_iter))
    fitted = linear.LinearForecast("lasso", 0.1).fit(
        rows[FEATURES].to_numpy(), rows["target_return"].to_numpy()
    )
    assert fitted.converged_ is converged


def test_warnings_other_than_convergence_still_surface(sandbox, monkeypatch):
    frame, folds, _ = sandbox
    rows = frame.iloc[folds[-1].train_rows]
    monkeypatch.setattr(linear, "Lasso", _FakeSolver(warn=False, at_max_iter=False, other=True))
    with pytest.warns(RuntimeWarning, match="unrelated"):
        fitted = linear.LinearForecast("lasso", 0.1).fit(
            rows[FEATURES].to_numpy(), rows["target_return"].to_numpy()
        )
    assert fitted.converged_ is True


def test_one_date_at_a_time_equals_whole_window_prediction(sandbox):
    frame, folds, _ = sandbox
    fold = folds[-1]
    gappy = frame.copy()
    test = gappy.iloc[fold.test_rows].index
    gappy.loc[test[::3], "feature_a"] = np.nan  # missing test features: training medians only
    result = run_walk_forward(
        gappy,
        [fold],
        feature_columns=FEATURES,
        make_estimator=lambda: linear.LinearForecast("ridge", 0.1),
        lockbox_start=fold.holdout_start,
    )
    (estimator,) = result.estimators
    whole = estimator.predict(gappy.iloc[fold.test_rows][FEATURES].to_numpy())
    np.testing.assert_allclose(result.predictions["prediction"].to_numpy(), whole, rtol=0, atol=0)
    medians = estimator.medians_
    row = gappy.iloc[[fold.test_rows[0]]][FEATURES].to_numpy(copy=True)
    row[0, 0] = np.nan
    filled = row.copy()
    filled[0, 0] = medians[0]
    assert estimator.predict(row)[0] == estimator.predict(filled)[0]


def _planted(frame, strength):
    rng = np.random.RandomState(9)
    planted = frame.copy()
    label = planted["target_return"].fillna(0.0).to_numpy()
    planted["feature_a"] = strength * label / label.std() + rng.randn(len(planted))
    planted["feature_b"] = rng.randn(len(planted))
    return planted


def test_selection_recovers_a_planted_signal_and_rejects_noise(sandbox):
    frame, folds, design = sandbox
    specs = _specs(design)
    strong = _planted(frame, 2.0)
    for name in ("ridge", "lasso", "elastic_net"):
        result = _run(strong, folds, specs[name], design)
        coefficients = pd.DataFrame(result.coefficient_rows)
        signal = coefficients[coefficients["feature"] == "feature_a"]["coefficient"]
        noise = coefficients[coefficients["feature"] == "feature_b"]["coefficient"]
        assert (signal > 0).all(), name
        assert noise.abs().max() < signal.min(), name
    noise_only = _planted(frame, 0.0)
    result = _run(noise_only, folds, specs["lasso"], design)
    chosen = pd.DataFrame(result.selection_rows).query("chosen")
    coefficients = pd.DataFrame(result.coefficient_rows)
    slopes = coefficients[coefficients["feature"] != "__intercept__"]["coefficient"]
    # On pure noise the inner validation should prefer heavy shrinkage: well
    # below what unpenalized OLS fits to the same noise on the same folds (a
    # fixed threshold would be vacuous at this label scale, SD ~0.02).
    assert chosen["penalty"].median() >= 0.1
    ols = [
        linear.LinearForecast("ols")
        .fit(
            noise_only.iloc[fold.train_rows][FEATURES].to_numpy(),
            noise_only.iloc[fold.train_rows]["target_return"].to_numpy(),
        )
        .coef_
        for fold in folds
    ]
    assert slopes.abs().sum() < 0.6 * np.abs(ols).sum()


@pytest.mark.parametrize(("method", "l1_ratio"), [("lasso", None), ("elastic_net", 0.5)])
def test_a_solver_stopped_before_convergence_says_so(sandbox, method, l1_ratio):
    frame, folds, _ = sandbox
    rows = frame.iloc[folds[-1].train_rows]
    X, y = rows[FEATURES].to_numpy(), rows["target_return"].to_numpy()
    options = {"l1_ratio": l1_ratio} if l1_ratio else {}
    stalled = linear.LinearForecast(method, 0.01, **options, tol=0.0, max_iter=2).fit(X, y)
    assert stalled.converged_ is False
    assert linear.has_converged(stalled) is False
    settled = linear.LinearForecast(method, 0.01, **options).fit(X, y)
    assert settled.converged_ is True
