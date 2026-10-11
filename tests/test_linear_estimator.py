"""
The real linear core inside the nested pipeline, under normal and
adversarial conditions: every fold-specific value comes from that fold's
eligible training rows only, penalty scales come from each fit's own rows,
the harness's one-date-at-a-time prediction equals whole-window prediction,
and selection behaves sensibly on planted signal and on pure noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        fold_rows(baseline, "coefficient_rows"), fold_rows(perturbed, "coefficient_rows")
    )
    pd.testing.assert_frame_equal(
        fold_rows(baseline, "selection_rows"), fold_rows(perturbed, "selection_rows")
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
    # On pure noise the inner validation should prefer heavy shrinkage.
    assert chosen["penalty"].median() >= 0.1
    assert (slopes.abs() < 0.05).mean() >= 0.5
