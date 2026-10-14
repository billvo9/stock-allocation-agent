"""The HistGradientBoosting challenger: estimator contract, nested plumbing, honest increments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from ml_helpers import config_with, run

from stock_agent.features.training import LabelSpec
from stock_agent.forecasting import nonlinear
from stock_agent.registry import open_registry

CHALLENGER = nonlinear.NAME


@pytest.fixture(scope="module")
def bent(tmp_path_factory):
    """30 names with a planted nonlinear signal (no linear content in the characteristic)."""

    root = tmp_path_factory.mktemp("bent")
    config = config_with(
        experiment="test_bent",
        data__params__signal="nonlinear",
        data__params__n_assets=30,
        data__params__signal_strength=0.0015,
        models__challengers=[CHALLENGER],
        evaluation__permutation_models=[],
    )
    return root, run(config, root)


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    """The linear planted panel of the runner tests, with the challenger added."""

    root = tmp_path_factory.mktemp("straight")
    config = config_with(
        experiment="test_straight",
        models__challengers=[CHALLENGER],
        evaluation__permutation_models=[],
    )
    return run(config, root)


# --- estimator ---


def matrix(seed=0, rows=400):
    rng = np.random.RandomState(seed)
    X = rng.randn(rows, 3)
    y = np.abs(X[:, 0]) - 0.8 + 0.1 * rng.randn(rows)
    return X, y


def test_the_estimator_is_deterministic_and_fitted_once():
    X, y = matrix()
    point = dict(nonlinear.GRID[1])
    first = nonlinear.GradientBoostingForecast(**point).fit(X, y)
    second = nonlinear.GradientBoostingForecast(**point).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    params = first.model_.get_params()
    assert params["early_stopping"] is False  # no random (non-temporal) validation split
    assert params["loss"] == "squared_error"
    with pytest.raises(RuntimeError, match="fitted once"):
        first.fit(X, y)
    with pytest.raises(RuntimeError, match="before predicting"):
        nonlinear.GradientBoostingForecast(**point).predict(X)


def test_missing_values_are_imputed_with_training_medians_only():
    X, y = matrix()
    X[::7, 1] = np.nan
    model = nonlinear.GradientBoostingForecast(**nonlinear.GRID[1]).fit(X, y)
    np.testing.assert_array_equal(model.imputer_.medians_, np.nanmedian(X, axis=0))
    test = np.array([[0.5, np.nan, 0.0]])
    filled = test.copy()
    filled[0, 1] = np.nanmedian(X[:, 1])
    np.testing.assert_array_equal(model.predict(test), model.predict(filled))


def test_predictions_are_row_wise_and_independent_of_the_batch():
    # The harness predicts one date at a time; an estimator whose output for a
    # row depends on the other rows of its batch (for example imputation refitted
    # on test rows) would read later dates' features. Only the fit rows may matter.
    X, y = matrix(rows=600)
    X[::5, 2] = np.nan
    model = nonlinear.GradientBoostingForecast(**nonlinear.GRID[1]).fit(X, y)
    test = matrix(seed=1, rows=120)[0]
    test[::3, 0] = np.nan
    test[1::4, 2] = np.nan
    test[60:] += 10.0  # the second half alone has very different medians
    whole = model.predict(test)
    one_by_one = np.concatenate([model.predict(test[i : i + 1]) for i in range(len(test))])
    halves = np.concatenate([model.predict(test[:60]), model.predict(test[60:])])
    shuffled = np.random.RandomState(2).permutation(len(test))
    np.testing.assert_array_equal(whole, one_by_one)
    np.testing.assert_array_equal(whole, halves)
    np.testing.assert_array_equal(whole[shuffled], model.predict(test[shuffled]))


def test_the_grid_is_a_pre_registered_complexity_path():
    penalties = [point["penalty"] for point in nonlinear.GRID]
    assert penalties == sorted(penalties, reverse=True) and len(set(penalties)) == 4
    leaves = [point["max_leaf_nodes"] for point in nonlinear.GRID]
    iterations = [point["max_iter"] for point in nonlinear.GRID]
    assert leaves == sorted(leaves) and iterations == sorted(iterations)  # larger penalty: simpler


def test_the_challenger_is_a_nested_candidate_whose_grid_is_part_of_its_identity():
    spec = nonlinear.challenger_spec(LabelSpec(10, 1), ["a", "b"])
    assert spec.role == "candidate" and "selection" in spec.params
    changed = replace(
        spec,
        params={
            **spec.params,
            "selection": {**spec.params["selection"], "grid": spec.params["selection"]["grid"][:2]},
        },
    )
    assert spec.describe()["variant_id"] != changed.describe()["variant_id"]


def test_the_challenger_learns_a_shape_a_line_cannot():
    X, y = matrix(rows=2000)
    model = nonlinear.GradientBoostingForecast(**nonlinear.GRID[0]).fit(X, y)
    assert np.corrcoef(model.predict(X), np.abs(X[:, 0]))[0, 1] > 0.8


# --- through the runner ---


def test_the_challenger_runs_on_the_same_folds_tuning_and_nulls(bent):
    root, result = bent
    tables = result.diagnostics.tables
    predictions = tables["predictions"]
    keys = ["fold_id", "date", "symbol"]
    ridge = predictions.query("model == 'ridge'")[keys].reset_index(drop=True)
    boosted = predictions.query("model == @CHALLENGER")[keys].reset_index(drop=True)
    pd.testing.assert_frame_equal(ridge, boosted)
    selection = tables["selection"].query("model == @CHALLENGER")
    folds = tables["folds"]
    assert len(selection) == len(folds) * len(nonlinear.GRID)
    assert selection.groupby("fold_id")["chosen"].sum().eq(1).all()
    assert tables["coefficients"].query("model == @CHALLENGER").empty
    checks = tables["checks"]
    inside = checks[(checks["check"] == "nested_selection_inside_training")]
    assert inside.set_index("model").loc[CHALLENGER, "passed"]
    baselines = {"zero", "per_symbol_mean", "ols"}
    compared = {
        metric.removeprefix("oos_r2_vs_")
        for metric in tables["metrics"].query("model == @CHALLENGER")["metric"]
        if metric.startswith("oos_r2_vs_")
    }
    assert baselines <= compared
    testing = result.record["multiple_testing"]
    assert testing["candidates_in_run"] == 2
    assert testing["grid_points_in_run"] == 6 + len(nonlinear.GRID)
    entry = open_registry(root / "registry").entry(f"test_xs.{CHALLENGER}@v1")
    assert entry.evidence["role"] == "candidate"


def test_the_increment_over_every_linear_baseline_is_reported_with_uncertainty(bent):
    _, result = bent
    metrics = result.tables["metrics"].query("model == @CHALLENGER")
    for metric in ("rank_ic_difference", "oos_r2_vs_reference", "mse_improvement_vs_reference"):
        assert set(metrics.query("metric == @metric")["reference"]) == {"ridge", "ols"}
    folds = metrics.query("metric == 'rank_ic_difference' and inference_method == 'fold_block_t'")
    assert (folds["se"] > 0).all() and folds["ci_low"].notna().all()


def test_a_planted_nonlinear_signal_gives_a_credible_increment(bent):
    _, result = bent
    boosted = result.record["summary"][CHALLENGER]
    assert boosted["rank_ic_difference_vs_ridge"] > 0.05
    assert boosted["rank_ic_difference_vs_ridge_t"] > 2.0
    assert boosted["mse_improvement_vs_ridge_t"] > 2.0
    assert boosted["oos_r2_vs_ridge"] > 0.0


def test_a_linear_truth_gives_no_credible_increment(straight):
    boosted = straight.record["summary"][CHALLENGER]
    assert abs(boosted["rank_ic_difference_vs_ridge_t"]) < 2.0
    assert boosted["mse_improvement_vs_ridge_t"] < 2.0
    assert straight.record["summary"]["ridge"]["coverage_weighted_rank_ic_t"] > 3.0
