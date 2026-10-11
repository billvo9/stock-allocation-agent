"""
OWNER EXERCISE: the contract of the T4 linear core (model_diagnostics/linear.py).

Every test below is marked xfail(strict=True, raises=NotImplementedError):
it is expected to fail while the core raises NotImplementedError, and turns
into an error the moment it passes ("XPASS(strict)"). When you implement a
function, delete the marker on its tests; they must then pass as written.
Do not weaken an assertion to make it pass (AGENTS.md).

See docs/research/linear_models.md "Owner exercise" for the order of work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics.controls import PooledMeanForecast
from stock_agent.model_diagnostics.linear import LinearForecast, Selection, select_hyperparameters

OWNER = pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="OWNER: implement model_diagnostics/linear.py"
)


def _data(n=400, seed=0, missing=False):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3) * np.array([1.0, 5.0, 0.1]) + np.array([0.0, 2.0, -1.0])
    y = 0.5 * (X[:, 0] - X[:, 0].mean()) + 0.01 * rng.randn(n) + 0.02
    if missing:
        X[::7, 1] = np.nan
    return X, y


# --- LinearForecast ---


@OWNER
def test_transform_parameters_are_statistics_of_the_training_rows():
    X, y = _data(missing=True)
    model = LinearForecast("ridge", 1.0).fit(X, y)
    medians = np.nanmedian(X, axis=0)
    imputed = np.where(np.isnan(X), medians, X)
    np.testing.assert_allclose(model.medians_, medians)
    np.testing.assert_allclose(model.means_, imputed.mean(axis=0))
    np.testing.assert_allclose(model.scales_, imputed.std(axis=0, ddof=0))


@OWNER
def test_predictions_are_row_wise():
    X, y = _data()
    model = LinearForecast("elastic_net", 0.1, 0.5).fit(X, y)
    together = model.predict(X[:5])
    one_by_one = np.concatenate([model.predict(X[i : i + 1]) for i in range(5)])
    np.testing.assert_allclose(together, one_by_one)


@OWNER
def test_ols_nests_the_pooled_mean():
    # On standardized (mean-zero) features the OLS intercept is the mean label:
    # the pooled-mean null is OLS with every slope restricted to zero, fitted on
    # the same rows. This is what makes Clark-West valid for ols vs the null.
    X, y = _data()
    ols = LinearForecast("ols").fit(X, y)
    assert ols.intercept_ == pytest.approx(PooledMeanForecast().fit(X, y).mean_)
    assert ols.alpha_ == 0.0


@OWNER
def test_penalty_scales_come_from_the_fit_rows():
    X, y = _data()
    n = len(y)
    ridge = LinearForecast("ridge", 0.1).fit(X, y)
    assert ridge.alpha_ == pytest.approx(0.1 * n)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    lambda_max = np.abs(Z.T @ (y - y.mean())).max() / n
    lasso = LinearForecast("lasso", 0.2).fit(X, y)
    assert lasso.alpha_ == pytest.approx(0.2 * lambda_max)
    net = LinearForecast("elastic_net", 0.2, 0.5).fit(X, y)
    assert net.alpha_ == pytest.approx(0.2 * lambda_max / 0.5)


@OWNER
@pytest.mark.parametrize(("method", "l1_ratio"), [("lasso", None), ("elastic_net", 0.5)])
def test_the_top_of_an_l1_grid_zeroes_every_coefficient(method, l1_ratio):
    X, y = _data()
    model = LinearForecast(method, 1.0, l1_ratio).fit(X, y)
    np.testing.assert_allclose(model.coef_, 0.0, atol=1e-12)
    assert model.converged_


@OWNER
@pytest.mark.parametrize(
    ("method", "penalty", "l1_ratio"),
    [("ridge", 1e-3, None), ("lasso", 0.01, None), ("ols", None, None)],
)
def test_a_planted_signal_keeps_its_sign_and_noise_stays_small(method, penalty, l1_ratio):
    X, y = _data()
    model = LinearForecast(method, penalty, l1_ratio).fit(X, y)
    assert model.coef_[0] > 0.3  # standardized: the planted 0.5 slope on a unit-variance feature
    assert np.abs(model.coef_[1:]).max() < 0.05


@OWNER
def test_a_fitted_model_is_never_refit():
    X, y = _data()
    model = LinearForecast("ridge", 1.0).fit(X, y)
    with pytest.raises(RuntimeError):
        model.fit(X, y)


# --- select_hyperparameters ---


def _inner(per_date):
    """per_date: list of (date, [(label, prediction), ...])."""

    rows = [
        {
            "fold_id": 0,
            "date": pd.Timestamp(day, tz="UTC"),
            "symbol": f"S{i}",
            "target_return": y,
            "prediction": f,
        }
        for day, cells in per_date
        for i, (y, f) in enumerate(cells)
    ]
    return pd.DataFrame(rows)


@OWNER
def test_selection_minimises_the_date_normalized_mse():
    # Point 0 errs on a 1-name date, point 1 on a 9-name date. Date-normalized MSE
    # weighs both dates equally: 0 scores (0.04 + 0)/2, 1 scores (0 + 0.01)/2.
    zero = _inner([("2024-01-02", [(0.2, 0.0)]), ("2024-01-03", [(0.1, 0.1)] * 9)])
    one = _inner([("2024-01-02", [(0.2, 0.2)]), ("2024-01-03", [(0.1, 0.0)] * 9)])
    grid = [{"penalty": 1.0}, {"penalty": 0.1}]
    selection = select_hyperparameters(grid, [zero, one])
    assert isinstance(selection, Selection)
    np.testing.assert_allclose(selection.scores, [0.02, 0.005])
    assert selection.chosen == 1
    assert selection.n_dates == (2, 2)


@OWNER
def test_ties_go_to_the_larger_penalty_then_the_larger_l1_ratio():
    same = _inner([("2024-01-02", [(0.1, 0.0), (0.0, 0.0)])])
    grid = [
        {"penalty": 0.1, "l1_ratio": 0.8},
        {"penalty": 1.0, "l1_ratio": 0.2},
        {"penalty": 1.0, "l1_ratio": 0.5},
    ]
    assert select_hyperparameters(grid, [same, same, same]).chosen == 2
