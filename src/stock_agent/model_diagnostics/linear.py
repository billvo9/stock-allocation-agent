"""
T4 estimator core: penalized linear forecasts and nested hyperparameter
selection (docs/research/linear_models.md). Implemented under the owner's
explicit delegation of 2026-10-11; the contract tests are
tests/test_linear_core.py.

nested.py builds the inner purged folds, runs each grid point through the
validation harness, calls `select_hyperparameters`, refits each outer fold
once with the chosen point, and records `coefficients` and `selection`
tables from the attributes listed below. Every value a fit computes (the
imputation medians, the standardization, the penalty scale, the
coefficients) comes from the rows that fit is given, which the harness
restricts to one fold's eligible (purged) training rows.

Contract of a fitted `LinearForecast` (read by nested.py):
    medians_   (n_features,) training-row medians used to impute NaN
    means_     (n_features,) training-row means of the imputed features
    scales_    (n_features,) training-row population std (1.0 if ~constant)
    coef_      (n_features,) coefficients on the standardized features
    intercept_ float
    alpha_     float, the penalty actually used (0.0 for OLS)
    converged_ bool, False if the solver stopped at max_iter
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from stock_agent.model_validation.preprocessing import MedianImputer, Standardizer

METHODS = ("ridge", "lasso", "elastic_net", "ols")
PENALIZED = ("ridge", "lasso", "elastic_net")
# Grid points whose scores differ from the best by at most
# TIE_TOLERANCE * max(1, |best|) are tied: absolute for scores below 1 (inner
# MSEs here are ~1e-3), relative above. The tie rule then decides (larger
# penalty, then larger l1_ratio).
TIE_TOLERANCE = 1e-12


def lambda_max(Z: np.ndarray, y: np.ndarray, l1_ratio: float = 1.0) -> float:
    """
    The smallest L1 penalty (sklearn's Lasso/ElasticNet objective, which
    divides the squared error by 2n) that sets every coefficient to zero:
        max_j |z_j' (y - mean y)| / (n * l1_ratio)
    for centred (standardized) Z.
    """

    centred = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    return float(np.max(np.abs(Z.T @ centred)) / (len(centred) * l1_ratio))


class LinearForecast:
    """
    Median-impute, standardize (training rows only), then a linear model on
    the standardized features, refit from scratch per fold:

        ridge        sklearn Ridge(alpha = penalty * n_fit)
        lasso        sklearn Lasso(alpha = penalty * lambda_max)
        elastic_net  sklearn ElasticNet(alpha = penalty * lambda_max, l1_ratio)
        ols          sklearn LinearRegression (no penalty; penalty must be None)

    lambda_max = max_j |x_j' (y - mean(y))| / (n_fit * l1_ratio) on the
    standardized fit rows (l1_ratio = 1 for lasso): the smallest penalty that
    sets every coefficient to zero. Each scale comes from the rows this fit
    sees, never from rows of another fold or an inner validation block.
    Lasso and Elastic Net use selection="cyclic", tol and max_iter as given.
    """

    def __init__(
        self,
        method: str,
        penalty: float | None = None,
        l1_ratio: float | None = None,
        *,
        tol: float = 1e-10,
        max_iter: int = 100_000,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}; got {method!r}.")
        if method == "ols" and penalty is not None:
            raise ValueError("OLS takes no penalty.")
        if method in PENALIZED and (penalty is None or not penalty > 0):
            raise ValueError(f"{method} needs a positive penalty.")
        if (method == "elastic_net") != (l1_ratio is not None):
            raise ValueError("l1_ratio is required for elastic_net and only for it.")
        if l1_ratio is not None and not 0.0 < l1_ratio < 1.0:
            raise ValueError("l1_ratio must lie strictly between 0 and 1.")
        self.method = method
        self.penalty = None if penalty is None else float(penalty)
        self.l1_ratio = None if l1_ratio is None else float(l1_ratio)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearForecast:
        if self._fitted:
            raise RuntimeError("A LinearForecast is fitted once; build a new one per fold.")
        matrix = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
            raise ValueError("X must be 2-D with one row per label.")
        if matrix.shape[0] < 2 or not np.isfinite(target).all():
            raise ValueError("A fit needs at least two rows with finite labels.")
        self._fitted = True  # past input validation: never refit this object, even on failure

        imputer = MedianImputer().fit(matrix)
        filled = imputer.transform(matrix)
        scaler = Standardizer().fit(filled)
        standardized = scaler.transform(filled)
        n_fit = len(target)
        self.medians_ = imputer.medians_
        self.means_ = scaler.mean_
        self.scales_ = scaler.scale_

        if self.method == "ols":
            model, self.alpha_ = LinearRegression(), 0.0
        elif self.method == "ridge":
            self.alpha_ = self.penalty * n_fit
            model = Ridge(alpha=self.alpha_, solver="cholesky")
        else:
            mix = 1.0 if self.method == "lasso" else self.l1_ratio
            self.alpha_ = self.penalty * lambda_max(standardized, target, mix)
            if not self.alpha_ > 0.0:  # constant labels or features: nothing to fit
                return self._constant(target, standardized.shape[1])
            options = {"alpha": self.alpha_, "tol": self.tol, "max_iter": self.max_iter}
            if self.method == "lasso":
                model = Lasso(selection="cyclic", **options)
            else:
                model = ElasticNet(l1_ratio=mix, selection="cyclic", **options)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(standardized, target)
        stalled = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:  # recorded as converged_ = False; any other warning still surfaces
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        iterations = np.atleast_1d(getattr(model, "n_iter_", 0) or 0)
        self.converged_ = bool(not stalled and (iterations < self.max_iter).all())
        self.coef_ = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        self.intercept_ = float(model.intercept_)
        return self

    def _constant(self, target: np.ndarray, width: int) -> LinearForecast:
        self.coef_ = np.zeros(width)
        self.intercept_ = float(np.mean(target))
        self.converged_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or not hasattr(self, "coef_"):
            raise RuntimeError("Fit the LinearForecast before predicting.")
        matrix = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isnan(matrix), self.medians_, matrix)
        standardized = (filled - self.means_) / self.scales_
        # Exactly row-wise: each row uses only the stored training statistics and
        # sums its features in the same order whatever the batch size (a matrix
        # product would let BLAS change the summation order with the shape).
        return self.intercept_ + np.sum(standardized * self.coef_, axis=1)


@dataclass(frozen=True)
class Selection:
    """The inner validation result for one outer fold (one row per grid point)."""

    chosen: int
    scores: tuple[float, ...]
    n_dates: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.n_dates):
            raise ValueError("scores and n_dates must have one entry per grid point.")
        if not 0 <= self.chosen < len(self.scores):
            raise ValueError("chosen must index a grid point.")


def select_hyperparameters(
    grid: Sequence[Mapping[str, float]],
    inner_predictions: Sequence[pd.DataFrame],
    *,
    target_column: str = "target_return",
) -> Selection:
    """
    Pre-registered selection rule. `inner_predictions[k]` holds grid point k's
    harness predictions over every inner fold (columns fold_id, date, symbol,
    prediction and the label).

    Criterion: the mean over inner test dates of the per-date cross-sectional
    mean squared error, mean_t [ (1/N_t) sum_i (y_it - f_it)^2 ] (each date
    weighs the same). Lowest wins. Tie rule (a total order): the larger
    `penalty`, then the larger `l1_ratio`. Returns every grid point's score
    and its number of inner dates so they can be stored.
    """

    if len(grid) != len(inner_predictions) or not grid:
        raise ValueError("One inner prediction frame per grid point is required.")
    scores, dates = [], []
    for frame in inner_predictions:
        errors = (frame[target_column] - frame["prediction"]) ** 2
        per_date = errors.groupby(frame["date"]).mean()
        scores.append(float(per_date.mean()) if len(per_date) else float("nan"))
        dates.append(int(per_date.size))
    finite = [k for k, score in enumerate(scores) if np.isfinite(score)]
    if not finite:
        raise ValueError("No grid point has an inner score.")
    best = min(scores[k] for k in finite)
    tolerance = TIE_TOLERANCE * max(1.0, abs(best))
    tied = [k for k in finite if scores[k] - best <= tolerance]
    chosen = max(
        tied,
        key=lambda k: (float(grid[k]["penalty"]), float(grid[k].get("l1_ratio", 0.0)), -k),
    )
    return Selection(chosen=chosen, scores=tuple(scores), n_dates=tuple(dates))


def has_converged(estimator: object) -> bool | None:
    """Plumbing helper: an estimator's converged_ flag, or None if it has none."""

    value = getattr(estimator, "converged_", None)
    return None if value is None else bool(np.asarray(value).all())
