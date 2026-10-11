"""
T4 estimator core: penalized linear forecasts and nested hyperparameter
selection. OWNER-IMPLEMENTED (learning core; see AGENTS.md "Autonomy" and
docs/research/linear_models.md "Owner exercise").

The plumbing around this module is complete: nested.py builds the inner
purged folds, runs each grid point through the validation harness, calls
`select_hyperparameters`, refits each outer fold once with the chosen point,
and records `coefficients` and `selection` tables from the attributes listed
below. The functions marked OWNER raise NotImplementedError until written;
tests/test_linear_core.py states their contract and turns green once they
are correct (remove the strict xfail markers then).

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ("ridge", "lasso", "elastic_net", "ols")
PENALIZED = ("ridge", "lasso", "elastic_net")


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearForecast:
        raise NotImplementedError(
            "OWNER: impute (training medians), standardize (training means and "
            "population std), compute the penalty scale from these rows, fit the "
            "sklearn model, and set medians_, means_, scales_, coef_, intercept_, "
            "alpha_ and converged_. Fit exactly once."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "OWNER: apply the stored medians, means and scales, then "
            "intercept_ + X_std @ coef_. Row-wise: a batch of one row gives the same "
            "value as that row inside a larger batch."
        )


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

    raise NotImplementedError(
        "OWNER: score each grid point with the date-normalized MSE, choose the "
        "lowest, break ties by larger penalty then larger l1_ratio, and return a "
        "Selection with every score and date count."
    )


def has_converged(estimator: object) -> bool | None:
    """Plumbing helper: an estimator's converged_ flag, or None if it has none."""

    value = getattr(estimator, "converged_", None)
    return None if value is None else bool(np.asarray(value).all())
