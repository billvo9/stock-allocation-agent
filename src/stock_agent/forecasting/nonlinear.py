"""
A simple nonlinear challenger: scikit-learn HistGradientBoostingRegressor
(already a dependency through scikit-learn 1.9.1; nothing new is installed).

It is evaluated exactly like the linear baselines (same folds, same nested
purged selection, same nulls, same checks), so any increment is measured on
identical rows.

Estimator (`GradientBoostingForecast`), fitted once per fold:
    1. median-impute each feature with the fit rows' medians (the features'
       contract policy impute_training_median; no scaling: trees are
       invariant to monotone transforms);
    2. HistGradientBoostingRegressor(loss="squared_error", early_stopping=False,
       random_state fixed). Early stopping is off because it would carve a
       random (not temporal) validation split out of the training rows.

Pre-registered grid: a one-dimensional complexity path. `penalty` is the
path index; a larger value is a simpler model, so T4's tie rule (the larger
penalty wins) prefers the simpler model, and the stored `selection` rows
keep their contract-1.2 columns (penalty; l1_ratio NaN).

    penalty  max_leaf_nodes  max_iter  learning_rate  min_samples_leaf  l2
    4        4               50        0.05           200               1.0
    3        8               100       0.05           100               1.0
    2        16              150       0.05           50                1.0
    1        31              200       0.05           20                0.0

Changing the grid is a new variant (it is part of the spec's variant id).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from stock_agent.features.training import LabelSpec
from stock_agent.model_diagnostics import nested
from stock_agent.model_validation.preprocessing import MedianImputer

NAME = "hist_gradient_boosting"
PREPROCESSING = "median impute (training rows only, per fit); no scaling"
GRID = (
    {
        "penalty": 4.0,
        "max_leaf_nodes": 4,
        "max_iter": 50,
        "learning_rate": 0.05,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
    },
    {
        "penalty": 3.0,
        "max_leaf_nodes": 8,
        "max_iter": 100,
        "learning_rate": 0.05,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "penalty": 2.0,
        "max_leaf_nodes": 16,
        "max_iter": 150,
        "learning_rate": 0.05,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "penalty": 1.0,
        "max_leaf_nodes": 31,
        "max_iter": 200,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
)
TIE_RULE = "larger penalty (the simpler model on the pre-registered complexity path)"


class GradientBoostingForecast:
    """Median imputation, then histogram gradient boosting; refit from scratch per fold."""

    def __init__(
        self,
        penalty: float,
        max_leaf_nodes: int,
        max_iter: int,
        learning_rate: float,
        min_samples_leaf: int,
        l2_regularization: float,
        *,
        random_state: int = 0,
    ) -> None:
        self.penalty = float(penalty)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.random_state = int(random_state)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingForecast:
        if self._fitted:
            raise RuntimeError("A GradientBoostingForecast is fitted once; build one per fold.")
        matrix = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
            raise ValueError("X must be 2-D with one row per label.")
        if matrix.shape[0] < 2 or not np.isfinite(target).all():
            raise ValueError("A fit needs at least two rows with finite labels.")
        self._fitted = True
        self.imputer_ = MedianImputer().fit(matrix)
        self.model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )
        self.model_.fit(self.imputer_.transform(matrix), target)
        self.n_iter_ = int(self.model_.n_iter_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or not hasattr(self, "model_"):
            raise RuntimeError("Fit the GradientBoostingForecast before predicting.")
        return np.asarray(
            self.model_.predict(self.imputer_.transform(np.asarray(X, dtype=np.float64))),
            dtype=np.float64,
        )


def challenger_spec(
    label_spec: LabelSpec,
    feature_columns: Sequence[str],
    design: nested.InnerDesign = nested.DEFAULT_DESIGN,
):
    """The challenger as a nested-selected candidate (T4's selection plumbing)."""

    from stock_agent.model_diagnostics.runner import ModelSpec  # avoid an import cycle

    return ModelSpec(
        NAME,
        "candidate",
        GradientBoostingForecast,
        tuple(feature_columns),
        params={
            "preprocessing": PREPROCESSING,
            "label_id": label_spec.label_id,
            "selection": {
                "grid": [dict(point) for point in GRID],
                "criterion": nested.CRITERION,
                "tie_rule": TIE_RULE,
                "inner": {
                    "block_sessions": design.block_sessions,
                    "n_windows": design.n_windows,
                    "min_train_sessions": design.min_train_sessions,
                },
            },
        },
        note="nonlinear challenger; complexity chosen by nested purged temporal validation",
    )
