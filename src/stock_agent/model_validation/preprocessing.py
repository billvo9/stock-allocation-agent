"""
Train-only preprocessing (numpy only).

Every transform learns its parameters in fit() from the rows it is given and
applies them unchanged in transform(). Objects are fit-once: fitting the same
instance twice raises, so a transform shared across folds (for example through
a closure in an estimator factory) fails loudly instead of silently carrying
one fold's state into another. The harness supplies a fresh pipeline per fold.

The interface mirrors scikit-learn's fit/transform/predict, so sklearn objects
can replace these if that dependency is approved.

Per-date cross-sectional transforms (ranks or z-scores within one date) are
not learned transforms: they use only same-date features, so they belong in
feature construction, never here, and never with labels as inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_matrix(X: Any) -> np.ndarray:
    matrix = np.array(X, dtype=np.float64, copy=True)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix; got shape {matrix.shape}.")
    return matrix


class _FitOnce:
    _fitted = False

    def _start_fit(self) -> None:
        if self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} is already fitted. Build a fresh instance per "
                "fold; reusing one would carry state across folds."
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use.")


class MedianImputer(_FitOnce):
    """Replace missing values with each column's training median."""

    def fit(self, X: Any, y: Any = None) -> MedianImputer:
        self._start_fit()
        matrix = _as_matrix(X)
        empty = np.flatnonzero(np.isnan(matrix).all(axis=0))
        if matrix.shape[0] == 0 or empty.size:
            raise ValueError(f"Columns {empty.tolist()} have no observed training values.")
        self.medians_ = np.nanmedian(matrix, axis=0)
        self._fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        matrix = _as_matrix(X)
        rows, cols = np.nonzero(np.isnan(matrix))
        matrix[rows, cols] = self.medians_[cols]
        return matrix


class Standardizer(_FitOnce):
    """
    Center and scale with training mean and population std (ddof=0).

    Constant training columns get scale 1.0. A column is constant when its
    std is within floating-point rounding of zero relative to its mean (a
    column of 0.1s has std ~1e-17, not 0), which would otherwise turn
    test-period values into ~1e15. Missing values must be imputed first.
    """

    def fit(self, X: Any, y: Any = None) -> Standardizer:
        self._start_fit()
        matrix = _as_matrix(X)
        if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
            raise ValueError("Standardizer needs non-empty, finite training data; impute first.")
        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        rounding = 10.0 * np.finfo(np.float64).eps * np.abs(self.mean_)
        scale[(scale == 0.0) | (scale <= rounding)] = 1.0
        self.scale_ = scale
        self._fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        return (_as_matrix(X) - self.mean_) / self.scale_


class Pipeline(_FitOnce):
    """Apply transforms in order, then a final estimator with fit/predict."""

    def __init__(self, transforms: Sequence[Any], estimator: Any) -> None:
        self.transforms = list(transforms)
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> Pipeline:
        self._start_fit()
        matrix = _as_matrix(X)
        for transform in self.transforms:
            matrix = transform.fit(matrix, y).transform(matrix)
        self.estimator.fit(matrix, np.asarray(y, dtype=np.float64))
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._require_fitted()
        matrix = _as_matrix(X)
        for transform in self.transforms:
            matrix = transform.transform(matrix)
        return np.asarray(self.estimator.predict(matrix), dtype=np.float64)
