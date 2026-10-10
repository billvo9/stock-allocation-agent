"""
Null models, controls, and the leakage canary, as ordinary estimators.

Every predictor here runs through model_validation.harness.run_walk_forward
with the same folds as any learned model: global purge, lockbox, fresh
estimator per fold, one predict call per date. None has a special path.

The harness hands estimators only the declared feature matrix. Predictors
that need a row's identity read it from numeric control columns added by
add_control_columns:

    control_symbol_code    position of the symbol in the sorted universe
    control_session_index  dense rank of the row's session date

Both are prefix-stable (adding later rows never changes an earlier value)
and label-blind. They are for controls only: a candidate model that used
them could quietly fit a per-symbol effect (see runner.ModelSpec).

Roles (recorded with every result):

    null            no information about the label; expected score zero
    control         a known, non-zero reference (benchmark, selection,
                    timeliness); its score is reported, never "accepted"
    canary          a memorizer whose only possible skill is leakage
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from stock_agent.model_validation.audit import datetime_ns

CONTROL_SYMBOL_CODE = "control_symbol_code"
CONTROL_SESSION_INDEX = "control_session_index"
CONTROL_COLUMNS = (CONTROL_SYMBOL_CODE, CONTROL_SESSION_INDEX)
CONTROL_PREFIX = "control_"

_KEY_STRIDE = 10_000_000


def symbol_codes(universe: Iterable[str]) -> dict[str, int]:
    """Alphabetical position of each symbol; recorded in the run record."""

    symbols = sorted(set(universe))
    if not symbols:
        raise ValueError("universe must not be empty.")
    return {symbol: code for code, symbol in enumerate(symbols)}


def add_control_columns(labeled: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    """
    Copy of `labeled` (same rows, same order) with the two control columns.

    Codes come from the sorted universe, never from first appearance, so a
    symbol's code does not depend on which rows are present.
    """

    codes = symbol_codes(universe)
    unknown = sorted(set(labeled["symbol"]) - set(codes))
    if unknown:
        raise ValueError(f"Symbols outside the universe: {unknown}")
    present = [column for column in CONTROL_COLUMNS if column in labeled.columns]
    if present:
        raise ValueError(f"Frame already has control columns: {present}")
    result = labeled.copy()
    result[CONTROL_SYMBOL_CODE] = result["symbol"].map(codes).astype(np.int64)
    dates = datetime_ns(result["date"])
    result[CONTROL_SESSION_INDEX] = np.unique(dates, return_inverse=True)[1].astype(np.int64)
    return result


def _matrix(X: np.ndarray, columns: int) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < columns:
        raise ValueError(f"Expected a 2-D matrix with at least {columns} column(s).")
    return matrix


def _integer_column(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all() or (values != np.round(values)).any():
        raise ValueError(f"{name} must hold finite integers.")
    return values.astype(np.int64)


def _training_mean(y: np.ndarray) -> float:
    targets = np.asarray(y, dtype=np.float64)
    if targets.size == 0 or not np.isfinite(targets).all():
        raise ValueError("Training labels must be non-empty and finite.")
    return float(targets.mean())


class ZeroForecast:
    """Null: predicts 0 for every row. Ignores features and labels."""

    def fit(self, X, y):
        self.n_train_ = len(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)


class PooledMeanForecast:
    """
    Null for timing and cross-sectional skill: the mean training label.

    It predicts one value per fold, so it has no cross-sectional ranking
    (rank IC is undefined) and is the Campbell-Thompson baseline for
    out-of-sample R^2 under a raw-return label.
    """

    def fit(self, X, y):
        self.mean_ = _training_mean(y)
        self.n_train_ = len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=np.float64)


class PerSymbolMeanForecast:
    """
    Selection control: each symbol's mean training label.

    Not a null. On a hindsight-selected universe the symbol with the best
    history keeps ranking first, which shows up as rank IC that comes from
    selection, not foresight. Symbols with fewer than `min_history`
    training rows fall back to the pooled training mean, and every fallback
    prediction is counted.

    Feature column 0 must be control_symbol_code.
    """

    def __init__(self, min_history: int = 63):
        if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
            raise ValueError("min_history must be a positive integer.")
        self.min_history = min_history

    def fit(self, X, y):
        codes = _integer_column(_matrix(X, 1)[:, 0], CONTROL_SYMBOL_CODE)
        targets = np.asarray(y, dtype=np.float64)
        self.pooled_mean_ = _training_mean(targets)
        keys, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=targets)
        usable = counts >= self.min_history
        self.symbol_means_ = dict(zip(keys[usable].tolist(), (sums / counts)[usable], strict=True))
        self.training_counts_ = dict(zip(keys.tolist(), counts.tolist(), strict=True))
        self.fallback_predictions_ = 0
        return self

    def predict(self, X):
        codes = _integer_column(_matrix(X, 1)[:, 0], CONTROL_SYMBOL_CODE)
        output = np.full(len(codes), self.pooled_mean_, dtype=np.float64)
        known = np.array([code in self.symbol_means_ for code in codes.tolist()], dtype=bool)
        output[known] = [self.symbol_means_[code] for code in codes[known].tolist()]
        self.fallback_predictions_ += int((~known).sum())
        return output


class FeatureScore:
    """
    Unfitted control: sign * one feature (for example momentum_20d).

    The sign is pre-registered, never chosen from development results. The
    output is a score, not a calibrated forecast: only rank and sign metrics
    are meaningful.
    """

    def __init__(self, sign: int = 1, column: int = 0):
        if sign not in (1, -1) or isinstance(sign, bool):
            raise ValueError("sign must be +1 or -1.")
        self.sign = sign
        self.column = column

    def fit(self, X, y):
        _matrix(X, self.column + 1)
        return self

    def predict(self, X):
        return self.sign * _matrix(X, self.column + 1)[:, self.column].copy()


class RandomNoiseScore:
    """
    Null with a defined ranking: iid standard-normal scores.

    The zero and pooled-mean nulls are constant within a date, so rank IC,
    hit rate and calibration are undefined for them. This null exercises
    those metrics under no skill. One fresh RandomState per fold (the seed
    is derived per fold by the runner); draws follow the harness's ascending
    date order, so results are deterministic.
    """

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        self.seed = seed

    def fit(self, X, y):
        self.random_state_ = np.random.RandomState(self.seed)
        return self

    def predict(self, X):
        return self.random_state_.randn(len(X))


class NearestKeyMemorizer:
    """
    Canary: predicts the label of the nearest training row of the same
    symbol (by session index; ties go to the earlier row).

    Under purged folds the nearest training label ends before the test
    window, so any skill it shows is that of a lagged label. With purging
    removed it copies labels that overlap the test labels and scores
    spuriously; the harness refuses such folds. A symbol with no training
    rows falls back to the pooled training mean (counted).

    Features: control_symbol_code, control_session_index.
    """

    def fit(self, X, y):
        matrix = _matrix(X, 2)
        keys = self._keys(matrix)
        order = np.argsort(keys, kind="stable")
        self.keys_ = keys[order]
        self.labels_ = np.asarray(y, dtype=np.float64)[order]
        self.pooled_mean_ = _training_mean(self.labels_)
        self.fallback_predictions_ = 0
        return self

    @staticmethod
    def _keys(matrix: np.ndarray) -> np.ndarray:
        codes = _integer_column(matrix[:, 0], CONTROL_SYMBOL_CODE)
        sessions = _integer_column(matrix[:, 1], CONTROL_SESSION_INDEX)
        if (sessions < 0).any() or (sessions >= _KEY_STRIDE).any() or (codes < 0).any():
            raise ValueError("Control columns are out of range.")
        return codes * _KEY_STRIDE + sessions

    def predict(self, X):
        keys = self._keys(_matrix(X, 2))
        symbol = keys // _KEY_STRIDE
        lo = np.searchsorted(self.keys_, symbol * _KEY_STRIDE)
        hi = np.searchsorted(self.keys_, (symbol + 1) * _KEY_STRIDE) - 1
        has_history = hi >= lo
        output = np.full(len(keys), self.pooled_mean_, dtype=np.float64)
        if has_history.any():
            k, lo_k, hi_k = keys[has_history], lo[has_history], hi[has_history]
            above = np.clip(np.searchsorted(self.keys_, k), lo_k, hi_k)
            below = np.clip(above - 1, lo_k, hi_k)
            use_below = np.abs(self.keys_[below] - k) <= np.abs(self.keys_[above] - k)
            output[has_history] = self.labels_[np.where(use_below, below, above)]
        self.fallback_predictions_ += int((~has_history).sum())
        return output
