"""
Walk-forward fit/predict loop.

For each fold the harness re-checks the fold's leakage invariants, builds a
fresh estimator from the factory, fits it on the fold's training rows only
(allowlisted feature columns, no keys, dates, or labels), and predicts the
test rows. Test labels never reach the estimator; they are attached to the
prediction frame afterwards, for scoring only.

predict() is called once per test date with that date's cross-section. A
single call over a whole window would let a batch-aware estimator (batch
z-scores, ranks, transductive methods) read features from later dates in
the window, and momentum_20d at t+21 equals the label at t. Per-date calls
make cross-sectional operations inside an estimator legitimate.

Metrics, tuning, and persistence are deliberately out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
)
from stock_agent.model_validation.audit import datetime_ns, frame_rows_sha256
from stock_agent.model_validation.checks import (
    assert_fold_is_leakage_safe,
    validate_feature_columns,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, WalkForwardFold

_ATOMIC = (str, bytes, int, float, bool, complex, type(None), np.ndarray, np.generic)


def _stateful_components(obj: Any) -> set[int]:
    """
    Identities of every mutable object reachable from an unfitted estimator.

    Used to reject a factory that shares any component (a transform, an inner
    estimator) with an earlier fold, even inside a freshly built wrapper.
    """

    seen: set[int] = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, _ATOMIC) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
        elif hasattr(current, "__dict__"):
            stack.extend(vars(current).values())
    return seen


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class WalkForwardResult:
    """Out-of-sample predictions keyed by fold, date, and symbol."""

    predictions: pd.DataFrame
    folds: tuple[WalkForwardFold, ...]
    estimators: tuple[Any, ...]


def run_walk_forward(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    feature_columns: Sequence[str],
    make_estimator: Callable[[], Estimator],
    lockbox_start: object = MODEL_HOLDOUT_START,
) -> WalkForwardResult:
    if not folds:
        raise ValueError("At least one fold is required.")
    if len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError("Fold ids must be unique.")
    scored_rows = np.concatenate([fold.test_rows for fold in folds])
    if len(np.unique(scored_rows)) != len(scored_rows):
        raise ValueError("A row is scored in more than one fold.")

    columns = validate_feature_columns(labeled, feature_columns)
    features = labeled.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    targets = labeled[TARGET_RETURN_COLUMN].to_numpy(dtype=np.float64)
    dates = datetime_ns(labeled["date"])
    label_columns = [TARGET_START_COLUMN, TARGET_END_COLUMN, TARGET_RETURN_COLUMN]

    frame_rows = frame_rows_sha256(labeled)
    estimators: list[Any] = []
    used_components: set[int] = set()
    frames = []
    for fold in folds:
        assert_fold_is_leakage_safe(
            labeled, fold, lockbox_start=lockbox_start, _frame_rows=frame_rows
        )

        estimator = make_estimator()
        components = _stateful_components(estimator)
        if components & used_components:
            raise ValueError(
                f"make_estimator shares an estimator or component with an earlier fold "
                f"(fold {fold.fold_id}); it must build a fresh, unfitted estimator per call."
            )
        used_components |= components
        estimators.append(estimator)

        estimator.fit(features[fold.train_rows], targets[fold.train_rows])

        test_dates = dates[fold.test_rows]
        prediction = np.empty(len(fold.test_rows), dtype=np.float64)
        for session in np.unique(test_dates):
            on_date = test_dates == session
            batch = fold.test_rows[on_date]
            output = np.asarray(estimator.predict(features[batch]), dtype=np.float64)
            if output.shape != (len(batch),):
                raise ValueError(
                    f"Fold {fold.fold_id}: expected {len(batch)} predictions for one date, "
                    f"got shape {output.shape}."
                )
            prediction[on_date] = output
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold.fold_id}: predictions must be finite.")

        scored = labeled.iloc[fold.test_rows][["date", "symbol", *label_columns]].copy()
        scored.insert(0, "fold_id", fold.fold_id)
        scored["prediction"] = prediction
        frames.append(scored)

    predictions = pd.concat(frames, ignore_index=True)
    return WalkForwardResult(
        predictions=predictions,
        folds=tuple(folds),
        estimators=tuple(estimators),
    )
