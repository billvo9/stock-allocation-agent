"""
Deliberately leaky reference for the memorizer canary. Never a model result.

The canary has two halves:

- the memorizer run through the purged harness, which must show no skill;
- the same memorizer trained WITHOUT purging (every labeled row dated
  before test_start), which must show spurious skill because overlapping
  labels copy the test returns.

The second half is the canary's positive control: it proves the measurement
layer would notice leakage if the harness let it in. It is computed here,
outside the harness (which refuses unpurged folds), its rows are tagged
role="canary_unsafe_reference", and it still respects the lockbox: no
training label may end on or after lockbox_start.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
)
from stock_agent.model_diagnostics import contract
from stock_agent.model_validation.audit import LeakageError, datetime_ns
from stock_agent.model_validation.checks import validate_feature_columns
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, WalkForwardFold, is_labeled

UNSAFE_ROLE = contract.UNSAFE_REFERENCE_ROLE  # never eligible as a candidate


def unpurged_reference_predictions(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    feature_columns: Sequence[str],
    make_estimator: Callable[[], object],
    lockbox_start: object = MODEL_HOLDOUT_START,
) -> pd.DataFrame:
    """
    Predictions from training on every labeled row dated before each fold's
    test_start (no purge). Same scored rows and per-date predict calls as
    the harness, so the two halves of the canary are directly comparable.
    """

    columns = validate_feature_columns(labeled, feature_columns)
    features = labeled.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    targets = labeled[TARGET_RETURN_COLUMN].to_numpy(dtype=np.float64)
    dates = datetime_ns(labeled["date"])
    ends = datetime_ns(labeled[TARGET_END_COLUMN])
    lockbox = datetime_ns(pd.Series([pd.Timestamp(lockbox_start)]))[0]
    labeled_mask = is_labeled(labeled)

    frames = []
    for fold in folds:
        start = datetime_ns(pd.Series([fold.test_start]))[0]
        train = np.flatnonzero(labeled_mask & (dates < start))
        if (ends[train] >= lockbox).any():
            raise LeakageError(
                f"Fold {fold.fold_id}: the unpurged reference would train on labels that "
                "mature in the lockbox."
            )
        estimator = make_estimator()
        estimator.fit(features[train], targets[train])
        test_dates = dates[fold.test_rows]
        prediction = np.empty(len(fold.test_rows), dtype=np.float64)
        for session in np.unique(test_dates):
            on_date = test_dates == session
            prediction[on_date] = estimator.predict(features[fold.test_rows[on_date]])
        scored = labeled.iloc[fold.test_rows][
            ["date", "symbol", TARGET_START_COLUMN, TARGET_END_COLUMN, TARGET_RETURN_COLUMN]
        ].copy()
        scored.insert(0, "fold_id", fold.fold_id)
        scored["prediction"] = prediction
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)
