"""
Leakage checks: the feature contract, prefix stability, and fold invariants.

The feature contract rejects columns that are labels, keys, timestamps, or
price/volume levels by name and dtype. A name check cannot catch a disguised
leak (e.g. a feature computed with a negative shift), so assert_prefix_stable
checks content: features for dates before T must be identical whether built
from all data or only from data dated before T.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
)
from stock_agent.model_validation.audit import LeakageError, datetime_ns, frame_rows_sha256
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

__all__ = [
    "LeakageError",
    "assert_fold_is_leakage_safe",
    "assert_prefix_stable",
    "validate_feature_columns",
]

# Vendor price and volume levels are back-adjusted with corporate actions
# announced later (yfinance OHLC and volume are split-adjusted; adjusted_close
# also embeds later dividends), so their levels encode the future. Ratios of
# them within a window are safe.
FORBIDDEN_FEATURES = frozenset(
    {
        "date",
        "symbol",
        "available_at",
        "vintage_date",
        "observation_date",
        TARGET_RETURN_COLUMN,
        TARGET_START_COLUMN,
        TARGET_END_COLUMN,
        "adjusted_close",
        "previous_adjusted_close",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)
FORBIDDEN_PREFIXES = ("target_", "label_", "fwd_", "forward_", "future_")
FORBIDDEN_SUFFIXES = ("_available_at", "_date")


def validate_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    """Return the feature list if it satisfies the feature contract; else raise."""

    columns = list(feature_columns)
    problems = []
    if not columns:
        problems.append("feature_columns is empty")
    if len(columns) != len(set(columns)):
        problems.append("feature_columns contains duplicates")
    for column in columns:
        if column not in frame.columns:
            problems.append(f"{column}: not in frame")
            continue
        lowered = column.lower()
        if (
            lowered in FORBIDDEN_FEATURES
            or lowered.startswith(FORBIDDEN_PREFIXES)
            or lowered.endswith(FORBIDDEN_SUFFIXES)
        ):
            problems.append(f"{column}: label, key, timestamp, or price/volume level")
            continue
        dtype = frame[column].dtype
        if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)) or (
            pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype)
        ):
            problems.append(f"{column}: dtype {dtype} is not numeric")
    if problems:
        raise ValueError("Invalid feature columns: " + "; ".join(problems))
    return columns


def assert_prefix_stable(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    *,
    cutoffs: Iterable[object],
    raw_time_column: str = "date",
    output_date_column: str = "date",
    key_columns: Sequence[str] = ("date", "symbol"),
    value_columns: Sequence[str] | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Raise LeakageError unless `build` is prefix-stable at every cutoff T.

    For each T, outputs dated before T must be identical (same keys, same
    values within rtol/atol) when built from all of `raw` and when built
    from only the raw rows whose `raw_time_column` is before T. Use the
    availability timestamp as `raw_time_column` for vintaged sources.
    """

    full = build(raw.copy())
    columns = (
        [c for c in full.columns if c not in key_columns]
        if value_columns is None
        else list(value_columns)
    )
    raw_times = pd.to_datetime(raw[raw_time_column], utc=True)

    for cutoff in cutoffs:
        cut = pd.Timestamp(cutoff)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        partial = build(raw.loc[raw_times < cut].copy())

        expected = _rows_before(full, output_date_column, cut, key_columns, columns)
        actual = _rows_before(partial, output_date_column, cut, key_columns, columns)
        if not expected[list(key_columns)].equals(actual[list(key_columns)]):
            raise LeakageError(f"Prefix instability at {cut}: output rows differ.")
        for column in columns:
            if not _values_equal(expected[column], actual[column], rtol=rtol, atol=atol):
                raise LeakageError(
                    f"Prefix instability at {cut}: column {column!r} changes when data "
                    "dated at or after the cutoff is added (future information leaks in)."
                )


def _rows_before(
    frame: pd.DataFrame,
    date_column: str,
    cutoff: pd.Timestamp,
    key_columns: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column], utc=True)
    subset = frame.loc[dates < cutoff, [*key_columns, *columns]].copy()
    subset[date_column] = pd.to_datetime(subset[date_column], utc=True)
    return subset.sort_values(list(key_columns)).reset_index(drop=True)


def _values_equal(left: pd.Series, right: pd.Series, *, rtol: float, atol: float) -> bool:
    if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
        return bool(
            np.allclose(
                left.to_numpy(dtype=np.float64),
                right.to_numpy(dtype=np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )
    return left.reset_index(drop=True).equals(right.reset_index(drop=True))


def assert_fold_is_leakage_safe(
    labeled: pd.DataFrame,
    fold: object,
    *,
    lockbox_start: object = MODEL_HOLDOUT_START,
    _frame_rows: str | None = None,
) -> None:
    """
    Re-derive a fold's invariants from the data, independently of the builder.

    - the fold belongs to this exact frame and row order (`_frame_rows` is the
      harness's own precomputed frame_rows_sha256 of `labeled`; never pass a
      hash taken from the fold);
    - row positions are unique and in bounds;
    - every used label enters after its feature date (no same-close labels);
    - every training label ended strictly before test_start, so no training
      row is dated at or after the window (no embargo is needed);
    - training is complete: every matured labeled row in scope is used;
    - the test set is complete: every labeled row in the window is scored
      unless its label matures in the holdout (no selection by outcome);
    - in development mode, holdout_start is no later than the owner's
      lockbox, the window ends by holdout_start, and no scored label ends in
      the holdout; holdout windows start at or after holdout_start;
    - train, purged, and test rows are disjoint.
    """

    if (_frame_rows or frame_rows_sha256(labeled)) != fold.frame_rows_sha256:
        raise LeakageError(f"Fold {fold.fold_id} was built from a different frame.")
    for name in ("train_rows", "test_rows", "purged_rows", "unlabeled_rows", "held_out_rows"):
        rows = getattr(fold, name)
        if len(rows) and (rows.min() < 0 or rows.max() >= len(labeled)):
            raise LeakageError(f"Fold {fold.fold_id} {name} has positions out of bounds.")
        if len(np.unique(rows)) != len(rows):
            raise LeakageError(f"Fold {fold.fold_id} {name} has duplicate positions.")

    dates = datetime_ns(labeled["date"])
    starts = datetime_ns(labeled[TARGET_START_COLUMN])
    ends = datetime_ns(labeled[TARGET_END_COLUMN])
    returns = labeled[TARGET_RETURN_COLUMN].to_numpy(dtype=np.float64)
    labeled_mask = np.isfinite(returns) & labeled[TARGET_END_COLUMN].notna().to_numpy()
    start = pd.Timestamp(fold.test_start).as_unit("ns").value
    end = pd.Timestamp(fold.test_end).as_unit("ns").value
    holdout = pd.Timestamp(fold.holdout_start).as_unit("ns").value
    lockbox = pd.Timestamp(lockbox_start).as_unit("ns").value

    train, test, purged = fold.train_rows, fold.test_rows, fold.purged_rows
    if not labeled_mask[train].all() or not labeled_mask[test].all():
        raise LeakageError(f"Fold {fold.fold_id} uses rows without a matured label.")
    used = np.concatenate([train, test])
    if (starts[used] <= dates[used]).any():
        raise LeakageError(f"Fold {fold.fold_id} uses a label entering at its own feature date.")
    if fold.mode == "development" and holdout > lockbox:
        raise LeakageError(
            f"Development fold {fold.fold_id} sets holdout_start after the lockbox start."
        )
    if (ends[train] >= start).any():
        raise LeakageError(f"Fold {fold.fold_id} trains on a label ending at or after test_start.")
    if (dates[train] >= start).any():
        raise LeakageError(f"Fold {fold.fold_id} trains on a row dated at or after test_start.")

    in_scope = np.ones(len(labeled), dtype=bool)
    if fold.train_start is not None:
        in_scope = dates >= pd.Timestamp(fold.train_start).as_unit("ns").value
    expected_train = np.flatnonzero(labeled_mask & in_scope & (ends < start))
    if not np.array_equal(np.sort(train), expected_train):
        raise LeakageError(f"Fold {fold.fold_id} training set is not the full purged set.")

    if ((dates[test] < start) | (dates[test] >= end)).any():
        raise LeakageError(f"Fold {fold.fold_id} scores rows outside its test window.")
    if fold.mode == "development":
        if end > holdout:
            raise LeakageError(f"Fold {fold.fold_id} test window reaches into the holdout.")
        if (ends[test] >= holdout).any():
            raise LeakageError(f"Fold {fold.fold_id} scores labels that mature in the holdout.")
    elif start < holdout:
        raise LeakageError(f"Holdout fold {fold.fold_id} starts before holdout_start.")

    scorable = labeled_mask & (dates >= start) & (dates < end)
    if fold.mode == "development":
        scorable &= ends < holdout
    if not np.array_equal(np.sort(test), np.flatnonzero(scorable)):
        raise LeakageError(
            f"Fold {fold.fold_id} test set is not every scorable row in its window "
            "(rows were added or dropped, e.g. selected by outcome)."
        )

    if (
        np.intersect1d(train, test).size
        or np.intersect1d(train, purged).size
        or np.intersect1d(test, purged).size
    ):
        raise LeakageError(f"Fold {fold.fold_id} train, purged, and test rows overlap.")
