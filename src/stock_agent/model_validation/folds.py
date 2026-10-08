"""
Purged expanding walk-forward folds.

Timing
    A row is a (date, symbol) observation. `date` is an exchange session date
    stored as 00:00 UTC whose features include that session's close.
    `target_end_date` is the session of the label's exit close; the label is
    known only after that close. Fold boundaries are therefore session dates
    (00:00 UTC); intraday boundaries are rejected.

Folds
    Each fold tests a half-open window [test_start, test_end). Its knowledge
    cutoff is test_start: a row trains only if target_end_date < test_start
    (strictly; a label ending on test_start is not known before that day's
    decision). Since target_end_date > date, training rows also precede the
    window. Training windows expand: every earlier matured row is used.

Purge (global across symbols)
    Labeled rows dated before test_start whose labels end on or after it are
    purged for every symbol, including symbols with no test rows. The names
    are strongly correlated, so another symbol's label over the test window
    carries information about the test labels.

No embargo
    An embargo removes training rows dated after a test window whose
    backward-looking features overlap the test labels. Expanding forward folds
    never train on rows after the window (asserted by
    checks.assert_fold_is_leakage_safe). K-fold, CPCV, or any inner CV that
    trains on later blocks would need a two-sided purge (also dropping later
    rows whose label windows overlap the test window) plus an embargo of at
    least the longest feature lookback plus the entry lag (21 sessions
    today): momentum_20d at row s equals the label of row s - 20 - entry_lag.

Entry after the feature date
    Labels must enter strictly after their row date (target_start_date >
    date; owner decision 2026-10-06, MODEL_LABEL_SPEC). Same-close labels are
    rejected: end-of-day data arrives after the close, and some macro series
    are published after the close on their date-level release day.

Lockbox
    mode="development": windows must end by holdout_start, and window rows
    whose labels end on or after holdout_start are not scored, so development
    results never use holdout-period returns. Development folds may not set
    holdout_start later than MODEL_HOLDOUT_START (the owner's lockbox).
    mode="holdout": windows start at or after holdout_start and are evaluated
    once per pre-registered model.

Nested selection
    Build inner folds from an outer fold's training rows with holdout_start set
    to the outer test_start; inner test labels then mature before the outer
    test window opens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
)
from stock_agent.model_validation.audit import datetime_ns, frame_rows_sha256

MODES = ("development", "holdout")

REQUIRED_COLUMNS = (
    "date",
    "symbol",
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    TARGET_END_COLUMN,
)

# Owner decision (2026-10-06): data from this session onward is the final
# holdout. Development folds may never treat it as development data.
MODEL_HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")


@dataclass(frozen=True, eq=False)
class WalkForwardFold:
    """
    One fold as read-only row positions into the frame it was built from.

    frame_rows_sha256 binds the positions to that frame's row order; the
    harness refuses to apply a fold to any other frame or ordering.
    """

    fold_id: int
    mode: str
    holdout_start: pd.Timestamp
    train_start: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: np.ndarray
    test_rows: np.ndarray
    purged_rows: np.ndarray
    unlabeled_rows: np.ndarray
    held_out_rows: np.ndarray
    frame_rows_sha256: str


def _session_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.normalize():
        raise ValueError(
            f"{name} must be a session date at 00:00 UTC, got {timestamp}. "
            "Intraday boundaries could admit labels whose closing price is not yet known."
        )
    return timestamp


def _require_session_dates(values: pd.Series, name: str) -> None:
    if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dtype.tz) != "UTC":
        raise ValueError(f"{name} must be tz-aware UTC datetimes; got {values.dtype}.")
    present = values.dropna()
    if not present.eq(present.dt.normalize()).all():
        raise ValueError(f"{name} must contain session dates at 00:00 UTC only.")


def is_labeled(frame: pd.DataFrame) -> np.ndarray:
    """Rows with a finite target and a known label end date."""

    returns = frame[TARGET_RETURN_COLUMN].to_numpy(dtype=np.float64)
    return np.isfinite(returns) & frame[TARGET_END_COLUMN].notna().to_numpy()


def validate_labeled_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Labeled frame is missing required columns: {missing}")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Labeled frame must have a default RangeIndex (use reset_index).")
    for column in ("date", TARGET_START_COLUMN, TARGET_END_COLUMN):
        _require_session_dates(frame[column], column)
    if frame["symbol"].isna().any():
        raise ValueError("Labeled frame contains missing symbols.")
    if frame.duplicated(subset=["date", "symbol"]).any():
        raise ValueError("Labeled frame contains duplicate (date, symbol) rows.")
    labeled = is_labeled(frame)
    dates = datetime_ns(frame["date"])[labeled]
    starts = datetime_ns(frame[TARGET_START_COLUMN])[labeled]
    ends = datetime_ns(frame[TARGET_END_COLUMN])[labeled]
    if frame[TARGET_START_COLUMN].isna().to_numpy()[labeled].any() or (starts <= dates).any():
        raise ValueError(
            "Every labeled row must enter after its feature date (target_start_date > date); "
            "same-close labels are rejected. Build labels with MODEL_LABEL_SPEC (entry_lag=1)."
        )
    if (ends <= starts).any():
        raise ValueError("Every labeled row must have target_end_date after target_start_date.")


def restrict_to_symbols(frame: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    """
    Keep only the modeled asset universe (e.g. config assets).

    Benchmark index rows must not enter the cross-section: they are price-
    return series and are built from the same stocks. Every requested symbol
    must be present.
    """

    wanted = list(symbols)
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("symbols must be a non-empty list without duplicates.")
    missing = sorted(set(wanted) - set(frame["symbol"]))
    if missing:
        raise ValueError(f"Requested symbols are absent from the frame: {missing}")
    return frame.loc[frame["symbol"].isin(wanted)].reset_index(drop=True)


def make_test_windows(
    session_dates: Iterable[object],
    *,
    start: object,
    end: object,
    block_sessions: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Consecutive half-open test windows of `block_sessions` sessions each,
    covering the sessions in [start, end). Boundaries are actual sessions,
    except the last window, which ends at `end`.
    """

    if isinstance(block_sessions, bool) or not isinstance(block_sessions, int):
        raise TypeError("block_sessions must be an integer.")
    if block_sessions <= 0:
        raise ValueError("block_sessions must be positive.")
    start_ts = _session_timestamp(start, "start")
    end_ts = _session_timestamp(end, "end")
    if start_ts >= end_ts:
        raise ValueError("start must be before end.")

    sessions = (
        pd.DatetimeIndex(pd.to_datetime(list(session_dates), utc=True)).unique().sort_values()
    )
    sessions = sessions[(sessions >= start_ts) & (sessions < end_ts)]
    if sessions.empty:
        raise ValueError("No sessions fall inside [start, end).")

    starts = list(sessions[::block_sessions])
    return list(zip(starts, [*starts[1:], end_ts], strict=True))


def _validate_windows(
    windows: Sequence[tuple[object, object]],
    *,
    mode: str,
    holdout_start: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        raise ValueError("At least one test window is required.")
    parsed = [
        (_session_timestamp(a, "test_start"), _session_timestamp(b, "test_end")) for a, b in windows
    ]
    for (start, end), following in zip(parsed, [*parsed[1:], None], strict=True):
        if start >= end:
            raise ValueError(f"Test window [{start}, {end}) is empty or reversed.")
        if following is not None and following[0] < end:
            raise ValueError("Test windows must be sorted and non-overlapping.")
        if mode == "development" and end > holdout_start:
            raise ValueError(
                f"Development window [{start}, {end}) reaches into the holdout "
                f"starting {holdout_start}. Use mode='holdout' to evaluate the lockbox."
            )
        if mode == "holdout" and start < holdout_start:
            raise ValueError(f"Holdout window [{start}, {end}) starts before {holdout_start}.")
    return parsed


def _ns(timestamp: pd.Timestamp) -> int:
    return pd.Timestamp(timestamp).as_unit("ns").value


def _positions(mask: np.ndarray) -> np.ndarray:
    positions = np.flatnonzero(mask).astype(np.int64)
    positions.setflags(write=False)
    return positions


def make_expanding_folds(
    labeled: pd.DataFrame,
    test_windows: Sequence[tuple[object, object]],
    *,
    holdout_start: object,
    mode: str = "development",
    train_start: object | None = None,
    lockbox_start: object = MODEL_HOLDOUT_START,
) -> tuple[WalkForwardFold, ...]:
    """
    Build purged expanding walk-forward folds (rules in the module docstring).

    `labeled` needs date, symbol, target_return, target_start_date and
    target_end_date, with session dates at 00:00 UTC and a default
    RangeIndex. Rows without a finite label are never trained or scored;
    they are counted per fold. `train_start` optionally drops older rows from
    every training set. In development mode `holdout_start` may be earlier
    than `lockbox_start` (e.g. inner folds of nested selection), never later.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}.")
    validate_labeled_frame(labeled)
    holdout = _session_timestamp(holdout_start, "holdout_start")
    if mode == "development" and holdout > _session_timestamp(lockbox_start, "lockbox_start"):
        raise ValueError(
            f"Development holdout_start {holdout} is later than the lockbox start "
            f"{lockbox_start}: development folds may not use lockbox data."
        )
    floor = None if train_start is None else _session_timestamp(train_start, "train_start")
    windows = _validate_windows(test_windows, mode=mode, holdout_start=holdout)

    # UTC int64 nanoseconds; NaT label ends only occur on unlabeled rows,
    # which every rule below masks out.
    dates = datetime_ns(labeled["date"])
    ends = datetime_ns(labeled[TARGET_END_COLUMN])
    labeled_mask = is_labeled(labeled)
    in_scope = np.ones(len(labeled), dtype=bool) if floor is None else dates >= _ns(floor)
    rows_sha256 = frame_rows_sha256(labeled)

    folds = []
    for fold_id, (test_start, test_end) in enumerate(windows):
        start = _ns(test_start)
        end = _ns(test_end)

        matured = labeled_mask & (ends < start)
        overlapping = labeled_mask & (dates < start) & (ends >= start)
        in_window = labeled_mask & (dates >= start) & (dates < end)
        if mode == "development":
            held_out = in_window & (ends >= _ns(holdout))
        else:
            held_out = np.zeros(len(labeled), dtype=bool)

        train = matured & in_scope
        test = in_window & ~held_out
        if not train.any():
            raise ValueError(f"Fold {fold_id} [{test_start}, {test_end}) has no training rows.")
        if not test.any():
            raise ValueError(f"Fold {fold_id} [{test_start}, {test_end}) has no scorable rows.")

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                mode=mode,
                holdout_start=holdout,
                train_start=floor,
                test_start=test_start,
                test_end=test_end,
                train_rows=_positions(train),
                test_rows=_positions(test),
                purged_rows=_positions(overlapping & in_scope),
                unlabeled_rows=_positions(~labeled_mask & in_scope & (dates < end)),
                held_out_rows=_positions(held_out),
                frame_rows_sha256=rows_sha256,
            )
        )

    return tuple(folds)
