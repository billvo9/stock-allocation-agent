"""
Canonical fingerprints and fold metadata for reproducible validation runs.

Hashes use an explicit byte layout rather than pandas' internal hashing, so a
fingerprint does not depend on the pandas or numpy version, the datetime unit
or timezone representation, the NaN bit pattern, or the sign of zero.

- frame_keys_sha256: the set of (date, symbol) keys; row order ignored.
- frame_content_sha256: keys plus the given columns' values; row order
  ignored; any change to a hashed key or value changes it. The dtype family
  is part of the hash, so the same numbers stored as int and as float hash
  differently.
- frame_rows_sha256: the keys in the frame's current row order. Folds bind to
  it because they store row positions.

Nothing here reads a clock or writes files: callers decide where metadata is
stored, and identical inputs always produce identical metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    LabelSpec,
)

if TYPE_CHECKING:
    from stock_agent.model_validation.folds import WalkForwardFold

KEY_COLUMNS = ("date", "symbol")


class LeakageError(Exception):
    """A leakage invariant does not hold."""


DECISION_CONVENTION = (
    "Row date d is an exchange session date stored as 00:00 UTC. Features include "
    "the close of d; macro availability is date-level and EDGAR is joined as of "
    "00:00 UTC d. The label enters at the close of the entry_lag-th later session "
    "of the same symbol and exits horizon sessions after entry; it is known only "
    "after the close of target_end_date. A fold trains on rows whose "
    "target_end_date is strictly before test_start."
)


def datetime_ns(values: pd.Series) -> np.ndarray:
    """Datetimes as UTC int64 nanoseconds; NaT maps to numpy's NaT sentinel."""

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        utc = values.dt.tz_convert("UTC").dt.tz_localize(None)
    elif pd.api.types.is_datetime64_dtype(values.dtype):
        utc = values  # naive datetimes are taken as UTC
    else:
        utc = pd.to_datetime(values, utc=True).dt.tz_localize(None)
    return np.asarray(utc, dtype="datetime64[ns]").view(np.int64)


def _canonical_float(values: pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64) + 0.0  # -0.0 -> 0.0
    array[np.isnan(array)] = np.nan  # one canonical NaN bit pattern
    return array.astype("<f8")


def _text_bytes(values: pd.Series) -> bytes:
    """Sorted table of distinct strings plus one int64 code per row (-1 = missing)."""

    missing = values.isna().to_numpy()
    text = values.astype(object).where(~missing, "").astype(str).to_numpy(dtype=object)
    uniques = np.unique(text[~missing].astype(str)) if (~missing).any() else np.array([], str)
    codes = np.full(len(text), -1, dtype=np.int64)
    codes[~missing] = np.searchsorted(uniques, text[~missing].astype(str))
    table = b"".join(
        len(encoded).to_bytes(8, "little") + encoded
        for encoded in (str(value).encode("utf-8") for value in uniques)
    )
    return len(uniques).to_bytes(8, "little") + table + codes.astype("<i8").tobytes()


def _column_bytes(values: pd.Series) -> tuple[bytes, bytes]:
    dtype = values.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return b"datetime", datetime_ns(values).astype("<i8").tobytes()
    if pd.api.types.is_bool_dtype(dtype) and not values.isna().any():
        return b"bool", np.asarray(values, dtype=np.uint8).tobytes()
    if pd.api.types.is_integer_dtype(dtype) and not values.isna().any():
        return b"int", np.asarray(values, dtype=np.int64).astype("<i8").tobytes()
    if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return b"float", _canonical_float(values.astype("float64")).tobytes()
    return b"text", _text_bytes(values)


def _canonical_order(frame: pd.DataFrame) -> np.ndarray:
    """Row order by (date, symbol), independent of the input order."""

    dates = datetime_ns(frame["date"])
    symbols = frame["symbol"].astype(str).to_numpy()
    return np.lexsort((symbols, dates))


def _sha256_frame(
    frame: pd.DataFrame, columns: Sequence[str], *, canonical_order: bool = True
) -> str:
    ordered = frame.iloc[_canonical_order(frame)] if canonical_order else frame
    digest = hashlib.sha256()
    digest.update(f"rows={len(ordered)};canonical={canonical_order}".encode())
    for column in columns:
        tag, payload = _column_bytes(ordered[column])
        name = str(column).encode("utf-8")
        digest.update(len(name).to_bytes(8, "little") + name + tag + payload)
    return digest.hexdigest()


def frame_keys_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint of the frame's set of (date, symbol) keys (row order ignored)."""

    return _sha256_frame(frame, KEY_COLUMNS)


def frame_rows_sha256(frame: pd.DataFrame) -> str:
    """
    Fingerprint of the (date, symbol) keys in the frame's current row order.

    Folds store row positions, so they bind to this order-sensitive hash: a
    reordered copy of the frame has the same key set but different rows at
    each position.
    """

    return _sha256_frame(frame, KEY_COLUMNS, canonical_order=False)


def frame_content_sha256(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Fingerprint of the row keys plus the given columns' values."""

    extra = [column for column in columns if column not in KEY_COLUMNS]
    return _sha256_frame(frame, [*KEY_COLUMNS, *extra])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _label_columns(frame: pd.DataFrame) -> list[str]:
    columns = [TARGET_RETURN_COLUMN, TARGET_END_COLUMN]
    if TARGET_START_COLUMN in frame.columns:
        columns.insert(1, TARGET_START_COLUMN)
    return columns


def describe_folds(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    feature_columns: Sequence[str],
    availability_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """
    One JSON-safe metadata row per fold.

    Together with the run manifest, each row is enough to reconstruct the
    fold: the date boundaries and purge rule select the rows, and the key
    and content fingerprints prove the reconstruction used identical data.
    For each availability column (e.g. EDGAR `available_at`), the latest
    timestamp seen by the training and test rows is recorded.
    """

    feature_columns = list(feature_columns)
    availability_columns = list(availability_columns)
    content_columns = [*feature_columns, *_label_columns(labeled), *availability_columns]
    frame_content = frame_content_sha256(labeled, content_columns)
    frame_keys = frame_keys_sha256(labeled)
    frame_rows = frame_rows_sha256(labeled)

    rows = []
    for fold in folds:
        if fold.frame_rows_sha256 != frame_rows:
            raise LeakageError(
                f"Fold {fold.fold_id} was built from a different frame or row order; "
                "its metadata would describe the wrong rows."
            )
        train = labeled.iloc[fold.train_rows]
        test = labeled.iloc[fold.test_rows]
        purged = labeled.iloc[fold.purged_rows]
        row = {
            "fold_id": fold.fold_id,
            "mode": fold.mode,
            "holdout_start": _iso(fold.holdout_start),
            "train_start": _iso(fold.train_start),
            "test_start": _iso(fold.test_start),
            "test_end_exclusive": _iso(fold.test_end),
            "knowledge_cutoff": _iso(fold.test_start),
            "purge_rule": (
                "train iff target_return is finite, target_end_date < test_start, "
                "and date >= train_start when set; applied to all symbols"
            ),
            "embargo_sessions": 0,
            "n_train": len(train),
            "n_test": len(test),
            "n_purged": len(purged),
            "n_unlabeled": len(fold.unlabeled_rows),
            "n_test_label_in_holdout": len(fold.held_out_rows),
            "train_first_date": _iso(train["date"].min()),
            "train_last_date": _iso(train["date"].max()),
            "train_max_target_end_date": _iso(train[TARGET_END_COLUMN].max()),
            "test_first_date": _iso(test["date"].min()),
            "test_last_date": _iso(test["date"].max()),
            "score_available_at": _iso(test[TARGET_END_COLUMN].max()),
            "train_rows_by_symbol": _json(train["symbol"].value_counts().sort_index().to_dict()),
            "test_rows_by_symbol": _json(test["symbol"].value_counts().sort_index().to_dict()),
            "feature_columns": _json(feature_columns),
            "frame_keys_sha256": frame_keys,
            "frame_rows_sha256": fold.frame_rows_sha256,
            "frame_content_sha256": frame_content,
            "train_keys_sha256": frame_keys_sha256(train),
            "test_keys_sha256": frame_keys_sha256(test),
            "purged_keys_sha256": frame_keys_sha256(purged),
            "train_content_sha256": frame_content_sha256(train, content_columns),
            "test_content_sha256": frame_content_sha256(test, content_columns),
        }
        for column in availability_columns:
            row[f"train_max_{column}"] = _iso(train[column].max())
            row[f"test_max_{column}"] = _iso(test[column].max())
        rows.append(row)

    return pd.DataFrame(rows)


def run_manifest(
    *,
    label_spec: LabelSpec,
    holdout_start: pd.Timestamp,
    universe: Sequence[str],
    excluded_symbols: Sequence[str],
    feature_columns: Sequence[str],
    input_files: Mapping[str, str | Path],
    git_sha: str | None,
) -> dict[str, object]:
    """Run-level provenance: label timing, lockbox, universe, inputs, versions."""

    return {
        "decision_convention": DECISION_CONVENTION,
        "label_spec": asdict(label_spec),
        "holdout_start": _iso(holdout_start),
        "universe": sorted(universe),
        "excluded_symbols": sorted(excluded_symbols),
        "feature_columns": list(feature_columns),
        "input_files_sha256": {name: file_sha256(path) for name, path in input_files.items()},
        "git_sha": git_sha,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
