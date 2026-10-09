"""
Placebo feature transforms that run through the same harness as any model.

Each returns a copy of the labeled frame with the same rows, keys and order
(so existing folds still bind) and the same labels; only feature values
move, and only between rows that were already observable. The estimator is
then fitted and evaluated on the placebo features exactly as on the real
ones.

block_permuted_features (null)
    Within each date, every row receives the feature vector of another
    symbol on the SAME date. The symbol mapping is held constant over
    blocks of `block_length` sessions, so each date's cross-section (the
    multiset of feature vectors) and the series' persistence within a block
    are preserved; only the link between a symbol's features and its own
    later returns is broken. Rows are never shuffled independently and
    values never cross dates, so no future information can enter. The
    mapping is drawn from SHA-256-derived keys of (seed, draw, block,
    symbol): label-blind, prefix-stable, independent of frame length.

    Caveat: the null assumes symbols are interchangeable. On a
    hindsight-selected universe a static tilt (always ranking the past
    winner first) looks significant against it; the static-ordering table
    in scoring shows how much of an IC a fixed ranking can produce.

stale_features (timeliness control, not a null)
    Row (t, s) receives the features of (t - k sessions, s): the same
    symbol's own older features, shifted backward only, never wrapped.
    It keeps symbol identity and any static tilt but removes timing, so
    IC(model) - IC(stale) asks whether the signal is timely. Rows without a
    session k back get NaN features; the harness refuses non-finite
    predictions rather than silently dropping them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_agent.model_diagnostics.seeds import unit_uniform
from stock_agent.model_validation.audit import datetime_ns

DEFAULT_PLACEBO_BLOCK_LENGTH = 63


def _validate(labeled: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    columns = list(feature_columns)
    missing = [column for column in ("date", "symbol", *columns) if column not in labeled.columns]
    if missing:
        raise ValueError(f"Frame is missing columns: {missing}")
    if not columns:
        raise ValueError("feature_columns must not be empty.")
    if labeled.duplicated(subset=["date", "symbol"]).any():
        raise ValueError("Frame contains duplicate (date, symbol) rows.")
    return columns


def session_positions(labeled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(sorted unique session dates as int64 ns, each row's session index)."""

    sessions, index = np.unique(datetime_ns(labeled["date"]), return_inverse=True)
    return sessions, index.astype(np.int64)


def block_ids(
    labeled: pd.DataFrame, *, block_length: int, anchor: object | None = None
) -> np.ndarray:
    """
    Block number of each row's session: floor((session - anchor) / length).

    Blocks are counted forward from `anchor` (default: the first session),
    never back from the last date, so adding later rows never moves an
    earlier block boundary. Sessions before the anchor get negative blocks.
    """

    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    sessions, index = session_positions(labeled)
    if anchor is None:
        anchor_index = 0
    else:
        anchor_ns = datetime_ns(pd.Series([pd.Timestamp(anchor)]))[0]
        anchor_index = int(np.searchsorted(sessions, anchor_ns))
    return np.floor_divide(index - anchor_index, block_length)


def block_permuted_features(
    labeled: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int,
    draw: int,
    block_length: int = DEFAULT_PLACEBO_BLOCK_LENGTH,
    anchor: object | None = None,
) -> pd.DataFrame:
    """Within-date, block-constant symbol permutation of the feature vectors."""

    columns = _validate(labeled, feature_columns)
    blocks = block_ids(labeled, block_length=block_length, anchor=anchor)
    symbols = labeled["symbol"].astype(str).to_numpy()
    cache: dict[tuple[int, str], float] = {}
    keys = np.empty(len(labeled), dtype=np.float64)
    for row, pair in enumerate(zip(blocks.tolist(), symbols.tolist(), strict=True)):
        if pair not in cache:
            cache[pair] = unit_uniform(seed, "block_permutation", draw, *pair)
        keys[row] = cache[pair]

    dates = datetime_ns(labeled["date"])
    by_name = np.lexsort((symbols, dates))
    by_key = np.lexsort((keys, dates))
    # Both orders group rows by date identically; within a date, the symbol
    # with the i-th smallest name takes the features of the row with the
    # i-th smallest key.
    source = np.empty(len(labeled), dtype=np.int64)
    source[by_name] = by_key

    result = labeled.copy()
    for column in columns:
        result[column] = labeled[column].to_numpy()[source]
    return result


def stale_features(
    labeled: pd.DataFrame, feature_columns: Sequence[str], *, lag_sessions: int
) -> pd.DataFrame:
    """Each row's features from the same symbol `lag_sessions` sessions earlier."""

    columns = _validate(labeled, feature_columns)
    if isinstance(lag_sessions, bool) or not isinstance(lag_sessions, int) or lag_sessions < 1:
        raise ValueError("lag_sessions must be a positive integer (backward shifts only).")
    _, index = session_positions(labeled)
    lookup = pd.DataFrame(
        {"session": index, "symbol": labeled["symbol"].to_numpy(), "row": np.arange(len(labeled))}
    )
    wanted = pd.DataFrame(
        {"session": index - lag_sessions, "symbol": labeled["symbol"].to_numpy()}
    ).merge(lookup, on=["session", "symbol"], how="left", validate="many_to_one")
    source = wanted["row"].to_numpy(dtype=np.float64)
    found = np.isfinite(source)

    result = labeled.copy()
    for column in columns:
        values = np.full(len(labeled), np.nan, dtype=np.float64)
        values[found] = labeled[column].to_numpy(dtype=np.float64)[source[found].astype(np.int64)]
        result[column] = values
    return result
