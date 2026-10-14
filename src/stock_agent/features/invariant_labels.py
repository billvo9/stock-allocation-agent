"""
Labels over a frozen eligible universe, and the prefix-invariance check.

`frozen_universe_labels` computes LabelSpec labels for the eligible keys of
a frozen universe:

- the raw forward return of each symbol uses its own price rows
  (features.training.add_forward_return_target: entry `entry_lag` rows after
  the feature date, exit `horizon` rows after entry);
- with benchmark="equal_weight", the excess return is taken over C_t = the
  universe's eligible members on t, through the T4 function
  add_equal_weight_excess_target applied to the eligible rows only. Its
  availability and fail-closed rules are unchanged: every row on t takes
  the latest member end and earliest member start, and if any eligible
  member lacks a finite raw label (for example it delisted before its exit)
  or fewer than min_cross_section members exist, every row on t is
  unlabeled.

The only difference from the T4 label is the definition of C_t: eligible
members of a point-in-time universe instead of "names with a feature row on
t". It is used by the synthetic experiment runner; applying it to real data
needs the owner's approval (label semantics, "Mandatory before
large-universe validation" in docs/research/linear_models.md).

`assert_labels_prefix_invariant` is the invariant: later data never changes
an earlier label. For every cutoff T, building from only the raw rows dated
before T gives
- the same label, entry and exit for every row it labels, and
- a label for every row whose full-data label ends before T (a matured label
  never depends on later data).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from stock_agent.features.frozen_universe import FrozenUniverse, eligible_keys
from stock_agent.features.training import (
    RAW_TARGET_RETURN_COLUMN,
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    LabelSpec,
    add_equal_weight_excess_target,
    add_forward_return_target,
)
from stock_agent.model_validation.audit import LeakageError

LABEL_COLUMNS = (TARGET_RETURN_COLUMN, TARGET_START_COLUMN, TARGET_END_COLUMN)


def frozen_universe_labels(
    prices: pd.DataFrame, universe: FrozenUniverse, spec: LabelSpec
) -> pd.DataFrame:
    """Labels for the universe's eligible keys: date, symbol and the label columns."""

    if spec.price_column != universe.rule.price_column:
        raise ValueError(
            f"Label price column {spec.price_column!r} differs from the universe's "
            f"{universe.rule.price_column!r}."
        )
    raw = add_forward_return_target(
        prices.loc[:, ["date", "symbol", spec.price_column]],
        horizon=spec.horizon,
        price_column=spec.price_column,
        entry_lag=spec.entry_lag,
    )
    raw["date"] = raw["date"].dt.as_unit("ns")
    keys = eligible_keys(universe)
    labeled = keys.merge(raw, on=["date", "symbol"], how="left", validate="one_to_one")
    if spec.benchmark is not None:
        symbols = sorted(universe.membership["symbol"].unique())
        labeled = add_equal_weight_excess_target(
            labeled, universe=symbols, min_cross_section=spec.min_cross_section
        )
    columns = ["date", "symbol", *LABEL_COLUMNS]
    if RAW_TARGET_RETURN_COLUMN in labeled.columns:
        columns.append(RAW_TARGET_RETURN_COLUMN)
    result = labeled.loc[:, columns].sort_values(["date", "symbol"]).reset_index(drop=True)
    for column in (TARGET_START_COLUMN, TARGET_END_COLUMN):
        result[column] = pd.to_datetime(result[column], utc=True).dt.as_unit("ns")
    return result


def _same(left: pd.Series, right: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(left.dtype):
        return (left.to_numpy() == right.to_numpy()) | (left.isna() & right.isna()).to_numpy()
    a = left.to_numpy(dtype=np.float64)
    b = right.to_numpy(dtype=np.float64)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def assert_labels_prefix_invariant(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    *,
    cutoffs: Iterable[object],
    time_column: str = "date",
) -> None:
    """Raise LeakageError unless later raw rows never change an earlier label."""

    full = build(raw.copy()).set_index(["date", "symbol"])
    times = pd.to_datetime(raw[time_column], utc=True)
    for cutoff in cutoffs:
        cut = pd.Timestamp(cutoff)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        partial = build(raw.loc[times < cut].copy()).set_index(["date", "symbol"])
        labeled = partial.loc[np.isfinite(partial[TARGET_RETURN_COLUMN].to_numpy(np.float64))]
        missing = labeled.index.difference(full.index)
        if len(missing):
            raise LeakageError(f"Prefix instability at {cut}: labeled rows vanish with more data.")
        aligned = full.loc[labeled.index]
        for column in LABEL_COLUMNS:
            if not _same(labeled[column], aligned[column]).all():
                raise LeakageError(
                    f"Prefix instability at {cut}: {column} of an earlier label changes when "
                    "later data is added (future information leaks into labels)."
                )
        matured = full.loc[
            np.isfinite(full[TARGET_RETURN_COLUMN].to_numpy(np.float64))
            & (full[TARGET_END_COLUMN] < cut).to_numpy()
        ]
        if len(matured.index.difference(labeled.index)):
            raise LeakageError(
                f"Prefix instability at {cut}: a label that matured before the cutoff is "
                "missing from the truncated build (it depends on later data)."
            )
