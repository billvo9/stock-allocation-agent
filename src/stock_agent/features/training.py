from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TRAIN_START = pd.Timestamp("2016-01-01")
DEFAULT_TRAIN_END = pd.Timestamp("2024-12-31")
DEFAULT_VALIDATION_START = pd.Timestamp("2025-01-01")
DEFAULT_VALIDATION_END = pd.Timestamp("2025-12-31")
DEFAULT_TEST_START = pd.Timestamp("2026-01-01")

TARGET_RETURN_COLUMN = "target_return"
TARGET_START_COLUMN = "target_start_date"
TARGET_END_COLUMN = "target_end_date"
# The raw forward return, kept beside an excess label (never a feature: the
# target_ prefix is refused by validate_feature_columns).
RAW_TARGET_RETURN_COLUMN = "target_return_raw"
BENCHMARKS = ("equal_weight",)


def _require_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}; got {value!r}.")
    return int(value)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class LabelSpec:
    """
    Explicit forward-return label definition.

    A row dated t carries features through the close of t. The position is
    entered at the close `entry_lag` observations later and held for
    `horizon` observations of the same symbol:

        raw return = price(t + entry_lag + horizon) / price(t + entry_lag) - 1

    With benchmark=None the label is the raw return. With
    benchmark="equal_weight" it is the raw return minus the equal-weight mean
    raw return of the universe's names on the same date (see
    add_equal_weight_excess_target), and the raw return is kept in
    target_return_raw. The timing fields have no defaults so every
    experiment states its timing.
    """

    horizon: int
    entry_lag: int
    price_column: str = "adjusted_close"
    benchmark: str | None = None
    min_cross_section: int = 2

    def __post_init__(self) -> None:
        _require_int(self.horizon, "horizon", minimum=1)
        _require_int(self.entry_lag, "entry_lag", minimum=0)
        _require_int(self.min_cross_section, "min_cross_section", minimum=2)
        if self.benchmark is not None and self.benchmark not in BENCHMARKS:
            raise ValueError(
                f"benchmark must be None or one of {BENCHMARKS}; got {self.benchmark!r}."
            )

    @property
    def label_id(self) -> str:
        timing = f"{self.price_column}:horizon={self.horizon}:entry_lag={self.entry_lag}"
        if self.benchmark is None:
            return f"forward_simple_return:{timing}"
        return (
            f"{self.benchmark}_excess_forward_simple_return:{timing}"
            f":min_cross_section={self.min_cross_section}"
        )

    def apply(self, frame: pd.DataFrame, *, universe: Sequence[str] | None = None) -> pd.DataFrame:
        labeled = add_forward_return_target(
            frame,
            horizon=self.horizon,
            price_column=self.price_column,
            entry_lag=self.entry_lag,
        )
        if self.benchmark is None:
            return labeled
        if universe is None:
            raise ValueError("An excess label needs the universe that defines its benchmark.")
        return add_equal_weight_excess_target(
            labeled, universe=universe, min_cross_section=self.min_cross_section
        )


# Label convention for model evaluation (owner decision, 2026-10-06): enter at
# the close after the feature date. End-of-day data arrives after the close,
# so trading at the observed close is infeasible, and some macro series are
# published after the close on their release dates.
MODEL_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1)
# Primary alpha target from T4 (owner decision, 2026-10-10): the point-in-time
# cross-sectional equal-weight excess return; the raw return stays stored.
MODEL_EXCESS_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1, benchmark="equal_weight")

EXCESS_AVAILABILITY_RULE = (
    "An equal-weight excess label on date t depends on every universe name present on t, so "
    "each row on t takes target_end_date = the latest member end date and target_start_date = "
    "the earliest member start date. If any present name lacks a finite raw label, or fewer "
    "than min_cross_section names are present, every row on t is unlabeled."
)


def add_equal_weight_excess_target(
    labeled: pd.DataFrame, *, universe: Sequence[str], min_cross_section: int = 2
) -> pd.DataFrame:
    """
    Replace the raw label with its point-in-time equal-weight excess.

    For date t, C_t = the universe names present on t (rows of the frame).
        target_return     = raw_i - mean over C_t of raw_j
        target_return_raw = raw_i
    Availability: the excess is known only once every member's label is
    known, so each row on t gets the latest member target_end_date and the
    earliest member target_start_date. Fail closed: if any present name has
    no finite raw label (for example a missing exit price), or |C_t| <
    min_cross_section, every row on t is unlabeled, so a missing price can
    never shrink the benchmark and truncation never changes a matured label.
    Symbols outside `universe` are refused (an index row must never enter
    the benchmark).
    """

    min_cross_section = _require_int(min_cross_section, "min_cross_section", minimum=2)
    members = set(universe)
    outside = sorted(set(labeled["symbol"]) - members)
    if outside:
        raise ValueError(f"Rows outside the benchmark universe: {outside}.")
    result = labeled.copy()
    raw = result[TARGET_RETURN_COLUMN].astype("float64")
    by_date = result.groupby("date", sort=False)
    complete = (
        by_date[TARGET_RETURN_COLUMN]
        .transform(lambda values: bool(np.isfinite(values.to_numpy(dtype="float64")).all()))
        .astype(bool)
    )
    size = by_date["symbol"].transform("size")
    valid = complete & (size >= min_cross_section)
    benchmark = by_date[TARGET_RETURN_COLUMN].transform("mean")
    end = by_date[TARGET_END_COLUMN].transform("max")
    start = by_date[TARGET_START_COLUMN].transform("min")
    result[RAW_TARGET_RETURN_COLUMN] = raw.where(valid)
    result[TARGET_RETURN_COLUMN] = (raw - benchmark).where(valid)
    result[TARGET_START_COLUMN] = start.where(valid)
    result[TARGET_END_COLUMN] = end.where(valid)
    return result


def add_forward_return_target(
    frame: pd.DataFrame,
    horizon: int = 20,
    price_column: str = "adjusted_close",
    entry_lag: int = 0,
) -> pd.DataFrame:
    """
    Add a forward-return target independently for each asset.

    target_return:
        price(t + entry_lag + horizon) / price(t + entry_lag) - 1

    Offsets count observations of the same symbol, never another symbol's
    rows. target_start_date is the entry observation's date and
    target_end_date the exit observation's date: the label is known only
    after the close of target_end_date. The default entry_lag=0 keeps the
    historical same-close convention; model evaluation uses MODEL_LABEL_SPEC.
    """

    horizon = _require_int(horizon, "horizon", minimum=1)
    entry_lag = _require_int(entry_lag, "entry_lag", minimum=0)

    required = {
        "date",
        "symbol",
        price_column,
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Training frame is missing required columns: {sorted(missing)}")

    result = frame.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    )

    if result["date"].isna().any():
        raise ValueError("Training frame contains invalid dates.")

    if result.duplicated(subset=["symbol", "date"]).any():
        raise ValueError("Training frame contains duplicate asset-date rows.")

    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = result.groupby(
        "symbol",
        sort=False,
    )

    entry_price = grouped[price_column].shift(-entry_lag)
    exit_price = grouped[price_column].shift(-(entry_lag + horizon))

    result[TARGET_RETURN_COLUMN] = exit_price / entry_price - 1.0

    result[TARGET_START_COLUMN] = grouped["date"].shift(-entry_lag)
    result[TARGET_END_COLUMN] = grouped["date"].shift(-(entry_lag + horizon))

    return result


def select_training_rows_asof(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Return rows whose forward-return labels were fully known
    before the specified training date.

    This prevents future target information from leaking into training.

    Pass a date, not an intraday time: target_end_date is a midnight-UTC
    market date whose label is known only after that day's close, so an
    intraday cutoff on that date would admit a label not yet observable.
    Model evaluation should use stock_agent.model_validation, which
    enforces date-level boundaries.
    """

    required = {
        "date",
        "target_return",
        "target_end_date",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Target frame is missing required columns: {sorted(missing)}")

    cutoff = pd.Timestamp(as_of)

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    result = frame.copy()

    result["target_end_date"] = pd.to_datetime(
        result["target_end_date"],
        errors="coerce",
        utc=True,
    )

    mask = (
        result["target_return"].notna()
        & result["target_end_date"].notna()
        & (result["target_end_date"] < cutoff)
    )

    return result.loc[mask].copy()


def split_temporal_dataset(
    frame: pd.DataFrame,
    *,
    train_start: str = "2016-01-01",
    train_end: str = "2024-12-31",
    validation_start: str = "2025-01-01",
    validation_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    test_end: str | None = None,
) -> TemporalSplit:
    """
    Split rows by feature date.

    This does not decide whether a label was available for training.
    Use select_training_rows_asof() for that purpose.

    Warning: this split does not purge. Training rows near train_end carry
    labels that end inside the validation period, and validation labels
    cross into the test period. Do not use it to evaluate models; use
    stock_agent.model_validation (purged expanding walk-forward folds).
    """

    result = frame.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    )

    if result["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")

    train_start_ts = pd.Timestamp(
        train_start,
        tz="UTC",
    )
    train_end_ts = pd.Timestamp(
        train_end,
        tz="UTC",
    )

    validation_start_ts = pd.Timestamp(
        validation_start,
        tz="UTC",
    )
    validation_end_ts = pd.Timestamp(
        validation_end,
        tz="UTC",
    )

    test_start_ts = pd.Timestamp(
        test_start,
        tz="UTC",
    )

    train = result.loc[
        result["date"].between(
            train_start_ts,
            train_end_ts,
        )
    ].copy()

    validation = result.loc[
        result["date"].between(
            validation_start_ts,
            validation_end_ts,
        )
    ].copy()

    test_mask = result["date"] >= test_start_ts

    if test_end is not None:
        test_end_ts = pd.Timestamp(
            test_end,
            tz="UTC",
        )
        test_mask &= result["date"] <= test_end_ts

    test = result.loc[test_mask].copy()

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
