from __future__ import annotations

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

        target_return = price(t + entry_lag + horizon) / price(t + entry_lag) - 1

    The fields have no defaults so every experiment states its timing.
    """

    horizon: int
    entry_lag: int
    price_column: str = "adjusted_close"

    def __post_init__(self) -> None:
        _require_int(self.horizon, "horizon", minimum=1)
        _require_int(self.entry_lag, "entry_lag", minimum=0)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return add_forward_return_target(
            frame,
            horizon=self.horizon,
            price_column=self.price_column,
            entry_lag=self.entry_lag,
        )


# Label convention for model evaluation (owner decision, 2026-10-06): enter at
# the close after the feature date. End-of-day data arrives after the close,
# so trading at the observed close is infeasible, and some macro series are
# published after the close on their release dates.
MODEL_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1)


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
