from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_TRAIN_START = pd.Timestamp("2016-01-01")
DEFAULT_TRAIN_END = pd.Timestamp("2024-12-31")
DEFAULT_VALIDATION_START = pd.Timestamp("2025-01-01")
DEFAULT_VALIDATION_END = pd.Timestamp("2025-12-31")
DEFAULT_TEST_START = pd.Timestamp("2026-01-01")


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def add_forward_return_target(
    frame: pd.DataFrame,
    horizon: int = 20,
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """
    Add a forward-return target independently for each asset.

    target_return:
        price(t + horizon) / price(t) - 1

    target_end_date records when the forward return becomes observable.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

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

    future_price = grouped[price_column].shift(-horizon)

    result["target_return"] = future_price / result[price_column] - 1.0

    result["target_end_date"] = grouped["date"].shift(-horizon)

    return result


def select_training_rows_asof(
    frame: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Return rows whose forward-return labels were fully known
    before the specified training date.

    This prevents future target information from leaking into training.
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
