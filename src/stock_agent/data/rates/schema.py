from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RATE_COLUMNS = [
    "date",
    "rate_id",
    "provider_series_id",
    "currency",
    "tenor",
    "annual_yield",
    "quote_convention",
    "source",
]


@dataclass(frozen=True)
class RateSeriesSpec:
    rate_id: str
    provider_series_id: str
    currency: str
    tenor: str
    quote_convention: str
    provider_release_id: int | None = None


def validate_rate_frame(
    frame: pd.DataFrame,
) -> None:
    missing_columns = set(RATE_COLUMNS) - set(frame.columns)

    if missing_columns:
        raise ValueError(f"Rate data missing columns: {sorted(missing_columns)}")

    if frame.empty:
        raise ValueError("Rate data is empty")

    if frame.isna().any(axis=None):
        raise ValueError("Rate data has missing values")

    if frame.duplicated(subset=["date", "rate_id"]).any():
        raise ValueError("Rate data contains duplicate (date, rate_id) rows.")
