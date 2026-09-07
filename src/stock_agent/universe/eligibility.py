from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "adjusted_close",
]


def build_dynamic_universe(
    frame: pd.DataFrame,
    min_price_observations: int = 60,
) -> pd.DataFrame:
    """
    Add point-in-time asset eligibility information.

    Each asset is evaluated independently using only
    observations available on or before that date.

    Assets are preserved even before they become
    investable.
    """

    # 1. Validate configuration.
    if min_price_observations <= 0:
        raise ValueError("min_price_observations must be positive.")

    # 2. Validate required columns.
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Dynamic-universe frame is missing required columns: {missing_columns}")

    result = frame.copy()

    # Preserve the caller's original row order.
    result["_row_id"] = range(len(result))

    # 3. Standardize dates.
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    )

    # 4. Reject invalid dates.
    if result["date"].isna().any():
        raise ValueError("Dynamic-universe frame contains invalid dates.")

    if result["symbol"].isna().any():
        raise ValueError("Dynamic-universe frame contains missing symbols.")

    # 5. Duplicate asset-date observations are
    # a data-quality error. Do not silently delete them.
    duplicate_mask = result.duplicated(
        subset=[
            "symbol",
            "date",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError("Dynamic-universe frame contains duplicate (symbol, date) rows.")

    # 6. Put observations into chronological order
    # before calculating historical availability.
    chronological = result.sort_values(
        [
            "symbol",
            "date",
        ]
    ).copy()

    price_values = pd.to_numeric(
        chronological["adjusted_close"],
        errors="coerce",
    )

    chronological["has_valid_price"] = price_values.notna() & price_values.gt(0)

    chronological["price_history_observations"] = (
        chronological["has_valid_price"].astype(int).groupby(chronological["symbol"]).cumsum()
    )

    # 7. Point-in-time history requirement.
    chronological["has_required_price_history"] = (
        chronological["price_history_observations"] >= min_price_observations
    )

    # 8. Version 1 eligibility rule.
    chronological["is_investable"] = (
        chronological["has_required_price_history"] & chronological["has_valid_price"]
    )

    # Return rows in the same order supplied by
    # the caller.
    result = chronological.sort_values("_row_id").drop(columns="_row_id").reset_index(drop=True)

    return result
