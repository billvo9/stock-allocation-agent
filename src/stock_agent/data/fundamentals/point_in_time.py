from __future__ import annotations

import pandas as pd

from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)

MARKET_KEY_COLUMNS = [
    "date",
    "symbol",
]


def _validate_market_frame(
    frame: pd.DataFrame,
) -> None:
    missing = [column for column in MARKET_KEY_COLUMNS if column not in frame.columns]

    if missing:
        raise ValueError(f"Market frame is missing required columns: {missing}")

    if frame.empty:
        raise ValueError("Market frame cannot be empty.")

    dates = pd.to_datetime(
        frame["date"],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError("Market frame contains invalid dates.")

    if frame["symbol"].isna().any():
        raise ValueError("Market frame contains missing symbols.")


def align_quarterly_fundamentals_asof(
    market_frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align the latest available quarterly
    fundamentals to each asset-date row.

    Only reports whose available_at is less
    than or equal to the market date may be used.
    """

    _validate_market_frame(market_frame)

    validate_quarterly_fundamental_frame(fundamentals)

    market = market_frame.copy()

    market["date"] = pd.to_datetime(
        market["date"],
        utc=True,
    )

    market["_row_order"] = range(len(market))

    usable = fundamentals[fundamentals["available_at"].notna()].copy()

    usable["available_at"] = pd.to_datetime(
        usable["available_at"],
        utc=True,
    )

    usable["period_end"] = pd.to_datetime(
        usable["period_end"],
        utc=True,
    )

    if usable.duplicated(
        subset=[
            "symbol",
            "available_at",
        ]
    ).any():
        raise ValueError(
            "Fundamental data contain multiple rows for the same (symbol, available_at)."
        )

    aligned_frames: list[pd.DataFrame] = []

    for symbol, market_group in market.groupby(
        "symbol",
        sort=False,
    ):
        left = market_group.sort_values("date").copy()

        right = usable[usable["symbol"] == symbol].sort_values("available_at").copy()

        if right.empty:
            result = left.copy()

            for column in QUARTERLY_FUNDAMENTAL_COLUMNS:
                if column == "symbol":
                    continue

                if column not in result.columns:
                    result[column] = pd.NA

        else:
            right = right.drop(columns=["symbol"])

            result = pd.merge_asof(
                left,
                right,
                left_on="date",
                right_on="available_at",
                direction="backward",
                allow_exact_matches=True,
            )

        aligned_frames.append(result)

    aligned = pd.concat(
        aligned_frames,
        ignore_index=True,
    )

    aligned["fundamental_available"] = aligned["period_end"].notna()

    aligned["fundamental_age_days"] = (
        (aligned["date"] - aligned["available_at"]).dt.total_seconds().div(86_400)
    )

    return aligned.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
