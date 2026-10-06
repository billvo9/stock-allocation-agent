from __future__ import annotations

import pandas as pd

from stock_agent.data.fundamentals.quarterly_schema import (
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


def _current_state_change_log(
    usable: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce a versioned history to the rows that become the CURRENT state.

    A versioned history may publish a new version of an older fiscal
    period (a late amendment) after a newer period exists. That version
    is real later knowledge about the older period, but it is not the
    company's latest reported quarter, so it must never become the
    as-of row. A same-instant re-version of the newer period carries any
    effect the amendment has on it.

    Rule (a pure function of the row SET, independent of row order):
        1. Stable-sort by (symbol, available_at, period_end).
        2. At each (symbol, available_at) keep the max-period_end row.
        3. Keep only rows whose period_end equals the per-symbol running
           max of period_end over knowledge time.
    """

    ordered = usable.sort_values(
        ["symbol", "available_at", "period_end"],
        kind="mergesort",
    )

    latest_period_per_instant = ordered.drop_duplicates(
        subset=["symbol", "available_at"],
        keep="last",
    )

    running_max_period = latest_period_per_instant.groupby("symbol", sort=False)[
        "period_end"
    ].cummax()

    return latest_period_per_instant[latest_period_per_instant["period_end"].eq(running_max_period)]


def align_quarterly_fundamentals_asof(
    market_frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align the current quarterly fundamentals state to each asset-date row.

    Only rows whose available_at is less than or equal to the market date
    may be used (backward as-of join, exact matches allowed). The rows are
    first reduced to the current-state change log (see
    _current_state_change_log), so a late amendment of an older fiscal
    period never replaces a newer period.

    The output follows the market frame's row order and does not depend
    on the row order of either input.
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
            "period_end",
            "available_at",
        ]
    ).any():
        raise ValueError(
            "Fundamental data contain multiple rows for the same "
            "(symbol, period_end, available_at)."
        )

    current_state = _current_state_change_log(usable)

    # merge_asof needs one key dtype on both sides; a private key column
    # keeps the caller-visible date / available_at dtypes unchanged.
    market["_asof_key"] = market["date"].astype("datetime64[ns, UTC]")

    right = current_state.copy()
    right["_asof_key"] = right["available_at"].astype("datetime64[ns, UTC]")

    left = market.sort_values(["_asof_key", "_row_order"], kind="mergesort")
    right = right.sort_values(["_asof_key", "symbol"], kind="mergesort")

    aligned = pd.merge_asof(
        left,
        right,
        on="_asof_key",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )

    aligned["fundamental_available"] = aligned["period_end"].notna()

    aligned["fundamental_age_days"] = (
        (aligned["date"] - aligned["available_at"]).dt.total_seconds().div(86_400)
    )

    return (
        aligned.sort_values("_row_order")
        .drop(columns=["_row_order", "_asof_key"])
        .reset_index(drop=True)
    )
