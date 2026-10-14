"""
Point-in-time selection of fundamental observations.

An observation carries `period_end` and `available_at` (when the filing, or
its amendment, could first be used). At a knowledge time, the current
observation of a symbol is the latest period known by then and that
period's latest version known by then: a late amendment of an older period
never becomes current. This is the as-of rule of
data.fundamentals.point_in_time, restated for plain observation frames.

A row dated d (a session at 00:00 UTC) may use an observation only if its
available_at <= d at 00:00 UTC (the panel rule in features.panel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def current_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    The change log of each symbol's current observation: at each
    availability time, the latest period known so far and that period's
    latest version. A late amendment of an older period never becomes
    current (the as-of rule of fundamentals.point_in_time).
    """

    ordered = fundamentals.sort_values(["symbol", "available_at", "period_end"], kind="mergesort")
    if ordered.duplicated(["symbol", "available_at", "period_end"]).any():
        raise ValueError("Two versions of one period share an availability time (ambiguous).")
    latest_period = ordered.groupby("symbol")["period_end"].cummax()
    current = ordered.loc[ordered["period_end"] == latest_period]
    return current.drop_duplicates(["symbol", "available_at"], keep="last").reset_index(drop=True)


def asof_fundamentals(keys: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, symbol) key, the current observation with
    available_at <= date (00:00 UTC of the session): `keys` order is kept,
    unmatched keys get NaN values and NaT timestamps.
    """

    log = current_fundamentals(fundamentals)
    left = keys.loc[:, ["date", "symbol"]].copy()
    left["_order"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"], utc=True).dt.as_unit("ns")
    right = log.rename(columns={"available_at": "fundamentals_available_at"}).copy()
    right["fundamentals_available_at"] = right["fundamentals_available_at"].dt.as_unit("ns")
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("fundamentals_available_at"),
        left_on="date",
        right_on="fundamentals_available_at",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)
