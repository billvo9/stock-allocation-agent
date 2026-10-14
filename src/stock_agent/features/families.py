"""
Feature families built on a frozen eligible universe, each with its contract.

    relative_technical   past returns of the symbol, ranked within the date
                         among eligible names (session_close)
    risk_liquidity       realized volatility and Amihud illiquidity from the
                         symbol's own prices and volumes (session_close)
    fundamental_quality  ratios from the latest filing available at 00:00 UTC
                         of the row date (per_row_timestamp)

`build_feature_frame` returns one row per eligible (date, symbol) key,
whatever the features' missingness, with the requested families' columns,
`fundamentals_available_at` when fundamentals are used, and
`history_sessions` (the symbol's price rows up to the date, for
missingness diagnostics). Every value at (d, s) uses only s's price rows
dated <= d, the same-date cross-section of eligible names, and filings
available by d; prefix stability is tested per family.

Prices are raw closes and volumes of a synthetic or split-free source.
Vendor-adjusted levels embed later corporate actions; same-symbol ratios
cancel them, but Amihud's dollar volume does not (see ml_features.md).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_agent.features.asof import asof_fundamentals
from stock_agent.features.contracts import (
    PER_ROW_TIMESTAMP,
    SESSION_CLOSE,
    FeatureContract,
    contract_index,
)
from stock_agent.features.frozen_universe import FrozenUniverse, eligible_keys

FUNDAMENTALS_AVAILABLE_AT = "fundamentals_available_at"
HISTORY_COLUMN = "history_sessions"

CONTRACTS: tuple[FeatureContract, ...] = (
    FeatureContract(
        "rel_return_5d",
        "relative_technical",
        SESSION_CLOSE,
        5,
        "impute_training_median",
        ("close",),
        "5-session return, ranked within the date among eligible names (-0.5 to 0.5)",
        cross_sectional="within_date_rank",
    ),
    FeatureContract(
        "rel_momentum_20d",
        "relative_technical",
        SESSION_CLOSE,
        20,
        "impute_training_median",
        ("close",),
        "20-session return, ranked within the date among eligible names",
        cross_sectional="within_date_rank",
    ),
    FeatureContract(
        "rel_momentum_120_20d",
        "relative_technical",
        SESSION_CLOSE,
        120,
        "impute_training_median",
        ("close",),
        "return from 120 to 20 sessions ago (skips the latest month), ranked within the date",
        cross_sectional="within_date_rank",
    ),
    FeatureContract(
        "volatility_20d",
        "risk_liquidity",
        SESSION_CLOSE,
        20,
        "impute_training_median",
        ("close",),
        "standard deviation (ddof 1) of the last 20 daily log returns",
    ),
    FeatureContract(
        "volatility_60d",
        "risk_liquidity",
        SESSION_CLOSE,
        60,
        "impute_training_median",
        ("close",),
        "standard deviation (ddof 1) of the last 60 daily log returns",
    ),
    FeatureContract(
        "log_amihud_20d",
        "risk_liquidity",
        SESSION_CLOSE,
        20,
        "impute_training_median",
        ("close", "volume"),
        "log of the mean over 20 sessions of |log return| / (close x volume) x 1e9",
    ),
    FeatureContract(
        "quality_gross_profitability",
        "fundamental_quality",
        PER_ROW_TIMESTAMP,
        0,
        "impute_training_median",
        ("gross_profit", "total_assets"),
        "gross profit / total assets of the latest period known at 00:00 UTC of the date",
        availability_column=FUNDAMENTALS_AVAILABLE_AT,
    ),
    FeatureContract(
        "quality_leverage",
        "fundamental_quality",
        PER_ROW_TIMESTAMP,
        0,
        "impute_training_median",
        ("total_liabilities", "total_assets"),
        "total liabilities / total assets of the latest period known at 00:00 UTC of the date",
        availability_column=FUNDAMENTALS_AVAILABLE_AT,
    ),
)
CONTRACTS_BY_NAME = contract_index(CONTRACTS)


def family_features(families: Sequence[str]) -> list[str]:
    """Feature names of the requested families, in contract order."""

    unknown = sorted(set(families) - {c.family for c in CONTRACTS})
    if unknown or not families:
        raise ValueError(f"Unknown or empty feature families: {unknown or families}")
    return [c.name for c in CONTRACTS if c.family in set(families)]


def _within_date_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    """(rank - 1) / (n - 1) - 0.5 among the date's finite values; NaN if n < 2."""

    ranks = values.groupby(dates).rank(method="average")
    counts = values.notna().groupby(dates).transform("sum")
    scaled = (ranks - 1.0) / (counts - 1.0) - 0.5
    return scaled.where(counts >= 2)


def _price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.loc[:, ["date", "symbol", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.as_unit("ns")
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    by_symbol = frame.groupby("symbol", sort=False)
    close = frame["close"].astype("float64")
    frame[HISTORY_COLUMN] = by_symbol.cumcount() + 1
    log_return = np.log(close / by_symbol["close"].shift(1))
    frame["_ret_5"] = close / by_symbol["close"].shift(5) - 1.0
    frame["_ret_20"] = close / by_symbol["close"].shift(20) - 1.0
    frame["_ret_120_20"] = by_symbol["close"].shift(20) / by_symbol["close"].shift(120) - 1.0
    returns = log_return.groupby(frame["symbol"], sort=False)
    frame["volatility_20d"] = returns.rolling(20, min_periods=20).std(ddof=1).to_numpy()
    frame["volatility_60d"] = returns.rolling(60, min_periods=60).std(ddof=1).to_numpy()
    dollar_volume = close * frame["volume"].astype("float64")
    impact = (log_return.abs() / dollar_volume.where(dollar_volume > 0)) * 1e9
    mean_impact = (
        impact.groupby(frame["symbol"], sort=False).rolling(20, min_periods=20).mean().to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_amihud_20d"] = np.where(mean_impact > 0, np.log(mean_impact), np.nan)
    return frame


def build_feature_frame(
    prices: pd.DataFrame,
    universe: FrozenUniverse,
    families: Sequence[str],
    *,
    fundamentals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, FeatureContract]]:
    """
    One row per eligible key with the requested families' features.
    Returns (frame, contracts of the returned features).
    """

    names = family_features(families)
    keys = eligible_keys(universe)
    computed = _price_features(prices)
    frame = keys.merge(
        computed.drop(columns=["close", "volume"]),
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    dates = frame["date"]
    frame["rel_return_5d"] = _within_date_rank(frame["_ret_5"], dates)
    frame["rel_momentum_20d"] = _within_date_rank(frame["_ret_20"], dates)
    frame["rel_momentum_120_20d"] = _within_date_rank(frame["_ret_120_20"], dates)
    frame = frame.drop(columns=["_ret_5", "_ret_20", "_ret_120_20"])

    extra = [HISTORY_COLUMN]
    if "fundamental_quality" in families:
        if fundamentals is None:
            raise ValueError("The fundamental_quality family needs fundamentals.")
        observed = asof_fundamentals(keys, fundamentals)
        assets = observed["total_assets"].astype("float64")
        frame["quality_gross_profitability"] = (observed["gross_profit"] / assets).where(assets > 0)
        frame["quality_leverage"] = (observed["total_liabilities"] / assets).where(assets > 0)
        frame[FUNDAMENTALS_AVAILABLE_AT] = pd.to_datetime(
            observed[FUNDAMENTALS_AVAILABLE_AT], utc=True
        ).dt.as_unit("ns")
        extra.append(FUNDAMENTALS_AVAILABLE_AT)
    result = frame.loc[:, ["date", "symbol", *names, *extra]].reset_index(drop=True)
    return result, {name: CONTRACTS_BY_NAME[name] for name in names}
