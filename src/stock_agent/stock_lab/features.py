"""
Deterministic, past-only features of one stock as of each session's close.

Point in time. The row for session d uses bars of d and earlier sessions
only; the return being predicted (of d+1 or later) never enters it. Every
window is a trailing window of sessions ending at d, computed on the
split-consistent level L (prices.py), so a split never looks like a crash.
Windows are strict: a window that is not complete (too early in the history,
or touching a session whose value is missing) gives a missing value, never a
value from a shorter window. Prefix stability (adding later bars never
changes an earlier row) is tested.

Features (FEATURE_VERSION; r is the price return, L the level):

    run_direction, run_length, stable_run, paused   the run in progress (runs.py)
    signed_run          +run_length rising, -run_length falling, 0 Stable
    up_count_20, stable_count_20, down_count_20      categories in the last 20
    longest_up_60, longest_down_60                   longest rising / falling
                                                     run reached in the last 60
    ret_k               L_d / L_(d-k) - 1, compounded from the k session returns
                        (missing when any is), for k in 1, 3, 5, 10, 20, 60
    vol_20, vol_60      standard deviation of r over 20 / 60 sessions (ddof 1)
    ewma_vol            sqrt of the exponentially weighted mean of r^2,
                        lambda 0.94 (RiskMetrics), from 20 returns on
    vol_ratio           vol_20 / vol_60 (is volatility rising?)
    vol_change_5        ewma_vol_d / ewma_vol_(d-5) - 1
    ma_gap_20, ma_gap_50  L_d / mean(L over 20 / 50 sessions) - 1
    ma_slope_20         SMA20_d / SMA20_(d-5) - 1
    drawdown_k          L_d / max(L over k sessions) - 1 for k in 20, 60, 252
    bounce_k            L_d / min(L over k sessions) - 1 for k in 20, 60
    momentum_accel      ret_5_d - ret_5_(d-5)
    rel_dollar_volume   close_d * volume_d / median of close * volume over the
                        20 sessions before d (dollar volume is split-invariant)
    gap                 open_d * split_factor_d / close_(d-1) - 1, only where a
                        genuine open exists
    band_over_vol       tau / ewma_vol: how wide the stable band is for this
                        stock now (large means many Stable sessions)
    ex_dividend         a dividend went ex on d (its return includes the drop)

Market or sector context is not included: it needs a source with its own
availability timestamps (a separate decision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import DEFAULT_BAND, Category, StableBand
from stock_agent.stock_lab.prices import on_calendar, session_returns
from stock_agent.stock_lab.runs import DEFAULT_RULE, RunRule, current_runs

FEATURE_VERSION = "stock_lab_features_v1"
RETURN_HORIZONS = (1, 3, 5, 10, 20, 60)
EWMA_LAMBDA = 0.94
EWMA_MIN_RETURNS = 20
LONGEST_WINDOW = 252  # the longest window: rows before it lack some features

NUMERIC_FEATURES = (
    "run_length",
    "stable_run",
    "paused",
    "signed_run",
    "up_count_20",
    "stable_count_20",
    "down_count_20",
    "longest_up_60",
    "longest_down_60",
    *(f"ret_{k}" for k in RETURN_HORIZONS),
    "vol_20",
    "vol_60",
    "ewma_vol",
    "vol_ratio",
    "vol_change_5",
    "ma_gap_20",
    "ma_gap_50",
    "ma_slope_20",
    "drawdown_20",
    "drawdown_60",
    "drawdown_252",
    "bounce_20",
    "bounce_60",
    "momentum_accel",
    "rel_dollar_volume",
    "gap",
    "band_over_vol",
)
FEATURE_COLUMNS = ("run_direction", *NUMERIC_FEATURES, "ex_dividend")


def _strict_rolling(values: pd.Series, window: int):
    return values.rolling(window, min_periods=window)


def ewma_volatility(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """
    sigma_t = sqrt(EW mean of r^2 through t), weights lambda^k on the k-th
    previous available return (missing returns are skipped, never filled);
    missing until EWMA_MIN_RETURNS returns have been seen.
    """

    squared = returns.pow(2)
    variance = squared.ewm(alpha=1.0 - lam, adjust=False, ignore_na=True).mean()
    seen = returns.notna().cumsum()
    return np.sqrt(variance.where(seen >= EWMA_MIN_RETURNS))


def build_features(
    bars: pd.DataFrame,
    *,
    band: StableBand = DEFAULT_BAND,
    rule: RunRule = DEFAULT_RULE,
    sessions=None,
) -> pd.DataFrame:
    """
    One row per session of `bars` (on `sessions` when given, so missing
    sessions become rows with missing features): `session`, the return
    columns of `prices.session_returns` and FEATURE_COLUMNS.
    """

    frame = on_calendar(bars, sessions)
    returns = session_returns(frame, band)
    r = returns["price_return"]
    level = returns["level"]
    categories = returns["category"]

    out = returns.copy()
    runs = current_runs(categories, rule)
    for column in runs.columns:
        out[column] = runs[column].to_numpy()
    sign = np.where(
        runs["run_direction"].eq(Category.INCREASE.value),
        1,
        np.where(runs["run_direction"].eq(Category.DECREASE.value), -1, 0),
    )
    out["signed_run"] = sign * runs["run_length"].to_numpy()

    known = categories.notna()
    for name, category in (
        ("up_count_20", Category.INCREASE),
        ("stable_count_20", Category.STABLE),
        ("down_count_20", Category.DECREASE),
    ):
        indicator = categories.eq(category.value).astype("float64").where(known)
        out[name] = _strict_rolling(indicator, 20).sum()
    for name, category in (
        ("longest_up_60", Category.INCREASE),
        ("longest_down_60", Category.DECREASE),
    ):
        lengths = runs["run_length"].where(runs["run_direction"].eq(category.value), 0)
        lengths = lengths.astype("float64").where(known)
        out[name] = _strict_rolling(lengths, 60).max()

    growth = np.log1p(r)
    for k in RETURN_HORIZONS:
        # Compounded over the k returns of the window, so a gap or a broken level
        # inside it makes the return missing (never one return across a gap).
        out[f"ret_{k}"] = np.expm1(_strict_rolling(growth, k).sum())
    out["vol_20"] = _strict_rolling(r, 20).std(ddof=1)
    out["vol_60"] = _strict_rolling(r, 60).std(ddof=1)
    out["ewma_vol"] = ewma_volatility(r)
    out["vol_ratio"] = out["vol_20"] / out["vol_60"]
    out["vol_change_5"] = out["ewma_vol"] / out["ewma_vol"].shift(5) - 1.0

    sma20 = _strict_rolling(level, 20).mean()
    sma50 = _strict_rolling(level, 50).mean()
    out["ma_gap_20"] = level / sma20 - 1.0
    out["ma_gap_50"] = level / sma50 - 1.0
    out["ma_slope_20"] = sma20 / sma20.shift(5) - 1.0
    for k in (20, 60, 252):
        out[f"drawdown_{k}"] = level / _strict_rolling(level, k).max() - 1.0
    for k in (20, 60):
        out[f"bounce_{k}"] = level / _strict_rolling(level, k).min() - 1.0
    out["momentum_accel"] = out["ret_5"] - out["ret_5"].shift(5)

    close = frame["close"]
    volume = frame["volume"].where(frame["volume"] > 0)
    dollar_volume = close * volume
    baseline = _strict_rolling(dollar_volume.shift(1), 20).median()
    out["rel_dollar_volume"] = dollar_volume / baseline

    previous_close = close.shift(1)
    out["gap"] = frame["open"] * frame["split_factor"] / previous_close - 1.0
    out["band_over_vol"] = band.tau / out["ewma_vol"]
    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["feature_version"] = FEATURE_VERSION
    return out


def latest_row(features: pd.DataFrame) -> pd.Series | None:
    """The row of the last session with a close; None when there is none."""

    present = features[features["close"].notna()]
    return None if present.empty else present.iloc[-1]
