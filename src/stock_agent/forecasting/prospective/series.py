"""
Per-asset series: approved F5.1 total returns, their daily log values, the
ETF track's features and the prospective track's two targets.

Daily log total return (from portfolio.descriptive.total_returns, OD8 group
A, approved 2026-10-14):

    r_t  = (f_t P_t + D_t) / P_{t-1} - 1     raw closes, as-paid cash, splits
    g_t  = log(1 + r_t)                       only when r_t is a one-session
                                              return (sessions_spanned == 1)

A multi-session return (a session without a close inside the span) or an
unavailable return gives g_t = NaN: it is never spread over the sessions it
spans and never filled.

Features at the close of session t (the ETF track's formulas,
forecasting.etf_data, on g instead of price-only log returns, so cash
distributions count):

    log_rv_k   = 0.5 log(mean_{j=t-k+1..t} g_j^2)               k = 5, 20, 60
    tsmom_k    = sum_{j=t-k+1..t} g_j / (sigma_60 sqrt(k))        k = 20, 60
    sigma_60   = sqrt(mean_{j=t-59..t} g_j^2)

Every window needs all k values (min_periods = k), so a missing return makes
the features that cover it NaN (imputed by the training median at fit time).

Targets of row t, with entry e = t + L and exit x = e + h in sessions:

    total_log_return         sum_{k=e+1..x} g_k
    log_realized_volatility  0.5 log(mean_{k=e+1..x} g_k^2)

Both need all h one-session returns; otherwise the row is unlabeled. They are
known after the exit close (target_end_date), like every label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
)
from stock_agent.portfolio import descriptive

FEATURES = ("tsmom_20d", "tsmom_60d", "log_rv_5d", "log_rv_20d", "log_rv_60d")
MIN_RETURNS = 60  # the longest feature window
UNADJUSTED = descriptive.UNADJUSTED
PER_SHARE_AS_PAID = descriptive.PER_SHARE_AS_PAID


class SeriesRefused(ValueError):
    """Inputs that the approved total-return formulas cannot use. `reason` is a code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _own(frame: pd.DataFrame, fund_id: str, columns: Sequence[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SeriesRefused("schema_mismatch", f"missing columns {missing}")
    return frame.loc[frame["fund_id"] == fund_id, list(columns)]


def _utc_dates(values: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")


def daily_log_returns(
    sessions: pd.DatetimeIndex,
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    fund_id: str,
) -> pd.DataFrame:
    """
    One row per session of `sessions`: date, close_observed, total_return,
    sessions_spanned, status, reason and log_return (g_t above).

    Only unadjusted closes are read; distribution amounts must be per share
    as paid (R1: nothing adjusted twice). F1 column names are accepted
    (`session_date`, `ex_date`, `effective_date`, `ratio_new_per_old`).
    """

    sessions = pd.DatetimeIndex(sessions).as_unit("ns")
    own = _own(prices, fund_id, ("session_date", "close", "adjustment_basis"))
    own = own.loc[own["adjustment_basis"] == UNADJUSTED]
    dates = _utc_dates(own["session_date"])
    if dates.has_duplicates:
        raise SeriesRefused(
            "multiple_vintages",
            f"{fund_id}: more than one unadjusted close for a session; choosing among "
            "vintages needs an approved rule",
        )
    closes = pd.Series(own["close"].to_numpy(dtype="float64"), index=dates).sort_index()
    paid = _own(distributions, fund_id, ("ex_date", "amount", "amount_basis"))
    if not (paid["amount_basis"] == PER_SHARE_AS_PAID).all():
        raise SeriesRefused(
            "amount_basis", f"{fund_id}: distribution amounts must be per share as paid"
        )
    # Several distributions on one ex-date (types, supplementals) are summed by total_returns.
    cash = pd.DataFrame(
        {"ex_date": _utc_dates(paid["ex_date"]), "amount": paid["amount"].astype("float64")}
    )
    split = _own(splits, fund_id, ("effective_date", "ratio_new_per_old"))
    ratios = pd.DataFrame(
        {
            "effective_date": _utc_dates(split["effective_date"]),
            "ratio": split["ratio_new_per_old"].astype("float64"),
        }
    )
    try:
        frame = descriptive.total_returns(
            sessions,
            closes,
            cash,
            ratios,
            close_basis=UNADJUSTED,
            amount_basis=PER_SHARE_AS_PAID,
        )
    except ValueError as error:
        raise SeriesRefused("total_return_inputs", f"{fund_id}: {error}") from None
    one_session = (frame["status"] == descriptive.OK) & (frame["sessions_spanned"] == 1)
    with np.errstate(invalid="ignore"):
        frame["log_return"] = np.where(one_session, np.log1p(frame["total_return"]), np.nan)
    frame["date"] = pd.DatetimeIndex(frame["date"]).as_unit("ns")
    return frame


def window_sums(values: object, k: int) -> np.ndarray:
    """
    At each position p, the sum of values[p-k+1..p], NaN when fewer than k
    values precede p or any of them is NaN. Each window is summed on its own
    (no running sum), so a value depends, bit for bit, only on its window.
    """

    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan)
    if len(array) >= k:
        result[k - 1 :] = np.lib.stride_tricks.sliding_window_view(array, k).sum(axis=1)
    return result


def features_from_log_returns(log_return: pd.Series) -> pd.DataFrame:
    """The five ETF-track features at every position of one asset's g series."""

    g = np.asarray(log_return, dtype=np.float64)
    vol_60 = np.sqrt(window_sums(g**2, 60) / 60)
    result = {}
    for k in (20, 60):
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"tsmom_{k}d"] = np.where(
                vol_60 > 0, window_sums(g, k) / (vol_60 * np.sqrt(k)), np.nan
            )
    for k in (5, 20, 60):
        mean_square = window_sums(g**2, k) / k
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"log_rv_{k}d"] = np.where(mean_square > 0, 0.5 * np.log(mean_square), np.nan)
    return pd.DataFrame(result, index=log_return.index).loc[:, list(FEATURES)]


def labels_from_log_returns(
    dates: pd.Series, log_return: pd.Series, *, horizon: int, entry_lag: int, target: str
) -> pd.DataFrame:
    """target_return, target_start_date and target_end_date for one asset and target."""

    g = np.asarray(log_return, dtype=np.float64)
    offset = entry_lag + horizon
    if target == "total_log_return":
        ending = window_sums(g, horizon)
    elif target == "log_realized_volatility":
        mean_square = window_sums(g**2, horizon) / horizon
        with np.errstate(divide="ignore", invalid="ignore"):
            ending = np.where(mean_square > 0, 0.5 * np.log(mean_square), np.nan)
    else:
        raise ValueError(f"Unknown target {target!r}.")
    value = pd.Series(ending).shift(-offset).to_numpy()  # the window ending at the exit row
    stamps = pd.Series(pd.DatetimeIndex(dates).as_unit("ns"))
    return pd.DataFrame(
        {
            TARGET_RETURN_COLUMN: value,
            TARGET_START_COLUMN: stamps.shift(-entry_lag).to_numpy(),
            TARGET_END_COLUMN: stamps.shift(-offset).to_numpy(),
        },
        index=log_return.index,
    )


def asset_frame(returns: pd.DataFrame, fund_id: str) -> pd.DataFrame:
    """date, symbol, close_observed, log_return and the features, one row per session."""

    frame = returns.loc[:, ["date", "close_observed", "log_return"]].reset_index(drop=True)
    frame.insert(1, "symbol", fund_id)
    features = features_from_log_returns(frame["log_return"])
    return pd.concat([frame, features], axis=1)
