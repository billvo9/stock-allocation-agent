"""
Zero realized volatility in the log-volatility target: the rule in force and
the research-only option B2 (docs/research/zero_vol_floor.md).

A volatility outcome of row t (as-of session t, entry e = t + L, exit
x = e + h) covers the h one-session log total returns g_{e+1..x}:

    sigma_raw      = sqrt(mean g^2)          measured, >= 0
    measured_zero  = every g is exactly 0    (so sigma_raw == 0)

CURRENT, "current_drop_and_miss" (in force; the default everywhere)

    y = 0.5 log(mean g^2) when sigma_raw > 0, undefined otherwise.
    Training drops a row whose y is undefined (both of its labels), and
    prospective scoring counts such an outcome as outside the band.
    floored is always False and sigma_floor is NaN.

B2, "research_b2_half_tick_floor" (research only; NOT approved by the owner)

    sigma_floor_h = 0.5 log(1 + tick / P_asof) / sqrt(h)     tick = $0.01
    floored       = sigma_raw < sigma_floor_h
    y             = log(sigma_floor_h) if floored, else 0.5 log(mean g^2)

    P_asof is the row's own unadjusted as-of close, known at the decision
    time: the floor never looks at the outcome window. One full tick move in
    the window gives sigma_raw = |log(1 +- tick / P)| / sqrt(h), about twice
    the floor, so only measured zeros and windows whose only movement is
    smaller than half a tick (for example a sub-tick distribution) are
    floored. No row is dropped, and the same y is the training label, the
    conformal calibration score's label and the prospective outcome.

Measured zero volatility stays distinguishable from the modeling floor: every
row records sigma_raw, measured_zero and floored under either rule. Where a
row is not floored, B2's y is the current rule's y bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_RETURN_COLUMN
from stock_agent.forecasting.prospective import series

CURRENT = "current_drop_and_miss"
B2 = "research_b2_half_tick_floor"
RULES = (CURRENT, B2)
DEFAULT = CURRENT
TICK = 0.01  # dollars: the price grid of the funds' closes
DIAGNOSTIC_COLUMNS = ("sigma_raw", "measured_zero", "floored", "sigma_floor")
DESCRIPTIONS = {
    CURRENT: (
        "Training drops a row whose realized volatility is 0 (both labels); prospective "
        "scoring counts it outside the band."
    ),
    B2: (
        "Research only, not approved: y = log max(sigma_raw, 0.5 log(1 + 0.01 / P_asof) / "
        "sqrt(h)) in training labels, calibration and scoring; no row dropped."
    ),
}


def check_rule(rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"Unknown zero-volatility rule {rule!r}; expected one of {RULES}.")
    return rule


def sigma_floor(as_of_close: object, horizon: int, tick: float = TICK) -> np.ndarray:
    """B2's floor on daily realized volatility: 0.5 log(1 + tick / P_asof) / sqrt(h)."""

    close = np.asarray(as_of_close, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.log1p(tick / close) / np.sqrt(horizon)


def as_of_closes(dates: pd.Series, prices: pd.DataFrame, fund_id: str) -> np.ndarray:
    """The fund's unadjusted close on each of `dates` (NaN where it has none)."""

    own = prices.loc[
        (prices["fund_id"] == fund_id) & (prices["adjustment_basis"] == series.UNADJUSTED),
        ["session_date", "close"],
    ]
    index = pd.DatetimeIndex(pd.to_datetime(own["session_date"], utc=True)).as_unit("ns")
    closes = pd.Series(own["close"].to_numpy(np.float64), index=index)
    wanted = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).as_unit("ns")
    return closes.reindex(wanted).to_numpy(np.float64)


def volatility_labels(
    dates: pd.Series,
    log_return: pd.Series,
    as_of_close: object,
    *,
    horizon: int,
    entry_lag: int,
    rule: str = DEFAULT,
) -> pd.DataFrame:
    """
    `series.labels_from_log_returns` for the log-volatility target under
    `rule`, plus sigma_raw, measured_zero, floored and sigma_floor per row.
    `as_of_close` is aligned with `dates` (the row's own as-of close).
    """

    check_rule(rule)
    labels = series.labels_from_log_returns(
        dates, log_return, horizon=horizon, entry_lag=entry_lag, target="log_realized_volatility"
    )
    g = np.asarray(log_return, dtype=np.float64)
    offset = entry_lag + horizon
    mean_square = series.window_sums(g**2, horizon) / horizon
    moved = np.where(np.isnan(g), np.nan, (g != 0.0).astype(np.float64))
    # The window ending at the exit row, as in labels_from_log_returns (NaN when incomplete).
    raw = pd.Series(np.sqrt(mean_square)).shift(-offset).to_numpy()
    moves = pd.Series(series.window_sums(moved, horizon)).shift(-offset).to_numpy()
    measured_zero = moves == 0.0
    if rule == CURRENT:
        floor = np.full(len(g), np.nan)
        floored = np.zeros(len(g), dtype=bool)
    else:
        floor = sigma_floor(as_of_close, horizon)
        floored = np.isfinite(raw) & np.isfinite(floor) & (raw < floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            floored_value = np.log(floor)
        labels[TARGET_RETURN_COLUMN] = np.where(
            floored, floored_value, labels[TARGET_RETURN_COLUMN].to_numpy(np.float64)
        )
    labels["sigma_raw"] = raw
    labels["measured_zero"] = measured_zero
    labels["floored"] = floored
    labels["sigma_floor"] = floor
    return labels


__all__ = [
    "B2",
    "CURRENT",
    "DEFAULT",
    "DESCRIPTIONS",
    "DIAGNOSTIC_COLUMNS",
    "RULES",
    "TICK",
    "as_of_closes",
    "check_rule",
    "sigma_floor",
    "volatility_labels",
]
