"""
Forecast targets: the category of each of the next five sessions, and of the
five-session total, for every decision session d (after d's close).

    daily target h (h = 1..5)   the category of r_(d+h) against the band tau
    cumulative target           the category of L_(d+5) / L_d - 1 against the
                                band tau_5 = tau * sqrt(5) (proposed default,
                                owner decision A5; configurable)

Availability. Daily target h needs a usable return at d+h. The cumulative
target needs usable returns at every session d+1..d+5 (a gap would make it
a different span). Every target also needs its last session on or before
`outcome_end`: outcomes after the cutoff (2024-12-31 for real data in
development) are censored, never resolved from the sealed period.

Columns (aligned with `frame`, one stock):
    target_<h>          category code (Decrease 0, Stable 1, Increase 2), Int64
    target_cum          category code of the five-session total
    target_return       the five-session total return (NaN when unavailable)
    target_start_date   the session d+1, and target_end_date the session d+5
                        (NaT beyond the data), for purged walk-forward folds
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import StableBand
from stock_agent.stock_lab.prices import categorize, category_codes

HORIZONS = (1, 2, 3, 4, 5)
CUMULATIVE_SESSIONS = 5
TARGET_COLUMNS = (*(f"target_{h}" for h in HORIZONS), "target_cum")


def cumulative_band(band: StableBand, sessions: int = CUMULATIVE_SESSIONS) -> StableBand:
    """tau * sqrt(sessions), capped at the band's maximum (proposed; owner decision A5)."""

    return StableBand(min(band.tau * math.sqrt(sessions), 0.05))


def _ahead(values: np.ndarray, h: int, fill) -> np.ndarray:
    """values[i + h] at position i, `fill` beyond the end."""

    out = np.full(len(values), fill, dtype=values.dtype)
    if h < len(values):
        out[: len(values) - h] = values[h:]
    return out


def build_targets(
    frame: pd.DataFrame,
    band: StableBand,
    *,
    outcome_end: pd.Timestamp | None = None,
    cumulative: StableBand | None = None,
) -> pd.DataFrame:
    """The target columns for `frame` (features.build_features output, one stock)."""

    sessions = pd.DatetimeIndex(frame["session"])
    returns = frame["price_return"].to_numpy(dtype="float64")
    level = frame["level"].to_numpy(dtype="float64")
    usable = np.isfinite(returns)
    # Both sides in nanoseconds (sessions are stored in microseconds).
    stamps = sessions.as_unit("ns").asi8 if len(sessions) else np.array([], dtype="int64")
    limit = None if outcome_end is None else pd.Timestamp(outcome_end).as_unit("ns").value
    codes = category_codes(categorize(frame["price_return"], band)).to_numpy(
        dtype="float64", na_value=np.nan
    )

    out = pd.DataFrame(index=frame.index)
    in_time = {}
    for h in HORIZONS:
        end = _ahead(stamps, h, np.iinfo("int64").min)
        exists = end != np.iinfo("int64").min
        in_time[h] = exists if limit is None else exists & (end <= limit)
        ok = _ahead(usable, h, False) & in_time[h]
        target = np.where(ok, _ahead(codes, h, np.nan), np.nan)
        out[f"target_{h}"] = pd.array(
            np.where(np.isnan(target), pd.NA, target).tolist(), dtype="Int64"
        )

    all_usable = np.logical_and.reduce([_ahead(usable, h, False) for h in HORIZONS])
    ok_cum = all_usable & in_time[CUMULATIVE_SESSIONS]
    with np.errstate(divide="ignore", invalid="ignore"):
        total = _ahead(level, CUMULATIVE_SESSIONS, np.nan) / level - 1.0
    total = np.where(ok_cum, total, np.nan)
    out["target_return"] = total
    cum_codes = category_codes(
        categorize(pd.Series(total, index=frame.index), cumulative or cumulative_band(band))
    )
    out["target_cum"] = cum_codes.to_numpy()
    start = pd.Series(sessions).shift(-1)
    end = pd.Series(sessions).shift(-CUMULATIVE_SESSIONS)
    out["target_start_date"] = pd.DatetimeIndex(start)
    out["target_end_date"] = pd.DatetimeIndex(end)
    return out
