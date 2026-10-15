"""
80% bands: split-conformal calibration of signed residuals on development
folds (registered as `split_conformal_signed_v1`).

Why not the plug-in. The ETF track's plug-in interval mu_hat +- z sigma_hat
sqrt(h) treats both forecasts as exact, so it runs narrow wherever
volatility is forecast with error (docs/research/ml_etf_track.md). A
conformal band instead takes its width from the model's own out-of-fold
errors, so the forecast errors, the Jensen gap and fat tails are all in it.

Scores, per asset, horizon and target, from out-of-fold development rows:

    return      u = (y - mu_hat) / s,   s = exp(v_hat) sqrt(h)   (the plug-in scale)
    volatility  u = y - v_hat,          s = 1                    (log scale)

Quantiles with the finite-sample conformal ranks (u sorted, n scores):

    lower = u_(floor((n + 1) q_lo)),   upper = u_(ceil((n + 1) q_hi))

and the band is point + s [lower, upper]; q_lo = 0.1 and q_hi = 0.9 give the
registered 80% band. A rank outside 1..n leaves the quantile unavailable
(too few scores), never extrapolated.

Assumptions. The finite-sample guarantee of split conformal needs
exchangeable scores from a model fitted without the calibration rows.
Neither holds exactly: overlapping h-session labels and volatility regimes
break exchangeability, and the frozen quantiles come from out-of-fold scores
of the fold models but are applied to a final refit trained on those same
labels. So the nominal level is a target, not a guarantee; coverage is
measured (the sequential estimate below has the same structure), never
assumed.

Development coverage (`sequential_bands`): fold b is scored with quantiles
from the scores of earlier folds whose labels ended before fold b's test
start, so the reported coverage is out of sample. The frozen band uses every
development score (labels ending before the lockbox). The sealed period is
never used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_END_COLUMN, TARGET_RETURN_COLUMN

METHOD = "split_conformal_signed_v1"
DESCRIPTION = (
    "Split-conformal quantiles of signed standardized out-of-fold residuals per asset, "
    "horizon and target, calibrated on development folds before 2025 only; return scale "
    "exp(volatility forecast) sqrt(h), volatility on the log scale."
)


def scores(frame: pd.DataFrame) -> np.ndarray:
    """Signed standardized residuals (y - point) / scale of scored rows."""

    y = frame[TARGET_RETURN_COLUMN].to_numpy(np.float64)
    return (y - frame["point"].to_numpy(np.float64)) / frame["scale"].to_numpy(np.float64)


def conformal_quantiles(
    values: np.ndarray, lower_q: float, upper_q: float
) -> tuple[float, float] | None:
    """The conformal lower and upper quantiles of `values`, or None if too few."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    if n == 0 or not np.isfinite(ordered).all():
        return None
    low_rank = math.floor((n + 1) * lower_q)
    high_rank = math.ceil((n + 1) * upper_q)
    if low_rank < 1 or high_rank > n:
        return None
    return float(ordered[low_rank - 1]), float(ordered[high_rank - 1])


def sequential_bands(
    frame: pd.DataFrame,
    fold_starts: Mapping[int, pd.Timestamp],
    *,
    lower_q: float,
    upper_q: float,
    min_calibration: int,
) -> pd.DataFrame:
    """
    `frame` (fold_id, date, symbol, point, scale, the label and its end date)
    with band_low and band_high for every fold that has at least
    `min_calibration` earlier matured scores (NaN otherwise), plus the
    calibration size used.
    """

    result = frame.copy()
    result["band_low"] = np.nan
    result["band_high"] = np.nan
    result["n_calibration"] = 0
    values = scores(frame)
    ends = pd.to_datetime(frame[TARGET_END_COLUMN], utc=True)
    folds = frame["fold_id"].to_numpy()
    for fold_id in sorted(set(folds)):
        start = fold_starts[int(fold_id)]
        known = (folds < fold_id) & (ends < start).to_numpy()
        if known.sum() < min_calibration:
            continue
        quantiles = conformal_quantiles(values[known], lower_q, upper_q)
        if quantiles is None:
            continue
        mine = folds == fold_id
        scale = frame["scale"].to_numpy(np.float64)[mine]
        point = frame["point"].to_numpy(np.float64)[mine]
        result.loc[mine, "band_low"] = point + scale * quantiles[0]
        result.loc[mine, "band_high"] = point + scale * quantiles[1]
        result.loc[mine, "n_calibration"] = int(known.sum())
    return result


def inside(frame: pd.DataFrame, low: str = "band_low", high: str = "band_high") -> np.ndarray:
    """Whether each row's label lies inside its band (both ends included)."""

    y = frame[TARGET_RETURN_COLUMN].to_numpy(np.float64)
    return (y >= frame[low].to_numpy(np.float64)) & (y <= frame[high].to_numpy(np.float64))
