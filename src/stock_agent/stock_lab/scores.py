"""
Proper scores, calibration and intervals for three-category probabilities
(numpy and the standard library only).

Probabilities are arrays (rows, 3) in code order Decrease, Stable, Increase;
outcomes are codes 0, 1, 2. Every score is a proper scoring rule (minimized
in expectation only by the true probabilities):

    Brier       sum_c (p_c - y_c)^2                      (0 best, 2 worst)
    log loss    -log p_y                                  (probabilities floored)
    RPS         sum_(k=1..2) (P_k - O_k)^2 / 2            ranked probability score:
                cumulative over the ordered categories Decrease < Stable <
                Increase, so predicting Decrease when Increase happens costs
                more than predicting Stable

and two decompositions the statistical review asked to score separately:

    moving      Brier of P(Moving) = 1 - P(Stable) against "not Stable"
    direction   Brier of P(Increase | Moving) = p_I / (p_I + p_D) against
                "Increase", on sessions that moved only

Intervals for a proportion use the Jeffreys interval (Beta(x + 1/2,
n - x + 1/2) quantiles), with an effective sample size when observations
are weighted.
"""

from __future__ import annotations

import math

import numpy as np

from stock_agent.stock_lab.estimators import FLOOR, floor

K = 3


def one_hot(outcomes: np.ndarray) -> np.ndarray:
    return np.eye(K)[outcomes.astype(int)]


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - one_hot(y)) ** 2).sum(axis=1)


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = FLOOR) -> np.ndarray:
    q = floor(p, eps)
    return -np.log(q[np.arange(len(y)), y.astype(int)])


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(p, axis=1)[:, :2]
    observed = np.cumsum(one_hot(y), axis=1)[:, :2]
    return ((cumulative - observed) ** 2).sum(axis=1) / 2.0


def moving_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    moving = 1.0 - p[:, 1]
    return (moving - (y != 1)) ** 2


def direction_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per row; NaN for sessions that did not move (Stable outcomes)."""

    with np.errstate(invalid="ignore", divide="ignore"):
        up = p[:, 2] / (p[:, 2] + p[:, 0])
    score = (up - (y == 2)) ** 2
    return np.where(y == 1, np.nan, score)


SCORES = {
    "brier": brier,
    "log_loss": log_loss,
    "rps": rps,
    "moving_brier": moving_brier,
    "direction_brier": direction_brier,
}


# --- the Jeffreys interval ---


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""

    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if beta_cdf(mid, a, b) < q:
            low = mid
        else:
            high = mid
        if high - low < 1e-12:
            break
    return (low + high) / 2.0


def jeffreys(successes: float, n: float, level: float = 0.9) -> tuple[float, float]:
    """
    The Jeffreys interval for a proportion: Beta(x + 1/2, n - x + 1/2)
    quantiles; the lower end is 0 when x = 0 and the upper 1 when x = n.
    `n` may be an effective sample size (successes scaled to match).
    """

    if n <= 0:
        return math.nan, math.nan
    a, b = successes + 0.5, n - successes + 0.5
    tail = (1.0 - level) / 2.0
    low = 0.0 if successes <= 0 else beta_ppf(tail, a, b)
    high = 1.0 if successes >= n else beta_ppf(1.0 - tail, a, b)
    return low, high


# --- calibration ---


def reliability(
    p: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 10,
    level: float = 0.9,
    groups: np.ndarray | None = None,
) -> list[dict]:
    """
    Per category and equal-width probability bin: how many forecasts, their
    mean predicted probability, the observed frequency and its Jeffreys
    interval. With `groups` (the forecasts' dates), the interval uses the
    number of distinct dates in the bin as its sample size: forecasts on one
    date (other stocks, other horizons) move together, so counting each as
    independent would make the interval too narrow. Bins without forecasts
    are omitted.
    """

    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for c in range(K):
        probability = p[:, c]
        hit = (y == c).astype("float64")
        index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            chosen = index == b
            n = int(chosen.sum())
            if not n:
                continue
            observed = float(hit[chosen].sum())
            effective = n if groups is None else len(set(groups[chosen].tolist()))
            low, high = jeffreys(observed / n * effective, effective, level)
            rows.append(
                {
                    "category": c,
                    "bin": b,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "n": n,
                    "dates": effective,
                    "mean_predicted": float(probability[chosen].mean()),
                    "observed": observed / n,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def calibration_line(probability: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    """
    (intercept, slope) of the logistic recalibration of a one-versus-rest
    probability: logit P(hit) = a + b logit(p). Perfect calibration is (0, 1);
    a slope below 1 means the forecasts are too extreme.
    """

    q = np.clip(probability, FLOOR, 1 - FLOOR)
    x = np.log(q / (1 - q))
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = mu * (1 - mu)
        gradient = design.T @ (hit - mu)
        hessian = design.T @ (design * weight[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone); NaN stays NaN."""

    finite = sorted((value, key) for key, value in p_values.items() if not math.isnan(value))
    m = len(finite)
    adjusted, running = {}, 0.0
    for rank, (value, key) in enumerate(finite):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[key] = running
    for key, value in p_values.items():
        if math.isnan(value):
            adjusted[key] = math.nan
    return adjusted


def one_se_choice(per_date: dict, simplest_first: list) -> tuple[object, list[dict]]:
    """
    The one-standard-error rule with PAIRED errors: the candidate with the
    lowest mean per-date score is the best; every candidate whose mean per-date
    difference from the best is within one standard error of that difference
    (std of the paired daily differences / sqrt(dates)) is eligible; the first
    eligible one in `simplest_first` is chosen. Pairing matters: shocks common
    to a date cancel in the difference, so its standard error is several times
    smaller than either score's own.
    """

    means = {key: float(series.mean()) for key, series in per_date.items()}
    best = min(means, key=means.get)
    rows = []
    eligible = set()
    for key in simplest_first:
        diff = (per_date[key] - per_date[best]).dropna()
        n = len(diff)
        se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else math.inf
        mean_diff = float(diff.mean()) if n else math.inf
        if key == best or mean_diff <= se:
            eligible.add(key)
        rows.append(
            {"candidate": key, "mean": means[key], "diff_vs_best": mean_diff, "paired_se": se}
        )
    chosen = next(key for key in simplest_first if key in eligible)
    return chosen, rows
