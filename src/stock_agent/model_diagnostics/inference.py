"""
Uncertainty for date-indexed statistics (numpy and the standard library only).

Every statistic in this package is reduced to one value per scored session
(a rank IC, a loss difference, a hit share) before inference. Rows of one
date are never treated as independent observations: the names in a
cross-section share market shocks, and adjacent dates share 19 of 20 label
returns under MODEL_LABEL_SPEC. Three estimates are reported side by side,
because none is exact with only about 24 independent 63-session blocks:

newey_west
    Bartlett-kernel long-run variance with an explicit lag (default 40;
    the research plan rejects lag 20, which over-rejected 7-13% at 5%).
    Normal critical values. Its size is simulated and recorded, not assumed.
fold_block_t
    A t-test on per-block means (blocks = folds) with t(B-1) critical
    values. Coarse but close to nominal size in simulation.
circular_block_bootstrap
    Politis-Romano circular blocks of whole sessions; every date keeps its
    cross-section. Mean statistics use the bootstrap SE with t(B-1)
    critical values; ratio statistics use percentile intervals.

Series are aligned on the scored-session calendar. A NaN marks a session
whose value is unavailable (for example an undefined IC); lags count
sessions, so a gap is never closed up and adjacent-looking values are not
treated as neighbours.

Randomness uses numpy's legacy RandomState, whose stream is frozen across
numpy versions (NEP 19).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

DEFAULT_HAC_LAG = 40
DEFAULT_BLOCK_LENGTH = 63
DEFAULT_BOOTSTRAP_REPS = 999
DEFAULT_CI_LEVEL = 0.95


@dataclass(frozen=True, eq=False)
class Inference:
    """
    One uncertainty estimate for one statistic.

    `p_value` tests `null_value` (two-sided). Unavailable estimates carry
    status "unavailable", a machine-readable reason, and NaN numbers; they
    are never reported as zero.
    """

    method: str
    estimate: float
    se: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    p_value: float = math.nan
    null_value: float = 0.0
    n: int = 0
    n_eff: float = math.nan
    status: str = "ok"
    reason: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)


def unavailable(
    method: str,
    reason: str,
    *,
    n: int = 0,
    estimate: float = math.nan,
    null_value: float = 0.0,
    **params,
):
    return Inference(
        method=method,
        estimate=estimate,
        null_value=null_value,
        n=n,
        status="unavailable",
        reason=reason,
        params=params,
    )


def _require_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}.")
    return int(value)


def _require_level(ci_level: float) -> float:
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}.")
    return float(ci_level)


# --- distributions (no scipy) ---


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("Incomplete beta continued fraction did not converge.")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""

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


def student_t_two_sided_p(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive.")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def student_t_cdf(t: float, df: float) -> float:
    tail = 0.5 * student_t_two_sided_p(t, df)
    return 1.0 - tail if t >= 0 else tail


def _bisect_quantile(cdf: Callable[[float], float], q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError(f"Quantile level must be in (0, 1); got {q}.")
    if q < 0.5:
        return -_bisect_quantile(cdf, 1.0 - q)
    lo, hi = 0.0, 1.0
    while cdf(hi) < q:
        hi *= 2.0
    while hi - lo > 1e-12 * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@functools.lru_cache(maxsize=256)
def normal_ppf(q: float) -> float:
    return _bisect_quantile(normal_cdf, q)


@functools.lru_cache(maxsize=256)
def student_t_ppf(q: float, df: float) -> float:
    return _bisect_quantile(lambda t: student_t_cdf(t, df), q)


# --- long-run variance ---


def _as_series(values: object) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D date series; got shape {array.shape}.")
    if np.isinf(array).any():
        raise ValueError("Date series must not contain infinities.")
    return array


def bartlett_weights(lag: int) -> np.ndarray:
    """w_l = 1 - l / (lag + 1) for l = 1..lag (positive semidefinite kernel)."""

    lag = _require_int(lag, "lag", 0)
    return 1.0 - np.arange(1, lag + 1, dtype=np.float64) / (lag + 1.0)


def _centered_autocovariances(values: np.ndarray, lag: int) -> tuple[np.ndarray, int, float]:
    """
    gamma_0..gamma_lag of a gappy series, each divided by the number of
    present values (the usual biased estimator, which keeps the Bartlett
    long-run variance non-negative). Pairs with a missing side drop out.
    """

    present = np.isfinite(values)
    n = int(present.sum())
    mean = float(values[present].mean())
    centered = np.where(present, values - mean, 0.0)
    gammas = np.empty(lag + 1, dtype=np.float64)
    gammas[0] = centered @ centered / n
    for k in range(1, lag + 1):
        gammas[k] = centered[k:] @ centered[:-k] / n
    return gammas, n, mean


def long_run_variance(values: object, *, lag: int = DEFAULT_HAC_LAG) -> float:
    """Newey-West (Bartlett) long-run variance: gamma_0 + 2 sum w_l gamma_l."""

    series = _as_series(values)
    lag = _require_int(lag, "lag", 0)
    present = int(np.isfinite(series).sum())
    if present <= lag:
        raise ValueError(f"Need more than lag={lag} present values; got {present}.")
    gammas, _, _ = _centered_autocovariances(series, lag)
    return float(gammas[0] + 2.0 * bartlett_weights(lag) @ gammas[1:])


def newey_west_mean(
    values: object,
    *,
    lag: int = DEFAULT_HAC_LAG,
    ci_level: float = DEFAULT_CI_LEVEL,
    null_value: float = 0.0,
) -> Inference:
    """
    Mean of a date series with a Newey-West standard error.

    When the mean exists but its uncertainty cannot be estimated (too few
    dates, a series shorter than the lag, zero variance), the estimate is
    kept and status/reason say why the standard error is missing.

    SE = sqrt(LRV / n); n_eff = n * gamma_0 / LRV, the number of independent
    dates carrying the same information about the mean.
    """

    series = _as_series(values)
    lag = _require_int(lag, "lag", 0)
    ci_level = _require_level(ci_level)
    params = {"lag": lag, "kernel": "bartlett", "ci_level": ci_level, "ci_method": "normal"}
    n = int(np.isfinite(series).sum())
    mean = float(np.nanmean(series)) if n else math.nan
    if n < 2:
        return unavailable("newey_west", "too_few_dates", n=n, estimate=mean, **params)
    if n <= lag:
        return unavailable("newey_west", "series_shorter_than_lag", n=n, estimate=mean, **params)
    gammas, n, mean = _centered_autocovariances(series, lag)
    lrv = float(gammas[0] + 2.0 * bartlett_weights(lag) @ gammas[1:])
    if not lrv > 0.0:
        return unavailable("newey_west", "zero_variance", n=n, estimate=mean, **params)
    se = math.sqrt(lrv / n)
    z = normal_ppf(0.5 + ci_level / 2.0)
    return Inference(
        method="newey_west",
        estimate=mean,
        se=se,
        ci_low=mean - z * se,
        ci_high=mean + z * se,
        p_value=normal_two_sided_p((mean - null_value) / se),
        null_value=null_value,
        n=n,
        n_eff=n * float(gammas[0]) / lrv,
        params=params,
    )


def effective_sample_size(values: object, *, lag: int = DEFAULT_HAC_LAG) -> float:
    """n * gamma_0 / LRV; NaN when undefined (constant or too-short series)."""

    return newey_west_mean(values, lag=lag).n_eff


def iid_mean(values: object, *, ci_level: float = DEFAULT_CI_LEVEL) -> Inference:
    """
    Mean with the naive iid standard error (sample std / sqrt(n)).

    Reported only as a contrast: with overlapping labels it is far too small.
    """

    series = _as_series(values)
    present = series[np.isfinite(series)]
    params = {"ci_level": ci_level, "ci_method": "normal"}
    if len(present) < 2:
        return unavailable("iid", "too_few_dates", n=len(present), **params)
    mean = float(present.mean())
    sd = float(present.std(ddof=1))
    if not sd > 0.0:
        return unavailable("iid", "zero_variance", n=len(present), estimate=mean, **params)
    se = sd / math.sqrt(len(present))
    z = normal_ppf(0.5 + ci_level / 2.0)
    return Inference(
        method="iid",
        estimate=mean,
        se=se,
        ci_low=mean - z * se,
        ci_high=mean + z * se,
        p_value=normal_two_sided_p(mean / se),
        n=len(present),
        n_eff=float(len(present)),
        params=params,
    )


# --- block t-test ---


def fold_block_t(
    values: object,
    blocks: object,
    *,
    ci_level: float = DEFAULT_CI_LEVEL,
    null_value: float = 0.0,
    min_blocks: int = 3,
) -> Inference:
    """
    t-test on per-block means with t(B - 1) critical values.

    `blocks` labels each session (for example its fold id). The estimate is
    the unweighted mean of block means, so it can differ slightly from the
    pooled mean when blocks hold different numbers of dates.
    """

    series = _as_series(values)
    labels = np.asarray(blocks)
    if labels.shape != series.shape:
        raise ValueError("blocks must label every session of the series.")
    ci_level = _require_level(ci_level)
    params = {"ci_level": ci_level, "ci_method": "student_t", "block_basis": "fold"}
    present = np.isfinite(series)
    if not present.any():
        return unavailable("fold_block_t", "too_few_dates", n=0, **params)
    keys, codes = np.unique(labels[present], return_inverse=True)
    sums = np.bincount(codes, weights=series[present])
    counts = np.bincount(codes)
    block_means = sums / counts
    n_blocks = len(keys)
    params["blocks"] = n_blocks
    estimate = float(block_means.mean())
    if n_blocks < min_blocks:
        return unavailable(
            "fold_block_t", "too_few_blocks", n=int(present.sum()), estimate=estimate, **params
        )
    sd = float(block_means.std(ddof=1))
    if not sd > 0.0:
        return unavailable(
            "fold_block_t", "zero_variance", n=int(present.sum()), estimate=estimate, **params
        )
    se = sd / math.sqrt(n_blocks)
    df = n_blocks - 1
    params["df"] = df
    critical = student_t_ppf(0.5 + ci_level / 2.0, df)
    return Inference(
        method="fold_block_t",
        estimate=estimate,
        se=se,
        ci_low=estimate - critical * se,
        ci_high=estimate + critical * se,
        p_value=student_t_two_sided_p((estimate - null_value) / se, df),
        null_value=null_value,
        n=int(present.sum()),
        params=params,
    )


# --- circular block bootstrap ---


@dataclass(frozen=True, eq=False)
class BlockDraws:
    """
    Circular block bootstrap draws over n sessions.

    Each replicate concatenates k = ceil(n / block_length) blocks of
    consecutive sessions (wrapping past the last session to the first) and
    truncates to n sessions: block j starts at starts[r, j] and has length
    lengths[j]. Draws are made once per run and shared by every model, so
    model-versus-baseline comparisons are paired.
    """

    n: int
    block_length: int
    reps: int
    seed: int
    starts: np.ndarray
    lengths: np.ndarray

    @property
    def blocks_per_series(self) -> int:
        return len(self.lengths)

    def indices(self) -> np.ndarray:
        """Materialized (reps, n) session positions (for inspection and tests)."""

        offsets = np.concatenate([np.arange(length) for length in self.lengths])
        block_of = np.repeat(np.arange(len(self.lengths)), self.lengths)
        return (self.starts[:, block_of] + offsets) % self.n


def draw_circular_blocks(
    n: int,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int,
) -> BlockDraws:
    n = _require_int(n, "n", 1)
    block_length = _require_int(block_length, "block_length", 1)
    reps = _require_int(reps, "reps", 1)
    seed = _require_int(seed, "seed", 0)
    count = -(-n // block_length)
    lengths = np.full(count, block_length, dtype=np.int64)
    lengths[-1] = n - block_length * (count - 1)
    starts = np.random.RandomState(seed).randint(0, n, size=(reps, count)).astype(np.int64)
    starts.setflags(write=False)
    lengths.setflags(write=False)
    return BlockDraws(n, block_length, reps, seed, starts, lengths)


def resampled_sums(values: object, draws: BlockDraws) -> np.ndarray:
    """
    Sum of `values` over each bootstrap replicate's sessions, shape (reps,).

    Uses cumulative sums of the series repeated twice, so each block sum is
    one subtraction. NaN entries contribute nothing; count presence with a
    second call on the presence indicator.
    """

    series = np.asarray(values, dtype=np.float64)
    if series.shape != (draws.n,):
        raise ValueError(f"Expected {draws.n} sessions; got shape {series.shape}.")
    filled = np.where(np.isfinite(series), series, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.concatenate([filled, filled]))])
    ends = draws.starts + draws.lengths
    return (cumulative[ends] - cumulative[draws.starts]).sum(axis=1)


def _min_sessions(draws: BlockDraws) -> int:
    return 2 * draws.block_length


def bootstrap_mean(
    values: object,
    draws: BlockDraws,
    *,
    ci_level: float = DEFAULT_CI_LEVEL,
    null_value: float = 0.0,
) -> Inference:
    """
    Mean of a date series: bootstrap SE with t(k - 1) critical values, where
    k = ceil(n / block_length) is the number of blocks per replicate.
    """

    series = _as_series(values)
    ci_level = _require_level(ci_level)
    params = {
        "block_length": draws.block_length,
        "reps": draws.reps,
        "seed": draws.seed,
        "ci_level": ci_level,
        "ci_method": "bootstrap_se_student_t",
        "scheme": "circular",
    }
    present = np.isfinite(series)
    n = int(present.sum())
    estimate = float(series[present].mean()) if n else math.nan
    if draws.n < _min_sessions(draws):
        return unavailable(
            "circular_block_bootstrap",
            "series_shorter_than_two_blocks",
            n=n,
            estimate=estimate,
            **params,
        )
    if n < 2:
        return unavailable(
            "circular_block_bootstrap", "too_few_dates", n=n, estimate=estimate, **params
        )
    counts = resampled_sums(present.astype(np.float64), draws)
    sums = resampled_sums(series, draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        replicates = sums / counts
    replicates = replicates[counts > 0]
    se = float(replicates.std(ddof=1))
    if not se > 0.0:
        return unavailable(
            "circular_block_bootstrap", "zero_variance", n=n, estimate=estimate, **params
        )
    df = draws.blocks_per_series - 1
    params["df"] = df
    critical = student_t_ppf(0.5 + ci_level / 2.0, df)
    return Inference(
        method="circular_block_bootstrap",
        estimate=estimate,
        se=se,
        ci_low=estimate - critical * se,
        ci_high=estimate + critical * se,
        p_value=student_t_two_sided_p((estimate - null_value) / se, df),
        null_value=null_value,
        n=n,
        params=params,
    )


def bootstrap_statistic(
    per_date: Mapping[str, object],
    statistic: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    draws: BlockDraws,
    *,
    ci_level: float = DEFAULT_CI_LEVEL,
    n: int,
) -> Inference:
    """
    A smooth function of per-date sums (for example 1 - SSE_model / SSE_base)
    with a percentile interval over date-block replicates.

    `statistic` receives a mapping of name -> array of sums (shape (1,) for
    the full sample, (reps,) for replicates) and returns the statistic per
    element. Percentile intervals are approximate with ~24 blocks.
    """

    ci_level = _require_level(ci_level)
    params = {
        "block_length": draws.block_length,
        "reps": draws.reps,
        "seed": draws.seed,
        "ci_level": ci_level,
        "ci_method": "percentile_uncalibrated",
        "scheme": "circular",
    }
    full = {
        name: np.array([np.nansum(np.asarray(values, dtype=np.float64))])
        for name, values in per_date.items()
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        estimate = float(statistic(full)[0])
    if not math.isfinite(estimate):
        return unavailable("circular_block_bootstrap", "undefined_statistic", n=n, **params)
    if draws.n < _min_sessions(draws):
        return Inference(
            method="circular_block_bootstrap",
            estimate=estimate,
            n=n,
            status="unavailable",
            reason="series_shorter_than_two_blocks",
            params=params,
        )
    replicated = {name: resampled_sums(values, draws) for name, values in per_date.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        replicates = np.asarray(statistic(replicated), dtype=np.float64)
    replicates = replicates[np.isfinite(replicates)]
    if len(replicates) < 2:
        return Inference(
            method="circular_block_bootstrap",
            estimate=estimate,
            n=n,
            status="unavailable",
            reason="undefined_replicates",
            params=params,
        )
    alpha = 1.0 - ci_level
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    params["valid_reps"] = len(replicates)
    return Inference(
        method="circular_block_bootstrap",
        estimate=estimate,
        se=float(replicates.std(ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
        null_value=math.nan,
        n=n,
        params=params,
    )


# --- Driscoll-Kraay regression ---


@dataclass(frozen=True, eq=False)
class RegressionResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    n: int
    n_dates: int
    status: str
    reason: str | None
    lag: int


def driscoll_kraay_ols(
    y: object,
    X: object,
    session_positions: object,
    *,
    n_sessions: int,
    lag: int = DEFAULT_HAC_LAG,
) -> RegressionResult:
    """
    Pooled OLS with Driscoll-Kraay standard errors.

    Row scores x_i * e_i are summed within each session first, so names that
    share a date are one observation; the session sums then get a Bartlett
    kernel over `lag` sessions (gaps kept). V = (X'X)^-1 S (X'X)^-1 with no
    small-sample correction: lag 0 equals date-clustered CR0, and HC0 when
    every session holds one row.
    """

    target = np.asarray(y, dtype=np.float64)
    design = np.asarray(X, dtype=np.float64)
    positions = np.asarray(session_positions, dtype=np.int64)
    lag = _require_int(lag, "lag", 0)
    k = design.shape[1]
    if design.ndim != 2 or len(target) != len(design) or len(positions) != len(design):
        raise ValueError("y, X, and session_positions must describe the same rows.")
    if not (np.isfinite(target).all() and np.isfinite(design).all()):
        raise ValueError("Regression inputs must be finite.")
    n_dates = len(np.unique(positions))
    nan_result = RegressionResult(
        np.full(k, np.nan), np.full((k, k), np.nan), len(target), n_dates, "unavailable", "", lag
    )
    if len(target) <= k:
        return _with_reason(nan_result, "too_few_rows")
    if np.linalg.matrix_rank(design) < k:
        return _with_reason(nan_result, "singular_design")
    if n_dates <= lag:
        return _with_reason(nan_result, "series_shorter_than_lag")
    xtx = design.T @ design
    bread = np.linalg.inv(xtx)
    beta = bread @ (design.T @ target)
    residuals = target - design @ beta
    session_scores = np.zeros((n_sessions, k), dtype=np.float64)
    np.add.at(session_scores, positions, design * residuals[:, None])
    meat = session_scores.T @ session_scores
    for distance, weight in zip(range(1, lag + 1), bartlett_weights(lag), strict=True):
        cross = session_scores[distance:].T @ session_scores[:-distance]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    return RegressionResult(beta, covariance, len(target), n_dates, "ok", None, lag)


def _with_reason(result: RegressionResult, reason: str) -> RegressionResult:
    return RegressionResult(
        result.coefficients,
        result.covariance,
        result.n,
        result.n_dates,
        "unavailable",
        reason,
        result.lag,
    )


# --- synthetic calibration ---


CALIBRATION_SIGNALS = ("iid", "momentum", "block_constant")


def simulate_null_ic_series(
    *,
    n_dates: int,
    n_names: int = 4,
    horizon: int = 20,
    entry_lag: int = 1,
    signal: str = "momentum",
    reps: int,
    seed: int,
    block_length: int = DEFAULT_BLOCK_LENGTH,
) -> np.ndarray:
    """
    Daily rank ICs of a signal with no skill, shape (reps, n_dates).

    Returns are iid across names and days, so every signal is independent of
    the labels, while the labels overlap exactly as in MODEL_LABEL_SPEC
    (sum of `horizon` daily returns starting `entry_lag` days later). The
    signal sets the IC's persistence:
      iid             fresh noise each day;
      momentum        trailing 20-day return (as momentum_20d);
      block_constant  one random ordering held for each 63-session block.
    """

    if signal not in CALIBRATION_SIGNALS:
        raise ValueError(f"signal must be one of {CALIBRATION_SIGNALS}; got {signal!r}.")
    rng = np.random.RandomState(_require_int(seed, "seed", 0))
    warmup = 20
    total = warmup + n_dates + entry_lag + horizon
    returns = rng.randn(reps, total, n_names)
    cumulative = np.concatenate([np.zeros((reps, 1, n_names)), np.cumsum(returns, axis=1)], axis=1)
    dates = np.arange(warmup, warmup + n_dates)
    # label at t: returns t+entry_lag+1 .. t+entry_lag+horizon
    labels = cumulative[:, dates + entry_lag + horizon + 1] - cumulative[:, dates + entry_lag + 1]
    if signal == "iid":
        scores = rng.randn(reps, n_dates, n_names)
    elif signal == "momentum":
        scores = cumulative[:, dates + 1] - cumulative[:, dates - 19]
    else:
        blocks = -(-n_dates // block_length)
        orderings = rng.rand(reps, blocks, n_names)
        scores = np.repeat(orderings, block_length, axis=1)[:, :n_dates]
    return _rank_correlation(scores, labels)


def _rank_correlation(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Spearman correlation along the last axis for continuous (tie-free) data."""

    score_ranks = scores.argsort(axis=-1).argsort(axis=-1).astype(np.float64)
    label_ranks = labels.argsort(axis=-1).argsort(axis=-1).astype(np.float64)
    score_ranks -= score_ranks.mean(axis=-1, keepdims=True)
    label_ranks -= label_ranks.mean(axis=-1, keepdims=True)
    numerator = (score_ranks * label_ranks).sum(axis=-1)
    denominator = np.sqrt((score_ranks**2).sum(axis=-1) * (label_ranks**2).sum(axis=-1))
    return numerator / denominator


def null_rejection_rates(
    series: np.ndarray,
    *,
    lag: int = DEFAULT_HAC_LAG,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    bootstrap_reps: int = 199,
    alpha: float = 0.05,
    seed: int,
) -> dict[str, float]:
    """
    Share of no-skill series whose mean is declared significant at `alpha`,
    per method, vectorized over rows of `series` (shape (reps, n_dates)).

    Uses the same Bartlett weights, fold-block layout (consecutive blocks of
    `block_length` sessions) and circular block draws as the estimators
    above; the shared helpers are tested for equality with them.
    """

    series = np.asarray(series, dtype=np.float64)
    reps, n = series.shape
    lag = _require_int(lag, "lag", 0)
    means = series.mean(axis=1)
    centered = series - means[:, None]
    gamma0 = (centered**2).sum(axis=1) / n
    lrv = gamma0.copy()
    for distance, weight in zip(range(1, lag + 1), bartlett_weights(lag), strict=True):
        lrv += 2.0 * weight * (centered[:, distance:] * centered[:, :-distance]).sum(axis=1) / n
    z = normal_ppf(1.0 - alpha / 2.0)
    rates = {
        "iid": float(np.mean(np.abs(means) / np.sqrt(series.var(axis=1, ddof=1) / n) > z)),
        "newey_west": float(np.mean(np.abs(means) / np.sqrt(lrv / n) > z)),
    }

    labels = np.arange(n) // block_length
    block_count = int(labels[-1]) + 1
    block_means = np.stack(
        [series[:, labels == block].mean(axis=1) for block in range(block_count)], axis=1
    )
    block_se = block_means.std(axis=1, ddof=1) / math.sqrt(block_count)
    critical = student_t_ppf(1.0 - alpha / 2.0, block_count - 1)
    rates["fold_block_t"] = float(np.mean(np.abs(block_means.mean(axis=1)) / block_se > critical))

    draws = draw_circular_blocks(n, block_length=block_length, reps=bootstrap_reps, seed=seed)
    filled = np.concatenate(
        [np.zeros((reps, 1)), np.cumsum(np.hstack([series, series]), axis=1)], axis=1
    )
    ends = draws.starts + draws.lengths
    replicate_means = (filled[:, ends] - filled[:, draws.starts]).sum(axis=2) / n
    bootstrap_se = replicate_means.std(axis=1, ddof=1)
    critical = student_t_ppf(1.0 - alpha / 2.0, draws.blocks_per_series - 1)
    rates["circular_block_bootstrap"] = float(np.mean(np.abs(means) / bootstrap_se > critical))
    return rates
