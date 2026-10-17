"""
The candidate decision tests for owner decision A7, "which test decides that a
challenger beats B1" (research side: the Lab's page never imports this module;
docs/research/stock_lab_a7_decision.md).

Every procedure reduces a per-date loss differential d_1..d_T (a challenger's
RPS minus B1's, averaged across stocks) to its mean, a standard error that does
not change when a constant is added to the series, and lower-tail critical
values for the t-ratio mean / se at given one-sided levels (the challenger is
better when the mean is negative):

    nw_lag10          Newey-West, Bartlett weights 1 - l/M for l < M, lag 10
                      (M = 11), normal critical values: the current decision
                      test (inference.newey_west_mean);
    fold_block_t      the t-test on the fold means, t(F - 1)
                      (inference.fold_block_t);
    nw_fixed_b_sqrt   Newey-West with M = ceil(1.3 sqrt(T)) (Lazarus, Lewis,
                      Stock and Watson 2018) and fixed-b critical values at
                      b = M / T;
    nw_fixed_b_tenth  Newey-West with M = ceil(0.1 T), fixed-b at b = M / T;
    cbb_block20       the circular block bootstrap's standard error, block 20,
                      t(k - 1) with k = ceil(T / 20) blocks (the current second
                      view, inference.bootstrap_mean);
    sb_politis_white  the stationary bootstrap's standard error at the
                      Politis-White expected block length, t(floor(T / L) - 1).

The bootstrap standard errors are their exact B -> infinity limits (closed
forms below), so no resampling seed enters. Fixed-b critical values are
quantiles of a seeded simulation of the Bartlett fixed-b limit (Kiefer and
Vogelsang 2005), checked against their published polynomials.

Every function works on a batch: `series` has shape (rows, T), one per-date
differential per row, every value present.

Randomness uses numpy's legacy RandomState, whose stream is frozen across
numpy versions (NEP 19).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stock_agent.model_diagnostics.inference import normal_ppf, student_t_ppf

PROCEDURES = (
    "nw_lag10",
    "fold_block_t",
    "nw_fixed_b_sqrt",
    "nw_fixed_b_tenth",
    "cbb_block20",
    "sb_politis_white",
)
CURRENT_LAG = 10  # the pre-registered Newey-West lag of the current decision test
CURRENT_BLOCK = 20  # the current second view's bootstrap block
SQRT_FACTOR = 1.3  # M = ceil(1.3 sqrt(T))
TENTH = 0.1  # M = ceil(0.1 T)
FIXED_B_STEPS = 1000  # discretization of the fixed-b limit (Kiefer and Vogelsang's)
FIXED_B_DRAWS = 50_000  # before symmetrization (100,000 values)
FIXED_B_SEED = 20261017
FIXED_B_CHUNK = 2_000
# Kiefer and Vogelsang (2005): the Bartlett kernel's fixed-b critical value
# function a0 + a1 b + a2 b^2 + a3 b^3, at the 95% and 97.5% points.
KV_BARTLETT = {
    0.95: (1.6449, 2.1859, 0.3142, -0.3427),
    0.975: (1.9600, 2.9694, 0.4160, -0.5324),
}
# The packet's critical values at the planned T = 4,788 (19 folds of 252
# dates), at holm_levels(0.05, 4) = 1.25%, 1.67%, 2.5% and 5% (reproduced in
# tests): the fold-block t's t(18), and fixed_b_table's default simulation.
PLANNED_FOLD_T = (-2.4450, -2.3043, -2.1009, -1.7341)
PLANNED_FIXED_B = {
    "nw_fixed_b_sqrt": (-2.3196, -2.2006, -2.0216, -1.6859),  # M = 90, b = 0.0188
    "nw_fixed_b_tenth": (-2.6372, -2.4837, -2.2456, -1.8390),  # M = 479, b = 0.1000
}


@dataclass(frozen=True, eq=False)
class Statistics:
    """One procedure on a batch: mean and se (rows,), critical values (rows, levels)."""

    mean: np.ndarray
    se: np.ndarray
    critical: np.ndarray

    def rejects(self, shift: float | np.ndarray = 0.0) -> np.ndarray:
        """(rows, levels): is (mean - shift) / se below each level's critical value?"""

        ratio = (self.mean - shift) / self.se
        return ratio[..., None] < self.critical


def holm_levels(alpha: float, tests: int) -> tuple[float, ...]:
    """The one-sided levels Holm's step-down compares with: alpha/m, ..., alpha/1."""

    return tuple(alpha / (tests - k) for k in range(tests))


# --- long-run variance ---


def _fft_length(minimum: int) -> int:
    """The smallest 2^a 3^b 5^c at least `minimum` (fast for numpy's FFT)."""

    best = 1 << max(1, (minimum - 1).bit_length())
    five = 1
    while five < best:
        three = five
        while three < best:
            two = three
            while two < minimum:
                two *= 2
            best = min(best, two)
            three *= 3
        five *= 5
    return best


def autocovariances(series: np.ndarray) -> np.ndarray:
    """
    gamma_0..gamma_(T-1) of each row about its own mean, divided by T (the
    biased estimator of inference.long_run_variance), by FFT: shape (rows, T).
    """

    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    length = x.shape[1]
    size = _fft_length(2 * length - 1)
    spectrum = np.fft.rfft(x, n=size, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    return np.fft.irfft(power, n=size, axis=1)[:, :length] / length


def bartlett_lrv(acov: np.ndarray, bandwidth: int) -> np.ndarray:
    """gamma_0 + 2 sum_(l < M) (1 - l/M) gamma_l: inference.long_run_variance at lag M - 1."""

    lags = np.arange(1, bandwidth)
    weights = 1.0 - lags / bandwidth
    return acov[:, 0] + 2.0 * acov[:, 1:bandwidth] @ weights


def sqrt_bandwidth(length: int) -> int:
    return math.ceil(SQRT_FACTOR * math.sqrt(length))


def tenth_bandwidth(length: int) -> int:
    return math.ceil(TENTH * length)


# --- fixed-b critical values ---


def kiefer_vogelsang(b: float, level: float) -> float:
    """The published Bartlett fixed-b critical value at `level` (0.95 or 0.975)."""

    a0, a1, a2, a3 = KV_BARTLETT[level]
    return a0 + a1 * b + a2 * b**2 + a3 * b**3


@functools.lru_cache(maxsize=8)
def _fixed_b_draws(bandwidth: int, steps: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    values = []
    for start in range(0, draws, FIXED_B_CHUNK):
        count = min(FIXED_B_CHUNK, draws - start)
        z = rng.standard_normal((count, steps))
        lrv = bartlett_lrv(autocovariances(z), bandwidth)
        values.append(math.sqrt(steps) * z.mean(axis=1) / np.sqrt(lrv))
    t = np.concatenate(values)
    out = np.sort(np.concatenate([t, -t]))
    out.setflags(write=False)
    return out


def fixed_b_draws(
    b: float,
    *,
    steps: int = FIXED_B_STEPS,
    draws: int = FIXED_B_DRAWS,
    seed: int = FIXED_B_SEED,
) -> np.ndarray:
    """
    Sorted draws of the Bartlett fixed-b limit t = W(1) / sqrt(Q_b): the
    Newey-West t-ratio of iid N(0, 1) series of `steps` values at bandwidth
    M = round(b * steps), symmetrized (each draw and its negative).
    """

    bandwidth = max(1, round(b * steps))
    return _fixed_b_draws(bandwidth, steps, draws, seed)


def fixed_b_critical_values(b: float, levels: Sequence[float], **kwargs) -> np.ndarray:
    """Lower-tail critical values at each one-sided level (negative numbers)."""

    return np.quantile(fixed_b_draws(b, **kwargs), np.asarray(levels), method="linear")


# --- block bootstrap variances (exact, B -> infinity) ---


def circular_block_variance(series: np.ndarray, block: int) -> np.ndarray:
    """
    The circular block bootstrap variance of the mean, as
    inference.draw_circular_blocks draws it: k = ceil(T / block) blocks with
    independent uniform starts, the last cut to T - (k - 1) block sessions.
    Block sums are independent, so
        Var*(mean*) = [(k - 1) V(block) + V(last)] / T^2,
    with V(l) the variance over start positions of a circular l-session sum.
    """

    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    rows, length = x.shape
    count = -(-length // block)
    last = length - block * (count - 1)
    cumulative = np.concatenate([np.zeros((rows, 1)), np.cumsum(np.hstack([x, x]), axis=1)], 1)

    def spread(width: int) -> np.ndarray:
        sums = cumulative[:, width : width + length] - cumulative[:, :length]
        return (sums**2).mean(axis=1)  # circular sums of a centred series have mean zero

    return ((count - 1) * spread(block) + spread(last)) / length**2


def stationary_bootstrap_variance(acov: np.ndarray, mean_block: np.ndarray) -> np.ndarray:
    """
    Politis and Romano (1994), Lemma 1: with q = 1 - 1/L,
        Var*(mean*) = [gamma_0 + 2 sum_(i=1)^(T-1) b(i) gamma_i] / T,
        b(i) = (1 - i/T) q^i + (i/T) q^(T - i).
    """

    length = acov.shape[1]
    q = (1.0 - 1.0 / np.asarray(mean_block, dtype=np.float64))[:, None]
    lags = np.arange(1, length, dtype=np.float64)[None, :]
    with np.errstate(under="ignore"):
        weights = (1.0 - lags / length) * q**lags + (lags / length) * q ** (length - lags)
    return (acov[:, 0] + 2.0 * (weights * acov[:, 1:]).sum(axis=1)) / length


def politis_white_block(acov: np.ndarray) -> np.ndarray:
    """
    The stationary bootstrap's expected block length (Politis and White 2004,
    with the Patton, Politis and White 2009 correction), per row:
      K = max(5, ceil(sqrt(log10 T))), m_max = ceil(sqrt(T)) + K;
      m = the smallest lag after which K autocorrelations in a row are below
          2 sqrt(log10(T) / T) in absolute value (m_max if none);
      M = min(2 max(m, 1), m_max), and the flat-top window
          lambda(s) = 1 for |s| <= 1/2, 2 (1 - |s|) for 1/2 < |s| <= 1;
      g = gamma_0 + 2 sum_(k<=M) lambda(k/M) gamma_k,
      G = 2 sum_(k<=M) lambda(k/M) k gamma_k;
      L = (G^2 / g^2)^(1/3) T^(1/3)   (= (2 G^2 / D_SB)^(1/3) T^(1/3), D_SB = 2 g^2),
    kept within [1, ceil(min(3 sqrt(T), T / 3))].
    """

    rows, length = acov.shape
    run = max(5, math.ceil(math.sqrt(math.log10(length))))
    m_max = math.ceil(math.sqrt(length)) + run
    cap = math.ceil(min(3.0 * math.sqrt(length), length / 3.0))
    threshold = 2.0 * math.sqrt(math.log10(length) / length)
    rho = np.abs(acov[:, 1 : m_max + run + 1] / acov[:, :1])  # lags 1..m_max + run
    small = np.concatenate([np.zeros((rows, 1)), np.cumsum(rho < threshold, axis=1)], axis=1)
    # Lags m + 1..m + run are all small, for m = 0..m_max.
    quiet = (small[:, run : run + m_max + 1] - small[:, : m_max + 1]) == run
    first = np.where(quiet.any(axis=1), quiet.argmax(axis=1), m_max)
    window = np.minimum(2 * np.maximum(first, 1), m_max)[:, None]
    lags = np.arange(1, m_max + 1)[None, :]
    s = lags / window
    flat_top = np.where(s <= 0.5, 1.0, np.where(s <= 1.0, 2.0 * (1.0 - s), 0.0))
    gamma = acov[:, 1 : m_max + 1]
    g = acov[:, 0] + 2.0 * (flat_top * gamma).sum(axis=1)
    big_g = 2.0 * (flat_top * lags * gamma).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        block = (big_g**2 / g**2) ** (1.0 / 3.0) * length ** (1.0 / 3.0)
    return np.clip(np.nan_to_num(block, nan=1.0, posinf=cap), 1.0, cap)


# --- the procedures ---


def _student_t_critical(levels: Sequence[float], df: np.ndarray) -> np.ndarray:
    """(rows, levels) lower-tail t critical values for each row's degrees of freedom."""

    df = np.asarray(df, dtype=np.int64)
    unique, inverse = np.unique(df, return_inverse=True)
    table = np.array([[student_t_ppf(level, float(d)) for level in levels] for d in unique])
    return table[inverse.reshape(-1)]


def _constant(values: np.ndarray, rows: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(values, dtype=np.float64), (rows, len(values)))


def procedure_statistics(
    series: np.ndarray,
    folds: np.ndarray,
    levels: Sequence[float],
    *,
    fixed_b: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Statistics]:
    """
    Every procedure in PROCEDURES on a batch of per-date series (rows, T).

    `folds` labels each date with its fold (0..F - 1). `fixed_b` may pass the
    procedures' fixed-b critical values at `levels` (computed here otherwise),
    so a simulation computes them once.
    """

    x = np.asarray(series, dtype=np.float64)
    rows, length = x.shape
    levels = tuple(levels)
    mean = x.mean(axis=1)
    acov = autocovariances(x)
    normal = np.array([normal_ppf(level) for level in levels])
    out = {}

    lrv = bartlett_lrv(acov, CURRENT_LAG + 1)
    out["nw_lag10"] = Statistics(mean, np.sqrt(lrv / length), _constant(normal, rows))

    labels = np.asarray(folds)
    fold_count = int(labels.max()) + 1
    member = np.zeros((length, fold_count))
    member[np.arange(length), labels] = 1.0
    fold_means = (x @ member) / member.sum(axis=0)
    fold_se = fold_means.std(axis=1, ddof=1) / math.sqrt(fold_count)
    fold_t = _constant([student_t_ppf(level, fold_count - 1) for level in levels], rows)
    out["fold_block_t"] = Statistics(fold_means.mean(axis=1), fold_se, fold_t)

    for name, bandwidth in (
        ("nw_fixed_b_sqrt", sqrt_bandwidth(length)),
        ("nw_fixed_b_tenth", tenth_bandwidth(length)),
    ):
        b = bandwidth / length
        critical = (fixed_b or {}).get(name)
        if critical is None:
            critical = fixed_b_critical_values(b, levels)
        se = np.sqrt(bartlett_lrv(acov, bandwidth) / length)
        out[name] = Statistics(mean, se, _constant(critical, rows))

    blocks = -(-length // CURRENT_BLOCK)
    cbb_t = _constant([student_t_ppf(level, blocks - 1) for level in levels], rows)
    out["cbb_block20"] = Statistics(mean, np.sqrt(circular_block_variance(x, CURRENT_BLOCK)), cbb_t)

    block = politis_white_block(acov)
    sb_df = np.maximum(np.floor(length / block) - 1, 1)
    out["sb_politis_white"] = Statistics(
        mean,
        np.sqrt(stationary_bootstrap_variance(acov, block)),
        _student_t_critical(levels, sb_df),
    )
    return out


def fixed_b_table(length: int, levels: Sequence[float], **kwargs) -> dict[str, np.ndarray]:
    """Both fixed-b procedures' lower-tail critical values at `levels` for T = `length`."""

    return {
        "nw_fixed_b_sqrt": fixed_b_critical_values(
            sqrt_bandwidth(length) / length, levels, **kwargs
        ),
        "nw_fixed_b_tenth": fixed_b_critical_values(
            tenth_bandwidth(length) / length, levels, **kwargs
        ),
    }


__all__: Sequence[str] = ()
