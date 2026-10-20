"""
How often the briefing's matched-comparison interval covers the true mean
difference (research side: the briefing never imports this module).

The interval (briefing.reliability.block_t_interval) is the mean of T per-date
differences with a standard error from blocks of COMPARISON_BLOCK consecutive
dates (a shorter remainder merged into the last whole block) and a t interval
on (blocks - 1) degrees of freedom; the briefing needs at least 100 dates. Its
coverage is the share of simulated records whose interval at the nominal level
contains the true mean. It is a reporting interval, never a decision test.

The persistence check (briefing.reliability.persistence_ratio) sets the same
SE from blocks of 40 dates against it, from 120 dates. Per size, a cell records
how often SE40 / SE20 exceeds each of `thresholds` (the briefing flags RPS above
reliability.PERSISTENCE_THRESHOLD): its false-alarm rate on independent dates
and the symmetric placebo, and its detection rate under persistence and a shift.

Series. A panel is one record of per-date differences d_1..d_T in time order,
simulated at the longest size and read at each size as its first T dates.
    ar1      d_t = phi d_(t-1) + sqrt(1 - phi^2) e_t with standardized Student-t
             shocks e_t, started at its stationary variance, for each
             persistence phi.
    panel    2 to 5 matched stocks a date (drawn per date):
             d_(t,s) = a_t + sqrt(rho) c_t + sqrt(1 - rho) e_(t,s), with a_t an
             ar1 date effect and c_t a shock the date's stocks share
             (cross-sectional correlation rho). d_t is the mean over the date's
             stocks, as matched_comparison takes it.
    break    independent dates (ar1 with phi 0) whose mean steps up by `delta`
             (in per-date standard deviations) at date T // 2 of each size T,
             centred so the mean over the T dates is 0 (`mid_record_step`)
    scores   RPS, Brier and log loss differences (lower is better), one cell
             each, on simulated prices: planned_simulation.garch_blocks
             (a common GARCH(1,1) variance, pairwise correlation 0.5, Student-t
             shocks, 2.5% daily volatility) for `names` stocks, each with two
             shadow stocks that are never scored; the band and the five-session
             total's band times sqrt(5). B1 (estimators.b0 / b1, updated session
             by session) reads each target's past categories only, so the
             five-session total's B1 at date t has seen the totals ending by t.
             For target "1" (the next session) and "cum" (the five-session total):
               placebo      B1 read from one shadow's history minus B1 read from
                            the other's: exactly antisymmetric, true mean 0
               own_history  B1 read from a shadow's history minus the stock's own
                            B1 (the A7 own-history null): true mean exactly 0
               volatility   a volatility-scaled forecaster (normal probabilities
                            of +-band / an EWMA volatility of the stock's own
                            returns) minus the stock's own B1: a real difference
                            that moves with the volatility regime. Its true mean
                            at size T is the mean of the first T dates over an
                            independent run of `truth_reps` panels.
             A score cell also records the skewness of its per-date differences
             (log-loss differences are skewed, and their coverage is per score).
The interval is location- and scale-equivariant, so coverage does not depend on
the mean or the scale of a series: ar1 and panel use mean 0.

The rule before 2026-10-20 counted a shorter last block as a whole one, with
the unweighted spread of the block means (`previous_interval`). Its coverage is
reported beside the current rule's.

Randomness uses numpy's legacy RandomState, whose stream is frozen across
numpy versions (NEP 19).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from stock_agent.stock_lab import planned_simulation as ps
from stock_agent.stock_lab import scores
from stock_agent.stock_lab.briefing import reliability
from stock_agent.stock_lab.size_simulation import categorize

SIZES = (100, 101, 110, 119, 120, 150, 200, 250, 300, 400, 500)
PERSISTENCE = (0.0, 0.25, 0.5, 0.75, 0.9)
PANEL_PERSISTENCE = (0.0, 0.5)
TARGETS = ("1", "cum")
SCORE_SERIES = ("placebo", "own_history", "volatility")
SCORE_NAMES = reliability.SCORE_NAMES  # rps, brier, log_loss
BREAKS = (0.5, 1.0)  # mid-record steps of the mean, in per-date standard deviations
THRESHOLDS = (1.2, reliability.PERSISTENCE_THRESHOLD)  # the starting point, and the one adopted
LAGS = (1, 2, 5, 10, 20)
SEED = 20261020
HORIZONS = 5
K = 3  # Decrease, Stable, Increase


@dataclass(frozen=True)
class CoverageDesign:
    reps: int = 10_000  # ar1 and panel records per cell
    score_reps: int = 4_000
    truth_reps: int = 4_000  # the volatility series' true mean
    seed: int = SEED
    sizes: tuple[int, ...] = SIZES
    persistence: tuple[float, ...] = PERSISTENCE
    panel_persistence: tuple[float, ...] = PANEL_PERSISTENCE
    breaks: tuple[float, ...] = BREAKS
    within_date: float = 0.5  # rho, the panel's cross-sectional correlation
    min_stocks: int = 2
    max_stocks: int = 5
    t_df: float = 5.0
    names: int = 4  # the watchlist
    half_life: float = 63.0
    warmup: int = 756  # three years of sessions before the first date
    correlation: float = 0.5
    daily_vol: float = 0.025
    arch: float = 0.06
    garch: float = 0.92
    tau: float = 0.005
    alpha_prior: float = 50.0
    kappa: float = 40.0
    vol_decay: float = 0.94
    chunk: int = 250
    level: float = reliability.LEVEL
    block: int = reliability.COMPARISON_BLOCK
    long_block: int = reliability.PERSISTENCE_BLOCK
    thresholds: tuple[float, ...] = THRESHOLDS


def previous_interval(
    values: np.ndarray, level: float = reliability.LEVEL, block: int = reliability.COMPARISON_BLOCK
) -> reliability.BlockInterval:
    """The rule before 2026-10-20: ceil(T / block) blocks, unweighted (module docstring)."""

    values = np.asarray(values, dtype="float64")
    dates = values.shape[-1]
    starts = np.arange(0, dates, block)
    means = np.add.reduceat(values, starts, axis=-1) / np.diff(np.append(starts, dates))
    blocks = len(starts)
    estimate = values.mean(axis=-1)
    se = means.std(axis=-1, ddof=1) / math.sqrt(blocks)
    q = reliability.t_quantile(level, blocks - 1)
    return reliability.BlockInterval(estimate, se, estimate - q * se, estimate + q * se, blocks)


# --- the series ---


def ar1(
    rng: np.random.RandomState, reps: int, dates: int, phi: float, *, t_df: float
) -> np.ndarray:
    """(reps, dates) AR(1) series with unit variance from the first date."""

    scale = math.sqrt((t_df - 2.0) / t_df)
    shocks = rng.standard_t(t_df, size=(reps, dates)) * scale
    out = np.empty_like(shocks)
    out[:, 0] = shocks[:, 0]
    innovation = math.sqrt(1.0 - phi**2)
    for t in range(1, dates):
        out[:, t] = phi * out[:, t - 1] + innovation * shocks[:, t]
    return out


def panel(
    rng: np.random.RandomState, reps: int, dates: int, phi: float, design: CoverageDesign
) -> np.ndarray:
    """(reps, dates) per-date means of a date-clustered panel (module docstring)."""

    scale = math.sqrt((design.t_df - 2.0) / design.t_df)
    effect = ar1(rng, reps, dates, phi, t_df=design.t_df)
    counts = rng.randint(design.min_stocks, design.max_stocks + 1, size=(reps, dates))
    common = rng.standard_t(design.t_df, size=(reps, dates)) * scale
    own = rng.standard_t(design.t_df, size=(reps, dates, design.max_stocks)) * scale
    present = np.arange(design.max_stocks) < counts[..., None]
    own_mean = (own * present).sum(axis=-1) / counts
    rho = design.within_date
    return effect + math.sqrt(rho) * common + math.sqrt(1.0 - rho) * own_mean


def mid_record_step(dates: int) -> np.ndarray:
    """A step of 1 at date dates // 2, centred so its mean over the `dates` dates is 0."""

    later = np.arange(dates) >= dates // 2
    return later - later.mean()


_erfc = np.frompyfunc(math.erfc, 1, 1)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-z / math.sqrt(2.0)).astype("float64")


class _B1:
    """B0 and B1 (estimators.b0 / b1) for every column, updated one category at a time."""

    def __init__(self, reps: int, columns: int, design: CoverageDesign):
        self.decay = 2.0 ** (-1.0 / design.half_life)
        self.alpha, self.kappa = design.alpha_prior, design.kappa
        self.sums = np.zeros((reps, columns, K))
        self.counts = np.zeros((reps, columns, K))
        self.pooled = np.zeros((reps, K))
        self.weight = 0.0
        self.seen = 0

    def update(self, codes: np.ndarray) -> None:
        one = np.eye(K)[codes]
        self.counts += one
        self.pooled += one.sum(axis=1)
        self.sums = self.decay * self.sums + one
        self.weight = self.decay * self.weight + 1.0
        self.seen += 1

    def cumulative(self) -> np.ndarray:
        """P(Decrease) and P(Decrease or Stable): (reps, columns, 2)."""

        prior = self.pooled / self.pooled.sum(axis=1, keepdims=True)
        b0 = (self.counts + self.alpha * prior[:, None, :]) / (self.seen + self.alpha)
        b1 = (self.sums + self.kappa * b0) / (self.weight + self.kappa)
        return np.cumsum(b1, axis=-1)[..., :2]


def _rps_difference(q: np.ndarray, p: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """RPS(q) - RPS(p) from cumulative probabilities (size_simulation's identity)."""

    return 0.5 * ((q - p) * (q + p - 2.0 * observed)).sum(axis=-1)


def _scored(cumulative: np.ndarray, y: np.ndarray, name: str) -> np.ndarray:
    """A score (the scorer's own function) of cumulative probabilities (..., 2) on codes y."""

    p = np.stack(
        [cumulative[..., 0], cumulative[..., 1] - cumulative[..., 0], 1.0 - cumulative[..., 1]],
        axis=-1,
    )
    return scores.SCORES[name](p.reshape(-1, K), y.reshape(-1)).reshape(y.shape)


def _differences(q: np.ndarray, p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Score(q) - score(p) per stock, for every score of SCORE_NAMES."""

    observed = np.stack([y == 0, y <= 1], axis=-1).astype("float64")
    out = {"rps": _rps_difference(q, p, observed)}
    for name in SCORE_NAMES:
        if name != "rps":
            out[name] = _scored(q, y, name) - _scored(p, y, name)
    return out


def score_panels(
    rng: np.random.RandomState, reps: int, dates: int, design: CoverageDesign
) -> dict[tuple[str, str, str], np.ndarray]:
    """{(target, series, score): (reps, dates)} per-date score differences (module docstring)."""

    names = design.names
    columns = 3 * names  # the stocks, then each stock's first and second shadow
    sessions = design.warmup + dates + HORIZONS
    blocks = ps.garch_blocks(
        rng,
        reps,
        sessions,
        columns,
        correlation=design.correlation,
        daily_vol=design.daily_vol,
        t_df=design.t_df,
        arch=design.arch,
        garch=design.garch,
    )
    returns = np.concatenate(list(blocks), axis=1)
    daily = categorize(returns, design.tau)
    level = np.concatenate([np.zeros((reps, 1, columns)), np.cumsum(returns, axis=1)], axis=1)
    # totals[:, t]: the category of the returns of sessions t + 1 .. t + 5
    totals = categorize(
        level[:, 1 + HORIZONS :] - level[:, 1:-HORIZONS], design.tau * math.sqrt(HORIZONS)
    )
    state = {target: _B1(reps, columns, design) for target in TARGETS}
    variance = np.full((reps, names), design.daily_vol**2)
    out = {
        (t, s, name): np.empty((reps, dates))
        for t in TARGETS
        for s in SCORE_SERIES
        for name in SCORE_NAMES
    }
    for t in range(design.warmup + dates):
        variance = (
            design.vol_decay * variance + (1.0 - design.vol_decay) * returns[:, t, :names] ** 2
        )
        state["1"].update(daily[:, t])
        if t >= HORIZONS:
            state["cum"].update(totals[:, t - HORIZONS])  # the total that ended at session t
        if t < design.warmup:
            continue
        low = _normal_cdf(-design.tau / np.sqrt(variance))  # the same z for the total
        volatility = np.stack([low, 1.0 - low], axis=-1)
        for target, b1 in state.items():
            cumulative = b1.cumulative()
            own, first, second = (cumulative[:, k * names : (k + 1) * names] for k in range(3))
            y = daily[:, t + 1, :names] if target == "1" else totals[:, t, :names]
            pairs = {
                "placebo": (first, second),
                "own_history": (first, own),
                "volatility": (volatility, own),
            }
            for series, (q, p) in pairs.items():
                for name, difference in _differences(q, p, y).items():
                    out[target, series, name][:, t - design.warmup] = difference.mean(axis=-1)
    return out


def _scores(seed: int, reps: int, design: CoverageDesign) -> dict[tuple[str, str, str], np.ndarray]:
    rng = np.random.RandomState(seed)
    parts: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for start in range(0, reps, design.chunk):
        found = score_panels(rng, min(design.chunk, reps - start), max(design.sizes), design)
        for key, values in found.items():
            parts.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in parts.items()}


# --- coverage ---


def autocorrelations(series: np.ndarray, lags: Sequence[int] = LAGS) -> dict[str, float]:
    """The mean over records of each lag's sample autocorrelation."""

    centred = series - series.mean(axis=1, keepdims=True)
    variance = (centred**2).mean(axis=1)
    return {
        str(lag): float(((centred[:, lag:] * centred[:, :-lag]).mean(axis=1) / variance).mean())
        for lag in lags
    }


def skewness(series: np.ndarray) -> float:
    """The mean over records of each record's sample skewness."""

    centred = series - series.mean(axis=1, keepdims=True)
    return float(((centred**3).mean(axis=1) / (centred**2).mean(axis=1) ** 1.5).mean())


def persistence(values: np.ndarray, design: CoverageDesign) -> dict | None:
    """
    How often SE40 / SE20 (reliability.persistence_ratio) exceeds each
    threshold, and its median and 90th percentile; None below 3 blocks of 40.
    A record without spread (NaN) counts as not flagged.
    """

    ratio = reliability.persistence_ratio(values, design.block, design.long_block)
    if ratio is None:
        return None
    return {
        "long_blocks": values.shape[-1] // design.long_block,
        "flag_rate": {str(level): float((ratio > level).mean()) for level in design.thresholds},
        "median_ratio": float(np.nanmedian(ratio)),
        "q90_ratio": float(np.nanquantile(ratio, 0.9)),
    }


def coverage(
    series: np.ndarray,
    truth: float | Mapping[int, float],
    design: CoverageDesign,
    *,
    step: float = 0.0,
) -> list[dict]:
    """
    Per size, for the current and the previous rule: the coverage, the misses
    on each side (the truth above the interval means the estimate was too
    low), the median width, and the mean SE over the simulated standard
    deviation of the mean (below 1: the SE is too small on average); and the
    persistence check's flag rates. `step` adds a centred mid-record step of
    that height at each size (the break series).
    """

    out = []
    reps = series.shape[0]
    for size in design.sizes:
        target = truth if isinstance(truth, float) else truth[size]
        values = series[:, :size] + step * mid_record_step(size) if step else series[:, :size]
        row: dict = {"dates": size}
        for name, interval in (
            ("current", reliability.block_t_interval),
            ("previous", previous_interval),
        ):
            found = interval(values, design.level, design.block)
            rate = float(((found.low <= target) & (target <= found.high)).mean())
            row[name] = {
                "blocks": found.blocks,
                "coverage": rate,
                "mc_se": math.sqrt(rate * (1.0 - rate) / reps),
                "truth_above": float((target > found.high).mean()),
                "truth_below": float((target < found.low).mean()),
                "median_width": float(np.median(found.high - found.low)),
                "se_ratio": float(found.se.mean() / found.estimate.std(ddof=1)),
            }
        row["persistence"] = persistence(values, design)
        out.append(row)
    return out


def simulate(design: CoverageDesign) -> dict:
    """Every series' coverage at every size, as a JSON-ready record section."""

    longest = max(design.sizes)
    if min(design.sizes) < design.block * reliability.MIN_COMPARISON_BLOCKS:
        raise ValueError("the briefing shows no interval below 100 dates")
    rng = np.random.RandomState(design.seed)
    cells = []
    for kind, levels in (("ar1", design.persistence), ("panel", design.panel_persistence)):
        for phi in levels:
            if kind == "ar1":
                series = ar1(rng, design.reps, longest, phi, t_df=design.t_df)
            else:
                series = panel(rng, design.reps, longest, phi, design)
            cells.append(
                {
                    "series": kind,
                    "phi": phi,
                    "reps": design.reps,
                    "autocorrelation": autocorrelations(series),
                    "sizes": coverage(series, 0.0, design),
                }
            )
    for delta in design.breaks:  # drawn after the cells above, which therefore do not move
        noise = ar1(rng, design.reps, longest, 0.0, t_df=design.t_df)
        cells.append(
            {
                "series": "break",
                "delta": delta,
                "reps": design.reps,
                "autocorrelation": autocorrelations(noise + delta * mid_record_step(longest)),
                "sizes": coverage(noise, 0.0, design, step=delta),
            }
        )
    found = _scores(design.seed + 1, design.score_reps, design)
    truth = _scores(design.seed + 2, design.truth_reps, design)
    for (target, name, score), series in found.items():
        run = truth[target, name, score]
        means = {size: float(run[:, :size].mean()) for size in design.sizes}
        mc_se = float(run[:, :longest].mean(axis=1).std(ddof=1)) / math.sqrt(design.truth_reps)
        cells.append(
            {
                "series": f"scores:{name}",
                "target": target,
                "score": score,
                "reps": design.score_reps,
                "true_mean": 0.0 if name != "volatility" else means[longest],
                "truth_run_mean": means[longest],  # 0 within Monte Carlo error unless volatility
                "truth_run_mc_se": mc_se,
                "autocorrelation": autocorrelations(series),
                "skewness": skewness(series),
                "sizes": coverage(series, means if name == "volatility" else 0.0, design),
            }
        )
    return {"design": asdict(design), "cells": cells}
