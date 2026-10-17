"""
The transparent probability estimators (numpy and pandas only).

Every probability is a fitted estimator's output, never a fixed number.
Probabilities are arrays of shape (rows, 3) in category-code order
(Decrease, Stable, Increase); each row sums to 1.

The fold contract (statistical review, 2026-10-16). Parameters are frozen
when a test window opens: the half-life H, the prior strengths, the B2
tilts and the B3 error distributions are fitted on training rows only.
Estimator STATE (weighted counts, volatility, the run in progress) is
computed online from data up to each decision session d, which is
information available after d's close. State never uses a later session.

B0  unconditional frequencies with a Dirichlet prior toward the pooled
    frequencies of every stock in the panel (all through d):
        p_c = (n_c + alpha * pi_c) / (N + alpha)
B1  exponentially weighted frequencies, half-life H sessions, with a prior
    of strength kappa toward B0:
        p_c = (S_c + kappa * B0_c) / (W + kappa),
        S_c = sum_k lambda^k 1[c_(d-k) = c],  W = sum_k lambda^k,  lambda = 2^(-1/H)
    (k = 0 is session d; a missing session decays the sums without adding).
    Kish's effective sample size n_eff = W^2 / sum_k lambda^(2k).
B2  a run-length tilt of B1, coherent over the five sessions:
        P(c' | state s) proportional to B1_d(c') * exp(theta_(s, c'))
    with state s = (the run's direction, its length capped at 4) under the
    "breaks" rule, theta_(s, Stable) = 0, and theta fitted per state by
    penalized maximum likelihood (a Gaussian prior, `ridge`) with log B1 as
    an offset; theta = 0 is exactly B1. Five-session marginals and path
    probabilities come from the chain: each step's category moves the state
    deterministically (same direction: length + 1, capped; otherwise the new
    direction with length 1), and B1_d is held at its value at d.
B3  volatility-scaled empirical CDF, per horizon h:
        P(Decrease) = F_h(-tau / sigma_d strictly), P(Increase) = 1 - F_h(tau / sigma_d)
    where sigma_d is the EWMA volatility at d (features.ewma_vol) and F_h the
    empirical CDF of z = r_t / sigma_(t-h) on training rows, shrunk toward the
    pooled ECDF with weight kappa_F. The horizon enters through F_h, since the
    EWMA forecast of sigma is flat in h.

Floors. Scores use probabilities floored at FLOOR and renormalized, so a
log loss is always finite; the floor is recorded with every evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import Category, StableBand

K = 3  # Decrease, Stable, Increase
UNIFORM = np.full(K, 1.0 / K)
FLOOR = 1e-3
MAX_RUN = 4
DIRECTIONS = (Category.DECREASE.value, Category.STABLE.value, Category.INCREASE.value)
STATES = tuple((direction, length) for direction in DIRECTIONS for length in range(1, MAX_RUN + 1))
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
N_STATES = len(STATES)


def floor(probabilities: np.ndarray, eps: float = FLOOR) -> np.ndarray:
    """Probabilities floored at `eps` and renormalized (rows sum to 1)."""

    clipped = np.maximum(probabilities, eps)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _codes(panel: pd.DataFrame) -> np.ndarray:
    return panel["category_code"].to_numpy(dtype="float64", na_value=np.nan)


# --- B0 and B1: online frequency state ---


@dataclass(frozen=True)
class FrequencyParams:
    alpha: float = 50.0  # B0's prior strength toward the pooled frequencies
    half_life: float = 63.0  # B1's half-life in sessions
    kappa: float = 40.0  # B1's prior strength toward B0


def pooled_frequencies(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per session date, the pooled frequencies of every stock's categories
    through that date (expanding), as columns 0..2; uniform before any.
    """

    codes = _codes(panel)
    indicator = pd.DataFrame(
        {c: (codes == c).astype("float64") for c in range(K)}, index=panel.index
    )
    indicator["session"] = panel["session"].to_numpy()
    per_date = indicator.groupby("session").sum().sort_index().cumsum()
    total = per_date.sum(axis=1)
    freq = per_date.div(total.where(total > 0), axis=0)
    return freq.fillna(1.0 / K)


def b0(panel: pd.DataFrame, params: FrequencyParams) -> np.ndarray:
    """B0 for every row of `panel` (columns symbol, session, category_code)."""

    pooled = pooled_frequencies(panel).reindex(panel["session"].to_numpy()).to_numpy()
    out = np.empty((len(panel), K))
    codes = _codes(panel)
    for rows in panel.groupby("symbol", sort=False).indices.values():
        counts = np.stack([np.cumsum(codes[rows] == c) for c in range(K)], axis=1)
        total = counts.sum(axis=1, keepdims=True)
        out[rows] = (counts + params.alpha * pooled[rows]) / (total + params.alpha)
    return out


def ew_state(codes: np.ndarray, half_life: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, W, W2) of B1 for one stock's codes in session order (module docstring)."""

    lam = 2.0 ** (-1.0 / half_life)
    n = len(codes)
    sums, weight, weight2 = np.zeros((n, K)), np.zeros(n), np.zeros(n)
    s, w, w2 = np.zeros(K), 0.0, 0.0
    for i, code in enumerate(codes):
        s, w, w2 = s * lam, w * lam, w2 * lam * lam
        if not math.isnan(code):
            s[int(code)] += 1.0
            w += 1.0
            w2 += 1.0
        sums[i], weight[i], weight2[i] = s, w, w2
    return sums, weight, weight2


def b1(
    panel: pd.DataFrame, params: FrequencyParams, prior: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(B1 probabilities, Kish n_eff) for every row; `prior` defaults to B0."""

    prior = b0(panel, params) if prior is None else prior
    out, n_eff = np.empty((len(panel), K)), np.empty(len(panel))
    codes = _codes(panel)
    for rows in panel.groupby("symbol", sort=False).indices.values():
        sums, weight, weight2 = ew_state(codes[rows], params.half_life)
        out[rows] = (sums + params.kappa * prior[rows]) / (weight + params.kappa)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            n_eff[rows] = np.where(weight2 > 0, weight**2 / weight2, 0.0)
    return out, n_eff


# --- B2: the run-length tilt ---


def states_of(panel: pd.DataFrame) -> np.ndarray:
    """The state index of each row ("breaks" rule), -1 where the run is unknown."""

    direction = panel["run_direction"].to_numpy(dtype=object)
    length = np.minimum(panel["run_length"].to_numpy(dtype="int64"), MAX_RUN)
    out = np.full(len(panel), -1, dtype="int64")
    for i, (d, n) in enumerate(zip(direction, length, strict=True)):
        if isinstance(d, str) and n >= 1:
            out[i] = STATE_INDEX[(d, int(n))]
    return out


def next_state(state: int, code: int) -> int:
    direction, length = STATES[state]
    category = DIRECTIONS[code]
    if category == direction:
        return STATE_INDEX[(direction, min(length + 1, MAX_RUN))]
    return STATE_INDEX[(category, 1)]


NEXT = np.array([[next_state(s, c) for c in range(K)] for s in range(N_STATES)])


@dataclass(frozen=True)
class Tilt:
    """theta[state, code], with theta[:, Stable] = 0."""

    theta: np.ndarray
    ridge: float
    rows_per_state: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def zero(cls) -> Tilt:
        return cls(np.zeros((N_STATES, K)), ridge=0.0)


def fit_tilt(
    base: np.ndarray,
    states: np.ndarray,
    outcomes: np.ndarray,
    *,
    ridge: float = 10.0,
    iterations: int = 25,
) -> Tilt:
    """
    theta per state by Newton's method on the penalized multinomial
    log-likelihood with offset log(base): each state is its own 2-parameter
    problem (Decrease and Increase against Stable), with penalty ridge/2 * |theta|^2.
    Rows with an unknown state or outcome are skipped.
    """

    theta = np.zeros((N_STATES, K))
    counts = []
    offset = np.log(np.maximum(base, 1e-12))
    known = (states >= 0) & np.isfinite(outcomes)
    free = np.array([0, 2])  # Decrease and Increase; Stable is the reference
    for s in range(N_STATES):
        rows = known & (states == s)
        counts.append(int(rows.sum()))
        if not rows.any():
            continue
        off = offset[rows]
        y = np.eye(K)[outcomes[rows].astype(int)]
        t = np.zeros(2)
        for _ in range(iterations):
            logits = off.copy()
            logits[:, free] += t
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            gradient = (p[:, free] - y[:, free]).sum(axis=0) + ridge * t
            pf = p[:, free]
            hessian = np.einsum("ni,nj->ij", pf, -pf)
            hessian[np.diag_indices(2)] += pf.sum(axis=0)
            hessian[np.diag_indices(2)] += ridge
            step = np.linalg.solve(hessian, gradient)
            t -= step
            if np.abs(step).max() < 1e-10:
                break
        theta[s, free] = t
    return Tilt(theta, ridge=ridge, rows_per_state=tuple(counts))


def tilt_step(base: np.ndarray, tilt: Tilt) -> np.ndarray:
    """P(c' | state) for every row and state: shape (rows, states, 3)."""

    logits = np.log(np.maximum(base, 1e-12))[:, None, :] + tilt.theta[None, :, :]
    logits -= logits.max(axis=2, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=2, keepdims=True)


def _advance(dist: np.ndarray, step: np.ndarray) -> np.ndarray:
    """The state distribution after one more session on the chain."""

    new = np.zeros_like(dist)
    for s in range(N_STATES):
        for c in range(K):
            new[:, NEXT[s, c]] += dist[:, s] * step[:, s, c]
    return new


def b2_marginals(base: np.ndarray, states: np.ndarray, tilt: Tilt, horizons: int = 5) -> np.ndarray:
    """
    The chain's marginal probabilities at d+1..d+horizons: shape
    (rows, horizons, 3). A row whose state is unknown (after a gap) takes B1
    itself for d+1 and continues on the chain from the state that category
    starts.
    """

    n = len(base)
    step = tilt_step(base, tilt)  # (n, states, 3)
    known = states >= 0
    dist = np.zeros((n, N_STATES))
    dist[np.flatnonzero(known), states[known]] = 1.0
    out = np.empty((n, horizons, K))
    out[:, 0] = np.where(known[:, None], np.einsum("ns,nsk->nk", dist, step), base)
    after = _advance(dist, step)
    for c in range(K):
        after[~known, STATE_INDEX[(DIRECTIONS[c], 1)]] = base[~known, c]
    dist = after
    for h in range(1, horizons):
        out[:, h] = np.einsum("ns,nsk->nk", dist, step)
        dist = _advance(dist, step)
    return out


def b2_path_probabilities(base: np.ndarray, states: np.ndarray, tilt: Tilt) -> dict:
    """
    Joint path probabilities from the chain, for full-path calibration:
    "all_increase" / "all_decrease" (all five sessions one direction), and
    the nine joint cells of (d+1, d+2) as "joint_<c1><c2>". Rows with an
    unknown state are NaN.
    """

    step = tilt_step(base, tilt)
    n = len(base)
    known = states >= 0
    rows = np.flatnonzero(known)
    out = {}
    for name, code in (("all_decrease", 0), ("all_increase", 2)):
        prob = np.full(n, np.nan)
        state = states[rows].copy()
        value = np.ones(len(rows))
        for _ in range(5):
            value *= step[rows, state, code]
            state = NEXT[state, code]
        prob[rows] = value
        out[name] = prob
    for c1 in range(K):
        for c2 in range(K):
            prob = np.full(n, np.nan)
            s1 = states[rows]
            s2 = NEXT[s1, c1]
            prob[rows] = step[rows, s1, c1] * step[rows, s2, c2]
            out[f"joint_{c1}{c2}"] = prob
    return out


# --- B3: the volatility-scaled empirical CDF ---


@dataclass(frozen=True)
class ErrorDistribution:
    """Sorted standardized returns per stock and pooled, for one horizon."""

    horizon: int
    pooled: np.ndarray
    by_stock: Mapping[str, np.ndarray]
    kappa: float = 250.0


def standardized(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """z_t = r_t / sigma_(t - horizon) for one stock (NaN where unavailable)."""

    sigma = frame["ewma_vol"].shift(horizon).to_numpy(dtype="float64")
    r = frame["price_return"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = r / sigma
    return np.where(np.isfinite(z) & (sigma > 0), z, np.nan)


def fit_errors(
    training: Mapping[str, pd.DataFrame], horizon: int, kappa: float = 250.0
) -> ErrorDistribution:
    by_stock = {}
    for symbol, frame in training.items():
        z = standardized(frame, horizon)
        by_stock[symbol] = np.sort(z[np.isfinite(z)])
    pooled = np.sort(np.concatenate([*by_stock.values(), np.array([])]))
    return ErrorDistribution(horizon, pooled, by_stock, kappa)


def panel_standardized(panel: pd.DataFrame, horizon: int, *, sessions: int = 1) -> np.ndarray:
    """
    z for every panel row, computed on each stock's whole history (so a gap or
    a row outside a training set never shifts the alignment):
    r_t / sigma_(t - horizon) for one session, or the `sessions`-session total
    L_t / L_(t - sessions) - 1 over sigma_(t - sessions) * sqrt(sessions).
    """

    grouped = panel.groupby("symbol", sort=False)
    sigma = grouped["ewma_vol"].shift(horizon if sessions == 1 else sessions)
    if sessions == 1:
        move = panel["price_return"]
    else:
        move = panel["level"] / grouped["level"].shift(sessions) - 1.0
        sigma = sigma * math.sqrt(sessions)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (move / sigma).to_numpy(dtype="float64")
    return np.where(np.isfinite(z) & (sigma.to_numpy() > 0), z, np.nan)


def errors_from_panel(
    panel: pd.DataFrame, z: np.ndarray, rows: np.ndarray, horizon: int, kappa: float = 250.0
) -> ErrorDistribution:
    """The ECDFs of `z` over the rows in the boolean mask `rows` (training rows only)."""

    symbols = panel["symbol"].to_numpy(dtype=object)
    by_stock = {}
    for symbol in dict.fromkeys(symbols[rows]):
        values = z[rows & (symbols == symbol)]
        by_stock[symbol] = np.sort(values[np.isfinite(values)])
    pooled = np.sort(np.concatenate([*by_stock.values(), np.array([])]))
    return ErrorDistribution(horizon, pooled, by_stock, kappa)


def _ecdf(sorted_values: np.ndarray, x: np.ndarray, side: str) -> np.ndarray:
    if len(sorted_values) == 0:
        return np.full(len(x), np.nan)
    return np.searchsorted(sorted_values, x, side=side) / len(sorted_values)


def b3(
    errors: ErrorDistribution,
    symbols: Sequence[str],
    sigma: np.ndarray,
    band: StableBand,
) -> np.ndarray:
    """B3 probabilities for rows with stock `symbols` and EWMA volatility `sigma`."""

    symbols = np.asarray(symbols, dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = band.tau / sigma
    lower = -upper
    out = np.full((len(sigma), K), np.nan)
    pooled_low = _ecdf(errors.pooled, lower, "left")
    pooled_high = _ecdf(errors.pooled, upper, "right")
    for symbol in dict.fromkeys(symbols):
        rows = symbols == symbol
        own = errors.by_stock.get(symbol, np.array([]))
        n = len(own)
        weight = n / (n + errors.kappa)
        low = (
            pooled_low[rows]
            if n == 0
            else (weight * _ecdf(own, lower[rows], "left") + (1 - weight) * pooled_low[rows])
        )
        high = (
            pooled_high[rows]
            if n == 0
            else (weight * _ecdf(own, upper[rows], "right") + (1 - weight) * pooled_high[rows])
        )
        out[rows, 0] = low
        out[rows, 2] = 1.0 - high
        out[rows, 1] = high - low
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    out[bad] = np.nan
    return out
