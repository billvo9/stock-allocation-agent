"""
The significance tests of the Stock Lab's evaluation, and how often each one
is fooled by a challenger with no skill (research side: the Lab's page never
imports this module; it reads the stored result).

The tests. Every comparison reduces to one loss differential per date (a
challenger's RPS minus B1's, averaged across stocks). Its mean gets three
tests, all one-sided toward "the challenger is better" (a negative mean):
    newey_west                Bartlett long-run variance, lag `lag`, normal
                              critical values (the pre-registered decision test);
    fold_block_t              a t-test on the per-fold means, t(folds - 1);
    circular_block_bootstrap  the bootstrap SE of the mean over circular blocks
                              of whole dates, t(blocks - 1).
`one_sided_tests` is the only place they are computed, for the evaluation and
for the simulation alike.

The size simulation. A test's size is how often it declares a difference when
there is none. Panels shaped like the evaluation's (its number of stocks and
test years, after a three-year warm-up) are drawn with returns that are
independent over time and equicorrelated across stocks (pairwise correlation
`correlation`, Student-t innovations), so no estimator has skill. The
challenger is a placebo, B1 at the same half-life read from the NEXT stock's
history, against one of two references (PLACEBOS):
    other_vs_own    the stock's own B1;
    other_vs_other  B1 read from the PREVIOUS stock's history.
At every date the two sides are equally good in expectation (neither history
says anything about the stock's next sessions, and the stocks are alike), so
each date's differential has mean exactly zero. The differential keeps what
makes inference hard on the real panel: B1's slow state (the half-life), the
five daily targets of one date overlapping the next date's, and stocks moving
together.
    other_vs_other is also symmetric in distribution (listing the stocks
    backwards swaps the two sides and flips its sign), so its size measures
    the tests alone.
    other_vs_own is not: the stock's own B1 later learns from the very
    sessions it was scored on, and the placebo never does. Real comparisons
    have that asymmetry too (B1, B3 and M1 learn from a stock's past in
    different ways), so its size shows how much it adds.
The share of panels each test rejects at the nominal level is its simulated
size. It is reported beside the results and never used to select anything.

Randomness uses numpy's legacy RandomState, whose stream is frozen across
numpy versions (NEP 19).
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from stock_agent.model_diagnostics.inference import (
    BlockDraws,
    Inference,
    bootstrap_mean,
    draw_circular_blocks,
    fold_block_t,
    newey_west_mean,
)

METHODS = ("newey_west", "fold_block_t", "circular_block_bootstrap")
PLACEBOS = ("other_vs_own", "other_vs_other")
HORIZONS = 5  # the five daily targets of one decision date
CHUNK = 50  # panels simulated at once
K = 3  # Decrease, Stable, Increase


def one_sided_tests(
    values: np.ndarray, blocks: np.ndarray, draws: BlockDraws, *, lag: int
) -> dict[str, Inference]:
    """The three tests of one per-date differential (module docstring)."""

    return {
        "newey_west": newey_west_mean(values, lag=lag),
        "fold_block_t": fold_block_t(values, blocks),
        "circular_block_bootstrap": bootstrap_mean(values, draws),
    }


def one_sided_p(result: Inference) -> float:
    """P(a mean this low | no difference), from the two-sided p; NaN if unavailable."""

    p = result.p_value
    if math.isnan(p):
        return math.nan
    return float(0.5 * p) if result.estimate < 0 else float(1.0 - 0.5 * p)


# --- the no-skill panels ---


def null_returns(
    rng: np.random.RandomState,
    reps: int,
    sessions: int,
    names: int,
    *,
    correlation: float,
    daily_vol: float,
    t_df: float,
) -> np.ndarray:
    """Returns (reps, sessions, names): a common and an own Student-t shock, unit variance each."""

    scale = math.sqrt((t_df - 2.0) / t_df)
    market = rng.standard_t(t_df, size=(reps, sessions, 1)) * scale
    own = rng.standard_t(t_df, size=(reps, sessions, names)) * scale
    return daily_vol * (math.sqrt(correlation) * market + math.sqrt(1.0 - correlation) * own)


def categorize(returns: np.ndarray, tau: float) -> np.ndarray:
    """Category codes as prices.categorize: Increase above tau, Decrease below -tau, else Stable."""

    return (1 + (returns > tau).astype(np.int8) - (returns < -tau).astype(np.int8)).astype(np.int8)


def placebo_differentials(
    codes: np.ndarray,
    half_lives: Sequence[float],
    *,
    alpha_prior: float,
    kappa: float,
    first: int,
) -> np.ndarray:
    """
    Per decision date t (from `first` to the last date with five outcomes), the
    mean over stocks and the five daily targets of RPS(B1 of the next stock)
    minus RPS(the reference's B1), for each placebo in PLACEBOS: shape
    (placebos, half-lives, reps, dates). Stocks are in a circle (the next of
    the last is the first). `codes` has shape (reps, sessions, names) with
    every session observed.

    B0 and B1 are the estimators' formulas (estimators.b0 / b1, tested equal),
    updated session by session for every half-life at once. Because B1's
    forecast is the same for the five targets, the mean RPS over them uses the
    targets' mean outcome indicators O_k:
        mean_h RPS(q, y_h) - mean_h RPS(p, y_h) = 1/2 sum_k (Q_k - P_k)(Q_k + P_k - 2 O_k)
    with Q, P the cumulative probabilities of {Decrease} and {Decrease, Stable}.
    """

    reps, sessions, names = codes.shape
    decay = (2.0 ** (-1.0 / np.asarray(half_lives, dtype=np.float64)))[:, None, None, None]
    eye = np.eye(K)
    sums = np.zeros((len(half_lives), reps, names, K))
    weight = np.zeros((len(half_lives), 1, 1, 1))
    counts = np.zeros((reps, names, K))
    pooled = np.zeros((reps, K))
    out = np.empty((len(PLACEBOS), len(half_lives), reps, sessions - HORIZONS - first))
    for t in range(sessions - HORIZONS):
        one = eye[codes[:, t]]
        counts += one
        pooled += one.sum(axis=1)
        sums = decay * sums + one
        weight = decay * weight + 1.0
        if t < first:
            continue
        prior = pooled / pooled.sum(axis=1, keepdims=True)
        b0 = (counts + alpha_prior * prior[:, None, :]) / (t + 1 + alpha_prior)
        b1 = (sums + kappa * b0) / (weight + kappa)
        own = np.cumsum(b1, axis=-1)[..., :2]
        following = np.roll(own, -1, axis=2)  # stock i reads stock i + 1's history
        previous = np.roll(own, 1, axis=2)  # and stock i - 1's
        future = codes[:, t + 1 : t + 1 + HORIZONS]
        observed = np.stack([(future == 0).mean(axis=1), (future <= 1).mean(axis=1)], axis=-1)
        for index, reference in enumerate((own, previous)):
            diff = 0.5 * ((following - reference) * (following + reference - 2.0 * observed))
            out[index, :, :, t - first] = diff.sum(axis=-1).mean(axis=-1)
    return out


# --- the simulation ---


@dataclass(frozen=True)
class SizeDesign:
    names: int
    years: int
    half_lives: tuple[float, ...]
    reps: int
    seed: int
    nw_lag: int
    block: int
    bootstrap_reps: int
    bootstrap_seed: int
    correlation: float = 0.4
    daily_vol: float = 0.015
    t_df: float = 5.0
    tau: float = 0.005
    alpha_prior: float = 50.0
    kappa: float = 40.0
    warmup_years: int = 3
    sessions_per_year: int = 252
    nominal: float = 0.05


def simulate_size(design: SizeDesign) -> dict:
    """The simulated size of each test (module docstring), as a JSON-ready record section."""

    return copy.deepcopy(_simulate(design))


@functools.lru_cache(maxsize=8)
def _simulate(design: SizeDesign) -> dict:
    record = {"use": "reported beside the comparisons; never used to select an estimator"}
    if design.names < 3:
        # The placebos read the next and the previous stock's histories: two other stocks.
        return {
            **record,
            "status": "unavailable",
            "reason": "needs_three_stocks",
            "design": _jsonable(asdict(design)),
            "rates": [],
            "null_mean": [],
        }
    rng = np.random.RandomState(design.seed)
    first = design.warmup_years * design.sessions_per_year
    dates = design.years * design.sessions_per_year
    sessions = first + dates + HORIZONS
    blocks = np.arange(dates) // design.sessions_per_year
    draws = draw_circular_blocks(
        dates, block_length=design.block, reps=design.bootstrap_reps, seed=design.bootstrap_seed
    )
    shape = (len(PLACEBOS), len(design.half_lives), len(METHODS))
    rejections = np.zeros(shape, dtype=np.int64)
    means = []
    for start in range(0, design.reps, CHUNK):
        count = min(CHUNK, design.reps - start)
        returns = null_returns(
            rng,
            count,
            sessions,
            design.names,
            correlation=design.correlation,
            daily_vol=design.daily_vol,
            t_df=design.t_df,
        )
        differentials = placebo_differentials(
            categorize(returns, design.tau),
            design.half_lives,
            alpha_prior=design.alpha_prior,
            kappa=design.kappa,
            first=first,
        )
        means.append(differentials.mean(axis=3))
        for index in np.ndindex(differentials.shape[:3]):
            tests = one_sided_tests(differentials[index], blocks, draws, lag=design.nw_lag)
            for j, method in enumerate(METHODS):
                rejections[index[0], index[1], j] += one_sided_p(tests[method]) < design.nominal
    means = np.concatenate(means, axis=2)
    limit = design.nominal + 2.0 * math.sqrt(design.nominal * (1 - design.nominal) / design.reps)
    rates = []
    for (p, i, j), count in np.ndenumerate(rejections):
        rate = float(count / design.reps)
        rates.append(
            {
                "placebo": PLACEBOS[p],
                "half_life": float(design.half_lives[i]),
                "method": METHODS[j],
                "rejections": int(count),
                "reps": design.reps,
                "rate": rate,
                "mc_se": math.sqrt(rate * (1 - rate) / design.reps),
                "exceeds_nominal": rate > limit,
            }
        )
    return {
        **record,
        "status": "ok",
        "reason": None,
        "design": {
            **_jsonable(asdict(design)),
            "dates": dates,
            "dgp": "independent over time, equicorrelated Student-t returns",
            "placebos": {
                "other_vs_own": "B1 from the next stock's history against the stock's own B1",
                "other_vs_other": (
                    "B1 from the next stock's history against B1 from the previous stock's"
                ),
            },
            "target": "mean RPS over the five daily targets, averaged across stocks per date",
            "one_sided": True,
            "limit": limit,
        },
        "rates": rates,
        "null_mean": [
            {
                "placebo": placebo,
                "half_life": float(half_life),
                "mean": float(means[p, i].mean()),
                "se": float(means[p, i].std(ddof=1) / math.sqrt(design.reps)),
            }
            for p, placebo in enumerate(PLACEBOS)
            for i, half_life in enumerate(design.half_lives)
        ],
    }


def _jsonable(values: Mapping) -> dict:
    return {
        key: list(value) if isinstance(value, tuple) else value for key, value in values.items()
    }


__all__: Sequence[str] = ()
