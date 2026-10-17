"""
Owner decision A7: the size and power of the candidate decision tests
(decision_tests) under the Stock Lab's planned conditions (research side: the
Lab's page never imports this module). docs/research/stock_lab_a7_decision.md
pre-registered this design and the selection criterion (`choose_procedure`)
before it was run. Nothing here reads real data or changes the evaluation's
rule: B1 stays the default.

Panels. N scored stocks over the planned history: a three-year warm-up, then
`years` yearly folds of `sessions_per_year` decision dates (2006-2024: 19
folds, T = 4,788). Each date's differential is the mean, over the stocks
present and the five daily targets h = 1..5, of a challenger's RPS minus B1's,
so neighbouring dates share four of their five outcomes.
    Returns    r_it = sigma_t (sqrt(rho) z_mt + sqrt(1 - rho) z_it), standardized
               Student-t shocks, with a common GARCH(1,1) variance
               sigma_t^2 = omega + a sigma_(t-1)^2 z_m,(t-1)^2 + b sigma_(t-1)^2.
               A common variance keeps the null exact: another stock's history
               says as much about it as the stock's own.
    Listings   the first max(4, ceil(0.7 N)) names have the full history; the
               other L enter the cross-sectional mean at test dates
               floor(k T / (L + 1)), k = 1..L (`listing_dates`). Their
               estimator state is warmed up on latent history.

Nulls. Every scored stock has `challengers` shadow stocks, with the same
return law and never scored. Challenger j forecasts the stock with B1 at the
same half-life read from its j-th shadow's history (`shadow_differentials`).
    own history   challenger j minus the stock's own B1: the decision
                  comparison. Each date's differential has mean exactly zero,
                  and the stock's own B1 later learns from the sessions it was
                  scored on, as in the real comparisons.
    symmetric     challenger k minus challenger j (the stepwise rule's
                  comparisons): exactly antisymmetric, so it measures the
                  tests alone.
Alternatives shift a challenger's per-date differential by -delta. Every
procedure's standard error is shift-invariant, so size and power come from the
same panels (decision_tests.Statistics.rejects).

Family. Holm across the challengers and the simplest-first stepwise rule are
applied as evaluation._score and evaluation._select apply them (`select`):
challenger 0 is the simplest. Reported, never used to choose.

Randomness uses numpy's legacy RandomState, whose stream is frozen across
numpy versions (NEP 19).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from stock_agent.stock_lab import decision_tests as dt
from stock_agent.stock_lab.size_simulation import categorize

UNIVERSE_SIZES = (4, 6, 10, 30)
HALF_LIVES = (21.0, 63.0, 126.0, 252.0)
DELTAS = (0.0005, 0.001)
NOMINAL = 0.05
REPS = 4000
SEED = 20261017  # plus the universe size, one stream per size
# The pre-registered criterion (choose_procedure).
MAX_DISTORTION = 0.01
CHOICE_DELTA = 0.0005
POWER_TIE = 0.01
SIMPLER_FIRST = (
    "fold_block_t",
    "nw_lag10",
    "nw_fixed_b_sqrt",
    "nw_fixed_b_tenth",
    "cbb_block20",
    "sb_politis_white",
)
HORIZONS = 5
K = 3  # Decrease, Stable, Increase
RETURN_BLOCK = 252  # sessions of shocks drawn at once


@dataclass(frozen=True)
class PlannedDesign:
    names: int
    reps: int = REPS
    seed: int = SEED
    half_lives: tuple[float, ...] = HALF_LIVES
    challengers: int = 4
    years: int = 19
    warmup_years: int = 3
    sessions_per_year: int = 252
    full_history_tenths: int = 7  # the first max(4, ceil(0.7 N)) names list before the panel
    min_full_history: int = 4
    correlation: float = 0.5
    daily_vol: float = 0.025
    t_df: float = 5.0
    arch: float = 0.06
    garch: float = 0.92
    tau: float = 0.005
    alpha_prior: float = 50.0
    kappa: float = 40.0
    nominal: float = NOMINAL
    deltas: tuple[float, ...] = DELTAS
    fixed_b_steps: int = dt.FIXED_B_STEPS
    fixed_b_draws: int = dt.FIXED_B_DRAWS
    fixed_b_seed: int = dt.FIXED_B_SEED
    chunk: int = 100

    @property
    def first(self) -> int:
        return self.warmup_years * self.sessions_per_year

    @property
    def dates(self) -> int:
        return self.years * self.sessions_per_year

    @property
    def sessions(self) -> int:
        return self.first + self.dates + HORIZONS


def planned_design(names: int, **changes) -> PlannedDesign:
    """The pre-registered design for one universe size (its own seed)."""

    return PlannedDesign(names=names, seed=SEED + names, **changes)


# --- the panels ---


def listing_dates(names: int, dates: int, *, tenths: int = 7, minimum: int = 4) -> np.ndarray:
    """Each name's first date in the cross-sectional mean (0 for the full-history names)."""

    full = min(names, max(minimum, -(-names * tenths // 10)))
    late = names - full
    entry = np.zeros(names, dtype=np.int64)
    entry[full:] = [k * dates // (late + 1) for k in range(1, late + 1)]
    return entry


def garch_blocks(
    rng: np.random.RandomState,
    reps: int,
    sessions: int,
    names: int,
    *,
    correlation: float,
    daily_vol: float,
    t_df: float,
    arch: float,
    garch: float,
) -> Iterator[np.ndarray]:
    """Returns (reps, <= RETURN_BLOCK, names), block by block (module docstring)."""

    scale = math.sqrt((t_df - 2.0) / t_df)
    omega = daily_vol**2 * (1.0 - arch - garch)
    variance = np.full(reps, daily_vol**2)
    for start in range(0, sessions, RETURN_BLOCK):
        count = min(RETURN_BLOCK, sessions - start)
        market = rng.standard_t(t_df, size=(reps, count)) * scale
        own = rng.standard_t(t_df, size=(reps, count, names)) * scale
        sigma = np.empty((reps, count))
        for t in range(count):
            sigma[:, t] = np.sqrt(variance)
            variance = omega + (arch * market[:, t] ** 2 + garch) * variance
        common = math.sqrt(correlation) * market[..., None]
        yield sigma[..., None] * (common + math.sqrt(1.0 - correlation) * own)


def panel_codes(rng: np.random.RandomState, reps: int, design: PlannedDesign) -> np.ndarray:
    """Category codes (reps, sessions, names * (1 + challengers)): scored stocks first."""

    total = design.names * (1 + design.challengers)
    blocks = garch_blocks(
        rng,
        reps,
        design.sessions,
        total,
        correlation=design.correlation,
        daily_vol=design.daily_vol,
        t_df=design.t_df,
        arch=design.arch,
        garch=design.garch,
    )
    return np.concatenate([categorize(block, design.tau) for block in blocks], axis=1)


def shadow_differentials(
    codes: np.ndarray,
    half_lives: Sequence[float],
    *,
    names: int,
    challengers: int,
    entry: np.ndarray,
    alpha_prior: float,
    kappa: float,
    first: int,
) -> np.ndarray:
    """
    Per decision date t (from `first` to the last date with five outcomes) and
    challenger j, the mean over the stocks present (date index >= entry) and
    the five daily targets of RPS(B1 from stock i's j-th shadow) minus
    RPS(stock i's own B1): shape (half-lives, reps, dates, challengers).

    Column i < names of `codes` is scored stock i; column names + i *
    challengers + j is its j-th shadow. B0 and B1 are the estimators' formulas
    (estimators.b0 / b1), updated session by session as in
    size_simulation.placebo_differentials, whose RPS identity this uses.
    """

    reps, sessions, total = codes.shape
    if total != names * (1 + challengers):
        raise ValueError("codes must hold every scored stock and its shadows")
    dates = sessions - HORIZONS - first
    present = np.arange(dates)[:, None] >= np.asarray(entry)[None, :]
    share = present / present.sum(axis=1, keepdims=True)
    decay = (2.0 ** (-1.0 / np.asarray(half_lives, dtype=np.float64)))[:, None, None, None]
    eye = np.eye(K)
    sums = np.zeros((len(half_lives), reps, total, K))
    weight = np.zeros((len(half_lives), 1, 1, 1))
    counts = np.zeros((reps, total, K))
    pooled = np.zeros((reps, K))
    out = np.empty((len(half_lives), reps, dates, challengers))
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
        cumulative = np.cumsum(b1, axis=-1)[..., :2]
        own = cumulative[:, :, :names, None]
        shadow = cumulative[:, :, names:].reshape(len(half_lives), reps, names, challengers, 2)
        future = codes[:, t + 1 : t + 1 + HORIZONS, :names]
        observed = np.stack([(future == 0).mean(axis=1), (future <= 1).mean(axis=1)], axis=-1)
        gap = shadow - own
        diff = 0.5 * (gap * (shadow + own - 2.0 * observed[None, :, :, None])).sum(axis=-1)
        out[:, :, t - first] = np.einsum("hrnc,n->hrc", diff, share[t - first])
    return out


# --- the simulation ---


def _pairs(challengers: int) -> list[tuple[int, int]]:
    """(k, j) with k > j: the later (more complex) challenger first."""

    return [(k, j) for k in range(challengers) for j in range(k)]


def simulate_panels(design: PlannedDesign) -> dict:
    """
    Every procedure's mean, standard error and critical values on every panel:
    "own" (half-lives, reps, challengers) and "pair" (half-lives, reps, pairs),
    plus the fixed-b critical values used.
    """

    rng = np.random.RandomState(design.seed)
    levels = dt.holm_levels(design.nominal, design.challengers)
    folds = np.arange(design.dates) // design.sessions_per_year
    entry = listing_dates(
        design.names,
        design.dates,
        tenths=design.full_history_tenths,
        minimum=design.min_full_history,
    )
    fixed_b = dt.fixed_b_table(
        design.dates,
        levels,
        steps=design.fixed_b_steps,
        draws=design.fixed_b_draws,
        seed=design.fixed_b_seed,
    )
    pairs = _pairs(design.challengers)
    width = {"own": design.challengers, "pair": len(pairs)}
    parts = {
        side: {p: {"mean": [], "se": [], "critical": []} for p in dt.PROCEDURES} for side in width
    }
    for start in range(0, design.reps, design.chunk):
        count = min(design.chunk, design.reps - start)
        diffs = shadow_differentials(
            panel_codes(rng, count, design),
            design.half_lives,
            names=design.names,
            challengers=design.challengers,
            entry=entry,
            alpha_prior=design.alpha_prior,
            kappa=design.kappa,
            first=design.first,
        )
        own = np.moveaxis(diffs, 3, 2)  # (half-lives, reps, challengers, dates)
        pair = np.stack([own[:, :, k] - own[:, :, j] for k, j in pairs], axis=2)
        series = np.concatenate([own, pair], axis=2).reshape(-1, design.dates)
        stats = dt.procedure_statistics(series, folds, levels, fixed_b=fixed_b)
        shape = (len(design.half_lives), count, width["own"] + width["pair"])
        for name, result in stats.items():
            mean = result.mean.reshape(shape)
            se = result.se.reshape(shape)
            critical = np.asarray(result.critical).reshape(*shape, len(levels))
            split = (("own", slice(0, width["own"])), ("pair", slice(width["own"], None)))
            for side, columns in split:
                parts[side][name]["mean"].append(mean[:, :, columns])
                parts[side][name]["se"].append(se[:, :, columns])
                parts[side][name]["critical"].append(critical[:, :, columns])
    panels = {
        side: {
            name: dt.Statistics(
                **{key: np.concatenate(values, axis=1) for key, values in parts[side][name].items()}
            )
            for name in dt.PROCEDURES
        }
        for side in width
    }
    return {"panels": panels, "levels": levels, "fixed_b": fixed_b, "pairs": pairs}


def _slice(stats: dt.Statistics, index: int) -> dt.Statistics:
    return dt.Statistics(stats.mean[index], stats.se[index], stats.critical[index])


def select(own: dt.Statistics, pair: dt.Statistics, shifts: np.ndarray) -> np.ndarray:
    """
    The selection on each panel: -1 keeps B1, j selects challenger j.

    `own` and `pair` hold (reps, challengers) and (reps, pairs) statistics with
    critical values at holm_levels(nominal, challengers). A challenger beats B1
    when Holm's step-down rejects it; the beating challengers are taken
    simplest first (lowest index), and a later one replaces the current choice
    only when its own differential against it is significant at the nominal
    level (unadjusted), as evaluation._select does. `shifts` (challengers,) is
    each challenger's true improvement delta.
    """

    shifts = np.asarray(shifts, dtype=np.float64)
    reps, challengers = own.mean.shape
    rejected = own.rejects(shifts[None, :])  # (reps, challengers, levels): alpha/m .. alpha
    counts = rejected.sum(axis=1)
    steps = np.cumprod(counts >= np.arange(1, challengers + 1)[None, :], axis=1).sum(axis=1)
    beating = np.zeros((reps, challengers), dtype=bool)
    chosen = steps > 0
    beating[chosen] = rejected[chosen, :, steps[chosen] - 1]
    index = {pair: n for n, pair in enumerate(_pairs(challengers))}
    pair_shift = np.array([shifts[k] - shifts[j] for k, j in _pairs(challengers)])
    moves = pair.rejects(pair_shift[None, :])[:, :, -1]  # (reps, pairs) at the nominal level
    selected = np.where(beating.any(axis=1), beating.argmax(axis=1), -1)
    for rep in np.flatnonzero(beating.sum(axis=1) > 1):
        winners = np.flatnonzero(beating[rep])
        current = winners[0]
        for challenger in winners[1:]:
            if moves[rep, index[(challenger, current)]]:
                current = challenger
        selected[rep] = current
    return selected


def _rate(hits: np.ndarray) -> dict:
    """The share of hits (reps, tests) with its Monte Carlo SE (panels are independent)."""

    per_panel = hits.reshape(len(hits), -1).mean(axis=1)
    return {
        "rate": float(per_panel.mean()),
        "mc_se": float(per_panel.std(ddof=1) / math.sqrt(len(per_panel))),
    }


def summarize(design: PlannedDesign, raw: Mapping) -> dict:
    """Size, power and the family's selection rates per half-life and procedure."""

    cells, null_mean = [], []
    challengers = design.challengers
    last = challengers - 1
    for h, half_life in enumerate(design.half_lives):
        means = raw["panels"]["own"]["nw_lag10"].mean[h]
        null_mean.append(
            {
                "half_life": half_life,
                "mean": float(means.mean()),
                "se": float(means.mean(axis=1).std(ddof=1) / math.sqrt(len(means))),
                "rms": float(np.sqrt((means**2).mean())),
            }
        )
        for name in dt.PROCEDURES:
            own = _slice(raw["panels"]["own"][name], h)
            pair = _slice(raw["panels"]["pair"][name], h)
            forward = pair.rejects()[..., -1]
            backward = dt.Statistics(-pair.mean, pair.se, pair.critical).rejects()[..., -1]
            cell = {
                "names": design.names,
                "half_life": half_life,
                "procedure": name,
                "size_own": _rate(own.rejects()[..., -1]),
                "size_symmetric": _rate(np.concatenate([forward, backward], axis=1)),
                # E[se^2] / E[mean^2]: how well the procedure's variance matches the true one.
                "variance_ratio": float((own.se**2).mean() / (own.mean**2).mean()),
                "power_own": [
                    {"delta": delta, **_rate(own.rejects(delta)[..., -1])}
                    for delta in design.deltas
                ],
                "family": [],
            }
            null_pick = select(own, pair, np.zeros(challengers))
            cell["family"].append(
                {"delta": 0.0, "true": None, "replaced": _rate(null_pick >= 0), "picked": None}
            )
            for delta in design.deltas:
                for true in (0, last):
                    shifts = np.zeros(challengers)
                    shifts[true] = delta
                    pick = select(own, pair, shifts)
                    cell["family"].append(
                        {
                            "delta": delta,
                            "true": int(true),
                            "replaced": _rate(pick >= 0),
                            "picked": _rate(pick == true),
                        }
                    )
            cells.append(cell)
    return {"cells": cells, "null_mean": null_mean}


def simulate_planned(design: PlannedDesign) -> dict:
    """The JSON-ready record of one universe size (module docstring)."""

    raw = simulate_panels(design)
    entry = listing_dates(
        design.names,
        design.dates,
        tenths=design.full_history_tenths,
        minimum=design.min_full_history,
    )
    return {
        "design": {
            **{k: list(v) if isinstance(v, tuple) else v for k, v in asdict(design).items()},
            "first": design.first,
            "dates": design.dates,
            "sessions": design.sessions,
            "listing_dates": entry.tolist(),
        },
        "levels": list(raw["levels"]),
        "fixed_b": {name: values.tolist() for name, values in raw["fixed_b"].items()},
        **summarize(design, raw),
    }


# --- the pre-registered choice ---


def choose_procedure(cells: Sequence[Mapping], *, nominal: float = NOMINAL) -> dict:
    """
    The pre-registered criterion (docs/research/stock_lab_a7_decision.md):
    distortion is the largest |size - nominal| over every cell (universe size,
    half-life, own-history and symmetric nulls); a procedure is admissible at
    most MAX_DISTORTION from nominal; among admissible ones the highest mean
    own-history power at CHOICE_DELTA wins; within POWER_TIE of the runner-up
    the smaller distortion wins, then the simpler (SIMPLER_FIRST). With none
    admissible, the smallest distortion (then the simpler).
    """

    distortion = {name: 0.0 for name in SIMPLER_FIRST}
    power: dict[str, list[float]] = {name: [] for name in SIMPLER_FIRST}
    for cell in cells:
        name = cell["procedure"]
        for null in ("size_own", "size_symmetric"):
            gap = round(abs(cell[null]["rate"] - nominal), 12)
            distortion[name] = max(distortion[name], gap)
        power[name] += [p["rate"] for p in cell["power_own"] if p["delta"] == CHOICE_DELTA]
    mean_power = {name: float(np.mean(values)) for name, values in power.items()}
    admissible = [name for name in SIMPLER_FIRST if distortion[name] <= MAX_DISTORTION]

    def calmer(names: Sequence[str]) -> str:
        return min(names, key=lambda name: (distortion[name], SIMPLER_FIRST.index(name)))

    if admissible:
        ranked = sorted(admissible, key=lambda name: (-mean_power[name], SIMPLER_FIRST.index(name)))
        top = ranked[:2]
        tied = len(top) == 2 and mean_power[top[0]] - mean_power[top[1]] <= POWER_TIE
        chosen = calmer(top) if tied else top[0]
        reason = "admissible; highest power" + (" (tied: smaller distortion)" if tied else "")
    else:
        chosen = calmer(SIMPLER_FIRST)
        reason = "none admissible; smallest distortion"
    return {
        "chosen": chosen,
        "reason": reason,
        "admissible": admissible,
        "distortion": distortion,
        "mean_power": mean_power,
        "criterion": {
            "max_distortion": MAX_DISTORTION,
            "choice_delta": CHOICE_DELTA,
            "power_tie": POWER_TIE,
            "simpler_first": list(SIMPLER_FIRST),
            "nominal": nominal,
        },
    }


__all__: Sequence[str] = ()
