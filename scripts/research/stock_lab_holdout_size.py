"""
SIMULATION ONLY: the size and power of the sealed holdout's deciding test, on
synthetic panels. It reads no data, no store and no calendar, and it contacts
no network. It never touches the holdout and does not change its registered
protocol (docs/research/stock_lab_holdout.md); it measures how far the one
answer can be trusted. The notes are in docs/research/stock_lab_holdout_size.md.

The test (the protocol's). d_j is decision j's pooled daily RPS differential,
B1 minus B0: the mean over the four stocks of the mean over the five daily
targets. The 443 decisions fall into 7 calendar-quarter blocks of BLOCK_SIZES
decisions. With m_1..m_7 the block means,
    t = mean(m) / (sd(m) / sqrt(7)),
and B1 beats B0 when t is below the 2.5% quantile of t(6), -2.447. The holdout
computes it with inference.fold_block_t and size_simulation.one_sided_p;
`block_t` is the same test on a batch of replications (tested equal).

The panels. Each replication is one synthetic history of four stocks:
    returns      planned_simulation.garch_blocks with decision A7's planned
                 settings: Student-t(5) shocks, pairwise correlation 0.5 and a
                 common GARCH(1,1) variance (daily volatility 2.5%, a = 0.06,
                 b = 0.92), so the stocks share persistent volatility regimes;
    categories   size_simulation.categorize at the stable band (0.5%);
    development  22 years (2003-2024); its last 252 sessions are the first
                 decision's window (2024);
    B0           frozen at the end of development, as registry._long_run (the
                 estimators.b0 formula at the last row): per stock,
                 (counts + alpha * pooled share) / (n + alpha), alpha = 50;
    B1           at each decision, as the frozen model under frozen_long_run
                 (registry._recency_weighted): the window's 251 categories (its
                 first session has no return inside it) weighted lambda^age,
                 lambda = 2^(-1/252), shrunk toward the frozen B0 with kappa = 40;
    outcomes     the five sessions after each decision. The first decision is
                 the last development session and the last is five sessions
                 before the end: 443 decisions over 447 outcome sessions.

Scenarios (SHIFTS). Under "none" the law of the categories never changes, so
B0 is right on average; B1 can gain only by tracking the volatility regime,
and it pays for its own sampling noise. The other scenarios change the law
from an onset, through a daily drift (share moves from Decrease to Increase)
or a volatility scale (share moves out of Stable):
    window   from the first session of the first decision's window (2024): the
             recent era differs from the long run, and B1's first window
             already sees the change;
    holdout  from the first outcome session (2025-01-02): B1 must learn it.
Every scenario uses the same shocks (common random numbers).

Reported per scenario (`summarize`):
    true_mean       E[mean(m)], the Monte Carlo mean: B1 is better on average
                    when it is negative;
    reject          how often the test says "B1 beats B0" at the nominal level:
                    power when true_mean < 0, a false-positive rate otherwise;
    boundary_size   how often it rejects once each replication's statistic is
                    recentred at true_mean (the fold-block se is unchanged when
                    a constant is added to the series): the false-positive rate
                    in a world with this dependence where B1 and B0 are exactly
                    equally good. This is the test's size;
    calibration     the recentred rejection rates at several nominal levels,
                    and the critical value that would give exactly `nominal`;
    variance_ratio  E[se^2] / Var(mean(m)): below 1, the test's se is too small;
    block_correlation  between the quarter means: the main reason for both;
    shape           the skewness of mean(m), and its correlation with se: when
                    B1 does unusually well, are its quarters also uneven?
The "none" scenario also gives the power against a pure RPS gain delta (the
statistic recentred at -delta), comparable with decision A7's.

Randomness uses numpy's legacy RandomState, whose stream is frozen across numpy
versions (NEP 19).

Usage (a few minutes; `--reps` and `--design` for a quick run):
    PYTHONPATH=src .venv/bin/python scripts/research/stock_lab_holdout_size.py \
        --out docs/research/stock_lab_holdout_size.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from stock_agent.model_diagnostics.inference import student_t_ppf
from stock_agent.stock_lab import planned_simulation as ps
from stock_agent.stock_lab.size_simulation import HORIZONS, K, categorize

SCHEMA = "stock_lab_block_t_size"  # the quarterly fold-block t's size and power
SCHEMA_VERSION = "1.0"
# Decisions per block of the registered protocol, 2025Q1 (with 2024-12-31) to 2026Q3 (with
# the October 2026 decisions), from the rules calendar the protocol uses (a test checks them).
BLOCK_SIZES = (61, 62, 64, 64, 61, 62, 69)
LEVELS = (0.005, 0.01, 0.025, 0.05, 0.1)
DELTAS = (0.0005, 0.001, 0.002)
SEED = 20261029
ONSETS = ("window", "holdout")
CATEGORIES = ("decrease", "stable", "increase")
PLANNED = ps.PlannedDesign(names=4)  # decision A7's planned settings


@dataclass(frozen=True)
class Shift:
    """A change in the categories' law from `onset` (module docstring): r -> scale * r + drift."""

    name: str
    drift: float = 0.0
    vol_scale: float = 1.0
    onset: str = "window"


SHIFTS = (
    Shift("none"),
    Shift("drift_5bp", drift=0.0005),
    Shift("drift_10bp", drift=0.001),
    Shift("drift_15bp", drift=0.0015),
    Shift("drift_20bp", drift=0.002),
    Shift("drift_30bp", drift=0.003),
    Shift("vol_x1.25", vol_scale=1.25),
    Shift("vol_x1.5", vol_scale=1.5),
    Shift("drift_20bp_vol_x1.5", drift=0.002, vol_scale=1.5),
    Shift("drift_20bp_from_holdout", drift=0.002, onset="holdout"),
    Shift("vol_x1.5_from_holdout", vol_scale=1.5, onset="holdout"),
)


@dataclass(frozen=True)
class HoldoutDesign:
    label: str = "primary"
    names: int = 4
    block_sizes: tuple[int, ...] = BLOCK_SIZES
    window: int = 252
    half_life: float = 252.0
    kappa: float = PLANNED.kappa
    alpha_prior: float = PLANNED.alpha_prior
    development_years: int = 22
    sessions_per_year: int = PLANNED.sessions_per_year
    correlation: float = PLANNED.correlation
    daily_vol: float = PLANNED.daily_vol
    t_df: float = PLANNED.t_df
    arch: float = PLANNED.arch
    garch: float = PLANNED.garch
    tau: float = PLANNED.tau
    nominal: float = 0.025
    shifts: tuple[Shift, ...] = SHIFTS
    reps: int = 20_000
    seed: int = SEED
    chunk: int = 250

    @property
    def decisions(self) -> int:
        return sum(self.block_sizes)

    @property
    def development(self) -> int:
        return self.development_years * self.sessions_per_year

    @property
    def fetch_start(self) -> int:
        """The first session of the first decision's window."""

        return self.development - self.window

    @property
    def first(self) -> int:
        """The first decision: the last development session."""

        return self.development - 1

    @property
    def sessions(self) -> int:
        return self.first + self.decisions + HORIZONS


# The primary design, and two sensitivities of its no-shift scenario: no volatility regimes,
# and regimes that last longer (a + b = 0.995 rather than 0.98).
DESIGNS = (
    HoldoutDesign(),
    HoldoutDesign(label="constant_volatility", arch=0.0, garch=0.0, shifts=SHIFTS[:1]),
    HoldoutDesign(label="persistent_volatility", arch=0.06, garch=0.935, shifts=SHIFTS[:1]),
)


# --- one replication batch ---


def shifted(returns: np.ndarray, shift: Shift, design: HoldoutDesign) -> np.ndarray:
    """The returns from the fetch start on, with `shift` applied from its onset."""

    if shift.onset not in ONSETS:
        raise ValueError(f"unknown onset {shift.onset!r}")
    out = returns[:, design.fetch_start :].copy()
    onset = 0 if shift.onset == "window" else design.first + 1 - design.fetch_start
    out[:, onset:] = shift.vol_scale * out[:, onset:] + shift.drift
    return out


def frozen_b0(counts: np.ndarray, alpha_prior: float) -> np.ndarray:
    """
    B0 at the end of development from each stock's category counts (reps, names, K):
    (counts + alpha * pooled share) / (n + alpha), the pooled share over every stock.
    """

    pooled = counts.sum(axis=1, keepdims=True)
    share = pooled / pooled.sum(axis=2, keepdims=True)
    return (counts + alpha_prior * share) / (counts.sum(axis=2, keepdims=True) + alpha_prior)


def window_b1(
    onehot: np.ndarray,
    prior: np.ndarray,
    *,
    half_life: float,
    kappa: float,
    window: int,
    first: int,
    count: int,
) -> np.ndarray:
    """
    B1 at the `count` decisions from session `first` of `onehot` (reps, sessions, names, K),
    shape (reps, count, names, K). A decision's window is the `window` sessions ending at
    it; the first has no return inside the window, so the last window - 1 categories count,
    weighted lambda^age: S = sum lambda^age e, W = sum lambda^age, and
    B1 = (S + kappa * prior) / (W + kappa), prior (reps, names, K) the frozen B0.
    S is updated decision by decision: S_s = lambda S_(s-1) + e_s - lambda^span e_(s-span).
    """

    lam = 2.0 ** (-1.0 / half_life)
    span = window - 1
    weights = lam ** np.arange(span - 1, -1, -1, dtype=np.float64)  # oldest first
    sums = np.einsum("j,rjnk->rnk", weights, onehot[:, first - span + 1 : first + 1])
    out = np.empty((onehot.shape[0], count, *onehot.shape[2:]))
    out[:, 0] = sums
    fade = lam**span
    for j in range(1, count):
        s = first + j
        sums = lam * sums + onehot[:, s] - fade * onehot[:, s - span]
        out[:, j] = sums
    return (out + kappa * prior[:, None]) / (weights.sum() + kappa)


def differentials(
    codes: np.ndarray, b1: np.ndarray, b0: np.ndarray, *, first: int, count: int
) -> np.ndarray:
    """
    d_j (reps, count): at decision first + j, the mean over stocks of the mean over the five
    daily targets of RPS(B1) - RPS(B0). B1 and B0 give every target the same forecast, so
    (size_simulation.placebo_differentials) with Q, P the cumulative probabilities of
    {Decrease} and {Decrease, Stable} and O_k the targets' mean outcome indicators,
        mean_h RPS(Q, y_h) - mean_h RPS(P, y_h) = 1/2 sum_k (Q_k - P_k)(Q_k + P_k - 2 O_k).
    """

    below = np.stack([codes == 0, codes <= 1], axis=-1).astype(np.float64)
    running = np.concatenate([np.zeros_like(below[:, :1]), np.cumsum(below, axis=1)], axis=1)
    start = first + 1
    observed = (
        running[:, start + HORIZONS : start + HORIZONS + count] - running[:, start : start + count]
    ) / HORIZONS
    q = np.cumsum(b1, axis=-1)[..., :2]
    p = np.cumsum(b0, axis=-1)[:, None, :, :2]
    return (0.5 * ((q - p) * (q + p - 2.0 * observed)).sum(axis=-1)).mean(axis=-1)


def block_means(series: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """(reps, blocks): each block's mean of consecutive decisions, `sizes` decisions each."""

    edges = np.cumsum((0, *sizes))
    if edges[-1] != series.shape[1]:
        raise ValueError("the blocks must cover every decision")
    return np.stack([series[:, a:b].mean(axis=1) for a, b in itertools.pairwise(edges)], axis=1)


def block_t(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fold-block t on a batch: (mean of the block means, sd / sqrt(blocks)) per row."""

    return means.mean(axis=1), means.std(axis=1, ddof=1) / math.sqrt(means.shape[1])


def simulate(design: HoldoutDesign) -> dict[str, dict[str, np.ndarray]]:
    """
    Per scenario: the block means (reps, blocks) and the holdout's category mix minus the
    frozen B0, averaged over the stocks (reps, K).
    """

    if design.fetch_start < 0 or design.window < 2:
        raise ValueError("development must hold the first decision's window")
    rng = np.random.RandomState(design.seed)
    first = design.first - design.fetch_start  # the first decision, within the fetched sessions
    eye = np.eye(K)
    parts: dict[str, dict[str, list]] = {s.name: {"means": [], "mix": []} for s in design.shifts}
    for start in range(0, design.reps, design.chunk):
        count = min(design.chunk, design.reps - start)
        blocks = ps.garch_blocks(
            rng,
            count,
            design.sessions,
            design.names,
            correlation=design.correlation,
            daily_vol=design.daily_vol,
            t_df=design.t_df,
            arch=design.arch,
            garch=design.garch,
        )
        returns = np.concatenate(list(blocks), axis=1)
        before = categorize(returns[:, : design.fetch_start], design.tau)
        base = np.stack([(before == k).sum(axis=1) for k in range(K)], axis=-1).astype(np.float64)
        for shift in design.shifts:
            codes = categorize(shifted(returns, shift, design), design.tau)
            onehot = eye[codes]
            b0 = frozen_b0(base + onehot[:, : first + 1].sum(axis=1), design.alpha_prior)
            b1 = window_b1(
                onehot,
                b0,
                half_life=design.half_life,
                kappa=design.kappa,
                window=design.window,
                first=first,
                count=design.decisions,
            )
            d = differentials(codes, b1, b0, first=first, count=design.decisions)
            parts[shift.name]["means"].append(block_means(d, design.block_sizes))
            mix = onehot[:, first + 1 :].mean(axis=1) - b0
            parts[shift.name]["mix"].append(mix.mean(axis=1))
    return {
        name: {key: np.concatenate(values) for key, values in found.items()}
        for name, found in parts.items()
    }


# --- the summaries ---


def _rate(hits: np.ndarray) -> dict:
    rate = float(np.mean(hits))
    return {"rate": rate, "mc_se": math.sqrt(rate * (1.0 - rate) / len(hits))}


def summarize(
    means: np.ndarray,
    mix: np.ndarray,
    *,
    nominal: float,
    levels: Sequence[float] = LEVELS,
    deltas: Sequence[float] = (),
) -> dict:
    """One scenario's record (module docstring) from its block means and mix shifts."""

    reps, blocks = means.shape
    df = blocks - 1
    critical = {level: student_t_ppf(level, df) for level in (*levels, nominal)}
    estimate, se = block_t(means)
    true_mean = float(estimate.mean())
    centred = (estimate - true_mean) / se
    correlation = np.corrcoef(means, rowvar=False)
    pairs = correlation[~np.eye(blocks, dtype=bool)]
    return {
        "true_mean": true_mean,
        "true_mean_mc_se": float(estimate.std(ddof=1) / math.sqrt(reps)),
        "sd_of_estimate": float(estimate.std(ddof=1)),
        "rms_se": float(np.sqrt(np.mean(se**2))),
        "b1_better_share": float(np.mean(estimate < 0)),
        "mix_shift_pp": {c: float(100.0 * mix[:, k].mean()) for k, c in enumerate(CATEGORIES)},
        "reject": {
            "means": "power" if true_mean < 0 else "false_positive_rate",
            **_rate(estimate / se < critical[nominal]),
        },
        "boundary_size": _rate(centred < critical[nominal]),
        "calibration": [
            {"level": level, "critical": critical[level], **_rate(centred < critical[level])}
            for level in levels
        ],
        "exact_critical": float(np.quantile(centred, nominal)),
        "variance_ratio": float(np.mean(se**2) / estimate.var(ddof=1)),
        "shape": {
            "estimate_skewness": float(np.mean((estimate - true_mean) ** 3) / estimate.std() ** 3),
            "estimate_se_correlation": float(np.corrcoef(estimate, se)[0, 1]),
        },
        "block_correlation": {
            "adjacent_mean": float(np.diag(correlation, 1).mean()),
            "all_pairs_mean": float(pairs.mean()),
        },
        "rps_gain_power": [
            {"delta": delta, **_rate((estimate - true_mean - delta) / se < critical[nominal])}
            for delta in deltas
        ],
    }


def run_design(design: HoldoutDesign) -> dict:
    """One design's JSON-ready record."""

    found = simulate(design)
    scenarios = []
    for shift in design.shifts:
        deltas = DELTAS if shift.name == "none" else ()
        summary = summarize(
            found[shift.name]["means"],
            found[shift.name]["mix"],
            nominal=design.nominal,
            deltas=deltas,
        )
        scenarios.append({"shift": asdict(shift), **summary})
    return {"design": design_record(design), "scenarios": scenarios}


def design_record(design: HoldoutDesign) -> dict:
    return {
        **_jsonable(asdict(design)),
        "decisions": design.decisions,
        "development_sessions": design.development,
        "sessions": design.sessions,
    }


def record(designs: Sequence[HoldoutDesign]) -> dict:
    return _rounded(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "simulation_only": True,
            "test": {
                "name": "fold_block_t_one_sided",
                "blocks": len(BLOCK_SIZES),
                "df": len(BLOCK_SIZES) - 1,
                "critical": student_t_ppf(designs[0].nominal, len(BLOCK_SIZES) - 1),
                "series": "pooled daily RPS differential, B1 minus B0, per decision",
            },
            "designs": [run_design(design) for design in designs],
        }
    )


def _jsonable(values: Mapping) -> dict:
    return {key: list(v) if isinstance(v, tuple) else v for key, v in values.items()}


def _rounded(value):
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, Mapping):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(item) for item in value]
    return value


# --- the tables ---


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown(found: Mapping) -> str:
    """The note's tables, from a record."""

    header = (
        "| Design | Scenario | Mix shift, pp (D / S / I) | True mean (x1e-4) | "
        "B1 better (realized) | Says 'beats B0' | Boundary size | Variance ratio | "
        "Adjacent-quarter corr. | Corr(mean, se) |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for design in found["designs"]:
        for row in design["scenarios"]:
            mix = row["mix_shift_pp"]
            lines.append(
                f"| {design['design']['label']} | {row['shift']['name']} | "
                f"{mix['decrease']:+.1f} / {mix['stable']:+.1f} / {mix['increase']:+.1f} | "
                f"{1e4 * row['true_mean']:+.2f} | {_pct(row['b1_better_share'])} | "
                f"{_pct(row['reject']['rate'])} ({row['reject']['means']}) | "
                f"{_pct(row['boundary_size']['rate'])} | {row['variance_ratio']:.2f} | "
                f"{row['block_correlation']['adjacent_mean']:+.2f} | "
                f"{row['shape']['estimate_se_correlation']:+.2f} |"
            )
    lines += ["", "| Design | Nominal | t(6) critical | Boundary rate | Exact critical |"]
    lines.append("|---|---|---|---|---|")
    for design in found["designs"]:
        none = design["scenarios"][0]
        for row in none["calibration"]:
            lines.append(
                f"| {design['design']['label']} | {_pct(row['level'])} | "
                f"{row['critical']:.3f} | {_pct(row['rate'])} | "
                + (f"{none['exact_critical']:.3f}" if row["level"] == 0.025 else "")
                + " |"
            )
    lines += ["", "| Design | RPS gain delta | Power |", "|---|---|---|"]
    for design in found["designs"]:
        for row in design["scenarios"][0]["rps_gain_power"]:
            lines.append(f"| {design['design']['label']} | {row['delta']} | {_pct(row['rate'])} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="where to write the JSON record")
    parser.add_argument("--reps", type=int, help="replications per design (default: each design's)")
    parser.add_argument(
        "--design", action="append", choices=[d.label for d in DESIGNS], help="only these"
    )
    args = parser.parse_args(argv)
    designs = [d for d in DESIGNS if not args.design or d.label in args.design]
    if args.reps:
        designs = [replace(d, reps=args.reps) for d in designs]
    found = record(designs)
    text = json.dumps(found, indent=2, sort_keys=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(markdown(found), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
