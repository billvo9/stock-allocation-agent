"""
How reliable the forecasts have been (numpy and the Stock Lab only): the
question "when the model said 60-70% for Increase, how often did Increase
happen, and was it more accurate than the recent-weighted baseline (B1)?".

Two kinds of evidence, never mixed:

    development  the stored walk-forward evaluation (pre-2025 history of a
                 watchlist chosen in hindsight, exploratory; or synthetic),
                 read exactly as stored: nothing is recomputed
    prospective  the recorded forecasts and their scores in the ledger, as
                 sessions closed after each forecast was made

A group of observations is one target definition: one horizon (sessions 1
to 5, or the five-session total), one stable band, one estimator and one
model version. Groups are never pooled (`MixedTargets`).

Rows are not independent trials: dates are the clusters
    Forecasts made on one date for several stocks are settled by the same
    sessions, and stocks move together. A bucket's observed frequency
    p = (hits) / (rows) therefore gets a cluster-robust (linearization)
    variance over forecast dates. With u_g = y_g - p * m_g (the hits y_g and
    rows m_g of date g, dates in time order, zero for dates the bucket does
    not hold), C dates and n rows:

        Var(p) = C / (C - 1) * max(G_0, G_0 + 2 * sum_{l=1..L} G_l) / n^2
        G_l    = sum_g u_g * u_(g+l)

    L is 0 for a single session's target (within one horizon, each date is
    settled by a different session). Five-session totals of dates up to
    four sessions apart share sessions, so L = 4 for them: the overlap's
    covariance is counted (a truncated-kernel HAC variance, floored at the
    lag-0 term).

    That variance defines an effective sample size n_eff = p (1 - p) / Var,
    capped at n, and the interval is the Jeffreys interval at n_eff (the
    effective-sample-size approach for clustered survey proportions). The
    effective count lies between the dates and the rows. When p is 0 or 1,
    or the variance is zero, it says nothing, so n_eff falls back to the
    independent units (the dates, or the dates / 5 for the total): the
    cautious reading.

    A bucket needs at least MIN_CLUSTERS independent units and an effective
    count of at least MIN_CLUSTERS; otherwise it is insufficient evidence
    (the counts are shown, the frequency is not).

    What the interval does not cover: it reflects date-to-date noise within
    the period observed. If the true frequency moves with long calm or
    volatile stretches the forecast ignores, the interval is too narrow;
    a short prospective record cannot tell.

Matched comparisons
    A model is compared with B1 only on exactly matched observations: the
    same stock, as-of session, horizon, band and outcome. The difference of
    each score (RPS, Brier, log loss; lower is better) is the mean over dates
    of the per-date mean difference. Per-date differences are serially
    dependent (the A7 packet measured it), so the standard error comes from
    blocks of COMPARISON_BLOCK consecutive dates, counted from the first
    matched date, with a shorter remainder merged into the last whole block
    (so a block holds 20 to 39 dates). With T dates in K blocks, block means
    m_k of n_k dates and the mean difference d:

        SE^2 = K / (K - 1) * sum_k n_k^2 (m_k - d)^2 / T^2

    the cluster-robust form of a single session's calibration variance, with
    blocks of dates as the clusters; with whole blocks it is the spread of
    the block means over sqrt(K). The interval is t on (K - 1) degrees of
    freedom; fewer than MIN_COMPARISON_BLOCKS blocks give no difference.

    Simulated coverage at a nominal 90% (stock_lab.comparison_coverage) is
    near nominal when the dependence dies out within a few dates, and too
    narrow under strong persistence. The interval is a reporting convention
    (owner, 2026-10-20), never a decision test: whether a challenger beats
    B1 is decided on development data only (A7).

The persistence check (reported only)
    Beside each score's SE, the same SE from blocks of PERSISTENCE_BLOCK
    dates (a remainder again merged into the last whole block), once at
    least MIN_PERSISTENCE_BLOCKS of them exist (120 dates), and the ratio
    SE40 / SE20. Every block of 40 joins whole blocks of 20, so the ratio
    is near 1 for independent dates and bounded above (by sqrt(5/2) at 120
    dates, when each block of 40 joins two of 20).
    It grows when per-date differences stay alike for longer than a block
    of 20, or when their mean shifts during the record. When the RPS ratio
    exceeds PERSISTENCE_THRESHOLD (the owner's choice, 2026-10-20), the
    comparison is marked persistence_suspected. The Brier and log loss
    ratios are reported, never flagged: a flag on any of three scores would
    fire far more often by chance. The flag never changes a number, an
    interval or a decision. Its simulated false-alarm and detection rates
    are in stock_lab.comparison_coverage.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from stock_agent.stock_lab.contracts import CATEGORY_CODES, Category
from stock_agent.stock_lab.reversal import INSUFFICIENT, LOW, LOW_SAMPLE, MIN_EFFECTIVE, OK
from stock_agent.stock_lab.scores import beta_ppf, brier, jeffreys, log_loss, rps

LEVEL = 0.9
BINS = 10
MIN_CLUSTERS = MIN_EFFECTIVE  # independent units (and effective count) before a frequency shows
MIN_SCORED = 250  # Model Health's threshold for a rolling score (health.stock_lab.MIN_SCORED)
MIN_SESSIONS = 50  # ... over at least this many settled sessions (health.stock_lab.MIN_SESSIONS)
CUMULATIVE_LAGS = 4  # five-session totals of dates up to four sessions apart share sessions
COMPARISON_BLOCK = 20  # dates per block for a comparison's standard error
MIN_COMPARISON_BLOCKS = 5  # so at least 100 matched dates
PERSISTENCE_BLOCK = 40  # dates per block for the persistence check's standard error
MIN_PERSISTENCE_BLOCKS = 3  # so at least 120 matched dates
PERSISTENCE_THRESHOLD = 1.3  # SE40 / SE20 of RPS above this: persistence suspected (2026-10-20)
PERSISTENCE_SCORE = "rps"  # the score whose ratio raises the flag (A7's score)
REFERENCE = "B1"
NONE = "none"
OUTCOME_MISMATCH = "outcome_mismatch"
SCORE_NAMES = ("rps", "brier", "log_loss")

__all__ = ["INSUFFICIENT", "LOW", "OK"]


class MixedTargets(ValueError):
    """Observations of different target definitions, which are never pooled."""


def cluster_ids(dates: Sequence[str] | np.ndarray, block: int) -> np.ndarray:
    """Each observation's block: its date's position among the distinct dates, // block."""

    if block < 1:
        raise ValueError("a block holds at least one date")
    _, inverse = np.unique(np.asarray(dates), return_inverse=True)  # ISO dates sort in time
    return inverse // block


def lags_for(target: str) -> int:
    return CUMULATIVE_LAGS if target == "cum" else 0


def units_of(dates: int, lags: int) -> int:
    """Independent units among `dates` consecutive dates whose targets overlap `lags` deep."""

    return math.ceil(dates / (lags + 1))


def t_quantile(level: float, df: int) -> float:
    """The two-sided t quantile: P(|T| <= q) = level, T ~ t(df) (via the beta inverse)."""

    x = beta_ppf(1.0 - level, df / 2.0, 0.5)
    return math.sqrt(df * (1.0 - x) / x)


@dataclass(frozen=True)
class ClusteredProportion:
    estimate: float
    low: float
    high: float
    n: int
    dates: int  # distinct dates holding observations
    units: int  # independent units: the dates, or the dates / 5 for the total
    n_eff: float


def clustered_proportion(
    hits: Sequence[float] | np.ndarray,
    positions: Sequence[int] | np.ndarray,
    level: float = LEVEL,
    *,
    lags: int = 0,
) -> ClusteredProportion:
    """
    A proportion of rows with a cluster-robust Jeffreys interval (module
    docstring). `positions` is each row's date as its position in the time
    order of all the group's dates (any integer labels when `lags` is 0).
    """

    hits = np.asarray(hits, dtype="float64")
    n = len(hits)
    if n == 0:
        return ClusteredProportion(math.nan, math.nan, math.nan, 0, 0, 0, 0.0)
    positions = np.asarray(positions)
    if lags:
        index = positions.astype(int) - int(positions.min())
    else:
        _, index = np.unique(positions, return_inverse=True)
    size = int(index.max()) + 1
    y = np.bincount(index, weights=hits, minlength=size)
    m = np.bincount(index, minlength=size).astype("float64")
    dates = int((m > 0).sum())
    units = units_of(dates, lags)
    p = float(y.sum() / n)
    n_eff = float(units)
    if 0.0 < p < 1.0 and dates >= 2:
        u = y - p * m
        lag0 = float((u * u).sum())
        total = lag0 + 2.0 * sum(float((u[:-lag] * u[lag:]).sum()) for lag in range(1, lags + 1))
        variance = dates / (dates - 1) * max(lag0, total) / n**2
        if variance > 0:
            n_eff = min(max(p * (1.0 - p) / variance, 1.0), float(n))
    low, high = jeffreys(p * n_eff, n_eff, level)
    return ClusteredProportion(p, low, high, n, dates, units, n_eff)


@dataclass(frozen=True)
class Bucket:
    low: float
    high: float
    n: int  # observations
    dates: int  # distinct forecast dates
    clusters: int  # independent units: the dates, or the dates / 5 for the five-session total
    mean_predicted: float
    observed: float | None  # None when the evidence is insufficient
    ci_low: float | None
    ci_high: float | None
    n_eff: float | None
    status: str  # ok, low_sample or insufficient


def _status(units: float, minimum: int) -> str:
    return INSUFFICIENT if units < minimum else LOW if units < LOW_SAMPLE else OK


def calibration_buckets(
    probability: Sequence[float] | np.ndarray,
    hit: Sequence[float] | np.ndarray,
    dates: Sequence[str] | np.ndarray,
    *,
    lags: int = 0,
    bins: int = BINS,
    level: float = LEVEL,
    min_clusters: int = MIN_CLUSTERS,
) -> tuple[Bucket, ...]:
    """
    Equal-width probability buckets (the evaluation's bins, scores.reliability)
    for one category of one target definition; empty buckets are omitted.
    """

    probability = np.asarray(probability, dtype="float64")
    hit = np.asarray(hit, dtype="float64")
    dates = np.asarray(dates)
    positions = cluster_ids(dates, 1)  # each date's place in the time order of all the dates
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        chosen = index == b
        if not chosen.any():
            continue
        found = clustered_proportion(hit[chosen], positions[chosen], level, lags=lags)
        status = _status(min(found.units, found.n_eff), min_clusters)
        shown = status != INSUFFICIENT
        out.append(
            Bucket(
                low=float(edges[b]),
                high=float(edges[b + 1]),
                n=found.n,
                dates=found.dates,
                clusters=found.units,
                mean_predicted=float(probability[chosen].mean()),
                observed=found.estimate if shown else None,
                ci_low=found.low if shown else None,
                ci_high=found.high if shown else None,
                n_eff=found.n_eff if shown else None,
                status=status,
            )
        )
    return tuple(out)


# --- prospective observations from stored records ---


@dataclass(frozen=True)
class Observation:
    """One scored target of one stored forecast, with the probabilities it was scored on."""

    symbol: str
    as_of_session: str
    target: str  # "1".."5" or "cum"
    band: float  # the band of this target (the cumulative band for "cum")
    estimator: str
    model_version: str
    p_decrease: float
    p_stable: float
    p_increase: float
    outcome: int  # Decrease 0, Stable 1, Increase 2
    rps: float
    brier: float
    log_loss: float
    session: str  # the session that settled it


def observations(
    forecasts: Sequence[Mapping],
    scores: Sequence[Mapping],
    *,
    superseded: Collection[str] = frozenset(),
) -> list[Observation]:
    """
    Every scored target of the live (not superseded, not refused) `forecasts`,
    joined to the probabilities stored with it. The caller passes only the
    records that existed when the briefing was prepared.
    """

    live = {
        record["forecast_id"]: record
        for record in forecasts
        if record["forecast_id"] not in superseded and record["refusal"] is None
    }
    out = []
    for score in scores:
        record = live.get(score["forecast_id"])
        if record is None or score["status"] != "scored":
            continue
        target = str(score["target"])
        if target == "cum":
            row, band = record["cumulative"], record["cumulative_band"]
        else:
            row, band = record["daily"][int(target) - 1], record["band"]
        if row is None:
            continue
        out.append(
            Observation(
                symbol=record["symbol"],
                as_of_session=record["as_of_session"],
                target=target,
                band=float(band),
                estimator=record["estimator"],
                model_version=record["model_version"],
                p_decrease=float(row["p_decrease"]),
                p_stable=float(row["p_stable"]),
                p_increase=float(row["p_increase"]),
                outcome=int(score["outcome"]),
                rps=float(score["rps"]),
                brier=float(score["brier"]),
                log_loss=float(score["log_loss"]),
                session=score["session"],
            )
        )
    return sorted(out, key=lambda o: (o.target, o.as_of_session, o.symbol))


def comparator_observations(
    forecasts: Sequence[Mapping],
    scores: Sequence[Mapping],
    *,
    superseded: Collection[str] = frozenset(),
) -> list[Observation]:
    """
    B1's side of each scored target: the B1 comparator recorded inside the same forecast
    (schema 1.1), on the outcome stored for that target, with B1's Brier score, log loss and
    RPS computed from those stored probabilities by the scorer's functions. The model version
    is the forecast's, so B1 is matched only with the model it was recorded beside. Forecasts
    of B1 itself, records without a comparator (schema 1.0) and refusals add nothing.
    """

    live = {
        record["forecast_id"]: record
        for record in forecasts
        if record["forecast_id"] not in superseded
        and record["refusal"] is None
        and record.get("comparator") is not None
        and record["estimator"] != REFERENCE
    }
    out = []
    for score in scores:
        record = live.get(score["forecast_id"])
        if record is None or score["status"] != "scored":
            continue
        target = str(score["target"])
        comparator = record["comparator"]
        if target == "cum":
            row, band = comparator["cumulative"], record["cumulative_band"]
        else:
            row, band = comparator["daily"][int(target) - 1], record["band"]
        if row is None:
            continue
        p = np.array([[row["p_decrease"], row["p_stable"], row["p_increase"]]], dtype="float64")
        y = np.array([int(score["outcome"])])
        out.append(
            Observation(
                symbol=record["symbol"],
                as_of_session=record["as_of_session"],
                target=target,
                band=float(band),
                estimator=REFERENCE,
                model_version=record["model_version"],
                p_decrease=float(row["p_decrease"]),
                p_stable=float(row["p_stable"]),
                p_increase=float(row["p_increase"]),
                outcome=int(score["outcome"]),
                rps=float(rps(p, y)[0]),
                brier=float(brier(p, y)[0]),
                log_loss=float(log_loss(p, y)[0]),
                session=score["session"],
            )
        )
    return sorted(out, key=lambda o: (o.target, o.as_of_session, o.symbol))


def definition(o: Observation) -> tuple[str, float, str, str]:
    """The target definition an observation belongs to."""

    return (o.target, o.band, o.estimator, o.model_version)


def _one(found: Sequence[Observation]) -> tuple[str, float, str, str]:
    keys = {definition(o) for o in found}
    if len(keys) != 1:
        raise MixedTargets(
            "observations of different horizons, stable bands, estimators or model versions are "
            "never pooled"
        )
    return keys.pop()


@dataclass(frozen=True)
class CalibrationGroup:
    target: str
    band: float
    estimator: str
    model_version: str
    category: str
    n: int
    dates: int
    buckets: tuple[Bucket, ...]


def calibration_group(found: Sequence[Observation], category: str) -> CalibrationGroup:
    """The buckets of one category for one target definition (refuses a mixed group)."""

    if not found:
        raise ValueError("a calibration group needs observations")
    target, band, estimator, version = _one(found)
    code = CATEGORY_CODES[Category(category)]
    buckets = calibration_buckets(
        [getattr(o, f"p_{category}") for o in found],
        [float(o.outcome == code) for o in found],
        [o.as_of_session for o in found],
        lags=lags_for(target),
    )
    return CalibrationGroup(
        target=target,
        band=band,
        estimator=estimator,
        model_version=version,
        category=category,
        n=len(found),
        dates=len({o.as_of_session for o in found}),
        buckets=buckets,
    )


@dataclass(frozen=True)
class Rolling:
    """The latest `window` forecast dates of one target definition."""

    n: int
    dates: int
    first: str | None
    last: str | None
    means: Mapping[str, float] | None  # None when insufficient
    status: str


def rolling(found: Sequence[Observation], window: int = 63) -> Rolling:
    """
    Mean scores over the latest `window` forecast dates, shown only from
    MIN_SCORED observations over MIN_SESSIONS dates (Model Health's rule).
    """

    if not found:
        return Rolling(0, 0, None, None, None, NONE)
    _one(found)
    kept_dates = sorted({o.as_of_session for o in found})[-window:]
    kept = [o for o in found if o.as_of_session >= kept_dates[0]]
    enough = len(kept) >= MIN_SCORED and len(kept_dates) >= MIN_SESSIONS
    means = {name: float(np.mean([getattr(o, name) for o in kept])) for name in SCORE_NAMES}
    return Rolling(
        len(kept),
        len(kept_dates),
        kept_dates[0],
        kept_dates[-1],
        means if enough else None,
        OK if enough else INSUFFICIENT,
    )


# --- matched comparisons ---


@dataclass(frozen=True)
class ScoreDifference:
    score: str
    candidate: float | None
    baseline: float | None
    difference: float | None  # candidate minus baseline; negative favours the candidate
    se: float | None
    low: float | None
    high: float | None
    se_long: float | None  # the SE from blocks of 40 dates; None below 3 such blocks
    se_ratio: float | None  # se_long / se; None without se_long or when se is zero


@dataclass(frozen=True)
class MatchedComparison:
    target: str
    band: float
    candidate_estimator: str
    baseline_estimator: str
    n: int
    dates: int
    blocks: int
    long_blocks: int  # blocks of 40 dates, for the persistence check
    unmatched_candidate: int
    unmatched_baseline: int
    differences: tuple[ScoreDifference, ...]
    status: str  # ok, insufficient, none or outcome_mismatch
    persistence_suspected: bool | None  # the RPS ratio above the threshold; None: not checked


def comparison_blocks(dates: int, block: int = COMPARISON_BLOCK) -> np.ndarray:
    """
    Each of `dates` consecutive dates' block: runs of `block` dates from the
    first, a shorter remainder merged into the last whole block. Earlier
    blocks never move as dates are added.
    """

    if block < 1:
        raise ValueError("a block holds at least one date")
    if dates < block:
        raise ValueError("fewer dates than one whole block")
    return np.minimum(np.arange(dates) // block, dates // block - 1)


@dataclass(frozen=True)
class BlockInterval:
    estimate: np.ndarray  # one per series: the mean over the last axis
    se: np.ndarray
    low: np.ndarray
    high: np.ndarray
    blocks: int


def block_se(
    values: Sequence[float] | np.ndarray, block: int = COMPARISON_BLOCK
) -> tuple[np.ndarray, int]:
    """
    The standard error of the mean of per-date values in time order (the
    last axis) from blocks of `block` dates, and the number of blocks
    (module docstring).
    """

    values = np.asarray(values, dtype="float64")
    dates = values.shape[-1]
    sizes = np.bincount(comparison_blocks(dates, block))
    blocks = len(sizes)
    if blocks < 2:
        raise ValueError("a block standard error needs at least two blocks")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    deviation = np.add.reduceat(values, starts, axis=-1) - sizes * values.mean(axis=-1)[..., None]
    return np.sqrt(blocks / (blocks - 1) * (deviation**2).sum(axis=-1)) / dates, blocks


def block_t_interval(
    values: Sequence[float] | np.ndarray, level: float = LEVEL, block: int = COMPARISON_BLOCK
) -> BlockInterval:
    """
    The mean of per-date values in time order (the last axis: one series, or
    one per row) with its block t interval (module docstring).
    """

    values = np.asarray(values, dtype="float64")
    estimate = values.mean(axis=-1)
    se, blocks = block_se(values, block)
    q = t_quantile(level, blocks - 1)
    return BlockInterval(estimate, se, estimate - q * se, estimate + q * se, blocks)


def persistence_ratio(
    values: Sequence[float] | np.ndarray,
    block: int = COMPARISON_BLOCK,
    long_block: int = PERSISTENCE_BLOCK,
) -> np.ndarray | None:
    """
    SE from blocks of `long_block` dates over SE from blocks of `block` dates
    for per-date values in time order (the last axis), NaN where the second
    is zero; None with fewer than MIN_PERSISTENCE_BLOCKS long blocks (module
    docstring).
    """

    values = np.asarray(values, dtype="float64")
    if values.shape[-1] // long_block < MIN_PERSISTENCE_BLOCKS:
        return None
    short, _ = block_se(values, block)
    long, _ = block_se(values, long_block)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(short > 0, long / short, np.nan)


def _by_key(found: Sequence[Observation], side: str) -> dict[tuple[str, str], Observation]:
    out: dict[tuple[str, str], Observation] = {}
    for o in found:
        key = (o.symbol, o.as_of_session)
        if key in out:
            raise ValueError(f"two {side} observations for {key}")
        out[key] = o
    return out


def matched_comparison(
    candidate: Sequence[Observation],
    baseline: Sequence[Observation],
    *,
    level: float = LEVEL,
    block: int = COMPARISON_BLOCK,
    min_blocks: int = MIN_COMPARISON_BLOCKS,
    long_block: int = PERSISTENCE_BLOCK,
    threshold: float = PERSISTENCE_THRESHOLD,
) -> MatchedComparison:
    """
    Candidate minus B1 on exactly matched observations, with each score's
    persistence ratio and the RPS flag (module docstring). The flag never
    changes a number.
    """

    target, band, candidate_estimator, _ = _one(candidate)
    base_target, base_band, baseline_estimator, _ = _one(baseline)
    if (base_target, base_band) != (target, band):
        raise MixedTargets("a comparison needs the same horizon and stable band on both sides")
    if baseline_estimator != REFERENCE or candidate_estimator == REFERENCE:
        raise MixedTargets("a comparison sets a challenger against B1")
    candidates, baselines = _by_key(candidate, "candidate"), _by_key(baseline, "baseline")
    pairs = [(o, baselines[key]) for key, o in candidates.items() if key in baselines]
    distinct = sorted({o.as_of_session for o, _ in pairs})
    blocks = len(distinct) // block  # whole blocks; a remainder joins the last
    common = {
        "target": target,
        "band": band,
        "candidate_estimator": candidate_estimator,
        "baseline_estimator": baseline_estimator,
        "n": len(pairs),
        "dates": len(distinct),
        "blocks": blocks,
        "long_blocks": len(distinct) // long_block,
        "unmatched_candidate": len(candidates) - len(pairs),
        "unmatched_baseline": len(baselines) - len(pairs),
    }
    empty = tuple(
        ScoreDifference(name, *(None,) * (len(fields(ScoreDifference)) - 1)) for name in SCORE_NAMES
    )
    if any(o.outcome != m.outcome for o, m in pairs):
        return MatchedComparison(
            **common, differences=empty, status=OUTCOME_MISMATCH, persistence_suspected=None
        )
    if not pairs or blocks < max(min_blocks, 2):  # a spread needs two blocks
        return MatchedComparison(
            **common,
            differences=empty,
            status=INSUFFICIENT if pairs else NONE,
            persistence_suspected=None,
        )
    position = {day: i for i, day in enumerate(distinct)}
    index = np.array([position[o.as_of_session] for o, _ in pairs])
    per_date = np.bincount(index).astype("float64")
    differences = []
    for name in SCORE_NAMES:
        a = np.bincount(index, weights=[getattr(o, name) for o, _ in pairs]) / per_date
        b = np.bincount(index, weights=[getattr(m, name) for _, m in pairs]) / per_date
        found = block_t_interval(a - b, level, block)
        ratio = persistence_ratio(a - b, block, long_block)
        differences.append(
            ScoreDifference(
                name,
                float(np.mean(a)),
                float(np.mean(b)),
                float(found.estimate),
                float(found.se),
                float(found.low),
                float(found.high),
                se_long=None if ratio is None else float(block_se(a - b, long_block)[0]),
                se_ratio=None if ratio is None or np.isnan(ratio) else float(ratio),
            )
        )
    (flagged,) = (d.se_ratio for d in differences if d.score == PERSISTENCE_SCORE)
    return MatchedComparison(
        **common,
        differences=tuple(differences),
        status=OK,
        persistence_suspected=None if flagged is None else flagged > threshold,
    )


# --- the development record, read as stored ---


@dataclass(frozen=True)
class DevelopmentBucket:
    low: float
    high: float
    n: int
    dates: int | None  # None when the record does not say (then insufficient)
    mean_predicted: float
    observed: float
    ci_low: float
    ci_high: float
    status: str  # a label from the distinct dates (no recomputation)


@dataclass(frozen=True)
class DevelopmentEvidence:
    run_id: str
    data_class: str
    created_at: str
    outcome_end: str | None
    stocks: tuple[str, ...]
    estimator: str
    target: str
    category: str
    candidate: Mapping[str, float]
    baseline: Mapping[str, float] | None  # None when the estimator is B1 itself
    n: int
    dates: int | None
    reported: Mapping[str, float | None] | None  # the stored comparison with B1, reported only
    selected: str
    rule: str
    buckets: tuple[DevelopmentBucket, ...]


def development_evidence(
    evaluation: Mapping, estimator: str, target: str = "target_1", category: str = "increase"
) -> DevelopmentEvidence | None:
    """
    The stored evaluation's numbers for `estimator` on one target: its scores
    and B1's on the same rows (the evaluation scores every model on the same
    rows), the stored comparison, and the reliability bins of one category
    (Jeffreys intervals with the distinct dates as the count, as the
    evaluation computed them). A bin whose record gives no date count is
    insufficient. None when the record has no metrics for the estimator.
    """

    metrics = {row["model"]: row for row in evaluation["metrics"] if row["target"] == target}
    found = metrics.get(estimator)
    if found is None:
        return None
    is_reference = estimator == REFERENCE
    reference = metrics.get(REFERENCE)
    comparison = next(
        (
            row
            for row in evaluation.get("comparisons", ())
            if row["model"] == estimator
            and row["target"] == target
            and row.get("vs", REFERENCE) == REFERENCE
        ),
        None,
    )
    code = CATEGORY_CODES[Category(category)]
    buckets = tuple(
        DevelopmentBucket(
            low=row["bin_low"],
            high=row["bin_high"],
            n=row["n"],
            dates=row.get("dates"),
            mean_predicted=row["mean_predicted"],
            observed=row["observed"],
            ci_low=row["ci_low"],
            ci_high=row["ci_high"],
            status=_status(row.get("dates") or 0, MIN_CLUSTERS),
        )
        for row in evaluation.get("reliability", ())
        if row["model"] == estimator and row["target"] == target and row["category"] == code
    )
    return DevelopmentEvidence(
        run_id=evaluation["run_id"],
        data_class=evaluation["data_class"],
        created_at=evaluation["created_at"],
        outcome_end=evaluation.get("config", {}).get("outcome_end"),
        stocks=tuple(evaluation.get("stocks", ())),
        estimator=estimator,
        target=target,
        category=category,
        candidate={name: found[name] for name in SCORE_NAMES},
        baseline=None
        if is_reference or reference is None
        else {name: reference[name] for name in SCORE_NAMES},
        n=found["n"],
        dates=found.get("dates"),
        reported=None
        if is_reference or comparison is None
        else {
            "rps_difference": comparison["mean_diff"],
            "p_newey_west": comparison.get("p_one_sided"),
            "p_fold_block_t": comparison.get("fold_t_p_one_sided"),
        },
        selected=evaluation["selection"]["selected"],
        rule=evaluation["selection"]["rule"],
        buckets=buckets,
    )
