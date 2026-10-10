"""
Model-agnostic out-of-sample scoring (numpy and pandas only).

Inputs are harness outputs: one row per scored (fold, date, symbol) with a
prediction and the scored label. Every function here runs after all
predictions of a run are complete; nothing is fitted, filtered, or dropped.
Rows a metric cannot use are counted and reported with a reason, never
removed silently: a date whose rank IC is undefined stays in the by-date
table with status "unavailable".

Conventions (tested):
- Rank IC is a Spearman correlation WITHIN one date (average ranks for
  ties), then averaged over dates. It needs at least `min_names` names
  and non-constant predictions and labels on that date; otherwise the date
  is unavailable with reason too_few_names / constant_prediction /
  constant_target. With 4 names it can only take 11 values (steps of 0.2).
- Inference uses one value per date (see inference.py).
- Same-date rank positions run 1..N (1 = highest prediction). A tied group
  spanning positions a..b gives each of those positions the group's mean
  label, so a constant predictor has a top-minus-bottom spread of exactly 0
  and no position is ever chosen at random.
- Pooled deciles use bucket boundaries of the whole out-of-sample sample.
  They are descriptive ("pooled"), mix folds and dates, and are flagged
  degenerate when predictions take fewer distinct values than bins.
- "score" outputs (unfitted controls, noise) are rankings, not calibrated
  forecasts: scale-dependent metrics (MAE, RMSE, R^2, Mincer-Zarnowitz,
  residuals) are unavailable for them with reason score_not_forecast.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_RETURN_COLUMN
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics.inference import BlockDraws, Inference
from stock_agent.model_validation.audit import datetime_ns

SCORED_COLUMNS = ("fold_id", "date", "symbol", "prediction")
OUTPUT_KINDS = ("forecast", "score")
EVALUATION_SCOPE = "evaluation"
MDE_POWER = 0.80


@dataclass(frozen=True, eq=False)
class ScoringContext:
    """
    The scored-session calendar shared by every model of a run.

    All models are scored on the same rows (same folds), so per-date series
    align on one calendar and the bootstrap draws are shared (paired).
    """

    sessions: np.ndarray  # sorted unique scored dates, int64 ns UTC
    fold_of_session: np.ndarray
    draws: BlockDraws
    target_column: str = TARGET_RETURN_COLUMN
    hac_lag: int = inf.DEFAULT_HAC_LAG
    ci_level: float = inf.DEFAULT_CI_LEVEL
    min_names: int = 3
    decile_bins: int = 10
    acf_max_lag: int = 40

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.sessions.astype("datetime64[ns]"), tz="UTC")


def build_context(
    reference: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
    **settings: object,
) -> ScoringContext:
    """Calendar, fold of each session, and shared bootstrap draws."""

    dates = datetime_ns(reference["date"])
    sessions, inverse = np.unique(dates, return_inverse=True)
    folds = reference["fold_id"].to_numpy()
    fold_of_session = np.full(len(sessions), -1, dtype=np.int64)
    fold_of_session[inverse] = folds
    if (fold_of_session[inverse] != folds).any():
        raise ValueError("A scored date belongs to more than one fold.")
    draws = inf.draw_circular_blocks(
        len(sessions), block_length=block_length, reps=bootstrap_reps, seed=seed
    )
    return ScoringContext(sessions, fold_of_session, draws, **settings)


def validate_scored(scored: pd.DataFrame, target_column: str) -> pd.DataFrame:
    missing = [c for c in (*SCORED_COLUMNS, target_column) if c not in scored.columns]
    if missing:
        raise ValueError(f"Scored frame is missing columns: {missing}")
    if scored.duplicated(subset=["date", "symbol"]).any():
        raise ValueError("Scored frame contains duplicate (date, symbol) rows.")
    for column in ("prediction", target_column):
        if not np.isfinite(scored[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Scored {column} must be finite; rows are never dropped.")
    return scored


def _positions(scored: pd.DataFrame, ctx: ScoringContext) -> np.ndarray:
    dates = datetime_ns(scored["date"])
    positions = np.searchsorted(ctx.sessions, dates)
    if (positions >= len(ctx.sessions)).any() or (ctx.sessions[positions] != dates).any():
        raise ValueError("Scored dates are not on the run's scored-session calendar.")
    return positions


def per_session(values: object, positions: np.ndarray, ctx: ScoringContext) -> np.ndarray:
    """Sum of row values per session (0 where a session has no rows)."""

    return np.bincount(
        positions, weights=np.asarray(values, dtype=np.float64), minlength=len(ctx.sessions)
    )


def session_mean(values: object, positions: np.ndarray, ctx: ScoringContext) -> np.ndarray:
    """Mean of row values per session; NaN where a session has no rows."""

    counts = np.bincount(positions, minlength=len(ctx.sessions)).astype(np.float64)
    sums = per_session(values, positions, ctx)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def mean_inferences(series: np.ndarray, ctx: ScoringContext, *, null_value: float = 0.0):
    """Newey-West, fold-block t, and circular block bootstrap for one date series."""

    return [
        inf.newey_west_mean(series, lag=ctx.hac_lag, ci_level=ctx.ci_level, null_value=null_value),
        inf.fold_block_t(series, ctx.fold_of_session, ci_level=ctx.ci_level, null_value=null_value),
        inf.bootstrap_mean(series, ctx.draws, ci_level=ctx.ci_level, null_value=null_value),
    ]


def point(estimate: float, *, n: int, status: str = "ok", reason: str | None = None) -> Inference:
    if status == "ok" and not math.isfinite(estimate):
        status, reason = "unavailable", reason or "undefined"
    return Inference(
        method="point", estimate=float(estimate), n=int(n), status=status, reason=reason
    )


def minimum_detectable(test: Inference, *, power: float = MDE_POWER) -> Inference:
    """
    Smallest true mean the decision test would detect with the given power
    at its own two-sided level: (t_{1-alpha/2, df} + t_{power, df}) * SE.
    """

    if test.status != "ok" or "df" not in test.params:
        return inf.unavailable(test.method, test.reason or "no_standard_error", n=test.n)
    df = test.params["df"]
    level = test.params.get("ci_level", inf.DEFAULT_CI_LEVEL)
    multiplier = inf.student_t_ppf(0.5 + level / 2.0, df) + inf.student_t_ppf(power, df)
    return Inference(
        method=test.method,
        estimate=multiplier * test.se,
        null_value=math.nan,
        n=test.n,
        params={"df": df, "ci_level": level, "power": power, "ci_method": None},
    )


@functools.lru_cache(maxsize=16)
def positive_ic_probability(n_names: int) -> float:
    """P(Spearman > 0) for n names under no skill (exact enumeration, n <= 8)."""

    if n_names < 2 or n_names > 8:
        return math.nan
    ranks = np.arange(n_names, dtype=np.float64)
    permutations = np.array(list(itertools.permutations(range(n_names))), dtype=np.float64)
    centered = ranks - ranks.mean()
    numerators = (permutations - ranks.mean()) @ centered
    return float(np.mean(numerators > 1e-12))


# --- rank IC ---


def rank_ic_by_date(scored: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """One row per scored date: fold_id, n_names, rank_ic, status, reason."""

    target = ctx.target_column
    grouped = scored.groupby("date", sort=True)
    n = grouped["prediction"].transform("size").to_numpy(dtype=np.float64)
    center = (n + 1.0) / 2.0
    dp = grouped["prediction"].rank(method="average").to_numpy() - center
    dt = grouped[target].rank(method="average").to_numpy() - center
    parts = pd.DataFrame(
        {
            "date": scored["date"].to_numpy(),
            "fold_id": scored["fold_id"].to_numpy(),
            "pp": dp * dp,
            "tt": dt * dt,
            "pt": dp * dt,
        }
    )
    per = parts.groupby("date", sort=True).agg(
        fold_id=("fold_id", "first"),
        n_names=("pp", "size"),
        pp=("pp", "sum"),
        tt=("tt", "sum"),
        pt=("pt", "sum"),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = per["pt"] / np.sqrt(per["pp"] * per["tt"])
    reason = np.select(
        [per["n_names"] < ctx.min_names, per["pp"] <= 0.0, per["tt"] <= 0.0],
        ["too_few_names", "constant_prediction", "constant_target"],
        default="",
    )
    ok = reason == ""
    return pd.DataFrame(
        {
            "date": per.index,
            "fold_id": per["fold_id"].to_numpy(dtype=np.int64),
            "n_names": per["n_names"].to_numpy(dtype=np.int64),
            "rank_ic": np.where(ok, ic.to_numpy(), np.nan),
            "status": np.where(ok, "ok", "unavailable"),
            "reason": np.array([value or None for value in reason.tolist()], dtype=object),
        }
    )


def ic_series(by_date: pd.DataFrame, ctx: ScoringContext) -> np.ndarray:
    series = np.full(len(ctx.sessions), np.nan)
    positions = np.searchsorted(ctx.sessions, datetime_ns(by_date["date"]))
    series[positions] = by_date["rank_ic"].to_numpy(dtype=np.float64)
    return series


def _dominant_reason(by_date: pd.DataFrame) -> str:
    reasons = by_date["reason"].dropna()
    return str(reasons.mode().iloc[0]) if len(reasons) else "no_defined_dates"


# --- hit rate ---


def hit_rate_summary(scored: pd.DataFrame, positions: np.ndarray, ctx: ScoringContext) -> dict:
    """
    Sign agreement on rows where neither prediction nor label is exactly 0.

    The benchmark is the hit rate expected if signs were independent,
    p*q + (1-p)*(1-q) (p = share of positive labels, q = share of positive
    predictions), not 50% and not the base rate alone.
    """

    prediction = scored["prediction"].to_numpy(dtype=np.float64)
    label = scored[ctx.target_column].to_numpy(dtype=np.float64)
    usable = (prediction != 0.0) & (label != 0.0)
    result = {
        "n_zero_prediction": int((prediction == 0.0).sum()),
        "n_zero_label": int((label == 0.0).sum()),
        "n_used": int(usable.sum()),
    }
    if not usable.any():
        return {**result, "status": "unavailable", "reason": "no_signed_predictions"}
    hits = (np.sign(prediction) == np.sign(label)).astype(np.float64)
    p = float((label[usable] > 0).mean())
    q = float((prediction[usable] > 0).mean())
    counts = np.bincount(positions[usable], minlength=len(ctx.sessions)).astype(np.float64)
    sums = np.bincount(positions[usable], weights=hits[usable], minlength=len(ctx.sessions))
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.where(counts > 0, sums / counts, np.nan)
    return {
        **result,
        "status": "ok",
        "reason": None,
        "hit_rate": float(hits[usable].mean()),
        "base_rate": p,
        "prediction_positive_share": q,
        "independence_expected": p * q + (1.0 - p) * (1.0 - q),
        "daily": daily,
    }


# --- squared-error comparisons ---


def aligned_pair(model: pd.DataFrame, baseline: pd.DataFrame, target: str) -> pd.DataFrame:
    """Join a model and a baseline on identical keys and identical labels, or raise."""

    keys = ["fold_id", "date", "symbol"]
    left = model[[*keys, "prediction", target]]
    right = baseline[[*keys, "prediction", target]]
    joined = left.merge(right, on=keys, how="outer", suffixes=("", "_baseline"), indicator=True)
    if (joined["_merge"] != "both").any() or len(joined) != len(model):
        raise ValueError("Model and baseline must be scored on exactly the same rows.")
    if not np.array_equal(joined[target].to_numpy(), joined[f"{target}_baseline"].to_numpy()):
        raise ValueError("Model and baseline carry different labels for the same rows.")
    return joined.drop(columns=["_merge", f"{target}_baseline"])


def r2_statistic(sums: Mapping[str, np.ndarray]) -> np.ndarray:
    return 1.0 - sums["sse_model"] / sums["sse_baseline"]


def oos_r2_vs(
    model: pd.DataFrame, baseline: pd.DataFrame, ctx: ScoringContext
) -> tuple[Inference, list[Inference], np.ndarray]:
    """
    R^2 = 1 - SSE_model / SSE_baseline over identical rows (never an average
    of per-fold R^2), with a percentile date-block interval; plus inference
    on the per-date mean squared-error improvement (baseline minus model;
    positive favours the model) and the cumulative Goyal-Welch curve.
    """

    target = ctx.target_column
    joined = aligned_pair(model, baseline, target)
    positions = _positions(joined, ctx)
    label = joined[target].to_numpy(dtype=np.float64)
    sse_model = per_session((label - joined["prediction"].to_numpy()) ** 2, positions, ctx)
    sse_base = per_session((label - joined["prediction_baseline"].to_numpy()) ** 2, positions, ctx)
    counts = np.bincount(positions, minlength=len(ctx.sessions)).astype(np.float64)
    if not sse_base.sum() > 0.0:
        r2 = inf.unavailable("circular_block_bootstrap", "zero_baseline_error", n=len(joined))
    else:
        r2 = inf.bootstrap_statistic(
            {"sse_model": sse_model, "sse_baseline": sse_base},
            r2_statistic,
            ctx.draws,
            ci_level=ctx.ci_level,
            n=len(joined),
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        improvement = np.where(counts > 0, (sse_base - sse_model) / counts, np.nan)
    return r2, mean_inferences(improvement, ctx), np.cumsum(sse_base - sse_model)


def cross_sectional_r2(scored: pd.DataFrame, positions: np.ndarray, ctx: ScoringContext):
    """
    R^2 of date-demeaned forecasts for date-demeaned labels (baseline: any
    forecast that is constant within each date). Removes market timing.
    """

    label = scored[ctx.target_column].to_numpy(dtype=np.float64)
    prediction = scored["prediction"].to_numpy(dtype=np.float64)
    label_dm = label - session_mean(label, positions, ctx)[positions]
    prediction_dm = prediction - session_mean(prediction, positions, ctx)[positions]
    sse_model = per_session((label_dm - prediction_dm) ** 2, positions, ctx)
    sse_base = per_session(label_dm**2, positions, ctx)
    if not sse_base.sum() > 0.0:
        return inf.unavailable("circular_block_bootstrap", "zero_baseline_error", n=len(scored))
    return inf.bootstrap_statistic(
        {"sse_model": sse_model, "sse_baseline": sse_base},
        r2_statistic,
        ctx.draws,
        ci_level=ctx.ci_level,
        n=len(scored),
    )


# --- Mincer-Zarnowitz ---


def _coefficient_inference(
    result: inf.RegressionResult, index: int, *, null_value: float, ctx: ScoringContext
) -> Inference:
    params = {
        "lag": result.lag,
        "kernel": "bartlett",
        "ci_method": "normal",
        "ci_level": ctx.ci_level,
    }
    method = "driscoll_kraay"
    if result.status != "ok":
        return inf.unavailable(
            method, result.reason or "unavailable", n=result.n, null_value=null_value, **params
        )
    estimate = float(result.coefficients[index])
    variance = float(result.covariance[index, index])
    if not variance > 0.0:
        return inf.unavailable(
            method,
            "zero_variance",
            n=result.n,
            estimate=estimate,
            null_value=null_value,
            **params,
        )
    se = math.sqrt(variance)
    z = inf.normal_ppf(0.5 + ctx.ci_level / 2.0)
    return Inference(
        method=method,
        estimate=estimate,
        se=se,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
        p_value=inf.normal_two_sided_p((estimate - null_value) / se),
        null_value=null_value,
        n=result.n,
        params={**params, "n_dates": result.n_dates},
    )


def mincer_zarnowitz(scored: pd.DataFrame, positions: np.ndarray, ctx: ScoringContext) -> dict:
    """
    y = a + b * prediction (pooled) and the within-date version on
    date-demeaned values, with Driscoll-Kraay standard errors. Ideal a = 0,
    b = 1; b < 1 can mean overconfidence or a noisy forecast. A forecast that
    is constant within every fold (zero, pooled mean) identifies b only from
    fold-to-fold drift of a fitted constant, so it is reported unavailable.
    """

    label = scored[ctx.target_column].to_numpy(dtype=np.float64)
    prediction = scored["prediction"].to_numpy(dtype=np.float64)
    names = ("mz_intercept", "mz_slope", "mz_within_date_slope")
    if (scored.groupby("fold_id")["prediction"].nunique() <= 1).all():
        nulls = {"mz_intercept": 0.0, "mz_slope": 1.0, "mz_within_date_slope": 1.0}
        return {
            name: inf.unavailable(
                "driscoll_kraay", "constant_within_fold", n=len(scored), null_value=nulls[name]
            )
            for name in names
        }
    design = np.column_stack([np.ones(len(prediction)), prediction])
    pooled = inf.driscoll_kraay_ols(
        label, design, positions, n_sessions=len(ctx.sessions), lag=ctx.hac_lag
    )
    label_dm = label - session_mean(label, positions, ctx)[positions]
    prediction_dm = prediction - session_mean(prediction, positions, ctx)[positions]
    if not np.any(np.abs(prediction_dm) > 0.0):
        within = inf.unavailable(
            "driscoll_kraay", "constant_within_date", n=len(scored), null_value=1.0
        )
    else:
        within = _coefficient_inference(
            inf.driscoll_kraay_ols(
                label_dm,
                prediction_dm[:, None],
                positions,
                n_sessions=len(ctx.sessions),
                lag=ctx.hac_lag,
            ),
            0,
            null_value=1.0,
            ctx=ctx,
        )
    return {
        "mz_intercept": _coefficient_inference(pooled, 0, null_value=0.0, ctx=ctx),
        "mz_slope": _coefficient_inference(pooled, 1, null_value=1.0, ctx=ctx),
        "mz_within_date_slope": within,
    }


# --- rank positions and buckets ---


def rank_position_values(scored: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """
    Realized label at each same-date rank position (1 = highest prediction).

    Columns: date, n_names, position, realized, realized_minus_date_mean.
    Tied names share their group's mean label at every position the group
    spans.
    """

    target = ctx.target_column
    frame = scored[["date", "symbol", "prediction", target]].copy()
    grouped = frame.groupby("date", sort=True)
    frame["n_names"] = grouped["prediction"].transform("size")
    frame["first_position"] = grouped["prediction"].rank(method="min", ascending=False)
    frame["date_mean"] = grouped[target].transform("mean")
    groups = (
        frame.groupby(["date", "first_position"], sort=True)
        .agg(
            n_names=("n_names", "first"),
            size=(target, "size"),
            realized=(target, "mean"),
            date_mean=("date_mean", "first"),
        )
        .reset_index()
    )
    sizes = groups["size"].to_numpy()
    expanded = groups.loc[np.repeat(groups.index.to_numpy(), sizes)].reset_index(drop=True)
    offsets = np.concatenate([np.arange(size) for size in sizes]) if len(sizes) else []
    expanded["position"] = (expanded["first_position"].to_numpy() + offsets).astype(np.int64)
    expanded["realized_minus_date_mean"] = expanded["realized"] - expanded["date_mean"]
    return expanded[["date", "n_names", "position", "realized", "realized_minus_date_mean"]]


def rank_position_occupancy(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Share of dates on which each symbol occupies each position (ties split
    evenly), per cross-section size. Shows whether position 1 is mostly one
    symbol, as for a static tilt.
    """

    frame = scored[["date", "symbol", "prediction"]].copy()
    grouped = frame.groupby("date", sort=True)
    frame["n_names"] = grouped["prediction"].transform("size")
    frame["first_position"] = grouped["prediction"].rank(method="min", ascending=False)
    frame["size"] = frame.groupby(["date", "first_position"])["symbol"].transform("size")
    sizes = frame["size"].to_numpy()
    expanded = frame.loc[np.repeat(frame.index.to_numpy(), sizes)].reset_index(drop=True)
    offsets = np.concatenate([np.arange(size) for size in sizes]) if len(sizes) else []
    expanded["position"] = (expanded["first_position"].to_numpy() + offsets).astype(np.int64)
    expanded["weight"] = 1.0 / expanded["size"]
    dates_per_size = frame.groupby("n_names")["date"].nunique()
    share = expanded.groupby(["n_names", "position", "symbol"])["weight"].sum().reset_index()
    share["share"] = share["weight"] / share["n_names"].map(dates_per_size)
    return share[["n_names", "position", "symbol", "share"]]


def pooled_decile_bins(scored: pd.DataFrame, bins: int) -> tuple[np.ndarray, str | None]:
    """
    Bin 1..bins of each row by pooled average rank of its prediction (ties
    share a bin). Returns (bins, degenerate_reason).
    """

    ranks = scored["prediction"].rank(method="average").to_numpy()
    n = len(ranks)
    assigned = np.minimum(bins, np.floor((ranks - 1.0) * bins / n).astype(np.int64) + 1)
    if (scored.groupby("date")["prediction"].nunique() <= 1).all():
        # e.g. the pooled mean: bins group whole folds, sorted by time
        return assigned, "constant_within_date"
    distinct = scored["prediction"].nunique()
    return assigned, ("fewer_distinct_predictions_than_bins" if distinct < bins else None)


# --- residuals ---


def autocorrelations(series: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF at lags 1..max_lag of a gappy session series (NaN = absent)."""

    present = np.isfinite(series)
    if present.sum() <= max_lag + 1:
        return np.full(max_lag, np.nan)
    mean = series[present].mean()
    centered = np.where(present, series - mean, 0.0)
    gamma0 = centered @ centered
    if not gamma0 > 0.0:
        return np.full(max_lag, np.nan)
    return np.array([centered[k:] @ centered[:-k] / gamma0 for k in range(1, max_lag + 1)])


# --- static orderings (selection reference, labels only) ---


def static_ordering_ics(
    scored: pd.DataFrame, ctx: ScoringContext, *, max_symbols: int = 6
) -> tuple[pd.DataFrame, str | None]:
    """
    Mean rank IC of every fixed ranking of the symbols (e.g. 24 orderings of
    4 names), using only labels. A per-symbol (tilt) model's IC should be
    read against this distribution: it shows how much IC a ranking that
    never changes can produce on this hindsight-selected universe.
    """

    symbols = sorted(scored["symbol"].unique())
    if len(symbols) > max_symbols:
        return pd.DataFrame(), "too_many_symbols"
    per_date = scored.groupby("date")["symbol"].nunique()
    if (per_date != len(symbols)).any() or len(symbols) < ctx.min_names:
        return pd.DataFrame(), "varying_cross_section"
    wide = scored.pivot(index="date", columns="symbol", values=ctx.target_column)[symbols]
    label_ranks = wide.rank(axis=1, method="average").to_numpy()
    label_dev = label_ranks - label_ranks.mean(axis=1, keepdims=True)
    orderings = list(itertools.permutations(range(len(symbols))))
    # score of symbol j under an ordering = its position from the top
    score_ranks = np.empty((len(orderings), len(symbols)))
    for row, ordering in enumerate(orderings):
        score_ranks[row, list(ordering)] = np.arange(len(symbols), 0, -1)
    score_dev = score_ranks - score_ranks.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ics = (label_dev @ score_dev.T) / np.sqrt(
            (label_dev**2).sum(axis=1, keepdims=True) * (score_dev**2).sum(axis=1)[None, :]
        )
    means = np.nanmean(ics, axis=0)
    labels = [">".join(symbols[j] for j in ordering) for ordering in orderings]
    return pd.DataFrame({"draw": np.arange(len(orderings)), "detail": labels, "value": means}), None


def compare_to_null(
    observed: float, null_values: Sequence[float], kind: str, *, exhaustive: bool = False
) -> Inference:
    """
    Observed minus the null mean, with a one-sided empirical p-value
    (1 + #{null >= observed}) / (1 + draws) for random draws.

    For an exhaustive enumeration (static orderings) there is no sampling
    and the model is not exchangeable with the enumerated rankings, so no
    p-value is reported; use static_ordering_share_at_least instead.
    """

    values = np.asarray([v for v in null_values if math.isfinite(v)], dtype=np.float64)
    params = {"null_kind": kind, "alternative": "greater", "draws": len(values)}
    if exhaustive and len(values) and math.isfinite(observed):
        return Inference(
            method="null_enumeration",
            estimate=float(observed - values.mean()),
            se=float(values.std(ddof=0)),
            null_value=math.nan,
            n=len(values),
            params=params,
        )
    if not math.isfinite(observed):
        return inf.unavailable("null_distribution", "observed_unavailable", **params)
    if len(values) == 0:
        return inf.unavailable("null_distribution", "no_null_draws", **params)
    return Inference(
        method="null_distribution",
        estimate=float(observed - values.mean()),
        se=float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        p_value=float((1 + (values >= observed).sum()) / (1 + len(values))),
        n=len(values),
        params=params,
    )


# --- assembling one model's evaluation rows ---


def metric_rows(
    model: str,
    role: str,
    metric: str,
    inferences: Sequence[Inference],
    *,
    fold_id: int | None = None,
) -> list[dict]:
    """Rows of the `metrics` table, one per inference method."""

    rows = []
    for item in inferences:
        params = item.params
        is_point = item.method == "point"
        rows.append(
            {
                "model": model,
                "role": role,
                "scope": EVALUATION_SCOPE,
                "fold_id": fold_id,
                "metric": metric,
                "inference_method": item.method,
                "estimate": item.estimate,
                "se": item.se,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "ci_level": params.get("ci_level", math.nan),
                "ci_method": params.get("ci_method"),
                "p_value": item.p_value,
                "null_value": math.nan if is_point else item.null_value,
                "n": item.n,
                "n_eff": item.n_eff,
                "hac_lag": params.get("lag"),
                "block_length": params.get("block_length"),
                "bootstrap_reps": params.get("reps"),
                "seed": params.get("seed"),
                "df": float(params.get("df", math.nan)),
                "status": item.status,
                "reason": item.reason,
            }
        )
    return rows


# Curve intervals are computed around the mean of a per-date series (for
# pooled deciles, the per-date bucket means; the plotted value of a decile is
# its mean over rows). The stored method, level and lag are the ones the
# inference call used, so a reader never has to assume them. Bounds are NaN
# when the interval is unavailable; the method fields still say what was tried.
CURVE_CI_METHOD = "newey_west_normal_over_dates"


def _curve_interval(summary: Inference) -> dict:
    params = summary.params
    method = f"{summary.method}_{params['ci_method']}_over_dates"
    if method != CURVE_CI_METHOD:
        raise ValueError(f"Unexpected curve interval method {method!r}.")
    return {
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "ci_method": method,
        "ci_level": params["ci_level"],
        "hac_lag": params["lag"],
    }


def curve_row(model: str, role: str, curve: str, value: float, **fields: object) -> dict:
    status = fields.pop("status", "ok")
    reason = fields.pop("reason", None)
    if status == "ok" and not math.isfinite(value):
        status, reason = "unavailable", reason or "undefined"
    return {
        "model": model,
        "role": role,
        "scope": EVALUATION_SCOPE,
        "curve": curve,
        "group": fields.get("group"),
        "fold_id": fields.get("fold_id"),
        "date": fields.get("date"),
        "x": float(fields.get("x", math.nan)),
        "x_label": fields.get("x_label"),
        "value": float(value),
        "ci_low": float(fields.get("ci_low", math.nan)),
        "ci_high": float(fields.get("ci_high", math.nan)),
        "ci_method": fields.get("ci_method"),
        "ci_level": float(fields.get("ci_level", math.nan)),
        "hac_lag": fields.get("hac_lag"),
        "n": fields.get("n"),
        "status": status,
        "reason": reason,
    }


@dataclass(frozen=True, eq=False)
class ModelScore:
    metrics: list[dict]
    curves: list[dict]
    rank_ic_by_date: pd.DataFrame
    mean_rank_ic: tuple[Inference, ...]  # newey_west, fold_block_t, bootstrap

    @property
    def n_dates_defined(self) -> int:
        return int((self.rank_ic_by_date["status"] == "ok").sum())

    def mean_rank_ic_by(self, method: str) -> Inference:
        return next(item for item in self.mean_rank_ic if item.method == method)


def _unavailable_all(reason: str, n: int = 0) -> list[Inference]:
    return [
        inf.unavailable(method, reason, n=n)
        for method in ("newey_west", "fold_block_t", "circular_block_bootstrap")
    ]


SCALE_METRICS = (
    "mean_residual",
    "mae",
    "rmse",
    "cross_sectional_r2",
    "mz_intercept",
    "mz_slope",
    "mz_within_date_slope",
)


def score_model(
    model: str,
    role: str,
    output_kind: str,
    scored: pd.DataFrame,
    ctx: ScoringContext,
    *,
    baselines: Mapping[str, pd.DataFrame],
    horizon: int,
) -> ModelScore:
    """Every evaluation metric and curve for one model's complete predictions."""

    if output_kind not in OUTPUT_KINDS:
        raise ValueError(f"output_kind must be one of {OUTPUT_KINDS}; got {output_kind!r}.")
    validate_scored(scored, ctx.target_column)
    scored = scored.sort_values(["date", "symbol"]).reset_index(drop=True)
    positions = _positions(scored, ctx)
    target = ctx.target_column
    metrics: list[dict] = []
    curves: list[dict] = []

    def add(metric: str, inferences: Sequence[Inference], fold_id: int | None = None) -> None:
        metrics.extend(metric_rows(model, role, metric, inferences, fold_id=fold_id))

    # rank IC
    by_date = rank_ic_by_date(scored, ctx)
    series = ic_series(by_date, ctx)
    defined = int((by_date["status"] == "ok").sum())
    if defined:
        ic_inferences = mean_inferences(series, ctx)
        add("mean_rank_ic", ic_inferences)
        add("mean_rank_ic_minimum_detectable", [minimum_detectable(ic_inferences[1])])
        add("share_dates_rank_ic_positive", [point(float((series > 0).sum()) / defined, n=defined)])
        names = by_date.loc[by_date["status"] == "ok", "n_names"]
        add(
            "share_dates_rank_ic_positive_expected_under_no_skill",
            [point(float(np.mean([positive_ic_probability(int(k)) for k in names])), n=defined)],
        )
    else:
        ic_inferences = _unavailable_all(_dominant_reason(by_date))
        add("mean_rank_ic", ic_inferences)
    add("n_dates_rank_ic_unavailable", [point(len(by_date) - defined, n=len(by_date))])
    for fold_id, group in by_date.groupby("fold_id", sort=True):
        values = group["rank_ic"].dropna()
        add(
            "mean_rank_ic",
            [point(values.mean() if len(values) else math.nan, n=len(values))],
            fold_id=int(fold_id),
        )
    for row in by_date.itertuples(index=False):
        curves.append(
            curve_row(
                model,
                role,
                "rank_ic",
                row.rank_ic,
                date=row.date,
                fold_id=int(row.fold_id),
                n=int(row.n_names),
                status=row.status,
                reason=row.reason,
            )
        )

    # hit rate
    hit = hit_rate_summary(scored, positions, ctx)
    if hit["status"] == "ok":
        hit_inferences = mean_inferences(hit["daily"], ctx, null_value=hit["independence_expected"])
        if hit["prediction_positive_share"] in (0.0, 1.0):
            # One sign for every prediction: the hit rate equals the base rate
            # and its benchmark by construction.
            hit_inferences = [
                dataclasses.replace(item, status="warning", reason="constant_sign_prediction")
                for item in hit_inferences
            ]
        add("hit_rate", hit_inferences)
        for name in ("base_rate", "prediction_positive_share", "independence_expected"):
            add(f"hit_{name}", [point(hit[name], n=hit["n_used"])])
    else:
        add("hit_rate", _unavailable_all(hit["reason"], n=hit["n_used"]))
    add("hit_rows_used", [point(hit["n_used"], n=len(scored))])
    add("hit_rows_zero_prediction", [point(hit["n_zero_prediction"], n=len(scored))])
    signs = (scored["prediction"] != 0) & (scored[target] != 0)
    for fold_id, group in scored[signs].groupby("fold_id", sort=True):
        hits = np.sign(group["prediction"]) == np.sign(group[target])
        add("hit_rate", [point(hits.mean(), n=len(group))], fold_id=int(fold_id))

    # same-date rank positions
    positions_table = rank_position_values(scored, ctx)
    for n_names, group in positions_table.groupby("n_names", sort=True):
        for position, rows in group.groupby("position", sort=True):
            row_positions = np.searchsorted(ctx.sessions, datetime_ns(rows["date"]))
            for curve, column in (
                ("rank_position_realized", "realized"),
                ("rank_position_realized_minus_date_mean", "realized_minus_date_mean"),
            ):
                series_p = np.full(len(ctx.sessions), np.nan)
                series_p[row_positions] = rows[column].to_numpy()
                summary = inf.newey_west_mean(series_p, lag=ctx.hac_lag, ci_level=ctx.ci_level)
                curves.append(
                    curve_row(
                        model,
                        role,
                        curve,
                        float(np.nanmean(series_p)),
                        group=f"n_names={n_names}",
                        x=float(position),
                        n=len(rows),
                        **_curve_interval(summary),
                    )
                )
    spread = np.zeros(len(ctx.sessions))
    for position_rule, sign in ((1, 1.0), (None, -1.0)):
        target_position = positions_table["n_names"] if position_rule is None else position_rule
        rows = positions_table[positions_table["position"] == target_position]
        spread[np.searchsorted(ctx.sessions, datetime_ns(rows["date"]))] += sign * rows[
            "realized"
        ].to_numpy(dtype=np.float64)
    add("top_minus_bottom", mean_inferences(spread, ctx))
    for row in rank_position_occupancy(scored).itertuples(index=False):
        curves.append(
            curve_row(
                model,
                role,
                "rank_position_occupancy",
                row.share,
                group=f"n_names={row.n_names}",
                x=float(row.position),
                x_label=row.symbol,
            )
        )

    # pooled deciles (descriptive calibration / bucket returns)
    bins, degenerate = pooled_decile_bins(scored, ctx.decile_bins)
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        bin_series = np.full(len(ctx.sessions), np.nan)
        daily = session_mean(scored[target].to_numpy()[in_bin], positions[in_bin], ctx)
        bin_series[:] = daily
        summary = inf.newey_west_mean(bin_series, lag=ctx.hac_lag, ci_level=ctx.ci_level)
        curves.append(
            curve_row(
                model,
                role,
                "pooled_decile_realized",
                float(scored[target].to_numpy()[in_bin].mean()),
                group="pooled",
                x=float(scored["prediction"].to_numpy()[in_bin].mean()),
                x_label=str(int(bin_id)),
                n=int(in_bin.sum()),
                **_curve_interval(summary),
                status="warning" if degenerate else "ok",
                reason=degenerate,
            )
        )

    # scale-dependent metrics (forecasts only)
    if output_kind == "score":
        for name in SCALE_METRICS:
            add(name, [inf.unavailable("point", "score_not_forecast", n=len(scored))])
        for baseline in baselines:
            for name in (f"oos_r2_vs_{baseline}", f"mse_improvement_vs_{baseline}"):
                add(name, [inf.unavailable("point", "score_not_forecast", n=len(scored))])
        return ModelScore(metrics, curves, by_date, tuple(ic_inferences))

    label = scored[target].to_numpy(dtype=np.float64)
    residual = label - scored["prediction"].to_numpy(dtype=np.float64)
    add("mean_residual", mean_inferences(session_mean(residual, positions, ctx), ctx))
    add("mae", [point(np.abs(residual).mean(), n=len(residual))])
    add("rmse", [point(math.sqrt((residual**2).mean()), n=len(residual))])
    for fold_id, rows in pd.DataFrame({"fold_id": scored["fold_id"], "e": residual}).groupby(
        "fold_id"
    ):
        e = rows["e"].to_numpy()
        add("mean_residual", [point(e.mean(), n=len(e))], fold_id=int(fold_id))
        add(
            "residual_std",
            [point(e.std(ddof=1) if len(e) > 1 else math.nan, n=len(e))],
            fold_id=int(fold_id),
        )
        add("mae", [point(np.abs(e).mean(), n=len(e))], fold_id=int(fold_id))
        add("rmse", [point(math.sqrt((e**2).mean()), n=len(e))], fold_id=int(fold_id))
    series_by_label = {"date_mean": session_mean(residual, positions, ctx)}
    for symbol in sorted(scored["symbol"].unique()):
        mask = (scored["symbol"] == symbol).to_numpy()
        symbol_series = np.full(len(ctx.sessions), np.nan)
        symbol_series[positions[mask]] = residual[mask]
        series_by_label[str(symbol)] = symbol_series
    for label_name, values in series_by_label.items():
        for lag, value in enumerate(autocorrelations(values, ctx.acf_max_lag), start=1):
            curves.append(
                curve_row(
                    model,
                    role,
                    "residual_acf",
                    value,
                    x=float(lag),
                    x_label=label_name,
                    group="overlap_expected" if lag < horizon else "beyond_overlap",
                    n=int(np.isfinite(values).sum()),
                )
            )
    add("cross_sectional_r2", [cross_sectional_r2(scored, positions, ctx)])
    for name, result in mincer_zarnowitz(scored, positions, ctx).items():
        add(name, [result])
    for baseline_name, baseline in baselines.items():
        if baseline_name == model:
            continue
        r2, improvement, cumulative = oos_r2_vs(scored, baseline, ctx)
        add(f"oos_r2_vs_{baseline_name}", [r2])
        add(f"mse_improvement_vs_{baseline_name}", improvement)
        for date, value in zip(ctx.dates, cumulative, strict=True):
            curves.append(
                curve_row(
                    model, role, "cumulative_sse_improvement", value, group=baseline_name, date=date
                )
            )
    return ModelScore(metrics, curves, by_date, tuple(ic_inferences))
