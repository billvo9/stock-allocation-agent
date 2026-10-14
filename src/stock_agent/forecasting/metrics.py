"""
Experiment-level metrics on top of T2's scoring (model_diagnostics.scoring).

Everything is computed from complete out-of-fold predictions on identical
rows, with one value per scored date before inference (fold-block t decides;
Newey-West and the circular block bootstrap are reported beside it, as in
T2). Rows reuse T2's metrics columns plus `reference`, the model a paired
comparison is against.

    rank_ic_coverage            share of scored dates with a defined rank IC
    coverage_weighted_rank_ic   mean rank IC with undefined dates counted as
                                zero (the PRD rung L1 estimand)
    rank_ic_difference          paired per-date difference (model - reference),
                                undefined dates counted as zero for both
    oos_r2_vs_reference         1 - SSE_model / SSE_reference on identical rows
    mse_improvement_vs_reference  per-date MSE(reference) - MSE(model)
    prior_trials                earlier registrations in the run's research scopes
                                (the multiple-testing count recorded per run)
    holm_adjusted_p             Holm step-down adjustment of the candidates'
                                one-sided fold-block p-values for
                                coverage_weighted_rank_ic > 0 (information)
    interval_coverage           realized share inside a central interval,
                                for interval forecasts only (calibration is
                                not computed for point forecasts)

None of these changes a T2 statistic; they are additional rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from stock_agent.model_diagnostics import contract, scoring
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics.inference import Inference
from stock_agent.model_validation.audit import datetime_ns

DECISION_METHOD = "fold_block_t"
COLUMNS = (*contract.SCHEMAS["metrics"], "reference")


def rows(
    model: str,
    role: str,
    metric: str,
    inferences: Sequence[Inference],
    *,
    reference: str | None = None,
) -> list[dict]:
    return [
        {**row, "reference": reference}
        for row in scoring.metric_rows(model, role, metric, inferences)
    ]


def _zero_filled_ic(scored: pd.DataFrame, ctx: scoring.ScoringContext) -> tuple[np.ndarray, int]:
    by_date = scoring.rank_ic_by_date(scored.sort_values(["date", "symbol"]), ctx)
    series = scoring.ic_series(by_date, ctx)
    defined = int(np.isfinite(series).sum())
    return np.where(np.isfinite(series), series, 0.0), defined


def coverage_metrics(model: str, role: str, scored: pd.DataFrame, ctx) -> list[dict]:
    """Rank-IC coverage and the coverage-weighted (zero-filled) mean rank IC."""

    filled, defined = _zero_filled_ic(scored, ctx)
    total = len(ctx.sessions)
    result = rows(model, role, "rank_ic_coverage", [scoring.point(defined / total, n=total)])
    if defined == 0:
        unavailable = [
            inf.unavailable(method, "no_defined_dates", n=total)
            for method in ("newey_west", DECISION_METHOD, "circular_block_bootstrap")
        ]
        return result + rows(model, role, "coverage_weighted_rank_ic", unavailable)
    return result + rows(
        model, role, "coverage_weighted_rank_ic", scoring.mean_inferences(filled, ctx)
    )


def increment_metrics(
    model: str,
    role: str,
    scored: pd.DataFrame,
    reference_name: str,
    reference: pd.DataFrame,
    ctx,
    *,
    forecast: bool = True,
) -> list[dict]:
    """Paired increments of `model` over `reference` on identical rows."""

    own, _ = _zero_filled_ic(scored, ctx)
    other, _ = _zero_filled_ic(reference, ctx)
    result = rows(
        model,
        role,
        "rank_ic_difference",
        scoring.mean_inferences(own - other, ctx),
        reference=reference_name,
    )
    if not forecast:
        return result
    r2, improvement, _ = scoring.oos_r2_vs(scored, reference, ctx)
    result += rows(model, role, "oos_r2_vs_reference", [r2], reference=reference_name)
    result += rows(
        model, role, "mse_improvement_vs_reference", improvement, reference=reference_name
    )
    return result


def one_sided_p(row: Mapping[str, object]) -> float:
    """One-sided p-value for estimate > null from a fold-block t row."""

    se, df = float(row["se"]), float(row["df"])
    if row["status"] != "ok" or not se > 0 or not math.isfinite(df):
        return math.nan
    t = (float(row["estimate"]) - float(row["null_value"])) / se
    return 1.0 - inf.student_t_cdf(t, df)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values; NaN inputs stay NaN and are not counted."""

    finite = sorted((p, name) for name, p in p_values.items() if math.isfinite(p))
    m = len(finite)
    adjusted: dict[str, float] = {name: math.nan for name in p_values}
    running = 0.0
    for rank, (p, name) in enumerate(finite):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def holm_rows(candidate_rows: Mapping[str, Mapping[str, object]], roles: Mapping[str, str]):
    """Holm-adjusted one-sided p of coverage_weighted_rank_ic > 0 across candidates."""

    raw = {name: one_sided_p(row) for name, row in candidate_rows.items()}
    adjusted = holm_adjust(raw)
    result = []
    for name in candidate_rows:
        item = Inference(
            method="holm_one_sided_fold_block_t",
            estimate=adjusted[name],
            p_value=adjusted[name],
            n=len([p for p in raw.values() if math.isfinite(p)]),
            status="ok" if math.isfinite(adjusted[name]) else "unavailable",
            reason=None if math.isfinite(adjusted[name]) else "no_test",
            params={"family": "coverage_weighted_rank_ic > 0", "raw_p": raw[name]},
        )
        result += rows(name, roles[name], "holm_adjusted_p", [item])
    return result


def prior_trials(registry, scopes) -> int:
    """
    Earlier registrations in the given research scopes. Every registration is a
    prior trial, even of the same variant: a rerun on new data or with a new
    seed is another look at the same question, so distinct variant ids would
    undercount it.
    """

    return sum(len(registry.entries(scope)) for scope in scopes)


def interval_coverage(
    model: str,
    role: str,
    scored: pd.DataFrame,
    ctx,
    *,
    level: float,
    target_column: str = "target_return",
) -> list[dict]:
    """
    Calibration of a central interval: per-date share of labels inside
    [interval_low, interval_high], inferred against the nominal level.
    """

    frame = scored.sort_values(["date", "symbol"]).reset_index(drop=True)
    inside = (
        (frame[target_column] >= frame["interval_low"])
        & (frame[target_column] <= frame["interval_high"])
    ).to_numpy(dtype=np.float64)
    positions = np.searchsorted(ctx.sessions, datetime_ns(frame["date"]))
    series = scoring.session_mean(inside, positions, ctx)
    metric = f"interval_coverage_{round(level * 100)}"
    return rows(model, role, metric, scoring.mean_inferences(series, ctx, null_value=level))


def table(metric_rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """The experiment metrics table (T2 metric columns plus `reference`)."""

    frame = pd.DataFrame(list(metric_rows), columns=list(COLUMNS))
    types = {
        "fold_id": "Int64",
        "n": "Int64",
        "hac_lag": "Int64",
        "block_length": "Int64",
        "bootstrap_reps": "Int64",
        "seed": "Int64",
    }
    for column in ("model", "role", "scope", "metric", "inference_method", "status"):
        types[column] = "string"
    for column in ("ci_method", "reason", "reference"):
        types[column] = "string"
    for column in (
        "estimate",
        "se",
        "ci_low",
        "ci_high",
        "ci_level",
        "p_value",
        "null_value",
        "n_eff",
        "df",
    ):
        types[column] = "float64"
    return frame.astype(types)
