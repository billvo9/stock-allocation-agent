"""Experiment metrics: coverage, zero-filled rank IC, paired increments, Holm, intervals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.forecasting import metrics as em
from stock_agent.model_diagnostics import scoring


def scored_frame(n_dates: int = 120, n_names: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates, tz="UTC")
    rows = []
    for index, date in enumerate(dates):
        labels = rng.randn(n_names) * 0.02
        for name in range(n_names):
            rows.append(
                {
                    "fold_id": index // 30,
                    "date": date,
                    "symbol": f"S{name}",
                    "prediction": labels[name] + rng.randn() * 0.02,
                    "target_return": labels[name],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def context():
    frame = scored_frame()
    ctx = scoring.build_context(frame, block_length=20, bootstrap_reps=49, seed=1)
    return frame, ctx


def _row(rows, metric, method="fold_block_t", reference=None):
    return next(
        r
        for r in rows
        if r["metric"] == metric
        and r["inference_method"] == method
        and (reference is None or r["reference"] == reference)
    )


def test_coverage_counts_constant_dates_as_zero(context):
    frame, ctx = context
    sessions = frame["date"].drop_duplicates().sort_values()
    constant = frame.copy()
    constant.loc[constant["date"].isin(sessions.iloc[:60]), "prediction"] = 0.0
    full = em.coverage_metrics("m", "candidate", frame, ctx)
    half = em.coverage_metrics("m", "candidate", constant, ctx)
    assert _row(full, "rank_ic_coverage", "point")["estimate"] == 1.0
    assert _row(half, "rank_ic_coverage", "point")["estimate"] == 0.5
    by_date = scoring.rank_ic_by_date(constant.sort_values(["date", "symbol"]), ctx)
    expected = by_date["rank_ic"].fillna(0.0).mean()
    weighted = _row(half, "coverage_weighted_rank_ic", "newey_west")["estimate"]
    assert weighted == pytest.approx(expected)
    assert weighted < by_date["rank_ic"].dropna().mean()  # zero-filled, not conditional


def test_a_model_never_defined_has_no_weighted_ic(context):
    frame, ctx = context
    rows = em.coverage_metrics("zero", "null", frame.assign(prediction=0.0), ctx)
    assert _row(rows, "rank_ic_coverage", "point")["estimate"] == 0.0
    assert _row(rows, "coverage_weighted_rank_ic")["reason"] == "no_defined_dates"


def test_increments_against_itself_are_exactly_zero(context):
    frame, ctx = context
    rows = em.increment_metrics("m", "candidate", frame, "m2", frame.copy(), ctx)
    assert _row(rows, "rank_ic_difference", "newey_west")["estimate"] == 0.0
    assert _row(rows, "oos_r2_vs_reference", "circular_block_bootstrap")["estimate"] == 0.0
    assert all(r["reference"] == "m2" for r in rows)


def test_increments_are_paired_and_signed_model_minus_reference(context):
    frame, ctx = context
    worse = frame.assign(
        prediction=frame["prediction"] + np.random.RandomState(3).randn(len(frame))
    )
    rows = em.increment_metrics("good", "candidate", frame, "worse", worse, ctx)
    assert _row(rows, "rank_ic_difference")["estimate"] > 0
    assert _row(rows, "mse_improvement_vs_reference")["estimate"] > 0
    assert _row(rows, "oos_r2_vs_reference", "circular_block_bootstrap")["estimate"] > 0


def test_holm_adjustment_matches_the_step_down_rule():
    adjusted = em.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": math.nan})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)  # monotone: max(0.06, 1 * 0.04)
    assert math.isnan(adjusted["d"])


def test_one_sided_p_uses_the_fold_block_t_distribution():
    row = {"status": "ok", "se": 0.01, "df": 9.0, "estimate": 0.02, "null_value": 0.0}
    p = em.one_sided_p(row)
    assert 0.03 < p < 0.04  # t = 2 with 9 df
    assert math.isnan(em.one_sided_p({**row, "status": "unavailable"}))


def test_interval_coverage_is_tested_against_its_nominal_level(context):
    frame, ctx = context
    band = frame.assign(
        interval_low=frame["prediction"] - 0.02, interval_high=frame["prediction"] + 0.02
    )
    rows = em.interval_coverage("m", "candidate", band, ctx, level=0.8)
    row = _row(rows, "interval_coverage_80", "fold_block_t")
    inside = (band["target_return"] >= band["interval_low"]) & (
        band["target_return"] <= band["interval_high"]
    )
    assert row["status"] == "ok" and row["null_value"] == 0.8
    assert row["estimate"] == pytest.approx(inside.groupby(band["fold_id"]).mean().mean())
    nowhere = frame.assign(interval_low=5.0, interval_high=6.0)
    empty = _row(em.interval_coverage("m", "c", nowhere, ctx, level=0.5), "interval_coverage_50")
    assert empty["estimate"] == 0.0


def test_the_table_keeps_t2_columns_plus_the_reference(context):
    frame, ctx = context
    table = em.table(em.coverage_metrics("m", "candidate", frame, ctx))
    assert list(table.columns) == list(em.COLUMNS)
    assert table["reference"].isna().all()


def test_every_earlier_registration_counts_as_a_prior_trial(tmp_path):
    from test_registry import Clock, register, synthetic_record

    from stock_agent.registry import Registry

    registry = Registry(tmp_path / "registry", tmp_path, clock=Clock())
    scope = "equity_xs:synthetic:h20"
    assert em.prior_trials(registry, [scope]) == 0
    for run in ("r1", "r2", "r3"):  # the same variant, three looks
        register(registry, synthetic_record(run))
    assert em.prior_trials(registry, [scope]) == 3
    assert em.prior_trials(registry, [scope, "other:scope"]) == 3
