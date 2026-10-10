"""
Out-of-sample scoring conventions, pinned with small hand-built frames:
within-date rank IC with average-rank ties, explicit unavailable statuses,
hit rate against the independence benchmark, R^2 against an explicit
baseline on identical rows, calibration slopes, and tie-aware rank
positions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics import scoring as s

SYMBOLS = ("A", "B", "C", "D")


def _frame(predictions, labels, *, dates=None, symbols=SYMBOLS, fold_of=None):
    """Rows date-major: predictions/labels shaped (n_dates, n_names)."""

    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_dates, names = predictions.shape
    dates = dates if dates is not None else pd.bdate_range("2022-01-03", periods=n_dates, tz="UTC")
    fold_of = fold_of if fold_of is not None else (np.arange(n_dates) // max(1, n_dates // 4))
    return pd.DataFrame(
        {
            "fold_id": np.repeat(fold_of, names).astype(np.int64),
            "date": np.repeat(dates, names),
            "symbol": np.tile(symbols[:names], n_dates),
            "prediction": predictions.ravel(),
            "target_return": labels.ravel(),
        }
    )


def _context(frame, block_length=2, **settings):
    return s.build_context(frame, block_length=block_length, bootstrap_reps=49, seed=0, **settings)


# --- rank IC ---


def test_rank_ic_is_computed_within_each_date_not_across_dates():
    # Each date: higher prediction -> lower label (IC = -1), but date levels
    # align, so a pooled correlation across dates is strongly positive.
    levels = np.arange(10.0)[:, None]
    predictions = levels + np.array([0.0, 0.1, 0.2, 0.3])
    labels = levels + np.array([0.3, 0.2, 0.1, 0.0])
    frame = _frame(predictions, labels)
    by_date = s.rank_ic_by_date(frame, _context(frame))
    np.testing.assert_allclose(by_date["rank_ic"], -1.0)
    assert np.corrcoef(frame["prediction"], frame["target_return"])[0, 1] > 0.9


def test_spearman_uses_average_ranks_for_ties():
    frame = _frame([[1, 1, 2, 3]], [[1, 2, 3, 4]])
    ic = s.rank_ic_by_date(frame, _context(frame))["rank_ic"].iloc[0]
    assert ic == pytest.approx(math.sqrt(0.9))


def test_rank_ic_is_invariant_to_a_date_constant_and_positive_scale():
    rng = np.random.RandomState(0)
    predictions, labels = rng.randn(6, 4), rng.randn(6, 4)
    base = _frame(predictions, labels)
    shifted = _frame(predictions * 3.0 + np.arange(6)[:, None], labels - 5.0)
    np.testing.assert_allclose(
        s.rank_ic_by_date(base, _context(base))["rank_ic"],
        s.rank_ic_by_date(shifted, _context(shifted))["rank_ic"],
    )


def test_four_name_rank_ic_lies_on_a_grid_of_steps_of_point_two():
    rng = np.random.RandomState(1)
    frame = _frame(rng.randn(200, 4), rng.randn(200, 4))
    values = s.rank_ic_by_date(frame, _context(frame))["rank_ic"].to_numpy()
    assert np.allclose(values * 5, np.round(values * 5))


def test_undefined_dates_stay_in_the_table_with_a_reason():
    predictions = [[1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]
    labels = [[1, 2, 3, 4], [5, 5, 5, 5], [1, 2, 3, 4], [1, 2, 3, 4]]
    frame = _frame(predictions, labels)
    frame = frame.drop(index=[12, 13]).reset_index(drop=True)  # 2 names on the last date
    by_date = s.rank_ic_by_date(frame, _context(frame))
    assert len(by_date) == 4
    reasons = [value if isinstance(value, str) else None for value in by_date["reason"]]
    assert reasons == [
        "constant_prediction",
        "constant_target",
        None,
        "too_few_names",
    ]
    assert by_date["status"].tolist() == ["unavailable", "unavailable", "ok", "unavailable"]
    assert by_date["rank_ic"].isna().tolist() == [True, True, False, True]


def test_constant_forecasts_report_unavailable_ic_never_zero():
    rng = np.random.RandomState(2)
    frame = _frame(np.zeros((20, 4)), rng.randn(20, 4))
    score = s.score_model(
        "zero", "null", "forecast", frame, _context(frame), baselines={}, horizon=2
    )
    rows = [r for r in score.metrics if r["metric"] == "mean_rank_ic" and r["fold_id"] is None]
    assert {r["status"] for r in rows} == {"unavailable"}
    assert {r["reason"] for r in rows} == {"constant_prediction"}
    assert all(math.isnan(r["estimate"]) for r in rows)


# --- hit rate ---


def test_hit_rate_drops_zero_signs_and_reports_the_independence_benchmark():
    predictions = [[1, -1, 0, 2], [1, 1, -1, 3]]
    labels = [[0.5, 0.5, 0.2, 0.0], [-1, 1, -1, 2]]
    frame = _frame(predictions, labels)
    ctx = _context(frame)
    hit = s.hit_rate_summary(frame, np.repeat([0, 1], 4), ctx)
    # usable rows: (1,.5) hit, (-1,.5) miss, (1,-1) miss, (1,1) hit, (-1,-1) hit, (3,2) hit
    assert (hit["n_used"], hit["n_zero_prediction"], hit["n_zero_label"]) == (6, 1, 1)
    assert hit["hit_rate"] == pytest.approx(4 / 6)
    p, q = 4 / 6, 4 / 6
    assert hit["independence_expected"] == pytest.approx(p * q + (1 - p) * (1 - q))
    zero = s.hit_rate_summary(_frame(np.zeros((2, 4)), labels), np.repeat([0, 1], 4), ctx)
    assert (zero["status"], zero["reason"]) == ("unavailable", "no_signed_predictions")


# --- R^2 against an explicit baseline ---


def _pair(seed=0, n_dates=40):
    rng = np.random.RandomState(seed)
    labels = rng.randn(n_dates, 4)
    model = _frame(labels + 0.5 * rng.randn(n_dates, 4), labels)
    base = _frame(np.zeros((n_dates, 4)), labels)
    return model, base


def test_oos_r2_is_one_minus_pooled_sse_ratio_on_identical_rows():
    model, base = _pair()
    ctx = _context(model, block_length=5)
    r2, improvement, cumulative = s.oos_r2_vs(model, base, ctx)
    sse_model = ((model["target_return"] - model["prediction"]) ** 2).sum()
    sse_base = (base["target_return"] ** 2).sum()
    assert r2.estimate == pytest.approx(1 - sse_model / sse_base)
    assert cumulative[-1] == pytest.approx(sse_base - sse_model)
    assert improvement[0].estimate > 0
    perfect = model.assign(prediction=model["target_return"])
    flawless_baseline, *_ = s.oos_r2_vs(model, perfect, ctx)
    assert (flawless_baseline.status, flawless_baseline.reason) == (
        "unavailable",
        "zero_baseline_error",
    )
    assert s.oos_r2_vs(perfect, base, ctx)[0].estimate == pytest.approx(1.0)
    same, *_ = s.oos_r2_vs(model, model.copy(), ctx)
    assert same.estimate == pytest.approx(0.0)


def test_oos_r2_refuses_a_baseline_on_other_rows_or_labels():
    model, base = _pair()
    ctx = _context(model, block_length=5)
    with pytest.raises(ValueError, match="same rows"):
        s.oos_r2_vs(model, base.iloc[:-1], ctx)
    relabeled = base.assign(target_return=base["target_return"] + 1.0)
    with pytest.raises(ValueError, match="different labels"):
        s.oos_r2_vs(model, relabeled, ctx)


# --- calibration ---


def test_mincer_zarnowitz_recovers_a_planted_slope():
    rng = np.random.RandomState(3)
    forecast = rng.randn(400, 4)
    labels = 0.02 + 0.5 * forecast + 0.3 * rng.randn(400, 4)
    frame = _frame(forecast, labels, fold_of=np.arange(400) // 100)
    ctx = _context(frame, block_length=20, hac_lag=5)
    mz = s.mincer_zarnowitz(frame, np.repeat(np.arange(400), 4), ctx)
    assert mz["mz_slope"].estimate == pytest.approx(0.5, abs=0.05)
    assert mz["mz_intercept"].estimate == pytest.approx(0.02, abs=0.05)
    assert mz["mz_slope"].ci_low < 0.5 < mz["mz_slope"].ci_high
    assert mz["mz_slope"].null_value == 1.0
    assert mz["mz_within_date_slope"].estimate == pytest.approx(0.5, abs=0.05)


def test_forecasts_constant_within_each_fold_have_no_calibration_slope():
    rng = np.random.RandomState(4)
    fold_of = np.arange(40) // 10
    frame = _frame(np.repeat(fold_of[:, None] * 0.01, 4, axis=1), rng.randn(40, 4), fold_of=fold_of)
    mz = s.mincer_zarnowitz(frame, np.repeat(np.arange(40), 4), _context(frame))
    assert {r.reason for r in mz.values()} == {"constant_within_fold"}


# --- rank positions and buckets ---


def test_rank_positions_share_tied_groups_mean_label():
    frame = _frame([[0.1, 0.1, 0.3, 0.2]], [[1.0, 3.0, 10.0, 20.0]])
    table = s.rank_position_values(frame, _context(frame))
    # C (0.3) first, D (0.2) second, A and B tied for third and fourth
    assert table["position"].tolist() == [1, 2, 3, 4]
    assert table["realized"].tolist() == [10.0, 20.0, 2.0, 2.0]
    assert table["realized_minus_date_mean"].tolist() == pytest.approx([1.5, 11.5, -6.5, -6.5])


def test_a_constant_predictor_has_exactly_zero_spread_and_even_occupancy():
    rng = np.random.RandomState(5)
    frame = _frame(np.ones((8, 4)), rng.randn(8, 4))
    table = s.rank_position_values(frame, _context(frame))
    spread = table.groupby("date").apply(
        lambda g: (
            g.loc[g["position"] == 1, "realized"].iloc[0]
            - g.loc[g["position"] == 4, "realized"].iloc[0]
        )
    )
    assert (spread == 0.0).all()
    occupancy = s.rank_position_occupancy(frame)
    np.testing.assert_allclose(occupancy["share"], 0.25)
    assert occupancy.groupby("position")["share"].sum().tolist() == pytest.approx([1.0] * 4)


def test_rank_positions_follow_the_number_of_names_on_each_date():
    frame = _frame([[3, 2, 1, 0], [3, 2, 1, 0]], [[1, 2, 3, 4], [1, 2, 3, 4]])
    frame = frame.drop(index=7).reset_index(drop=True)  # 3 names on the second date
    table = s.rank_position_values(frame, _context(frame))
    assert table.groupby("date")["position"].max().tolist() == [4, 3]
    assert set(table["n_names"]) == {3, 4}


def test_pooled_deciles_are_exact_and_flag_degenerate_inputs():
    frame = _frame(np.arange(100.0).reshape(25, 4), np.zeros((25, 4)))
    bins, reason = s.pooled_decile_bins(frame, 10)
    assert reason is None
    assert np.bincount(bins)[1:].tolist() == [10] * 10
    constant, reason = s.pooled_decile_bins(frame.assign(prediction=1.0), 10)
    assert len(np.unique(constant)) == 1
    assert reason == "constant_within_date"
    two_values = frame.assign(prediction=np.tile([1.0, 2.0], 50))
    assert s.pooled_decile_bins(two_values, 10)[1] == "fewer_distinct_predictions_than_bins"


# --- whole-model scoring ---


def test_score_model_refuses_non_finite_inputs_rather_than_dropping_rows():
    model, _ = _pair()
    holed = model.copy()
    holed.loc[3, "prediction"] = np.nan
    with pytest.raises(ValueError, match="never dropped"):
        s.score_model("m", "control", "forecast", holed, _context(model), baselines={}, horizon=2)


def test_score_outputs_skip_scale_dependent_metrics_with_a_reason():
    model, base = _pair()
    ctx = _context(model, block_length=5)
    score = s.score_model("m", "control", "score", model, ctx, baselines={"zero": base}, horizon=2)
    reasons = {r["metric"]: r["reason"] for r in score.metrics if r["metric"] in s.SCALE_METRICS}
    assert set(reasons) == set(s.SCALE_METRICS)
    assert set(reasons.values()) == {"score_not_forecast"}


def test_metric_row_counts_cover_every_scored_row_and_date():
    model, base = _pair()
    ctx = _context(model, block_length=5)
    score = s.score_model(
        "m", "control", "forecast", model, ctx, baselines={"zero": base}, horizon=2
    )
    rmse = next(r for r in score.metrics if r["metric"] == "rmse" and r["fold_id"] is None)
    assert rmse["n"] == len(model)
    ic_curve = [r for r in score.curves if r["curve"] == "rank_ic"]
    assert len(ic_curve) == model["date"].nunique()


def test_bootstrap_resamples_whole_dates():
    # Every date's IC is exactly 1; any resample of whole dates keeps 1, so
    # the bootstrap has zero spread. A row-level resample would not.
    labels = np.random.RandomState(6).randn(30, 4)
    frame = _frame(labels, labels)
    ctx = _context(frame, block_length=5)
    series = s.ic_series(s.rank_ic_by_date(frame, ctx), ctx)
    assert ctx.draws.n == 30  # resampling units are dates, not the 120 rows
    boot = s.inf.bootstrap_mean(series, ctx.draws)
    assert boot.estimate == 1.0
    assert (boot.status, boot.reason) == ("unavailable", "zero_variance")
    with pytest.raises(ValueError, match="Expected 30 sessions"):
        s.inf.resampled_sums(frame["prediction"].to_numpy(), ctx.draws)


# --- reference distributions ---


def test_static_orderings_enumerate_every_fixed_ranking():
    rng = np.random.RandomState(7)
    labels = rng.randn(50, 4) + np.array([0.0, 0.0, 0.0, 1.0])  # D usually best
    frame = _frame(rng.randn(50, 4), labels)
    table, reason = s.static_ordering_ics(frame, _context(frame))
    assert reason is None and len(table) == 24
    best = table.loc[table["value"].idxmax(), "detail"]
    assert best.startswith("D>")
    pairs = dict(zip(table["detail"], table["value"], strict=True))
    assert pairs["A>B>C>D"] == pytest.approx(-pairs["D>C>B>A"])
    varying = frame.drop(index=0).reset_index(drop=True)
    assert s.static_ordering_ics(varying, _context(varying))[1] == "varying_cross_section"


def test_compare_to_null_reports_a_one_sided_empirical_p_value():
    result = s.compare_to_null(0.3, [0.1, 0.2, 0.3, 0.4], "block_permutation")
    assert result.estimate == pytest.approx(0.3 - 0.25)
    assert result.p_value == pytest.approx((1 + 2) / (1 + 4))
    assert s.compare_to_null(math.nan, [0.1], "x").reason == "observed_unavailable"


def test_cross_sectional_r2_values():
    rng = np.random.RandomState(8)
    labels = rng.randn(40, 4)
    ctx = _context(_frame(labels, labels), block_length=5)
    positions = np.repeat(np.arange(40), 4)
    perfect = _frame(labels + np.arange(40)[:, None], labels)  # right within date, wrong level
    assert s.cross_sectional_r2(perfect, positions, ctx).estimate == pytest.approx(1.0)
    flat = _frame(np.repeat(rng.randn(40, 1), 4, axis=1), labels)
    assert s.cross_sectional_r2(flat, positions, ctx).estimate == pytest.approx(0.0)


def test_top_minus_bottom_and_minimum_detectable_ic_in_the_scored_metrics():
    rng = np.random.RandomState(9)
    labels = rng.randn(60, 4)
    predictions = rng.randn(60, 4)
    frame = _frame(predictions, labels)
    ctx = _context(frame, block_length=5, hac_lag=3)
    score = s.score_model("m", "control", "score", frame, ctx, baselines={}, horizon=2)
    order = np.argsort(-predictions, axis=1)
    rows = np.arange(60)
    spread = labels[rows, order[:, 0]] - labels[rows, order[:, -1]]
    top = next(
        r
        for r in score.metrics
        if r["metric"] == "top_minus_bottom" and r["inference_method"] == "newey_west"
    )
    assert top["estimate"] == pytest.approx(spread.mean())
    fold = next(
        r
        for r in score.metrics
        if r["metric"] == "mean_rank_ic"
        and r["inference_method"] == "fold_block_t"
        and r["fold_id"] is None
    )
    mde = next(r for r in score.metrics if r["metric"] == "mean_rank_ic_minimum_detectable")
    df = fold["df"]  # the decision test's own critical values and power 0.8
    multiplier = s.inf.student_t_ppf(0.975, df) + s.inf.student_t_ppf(0.80, df)
    assert df == 3
    assert mde["inference_method"] == "fold_block_t"
    assert mde["estimate"] == pytest.approx(multiplier * fold["se"])
    assert multiplier == pytest.approx(3.182446 + 0.978472, rel=1e-5)


def test_positive_ic_probability_under_no_skill_is_exact():
    # 4 names: 2 of the 24 orderings give Spearman exactly 0.
    assert s.positive_ic_probability(4) == pytest.approx((1 - 2 / 24) / 2)
    assert s.positive_ic_probability(2) == pytest.approx(0.5)
    assert math.isnan(s.positive_ic_probability(12))


def test_deciles_of_a_forecast_constant_within_each_date_are_flagged():
    fold_of = np.arange(40) // 10
    frame = _frame(
        np.repeat(fold_of[:, None] * 0.01, 4, axis=1), np.zeros((40, 4)), fold_of=fold_of
    )
    _, reason = s.pooled_decile_bins(frame, 10)
    assert reason == "constant_within_date"


def test_constant_sign_predictions_get_a_hit_rate_warning():
    rng = np.random.RandomState(10)
    frame = _frame(np.abs(rng.randn(60, 4)) + 0.1, rng.randn(60, 4))
    ctx = _context(frame, block_length=5, hac_lag=3)
    score = s.score_model("m", "control", "score", frame, ctx, baselines={}, horizon=2)
    rows = [r for r in score.metrics if r["metric"] == "hit_rate" and r["fold_id"] is None]
    assert {(r["status"], r["reason"]) for r in rows} == {("warning", "constant_sign_prediction")}


def test_static_orderings_report_a_share_not_a_p_value():
    result = s.compare_to_null(0.3, [0.1, 0.2, 0.3, 0.4], "static_ordering", exhaustive=True)
    assert result.method == "null_enumeration"
    assert math.isnan(result.p_value)
    assert result.estimate == pytest.approx(0.05)


def test_curve_intervals_store_the_settings_their_inference_used():
    rng = np.random.RandomState(5)
    frame = _frame(rng.randn(40, 4), rng.randn(40, 4))
    ctx = _context(frame, ci_level=0.8, hac_lag=2)
    score = s.score_model("m", "control", "score", frame, ctx, baselines={}, horizon=2)
    with_interval = [
        r
        for r in score.curves
        if r["curve"]
        in (
            "rank_position_realized",
            "rank_position_realized_minus_date_mean",
            "pooled_decile_realized",
        )
    ]
    assert with_interval
    for row in with_interval:
        assert (row["ci_method"], row["ci_level"], row["hac_lag"]) == (s.CURVE_CI_METHOD, 0.8, 2)
        if not math.isnan(row["ci_low"]):  # the stored bounds are the nominal 80% interval
            assert row["ci_low"] < row["value"] < row["ci_high"]
    others = [r for r in score.curves if r not in with_interval]
    assert all(r["ci_method"] is None and math.isnan(r["ci_level"]) for r in others)
