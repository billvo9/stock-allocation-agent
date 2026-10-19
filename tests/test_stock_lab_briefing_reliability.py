"""
The briefing's reliability statistics: how often forecasts in a probability
range came true, with intervals in which forecast dates are the clusters
(and overlapping five-session totals keep their covariance), and
model-versus-B1 comparisons on exactly matched observations. Synthetic
numbers only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.health import stock_lab as health_stock_lab
from stock_agent.stock_lab.briefing import reliability
from stock_agent.stock_lab.briefing.reliability import (
    INSUFFICIENT,
    OUTCOME_MISMATCH,
    MixedTargets,
    Observation,
    calibration_buckets,
    calibration_group,
    cluster_ids,
    clustered_proportion,
    development_evidence,
    matched_comparison,
    observations,
    rolling,
    t_quantile,
)
from stock_agent.stock_lab.reversal import MIN_EFFECTIVE
from stock_agent.stock_lab.scores import jeffreys

DATES = [f"20{year}-{month:02d}-{day:02d}" for year in (26, 27) for month in range(1, 13)
         for day in range(1, 29)]  # fmt: skip


# --- the clustering unit ---


def test_dates_are_grouped_into_consecutive_blocks():
    dates = np.array(["d3", "d1", "d2", "d1", "d4", "d5"])
    assert cluster_ids(dates, 1).tolist() == [2, 0, 1, 0, 3, 4]
    assert cluster_ids(dates, 2).tolist() == [1, 0, 0, 0, 1, 2]
    with pytest.raises(ValueError):
        cluster_ids(dates, 0)


def test_the_t_quantile_matches_the_table():
    assert t_quantile(0.9, 1) == pytest.approx(6.314, abs=1e-3)
    assert t_quantile(0.9, 4) == pytest.approx(2.132, abs=1e-3)
    assert t_quantile(0.9, 30) == pytest.approx(1.697, abs=1e-3)


# --- a proportion with clustered observations ---


def test_a_worked_example_of_the_cluster_robust_count():
    # Dates a (2 hits of 3), b (0 of 2), c (1 of 1): p = 1/2, u = (0.5, -1, 0.5),
    # Var = 3/2 * 1.5 / 36 = 0.0625, so n_eff = 0.25 / 0.0625 = 4.
    result = clustered_proportion([1, 1, 0, 0, 0, 1], ["a", "a", "a", "b", "b", "c"])
    assert result.estimate == 0.5 and result.dates == 3
    assert result.n_eff == pytest.approx(4.0)
    assert (result.low, result.high) == pytest.approx(jeffreys(2, 4, 0.9))


def test_one_observation_per_date_is_the_plain_interval_less_one_date():
    rng = np.random.default_rng(1)
    hits = (rng.random(200) < 0.3).astype(float)
    result = clustered_proportion(hits, np.arange(200))
    assert result.n_eff == pytest.approx(199)  # the C / (C - 1) small-sample factor
    plain_low, plain_high = jeffreys(hits.sum(), 200, 0.9)
    assert result.low == pytest.approx(plain_low, abs=0.01)
    assert result.high == pytest.approx(plain_high, abs=0.01)


def test_rows_that_move_together_count_as_one_date():
    # Ten stocks on each of 40 dates share the day's outcome exactly: 400 rows,
    # but only 40 independent pieces of evidence.
    rng = np.random.default_rng(2)
    day_hits = (rng.random(40) < 0.6).astype(float)
    hits = np.repeat(day_hits, 10)
    result = clustered_proportion(hits, np.repeat(np.arange(40), 10))
    assert result.n == 400 and result.dates == 40
    assert result.n_eff == pytest.approx(39)
    by_dates = jeffreys(day_hits.sum(), 40, 0.9)
    by_rows = jeffreys(hits.sum(), 400, 0.9)
    width = result.high - result.low
    assert width == pytest.approx(by_dates[1] - by_dates[0], rel=0.1)
    assert width > 2.5 * (by_rows[1] - by_rows[0])


def test_no_hits_keeps_an_honest_upper_bound():
    result = clustered_proportion(np.zeros(120), np.repeat(np.arange(30), 4))
    assert result.estimate == 0.0 and result.low == 0.0
    assert result.n_eff == 30  # one observation per date, not 120
    assert result.high == pytest.approx(jeffreys(0, 30, 0.9)[1])


def test_a_zero_variance_falls_back_to_the_dates_not_the_rows():
    hits = np.tile([1.0, 0.0], 50)  # perfectly balanced inside every date
    result = clustered_proportion(hits, np.repeat(np.arange(50), 2))
    assert result.n_eff == 50


def test_the_clustered_interval_holds_its_coverage_where_counting_rows_does_not():
    # Each date's true probability is 0.40 or 0.90 (mean 0.65) and its ten stocks
    # share it, so stocks on one date are correlated (design effect about 3.5).
    rng = np.random.default_rng(20261018)
    reps, dates, stocks, truth = 400, 60, 10, 0.65
    clustered = naive = 0
    clusters = np.repeat(np.arange(dates), stocks)
    for _ in range(reps):
        daily = np.where(rng.random(dates) < 0.5, 0.40, 0.90)
        hits = (rng.random((dates, stocks)) < daily[:, None]).astype(float).ravel()
        result = clustered_proportion(hits, clusters)
        clustered += result.low <= truth <= result.high
        low, high = jeffreys(hits.sum(), hits.size, 0.9)
        naive += low <= truth <= high
    assert 0.85 <= clustered / reps <= 0.95
    assert naive / reps <= 0.75


def test_overlapping_five_session_totals_keep_their_coverage_with_the_lag_terms():
    # Five correlated stocks; each date's target is the total of the next five
    # sessions, so totals of dates up to four apart share sessions.
    rng = np.random.default_rng(20261019)
    reps, dates, stocks, rho, truth = 200, 250, 5, 0.5, 0.4
    threshold = 0.2533 * math.sqrt(5)  # P(N(0, 5) > threshold) = 0.40
    positions = np.repeat(np.arange(dates), stocks)
    covered = {0: 0, 4: 0}
    for _ in range(reps):
        common = rng.standard_normal(dates + 5)[:, None]
        returns = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal(
            (dates + 5, stocks)
        )
        level = np.vstack([np.zeros((1, stocks)), np.cumsum(returns, axis=0)])
        totals = level[6 : dates + 6] - level[1 : dates + 1]
        hits = (totals > threshold).astype(float).ravel()
        for lags in covered:
            result = clustered_proportion(hits, positions, lags=lags)
            covered[lags] += result.low <= truth <= result.high
    assert covered[4] / reps >= 0.85
    assert covered[0] / reps <= 0.70


def test_a_small_changing_panel_at_a_low_rate_keeps_its_coverage():
    # Two to five stocks per date, 25 dates, a true rate near 10%.
    rng = np.random.default_rng(7)
    shown = covered = 0
    for _ in range(600):
        daily = np.clip(0.1 + rng.normal(0, 0.03, 25), 0.01, 0.5)
        counts = rng.integers(2, 6, 25)
        hits = np.concatenate([(rng.random(k) < p).astype(float) for k, p in zip(counts, daily)])
        positions = np.repeat(np.arange(25), counts)
        result = clustered_proportion(hits, positions)
        if min(result.units, result.n_eff) >= 20:
            shown += 1
            covered += result.low <= 0.1 <= result.high
    assert shown > 300 and covered / shown >= 0.85


# --- calibration buckets ---


def _calibrated(rng, dates, stocks, probability):
    p = np.full(dates * stocks, probability)
    hits = (rng.random(p.size) < p).astype(float)
    return p, hits, np.repeat(np.array(DATES[:dates]), stocks)


def test_a_sixty_five_percent_forecast_lands_in_the_sixty_to_seventy_bucket():
    rng = np.random.default_rng(3)
    p, hits, dates = _calibrated(rng, 100, 4, 0.65)
    (bucket,) = calibration_buckets(p, hits, dates)
    assert (bucket.low, bucket.high) == pytest.approx((0.6, 0.7))
    assert bucket.n == 400 and bucket.dates == 100 and bucket.clusters == 100
    assert bucket.mean_predicted == pytest.approx(0.65)
    assert bucket.observed == pytest.approx(hits.mean())
    assert bucket.ci_low <= bucket.observed <= bucket.ci_high
    assert bucket.status == "ok"


def test_buckets_with_too_few_dates_say_insufficient_and_show_no_frequency():
    rng = np.random.default_rng(4)
    p, hits, dates = _calibrated(rng, MIN_EFFECTIVE - 1, 5, 0.35)
    (bucket,) = calibration_buckets(p, hits, dates)
    assert bucket.status == INSUFFICIENT
    assert bucket.n == 5 * (MIN_EFFECTIVE - 1) and bucket.dates == MIN_EFFECTIVE - 1
    assert bucket.observed is None and bucket.ci_low is None and bucket.ci_high is None


def test_a_probability_of_one_falls_in_the_last_bucket_and_empty_buckets_are_omitted():
    p = np.array([0.05, 1.0, 0.95])
    hits = np.array([0.0, 1.0, 1.0])
    buckets = calibration_buckets(p, hits, np.array(["a", "b", "c"]))
    assert [(round(b.low, 1), round(b.high, 1), b.n) for b in buckets] == [
        (0.0, 0.1, 1),
        (0.9, 1.0, 2),
    ]


def test_overlapping_totals_count_one_unit_per_five_dates():
    rng = np.random.default_rng(5)
    p, hits, dates = _calibrated(rng, 100, 2, 0.45)
    (bucket,) = calibration_buckets(p, hits, dates, lags=4)
    assert bucket.dates == 100 and bucket.clusters == 20


def test_a_well_calibrated_forecaster_looks_calibrated():
    rng = np.random.default_rng(6)
    probability = rng.uniform(0.05, 0.95, size=(250, 4))
    hits = (rng.random(probability.shape) < probability).astype(float)
    dates = np.repeat(np.array(DATES[:250]), 4)
    buckets = calibration_buckets(probability.ravel(), hits.ravel(), dates)
    covered = [
        b.ci_low <= b.mean_predicted <= b.ci_high for b in buckets if b.status != INSUFFICIENT
    ]
    assert len(covered) >= 8 and sum(covered) >= len(covered) - 1


# --- observations from stored records ---


DISTINCT = [(0.1, 0.2, 0.7), (0.2, 0.2, 0.6), (0.3, 0.2, 0.5), (0.4, 0.2, 0.4), (0.5, 0.2, 0.3)]


def _forecast(fid, symbol, as_of, *, estimator="B1", band=0.005, p=None, version="m1"):
    rows = DISTINCT if p is None else [p] * 5
    return {
        "forecast_id": fid,
        "symbol": symbol,
        "as_of_session": as_of,
        "estimator": estimator,
        "model_version": version,
        "band": band,
        "cumulative_band": band * math.sqrt(5),
        "refusal": None,
        "daily": [
            {"horizon": h, "session": f"s{h}", "p_decrease": d, "p_stable": s, "p_increase": i}
            for h, (d, s, i) in enumerate(rows, start=1)
        ],
        "cumulative": {"p_decrease": 0.35, "p_stable": 0.2, "p_increase": 0.45},
    }


def _score(fid, target, outcome, *, status="scored"):
    return {
        "forecast_id": fid,
        "target": target,
        "session": f"s{target}",
        "status": status,
        "outcome": outcome if status == "scored" else None,
        "rps": 0.2 if status == "scored" else None,
        "brier": 0.6 if status == "scored" else None,
        "log_loss": 1.0 if status == "scored" else None,
    }


def test_observations_join_each_score_to_its_own_horizon():
    forecasts = [_forecast("f1", "AAA", "2026-01-02")]
    scores = [
        _score("f1", "1", 2),
        _score("f1", "3", 0),
        _score("f1", "cum", 0),
        _score("f1", "2", None, status="unscorable"),
    ]
    found = {obs.target: obs for obs in observations(forecasts, scores)}
    assert set(found) == {"1", "3", "cum"}  # an unscorable target is no observation
    assert (found["1"].p_increase, found["3"].p_increase) == (0.7, 0.5)  # daily[0], daily[2]
    assert found["1"].band == 0.005
    assert found["cum"].p_increase == 0.45 and found["cum"].band == pytest.approx(0.005 * 5**0.5)


def test_superseded_and_refused_forecasts_give_no_observations():
    refused = {**_forecast("f2", "AAA", "2026-01-05"), "refusal": {"reason": "x"}, "daily": []}
    forecasts = [_forecast("f1", "AAA", "2026-01-02"), refused]
    scores = [_score("f1", "1", 2), _score("f2", "1", 1)]
    assert observations(forecasts, scores, superseded={"f1"}) == []


def test_a_group_never_mixes_horizons_bands_estimators_or_model_versions():
    base = observations([_forecast("f1", "AAA", "2026-01-02")], [_score("f1", "1", 2)])[0]
    for change in ({"target": "2"}, {"band": 0.01}, {"estimator": "B3"}, {"model_version": "m2"}):
        with pytest.raises(MixedTargets):
            calibration_group([base, Observation(**{**base.__dict__, **change})], "increase")


def test_a_group_counts_observations_and_distinct_forecast_dates():
    forecasts = [
        _forecast(f"f{i}{s}", s, DATES[i], p=(0.2, 0.15, 0.65))
        for i in range(30)
        for s in ("AAA", "BBB", "CCC")
    ]
    scores = [_score(f["forecast_id"], "1", 2 if k % 3 else 1) for k, f in enumerate(forecasts)]
    group = calibration_group(observations(forecasts, scores), "increase")
    assert (group.target, group.band, group.estimator, group.model_version) == (
        "1",
        0.005,
        "B1",
        "m1",
    )
    assert group.n == 90 and group.dates == 30
    (bucket,) = group.buckets
    assert bucket.mean_predicted == pytest.approx(0.65) and bucket.dates == 30


def test_the_rolling_summary_shows_means_only_from_enough_data_and_keeps_its_window():
    forecasts = [
        _forecast(f"f{i}{s}", s, DATES[i], p=(0.3, 0.3, 0.4)) for i in range(70) for s in "ABCD"
    ]
    found = observations(forecasts, [_score(f["forecast_id"], "1", 2) for f in forecasts])
    summary = rolling(found)
    assert summary.status == "ok" and summary.dates == 63 and summary.n == 252
    assert (summary.first, summary.last) == (DATES[7], DATES[69])
    assert summary.means == pytest.approx({"rps": 0.2, "brier": 0.6, "log_loss": 1.0})
    short = rolling([o for o in found if o.as_of_session <= DATES[40]])
    assert short.status == INSUFFICIENT and short.means is None


# --- matched comparisons ---


def _paired(rng, dates, stocks, edge, noise=0.0):
    candidate, baseline = [], []
    for i in range(dates):
        shock = rng.normal(0, 0.05)
        for s in range(stocks):
            base_rps = 0.2 + shock + rng.normal(0, 0.02)
            common = {
                "symbol": f"S{s}",
                "as_of_session": DATES[i],
                "target": "1",
                "band": 0.005,
                "model_version": "m",
                "p_decrease": 0.3,
                "p_stable": 0.3,
                "p_increase": 0.4,
                "outcome": 2,
                "session": DATES[i],
                "brier": 0.6,
                "log_loss": 1.0,
            }
            baseline.append(Observation(**common, estimator="B1", rps=base_rps))
            candidate.append(
                Observation(**common, estimator="B3", rps=base_rps - edge + rng.normal(0, noise))
            )
    return candidate, baseline


def test_identical_forecasts_differ_by_exactly_zero():
    rng = np.random.default_rng(7)
    _, baseline = _paired(rng, 120, 3, 0.0)
    twin = [Observation(**{**o.__dict__, "estimator": "B3"}) for o in baseline]
    result = matched_comparison(twin, baseline)
    rps = next(d for d in result.differences if d.score == "rps")
    assert rps.difference == pytest.approx(0.0, abs=1e-12) and rps.se == pytest.approx(0.0)


def test_the_difference_is_the_mean_of_per_date_differences_on_matched_rows_only():
    rng = np.random.default_rng(8)
    candidate, baseline = _paired(rng, 120, 3, 0.01, noise=0.01)
    extra = Observation(**{**candidate[0].__dict__, "symbol": "ONLY_CANDIDATE"})
    result = matched_comparison([*candidate, extra], baseline[:-1])
    assert result.n == 120 * 3 - 1
    assert (result.unmatched_candidate, result.unmatched_baseline) == (2, 0)
    assert (result.dates, result.blocks, result.status) == (120, 6, "ok")
    rps = next(d for d in result.differences if d.score == "rps")
    assert rps.difference == pytest.approx(-0.01, abs=0.004)
    assert rps.candidate - rps.baseline == pytest.approx(rps.difference)
    assert rps.low < rps.difference < rps.high
    assert rps.high - rps.difference == pytest.approx(t_quantile(0.9, 5) * rps.se)


def test_a_comparison_refuses_to_mix_definitions_or_to_compare_without_b1():
    rng = np.random.default_rng(9)
    candidate, baseline = _paired(rng, 100, 2, 0.0)
    other = [Observation(**{**o.__dict__, "band": 0.01}) for o in baseline]
    with pytest.raises(MixedTargets):
        matched_comparison(candidate, other)
    mixed = [*candidate[:-1], Observation(**{**candidate[-1].__dict__, "target": "2"})]
    with pytest.raises(MixedTargets):
        matched_comparison(mixed, baseline)
    not_b1 = [Observation(**{**o.__dict__, "estimator": "B0"}) for o in baseline]
    with pytest.raises(MixedTargets):
        matched_comparison(candidate, not_b1)
    with pytest.raises(ValueError, match="two candidate"):
        matched_comparison([*candidate, candidate[0]], baseline)


def test_a_small_comparison_says_insufficient_and_gives_no_difference():
    rng = np.random.default_rng(10)
    dates = reliability.COMPARISON_BLOCK * reliability.MIN_COMPARISON_BLOCKS - 1
    candidate, baseline = _paired(rng, dates, 4, 0.01)
    result = matched_comparison(candidate, baseline)
    assert result.status == INSUFFICIENT
    assert all(d.difference is None and d.low is None for d in result.differences)


def test_matched_rows_that_differ_in_outcome_are_reported_not_compared():
    rng = np.random.default_rng(11)
    candidate, baseline = _paired(rng, 100, 1, 0.0)
    broken = [*baseline[:-1], Observation(**{**baseline[-1].__dict__, "outcome": 0})]
    result = matched_comparison(candidate, broken)
    assert result.status == OUTCOME_MISMATCH
    assert all(d.difference is None for d in result.differences)


def test_no_matched_rows_is_reported_not_guessed():
    rng = np.random.default_rng(12)
    candidate, baseline = _paired(rng, 100, 1, 0.0)
    moved = [Observation(**{**o.__dict__, "symbol": "ELSEWHERE"}) for o in baseline]
    result = matched_comparison(candidate, moved)
    assert result.status == "none" and result.n == 0


# --- the development record, read as stored ---


def _evaluation():
    metrics = [
        {"model": m, "target": "target_1", "n": 2301, "dates": 767, "rps": r, "brier": b,
         "log_loss": ll}
        for m, r, b, ll in (("B1", 0.2067, 0.6298, 1.0465), ("B3", 0.2058, 0.6250, 1.0389))
    ]  # fmt: skip
    reliability_rows = [
        {"model": "B3", "target": "target_1", "category": 2, "bin_low": 0.3, "bin_high": 0.4,
         "n": 900, "dates": 700, "mean_predicted": 0.36, "observed": 0.37, "ci_low": 0.34,
         "ci_high": 0.40},
        {"model": "B3", "target": "target_1", "category": 2, "bin_low": 0.4, "bin_high": 0.5,
         "n": 15, "mean_predicted": 0.41, "observed": 0.27, "ci_low": 0.12, "ci_high": 0.48},
        {"model": "B3", "target": "target_1", "category": 0, "bin_low": 0.3, "bin_high": 0.4,
         "n": 1, "dates": 1, "mean_predicted": 0.3, "observed": 0.3, "ci_low": 0.1, "ci_high": 0.5},
    ]  # fmt: skip
    comparisons = [
        {"model": "B3", "target": "target_1", "vs": "B1", "mean_diff": -0.0009, "dates": 767,
         "p_one_sided": 0.03, "fold_t_p_one_sided": 0.12, "nw_se": 0.0004},
    ]  # fmt: skip
    return {
        "run_id": "S-test",
        "data_class": "synthetic",
        "created_at": "2024-12-17T21:01:00.000000Z",
        "stocks": ["DEMO_A", "DEMO_B"],
        "config": {"outcome_end": "2024-12-17"},
        "metrics": metrics,
        "reliability": reliability_rows,
        "comparisons": comparisons,
        "selection": {"selected": "B1", "rule": "keep B1 unless ..."},
    }


def test_development_evidence_is_read_from_the_stored_record_without_recomputation():
    found = development_evidence(_evaluation(), "B3")
    assert (found.run_id, found.data_class, found.outcome_end) == (
        "S-test",
        "synthetic",
        "2024-12-17",
    )
    assert found.candidate["rps"] == 0.2058 and found.baseline["rps"] == 0.2067
    assert (found.n, found.dates) == (2301, 767)
    assert found.reported == {
        "rps_difference": -0.0009,
        "p_newey_west": 0.03,
        "p_fold_block_t": 0.12,
    }
    stored, undated = found.buckets  # the Increase category only
    assert (stored.low, stored.high, stored.observed, stored.dates) == (0.3, 0.4, 0.37, 700)
    assert stored.status == "ok"
    assert undated.dates is None and undated.status == INSUFFICIENT  # no date count: not trusted


def test_for_b1_itself_there_is_no_comparison():
    found = development_evidence(_evaluation(), "B1")
    assert found.baseline is None and found.reported is None


def test_the_thresholds_match_model_health():
    assert reliability.MIN_SCORED == health_stock_lab.MIN_SCORED
    assert reliability.MIN_SESSIONS == health_stock_lab.MIN_SESSIONS
    assert reliability.MIN_CLUSTERS == MIN_EFFECTIVE
