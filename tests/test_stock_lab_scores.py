"""Proper scores, intervals and calibration (stock_agent.stock_lab.scores), by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.stock_lab import scores


def test_scores_by_hand():
    p = np.array([[0.2, 0.3, 0.5]])
    up, down = np.array([2]), np.array([0])
    assert scores.brier(p, up)[0] == pytest.approx(0.04 + 0.09 + 0.25)
    assert scores.log_loss(p, up)[0] == pytest.approx(-math.log(0.5))
    # RPS: cumulative (0.2, 0.5) vs observed (0, 0) for Increase; (1, 1) for Decrease.
    assert scores.rps(p, up)[0] == pytest.approx((0.04 + 0.25) / 2)
    assert scores.rps(p, down)[0] == pytest.approx((0.64 + 0.25) / 2)
    assert scores.moving_brier(p, up)[0] == pytest.approx((0.7 - 1) ** 2)
    assert scores.direction_brier(p, up)[0] == pytest.approx((0.5 / 0.7 - 1) ** 2)
    assert math.isnan(scores.direction_brier(p, np.array([1]))[0])


def test_rps_charges_more_for_the_far_miss():
    sure_down = np.array([[1.0, 0.0, 0.0]])
    sure_stable = np.array([[0.0, 1.0, 0.0]])
    up = np.array([2])
    assert scores.rps(sure_down, up)[0] > scores.rps(sure_stable, up)[0]
    assert scores.brier(sure_down, up)[0] == scores.brier(sure_stable, up)[0]


@pytest.mark.parametrize("name", list(scores.SCORES))
def test_every_score_is_proper_the_truth_scores_best(name):
    rng = np.random.RandomState(0)
    truth = np.array([0.2, 0.5, 0.3])
    y = rng.choice(3, size=200_000, p=truth)
    function = scores.SCORES[name]
    best = np.nanmean(function(np.tile(truth, (len(y), 1)), y))
    for other in ([0.33, 0.34, 0.33], [0.1, 0.6, 0.3], [0.3, 0.4, 0.3]):
        worse = np.nanmean(function(np.tile(other, (len(y), 1)), y))
        assert best <= worse + 1e-4, (name, other)


def test_the_jeffreys_interval_matches_beta_quantiles():
    stats = pytest.importorskip("scipy.stats")
    for x, n in ((3, 10), (37, 120), (0, 25), (25, 25), (12.4, 31.7)):
        low, high = scores.jeffreys(x, n, 0.9)
        if x > 0:
            assert low == pytest.approx(stats.beta.ppf(0.05, x + 0.5, n - x + 0.5), abs=1e-8)
        else:
            assert low == 0.0
        if x < n:
            assert high == pytest.approx(stats.beta.ppf(0.95, x + 0.5, n - x + 0.5), abs=1e-8)
        else:
            assert high == 1.0
    assert all(math.isnan(v) for v in scores.jeffreys(0, 0))


def test_a_calibrated_forecaster_has_slope_one_and_an_overconfident_one_less():
    rng = np.random.RandomState(1)
    p = rng.uniform(0.05, 0.95, size=50_000)
    hit = (rng.uniform(size=len(p)) < p).astype(float)
    intercept, slope = scores.calibration_line(p, hit)
    assert abs(intercept) < 0.05 and slope == pytest.approx(1.0, abs=0.05)
    extreme = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))  # twice the log-odds
    assert scores.calibration_line(extreme, hit)[1] == pytest.approx(0.5, abs=0.05)


def test_reliability_bins_count_every_forecast_once_per_category():
    rng = np.random.RandomState(2)
    p = rng.dirichlet([2, 2, 2], size=1000)
    y = rng.choice(3, size=1000)
    rows = scores.reliability(p, y, bins=10)
    for c in range(3):
        assert sum(r["n"] for r in rows if r["category"] == c) == 1000
    assert all(r["ci_low"] <= r["observed"] <= r["ci_high"] for r in rows)


def test_holm_is_step_down_and_monotone():
    adjusted = scores.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": math.nan})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
    assert math.isnan(adjusted["d"])


def test_the_one_se_rule_uses_paired_errors_and_prefers_the_simplest():
    import pandas as pd

    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.RandomState(0)
    common = rng.normal(0.2, 0.05, 400)  # a shock shared by every candidate on a date
    best = common + rng.normal(0, 0.001, 400)
    per_date = {
        # 0.004 worse every date: clearly worse once the common shock is paired away.
        "simple": pd.Series(best + 0.004 + rng.normal(0, 0.001, 400), index=dates),
        # Exactly as good as the best on average: within one paired SE.
        "middle": pd.Series(best + rng.normal(0, 0.0005, 400), index=dates),
        "complex": pd.Series(best, index=dates),
    }
    chosen, rows = scores.one_se_choice(per_date, ["simple", "middle", "complex"])
    # Unpaired SEs (about 0.0025) would make "simple" eligible; paired ones do not.
    assert chosen == "middle"
    by_name = {row["candidate"]: row for row in rows}
    assert by_name["simple"]["paired_se"] < 0.001 < by_name["simple"]["diff_vs_best"] * 4
