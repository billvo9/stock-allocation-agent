"""
Dependence-aware inference: exact hand examples, the Driscoll-Kraay
special cases, the circular block layout, and seeded calibration under no
skill. Oracles are written independently of the module's helpers.

Calibration bounds come from a 20-seed offline sweep (R = 400 series each):
MA(19) at T = 1000 rejected at a nominal 5% with
  iid SE 0.60-0.69, Newey-West lag 40 0.065-0.12, lag 20 0.09-0.15,
  fold-block t 0.035-0.085, circular block bootstrap 0.04-0.09;
the 4-name momentum IC process at 756 dates with
  iid SE 0.47-0.58, Newey-West 0.06-0.11, fold-block t 0.04-0.083,
  bootstrap 0.043-0.08.
Each bound below sits outside that range by at least ~1.5 Monte Carlo SEs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.model_diagnostics import inference as inf


def _naive_lrv(x, lag):
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    total = sum(d[t] * d[t] for t in range(n)) / n
    for k in range(1, lag + 1):
        gamma = sum(d[t] * d[t - k] for t in range(k, n)) / n
        total += 2 * (1 - k / (lag + 1)) * gamma
    return total


# --- Newey-West ---


def test_newey_west_hand_example_with_bartlett_weights_and_divisor_n():
    # x = 1..5: gamma0 = 2, gamma1 = 0.8, LRV = 2 + 2 * (1 - 1/2) * 0.8 = 2.8
    result = inf.newey_west_mean([1, 2, 3, 4, 5], lag=1)
    assert result.se**2 == pytest.approx(0.56)
    assert inf.long_run_variance([1, 2, 3, 4, 5], lag=1) == pytest.approx(2.8)
    # x = [1, -1, 2, 0, 3]: gamma0 = 2, gamma1 = -1, LRV = 1, SE = sqrt(1/5)
    assert inf.newey_west_mean([1, -1, 2, 0, 3], lag=1).se == pytest.approx(0.4472135955)
    np.testing.assert_allclose(inf.bartlett_weights(3), [0.75, 0.5, 0.25])


def test_newey_west_matches_a_naive_loop_and_lag_zero_is_the_iid_divisor_n_se():
    x = np.random.RandomState(4).randn(60).cumsum()
    for lag in (0, 3, 10):
        assert inf.long_run_variance(x, lag=lag) == pytest.approx(_naive_lrv(x, lag))
    assert inf.newey_west_mean(x, lag=0).se == pytest.approx(x.std(ddof=0) / math.sqrt(len(x)))


def test_bartlett_long_run_variance_is_non_negative_for_an_alternating_series():
    x = np.tile([1.0, -1.0], 50)
    assert inf.long_run_variance(x, lag=10) >= 0.0


def test_default_lag_is_the_research_minimum_and_is_recorded():
    assert inf.DEFAULT_HAC_LAG == 40
    result = inf.newey_west_mean(np.random.RandomState(0).randn(200))
    assert result.params["lag"] == 40
    assert result.params["kernel"] == "bartlett"


def test_gaps_keep_their_place_on_the_calendar():
    # Pairs with a missing side drop out; lags count sessions, not values.
    x = np.array([1.0, np.nan, 3.0, 2.0, np.nan, 5.0, 4.0])
    present = x[np.isfinite(x)]
    d = np.where(np.isfinite(x), x - present.mean(), 0.0)
    n = len(present)
    expected = d @ d / n + 2 * 0.5 * (d[1:] @ d[:-1]) / n
    assert inf.long_run_variance(x, lag=1) == pytest.approx(expected)
    closed_up = inf.long_run_variance(present, lag=1)
    assert closed_up != pytest.approx(expected)


def test_short_or_constant_series_are_unavailable_not_zero():
    assert inf.newey_west_mean([1.0, 2.0], lag=5).reason == "series_shorter_than_lag"
    constant = inf.newey_west_mean(np.ones(100), lag=5)
    assert (constant.status, constant.reason) == ("unavailable", "zero_variance")
    assert math.isnan(constant.se)
    with pytest.raises(ValueError, match="more than lag"):
        inf.long_run_variance([1.0, 2.0, 3.0], lag=3)
    with pytest.raises(TypeError):
        inf.newey_west_mean([1.0, 2.0, 3.0], lag=1.5)


def test_effective_sample_size_is_n_gamma0_over_lrv():
    # gamma0 = 2, LRV = 2.8 -> n_eff = 5 * 2 / 2.8
    assert inf.effective_sample_size([1, 2, 3, 4, 5], lag=1) == pytest.approx(5 * 2 / 2.8)
    assert math.isnan(inf.effective_sample_size(np.ones(50), lag=3))


# --- distributions ---


@pytest.mark.parametrize(
    ("df", "quantile"), [(1, 12.7062047362), (10, 2.2281388520), (23, 2.0686576104)]
)
def test_student_t_quantiles_match_tables(df, quantile):
    assert inf.student_t_ppf(0.975, df) == pytest.approx(quantile, abs=1e-8)
    assert inf.student_t_two_sided_p(quantile, df) == pytest.approx(0.05, abs=1e-10)


def test_normal_quantile_and_p_value():
    assert inf.normal_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert inf.normal_two_sided_p(1.959963985) == pytest.approx(0.05, abs=1e-9)


# --- fold-block t ---


def test_fold_block_t_uses_block_means_and_t_critical_values():
    values = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 7.0, np.nan, 0.0])
    blocks = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    result = inf.fold_block_t(values, blocks)
    block_means = np.array([2.0, 2.0, 6.0, 0.0])
    se = block_means.std(ddof=1) / 2.0
    assert result.estimate == pytest.approx(block_means.mean())
    assert result.se == pytest.approx(se)
    assert result.params["df"] == 3
    half_width = inf.student_t_ppf(0.975, 3) * se
    assert result.ci_high - result.estimate == pytest.approx(half_width)


def test_fold_block_t_needs_three_blocks():
    result = inf.fold_block_t([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    assert (result.status, result.reason) == ("unavailable", "too_few_blocks")


# --- circular block bootstrap ---


def test_circular_blocks_wrap_past_the_last_session():
    draws = inf.draw_circular_blocks(10, block_length=3, reps=5, seed=0)
    assert draws.blocks_per_series == 4
    np.testing.assert_array_equal(draws.lengths, [3, 3, 3, 1])
    fixed = inf.BlockDraws(10, 3, 1, 0, np.array([[8, 0, 4, 9]]), np.array([3, 3, 3, 1]))
    np.testing.assert_array_equal(fixed.indices()[0], [8, 9, 0, 0, 1, 2, 4, 5, 6, 9])


def test_resampled_sums_equal_sums_over_materialized_indices():
    values = np.random.RandomState(1).randn(50)
    values[[3, 17]] = np.nan
    draws = inf.draw_circular_blocks(50, block_length=7, reps=30, seed=3)
    expected = np.nansum(values[draws.indices()], axis=1)
    np.testing.assert_allclose(inf.resampled_sums(values, draws), expected)


def test_bootstrap_is_reproducible_from_its_recorded_seed():
    x = np.random.RandomState(2).randn(300)
    first = inf.bootstrap_mean(x, inf.draw_circular_blocks(300, block_length=20, reps=99, seed=7))
    again = inf.bootstrap_mean(x, inf.draw_circular_blocks(300, block_length=20, reps=99, seed=7))
    other = inf.bootstrap_mean(x, inf.draw_circular_blocks(300, block_length=20, reps=99, seed=8))
    assert first.se == again.se
    assert first.se != other.se
    assert (first.params["seed"], first.params["reps"], first.params["block_length"]) == (7, 99, 20)


def test_bootstrap_needs_two_blocks_of_sessions():
    draws = inf.draw_circular_blocks(100, block_length=63, reps=9, seed=0)
    result = inf.bootstrap_mean(np.arange(100.0), draws)
    assert (result.status, result.reason) == ("unavailable", "series_shorter_than_two_blocks")


def test_bootstrap_statistic_resamples_numerator_and_denominator_together():
    rng = np.random.RandomState(5)
    model = rng.rand(200)
    base = model + rng.rand(200)
    draws = inf.draw_circular_blocks(200, block_length=20, reps=199, seed=1)
    result = inf.bootstrap_statistic(
        {"sse_model": model, "sse_baseline": base},
        lambda s: 1 - s["sse_model"] / s["sse_baseline"],
        draws,
        n=200,
    )
    assert result.estimate == pytest.approx(1 - model.sum() / base.sum())
    assert result.ci_low < result.estimate < result.ci_high
    assert result.params["ci_method"] == "percentile_uncalibrated"


# --- Driscoll-Kraay ---


def _design(n, rng):
    return np.column_stack([np.ones(n), rng.randn(n)])


def test_driscoll_kraay_lag_zero_one_row_per_date_is_hc0():
    rng = np.random.RandomState(0)
    X = _design(40, rng)
    y = X @ [0.1, 0.5] + rng.randn(40)
    result = inf.driscoll_kraay_ols(y, X, np.arange(40), n_sessions=40, lag=0)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    hc0 = bread @ (X.T * e**2) @ X @ bread
    np.testing.assert_allclose(result.coefficients, beta)
    np.testing.assert_allclose(result.covariance, hc0)


def test_driscoll_kraay_lag_zero_several_rows_per_date_is_date_clustered():
    rng = np.random.RandomState(1)
    dates = np.repeat(np.arange(30), 4)
    X = _design(120, rng)
    y = X @ [0.0, 1.0] + rng.randn(120)
    result = inf.driscoll_kraay_ols(y, X, dates, n_sessions=30, lag=0)
    e = y - X @ result.coefficients
    meat = np.zeros((2, 2))
    for date in range(30):
        score = (X[dates == date] * e[dates == date, None]).sum(axis=0)
        meat += np.outer(score, score)
    bread = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(result.covariance, bread @ meat @ bread)


def test_driscoll_kraay_exceeds_ols_when_names_share_a_date_shock():
    rng = np.random.RandomState(2)
    n_dates, names = 300, 4
    dates = np.repeat(np.arange(n_dates), names)
    # Both the regressor and the error are mostly common to all names on a
    # date, so the names add almost no independent information.
    x = np.repeat(rng.randn(n_dates), names) + 0.1 * rng.randn(n_dates * names)
    y = np.repeat(rng.randn(n_dates), names) + 0.1 * rng.randn(n_dates * names)
    X = np.column_stack([np.ones(len(x)), x])
    result = inf.driscoll_kraay_ols(y, X, dates, n_sessions=n_dates, lag=0)
    e = y - X @ result.coefficients
    ols = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - 2)
    for index in (0, 1):
        ratio = math.sqrt(result.covariance[index, index] / ols[index, index])
        assert ratio > 0.8 * math.sqrt(names)


def test_driscoll_kraay_reports_a_singular_design():
    X = np.column_stack([np.ones(20), np.zeros(20)])
    result = inf.driscoll_kraay_ols(np.arange(20.0), X, np.arange(20), n_sessions=20, lag=0)
    assert (result.status, result.reason) == ("unavailable", "singular_design")


# --- calibration under no skill (seeded) ---


def _ma19(seed, T=1000, R=400):
    shocks = np.random.RandomState(seed).randn(R, T + 19)
    cumulative = np.concatenate([np.zeros((R, 1)), np.cumsum(shocks, axis=1)], axis=1)
    return (cumulative[:, 20:] - cumulative[:, :-20]) / math.sqrt(20)


def test_simulated_rejection_rates_on_overlapping_label_noise():
    series = _ma19(0)
    rates = inf.null_rejection_rates(series, lag=40, seed=100)
    lag20 = inf.null_rejection_rates(series, lag=20, seed=100)
    assert 0.55 <= rates["iid"] <= 0.75  # naive SE: badly over-rejects (theory ~0.66)
    assert 0.04 <= rates["newey_west"] <= 0.14  # lag 40: mild, recorded over-rejection
    assert lag20["newey_west"] > rates["newey_west"]
    assert 0.02 <= rates["fold_block_t"] <= 0.10
    assert 0.02 <= rates["circular_block_bootstrap"] <= 0.11


def test_simulated_rejection_rates_on_four_name_momentum_ic():
    series = inf.simulate_null_ic_series(n_dates=756, signal="momentum", reps=400, seed=0)
    assert series.shape == (400, 756)
    assert set(np.round(np.unique(series), 6)) <= set(np.round(np.arange(-1, 1.01, 0.2), 6))
    rates = inf.null_rejection_rates(series, lag=40, seed=100)
    assert rates["iid"] > 0.40
    assert 0.04 <= rates["newey_west"] <= 0.14
    assert 0.02 <= rates["fold_block_t"] <= 0.10
    assert 0.02 <= rates["circular_block_bootstrap"] <= 0.11


def test_bartlett_lag_40_recovers_most_of_the_ma19_long_run_variance():
    series = _ma19(1, T=1500, R=800)
    centered = series - series.mean(axis=1, keepdims=True)
    lrvs = [inf.long_run_variance(row, lag=40) for row in series[:200]]
    assert 14.0 <= np.mean(lrvs) <= 18.0  # true 20; Bartlett at lag 40 is biased down
    assert (centered**2).mean() == pytest.approx(1.0, abs=0.1)


def test_vectorized_rejection_rates_equal_the_estimators_row_by_row():
    # Enough rows that each method rejects some and accepts others, so an
    # exact match is informative.
    series = _ma19(3, T=252, R=200)
    blocks = np.arange(252) // 63
    draws = inf.draw_circular_blocks(252, block_length=63, reps=199, seed=9)
    z = inf.normal_ppf(0.975)
    t_blocks = inf.student_t_ppf(0.975, 3)
    t_boot = inf.student_t_ppf(0.975, draws.blocks_per_series - 1)
    expected = {"newey_west": [], "fold_block_t": [], "circular_block_bootstrap": []}
    for row in series:
        hac = inf.newey_west_mean(row, lag=40)
        block = inf.fold_block_t(row, blocks)
        boot = inf.bootstrap_mean(row, draws)
        expected["newey_west"].append(abs(hac.estimate) / hac.se > z)
        expected["fold_block_t"].append(abs(block.estimate) / block.se > t_blocks)
        expected["circular_block_bootstrap"].append(abs(boot.estimate) / boot.se > t_boot)
    rates = inf.null_rejection_rates(series, lag=40, seed=9, bootstrap_reps=199)
    for method, decisions in expected.items():
        assert 0 < sum(decisions) < len(decisions), method
        assert rates[method] == pytest.approx(np.mean(decisions)), method
