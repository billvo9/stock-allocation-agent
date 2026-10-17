"""
The candidate decision tests for owner decision A7
(stock_agent.stock_lab.decision_tests): each procedure against the shared
estimators or an independent computation, shift invariance, and the fixed-b
critical values the packet records.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from stock_agent.model_diagnostics.inference import (
    bootstrap_mean,
    draw_circular_blocks,
    fold_block_t,
    long_run_variance,
    newey_west_mean,
    normal_ppf,
    student_t_ppf,
)
from stock_agent.stock_lab import decision_tests as dt

LEVELS = dt.holm_levels(0.05, 4)


def ar1(rng: np.random.RandomState, rows: int, length: int, phi: float) -> np.ndarray:
    shocks = rng.standard_normal((rows, length))
    out = np.empty_like(shocks)
    out[:, 0] = shocks[:, 0]
    for t in range(1, length):
        out[:, t] = phi * out[:, t - 1] + shocks[:, t]
    return out


def test_holm_compares_with_alpha_over_the_remaining_tests():
    assert dt.holm_levels(0.05, 4) == pytest.approx((0.0125, 0.05 / 3, 0.025, 0.05))


def test_autocovariances_and_bartlett_weights_are_the_shared_newey_west():
    series = ar1(np.random.RandomState(0), 2, 400, 0.6)
    acov = dt.autocovariances(series)
    centered = series[0] - series[0].mean()
    assert acov[0, 7] == pytest.approx(centered[7:] @ centered[:-7] / 400, rel=1e-12)
    for lag in (0, 10, 89):
        expected = long_run_variance(series[1], lag=lag)
        assert dt.bartlett_lrv(acov, lag + 1)[1] == pytest.approx(expected, rel=1e-10)
    assert dt._fft_length(9575) == 9600 and dt._fft_length(1999) == 2000


def test_the_current_tests_are_the_shared_estimators():
    rng = np.random.RandomState(1)
    series = ar1(rng, 3, 600, 0.3) * 0.01 - 0.001
    folds = np.arange(600) // 200
    stats = dt.procedure_statistics(series, folds, LEVELS, fixed_b={})
    for row in range(3):
        nw = newey_west_mean(series[row], lag=10)
        fold = fold_block_t(series[row], folds)
        assert stats["nw_lag10"].se[row] == pytest.approx(nw.se, rel=1e-10)
        assert stats["nw_lag10"].mean[row] == pytest.approx(nw.estimate, rel=1e-12)
        assert stats["fold_block_t"].se[row] == pytest.approx(fold.se, rel=1e-12)
        assert stats["fold_block_t"].mean[row] == pytest.approx(fold.estimate, rel=1e-12)
    np.testing.assert_allclose(stats["nw_lag10"].critical[0], [normal_ppf(q) for q in LEVELS])
    np.testing.assert_allclose(
        stats["fold_block_t"].critical[0], [student_t_ppf(q, 2) for q in LEVELS]
    )
    # 600 dates in blocks of 20: 30 blocks, t(29).
    np.testing.assert_allclose(
        stats["cbb_block20"].critical[0], [student_t_ppf(q, 29) for q in LEVELS]
    )


def test_the_circular_block_variance_is_exact_by_enumeration():
    """Seven dates, block 3: blocks of 3, 3 and 1 sessions with independent
    starts. Enumerating all 7^3 start triples gives the bootstrap variance."""

    x = np.array([0.3, -1.2, 0.8, 2.0, -0.4, 0.1, 1.5])
    doubled = np.concatenate([x, x])
    means = [
        (doubled[a : a + 3].sum() + doubled[b : b + 3].sum() + doubled[c]) / 7
        for a, b, c in itertools.product(range(7), repeat=3)
    ]
    assert dt.circular_block_variance(x[None], 3)[0] == pytest.approx(np.var(means), rel=1e-12)


def test_the_circular_block_variance_is_the_shared_bootstrap_without_resampling_noise():
    series = ar1(np.random.RandomState(2), 1, 500, 0.6)
    draws = draw_circular_blocks(500, block_length=20, reps=40_000, seed=3)
    exact = math.sqrt(dt.circular_block_variance(series, 20)[0])
    assert exact == pytest.approx(bootstrap_mean(series[0], draws).se, rel=0.02)


def test_the_stationary_bootstrap_variance_matches_resampling():
    """Politis-Romano's closed form against the resampling scheme itself: each
    session continues the block with probability 1 - 1/L (wrapping), or starts
    a new block at a uniform position."""

    x = ar1(np.random.RandomState(4), 1, 80, 0.5)[0]
    rng = np.random.RandomState(5)
    reps, length, block = 20_000, len(x), 4.0
    index = np.empty((reps, length), dtype=np.int64)
    index[:, 0] = rng.randint(length, size=reps)
    restart = rng.rand(reps, length) < 1.0 / block
    fresh = rng.randint(length, size=(reps, length))
    for t in range(1, length):
        index[:, t] = np.where(restart[:, t], fresh[:, t], (index[:, t - 1] + 1) % length)
    resampled = x[index].mean(axis=1).var()
    exact = dt.stationary_bootstrap_variance(dt.autocovariances(x[None]), np.array([block]))[0]
    assert exact == pytest.approx(resampled, rel=0.04)
    iid = dt.stationary_bootstrap_variance(dt.autocovariances(x[None]), np.array([1.0]))[0]
    assert iid == pytest.approx(x.var() / length, rel=1e-12)  # L = 1 is the iid bootstrap


def test_the_politis_white_block_follows_its_formula():
    """An independent scalar computation of the rule on known autocovariances
    (an AR(1) with phi = 0.8 at T = 400), then the behaviour it implies."""

    length, phi = 400, 0.8
    acov = phi ** np.arange(length)[None, :] / (1 - phi**2)
    run = 5  # max(5, ceil(sqrt(log10 400)))
    m_max = 20 + run
    threshold = 2 * math.sqrt(math.log10(length) / length)
    rho = phi ** np.arange(length)
    m = next(m for m in range(m_max + 1) if all(rho[m + k] < threshold for k in range(1, run + 1)))
    window = min(2 * max(m, 1), m_max)
    weight = [1.0 if k / window <= 0.5 else max(0.0, 2 * (1 - k / window)) for k in range(41)]
    g = acov[0, 0] + 2 * sum(weight[k] * acov[0, k] for k in range(1, window + 1))
    big_g = 2 * sum(weight[k] * k * acov[0, k] for k in range(1, window + 1))
    expected = min((big_g**2 / g**2) ** (1 / 3) * length ** (1 / 3), math.ceil(3 * 20))
    assert dt.politis_white_block(acov)[0] == pytest.approx(expected, rel=1e-12)

    rng = np.random.RandomState(6)
    noise = dt.politis_white_block(dt.autocovariances(rng.standard_normal((20, 2000))))
    persistent = dt.politis_white_block(dt.autocovariances(ar1(rng, 20, 2000, 0.9)))
    assert np.median(noise) < 3 and np.median(persistent) > 10
    assert (persistent <= math.ceil(3 * math.sqrt(2000))).all() and (noise >= 1).all()


def test_every_standard_error_ignores_a_constant_shift():
    """Power comes from shifting the mean of null series, so no procedure's
    standard error, critical value or block length may depend on the level."""

    series = ar1(np.random.RandomState(7), 4, 760, 0.7) * 0.002
    folds = np.arange(760) // 190
    fixed_b = {"nw_fixed_b_sqrt": np.array(LEVELS), "nw_fixed_b_tenth": np.array(LEVELS)}
    base = dt.procedure_statistics(series, folds, LEVELS, fixed_b=fixed_b)
    moved = dt.procedure_statistics(series - 0.001, folds, LEVELS, fixed_b=fixed_b)
    assert tuple(base) == dt.PROCEDURES
    for name in dt.PROCEDURES:
        np.testing.assert_allclose(moved[name].se, base[name].se, rtol=1e-9)
        np.testing.assert_allclose(moved[name].critical, base[name].critical)
        np.testing.assert_allclose(moved[name].mean, base[name].mean - 0.001, rtol=1e-9)
        shifted = base[name].rejects(0.001)
        np.testing.assert_array_equal(shifted, moved[name].rejects())


# --- fixed-b ---


def test_fixed_b_draws_are_seeded_symmetric_and_need_the_planned_bandwidth():
    first = dt.fixed_b_draws(0.1, steps=200, draws=500, seed=3)
    assert np.array_equal(first, dt.fixed_b_draws(0.1, steps=200, draws=500, seed=3))
    assert not np.array_equal(first, dt.fixed_b_draws(0.1, steps=200, draws=500, seed=4))
    assert np.allclose(first, -first[::-1]) and len(first) == 1000
    assert not first.flags.writeable
    assert dt.sqrt_bandwidth(4788) == 90 and dt.tenth_bandwidth(4788) == 479


def test_fixed_b_critical_values_meet_the_normal_limit_and_kiefer_vogelsang():
    """As b -> 0 the fixed-b limit is the normal. At b = 0.019 and 0.1 the
    simulated 95% and 97.5% points agree with Kiefer and Vogelsang's cubic
    approximation to within 0.05 (it reads about 0.03 higher at b = 0.1)."""

    tiny = dt.fixed_b_critical_values(0.001, [0.05, 0.025])
    np.testing.assert_allclose(tiny, [normal_ppf(0.05), normal_ppf(0.025)], atol=0.03)
    for b in (0.019, 0.1):
        simulated = -dt.fixed_b_critical_values(b, [0.05, 0.025])
        published = [dt.kiefer_vogelsang(b, 0.95), dt.kiefer_vogelsang(b, 0.975)]
        np.testing.assert_allclose(simulated, published, atol=0.05)
        assert (simulated > [1.6449, 1.96]).all()  # wider than the normal's


def test_the_planned_fixed_b_critical_values_are_reproduced():
    """The fixed-b critical values at T = 4,788 (the planned 19 folds) at the
    four Holm levels, as recorded in the packet."""

    table = dt.fixed_b_table(4788, LEVELS)
    for name, recorded in dt.PLANNED_FIXED_B.items():
        np.testing.assert_allclose(table[name], recorded, rtol=0, atol=5e-5)
