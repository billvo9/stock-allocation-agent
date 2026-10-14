"""The exact long-only mean-variance solver used by the optimizer baseline."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from stock_agent.rl import optimizer as opt


def _objective(w, mu, cov, gamma):
    return 0.5 * gamma * w @ cov @ w - mu @ w


def _grid(n, step=0.02):
    ticks = np.arange(0.0, 1.0 + 1e-9, step)
    for point in itertools.product(ticks, repeat=n - 1):
        if sum(point) <= 1.0 + 1e-9:
            yield np.array([*point, 1.0 - sum(point)])


def test_one_risky_asset_and_cash_has_the_closed_form_merton_weight():
    mu = np.array([3e-4, 0.0])
    cov = np.diag([1e-4, 0.0])
    solution = opt.solve_mean_variance(mu, cov, risk_aversion=5.0)
    assert solution.weights[0] == pytest.approx(3e-4 / (5.0 * 1e-4), abs=1e-12)
    assert solution.weights.sum() == pytest.approx(1.0, abs=1e-15)


def test_the_solution_beats_every_point_of_a_fine_grid():
    rng = np.random.RandomState(0)
    for _ in range(5):
        a = rng.normal(size=(3, 3)) * 0.01
        cov = np.zeros((4, 4))
        cov[:3, :3] = a @ a.T + 1e-6 * np.eye(3)
        mu = np.append(rng.normal(0, 5e-4, size=3), 0.0)
        best = opt.solve_mean_variance(mu, cov, risk_aversion=4.0)
        floor = min(_objective(w, mu, cov, 4.0) for w in _grid(4))
        assert _objective(best.weights, mu, cov, 4.0) <= floor + 1e-12


def test_the_solution_satisfies_the_kkt_conditions():
    mu = np.array([4e-4, 1e-4, 8e-5, 0.0])
    vol = np.array([0.0095, 0.003, 0.0004])
    corr = np.array([[1.0, -0.2, 0.0], [-0.2, 1.0, 0.3], [0.0, 0.3, 1.0]])
    cov = np.zeros((4, 4))
    cov[:3, :3] = corr * np.outer(vol, vol)
    solution = opt.solve_mean_variance(mu, cov, risk_aversion=5.0)
    w = solution.weights
    gradient = 5.0 * cov @ w - mu
    free = w > 1e-12
    nu = gradient[free].mean()
    np.testing.assert_allclose(gradient[free], nu, atol=1e-12)
    assert np.all(gradient[~free] >= nu - 1e-12)


def test_caps_bind_and_the_excess_goes_where_the_objective_prefers():
    mu = np.array([5e-4, 0.0])
    cov = np.diag([1e-6, 0.0])
    solution = opt.solve_mean_variance(mu, cov, risk_aversion=1.0, upper=np.array([0.4, 1.0]))
    np.testing.assert_allclose(solution.weights, [0.4, 0.6], atol=1e-12)


def test_a_cap_set_that_cannot_hold_the_portfolio_is_rejected():
    with pytest.raises(ValueError, match="caps"):
        opt.solve_mean_variance(
            np.zeros(2), np.eye(2), risk_aversion=1.0, upper=np.array([0.3, 0.3])
        )


def test_non_finite_or_misshaped_inputs_are_rejected():
    with pytest.raises(ValueError):
        opt.solve_mean_variance(np.array([np.nan, 0.0]), np.eye(2), risk_aversion=1.0)
    with pytest.raises(ValueError):
        opt.solve_mean_variance(np.zeros(3), np.eye(2), risk_aversion=1.0)
    with pytest.raises(ValueError, match="risk_aversion"):
        opt.solve_mean_variance(np.zeros(2), np.eye(2), risk_aversion=0.0)


def test_the_solver_is_deterministic():
    mu = np.array([4e-4, 1e-4, 8e-5, 0.0])
    cov = np.diag([9e-5, 1e-5, 1.4e-7, 0.0])
    first = opt.solve_mean_variance(mu, cov, risk_aversion=5.0)
    np.testing.assert_array_equal(first.weights, opt.solve_mean_variance(mu, cov, 5.0).weights)
