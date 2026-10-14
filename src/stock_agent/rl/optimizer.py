"""
An exact long-only mean-variance solver for small universes (no SciPy).

    maximize  mu'w - (gamma / 2) w' Sigma w
    s.t.      sum(w) = 1,  0 <= w_i <= u_i

The problem is a convex quadratic program. Its optimum has some pattern of
weights at zero, at their cap, or strictly between; on that pattern the KKT
conditions are a small linear system:

    gamma * Sigma_SS w_S - nu * 1 = mu_S - gamma * Sigma_SU u_U
    1' w_S                         = 1 - sum(u_U)

For n <= 8 assets every pattern is enumerated (at most 3^n systems), each
feasible solution is evaluated, and the best is returned. Every candidate is
feasible and the optimum's own pattern is among them, so the result is the
global optimum, deterministically, without iteration counts or tolerances to
tune. A singular pattern (two riskless assets both free) is skipped.

Daily units throughout: mu is a per-session mean return and Sigma a
per-session covariance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

MAX_ASSETS = 8
FEASIBILITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MeanVarianceSolution:
    weights: np.ndarray
    objective: float
    patterns_evaluated: int


def _validate(mu, cov, risk_aversion, upper):
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    n = mu.shape[0] if mu.ndim == 1 else -1
    if mu.ndim != 1 or not 1 <= n <= MAX_ASSETS:
        raise ValueError(f"mu must be a vector of 1 to {MAX_ASSETS} assets.")
    if cov.shape != (n, n):
        raise ValueError("cov must be an n x n matrix.")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))):
        raise ValueError("mu and cov must be finite.")
    if not np.allclose(cov, cov.T, atol=1e-15):
        raise ValueError("cov must be symmetric.")
    if not risk_aversion > 0:
        raise ValueError("risk_aversion must be positive.")
    caps = np.ones(n) if upper is None else np.asarray(upper, dtype=np.float64)
    if caps.shape != (n,) or np.any(caps < 0) or not np.all(np.isfinite(caps)):
        raise ValueError("upper must be a finite, non-negative vector of length n.")
    if caps.sum() < 1.0 - FEASIBILITY_TOLERANCE:
        raise ValueError("The caps cannot hold a fully invested portfolio.")
    return mu, cov, np.minimum(caps, 1.0)


def solve_mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    upper: np.ndarray | None = None,
) -> MeanVarianceSolution:
    mu, cov, caps = _validate(mu, cov, risk_aversion, upper)
    n = mu.shape[0]
    gamma = float(risk_aversion)
    states_per_asset = [
        ("zero", "free", "cap") if caps[i] < 1.0 else ("zero", "free") for i in range(n)
    ]
    best_weights: np.ndarray | None = None
    best_value = np.inf
    evaluated = 0
    for pattern in itertools.product(*states_per_asset):
        free = np.array([s == "free" for s in pattern])
        capped = np.array([s == "cap" for s in pattern])
        weights = np.where(capped, caps, 0.0)
        remaining = 1.0 - weights.sum()
        if not free.any():
            if abs(remaining) > FEASIBILITY_TOLERANCE:
                continue
        else:
            k = int(free.sum())
            system = np.zeros((k + 1, k + 1))
            system[:k, :k] = gamma * cov[np.ix_(free, free)]
            system[:k, k] = -1.0
            system[k, :k] = 1.0
            rhs = np.append(mu[free] - gamma * cov[np.ix_(free, capped)] @ caps[capped], remaining)
            try:
                solution = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                continue
            free_weights = solution[:k]
            if not np.all(np.isfinite(free_weights)):
                continue
            if np.any(free_weights < -FEASIBILITY_TOLERANCE) or np.any(
                free_weights > caps[free] + FEASIBILITY_TOLERANCE
            ):
                continue
            weights[free] = np.clip(free_weights, 0.0, caps[free])
        evaluated += 1
        value = 0.5 * gamma * weights @ cov @ weights - mu @ weights
        if value < best_value - 1e-18:
            best_value, best_weights = float(value), weights
    if best_weights is None:
        raise ValueError("No feasible portfolio was found.")
    return MeanVarianceSolution(
        weights=best_weights / best_weights.sum(),
        objective=-best_value,
        patterns_evaluated=evaluated,
    )
