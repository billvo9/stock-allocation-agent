"""
Rule-based allocation baselines for the RL comparison.

Every policy has the same interface and sees the same point-in-time
`Observation` at each decision of the same environment:

    decide(observation) -> target weights (long-only, sum 1) | None (hold)

Registered baselines (parameters fixed before any result is seen):

    buy_and_hold      the fixed balanced weights, bought once, never rebalanced
    equal_weight      1/3 in each fund, no cash, at every decision
    fixed_balanced    60% equity, 30% bond, 10% short Treasury, 0% cash
    InverseVolatility weights proportional to 1 / trailing volatility (funds)
    MeanVariance      long-only mean-variance on the trailing window's sample
                      mean and covariance (the same 60 sessions the learned
                      policies' features use), cash as the riskless asset

The cash account earns 0 under decision D4 (a) (pending approval), so the
short-Treasury fund is the natural conservative holding; the registered
weights reflect that.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from stock_agent.rl.observation import Observation
from stock_agent.rl.optimizer import solve_mean_variance
from stock_agent.rl.scenarios import CASH_INDEX

FIXED_BALANCED_WEIGHTS = np.array([0.6, 0.3, 0.1, 0.0])
FIXED_BALANCED_WEIGHTS.flags.writeable = False


class PolicyInputError(ValueError):
    """A policy cannot form a decision from the observation it was given."""


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def decide(self, observation: Observation) -> np.ndarray | None: ...


def _registered(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64).copy()
    if np.any(array < 0) or abs(array.sum() - 1.0) > 1e-12:
        raise ValueError("Registered weights must be long-only and sum to 1.")
    array.flags.writeable = False
    return array


class ConstantMix:
    """Rebalance to the same registered weights at every decision."""

    def __init__(self, name: str, weights) -> None:
        self.name = name
        self.weights = _registered(weights)

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> np.ndarray:
        return self.weights.copy()


class BuyAndHold:
    """Buy the registered weights at the first decision, then never trade."""

    def __init__(self, name: str, weights) -> None:
        self.name = name
        self.weights = _registered(weights)
        self._bought = False

    def reset(self) -> None:
        self._bought = False

    def decide(self, observation: Observation) -> np.ndarray | None:
        if self._bought:
            return None
        self._bought = True
        return self.weights.copy()


def buy_and_hold() -> BuyAndHold:
    return BuyAndHold("buy_and_hold", FIXED_BALANCED_WEIGHTS)


def equal_weight() -> ConstantMix:
    return ConstantMix("equal_weight", [1 / 3, 1 / 3, 1 / 3, 0.0])


def fixed_balanced() -> ConstantMix:
    return ConstantMix("fixed_balanced", FIXED_BALANCED_WEIGHTS)


def _trailing(observation: Observation, window: int) -> np.ndarray:
    returns = observation.snapshot.returns
    if returns.shape[0] < window:
        raise PolicyInputError(f"Fewer than {window} sessions of history.")
    return returns[-window:]


class InverseVolatility:
    """w_i proportional to 1 / sigma_i over the funds; cash 0."""

    def __init__(self, window: int = 60, cash_index: int = CASH_INDEX) -> None:
        self.name = "inverse_volatility"
        self.window = window
        self.cash_index = cash_index

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> np.ndarray:
        window = _trailing(observation, self.window)
        n = window.shape[1]
        needed = max(2, math.ceil(self.window / 2))
        inverse = np.zeros(n)
        for asset in range(n):
            if asset == self.cash_index:
                continue
            values = window[:, asset][np.isfinite(window[:, asset])]
            if values.size < needed:
                raise PolicyInputError("Too few visible returns for a volatility estimate.")
            sigma = float(values.std(ddof=1))
            if not sigma > 0:
                raise PolicyInputError("A fund has zero estimated volatility.")
            inverse[asset] = 1.0 / sigma
        return inverse / inverse.sum()


class MeanVariance:
    """
    Myopic long-only mean-variance on trailing sample estimates.

    mu: for the funds, the sample mean over the window's complete rows (rows
    where every fund's return is visible), the same rows as Sigma; for cash,
    the mean of its visible returns. Scaled by (1 - mean_shrinkage) (a
    registered sensitivity; 0 by default).
    Sigma: the funds' sample covariance over those complete rows, plus
    `ridge` times its average variance on the diagonal; cash is riskless. Sample means over 60 sessions are very noisy, so this
    baseline is expected to trade often; that is its honest behaviour.
    """

    def __init__(
        self,
        risk_aversion: float = 5.0,
        window: int = 60,
        mean_shrinkage: float = 0.0,
        ridge: float = 1e-6,
        upper: np.ndarray | None = None,
        cash_index: int = CASH_INDEX,
    ) -> None:
        if not 0.0 <= mean_shrinkage <= 1.0:
            raise ValueError("mean_shrinkage must be in [0, 1].")
        self.name = "mean_variance"
        self.risk_aversion = risk_aversion
        self.window = window
        self.mean_shrinkage = mean_shrinkage
        self.ridge = ridge
        self.upper = upper
        self.cash_index = cash_index

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> np.ndarray:
        window = _trailing(observation, self.window)
        n = window.shape[1]
        funds = np.array([i != self.cash_index for i in range(n)])
        rows = window[np.all(np.isfinite(window[:, funds]), axis=1)]
        if rows.shape[0] < max(int(funds.sum()) + 2, math.ceil(self.window / 2)):
            raise PolicyInputError("Too few complete rows for a covariance estimate.")
        mu = np.zeros(n)
        mu[funds] = rows[:, funds].mean(axis=0)
        cash = window[:, self.cash_index]
        mu[self.cash_index] = (
            float(cash[np.isfinite(cash)].mean()) if np.isfinite(cash).any() else 0
        )
        mu *= 1.0 - self.mean_shrinkage
        cov = np.zeros((n, n))
        fund_cov = np.cov(rows[:, funds], rowvar=False, ddof=1)
        k = int(funds.sum())
        fund_cov = fund_cov + self.ridge * np.trace(fund_cov) / k * np.eye(k)
        cov[np.ix_(funds, funds)] = fund_cov
        return solve_mean_variance(mu, cov, self.risk_aversion, self.upper).weights
