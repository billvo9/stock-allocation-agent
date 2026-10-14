"""
T5a engine contract: the one per-step transition shared by the backtest
engine, constrained optimizers and the RL environment (spec F8.1,
`docs/research/t5_portfolio_formulas.md`, section 8).

STATUS: INTERFACE ONLY. OD8 group B (decisions D1-D8 in
`docs/research/t5a_open_decisions.md`) awaits the owner's approval, so every
function body here raises `NotImplementedError`. No execution, cost,
constraint or reward formula is live until that approval is recorded. The
data types below validate structure only (shapes, finiteness, signs).

Conventions. Vectors are 1-D float arrays over one asset order; the
component at `cash_index` is the cash account.

- `w_pre`: the drifted weights at execution close s, just before any trade
  there (w-tilde).
- `target`: the requested post-trade weights at close s, already passed
  through `feasible_target`; None means no trade at s.
- `r_next`: each asset's return from close s to close s+1; the cash
  component is the cash account's return for that session. The post-trade
  holdings earn it. No decision may ever see it.
- `cost_rate`: each asset's all-in cost per unit of traded notional
  (`CostModel.rates_for`). The cash component is the funding leg.

The function sees no dates: the execution lag (D1) and the rebalance
schedule (D7) live in the caller (`stock_agent.portfolio.schedule`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np

PENDING_APPROVAL = "awaiting OD8 group B approval (D1–D8)"

# Step reasons (machine-readable). Schedule-level reasons, such as a
# deferred rebalance, live in `stock_agent.portfolio.schedule`.
TRADED = "traded"
NO_TARGET = "no_target"  # target is None: pure drift, no cost
NO_TRADE = "no_trade"  # a target equal to w_pre: pure drift, no cost
STEP_REASONS = frozenset({TRADED, NO_TARGET, NO_TRADE})

# Constraints that can change a requested target (D5).
CAP = "w_max"
TURNOVER_LIMIT = "tau_max"


class InvalidTarget(ValueError):
    """
    A requested target breaks a rule no policy may break (negative weights,
    weights not summing to 1, a wrong shape). D5: such a target is an error,
    never silently repaired.
    """


def _frozen_vector(value: object, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(f"{name} must be a 1-D vector over at least two assets")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    vector.flags.writeable = False
    return vector


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, eq=False)
class StepResult:
    """
    The outcome of one execution close s and the session after it.

    - `w_post_trade`: the weights right after the trade at close s (w*).
    - `w_next`: the drifted weights at close s+1, before any trade there.
    - `gross_return`: the post-trade holdings' return over the session.
    - `cost`: C, the trading cost as a fraction of pre-trade wealth.
    - `net_return`: R_net, wealth at close s+1 over pre-trade wealth at
      close s, minus 1. The RL reward is built from it (D8).
    - `turnover_one_way`: half the summed absolute weight change over every
      component, cash included (a full switch is 1).
    - `traded_notional`: the summed absolute weight change over the
      non-cash components, as a fraction of pre-trade wealth.
    - `reason`: one of `STEP_REASONS`.

    Arrays are read-only copies.
    """

    w_post_trade: np.ndarray
    w_next: np.ndarray
    gross_return: float
    cost: float
    net_return: float
    turnover_one_way: float
    traded_notional: float
    reason: str

    def __post_init__(self) -> None:
        post = _frozen_vector(self.w_post_trade, "w_post_trade")
        following = _frozen_vector(self.w_next, "w_next")
        if post.shape != following.shape:
            raise ValueError("w_post_trade and w_next must have the same shape")
        object.__setattr__(self, "w_post_trade", post)
        object.__setattr__(self, "w_next", following)
        for name in ("gross_return", "cost", "net_return", "turnover_one_way", "traded_notional"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(self.cost, self.turnover_one_way, self.traded_notional) < 0:
            raise ValueError("cost, turnover and traded notional are non-negative")
        if self.reason not in STEP_REASONS:
            raise ValueError(f"unknown step reason {self.reason!r}")


@dataclass(frozen=True)
class CostModel:
    """
    Per-asset all-in trading cost per unit of traded notional (D2; F2.2,
    F2.6).

    - `rates`: asset name to its all-in rate as a fraction (0.0005 is 5 bp):
      commission, half-spread and slippage together, since there is no
      spread or slippage data to separate them. The rates are a registered
      assumption, never tuned on results.
    - `multiplier`: the sensitivity scale (D2 shows every backtest at 0x, 1x
      and 2x).
    """

    rates: Mapping[str, float]
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        rates = {
            str(asset): _finite(rate, f"rate of {asset}") for asset, rate in self.rates.items()
        }
        if not rates:
            raise ValueError("a cost model needs at least one rate")
        if any(rate < 0 for rate in rates.values()):
            raise ValueError("cost rates are non-negative")
        if _finite(self.multiplier, "multiplier") < 0:
            raise ValueError("the multiplier is non-negative")
        object.__setattr__(self, "rates", MappingProxyType(rates))

    def rates_for(self, assets: Sequence[str]) -> np.ndarray:
        """The `cost_rate` vector for `assets`, in that order, scaled by `multiplier`."""

        raise NotImplementedError(PENDING_APPROVAL)


@dataclass(frozen=True)
class Constraints:
    """
    The one feasibility set applied identically to every compared policy
    (D5). Registered per experiment, never per policy.

    - `cash_index`: the cash account's component; exempt from the cap.
    - `w_max`: the per-asset cap; 1.0 means no cap.
    - `tau_max`: the one-way turnover limit per rebalance; None means no limit.
    """

    cash_index: int
    w_max: float = 1.0
    tau_max: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cash_index, bool) or not isinstance(self.cash_index, int):
            raise TypeError("cash_index must be an int")
        if self.cash_index < 0:
            raise ValueError("cash_index must be non-negative")
        if not 0.0 < _finite(self.w_max, "w_max") <= 1.0:
            raise ValueError("w_max must lie in (0, 1]")
        if self.tau_max is not None and not 0.0 < _finite(self.tau_max, "tau_max") <= 1.0:
            raise ValueError("tau_max must lie in (0, 1], or be None for no limit")


@dataclass(frozen=True, eq=False)
class FeasibleTarget:
    """
    The output of the feasibility map (D5), recorded so no policy gains from
    a different API.

    - `requested`: the policy's target.
    - `weights`: the target handed to `portfolio_step`.
    - `distance_one_way`: half the summed absolute difference between
      `weights` and `requested` (0 when no constraint binds).
    - `binding`: the constraints that changed the target (`CAP`,
      `TURNOVER_LIMIT`), in the order applied.
    - `alpha`: the fraction of the trade executed under the turnover limit
      (1.0 when it does not bind).
    """

    requested: np.ndarray
    weights: np.ndarray
    distance_one_way: float
    binding: tuple[str, ...] = field(default=())
    alpha: float = 1.0

    def __post_init__(self) -> None:
        requested = _frozen_vector(self.requested, "requested")
        weights = _frozen_vector(self.weights, "weights")
        if requested.shape != weights.shape:
            raise ValueError("requested and weights must have the same shape")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "weights", weights)
        if _finite(self.distance_one_way, "distance_one_way") < 0:
            raise ValueError("distance_one_way is non-negative")
        if not 0.0 <= _finite(self.alpha, "alpha") <= 1.0:
            raise ValueError("alpha lies in [0, 1]")
        unknown = set(self.binding) - {CAP, TURNOVER_LIMIT}
        if unknown:
            raise ValueError(f"unknown binding constraints {sorted(unknown)}")
        object.__setattr__(self, "binding", tuple(self.binding))


class Transition(Protocol):
    """
    The per-step contract every simulator uses: the backtest engine, the RL
    environment and any optimizer replay. `portfolio_step` is the one
    production implementation; tests substitute mutants through this seam.
    """

    def __call__(
        self,
        w_pre: np.ndarray,
        target: np.ndarray | None,
        r_next: np.ndarray,
        cost_rate: np.ndarray,
        cash_index: int,
    ) -> StepResult: ...


def portfolio_step(
    w_pre: np.ndarray,
    target: np.ndarray | None,
    r_next: np.ndarray,
    cost_rate: np.ndarray,
    cash_index: int,
) -> StepResult:
    """
    One execution close and the session after it (F8.1). Proposed, PENDING
    the owner's approval (D2, D3, D8):

    1. Trade at close s: delta = target - w_pre; cost
       C = sum over non-cash i of cost_rate[i] * |delta[i]| (the cash leg is
       free); pre-trade wealth W- becomes W+ = W-(1 - C).
    2. Earn r_next: 1 + R_net = (1 - C)(1 + target . r_next).
    3. Drift: w_next = target * (1 + r_next) / (1 + target . r_next).

    No target, or a target equal to `w_pre`, is pure drift with C = 0. A
    non-finite return on a held asset raises; the caller records a reason.

    Raises NotImplementedError until OD8 group B is approved.
    """

    raise NotImplementedError(PENDING_APPROVAL)


def feasible_target(
    requested: np.ndarray,
    w_pre: np.ndarray,
    constraints: Constraints,
) -> FeasibleTarget:
    """
    The deterministic feasibility map every policy's target passes through
    before `portfolio_step` (D5). Proposed, PENDING the owner's approval:

    1. Long only and summing to 1, else `InvalidTarget` (never repaired).
    2. Cap each non-cash weight at `w_max`; the excess goes to cash.
    3. If the one-way turnover from `w_pre` exceeds `tau_max`, trade part
       way: w* = w_pre + alpha (target - w_pre), alpha = tau_max / turnover.

    Drift above the cap between rebalances is allowed and reported, not sold.

    Raises NotImplementedError until OD8 group B is approved.
    """

    raise NotImplementedError(PENDING_APPROVAL)
