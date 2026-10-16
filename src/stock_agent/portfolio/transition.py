"""
T5a engine contract: the one per-step transition shared by the backtest
engine, constrained optimizers and the RL environment (spec F8.1,
`docs/research/t5_portfolio_formulas.md`, section 8).

STATUS: APPROVED. The owner approved OD8 group B (decisions D1-D8 in
`docs/research/t5a_open_decisions.md`) on 2026-10-15. This module is the
single place where trading costs (D2), the feasibility map (D5) and drift
are defined.

Conventions. Vectors are 1-D float arrays over one asset order; the
component at `cash_index` is the cash account.

- `w_pre`: the drifted weights at execution close s, just before any trade
  there (w-tilde).
- `target`: the requested post-trade weights at close s, already passed
  through `feasible_target`; None means no trade at s.
- `r_next`: each asset's return from close s to the next valued close (one
  session, or a multi-session span across a gap); the cash component is the
  cash account's return over it. The post-trade holdings earn it. No
  decision may ever see it.
- `cost_rate`: each asset's all-in cost per unit of traded notional
  (`CostModel.rates_for`). The cash component is the funding leg: it is
  never charged, whatever its entry.

The function sees no dates: the execution lag (D1) and the rebalance
schedule (D7) live in the caller (`stock_agent.portfolio.schedule`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np

# Step reasons (machine-readable). Schedule-level reasons, such as a
# deferred rebalance, live in `stock_agent.portfolio.schedule`.
TRADED = "traded"
NO_TARGET = "no_target"  # target is None: pure drift, no cost
NO_TRADE = "no_trade"  # a target within DUST of w_pre: pure drift, no cost
STEP_REASONS = frozenset({TRADED, NO_TARGET, NO_TRADE})

# Constraints that can change a requested target (D5).
CAP = "w_max"
TURNOVER_LIMIT = "tau_max"

# A requested target may miss a sum of 1 by floating-point residue only.
SUM_TOLERANCE = 1e-9
# A weight change below this is dust: no trade, no cost (spec 2 (f)).
DUST = 1e-12

# D2 / F2.6: the provisional all-in rates per asset class, per unit of traded
# notional (commission, half-spread and slippage together). Approved on
# 2026-10-15 as provisional assumptions: registered here, never tuned on
# results, and every backtest is shown at each SENSITIVITY_MULTIPLIERS.
SINGLE_STOCK = "single_stock"
BROAD_ETF = "broad_etf"  # broad equity and Treasury ETFs
TBILL_FUND = "tbill_fund"  # a T-bill fund such as SHV
CASH_ACCOUNT = "cash"  # the funding leg: trading it is free
PROVISIONAL_RATES = MappingProxyType(
    {SINGLE_STOCK: 0.0010, BROAD_ETF: 0.0005, TBILL_FUND: 0.0002, CASH_ACCOUNT: 0.0}
)
SENSITIVITY_MULTIPLIERS = (0.0, 1.0, 2.0)


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


def _simplex(
    value: object, name: str, size: int | None = None, error: type[ValueError] = ValueError
) -> np.ndarray:
    """A long-only weight vector summing to 1, as a fresh array; never repaired."""

    weights = np.array(value, dtype=float)
    if weights.ndim != 1 or weights.size < 2:
        raise error(f"{name} must be a 1-D vector over at least two assets")
    if size is not None and weights.size != size:
        raise error(f"{name} has {weights.size} weights, expected {size}")
    if not np.isfinite(weights).all():
        raise error(f"{name} must be finite")
    if (weights < 0).any():
        raise error(f"{name} has a negative weight; weights are long-only")
    if abs(float(weights.sum()) - 1.0) > SUM_TOLERANCE:
        raise error(f"{name} must sum to 1, not {float(weights.sum())!r}")
    return weights


def check_target(target: object, size: int) -> np.ndarray:
    """A requested target as a fresh array, or `InvalidTarget` (D5: never repaired)."""

    return _simplex(target, "target", size, InvalidTarget)


def _cash(cash_index: int, size: int) -> int:
    if isinstance(cash_index, bool) or not isinstance(cash_index, int | np.integer):
        raise TypeError("cash_index must be an int")
    if not 0 <= cash_index < size:
        raise ValueError(f"cash_index {cash_index} is outside the {size} assets")
    return int(cash_index)


@dataclass(frozen=True, eq=False)
class StepResult:
    """
    The outcome of one execution close s and the span after it.

    - `w_post_trade`: the weights right after the trade at close s (w*).
    - `w_next`: the drifted weights at the span's end, before any trade there.
    - `gross_return`: the post-trade holdings' return over the span.
    - `cost`: C, the trading cost as a fraction of pre-trade wealth.
    - `net_return`: R_net, wealth at the span's end over pre-trade wealth at
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
      assumption, never tuned on results; `provisional` builds them from
      the approved per-class rates.
    - `multiplier`: the sensitivity scale (D2 shows every backtest at 0x, 1x
      and 2x; see `sensitivity`).
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

    @classmethod
    def provisional(cls, asset_classes: Mapping[str, str], multiplier: float = 1.0) -> CostModel:
        """The D2 provisional rates for `asset_classes` (asset name to asset class)."""

        unknown = sorted(set(asset_classes.values()) - set(PROVISIONAL_RATES))
        if unknown:
            raise ValueError(f"no provisional rate for asset classes {unknown}")
        rates = {asset: PROVISIONAL_RATES[klass] for asset, klass in asset_classes.items()}
        return cls(rates, multiplier)

    def rates_for(self, assets: Sequence[str]) -> np.ndarray:
        """The `cost_rate` vector for `assets`, in that order, scaled by `multiplier`."""

        missing = [asset for asset in assets if asset not in self.rates]
        if missing:
            raise KeyError(f"no cost rate for {missing}")
        return np.array([self.rates[asset] for asset in assets], dtype=float) * self.multiplier

    def sensitivity(self) -> tuple[CostModel, ...]:
        """The registered rates at 0x, 1x and 2x (D2): one model per multiplier."""

        return tuple(dataclasses.replace(self, multiplier=m) for m in SENSITIVITY_MULTIPLIERS)


@dataclass(frozen=True)
class Constraints:
    """
    The one feasibility set applied identically to every compared policy
    (D5). Registered per experiment, never per policy.

    - `cash_index`: the cash account's component; exempt from the cap.
    - `w_max`: the per-asset cap; 1.0 (the default) means no cap.
    - `tau_max`: the one-way turnover limit per rebalance; None (the default)
      means no limit.
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
    One execution close and the span after it (F8.1; D2, D3, D8 approved
    2026-10-15):

    1. Trade at close s: delta = target - w_pre, from the drifted weights
       (F2.1); cost C = sum over non-cash i of cost_rate[i] * |delta[i]|, the
       cash leg being free (F2.2); pre-trade wealth W- becomes W+ = W-(1 - C)
       (F2.3, multiplicative; the O(c^2) fixed point is ignored).
    2. Earn r_next: 1 + R_net = (1 - C)(1 + target . r_next).
    3. Drift: w_next = target * (1 + r_next) / (1 + target . r_next).

    No target, or a target within `DUST` of `w_pre` in every component, is
    pure drift with C = 0. Weight changes below `DUST` are zeroed. A
    non-finite return on a held asset raises; the caller records a reason.
    A cost of 1 or more, which would wipe out wealth, raises.
    """

    pre = _simplex(w_pre, "w_pre")
    n = pre.size
    cash = _cash(cash_index, n)
    rates = np.array(cost_rate, dtype=float)
    if rates.shape != (n,) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError(f"cost_rate must be a finite, non-negative vector over {n} assets")
    returns = np.array(r_next, dtype=float)
    if returns.shape != (n,):
        raise ValueError(f"r_next must be a vector over {n} assets")

    if target is None:
        post, delta, reason = pre, np.zeros(n), NO_TARGET
    else:
        requested = check_target(target, n)
        delta = requested - pre
        delta[np.abs(delta) < DUST] = 0.0
        if delta.any():
            post, reason = requested, TRADED
        else:
            post, reason = pre, NO_TRADE

    risky = np.arange(n) != cash
    cost = float(np.sum(rates[risky] * np.abs(delta[risky])))
    if cost >= 1.0:
        raise ValueError(f"the trading cost {cost!r} would consume the whole portfolio")

    held = post > 0
    if not np.isfinite(returns[held]).all():
        raise ValueError("r_next is not finite for a held asset")
    if (returns[held] < -1.0).any():
        raise ValueError("a held asset cannot lose more than 100%")
    earned = np.where(held, returns, 0.0)
    gross = float(np.dot(post, earned))
    if 1.0 + gross <= 0.0:
        raise ValueError("the post-trade holdings lose all their value")
    net = (1.0 - cost) * (1.0 + gross) - 1.0
    w_next = post * (1.0 + earned) / (1.0 + gross)

    return StepResult(
        w_post_trade=post,
        w_next=w_next,
        gross_return=gross,
        cost=cost,
        net_return=net,
        turnover_one_way=float(0.5 * np.sum(np.abs(delta))),
        traded_notional=float(np.sum(np.abs(delta[risky]))),
        reason=reason,
    )


def feasible_target(
    requested: np.ndarray,
    w_pre: np.ndarray,
    constraints: Constraints,
) -> FeasibleTarget:
    """
    The deterministic feasibility map every policy's target passes through
    before `portfolio_step` (D5, approved 2026-10-15):

    1. Long only and summing to 1, else `InvalidTarget` (never repaired).
    2. Cap each non-cash weight at `w_max`; the excess goes to cash.
    3. If the one-way turnover from `w_pre` exceeds `tau_max`, trade part
       way: w* = w_pre + alpha (target - w_pre), alpha = tau_max / turnover.

    A target inside the set is returned unchanged. Drift above the cap
    between rebalances is allowed and reported, not sold; a partial trade
    under the turnover limit can therefore leave a weight above the cap.
    """

    pre = np.array(w_pre, dtype=float)
    if pre.ndim != 1 or pre.size < 2 or not np.isfinite(pre).all():
        raise ValueError("w_pre must be a finite 1-D vector over at least two assets")
    n = pre.size
    cash = _cash(constraints.cash_index, n)
    wanted = check_target(requested, n)

    weights = wanted.copy()
    binding: list[str] = []
    over = (np.arange(n) != cash) & (weights > constraints.w_max)
    if over.any():
        weights[cash] += float(np.sum(weights[over] - constraints.w_max))
        weights[over] = constraints.w_max
        binding.append(CAP)
    alpha = 1.0
    if constraints.tau_max is not None:
        turnover = 0.5 * float(np.sum(np.abs(weights - pre)))
        if turnover > constraints.tau_max:
            alpha = constraints.tau_max / turnover
            weights = pre + alpha * (weights - pre)
            binding.append(TURNOVER_LIMIT)
    return FeasibleTarget(
        requested=wanted,
        weights=weights,
        distance_one_way=0.5 * float(np.sum(np.abs(weights - wanted))),
        binding=tuple(binding),
        alpha=alpha,
    )
