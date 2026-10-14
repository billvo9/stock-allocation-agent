"""
The per-step transition contract the RL environment injects (spec F8.1).

    portfolio_step(w_pre, target | None, r_next, cost_rate, cash_index) -> StepResult

One call books one close: trade from the drifted weights `w_pre` to `target`
at this close (no trade when `target` is None), then earn `r_next` over the
following span and drift. Every vector has length n with the cash account at
`cash_index`. `cost_rate` is a per-asset vector (cash entry 0).

The environment never computes a return, a cost or a drift itself. It only
schedules calls (decision timing, gaps) and reads `StepResult`. The one
function shared by backtests, optimizers, rule baselines and learned
policies is what keeps their comparison fair.

`check_step_result` enforces the shared invariants on whatever transition is
injected, so a defective transition fails loudly instead of training a policy
on broken economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ENGINEERING_ONLY = "engineering_only_unapproved_transition"
APPROVED = "approved"

SIMPLEX_TOLERANCE = 1e-9


class TransitionContractError(ValueError):
    """An injected transition returned a result that breaks the F8.1 invariants."""


@dataclass(frozen=True)
class StepResult:
    w_post_trade: np.ndarray
    w_next: np.ndarray
    gross_return: float
    cost: float
    net_return: float
    turnover_one_way: float
    traded_notional: float


class PortfolioStep(Protocol):
    def __call__(
        self,
        w_pre: np.ndarray,
        target: np.ndarray | None,
        r_next: np.ndarray,
        cost_rate: np.ndarray,
        cash_index: int,
    ) -> StepResult: ...


@dataclass(frozen=True)
class TransitionInfo:
    """
    Identity of an injected transition, recorded with every artifact.

    status is `approved` only for the owner-approved Track A implementation;
    anything else labels every result built on it as engineering only.
    """

    name: str
    version: str
    status: str

    @property
    def approved(self) -> bool:
        return self.status == APPROVED

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "status": self.status}


def _check_simplex(name: str, weights: np.ndarray, n_assets: int) -> None:
    weights = np.asarray(weights)
    if weights.shape != (n_assets,):
        raise TransitionContractError(f"{name} has shape {weights.shape}, expected ({n_assets},).")
    if not np.all(np.isfinite(weights)):
        raise TransitionContractError(f"{name} is not finite.")
    if np.any(weights < -SIMPLEX_TOLERANCE):
        raise TransitionContractError(f"{name} has a negative weight.")
    if abs(float(weights.sum()) - 1.0) > SIMPLEX_TOLERANCE:
        raise TransitionContractError(f"{name} sums to {float(weights.sum())!r}, not 1.")


def check_step_result(result: StepResult, *, n_assets: int) -> StepResult:
    """Validate the F8.1 invariants of one result; return it unchanged."""

    _check_simplex("w_post_trade", result.w_post_trade, n_assets)
    _check_simplex("w_next", result.w_next, n_assets)
    scalars = {
        "gross_return": result.gross_return,
        "cost": result.cost,
        "net_return": result.net_return,
        "turnover_one_way": result.turnover_one_way,
        "traded_notional": result.traded_notional,
    }
    for name, value in scalars.items():
        if not np.isfinite(value):
            raise TransitionContractError(f"{name} is not finite.")
    if not 0.0 <= result.cost < 1.0:
        raise TransitionContractError(f"cost {result.cost!r} is outside [0, 1).")
    if result.net_return > result.gross_return + 1e-15:
        raise TransitionContractError("net_return exceeds gross_return.")
    if result.net_return <= -1.0:
        raise TransitionContractError("net_return implies non-positive wealth.")
    if result.turnover_one_way < 0 or result.traded_notional < 0:
        raise TransitionContractError("turnover and traded notional must be non-negative.")
    return result
