"""
The registered rules as engine policies, using the very classes the
registered synthetic comparison uses (`stock_agent.rl.policies`, #88), so
their definitions cannot drift apart:

    buy_and_hold        `BuyAndHold`: equal weights over the universe at the
                        build decision, then None (hold) at every later one
    equal_weight        `ConstantMix`: 1/n in each holding at every decision
    inverse_volatility  `InverseVolatility`: w_i proportional to 1 / sigma_i,
                        sigma_i the sample standard deviation (ddof 1) of the
                        holding's visible returns in the last `window`
                        sessions (at least half of them must be visible; a
                        return spanning a missing session counts once)

The funding leg gets 0% in every target, so every portfolio is fully
invested. A policy sees only the engine's `returns[: d + 1]` at decision d
(`portfolio.engine.collect_targets`): nothing after close d.

The RL classes read an observation; here they get a minimal one holding only
the return history (`_History`), the one field these three rules read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_agent.portfolio.engine import Policy
from stock_agent.rl.policies import BuyAndHold, ConstantMix, InverseVolatility


@dataclass(frozen=True)
class _Snapshot:
    returns: np.ndarray


@dataclass(frozen=True)
class _History:
    """The part of an RL observation the three registered rules read."""

    snapshot: _Snapshot


def equal_weights(n_assets: int, cash_index: int) -> np.ndarray:
    """1/n on every holding and 0 on the funding leg (n = n_assets - 1)."""

    weights = np.full(n_assets, 1.0 / (n_assets - 1))
    weights[cash_index] = 0.0
    return weights


def make_policy(rule: str, n_assets: int, cash_index: int, *, window: int) -> Policy:
    """A fresh engine policy for `rule` over `n_assets` columns (funding leg included)."""

    if rule == "buy_and_hold":
        decider = BuyAndHold("buy_and_hold", equal_weights(n_assets, cash_index))
    elif rule == "equal_weight":
        decider = ConstantMix("equal_weight", equal_weights(n_assets, cash_index))
    elif rule == "inverse_volatility":
        decider = InverseVolatility(window=window, cash_index=cash_index)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    decider.reset()

    def policy(history: np.ndarray) -> np.ndarray | None:
        return decider.decide(_History(_Snapshot(history)))

    return policy
