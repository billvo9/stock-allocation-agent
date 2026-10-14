"""
ENGINEERING STAND-IN: the PROPOSED (unapproved) per-step rules D2 / F8.1.

This is not the project's economics. It exists so the RL environment and the
PPO pipeline can be built and tested before Track A's approved
`portfolio_step` merges. Every result built on it carries the label
`engineering_only_unapproved_transition`. Delete it, and inject Track A's
function, once the owner approves D1-D8 and that implementation merges.

Proposed rules (t5_portfolio_formulas.md sections 2 and 8, decision packet
D2 and D8):

    delta = target - w_pre                      (trade from drifted weights)
    C     = sum_{i != cash} c_i * |delta_i|      (cash is the free funding leg)
    1 + R_net = (1 - C) * (1 + w_post . r)       (multiplicative cost)
    w_next = w_post * (1 + r) / (1 + w_post . r) (drift identity)
    turnover_one_way = 0.5 * sum_all |delta_i|
    traded_notional  = sum_{i != cash} |delta_i|
"""

from __future__ import annotations

import numpy as np

from stock_agent.rl.transition import (
    ENGINEERING_ONLY,
    SIMPLEX_TOLERANCE,
    StepResult,
    TransitionInfo,
)

TRANSITION_INFO = TransitionInfo(
    name="rl.reference_transition.proposed_portfolio_step",
    version="0.1.0",
    status=ENGINEERING_ONLY,
)

DUST = 1e-12


def _as_weights(name: str, values: np.ndarray, n_assets: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (n_assets,):
        raise ValueError(f"{name} has shape {weights.shape}, expected ({n_assets},).")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must be finite.")
    if np.any(weights < -SIMPLEX_TOLERANCE):
        raise ValueError(f"{name} has a negative weight; long-only weights are required.")
    if abs(float(weights.sum()) - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"{name} must sum to 1 (got {float(weights.sum())!r}).")
    return np.clip(weights, 0.0, None)


def proposed_portfolio_step(
    w_pre: np.ndarray,
    target: np.ndarray | None,
    r_next: np.ndarray,
    cost_rate: np.ndarray,
    cash_index: int,
) -> StepResult:
    w_pre = np.asarray(w_pre, dtype=np.float64)
    n_assets = w_pre.shape[0]
    w_pre = _as_weights("w_pre", w_pre, n_assets)
    rates = np.asarray(cost_rate, dtype=np.float64)
    if rates.shape != (n_assets,) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("cost_rate must be a finite, non-negative vector of length n.")
    returns = np.asarray(r_next, dtype=np.float64)
    if returns.shape != (n_assets,):
        raise ValueError(f"r_next has shape {returns.shape}, expected ({n_assets},).")

    w_post = w_pre.copy() if target is None else _as_weights("target", target, n_assets)
    delta = w_post - w_pre
    delta[np.abs(delta) < DUST] = 0.0
    risky = np.ones(n_assets, dtype=bool)
    risky[cash_index] = False
    cost = float(np.sum(rates[risky] * np.abs(delta[risky])))
    if cost >= 1.0:
        raise ValueError(f"Trading cost {cost!r} would consume the whole portfolio.")

    held = w_post > 0
    if not np.all(np.isfinite(returns[held])):
        raise ValueError("r_next is not finite for a held asset.")
    earned = np.where(held, returns, 0.0)
    gross = float(np.dot(w_post, earned))
    net = (1.0 - cost) * (1.0 + gross) - 1.0
    w_next = w_post * (1.0 + earned) / (1.0 + gross)

    return StepResult(
        w_post_trade=w_post,
        w_next=w_next,
        gross_return=gross,
        cost=cost,
        net_return=float(net),
        turnover_one_way=float(0.5 * np.sum(np.abs(delta))),
        traded_notional=float(np.sum(np.abs(delta[risky]))),
    )
