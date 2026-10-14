"""
Detectors for the two timing defects a T5a simulator must never have
(spec F1.1 and section 8 (g)):

- same-close execution: holdings earning a return dated at or before the
  close at which they were bought (lag 0 on close-based decisions);
- a caller that lets a decision see the return its holdings will earn
  (r_next, or any later row).

Each detector reaches the code under test through a seam (an
`EpisodeRunner`, a target collector), so one check runs on the oracle, the
engine and the RL environment alike.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from reference.replay import EpisodeRunner, ReplayScenario

Policy = Callable[[list[list[float]]], Sequence[float] | None]
Collector = Callable[[Policy, np.ndarray], Mapping[int, Sequence[float] | None]]


def switch_scenario(jump: float, *, lag: int = 1) -> ReplayScenario:
    """
    D1's switch on assets (A, B, cash), sessions 0..3, no costs. Decision 0
    builds into B (always flat); decision 1 switches to 100% A. A returns
    `jump` at session 2 and -5% at session 3. With lag 1 the switch executes
    at close 2, so A's session-2 return is never earned: W = 0.950 for any
    `jump`. A same-close engine earns it: W = 0.95 (1 + jump), 1.045 at +10%.
    """

    nan = float("nan")
    returns = np.array(
        [
            [nan, nan, nan],
            [0.0, 0.0, 0.0],
            [jump, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
        ]
    )
    return ReplayScenario(
        returns=returns,
        targets={0: [0.0, 1.0, 0.0], 1: [1.0, 0.0, 0.0]},
        cash_index=2,
        cost_rate=[0.0, 0.0, 0.0],
        lag=lag,
    )


def final_wealth(runner: EpisodeRunner, scenario: ReplayScenario) -> float:
    path = runner(scenario)
    known = [value for value in path.wealth if value is not None]
    return known[-1] / scenario.initial_wealth


def earns_before_execution(runner: EpisodeRunner, *, tolerance: float = 1e-12) -> bool:
    """
    True when the runner's switch result depends on A's return dated at the
    decision's next session, i.e. the holdings earned a return from before
    their execution close.
    """

    up = final_wealth(runner, switch_scenario(0.10))
    down = final_wealth(runner, switch_scenario(-0.30))
    return abs(up - down) > tolerance


def peeking_policy(cash_index: int) -> Policy:
    """
    A policy sensitive to the newest row it is shown: weights proportional to
    exp(50 x latest return) over the non-cash assets, nothing in cash.
    """

    def policy(history: list[list[float]]) -> list[float]:
        latest = history[-1]
        scores = [
            0.0 if i == cash_index or not math.isfinite(r) else math.exp(50.0 * r)
            for i, r in enumerate(latest)
        ]
        if sum(scores) == 0.0:
            scores = [0.0 if i == cash_index else 1.0 for i in range(len(latest))]
        total = sum(scores)
        return [score / total for score in scores]

    return policy


def future_sensitive_decisions(
    collect: Collector, policy: Policy, returns: np.ndarray
) -> list[int]:
    """
    Decision sessions whose target changes when every return after the
    decision is replaced. Empty means the caller is causal for this policy
    and these returns; the check needs a policy sensitive to the newest row
    (`peeking_policy`) and returns that differ across assets.
    """

    returns = np.asarray(returns, dtype=float)
    base = collect(policy, returns)
    violations = []
    for decision, target in sorted(base.items()):
        perturbed = returns.copy()
        perturbed[decision + 1 :] = -perturbed[decision + 1 :] - 0.01
        changed = collect(policy, perturbed)[decision]
        if (target is None) != (changed is None) or (
            target is not None and not np.allclose(target, changed, rtol=0.0, atol=1e-15)
        ):
            violations.append(decision)
    return violations
