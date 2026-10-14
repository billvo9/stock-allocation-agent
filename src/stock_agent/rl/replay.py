"""
Rollouts and their records: one function runs every policy on the environment.

`rollout(env, policy)` is the only way the comparison harness runs a policy,
so rule baselines, the optimizer, learned policies and PPO all reach the
transition through the same `AllocationEnv.step`. A recorded target
sequence replays exactly through `ScriptedPolicy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_agent.rl.env import AllocationEnv, LedgerRow
from stock_agent.rl.observation import Observation


@dataclass(frozen=True)
class DecisionRecord:
    decision_index: int
    decision_session: int
    target: np.ndarray | None
    reward: float
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeRecord:
    policy: str
    identity: dict
    decisions: tuple[DecisionRecord, ...]
    ledger: tuple[LedgerRow, ...]
    wealth_path: np.ndarray

    @property
    def total_reward(self) -> float:
        return float(sum(decision.reward for decision in self.decisions))

    def targets(self) -> list[np.ndarray | None]:
        return [decision.target for decision in self.decisions]

    def summary(self) -> dict[str, object]:
        """
        Engineering facts about the episode, measured from the first
        execution close (where every policy starts from cash, decision D3).
        Not a strategy comparison: costs, turnover and wealth come from the
        injected transition and inherit its approval status.
        """

        first_execution = self.decisions[0].decision_index + 1
        wealth = self.wealth_path[first_execution:] / self.wealth_path[first_execution]
        peaks = np.maximum.accumulate(wealth)
        traded = [row for row in self.ledger if row.target is not None]
        return {
            "policy": self.policy,
            "transition_status": self.identity["transition"]["status"],
            "n_decisions": len(self.decisions),
            "n_trades": sum(row.cost > 0 for row in self.ledger),
            "terminal_wealth": float(wealth[-1]),
            "log_growth": self.total_reward,
            "max_drawdown": float(np.max(1.0 - wealth / peaks)),
            "total_cost": float(sum(row.cost * row.wealth_before for row in self.ledger)),
            "mean_turnover_one_way": float(np.mean([row.turnover_one_way for row in traded]))
            if traded
            else 0.0,
            "fallback_decisions": sum(bool(d.fallback_reasons) for d in self.decisions),
        }


class ScriptedPolicy:
    """Replays a recorded target sequence (None = hold)."""

    name = "scripted"

    def __init__(self, targets: list[np.ndarray | None]) -> None:
        self.targets = list(targets)
        self._next = 0

    def reset(self) -> None:
        self._next = 0

    def decide(self, observation: Observation) -> np.ndarray | None:
        if self._next >= len(self.targets):
            raise ValueError("The script has no target left for this decision.")
        target = self.targets[self._next]
        self._next += 1
        return None if target is None else np.array(target, dtype=np.float64)


def rollout(env: AllocationEnv, policy) -> EpisodeRecord:
    policy.reset()
    observation, info = env.reset()
    decisions: list[DecisionRecord] = []
    done = False
    while not done:
        target = policy.decide(observation)
        reasons = tuple(getattr(policy, "last_reasons", ()))
        stored = None if target is None else np.array(target, dtype=np.float64)
        next_observation, reward, terminated, truncated, next_info = env.step(stored)
        decisions.append(
            DecisionRecord(
                decision_index=info["decision_index"],
                decision_session=info["decision_session"],
                target=stored,
                reward=float(reward),
                fallback_reasons=reasons,
            )
        )
        observation, info = next_observation, next_info
        done = terminated or truncated
    return EpisodeRecord(
        policy=policy.name,
        identity=env.identity(),
        decisions=tuple(decisions),
        ledger=tuple(env.ledger),
        wealth_path=env.wealth_path,
    )
