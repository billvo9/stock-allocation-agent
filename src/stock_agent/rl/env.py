"""
The allocation environment: a scheduler around an injected transition.

The environment holds no economics. It decides WHEN the injected
`portfolio_step` is called and with which inputs; every return, cost and
drift comes from that function, and the reward from `rl.reward`.

Timeline. The environment works on the window's complete sessions
c_0 < c_1 < ... < c_{J-1} (every asset has a close). One transition call per
complete session c_m books the trade at the close of c_m (if one executes
there) and earns each asset's span return over (c_m, c_{m+1}]:

- Decisions (D1, L = 1, with D6): a decision at c_j uses information at the
  close of c_j, executes at the close of c_{j+1} and first earns
  (c_{j+1}, c_{j+2}]. No decision is taken on a gap session; a gap between
  complete sessions is held through (share counts fixed) and valued once,
  by the span return, at the next complete session.
  This is the universe-complete form of D6: a trade waits for a session where
  EVERY asset has a close. It equals D6 whenever a policy holds or targets
  every asset (all softmax policies do) and is stricter otherwise, but the
  same for every compared policy.
- Schedule (D7): decisions at indices j >= o with (j - o) mod k = 0, the
  last one at J - 3 so its trade still earns one span.
- Start (D3): 100% cash. The first execution close pays the build.

Episode accounting. reset() books calls c_0..c_o (cash; identical for every
policy) and returns the observation at c_o. step(target) books the calls
c_{j+1}..c_{j'} up to the next decision j' and returns their summed log
growth: everything the action at c_j influences. The call at c_{j'} is
booked before the observation at c_{j'} is built, but the observation reads
only rows <= c_{j'} of the market data, the post-trade weights at c_{j'}
(which do not depend on the next span's returns) and wealth marked at
closes <= c_{j'}. A test proves the observation is invariant to every
return after c_{j'}.

The window end is a time limit (truncated=True): the investor would carry
on, so learners may bootstrap from the final observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_agent.rl.observation import (
    Observation,
    ObservationBuilder,
    ScenarioLoader,
    snapshot_at,
)
from stock_agent.rl.reward import REWARD_INFO, log_growth_reward
from stock_agent.rl.scenarios import ScenarioData, generate_scenario
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import PortfolioStep, TransitionInfo, check_step_result


@dataclass(frozen=True)
class EnvConfig:
    rebalance_every: int = 1
    rebalance_offset: int = 0
    validate_transition: bool = True

    def __post_init__(self) -> None:
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be at least 1.")
        if not 0 <= self.rebalance_offset < self.rebalance_every:
            raise ValueError("rebalance_offset must be in [0, rebalance_every).")

    def to_dict(self) -> dict[str, object]:
        return {
            "rebalance_every": self.rebalance_every,
            "rebalance_offset": self.rebalance_offset,
            "validate_transition": self.validate_transition,
        }


@dataclass(frozen=True)
class LedgerRow:
    """One transition call: the close of `session`, earning to `next_session`."""

    index: int
    session: int
    next_session: int
    decision_session: int | None
    target: np.ndarray | None
    w_pre: np.ndarray
    w_post_trade: np.ndarray
    w_next: np.ndarray
    cost_rates: np.ndarray
    gross_return: float
    cost: float
    net_return: float
    turnover_one_way: float
    traded_notional: float
    wealth_before: float
    wealth_after: float
    reward: float


class AllocationEnv:
    """Gymnasium-style reset/step over one scenario window (no gymnasium import)."""

    def __init__(
        self,
        data: ScenarioData,
        *,
        start: int,
        stop: int,
        transition: PortfolioStep,
        transition_info: TransitionInfo,
        builder: ObservationBuilder,
        config: EnvConfig | None = None,
    ) -> None:
        self.data = data
        self.start, self.stop = int(start), int(stop)
        self.transition = transition
        self.transition_info = transition_info
        self.builder = builder
        self.config = config or EnvConfig()
        if builder.asset_names != data.asset_names or builder.cash_index != data.cash_index:
            raise ValueError("The observation builder was made for a different universe.")
        self.complete = data.complete_sessions(self.start, self.stop)
        if len(self.complete) < 3:
            raise ValueError("An episode window needs at least three complete sessions.")
        self.first_decision_index = self.config.rebalance_offset
        if self.first_decision_index > len(self.complete) - 3:
            raise ValueError("The rebalance offset leaves no decision inside the window.")
        self._ready = False

    @property
    def n_assets(self) -> int:
        return self.data.n_assets

    @property
    def last_call_index(self) -> int:
        return len(self.complete) - 2

    @property
    def wealth_path(self) -> np.ndarray:
        """Marked wealth at the closes c_0 .. c_m booked so far (W = 1 at c_0)."""

        return np.asarray(self._wealth, dtype=np.float64)

    def identity(self) -> dict[str, object]:
        return {
            "scenario": self.data.identity(),
            "window": [self.start, self.stop],
            "config": self.config.to_dict(),
            "transition": self.transition_info.to_dict(),
            "reward": REWARD_INFO.to_dict(),
        }

    def reset(self) -> tuple[Observation, dict]:
        cash = np.zeros(self.n_assets)
        cash[self.data.cash_index] = 1.0
        self._w_pre = cash
        self._wealth = [1.0]
        self.ledger: list[LedgerRow] = []
        self._post_trade: list[np.ndarray] = []
        for m in range(self.first_decision_index + 1):
            self._book(m, target=None, decision_session=None)
        self._j = self.first_decision_index
        self._ready = True
        return self._observe(self._j), self._info(self._j, None)

    def step(self, target: np.ndarray | None) -> tuple[Observation, float, bool, bool, dict]:
        if not self._ready:
            raise RuntimeError("The episode has ended (or never started); call reset().")
        j = self._j
        candidate = j + self.config.rebalance_every
        next_decision = candidate if candidate <= len(self.complete) - 3 else None
        last = self.last_call_index if next_decision is None else next_decision
        decided_at = int(self.complete[j])
        reward = 0.0
        for m in range(j + 1, last + 1):
            first = m == j + 1
            row = self._book(
                m,
                target=target if first else None,
                decision_session=decided_at if first else None,
            )
            reward += row.reward
        truncated = next_decision is None
        self._j = last
        self._ready = not truncated
        info = self._info(last, j)
        return self._observe(last), reward, False, truncated, info

    def _book(self, m: int, *, target: np.ndarray | None, decision_session: int | None):
        session, next_session = int(self.complete[m]), int(self.complete[m + 1])
        rates = self.data.cost_rates_at(session)
        target_array = None if target is None else np.asarray(target, dtype=np.float64).copy()
        result = self.transition(
            self._w_pre,
            target_array,
            self.data.span_returns(session, next_session),
            rates,
            self.data.cash_index,
        )
        if self.config.validate_transition:
            check_step_result(result, n_assets=self.n_assets)
        wealth_before = self._wealth[-1]
        wealth_after = wealth_before * (1.0 + result.net_return)
        row = LedgerRow(
            index=m,
            session=session,
            next_session=next_session,
            decision_session=decision_session,
            target=target_array,
            w_pre=self._w_pre,
            w_post_trade=np.asarray(result.w_post_trade, dtype=np.float64),
            w_next=np.asarray(result.w_next, dtype=np.float64),
            cost_rates=rates,
            gross_return=float(result.gross_return),
            cost=float(result.cost),
            net_return=float(result.net_return),
            turnover_one_way=float(result.turnover_one_way),
            traded_notional=float(result.traded_notional),
            wealth_before=wealth_before,
            wealth_after=wealth_after,
            reward=log_growth_reward(result),
        )
        self.ledger.append(row)
        self._post_trade.append(row.w_post_trade)
        self._wealth.append(wealth_after)
        self._w_pre = row.w_next
        return row

    def _observe(self, j: int) -> Observation:
        marked = np.asarray(self._wealth[: j + 1])
        drawdown = float(1.0 - marked[-1] / marked.max())
        return self.builder.build(
            snapshot_at(self.data, int(self.complete[j])), self._post_trade[j], drawdown
        )

    def _info(self, j: int, previous: int | None) -> dict:
        info = {
            "decision_index": j,
            "decision_session": int(self.complete[j]),
            "terminal": not self._ready,
            "transition": self.transition_info.to_dict(),
            "reward": REWARD_INFO.to_dict(),
        }
        if previous is not None:
            executed = self.ledger[previous + 1]
            info.update(
                executed_session=executed.session,
                cost=executed.cost,
                turnover_one_way=executed.turnover_one_way,
                sessions_spanned=int(self.complete[j + 1] - executed.session),
            )
        return info


def fold_envs(
    fold: FoldSpec,
    *,
    transition: PortfolioStep,
    transition_info: TransitionInfo,
    builder: ObservationBuilder,
    config: EnvConfig | None = None,
    loader: ScenarioLoader = generate_scenario,
) -> list[AllocationEnv]:
    """One environment per (scenario, seed) of the fold, over the fold's window."""

    return [
        AllocationEnv(
            loader(scenario_id, seed),
            start=fold.start,
            stop=fold.stop,
            transition=transition,
            transition_info=transition_info,
            builder=builder,
            config=config,
        )
        for scenario_id, seed in fold.episodes()
    ]
