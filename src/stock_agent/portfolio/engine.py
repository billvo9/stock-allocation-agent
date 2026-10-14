"""
T5a engine contract: the caller that turns decisions and returns into a
wealth path through `portfolio_step` (spec F8.1 with F1.1, F2.5, F7.1).

STATUS: INTERFACE ONLY. OD8 group B (D1-D8 in
`docs/research/t5a_open_decisions.md`) awaits the owner's approval, so every
function body raises `NotImplementedError`. After approval,
`evaluation.baseline.run_rebalanced_backtest` moves onto `simulate`, and the
RL environment must reproduce its wealth from the same targets and returns
(the replay-equivalence harness in `tests/reference/replay.py`).

Inputs (sessions are integer positions 0..T):

- `returns[t, i]`: asset i's return from its previous close to close t
  ("dated t"); the cash account's column is its session return. Row 0 is
  unused. NaN where the asset has no close at t; after such a gap the row
  holds the labelled multi-session return from the last close
  (`portfolio.descriptive.total_returns`, `sessions_spanned`).
- `has_close[t, i]`: whether asset i closed at t. Default: the finite
  entries of `returns`, with row 0 all True. A close whose return is
  unavailable (a distribution inside a gap) stops a held position with
  `DISTRIBUTION_IN_GAP`.
- `targets[d]`: the requested target decided at session d, from
  `collect_targets`. The earliest decision is the build from 100% cash.

Wealth is the pre-trade mark-to-market value at each close from the build's
execution close onward, None where a held asset has no close (never carried
flat).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from stock_agent.portfolio.descriptive import DISTRIBUTION_IN_GAP
from stock_agent.portfolio.schedule import ScheduledExecution, ScheduleSpec
from stock_agent.portfolio.transition import (
    PENDING_APPROVAL,
    Constraints,
    StepResult,
    Transition,
    portfolio_step,
)

COMPLETE = "complete"
STOPPED = "stopped"

# A rules policy: given the return history through close d (rows 0..d, a
# copy), the requested target, or None to keep the drifted holdings.
Policy = Callable[[np.ndarray], np.ndarray | None]


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """
    - `sessions`, `wealth`: the valued closes and the pre-trade wealth at
      each (None where unavailable).
    - `executions`: every decision with the session it executed at after
      any deferral, or why it did not.
    - `steps`: the `portfolio_step` results in order, one per valued span.
    - `status`: `COMPLETE`, or `STOPPED` with `reason` (for example
      `DISTRIBUTION_IN_GAP`).
    """

    sessions: tuple[int, ...]
    wealth: tuple[float | None, ...]
    executions: tuple[ScheduledExecution, ...]
    steps: tuple[StepResult, ...]
    status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.sessions) != len(self.wealth):
            raise ValueError("sessions and wealth must have the same length")
        if list(self.sessions) != sorted(set(self.sessions)):
            raise ValueError("sessions must be strictly increasing")
        if self.status not in {COMPLETE, STOPPED}:
            raise ValueError(f"unknown status {self.status!r}")
        if (self.status == STOPPED) != (self.reason is not None):
            raise ValueError("a stopped simulation, and only a stopped one, has a reason")
        if self.reason is not None and self.reason != DISTRIBUTION_IN_GAP:
            raise ValueError(f"unknown stop reason {self.reason!r}")


def collect_targets(
    policy: Policy,
    returns: np.ndarray,
    schedule: ScheduleSpec,
) -> dict[int, np.ndarray | None]:
    """
    Call `policy` once per decision session d of `schedule`, giving it a copy
    of `returns[: d + 1]` only, so no decision can see the return its
    holdings will earn.
    """

    raise NotImplementedError(PENDING_APPROVAL)


def simulate(
    returns: np.ndarray,
    targets: Mapping[int, np.ndarray | None],
    *,
    lag: int,
    cost_rate: np.ndarray,
    constraints: Constraints,
    initial_wealth: float,
    has_close: np.ndarray | None = None,
    transition: Transition = portfolio_step,
) -> SimulationResult:
    """
    Build from 100% cash at the first decision's execution close, then
    execute each decision d at close d + lag (deferred past missing closes),
    pass every target through `feasible_target`, and value the holdings
    through `transition`.
    """

    raise NotImplementedError(PENDING_APPROVAL)
