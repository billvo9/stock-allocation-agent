"""
T5a engine: the caller that turns decisions and returns into a wealth path
through `portfolio_step` (spec F8.1 with F1.1, F2.5, F7.1; D1-D8 approved by
the owner on 2026-10-15). `evaluation.baseline.run_rebalanced_backtest` runs
on `simulate`, and the RL environment reproduces its wealth from the same
targets and returns (the replay-equivalence tests).

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
  `collect_targets`, or None to keep the drifted holdings. The earliest
  decision is the build from 100% cash (D3).

Rules (D1, D3, D5, D6):

- Decision d executes at close d + lag. The build executes at the first
  decision's execution close from 100% cash and pays the D2 cost; there is
  no exit charge.
- Every target passes through `feasible_target` at its execution close.
- A rebalance waits for the next session where every held or targeted asset
  has a close (`UNEXECUTABLE_MISSING_CLOSE`). A newer target decided while
  one waits supersedes it (`SUPERSEDED_DEFERRED_TARGET`); a newer None (a
  hold, not a target) leaves the waiting target in place.
- Holdings are valued at each close where every held asset has one. Across
  a gap the share counts are held and the span's growth is booked in one
  `portfolio_step` call at the first close valued again; wealth at the
  sessions inside the gap is None, never carried flat.
- A held asset whose close has no available return stops the run
  (`DISTRIBUTION_IN_GAP`).

Wealth is the pre-trade mark-to-market value at each close from the build's
execution close onward, None where a held asset has no close.

Cash (D4 (a), approved for synthetic engineering only). Callers supply the
cash column as `UNINVESTED_CASH_RETURN` (0%) and label results with
`CASH_RETURN_LABEL`. The engine itself books whatever cash return it is
given, so its mechanics are tested with nonzero rates too; a real cash rate
is group C and is not approved.

Causality: `simulate` trusts the targets it is given. Only `collect_targets`
enforces that a decision at d sees nothing after close d, so targets built
any other way (an optimizer, an RL policy) must be checked at their own
source, for example with the detectors in `tests/reference/timing.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from stock_agent.portfolio.descriptive import DISTRIBUTION_IN_GAP
from stock_agent.portfolio.schedule import (
    BUILD,
    NO_EXECUTION_SESSION,
    REBALANCE,
    SUPERSEDED_DEFERRED_TARGET,
    UNEXECUTABLE_MISSING_CLOSE,
    ScheduledExecution,
    ScheduleSpec,
    decision_sessions,
    next_executable_session,
)
from stock_agent.portfolio.transition import (
    Constraints,
    CostModel,
    FeasibleTarget,
    StepResult,
    Transition,
    check_target,
    feasible_target,
    portfolio_step,
)

COMPLETE = "complete"
STOPPED = "stopped"

# D4 (a), approved 2026-10-15 for synthetic engineering only.
UNINVESTED_CASH_RETURN = 0.0
CASH_RETURN_LABEL = "uninvested cash earns 0% (synthetic engineering only; group C pending)"

# A rules policy: given the return history through close d (rows 0..d, a
# copy), the requested target, or None to keep the drifted holdings.
Policy = Callable[[np.ndarray], np.ndarray | None]


@dataclass(frozen=True, eq=False)
class TradeRecord:
    """One executed target: the decision, its execution close, and the D5 record."""

    decision: int
    session: int
    feasibility: FeasibleTarget


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """
    - `sessions`, `wealth`: the valued closes and the pre-trade wealth at
      each (None where unavailable).
    - `executions`: every decision with the session it executed at after
      any deferral, or why it did not, in decision order.
    - `steps`: the `portfolio_step` results in order, one per valued span.
    - `status`: `COMPLETE`, or `STOPPED` with `reason` (for example
      `DISTRIBUTION_IN_GAP`).
    - `spans`: each step's (start, end) closes, aligned with `steps`.
    - `trades`: every executed target with its feasibility record (the
      distance between the requested and the executed target).
    """

    sessions: tuple[int, ...]
    wealth: tuple[float | None, ...]
    executions: tuple[ScheduledExecution, ...]
    steps: tuple[StepResult, ...]
    status: str
    reason: str | None = None
    spans: tuple[tuple[int, int], ...] = ()
    trades: tuple[TradeRecord, ...] = ()

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
        if self.spans and len(self.spans) != len(self.steps):
            raise ValueError("spans and steps must align")

    @property
    def final_wealth(self) -> float | None:
        """The wealth at the last valued session (None when it is unavailable)."""

        return self.wealth[-1] if self.wealth else None


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

    table = np.asarray(returns, dtype=float)
    targets: dict[int, np.ndarray | None] = {}
    for decision in decision_sessions(schedule, len(table)):
        target = policy(table[: decision + 1].copy())
        targets[decision] = None if target is None else np.array(target, dtype=float)
    return targets


def _table(returns: np.ndarray) -> np.ndarray:
    table = np.array(returns, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 2:
        raise ValueError("returns must be a (sessions, assets) table over at least two assets")
    return table


def _close_mask(table: np.ndarray, has_close: np.ndarray | None) -> np.ndarray:
    if has_close is None:
        closes = np.isfinite(table)
        closes[0] = True
        return closes
    closes = np.array(has_close, dtype=bool)
    if closes.shape != table.shape:
        raise ValueError("has_close must have the shape of returns")
    orphan = np.isfinite(table[1:]) & ~closes[1:]
    if orphan.any():
        t, i = np.argwhere(orphan)[0]
        raise ValueError(f"asset {i} has a return but no close at session {t + 1}")
    return closes


def _span_returns(table: np.ndarray, closes: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Each asset's return from its close at `start` to its close at `end`: the
    chained returns of its closes in (start, end] (a gap's return is already
    in the next close's row). NaN for an asset without a close at `end`.
    """

    if end == start + 1:
        span = table[end].copy()
    else:
        window = np.where(closes[start + 1 : end + 1], 1.0 + table[start + 1 : end + 1], 1.0)
        span = np.prod(window, axis=0) - 1.0
    span[~closes[end]] = np.nan
    return span


def _held(weights: np.ndarray, cash: int) -> set[int]:
    return {int(i) for i in np.flatnonzero(weights > 0) if i != cash}


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

    Every step goes through `transition` and the engine uses its result: no
    cost, return or drift formula is computed here (F8.1, one function).
    The targets are trusted as given; causality is `collect_targets`' job.
    """

    table = _table(returns)
    n_rows, n = table.shape
    last = n_rows - 1
    closes = _close_mask(table, has_close)
    cash = constraints.cash_index
    if cash >= n:
        raise ValueError(f"cash_index {cash} is outside the {n} assets")
    rates = np.array(cost_rate, dtype=float)
    if rates.shape != (n,):
        raise ValueError(f"cost_rate must be a vector over {n} assets")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError("lag must be a non-negative int")
    wealth = float(initial_wealth)
    if not np.isfinite(wealth) or wealth <= 0:
        raise ValueError("initial_wealth must be positive and finite")

    plan: dict[int, np.ndarray | None] = {}
    for decision, target in targets.items():
        if isinstance(decision, bool) or not isinstance(decision, int | np.integer):
            raise TypeError("decision sessions must be ints")
        if not 0 <= decision < n_rows:
            raise ValueError(f"decision session {decision} is outside the data")
        plan[int(decision)] = None if target is None else check_target(target, n)
    if not plan:
        raise ValueError("no decisions")
    build = min(plan)
    start = build + lag
    if start > last - 1:
        raise ValueError("the build cannot execute inside the data")
    if not closes[start:, cash].all() or not np.isfinite(table[start + 1 :, cash]).all():
        raise ValueError("the cash account needs a finite return at every session")
    due = {decision + lag: decision for decision in plan}

    def kind(decision: int) -> str:
        return BUILD if decision == build else REBALANCE

    def closed(session: int, asset: int) -> bool:
        return bool(closes[session, asset])

    weights = np.zeros(n)
    weights[cash] = 1.0
    anchor, trade = start, None
    pending: tuple[int, int] | None = None  # (decision, planned execution)
    sessions, path = [start], [wealth]
    steps, spans, trades = [], [], []
    executions: dict[int, ScheduledExecution] = {}
    status, reason = COMPLETE, None

    def decide(session: int, valued: bool) -> None:
        nonlocal pending, trade
        if session in due:
            decision = due[session]
            if plan[decision] is None:
                executions[decision] = ScheduledExecution(decision, session, kind(decision))
            else:
                if pending is not None:
                    executions[pending[0]] = ScheduledExecution(
                        pending[0], None, kind(pending[0]), SUPERSEDED_DEFERRED_TARGET
                    )
                pending = (decision, session)
        if pending is None:
            return
        decision, planned = pending
        if session == last:
            executions[decision] = ScheduledExecution(
                decision, None, kind(decision), NO_EXECUTION_SESSION
            )
            pending = None
            return
        if not valued:
            return
        involved = _held(weights, cash) | _held(plan[decision], cash)
        if next_executable_session(session, involved, closed, n_rows) != session:
            return
        trade = feasible_target(plan[decision], weights, constraints)
        trades.append(TradeRecord(decision, session, trade))
        executions[decision] = ScheduledExecution(
            decision,
            session,
            kind(decision),
            None if session == planned else UNEXECUTABLE_MISSING_CLOSE,
        )
        pending = None

    decide(start, True)
    for session in range(start + 1, last + 1):
        held = _held(weights if trade is None else trade.weights, cash)
        if any(closes[session, i] and not np.isfinite(table[session, i]) for i in held):
            status, reason = STOPPED, DISTRIBUTION_IN_GAP
            break
        valued = all(closes[session, i] for i in held)
        if valued:
            step = transition(
                weights.copy(),
                None if trade is None else np.array(trade.weights),
                _span_returns(table, closes, anchor, session),
                rates.copy(),
                cash,
            )
            steps.append(step)
            spans.append((anchor, session))
            wealth *= 1.0 + step.net_return
            weights = np.array(step.w_next, dtype=float)
            anchor, trade = session, None
        sessions.append(session)
        path.append(wealth if valued else None)
        decide(session, valued)

    if status == COMPLETE:
        for planned, decision in due.items():
            if planned > last:
                executions[decision] = ScheduledExecution(
                    decision, None, kind(decision), NO_EXECUTION_SESSION
                )
    return SimulationResult(
        sessions=tuple(sessions),
        wealth=tuple(path),
        executions=tuple(executions[d] for d in sorted(executions)),
        steps=tuple(steps),
        status=status,
        reason=reason,
        spans=tuple(spans),
        trades=tuple(trades),
    )


def simulate_cost_sensitivity(
    returns: np.ndarray,
    targets: Mapping[int, np.ndarray | None],
    *,
    assets: Sequence[str],
    cost_model: CostModel,
    lag: int,
    constraints: Constraints,
    initial_wealth: float,
    has_close: np.ndarray | None = None,
    transition: Transition = portfolio_step,
) -> dict[float, SimulationResult]:
    """
    The same targets and returns at 0x, 1x and 2x the registered cost rates
    (D2): one simulation per multiplier, keyed by the multiplier.
    """

    return {
        model.multiplier: simulate(
            returns,
            targets,
            lag=lag,
            cost_rate=model.rates_for(assets),
            constraints=constraints,
            initial_wealth=initial_wealth,
            has_close=has_close,
            transition=transition,
        )
        for model in cost_model.sensitivity()
    }
