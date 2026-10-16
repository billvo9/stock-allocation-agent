"""
T5a schedule contract: when decisions are taken and when they execute
(D1 execution lag, D6 deferral, D7 rebalance offsets; spec F1.1, F1.3,
F7.1 and F7.2).

STATUS: APPROVED by the owner on 2026-10-15 (D1-D8 in
`docs/research/t5a_open_decisions.md`).

Sessions are integer positions on the trading calendar (0, 1, 2, ...); the
caller maps positions to calendar sessions, so nothing here reads a clock.

- A decision at session d uses only information available at the close
  of d. It executes at the close of d + lag; the new holdings first earn
  the return dated d + lag + 1 (D1). The lag is bound to the label's
  `entry_lag` (`execution_lag`).
- `first_decision` is the one common build decision: every compared
  portfolio and benchmark builds from 100% cash at first_decision + lag
  (D3, F7.1).
- Later rebalance decisions are the sessions d > first_decision with
  (d - first_decision) mod every == offset, with one registered offset
  shared by every compared portfolio (D7). All k offsets form one variant
  with a k-member sensitivity family (F7.2, `offset_family`).
- If a held or targeted asset has no close at the execution session, the
  whole rebalance waits for the next session where all of them have one
  (D6, `unexecutable_missing_close`). A newer target decided meanwhile
  supersedes the deferred one (`superseded_deferred_target`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection
from dataclasses import dataclass

from stock_agent.features.training import MODEL_LABEL_SPEC
from stock_agent.portfolio.descriptive import DISTRIBUTION_IN_GAP

BUILD = "build"
REBALANCE = "rebalance"
EXECUTION_KINDS = frozenset({BUILD, REBALANCE})

# Schedule and engine reasons (machine-readable).
UNEXECUTABLE_MISSING_CLOSE = "unexecutable_missing_close"  # deferred (D6)
NO_EXECUTION_SESSION = "no_execution_session"  # no close with a following session (F1 edge)
SUPERSEDED_DEFERRED_TARGET = "superseded_deferred_target"  # a newer target replaced it (D6)
REASONS = frozenset(
    {
        UNEXECUTABLE_MISSING_CLOSE,
        NO_EXECUTION_SESSION,
        SUPERSEDED_DEFERRED_TARGET,
        DISTRIBUTION_IN_GAP,
    }
)


@dataclass(frozen=True)
class ScheduleSpec:
    """
    One registered rebalance schedule.

    - `every`: k, the number of decision sessions between rebalances.
    - `offset`: o in [0, k), the registered phase (D7).
    - `lag`: L, execution sessions after the decision (D1).
    - `first_decision`: the common build decision session.
    """

    every: int
    offset: int
    lag: int
    first_decision: int = 0

    def __post_init__(self) -> None:
        for name in ("every", "offset", "lag", "first_decision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.every < 1:
            raise ValueError("every must be at least 1")
        if not 0 <= self.offset < self.every:
            raise ValueError("offset must lie in [0, every)")
        if self.lag < 0 or self.first_decision < 0:
            raise ValueError("lag and first_decision are non-negative")


@dataclass(frozen=True)
class ScheduledExecution:
    """
    One decision and where it executes. `execution` is None when it cannot
    execute inside the data; `reason` then says why. An executed decision
    may carry `UNEXECUTABLE_MISSING_CLOSE` when it ran later than planned.
    """

    decision: int
    execution: int | None
    kind: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in EXECUTION_KINDS:
            raise ValueError(f"unknown execution kind {self.kind!r}")
        if self.reason is not None and self.reason not in REASONS:
            raise ValueError(f"unknown schedule reason {self.reason!r}")
        if self.execution is not None and self.execution < self.decision:
            raise ValueError("an execution cannot precede its decision")
        if self.execution is None and self.reason is None:
            raise ValueError("an unexecuted decision needs a reason")


def _sessions(n_sessions: int) -> int:
    if isinstance(n_sessions, bool) or not isinstance(n_sessions, int) or n_sessions < 0:
        raise ValueError("n_sessions must be a non-negative int")
    return n_sessions


def execution_lag() -> int:
    """
    The registered execution lag L (D1, F1.3), bound to
    `MODEL_LABEL_SPEC.entry_lag` so backtests and labels agree.
    """

    return MODEL_LABEL_SPEC.entry_lag


def decision_sessions(spec: ScheduleSpec, n_sessions: int) -> tuple[int, ...]:
    """The build decision and every rebalance decision among sessions 0..n_sessions-1."""

    n_sessions = _sessions(n_sessions)
    first = spec.first_decision
    if first >= n_sessions:
        return ()
    later = (d for d in range(first + 1, n_sessions) if (d - first) % spec.every == spec.offset)
    return (first, *later)


def execution_plan(spec: ScheduleSpec, n_sessions: int) -> tuple[ScheduledExecution, ...]:
    """
    Every decision of `decision_sessions` with its planned execution session
    and kind, before any deferral. A decision executes only at a close with a
    following session, so one planned at or after the last session records
    `NO_EXECUTION_SESSION`.
    """

    plan = []
    for decision in decision_sessions(spec, n_sessions):
        kind = BUILD if decision == spec.first_decision else REBALANCE
        execution = decision + spec.lag
        if execution <= n_sessions - 2:
            plan.append(ScheduledExecution(decision, execution, kind))
        else:
            plan.append(ScheduledExecution(decision, None, kind, NO_EXECUTION_SESSION))
    return tuple(plan)


def offset_family(spec: ScheduleSpec) -> tuple[ScheduleSpec, ...]:
    """
    All k offsets of `spec`, sharing its build decision and lag: one variant
    with a k-member sensitivity family, never k independent trials (F7.2).
    """

    return tuple(dataclasses.replace(spec, offset=offset) for offset in range(spec.every))


def next_executable_session(
    planned: int,
    involved: Collection[int],
    has_close: Callable[[int, int], bool],
    n_sessions: int,
) -> int | None:
    """
    The first session at or after `planned` where every involved (held or
    targeted) asset has a close and a following session exists, or None (D6
    deferral).
    `has_close(session, asset)` reports whether the asset closed then.
    """

    n_sessions = _sessions(n_sessions)
    for session in range(planned, n_sessions - 1):
        if all(has_close(session, asset) for asset in involved):
            return session
    return None
