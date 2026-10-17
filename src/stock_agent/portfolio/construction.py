"""
The initial construction (the build) under a registered constraint set: what
the first executed target actually holds, and when the portfolio is fully
built.

The owner's 2026-10-17 confirmation of D5 (verbatim, recorded in
`docs/research/t5a_open_decisions.md`): "An explicitly registered turnover
cap also applies to the initial build. Report actual executed weights,
uninvested cash and any incomplete construction so results are not
misleading."

`build_report(trades, cash_index)` reads the D5 feasibility records of the
executed targets in execution order (the engine's `SimulationResult.trades`,
or the RL environment's ledger rows) and reports:

- the build: the first executed target, at its execution close, with its
  requested and executed weights (after the D5 map) and the uninvested cash
  it leaves: the executed cash weight, which holds the cap's excess, the part
  of a turnover-limited build not yet invested, and any cash the target asked
  for;
- `incomplete_construction`: the turnover limit bound at the build, so the
  executed weights fall short of the (capped) target;
- the first execution, at or after the build, where the turnover limit did
  not bind: the portfolio first reached a target it asked for. Its session,
  and the sessions from the build to it, are None when the limit bound at
  every execution through the end of the data (never fully built), for
  example a portfolio that trades only once.

Pure reporting: no weight, cost or formula changes here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from stock_agent.portfolio.transition import TURNOVER_LIMIT, FeasibleTarget


@dataclass(frozen=True)
class BuildReport:
    """
    - `session`: the build's execution close.
    - `requested`, `executed`: the build target as the policy asked for it,
      and as executed after the D5 map.
    - `uninvested_cash`: the executed cash weight at the build.
    - `binding`, `alpha`: the constraints that changed the build, and the
      fraction of the trade executed under the turnover limit.
    - `incomplete_construction`: True when the turnover limit left the build
      partial.
    - `fully_built_session`: the first execution at or after the build whose
      turnover limit did not bind, or None (never fully built in the data).
    - `sessions_until_fully_built`: `fully_built_session - session`, or None.
    """

    session: int
    requested: tuple[float, ...]
    executed: tuple[float, ...]
    uninvested_cash: float
    binding: tuple[str, ...]
    alpha: float
    incomplete_construction: bool
    fully_built_session: int | None
    sessions_until_fully_built: int | None

    def __post_init__(self) -> None:
        if len(self.requested) != len(self.executed):
            raise ValueError("requested and executed must have the same length")
        if (self.fully_built_session is None) != (self.sessions_until_fully_built is None):
            raise ValueError("a fully built session and its distance go together")
        if self.fully_built_session is not None and self.fully_built_session < self.session:
            raise ValueError("the build cannot be complete before it starts")
        if self.incomplete_construction != (TURNOVER_LIMIT in self.binding):
            raise ValueError("construction is incomplete exactly when the turnover limit binds")
        if not self.incomplete_construction and self.sessions_until_fully_built != 0:
            raise ValueError("a build the turnover limit did not bind is complete at once")

    def to_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "requested_weights": list(self.requested),
            "executed_weights": list(self.executed),
            "uninvested_cash": self.uninvested_cash,
            "binding": list(self.binding),
            "alpha": self.alpha,
            "incomplete_construction": self.incomplete_construction,
            "fully_built_session": self.fully_built_session,
            "sessions_until_fully_built": self.sessions_until_fully_built,
        }


def build_report(
    trades: Sequence[tuple[int, FeasibleTarget]], cash_index: int
) -> BuildReport | None:
    """
    The build report of the executed targets `trades`, each (execution
    session, D5 feasibility record), in execution order; None when no target
    was ever executed (the portfolio stayed in cash).
    """

    if not trades:
        return None
    sessions = [int(session) for session, _ in trades]
    if sessions != sorted(sessions):
        raise ValueError("trades must be in execution order")
    session, build = trades[0]
    if not 0 <= cash_index < build.weights.size:
        raise ValueError(f"cash_index {cash_index} is outside the {build.weights.size} assets")
    complete = next(
        (int(at) for at, record in trades if TURNOVER_LIMIT not in record.binding), None
    )
    return BuildReport(
        session=int(session),
        requested=tuple(float(w) for w in build.requested),
        executed=tuple(float(w) for w in build.weights),
        uninvested_cash=float(build.weights[cash_index]),
        binding=tuple(build.binding),
        alpha=float(build.alpha),
        incomplete_construction=TURNOVER_LIMIT in build.binding,
        fully_built_session=complete,
        sessions_until_fully_built=None if complete is None else complete - int(session),
    )
