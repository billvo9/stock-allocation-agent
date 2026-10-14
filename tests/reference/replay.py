"""
Replay-equivalence harness: proves that two simulators (the backtest engine,
the RL environment driven by a scripted policy, the reference oracle)
produce the same wealth path from the same targets and returns.

A runner adapts one simulator to `EpisodeRunner`: it takes a
`ReplayScenario` and returns a `ReplayPath`. The scenario fixes everything
the economics depend on (returns, the targets decided at each session, the
lag, the cost rates and the constraints), so any difference in the paths is
a difference in the simulators. Comparison is exact up to `tolerance`,
relative to the reference's starting wealth.

Adapters: `oracle_runner` here; the engine adapter lives with the engine
tests; the RL environment adapter belongs with the RL tests (track C).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from reference import t5a_reference as oracle

TOLERANCE = 1e-12


@dataclass(frozen=True, eq=False)
class ReplayScenario:
    """
    - `returns`: (T+1, n), the engine's layout (row t is the return dated t;
      row 0 unused; NaN where an asset has no close).
    - `targets`: decision session to requested target (None keeps the
      drifted holdings). The earliest decision is the build from cash.
    - `has_close`: optional (T+1, n) close mask; default from `returns`.
    """

    returns: np.ndarray
    targets: Mapping[int, Sequence[float] | None]
    cash_index: int
    cost_rate: Sequence[float]
    lag: int
    initial_wealth: float = 1.0
    has_close: np.ndarray | None = None
    w_max: float = 1.0
    tau_max: float | None = None


@dataclass(frozen=True)
class ReplayPath:
    """Pre-trade wealth at each valued close (None where unavailable)."""

    sessions: tuple[int, ...]
    wealth: tuple[float | None, ...]
    status: str = oracle.COMPLETE
    reason: str | None = None


class EpisodeRunner(Protocol):
    def __call__(self, scenario: ReplayScenario) -> ReplayPath: ...


@dataclass(frozen=True)
class ReplayReport:
    equivalent: bool
    max_difference: float
    first_divergence: int | None = None
    detail: str = ""
    differences: tuple[float, ...] = field(default=(), repr=False)


def compare_paths(
    reference: ReplayPath, candidate: ReplayPath, *, tolerance: float = TOLERANCE
) -> ReplayReport:
    """Same sessions, same availability, same status, wealth within tolerance."""

    if reference.sessions != candidate.sessions:
        first = next(
            (a for a, b in zip(reference.sessions, candidate.sessions, strict=False) if a != b),
            min(len(reference.sessions), len(candidate.sessions)),
        )
        return ReplayReport(False, float("inf"), first, "the valued sessions differ")
    if (reference.status, reference.reason) != (candidate.status, candidate.reason):
        return ReplayReport(
            False,
            float("inf"),
            None,
            f"status {candidate.status}/{candidate.reason}, "
            f"expected {reference.status}/{reference.reason}",
        )
    known = [value for value in reference.wealth if value is not None]
    scale = abs(known[0]) if known else 1.0
    differences = []
    for session, a, b in zip(reference.sessions, reference.wealth, candidate.wealth, strict=True):
        if (a is None) != (b is None):
            return ReplayReport(False, float("inf"), session, "wealth availability differs")
        differences.append(0.0 if a is None else abs(a - b) / scale)
    worst = max(differences, default=0.0)
    first = next(
        (s for s, d in zip(reference.sessions, differences, strict=True) if d > tolerance),
        None,
    )
    detail = "" if first is None else f"wealth differs by {worst:.3e} (relative)"
    return ReplayReport(first is None, worst, first, detail, tuple(differences))


def replay(
    runners: Mapping[str, EpisodeRunner],
    scenario: ReplayScenario,
    *,
    reference: str,
    tolerance: float = TOLERANCE,
) -> dict[str, ReplayReport]:
    """Run every runner on `scenario` and compare each with `reference`."""

    expected = runners[reference](scenario)
    return {
        name: compare_paths(expected, runner(scenario), tolerance=tolerance)
        for name, runner in runners.items()
        if name != reference
    }


def oracle_runner(scenario: ReplayScenario) -> ReplayPath:
    result = oracle.run(
        scenario.returns,
        scenario.targets,
        lag=scenario.lag,
        cost_rate=scenario.cost_rate,
        cash_index=scenario.cash_index,
        initial_wealth=scenario.initial_wealth,
        has_close=scenario.has_close,
        w_max=scenario.w_max,
        tau_max=scenario.tau_max,
    )
    return ReplayPath(tuple(result.sessions), tuple(result.wealth), result.status, result.reason)
