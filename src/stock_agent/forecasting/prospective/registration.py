"""
The arithmetic of the proposed prospective registration rules
(docs/research/prospective_registration.md). Proposed only: nothing here is
active, records anything or reads data, and none of it is approved.

Session numbering. Session 1 is the first eligible close after registration.
Sessions 1..TRIAL_SESSIONS are the trial (a registered exclusion). The
confirmatory phase starts at session TRIAL_SESSIONS + 1 = 21.

Confirmatory outcomes, per fund and horizon h. The forecasts recorded at
sessions 21, 21 + h, 21 + 2h, ... (a phase fixed now), so no two outcome
windows overlap. A forecast at session s covers the h sessions after its
entry session s + L (L = the entry lag, 1), so it matures at the close of
s + L + h.

Checkpoints. Checkpoint c is the close of session TRIAL_SESSIONS + c with
c = 252 ("12 months") or 504 ("24 months"). Its outcomes are those matured
by then:

    n(h, c) = floor((TRIAL_SESSIONS + c - (TRIAL_SESSIONS + 1) - L - h) / h) + 1

A cell with fewer than MIN_OUTCOMES registered outcomes at a checkpoint is
not tested there. Which cells are tested is fixed now from these counts,
never from the realized data: the family at a checkpoint is every fund x
horizon cell with n(h, c) >= MIN_OUTCOMES, and a cell in the family whose
realized count falls short (missed sessions) enters Holm with p = 1.

Power (normal approximation, one-sided test of a mean loss differential d
with standard deviation s_d over n independent outcomes, at level a):

    power(delta, n, a) = Phi(delta sqrt(n) - z_{1-a}),      delta = E[d] / s_d
    MDE(n, a)          = (z_{1-a} + z_{0.8}) / sqrt(n)        (80% power)

The honest level for a cell is the first Holm step, a = ALPHA / m, with m
the family size at that checkpoint: the level a lone true effect must beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from stock_agent.model_diagnostics import inference as inf

TRIAL_SESSIONS = 20
CHECKPOINTS = {"12_months": 252, "24_months": 504}  # sessions after the trial's last session
MIN_OUTCOMES = 20
ALPHA_PER_CHECKPOINT = 0.025  # the family-wise level split equally over the two checkpoints
POWER_TARGET = 0.8
ENTRY_LAG = 1
FUNDS = ("VTI", "AGG", "SHV")
HORIZONS = (5, 20, 30)
TARGETS = ("log_realized_volatility", "total_log_return")
FORECASTS_PER_SESSION = len(FUNDS) * len(HORIZONS) * len(TARGETS)


def first_confirmatory_session() -> int:
    return TRIAL_SESSIONS + 1


def confirmatory_sessions(horizon: int, checkpoint: int, entry_lag: int = ENTRY_LAG) -> list[int]:
    """The registered forecast sessions of one cell whose outcomes matured by `checkpoint`."""

    last = TRIAL_SESSIONS + checkpoint
    first = first_confirmatory_session()
    return [s for s in range(first, last + 1, horizon) if s + entry_lag + horizon <= last]


def confirmatory_outcomes(horizon: int, checkpoint: int, entry_lag: int = ENTRY_LAG) -> int:
    return len(confirmatory_sessions(horizon, checkpoint, entry_lag))


def family(
    checkpoint: int,
    funds: Sequence[str] = FUNDS,
    horizons: Sequence[int] = HORIZONS,
) -> list[tuple[str, int]]:
    """The fund x horizon cells tested at a checkpoint, fixed by the registered counts."""

    return [
        (fund, h)
        for h in horizons
        for fund in funds
        if confirmatory_outcomes(h, checkpoint) >= MIN_OUTCOMES
    ]


def first_holm_level(checkpoint: int, tests_per_cell: int = 1) -> float | None:
    size = len(family(checkpoint)) * tests_per_cell
    return ALPHA_PER_CHECKPOINT / size if size else None


def power(effect: float, n: int, alpha: float) -> float:
    """One-sided power of a mean test at standardized effect E[d] / s_d (normal approx)."""

    return inf.normal_cdf(effect * math.sqrt(n) - inf.normal_ppf(1.0 - alpha))


def minimum_detectable_effect(n: int, alpha: float, target: float = POWER_TARGET) -> float:
    """The standardized effect E[d] / s_d detected with `target` power (normal approx)."""

    return (inf.normal_ppf(1.0 - alpha) + inf.normal_ppf(target)) / math.sqrt(n)


def outcomes_for_power(effect: float, alpha: float, target: float = POWER_TARGET) -> int | None:
    """The independent outcomes needed for `target` power at `effect` (None if effect <= 0)."""

    if not effect > 0:
        return None
    return math.ceil(((inf.normal_ppf(1.0 - alpha) + inf.normal_ppf(target)) / effect) ** 2)


def binomial_acceptance(n: int, p: float, alpha: float) -> tuple[int, int]:
    """
    The hit counts [low, high] an exact two-sided binomial test of coverage p
    at level alpha does not reject (the equal-tailed rule: each tail <= alpha / 2).
    """

    pmf = [math.comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(n + 1)]
    low, tail = 0, 0.0
    while low <= n and tail + pmf[low] <= alpha / 2:
        tail += pmf[low]
        low += 1
    high, tail = n, 0.0
    while high >= 0 and tail + pmf[high] <= alpha / 2:
        tail += pmf[high]
        high -= 1
    return low, high


def schedule_rows() -> list[dict[str, object]]:
    """Per horizon: registered outcomes, whether tested, and the MDE at each checkpoint."""

    rows = []
    for h in HORIZONS:
        row: dict[str, object] = {"horizon": h}
        for name, checkpoint in CHECKPOINTS.items():
            n = confirmatory_outcomes(h, checkpoint)
            level = first_holm_level(checkpoint)
            tested = n >= MIN_OUTCOMES and level is not None
            row[f"n_{name}"] = n
            row[f"tested_{name}"] = tested
            row[f"mde_{name}"] = minimum_detectable_effect(n, level) if tested else None
        rows.append(row)
    return rows


def coverage_rows(level: float = 0.8) -> list[dict[str, object]]:
    """Per tested horizon and checkpoint: the hit share the coverage test accepts."""

    rows = []
    for name, checkpoint in CHECKPOINTS.items():
        alpha = first_holm_level(checkpoint, tests_per_cell=len(TARGETS))
        for h in HORIZONS:
            n = confirmatory_outcomes(h, checkpoint)
            if n < MIN_OUTCOMES or alpha is None:
                continue
            low, high = binomial_acceptance(n, level, alpha)
            rows.append(
                {
                    "checkpoint": name,
                    "horizon": h,
                    "n": n,
                    "alpha": alpha,
                    "accepted_low": low / n,
                    "accepted_high": high / n,
                }
            )
    return rows


__all__ = [
    "ALPHA_PER_CHECKPOINT",
    "CHECKPOINTS",
    "ENTRY_LAG",
    "FORECASTS_PER_SESSION",
    "FUNDS",
    "HORIZONS",
    "MIN_OUTCOMES",
    "POWER_TARGET",
    "TARGETS",
    "TRIAL_SESSIONS",
    "binomial_acceptance",
    "confirmatory_outcomes",
    "confirmatory_sessions",
    "coverage_rows",
    "family",
    "first_holm_level",
    "minimum_detectable_effect",
    "outcomes_for_power",
    "power",
    "schedule_rows",
]
