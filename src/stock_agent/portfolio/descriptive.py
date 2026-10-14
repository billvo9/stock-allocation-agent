"""
Result types and constants for descriptive fund metrics.

Types only, no formulas. The formulas are OD8 group A in
`docs/research/t5_portfolio_formulas.md` (F5.4 growth of $10,000 and its
start rule, F6.5 worst decline, F6.6 calendar-year returns). They are pending
the owner's approval and arrive with ticket T5-A, which implements them in
this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

GROWTH_START_VALUE = 10_000.0
CALENDAR_COLUMNS = ("year", "total_return", "partial", "losing")


class MetricUnavailable(Exception):
    """
    A metric cannot be given. `reason` is a machine-readable code; `detail`
    says why in plain words a reader can understand.
    """

    def __init__(self, metric: str, reason: str, detail: str) -> None:
        super().__init__(f"{metric} unavailable ({reason}): {detail}")
        self.metric = metric
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class DrawdownSummary:
    """
    The worst decline of one growth series (F6.5, pending OD8).

    - `max_drawdown`: the largest percentage decline from a previous high,
      as a positive fraction (0.25 is a 25% decline).
    - `peak_date`, `trough_date`: the sessions of that episode's high and low.
    - `recovery_date`: the first session back at or above the peak, or None.
    - `duration_sessions`, `duration_days`: from the peak to the recovery when
      recovered; otherwise from the peak to the series' last session with a
      value (a trailing gap is not counted).
    - `dollar_decline`: that episode's peak value minus its trough value, in
      dollars of the growth series (positive).
    """

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    duration_sessions: int
    duration_days: int
    dollar_decline: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown is a fraction in [0, 1]")
        if self.trough_date < self.peak_date:
            raise ValueError("the trough cannot precede the peak")
        if self.recovery_date is not None and self.recovery_date < self.trough_date:
            raise ValueError("the recovery cannot precede the trough")
        if min(self.duration_sessions, self.duration_days) < 0 or self.dollar_decline < 0:
            raise ValueError("durations and the dollar decline are non-negative")

    @property
    def recovered(self) -> bool:
        """True exactly when a recovery session is recorded."""

        return self.recovery_date is not None
