"""
The descriptive-metrics provider the Explore page calls, and its result shapes.

The formulas behind these metrics are OD8 group A in
`docs/research/t5_portfolio_formulas.md`: the start rule and the
hypothetical growth of $10,000 (F5.4), the worst decline (F6.5) and
calendar-year returns (F6.6). The owner has not approved them, so this
module defines only the protocol and the shapes. Ticket T5-A implements
them. Until then the app runs with `PendingMetrics`, which computes nothing
and marks every metric unavailable.

Unavailability is structured, never free text: a provider raises
`MetricUnavailable` with a machine-readable `reason` and a `detail` in
words. The page catches only that exception, per metric and per fund; any
other exception is a defect and propagates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from stock_agent.app.history import FundHistory

GROWTH_START_VALUE = 10_000.0
CALENDAR_COLUMNS = ("year", "total_return", "partial", "losing")
PENDING_REASON = "pending_od8_group_a"
PENDING_DETAIL = "calculation pending approval of OD8 group A"


class MetricUnavailable(Exception):
    """A provider cannot give `metric`; `reason` is a machine-readable code."""

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
    - `peak_date`, `trough_date`: the session dates of that episode's high
      and low.
    - `recovery_date`: the first session back at or above the peak, or None.
    - `duration_sessions`, `duration_days`: from the peak to the recovery
      when recovered; otherwise from the peak to the series' last session
      (the decline is still open at the end).
    - `dollar_decline`: that episode's peak value minus its trough value, in
      dollars of the growth series (positive).
    - `recovered`: True exactly when `recovery_date` is set.
    """

    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    duration_sessions: int
    duration_days: int
    dollar_decline: float
    recovered: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown is a fraction in [0, 1]")
        if self.recovered != (self.recovery_date is not None):
            raise ValueError("recovered must match whether a recovery_date is set")
        if self.trough_date < self.peak_date:
            raise ValueError("the trough cannot precede the peak")
        if self.recovery_date is not None and self.recovery_date < self.trough_date:
            raise ValueError("the recovery cannot precede the trough")
        if min(self.duration_sessions, self.duration_days) < 0 or self.dollar_decline < 0:
            raise ValueError("durations and the dollar decline are non-negative")


@runtime_checkable
class DescriptiveMetrics(Protocol):
    """
    What the Explore page needs from a metrics provider (implemented by T5-A).

    The page passes only records inside the selected window, so no result
    can depend on later data.
    """

    def common_start(
        self, histories: Sequence[FundHistory], window_start: pd.Timestamp
    ) -> pd.Timestamp | None:
        """
        The first session on or after `window_start` at which every fund in
        `histories` has an available observation; None when there is none.
        The start is fixed by this rule, never chosen by the caller (F5.4).
        """
        ...

    def growth_index(
        self, daily_total_returns: pd.Series, start_value: float = GROWTH_START_VALUE
    ) -> pd.Series:
        """
        Hypothetical growth of `start_value`, indexed by session date like
        `daily_total_returns`, whose first label is t0. The first value is
        `start_value` at t0 (F5.4). Unavailable returns are NaN on input.
        """
        ...

    def drawdown_summary(self, growth: pd.Series) -> DrawdownSummary:
        """The worst decline of a growth series (F6.5)."""
        ...

    def calendar_year_returns(self, growth: pd.Series) -> pd.DataFrame:
        """
        One row per calendar year of `growth`, with the columns
        `CALENDAR_COLUMNS`: `year` (int), `total_return` (a fraction),
        `partial` (the year is not fully covered by the series) and `losing`
        (the unrounded return is below zero). Partial years are never
        annualized (F6.6).
        """
        ...


class PendingMetrics:
    """
    The provider the app runs with until T5-A: every metric is unavailable,
    with reason `PENDING_REASON`. It computes nothing.
    """

    def _pending(self, metric: str) -> MetricUnavailable:
        return MetricUnavailable(metric, PENDING_REASON, PENDING_DETAIL)

    def common_start(
        self, histories: Sequence[FundHistory], window_start: pd.Timestamp
    ) -> pd.Timestamp | None:
        raise self._pending("common_start")

    def growth_index(
        self, daily_total_returns: pd.Series, start_value: float = GROWTH_START_VALUE
    ) -> pd.Series:
        raise self._pending("growth_index")

    def drawdown_summary(self, growth: pd.Series) -> DrawdownSummary:
        raise self._pending("drawdown_summary")

    def calendar_year_returns(self, growth: pd.Series) -> pd.DataFrame:
        raise self._pending("calendar_year_returns")
