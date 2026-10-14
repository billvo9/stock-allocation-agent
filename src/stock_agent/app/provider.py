"""
The descriptive-metrics provider the Explore page calls.

The formulas behind these metrics are OD8 group A in
`docs/research/t5_portfolio_formulas.md`: the start rule and the
hypothetical growth of $10,000 (F5.4), the worst decline (F6.5) and
calendar-year returns (F6.6). The owner has not approved them, so this
module defines only the protocol. Ticket T5-A implements it in
`stock_agent.portfolio`, which holds the result types and never imports the
app: the protocol uses only pandas types and those result types. Until then
the app runs with `PendingMetrics`, which computes nothing and marks every
metric unavailable.

Unavailability is structured, never free text: a provider raises
`MetricUnavailable` with a machine-readable `reason` and a `detail` in plain
words. The page catches only that exception, per metric and per fund; any
other exception is a defect and propagates. The page also checks every
result against the contract below and raises ValueError on a violation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd

from stock_agent.portfolio.descriptive import (
    GROWTH_START_VALUE,
    DrawdownSummary,
    MetricUnavailable,
)

# The machine reason keeps the decision code; the words shown to readers do not.
PENDING_REASON = "pending_od8_group_a"
PENDING_DETAIL = "awaiting the project owner's approval of the descriptive formulas"


@runtime_checkable
class DescriptiveMetrics(Protocol):
    """
    What the Explore page needs from a metrics provider (implemented by T5-A).

    The page passes only data inside the selected window, so no result can
    depend on later data.
    """

    def common_start(
        self, available_sessions: Mapping[str, pd.DatetimeIndex], window_start: pd.Timestamp
    ) -> pd.Timestamp | None:
        """
        The shared start t0 of the selected funds (F5.4), or None when there
        is none.

        `available_sessions` maps each selected fund to the sorted sessions
        inside the window at which it has an available close. t0 is the first
        session, on or after the latest of the funds' individual first
        available closes in the window, at which every selected fund has a
        close. Growth starts at `GROWTH_START_VALUE` on that close, and the
        first applied return is the next session's. The start is fixed by
        this rule, never chosen by the caller. The final rule is an owner
        decision under OD8 group A.
        """
        ...

    def growth_index(
        self, daily_total_returns: pd.Series, start_value: float = GROWTH_START_VALUE
    ) -> pd.Series:
        """
        Hypothetical growth of `start_value` (F5.4), indexed by the same
        session dates as `daily_total_returns`, whose first label is t0.

        Contract checked by the page:
        - the first value is `start_value` at t0 (t0's own return is not
          applied);
        - every value present is finite and positive (the chart's axis is
          logarithmic);
        - on every session whose input return is missing (NaN), the growth is
          NaN: a gap is never filled.

        What the growth does after a gap (resume from the last value, stay
        unavailable, or something else) is an owner formula decision under
        OD8 group A. This protocol encodes no rule for it.
        """
        ...

    def drawdown_summary(self, growth: pd.Series) -> DrawdownSummary:
        """
        The worst decline of a growth series (F6.5). Its peak, trough and
        recovery dates are sessions of `growth` that have a value.
        """
        ...

    def calendar_year_returns(self, growth: pd.Series) -> pd.DataFrame:
        """
        One row per calendar year of `growth` (F6.6), with the columns
        `CALENDAR_COLUMNS` of `stock_agent.portfolio.descriptive`: `year`
        (int, unique, a year in which `growth` has a value), `total_return`
        (a fraction), `partial` (the year is not fully covered by the series)
        and `losing` (exactly `total_return < 0` on the unrounded value).
        Partial years are never annualized.
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
        self, available_sessions: Mapping[str, pd.DatetimeIndex], window_start: pd.Timestamp
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
