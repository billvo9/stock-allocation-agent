"""
The descriptive-metrics interface the Explore page calls.

The calculations are OD8 group A of `docs/research/t5_portfolio_formulas.md`,
approved on 2026-10-14 and implemented in `stock_agent.portfolio.descriptive`
(`DescriptiveCalculator`), which never imports the app. This module states
the interface only, in pandas types and the result types from
`stock_agent.portfolio.descriptive`, so tests can substitute a fake.

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

from stock_agent.portfolio.descriptive import GROWTH_START_VALUE, DrawdownSummary


@runtime_checkable
class DescriptiveMetrics(Protocol):
    """
    What the Explore page needs from a metrics provider.

    The page passes only data inside the selected window, so no result can
    depend on later data.
    """

    def common_start(
        self, available_sessions: Mapping[str, pd.DatetimeIndex], window_start: pd.Timestamp
    ) -> pd.Timestamp | None:
        """
        The shared start t0 (R4), or None. `available_sessions` maps each
        selected fund to its sessions with a close inside the window. t0 is
        the first session, on or after the latest of the funds' first closes,
        at which every fund has a close.
        """
        ...

    def growth_index(
        self,
        returns: pd.Series,
        sessions_spanned: pd.Series,
        start_value: float = GROWTH_START_VALUE,
    ) -> pd.Series:
        """
        Hypothetical growth of `start_value` (A4), indexed like `returns`,
        whose first session is t0. Checked by the page: the first value is
        `start_value`; every value present is finite and positive; a session
        whose return is missing has no value (a gap is never filled).
        """
        ...

    def drawdown_summary(self, growth: pd.Series) -> DrawdownSummary:
        """The worst decline (A6); its dates are sessions of `growth` with a value."""
        ...

    def calendar_year_returns(
        self, growth: pd.Series, year_end_sessions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        One row per calendar year (A7, R5d), with the columns `CALENDAR_COLUMNS`:
        `year` (unique), `total_return`, `partial`, `losing` (exactly
        `total_return < 0` when available), `status` and `reason`.
        """
        ...
