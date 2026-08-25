from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestHistory:
    """Time-series output from a portfolio backtest."""

    portfolio_returns: pd.Series
    wealth: pd.Series
