from __future__ import annotations

import pandas as pd

from stock_agent.data.rates.transform import (
    prepare_risk_free_returns,
)
from stock_agent.evaluation.results import (
    BacktestHistory,
)


def prepare_aligned_risk_free_returns(
    history: BacktestHistory,
    rates: pd.DataFrame,
    rate_id: str,
) -> pd.Series:
    """Build risk-free returns aligned to portfolio returns."""

    portfolio_dates = history.wealth.index

    if not isinstance(
        portfolio_dates,
        pd.DatetimeIndex,
    ):
        raise TypeError("Backtest wealth history must use a DatetimeIndex.")

    risk_free_returns = prepare_risk_free_returns(
        rates=rates,
        portfolio_dates=portfolio_dates,
        rate_id=rate_id,
    )

    if not risk_free_returns.index.equals(history.portfolio_returns.index):
        raise RuntimeError("Portfolio and risk-free return histories are misaligned.")

    return risk_free_returns
