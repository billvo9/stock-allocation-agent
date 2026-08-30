from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_agent.evaluation.baseline import (
    BaselineResult,
)
from stock_agent.evaluation.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from stock_agent.evaluation.risk_adjusted import (
    prepare_aligned_risk_free_returns,
)


@dataclass(frozen=True)
class PerformanceMetrics:
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float


def calculate_performance_metrics(
    result: BaselineResult,
    rates: pd.DataFrame,
    rate_id: str,
) -> PerformanceMetrics:
    """Calculate performance metrics for one backtest result."""

    if result.history is None:
        raise ValueError("Backtest result does not contain history.")

    history = result.history

    risk_free_returns = prepare_aligned_risk_free_returns(
        history=history,
        rates=rates,
        rate_id=rate_id,
    )

    returns = history.portfolio_returns

    return PerformanceMetrics(
        cagr=calculate_cagr(returns),
        annualized_volatility=(calculate_annualized_volatility(returns)),
        sharpe=calculate_sharpe_ratio(
            returns=returns,
            risk_free_returns=risk_free_returns,
        ),
        sortino=calculate_sortino_ratio(
            returns=returns,
            risk_free_returns=risk_free_returns,
        ),
        calmar=calculate_calmar_ratio(returns),
    )
