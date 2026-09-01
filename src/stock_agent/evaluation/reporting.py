from __future__ import annotations

import pandas as pd

from stock_agent.evaluation.baseline import (
    BaselineResult,
)
from stock_agent.evaluation.performance import (
    PerformanceMetrics,
)


def build_comparison_table(
    results: dict[str, BaselineResult],
    metrics: dict[str, PerformanceMetrics],
) -> pd.DataFrame:
    rows = []

    if results.keys() != metrics.keys():
        raise ValueError("Results and metrics must contain the same portfolio names.")

    for name, result in results.items():
        performance = metrics[name]

        rows.append(
            {
                "portfolio": name,
                "cagr": performance.cagr,
                "volatility": (performance.annualized_volatility),
                "sharpe": performance.sharpe,
                "sortino": performance.sortino,
                "calmar": performance.calmar,
                "max_drawdown": (result.max_drawdown),
                "turnover": (result.total_turnover),
                "final_wealth": (result.final_wealth),
            }
        )

    return pd.DataFrame(rows).set_index("portfolio")


def build_normalized_wealth_frame(
    results: dict[str, BaselineResult],
    base_value: float = 100.0,
) -> pd.DataFrame:
    """Return normalized wealth histories for comparison."""

    if not results:
        raise ValueError("At least one result is required.")

    normalized_series = {}

    reference_index = None

    for name, result in results.items():
        if result.history is None:
            raise ValueError(f"{name} does not contain history.")

        wealth = result.history.wealth

        if reference_index is None:
            reference_index = wealth.index
        elif not reference_index.equals(wealth.index):
            raise ValueError("Wealth histories must use the same dates.")

        normalized_series[name] = wealth / wealth.iloc[0] * base_value

    return pd.DataFrame(normalized_series)


def build_drawdown_frame(
    results: dict[str, BaselineResult],
) -> pd.DataFrame:
    """Return drawdown histories for multiple portfolios."""

    drawdown_series = {}

    reference_index = None

    for name, result in results.items():
        if result.history is None:
            raise ValueError(f"{name} does not contain history.")

        wealth = result.history.wealth

        if reference_index is None:
            reference_index = wealth.index
        elif not reference_index.equals(wealth.index):
            raise ValueError("Wealth histories must use the same dates.")

        running_peak = wealth.cummax()

        drawdown_series[name] = wealth / running_peak - 1.0

    return pd.DataFrame(drawdown_series)
