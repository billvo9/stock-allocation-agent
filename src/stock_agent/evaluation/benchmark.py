from __future__ import annotations

import pandas as pd

from stock_agent.evaluation.baseline import (
    BaselineResult,
    prepare_feature_matrix,
    run_rebalanced_backtest,
)
from stock_agent.portfolio.transition import BROAD_ETF, PROVISIONAL_RATES

# D2/D3: a benchmark pays the D2 cost to build. The benchmarks are broad equity
# price indices, bought through their tradable proxy, a broad index ETF, so the
# build pays D2's provisional broad-ETF rate (5 bp).
BENCHMARK_BUILD_RATE = PROVISIONAL_RATES[BROAD_ETF]


def run_buy_and_hold_benchmark(
    frame: pd.DataFrame,
    symbol: str,
    initial_wealth: float = 100_000.0,
    transaction_cost_rate: float = BENCHMARK_BUILD_RATE,
    build_date: pd.Timestamp | None = None,
) -> BaselineResult:
    """
    Run a fully invested buy-and-hold benchmark on the approved engine: like
    every portfolio it starts as 100% cash, builds at the common first
    execution close and pays the D2 cost to build (D3, D7), and it is never
    rebalanced. A fund total-return benchmark (F4.1) is group C and not
    approved yet, so the benchmark is still the price index.
    """

    if not symbol:
        raise ValueError("Benchmark symbol cannot be empty.")

    returns = prepare_feature_matrix(
        frame=frame,
        symbols=[symbol],
        column="daily_return",
    )

    target_weights = pd.DataFrame(
        data=1.0,
        index=returns.index,
        columns=[symbol],
    )

    return run_rebalanced_backtest(
        returns=returns,
        target_weights=target_weights,
        initial_wealth=initial_wealth,
        rebalance_every=max(len(returns), 1),
        transaction_cost_rate=transaction_cost_rate,
        build_date=build_date,
    )
