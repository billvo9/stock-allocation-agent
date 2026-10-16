from __future__ import annotations

import pandas as pd

from stock_agent.evaluation.baseline import (
    BaselineResult,
    prepare_feature_matrix,
    run_rebalanced_backtest,
)


def run_buy_and_hold_benchmark(
    frame: pd.DataFrame,
    symbol: str,
    initial_wealth: float = 100_000.0,
) -> BaselineResult:
    """
    Run a fully invested buy-and-hold benchmark on the approved engine: it
    builds from cash at the common first execution close (D3, D7) and is
    never rebalanced. The benchmarks are price indices, which cannot be
    traded, so the build costs nothing here; a fund total-return benchmark
    with a D2 build cost (F4.1) is group C and is not approved yet.
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
        transaction_cost_rate=0.0,
    )
