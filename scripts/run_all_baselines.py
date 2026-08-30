from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd

from stock_agent.config import (
    load_asset_symbols,
    load_risk_free_rate_config,
)
from stock_agent.evaluation.baseline import (
    BaselineResult,
    prepare_momentum_targets,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_momentum_baseline,
)
from stock_agent.evaluation.performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)
from stock_agent.evaluation.results import (
    BacktestHistory,
)
from stock_agent.evaluation.risk_adjusted import (
    prepare_aligned_risk_free_returns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

ASSET_CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"

RATE_CONFIG_PATH = PROJECT_ROOT / "config" / "rates.yaml"

RATES_PATH = PROJECT_ROOT / "data" / "raw" / "rates.parquet"


def require_history(
    result: BaselineResult,
) -> BacktestHistory:
    """Return backtest history or fail if history was not recorded."""

    if result.history is None:
        raise RuntimeError("Backtest result does not contain history.")

    return result.history


def print_result(
    name: str,
    result: BaselineResult,
    metrics: PerformanceMetrics,
) -> None:
    """Print one baseline summary."""

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Initial wealth:         ${result.initial_wealth:,.2f}")
    print(f"Final wealth:           ${result.final_wealth:,.2f}")
    print(f"Cumulative return:      {result.cumulative_return:.2%}")
    print(f"Maximum drawdown:       {result.max_drawdown:.2%}")
    print(f"Total turnover:         {result.total_turnover:.4f}")
    print(f"CAGR:                   {metrics.cagr:.2%}")
    print(f"Annualized volatility:  {metrics.annualized_volatility:.2%}")
    print(f"Sharpe ratio:           {metrics.sharpe:.4f}")
    print(f"Sortino ratio:          {metrics.sortino:.4f}")
    print(f"Calmar ratio:           {metrics.calmar:.4f}")


def validate_common_history(
    *histories: BacktestHistory,
) -> None:
    """Require all baselines to use identical realized return dates."""

    if not histories:
        raise ValueError("At least one backtest history is required.")

    reference_index = histories[0].portfolio_returns.index

    for history in histories[1:]:
        if not reference_index.equals(history.portfolio_returns.index):
            raise RuntimeError("Baseline histories do not share the same evaluation dates.")


def print_risk_free_alignment_sample(
    history: BacktestHistory,
    rates: pd.DataFrame,
    rate_id: str,
) -> None:
    """Print a temporary real-data check of risk-free alignment."""

    risk_free_returns = prepare_aligned_risk_free_returns(
        history=history,
        rates=rates,
        rate_id=rate_id,
    )

    comparison = pd.DataFrame(
        {
            "portfolio_return": (history.portfolio_returns),
            "risk_free_return": (risk_free_returns),
        }
    )

    portfolio_dates = history.wealth.index

    if not isinstance(
        portfolio_dates,
        pd.DatetimeIndex,
    ):
        raise TypeError("Backtest wealth history must use a DatetimeIndex.")

    comparison["calendar_days"] = [(end - start).days for start, end in pairwise(portfolio_dates)]

    print("\nRisk-free alignment sample (multi-day periods)")
    print("---------------------------------------")

    multi_day = comparison[comparison["calendar_days"] > 1].head(10)

    if multi_day.empty:
        print("No multi-day periods found in this sample.")
    else:
        print(multi_day)

    print("\nFirst 5 aligned observations")
    print("----------------------------")
    print(comparison.head(5))

    print("\nLast 5 aligned observations")
    print("---------------------------")
    print(comparison.tail(5))


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    rates = pd.read_parquet(RATES_PATH)

    symbols = load_asset_symbols(ASSET_CONFIG_PATH)

    rate_config = load_risk_free_rate_config(RATE_CONFIG_PATH)
    rate_id = rate_config.spec.rate_id

    features["date"] = pd.to_datetime(features["date"])

    lookback = 20

    momentum_targets = prepare_momentum_targets(
        frame=features,
        symbols=symbols,
        lookback=lookback,
    )

    comparison_start = momentum_targets.index[0]

    comparison_features = features[features["date"] >= comparison_start].copy()

    equal_weight_result = run_equal_weight_baseline(
        frame=comparison_features,
        symbols=symbols,
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
    )

    inverse_vol_result = run_inverse_volatility_baseline(
        frame=comparison_features,
        symbols=symbols,
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
    )

    momentum_result = run_momentum_baseline(
        frame=features,
        symbols=symbols,
        lookback=lookback,
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
        cash_symbol="CASH",
        cash_return=0.0,
    )

    equal_history = require_history(equal_weight_result)
    inverse_history = require_history(inverse_vol_result)
    momentum_history = require_history(momentum_result)

    validate_common_history(
        equal_history,
        inverse_history,
        momentum_history,
    )

    equal_metrics = calculate_performance_metrics(
        result=equal_weight_result,
        rates=rates,
        rate_id=rate_id,
    )

    inverse_metrics = calculate_performance_metrics(
        result=inverse_vol_result,
        rates=rates,
        rate_id=rate_id,
    )

    momentum_metrics = calculate_performance_metrics(
        result=momentum_result,
        rates=rates,
        rate_id=rate_id,
    )

    print(f"\nCommon evaluation start: {comparison_start.date()}")

    print_result(
        "Equal Weight",
        equal_weight_result,
        equal_metrics,
    )

    print_result(
        "Inverse Volatility",
        inverse_vol_result,
        inverse_metrics,
    )

    print_result(
        "Momentum",
        momentum_result,
        momentum_metrics,
    )

    stock_targets = momentum_targets[symbols]

    max_stock_weight = stock_targets.max(axis=1).max()

    average_max_stock_weight = stock_targets.max(axis=1).mean()

    cash_usage_rate = (momentum_targets["CASH"] > 0).mean()

    full_cash_rate = (momentum_targets["CASH"] == 1.0).mean()

    print("\nMomentum diagnostics")
    print("--------------------")
    print(f"Maximum single-stock weight: {max_stock_weight:.2%}")
    print(f"Average largest position:     {average_max_stock_weight:.2%}")
    print(f"Dates with cash exposure:     {cash_usage_rate:.2%}")
    print(f"Dates at 100% cash:           {full_cash_rate:.2%}")

    # Temporary diagnostic for this stage. Once real-data alignment
    # is visually confirmed, this can be removed from normal output.
    print_risk_free_alignment_sample(
        history=momentum_history,
        rates=rates,
        rate_id=rate_id,
    )


if __name__ == "__main__":
    main()
