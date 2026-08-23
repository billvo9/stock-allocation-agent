from pathlib import Path

import pandas as pd

from stock_agent.config import load_asset_symbols
from stock_agent.evaluation.baseline import (
    prepare_momentum_targets,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_momentum_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"


def print_result(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Initial wealth:     ${result.initial_wealth:,.2f}")
    print(f"Final wealth:       ${result.final_wealth:,.2f}")
    print(f"Cumulative return:  {result.cumulative_return:.2%}")
    print(f"Maximum drawdown:   {result.max_drawdown:.2%}")
    print(f"Total turnover:     {result.total_turnover:.4f}")


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    symbols = load_asset_symbols(CONFIG_PATH)

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
    print(f"\nCommon evaluation start: {comparison_start.date()}")

    print_result("Equal Weight", equal_weight_result)
    print_result("Inverse Volatility", inverse_vol_result)
    print_result("Momentum", momentum_result)

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


if __name__ == "__main__":
    main()
