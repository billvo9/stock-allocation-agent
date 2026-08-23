from pathlib import Path

import pandas as pd

from stock_agent.config import load_asset_symbols
from stock_agent.evaluation.baseline import (
    run_momentum_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)

    symbols = load_asset_symbols(CONFIG_PATH)

    result = run_momentum_baseline(
        frame=features,
        symbols=symbols,
        lookback=20,
        initial_wealth=100_000,
        rebalance_every=5,
        transaction_cost_rate=0.001,
        cash_symbol="CASH",
        cash_return=0.0,
    )

    print(f"Initial wealth:     ${result.initial_wealth:,.2f}")
    print(f"Final wealth:       ${result.final_wealth:,.2f}")
    print(f"Cumulative return:  {result.cumulative_return:.2%}")
    print(f"Maximum drawdown:   {result.max_drawdown:.2%}")
    print(f"Total turnover:     {result.total_turnover:.4f}")


if __name__ == "__main__":
    main()
