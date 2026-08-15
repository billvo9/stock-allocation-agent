from pathlib import Path

import pandas as pd

from stock_agent.evaluation.baseline import run_equal_weight_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

SYMBOLS = ["MU", "SNDK", "MRVL"]

def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)

    result = run_equal_weight_baseline(
        frame=features,
        symbols=SYMBOLS,
        initial_wealth=100_000,
        rebalance_every=30,
        transaction_cost_rate=0.001,
    )

    print(f"Initial wealth:     ${result.initial_wealth:,.2f}")
    print(f"Final wealth:       ${result.final_wealth:,.2f}")
    print(f"Cumulative return:  {result.cumulative_return:.2%}")
    print(f"Maximum drawdown:   {result.max_drawdown:.2%}")
    print(f"Total turnover:     {result.total_turnover:.4f}")

    print(
        features.loc[
            features["symbol"].isin(SYMBOLS),
            "daily_return",
        ].describe()
    )

    print(
        features.loc[
            features["symbol"].isin(SYMBOLS),
            "daily_return",
        ].abs().nlargest(10)
    )

    print(
        features.loc[
            features["symbol"].isin(SYMBOLS),
            ["date", "symbol", "daily_return"],
        ]
        .sort_values("daily_return")
        .head(10)
    )

    print(
        features.loc[
            features["symbol"].isin(SYMBOLS),
            ["date", "symbol", "daily_return"],
        ]
        .sort_values("daily_return", ascending=False)
        .head(10)
    )

if __name__ == "__main__":
    main()