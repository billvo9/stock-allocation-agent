from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stock_agent.config import (
    load_asset_symbols,
    load_benchmark_symbols,
)
from stock_agent.evaluation.baseline import (
    prepare_momentum_targets,
    run_equal_weight_baseline,
    run_inverse_volatility_baseline,
    run_momentum_baseline,
)
from stock_agent.evaluation.benchmark import (
    run_buy_and_hold_benchmark,
)
from stock_agent.evaluation.reporting import (
    build_drawdown_frame,
    build_normalized_wealth_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

ASSET_CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"

REPORT_DIR = PROJECT_ROOT / "reports" / "baseline_comparison"


def build_results():
    """Run strategies and passive benchmarks on a common period."""

    features = pd.read_parquet(FEATURES_PATH)

    features["date"] = pd.to_datetime(features["date"])

    symbols = load_asset_symbols(ASSET_CONFIG_PATH)

    benchmark_symbols = load_benchmark_symbols(ASSET_CONFIG_PATH)

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

    benchmark_results = {
        symbol: run_buy_and_hold_benchmark(
            frame=comparison_features,
            symbol=symbol,
            initial_wealth=100_000,
        )
        for symbol in benchmark_symbols
    }

    return {
        "Equal Weight": (equal_weight_result),
        "Inverse Volatility": (inverse_vol_result),
        "Momentum": momentum_result,
        **benchmark_results,
    }


def plot_normalized_wealth(
    normalized_wealth: pd.DataFrame,
) -> Path:
    """Plot normalized wealth for all portfolios."""

    output_path = REPORT_DIR / "normalized_wealth.png"

    plt.figure(figsize=(12, 7))

    for column in normalized_wealth.columns:
        plt.plot(
            normalized_wealth.index,
            normalized_wealth[column],
            label=column,
        )

    plt.title("Normalized portfolio wealth")
    plt.xlabel("Date")
    plt.ylabel("Growth of 100")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


def plot_drawdowns(
    drawdowns: pd.DataFrame,
) -> Path:
    """Plot portfolio drawdowns through time."""

    output_path = REPORT_DIR / "drawdowns.png"

    plt.figure(figsize=(12, 7))

    drawdown_percent = drawdowns * 100.0

    for column in drawdown_percent.columns:
        plt.plot(
            drawdown_percent.index,
            drawdown_percent[column],
            label=column,
        )

    plt.title("Portfolio drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = build_results()

    normalized_wealth = build_normalized_wealth_frame(results)

    drawdowns = build_drawdown_frame(results)

    wealth_path = plot_normalized_wealth(normalized_wealth)

    drawdown_path = plot_drawdowns(drawdowns)

    print("Saved normalized wealth chart:")
    print(wealth_path)

    print()
    print("Saved drawdown chart:")
    print(drawdown_path)


if __name__ == "__main__":
    main()
