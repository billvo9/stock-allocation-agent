from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.features.build import (
    build_model_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

MACRO_VINTAGES_PATH = PROJECT_ROOT / "data" / "raw" / "macro_vintages.parquet"


def main() -> None:
    if not MACRO_VINTAGES_PATH.exists():
        raise FileNotFoundError(f"Macro vintage data not found: {MACRO_VINTAGES_PATH}")

    macro_vintages = pd.read_parquet(MACRO_VINTAGES_PATH)

    features = build_model_dataset(
        macro_vintages=macro_vintages,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved {len(features):,} feature rows")
    print(f"Output: {OUTPUT_PATH}")
    print()

    history_summary = (
        features.groupby("symbol")
        .agg(
            rows=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_index()
    )

    print("Feature history by symbol:")
    print(history_summary)

    print("Feature date range:")
    print(
        features["date"].min(),
        "to",
        features["date"].max(),
    )

    print()
    print("Combined feature panel:")
    print(f"Rows: {len(features):,}")
    print(
        "Asset-date duplicates:",
        features.duplicated(["date", "symbol"]).sum(),
    )

    print()
    print("Macro feature availability:")

    macro_summary = features[
        [
            "date",
            "macro_coverage_ratio",
        ]
    ].drop_duplicates(subset=["date"])

    print(macro_summary["macro_coverage_ratio"].describe())

    print()
    print("Macro feature missingness:")

    macro_columns = [
        column
        for column in features.columns
        if (column.startswith("us_") or column == "macro_coverage_ratio")
    ]

    print(features[macro_columns].isna().mean().mul(100).sort_values(ascending=False))


if __name__ == "__main__":
    main()
