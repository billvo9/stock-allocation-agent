from __future__ import annotations

from pathlib import Path

from stock_agent.features.build import (
    build_model_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"


def main() -> None:
    features = build_model_dataset()

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


if __name__ == "__main__":
    main()
