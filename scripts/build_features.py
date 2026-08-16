from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SQL_PATH = PROJECT_ROOT / "sql" / "rolling_features.sql"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rolling_features.parquet"

INVESTABLE_SYMBOLS = ["MU", "SNDK", "MRVL"]

BECHMARK_SYMBOLS = ["NASDAQ_COMPOSITE", "SP500", "DOW_JONES"]


def load_feature_query() -> str:
    """Read the SQL feature query from disk."""
    return SQL_PATH.read_text(encoding="utf-8")


def run_feature_query() -> pd.DataFrame:
    """Execute rolling feature SQL against the raw parquet data."""
    query = load_feature_query()

    try:
        frame = duckdb.sql(query).df()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to build SQL market fetures. Check {SQL_PATH}. Original error: {exc}"
        ) from exc

    if frame.empty:
        raise RuntimeError("Feature query returned zero rows.")

    return frame


def find_common_investable_dates(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return dates for which all investable symbols are available."""
    investable = frame[frame["symbol"].isin(INVESTABLE_SYMBOLS)]
    counts = investable.groupby("date")["symbol"].nunique()

    return counts[counts == len(INVESTABLE_SYMBOLS)].index


def build_model_dataset() -> pd.DataFrame:
    frame = run_feature_query()

    # Remove incomplete rolling-feature row.

    frame = frame.dropna(
        subset=[
            "daily_return",
            "momentum_20d",
            "volatility_20d",
        ]
    )

    common_dates = find_common_investable_dates(frame)

    frame = frame[frame["date"].isin(common_dates)].copy()

    if frame.empty:
        raise RuntimeError("No common dates remain after feature filtering.")

    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


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

    print("Rows by symbol:")
    print(features.groupby("symbol").size())
    print()

    print("Feature date range:")
    print(
        features["date"].min(),
        "to",
        features["date"].max(),
    )


if __name__ == "__main__":
    main()
