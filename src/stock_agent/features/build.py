from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from stock_agent.config import load_asset_symbols

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SQL_PATH = PROJECT_ROOT / "sql" / "rolling_features.sql"

CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"


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
            f"Unable to build SQL market features. Check {SQL_PATH}. Original error: {exc}"
        ) from exc

    if frame.empty:
        raise RuntimeError("Feature query returned zero rows.")

    return frame


def find_common_investable_dates(
    frame: pd.DataFrame,
    investable_symbols: list[str],
) -> pd.Index:
    """Return dates for which all investable symbols are available."""

    investable = frame[frame["symbol"].isin(investable_symbols)]

    counts = investable.groupby("date")["symbol"].nunique()

    return counts[counts == len(investable_symbols)].index


def build_model_dataset() -> pd.DataFrame:
    frame = run_feature_query()

    frame = frame.dropna(
        subset=[
            "daily_return",
            "momentum_20d",
            "volatility_20d",
        ]
    )

    symbols = load_asset_symbols(CONFIG_PATH)

    common_dates = find_common_investable_dates(
        frame,
        symbols,
    )

    frame = frame[frame["date"].isin(common_dates)].copy()

    if frame.empty:
        raise RuntimeError("No common dates remain after feature filtering.")

    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)
