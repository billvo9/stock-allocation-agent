from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SQL_PATH = PROJECT_ROOT / "sql" / "rolling_features.sql"

REQUIRED_MODEL_FEATURES = [
    "daily_return",
    "momentum_20d",
    "volatility_20d",
]


def load_feature_query() -> str:
    """Read the SQL feature query from disk."""

    return SQL_PATH.read_text(encoding="utf-8")


def run_feature_query() -> pd.DataFrame:
    """
    Execute rolling-feature SQL against
    the raw market data.
    """

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
    """
    Return dates for which all requested
    symbols are available.

    This helper is intended for synchronized
    evaluation and comparison. It should not
    define the master research dataset.
    """

    investable = frame[frame["symbol"].isin(investable_symbols)]

    counts = investable.groupby("date")["symbol"].nunique()

    return counts[counts == len(investable_symbols)].index


def build_model_dataset() -> pd.DataFrame:
    """
    Build the market-feature research panel.

    Assets retain their independently available
    histories. Newly listed assets do not truncate
    older assets to a common start date.
    """

    frame = run_feature_query()

    missing_columns = [column for column in REQUIRED_MODEL_FEATURES if column not in frame.columns]

    if missing_columns:
        raise RuntimeError(f"Feature query is missing required model columns: {missing_columns}")

    frame = frame.dropna(subset=REQUIRED_MODEL_FEATURES).copy()

    if frame.empty:
        raise RuntimeError("No feature rows remain after required-feature filtering.")

    if frame.duplicated(
        subset=[
            "date",
            "symbol",
        ]
    ).any():
        raise RuntimeError("Feature dataset contains duplicate (date, symbol) rows.")

    return frame.sort_values(
        [
            "date",
            "symbol",
        ],
        kind="stable",
    ).reset_index(drop=True)
