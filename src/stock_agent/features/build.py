from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from stock_agent.features.panel import attach_macro_features

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


def build_model_dataset(
    macro_vintages: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the model-ready asset-date feature panel.

    Assets retain their independently available histories.
    Newly listed assets do not truncate older assets to a
    common start date.

    Optional point-in-time macro features are attached
    without changing the asset-date universe.
    """

    frame = run_feature_query()

    missing_columns = [column for column in REQUIRED_MODEL_FEATURES if column not in frame.columns]

    if missing_columns:
        raise RuntimeError(f"Feature query is missing required model columns: {missing_columns}")

    frame = frame.dropna(subset=REQUIRED_MODEL_FEATURES).copy()

    if frame.empty:
        raise RuntimeError("No feature rows remain after required-feature filtering.")

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
        utc=True,
    ).dt.normalize()

    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Model feature frame contains duplicate (date, symbol) rows.")

    if macro_vintages is not None:
        original_row_count = len(frame)

        original_keys = set(
            zip(
                frame["date"],
                frame["symbol"],
            )
        )

        frame = attach_macro_features(
            asset_panel=frame,
            macro_vintages=macro_vintages,
        )

        if len(frame) != original_row_count:
            raise RuntimeError("Macro integration changed the number of asset-date rows.")

        resulting_keys = set(
            zip(
                frame["date"],
                frame["symbol"],
            )
        )

        if resulting_keys != original_keys:
            raise RuntimeError("Macro integration changed asset-date keys.")

    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)
