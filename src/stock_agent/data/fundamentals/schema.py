from __future__ import annotations

import pandas as pd

METADATA_COLUMNS = [
    "symbol",
    "provider_symbol",
    "sector",
    "sector_key",
    "industry",
    "industry_key",
    "country",
    "exchange",
    "quote_type",
    "currency",
    "source",
    "retrieved_at",
]


FUNDAMENTAL_SNAPSHOT_COLUMNS = [
    "symbol",
    "provider_symbol",
    "market_cap",
    "enterprise_value",
    "forward_pe",
    "trailing_pe",
    "price_to_book",
    "forward_eps",
    "trailing_eps",
    "beta",
    "dividend_yield",
    "shares_outstanding",
    "source",
    "retrieved_at",
]


NUMERIC_SNAPSHOT_COLUMNS = [
    "market_cap",
    "enterprise_value",
    "forward_pe",
    "trailing_pe",
    "price_to_book",
    "forward_eps",
    "trailing_eps",
    "beta",
    "dividend_yield",
    "shares_outstanding",
]


REQUIRED_COLUMNS = [
    "symbol",
    "provider_symbol",
    "source",
    "retrieved_at",
]


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]

    if missing:
        raise ValueError(f"Fundamental data are missing required columns: {missing}")


def _validate_required_values(
    frame: pd.DataFrame,
) -> None:
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Fundamental data contain missing {column} values.")

    for column in [
        "symbol",
        "provider_symbol",
        "source",
    ]:
        values = frame[column].astype(str).str.strip()

        if values.eq("").any():
            raise ValueError(f"Fundamental data contain empty {column} values.")


def _validate_retrieved_at(
    frame: pd.DataFrame,
) -> None:
    retrieved_at = pd.to_datetime(
        frame["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    if retrieved_at.isna().any():
        raise ValueError("Fundamental data contain invalid retrieved_at values.")


def validate_metadata_frame(
    frame: pd.DataFrame,
) -> None:
    """Validate canonical instrument metadata."""

    _validate_columns(
        frame,
        METADATA_COLUMNS,
    )

    if frame.empty:
        raise ValueError("Instrument metadata cannot be empty.")

    _validate_required_values(frame)
    _validate_retrieved_at(frame)

    if frame.duplicated(subset=["symbol"]).any():
        raise ValueError("Instrument metadata contain duplicate symbols.")


def validate_fundamental_snapshot_frame(
    frame: pd.DataFrame,
) -> None:
    """Validate canonical fundamental snapshots."""

    _validate_columns(
        frame,
        FUNDAMENTAL_SNAPSHOT_COLUMNS,
    )

    if frame.empty:
        raise ValueError("Fundamental snapshots cannot be empty.")

    _validate_required_values(frame)
    _validate_retrieved_at(frame)

    for column in NUMERIC_SNAPSHOT_COLUMNS:
        original = frame[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid = original.notna() & converted.isna()

        if invalid.any():
            raise ValueError(f"Fundamental snapshot contains invalid numeric values in {column}.")

    if frame.duplicated(
        subset=[
            "symbol",
            "retrieved_at",
        ]
    ).any():
        raise ValueError("Fundamental snapshots contain duplicate (symbol, retrieved_at) rows.")
