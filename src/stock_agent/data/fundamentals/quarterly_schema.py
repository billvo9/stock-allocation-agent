from __future__ import annotations

import numpy as np
import pandas as pd

QUARTERLY_FUNDAMENTAL_COLUMNS = [
    # Identity
    "symbol",
    "provider_symbol",
    # Point-in-time timing
    "period_end",
    "available_at",
    "retrieved_at",
    # Income statement
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "diluted_average_shares",
    # Balance sheet
    "total_assets",
    "total_debt",
    "cash_and_cash_equivalents",
    "inventory",
    "stockholders_equity",
    # Cash flow
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    # Filing / provenance
    "filing_date",
    "sec_form_type",
    "sec_accession_number",
    "availability_source",
    "currency",
    "source",
]


NUMERIC_QUARTERLY_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "diluted_average_shares",
    "total_assets",
    "total_debt",
    "cash_and_cash_equivalents",
    "inventory",
    "stockholders_equity",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
]


REQUIRED_QUARTERLY_COLUMNS = [
    "symbol",
    "provider_symbol",
    "period_end",
    "retrieved_at",
    "source",
]


STRING_REQUIRED_COLUMNS = [
    "symbol",
    "provider_symbol",
    "source",
]


def _validate_columns(
    frame: pd.DataFrame,
) -> None:
    missing = [column for column in QUARTERLY_FUNDAMENTAL_COLUMNS if column not in frame.columns]

    if missing:
        raise ValueError(f"Quarterly fundamental data are missing required columns: {missing}")


def _validate_required_values(
    frame: pd.DataFrame,
) -> None:
    for column in REQUIRED_QUARTERLY_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Quarterly fundamental data contain missing {column} values.")

    for column in STRING_REQUIRED_COLUMNS:
        values = frame[column].astype(str).str.strip()

        if values.eq("").any():
            raise ValueError(f"Quarterly fundamental data contain empty {column} values.")


def _validate_timestamps(
    frame: pd.DataFrame,
) -> None:
    period_end = pd.to_datetime(
        frame["period_end"],
        errors="coerce",
        utc=True,
    )

    retrieved_at = pd.to_datetime(
        frame["retrieved_at"],
        errors="coerce",
        utc=True,
    )

    available_at = pd.to_datetime(
        frame["available_at"],
        errors="coerce",
        utc=True,
    )

    filing_date = pd.to_datetime(
        frame["filing_date"],
        errors="coerce",
        utc=True,
    )

    if period_end.isna().any():
        raise ValueError("Quarterly fundamental data contain invalid period_end values.")

    if retrieved_at.isna().any():
        raise ValueError("Quarterly fundamental data contain invalid retrieved_at values.")

    # available_at is intentionally nullable.
    invalid_available_at = frame["available_at"].notna() & available_at.isna()

    if invalid_available_at.any():
        raise ValueError("Quarterly fundamental data contain invalid available_at values.")

    # filing_date is also intentionally nullable.
    invalid_filing_date = frame["filing_date"].notna() & filing_date.isna()

    if invalid_filing_date.any():
        raise ValueError("Quarterly fundamental data contain invalid filing_date values.")

    known_availability = available_at.notna()

    if (available_at[known_availability] < period_end[known_availability]).any():
        raise ValueError("available_at cannot precede period_end.")

    if (available_at[known_availability] > retrieved_at[known_availability]).any():
        raise ValueError("available_at cannot be later than retrieved_at.")

    known_filing_date = filing_date.notna()

    if (filing_date[known_filing_date] < period_end[known_filing_date]).any():
        raise ValueError("filing_date cannot precede period_end.")


def _validate_numeric_columns(
    frame: pd.DataFrame,
) -> None:
    for column in NUMERIC_QUARTERLY_COLUMNS:
        original = frame[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid_type = original.notna() & converted.isna()

        if invalid_type.any():
            raise ValueError(
                f"Quarterly fundamental data contain invalid numeric values in {column}."
            )

        finite_values = converted[converted.notna()].to_numpy(dtype=float)

        if not np.isfinite(finite_values).all():
            raise ValueError(f"Quarterly fundamental data contain non-finite values in {column}.")


def validate_quarterly_fundamental_frame(
    frame: pd.DataFrame,
) -> None:
    """
    Validate canonical quarterly fundamental data.

    Accounting values and availability metadata may
    legitimately be missing. Identity, period_end,
    retrieved_at, and source must always be present.
    """

    _validate_columns(frame)

    if frame.empty:
        raise ValueError("Quarterly fundamental data cannot be empty.")

    _validate_required_values(frame)

    _validate_timestamps(frame)

    _validate_numeric_columns(frame)

    if frame.duplicated(
        subset=[
            "symbol",
            "period_end",
            "retrieved_at",
        ]
    ).any():
        raise ValueError(
            "Quarterly fundamental data contain duplicate (symbol, period_end, retrieved_at) rows."
        )
