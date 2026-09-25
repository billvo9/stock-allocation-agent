from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from edgar import Company

EDGAR_FINANCIAL_FORMS = (
    "10-Q",
    "10-K",
    "10-Q/A",
    "10-K/A",
)

EDGAR_FILING_COLUMNS = [
    "form",
    "period_end",
    "filing_date",
    "accepted_at",
    "accession_number",
    "is_xbrl",
]


CompanyFactory = Callable[[str], Any]


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
) -> None:
    missing_columns = sorted(required_columns - set(frame.columns))

    if missing_columns:
        raise ValueError(f"EDGAR filing metadata are missing required columns: {missing_columns}")


def list_financial_filings(
    provider_symbol: str,
    company_factory: CompanyFactory = Company,
) -> pd.DataFrame:
    """
    Return normalized SEC financial-filing metadata.

    This function performs filing discovery only.
    It does not extract or transform accounting data.

    The returned timestamps establish when each filing
    and reporting period existed for point-in-time
    fundamental processing.
    """

    provider_symbol = provider_symbol.strip()

    if not provider_symbol:
        raise ValueError("Provider symbol cannot be empty.")

    company = company_factory(provider_symbol)

    filings = company.get_filings(
        form=list(EDGAR_FINANCIAL_FORMS),
    )

    raw = filings.to_pandas().copy()

    if raw.empty:
        return pd.DataFrame(columns=EDGAR_FILING_COLUMNS)

    _validate_required_columns(
        raw,
        {
            "form",
            "filing_date",
            "reportDate",
            "acceptanceDateTime",
            "accession_number",
            "isXBRL",
        },
    )

    frame = pd.DataFrame(
        {
            "form": raw["form"],
            "period_end": pd.to_datetime(
                raw["reportDate"],
                errors="coerce",
                utc=True,
            ).dt.normalize(),
            "filing_date": pd.to_datetime(
                raw["filing_date"],
                errors="coerce",
                utc=True,
            ).dt.normalize(),
            "accepted_at": pd.to_datetime(
                raw["acceptanceDateTime"],
                errors="coerce",
                utc=True,
            ),
            "accession_number": raw["accession_number"],
            "is_xbrl": raw["isXBRL"].astype("boolean"),
        },
        columns=EDGAR_FILING_COLUMNS,
    )

    frame["form"] = frame["form"].astype("string")

    frame["accession_number"] = frame["accession_number"].astype("string")

    frame = frame.dropna(
        subset=[
            "form",
            "period_end",
            "filing_date",
            "accession_number",
        ]
    )

    return frame.sort_values(
        [
            "period_end",
            "filing_date",
            "accepted_at",
            "accession_number",
        ]
    ).reset_index(drop=True)
