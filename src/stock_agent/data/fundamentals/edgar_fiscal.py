from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

_QUARTERLY_FORMS = {
    "10-Q",
    "10-Q/A",
}

_ANNUAL_FORMS = {
    "10-K",
    "10-K/A",
}

_QUARTER_NUMBER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "FY": 4,
}


class EdgarFiscalIdentityError(ValueError):
    """Raised when authoritative EDGAR fiscal identity cannot be resolved safely."""


@dataclass(frozen=True)
class EdgarFiscalIdentity:
    """
    Canonical fiscal identity for one SEC filing.

    fiscal_period preserves the EDGAR/XBRL fiscal label:

        Q1
        Q2
        Q3
        FY

    fiscal_quarter converts that identity into the canonical
    quarter number used by the project's quarterly data model.
    """

    fiscal_year: int
    fiscal_period: str
    fiscal_quarter: int


def _normalize_form(form: str) -> str:
    normalized = str(form).strip().upper()

    if not normalized:
        raise EdgarFiscalIdentityError("SEC form cannot be empty.")

    if normalized not in _QUARTERLY_FORMS | _ANNUAL_FORMS:
        raise EdgarFiscalIdentityError(f"Unsupported SEC form for fiscal identity: {normalized!r}.")

    return normalized


def _normalize_period_end(
    period_end: str | pd.Timestamp,
) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(period_end)
    except (TypeError, ValueError) as exc:
        raise EdgarFiscalIdentityError(f"Invalid period_end: {period_end!r}.") from exc

    if pd.isna(timestamp):
        raise EdgarFiscalIdentityError("period_end cannot be missing.")

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.normalize()


def _period_end_from_reporting_period(
    period: dict[str, Any],
) -> pd.Timestamp | None:
    """
    Return a normalized reporting-period end date.

    Quarterly and annual fiscal periods are duration periods and
    normally expose `end_date`. Malformed or incomplete provider
    metadata is ignored here and handled by the caller through a
    clear fiscal-identity error.
    """

    raw_end_date = period.get("end_date")

    if raw_end_date is None:
        return None

    try:
        timestamp = pd.Timestamp(raw_end_date)
    except (TypeError, ValueError):
        return None

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.normalize()


def _normalize_fiscal_year(
    value: object,
) -> int | None:
    if value is None or pd.isna(value):
        return None

    try:
        fiscal_year = int(value)
    except (TypeError, ValueError):
        return None

    if fiscal_year <= 0:
        return None

    return fiscal_year


def _normalize_fiscal_period(
    value: object,
) -> str | None:
    if value is None or pd.isna(value):
        return None

    fiscal_period = str(value).strip().upper()

    if not fiscal_period:
        return None

    return fiscal_period


def infer_edgar_fiscal_identity(
    *,
    form: str,
    period_end: str | pd.Timestamp,
    reporting_periods: list[dict[str, Any]],
) -> EdgarFiscalIdentity:
    """
    Resolve the authoritative fiscal identity of one SEC filing.

    Fiscal identity comes from EDGAR/XBRL reporting-period metadata,
    not from calendar-month assumptions.

    Rules
    -----
    10-Q / 10-Q/A:
        Accept a matching Q1, Q2, or Q3 fiscal period.

    10-K / 10-K/A:
        Accept a matching FY fiscal period and normalize it to
        canonical fiscal quarter 4.

    Reporting periods for other period-end dates are ignored.

    Duplicate metadata describing the same fiscal identity is safe.
    Conflicting fiscal identities for the same filing are rejected
    rather than guessed.
    """

    normalized_form = _normalize_form(form)
    target_period_end = _normalize_period_end(period_end)

    if reporting_periods is None:
        raise EdgarFiscalIdentityError("EDGAR reporting_periods cannot be missing.")

    if normalized_form in _QUARTERLY_FORMS:
        allowed_periods = {
            "Q1",
            "Q2",
            "Q3",
        }
    else:
        allowed_periods = {"FY"}

    candidates: set[tuple[int, str]] = set()

    for period in reporting_periods:
        if not isinstance(period, dict):
            continue

        reporting_period_end = _period_end_from_reporting_period(period)

        if reporting_period_end is None:
            continue

        if reporting_period_end != target_period_end:
            continue

        fiscal_year = _normalize_fiscal_year(period.get("fiscal_year"))

        fiscal_period = _normalize_fiscal_period(period.get("fiscal_period"))

        if fiscal_year is None or fiscal_period is None:
            continue

        if fiscal_period not in allowed_periods:
            continue

        candidates.add(
            (
                fiscal_year,
                fiscal_period,
            )
        )

    if not candidates:
        raise EdgarFiscalIdentityError(
            "Unable to determine EDGAR fiscal identity for "
            f"form={normalized_form!r}, "
            f"period_end={target_period_end.date()}."
        )

    if len(candidates) > 1:
        candidate_text = ", ".join(
            f"FY{fiscal_year}/{fiscal_period}" for fiscal_year, fiscal_period in sorted(candidates)
        )

        raise EdgarFiscalIdentityError(
            "Conflicting EDGAR fiscal identities were found for "
            f"form={normalized_form!r}, "
            f"period_end={target_period_end.date()}: "
            f"{candidate_text}."
        )

    fiscal_year, fiscal_period = next(iter(candidates))

    return EdgarFiscalIdentity(
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        fiscal_quarter=_QUARTER_NUMBER[fiscal_period],
    )
