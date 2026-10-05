from __future__ import annotations

import math
from typing import Any

import pandas as pd

from stock_agent.data.fundamentals.edgar_concepts import (
    resolve_statement_concept,
)
from stock_agent.data.fundamentals.edgar_periods import (
    derive_quarter_from_ytd,
    get_direct_quarter_value,
    get_fy_value,
    get_ytd_value,
)
from stock_agent.data.fundamentals.edgar_reconciliation import (
    FiscalFlowObservation,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)


class EdgarQuarterlyBuildError(ValueError):
    """Raised when an EDGAR quarterly row cannot be assembled safely."""


# Canonical flow metric -> statement it is read from.
FLOW_METRIC_STATEMENTS = {
    "revenue": "income_statement",
    "gross_profit": "income_statement",
    "operating_income": "income_statement",
    "net_income": "income_statement",
    "operating_cash_flow": "cash_flow_statement",
}


def _require_nonempty_string(
    value: str,
    *,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EdgarQuarterlyBuildError(f"{name} must be a non-empty string.")

    return value.strip()


def _to_utc_timestamp(
    value: Any,
    *,
    name: str,
    normalize: bool = False,
) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EdgarQuarterlyBuildError(f"{name} must be a valid timestamp.") from exc

    if pd.isna(timestamp):
        raise EdgarQuarterlyBuildError(f"{name} cannot be missing.")

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    if normalize:
        timestamp = timestamp.normalize()

    return timestamp


_SEC_TIMEZONE = "America/New_York"

# SEC-assigned filing time (ET wall clock) for submissions deferred to the
# next business day. Built as a wall-clock time, not midnight + 6 hours, so
# it stays 06:00 on daylight-saving transition dates.
_SEC_DEFERRED_FILING_TIME = "06:00"


def _resolve_available_at(
    *,
    filing_date: pd.Timestamp,
    accepted_at: Any | None,
) -> tuple[pd.Timestamp, str]:
    """
    Determine the earliest time the pipeline may use the filing.

    Three distinct timestamps:

        accepted_at  = SEC acceptance time (when EDGAR accepted it)
        filing_date  = SEC-assigned filing date (a business day)
        available_at = earliest timestamp our point-in-time pipeline
                       permits use (what this function returns)

    EDGAR rule (Regulation S-T Rule 13; EDGAR Filer Manual): most live
    submissions transmitted after 5:30 p.m. ET receive a filing date of
    6:00 a.m. ET on the next business day and are not disseminated until
    that business day. 10-Q, 10-K, and their amendments have no same-day
    exception. Ownership and certain registration forms (3, 4, 5, 144,
    *MEF, POS 462B) do, but this pipeline does not ingest them.

    The SEC-assigned filing_date is the signal: SEC applies the cutoff and
    per-form exceptions, and weekends and holidays, when assigning it.

        accepted_at missing:
            filing_date + 1 day                (sec_filing_date_plus_1d)

        filing_date later than the ET calendar date of accepted_at:
            06:00 ET on filing_date, in UTC    (sec_deferred_filing_date_6am)

            06:00 ET is the SEC-assigned next-business-day filing time,
            used as our modeling proxy for availability under the
            documented EDGAR rule. It is not an independently measured
            dissemination time.

        otherwise (same-day filing date, or inconsistent metadata where
        the filing date precedes acceptance):
            accepted_at                        (sec_acceptance_datetime)

    available_at is never earlier than accepted_at.
    """

    if accepted_at is None:
        return (
            filing_date + pd.Timedelta(days=1),
            "sec_filing_date_plus_1d",
        )

    accepted = _to_utc_timestamp(
        accepted_at,
        name="accepted_at",
    )

    accepted_et_date = accepted.tz_convert(_SEC_TIMEZONE).date()

    if filing_date.date() > accepted_et_date:
        deferred = (
            pd.Timestamp(f"{filing_date.date()} {_SEC_DEFERRED_FILING_TIME}")
            .tz_localize(_SEC_TIMEZONE)
            .tz_convert("UTC")
        )

        return (
            deferred,
            "sec_deferred_filing_date_6am",
        )

    return (
        accepted,
        "sec_acceptance_datetime",
    )


def resolve_edgar_available_at(
    *,
    filing_date: Any,
    accepted_at: Any | None,
) -> tuple[pd.Timestamp, str]:
    """
    Public entry point to the availability rule for raw filing metadata.

    Normalizes filing_date to its SEC calendar date and treats a missing
    accepted_at (None or NaT) as absent, exactly as build_edgar_quarter
    does, so callers can decide whether a filing is usable yet before
    loading it.
    """

    normalized_filing_date = _to_utc_timestamp(
        filing_date,
        name="filing_date",
        normalize=True,
    )

    if accepted_at is not None and pd.isna(accepted_at):
        accepted_at = None

    return _resolve_available_at(
        filing_date=normalized_filing_date,
        accepted_at=accepted_at,
    )


def _resolve_metric_row(
    frame: pd.DataFrame | None,
    *,
    canonical_name: str,
    symbol: str,
    accession_number: str,
) -> pd.Series | None:
    """
    Resolve one canonical accounting metric.

    Missing concepts are allowed and return None.

    Ambiguous concepts are considered unsafe and are
    converted into a contextual project-level error.
    """

    if frame is None:
        return None

    if not isinstance(frame, pd.DataFrame):
        raise EdgarQuarterlyBuildError(
            f"{canonical_name} statement input for {symbol} must be a pandas DataFrame."
        )

    if frame.empty:
        return None

    try:
        return resolve_statement_concept(
            frame,
            canonical_name,
        )
    except ValueError as exc:
        raise EdgarQuarterlyBuildError(
            "Ambiguous EDGAR concept resolution for "
            f"{canonical_name}; symbol={symbol}, "
            f"accession={accession_number}. "
            "Refusing to choose a financial concept "
            "silently."
        ) from exc


def _extract_quarterly_flow(
    row: pd.Series | None,
    *,
    period_end: pd.Timestamp,
    fiscal_quarter: int,
) -> float | None:
    """
    Extract one quarterly flow metric.

    Priority:
        1. Use a direct standalone-quarter value when available.
        2. For Q1, use YTD because Q1 YTD equals standalone Q1.
        3. Defer Q2/Q3/Q4 cross-filing derivation to edgar_history.
        4. Return missing rather than guess.
    """

    if row is None:
        return None

    direct_value = get_direct_quarter_value(
        row,
        period_end=period_end,
        fiscal_quarter=fiscal_quarter,
    )

    if direct_value is not None:
        return direct_value

    # Only Q1 can be derived safely from one filing because
    # Q1 YTD is identical to standalone Q1.
    if fiscal_quarter != 1:
        return None

    current_ytd = get_ytd_value(
        row,
        period_end=period_end,
    )

    if current_ytd is None:
        return None

    return derive_quarter_from_ytd(
        current_ytd=current_ytd,
        previous_ytd=None,
        fiscal_quarter=1,
    )


def _extract_statement_flow(
    frame: pd.DataFrame | None,
    *,
    canonical_name: str,
    symbol: str,
    accession_number: str,
    period_end: pd.Timestamp,
    fiscal_quarter: int,
) -> float | None:
    row = _resolve_metric_row(
        frame,
        canonical_name=canonical_name,
        symbol=symbol,
        accession_number=accession_number,
    )

    return _extract_quarterly_flow(
        row,
        period_end=period_end,
        fiscal_quarter=fiscal_quarter,
    )


def extract_fiscal_flow_observations(
    *,
    symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    period_end: str | pd.Timestamp,
    available_at: str | pd.Timestamp,
    accession_number: str,
    income_statement: pd.DataFrame | None,
    cash_flow_statement: pd.DataFrame | None,
) -> list[FiscalFlowObservation]:
    """
    Read what ONE filing reports for each flow metric.

    For the filing's own period end, each observation carries the
    direct standalone-quarter value, the YTD value, and the
    full-fiscal-year value, whichever are reported. Nothing is
    derived here: cross-filing arithmetic belongs to reconciliation.

    One observation is returned per flow metric. A metric whose
    concept or statement is missing yields an observation with no
    values, so the gap stays visible downstream.

    Ambiguous concept resolution raises EdgarQuarterlyBuildError.
    """

    symbol = _require_nonempty_string(
        symbol,
        name="symbol",
    )

    accession_number = _require_nonempty_string(
        accession_number,
        name="accession_number",
    )

    if fiscal_quarter not in {1, 2, 3, 4}:
        raise EdgarQuarterlyBuildError("fiscal_quarter must be between 1 and 4.")

    normalized_period_end = _to_utc_timestamp(
        period_end,
        name="period_end",
        normalize=True,
    )

    normalized_available_at = _to_utc_timestamp(
        available_at,
        name="available_at",
    )

    statements = {
        "income_statement": income_statement,
        "cash_flow_statement": cash_flow_statement,
    }

    observations: list[FiscalFlowObservation] = []

    for metric_name, statement_name in FLOW_METRIC_STATEMENTS.items():
        row = _resolve_metric_row(
            statements[statement_name],
            canonical_name=metric_name,
            symbol=symbol,
            accession_number=accession_number,
        )

        direct_value = ytd_value = fy_value = None

        if row is not None:
            direct_value = get_direct_quarter_value(
                row,
                period_end=normalized_period_end,
                fiscal_quarter=fiscal_quarter,
            )

            ytd_value = get_ytd_value(
                row,
                period_end=normalized_period_end,
            )

            fy_value = get_fy_value(
                row,
                period_end=normalized_period_end,
            )

        for value in (direct_value, ytd_value, fy_value):
            if value is not None and not math.isfinite(value):
                raise EdgarQuarterlyBuildError(
                    f"EDGAR {metric_name} for {symbol}, accession "
                    f"{accession_number} contains a non-finite value."
                )

        observations.append(
            FiscalFlowObservation(
                symbol=symbol,
                metric_name=metric_name,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                period_end=normalized_period_end,
                available_at=normalized_available_at,
                accession_number=accession_number,
                direct_value=direct_value,
                ytd_value=ytd_value,
                fy_value=fy_value,
            )
        )

    return observations


def build_edgar_quarter(
    *,
    symbol: str,
    provider_symbol: str,
    period_end: str | pd.Timestamp,
    fiscal_quarter: int,
    filing_date: str | pd.Timestamp,
    retrieved_at: str | pd.Timestamp,
    sec_form_type: str,
    sec_accession_number: str,
    income_statement: pd.DataFrame | None,
    cash_flow_statement: pd.DataFrame | None,
    balance_sheet: pd.DataFrame | None = None,
    accepted_at: str | pd.Timestamp | None = None,
    currency: str = "USD",
    source: str = "edgar",
) -> pd.DataFrame:
    """
    Assemble one leakage-safe canonical quarterly row
    from EDGAR financial statements.

    Missing individual accounting metrics remain missing.
    Ambiguous mappings raise a contextual error instead
    of silently selecting a possibly incorrect value.

    balance_sheet is accepted now so the public interface
    will not need to change when balance-sheet metrics are
    added in the next expansion.
    """

    symbol = _require_nonempty_string(
        symbol,
        name="symbol",
    )

    provider_symbol = _require_nonempty_string(
        provider_symbol,
        name="provider_symbol",
    )

    sec_form_type = _require_nonempty_string(
        sec_form_type,
        name="sec_form_type",
    )

    sec_accession_number = _require_nonempty_string(
        sec_accession_number,
        name="sec_accession_number",
    )

    currency = _require_nonempty_string(
        currency,
        name="currency",
    )

    source = _require_nonempty_string(
        source,
        name="source",
    )

    if fiscal_quarter not in {1, 2, 3, 4}:
        raise EdgarQuarterlyBuildError("fiscal_quarter must be between 1 and 4.")

    normalized_period_end = _to_utc_timestamp(
        period_end,
        name="period_end",
        normalize=True,
    )

    normalized_filing_date = _to_utc_timestamp(
        filing_date,
        name="filing_date",
        normalize=True,
    )

    normalized_retrieved_at = _to_utc_timestamp(
        retrieved_at,
        name="retrieved_at",
    )

    available_at, availability_source = _resolve_available_at(
        filing_date=normalized_filing_date,
        accepted_at=accepted_at,
    )

    if available_at < normalized_period_end:
        raise EdgarQuarterlyBuildError(
            "EDGAR available_at cannot be before "
            f"period_end for {symbol}; "
            f"period_end={normalized_period_end}, "
            f"available_at={available_at}."
        )

    if normalized_retrieved_at < available_at:
        raise EdgarQuarterlyBuildError(f"retrieved_at cannot be before available_at for {symbol}.")

    if income_statement is None and cash_flow_statement is None:
        raise EdgarQuarterlyBuildError(
            f"No usable EDGAR statements were supplied "
            f"for {symbol}, accession "
            f"{sec_accession_number}."
        )

    # balance_sheet is deliberately accepted but not yet
    # consumed. The next feature expansion will add the
    # stock/balance-sheet metrics through the same interface.
    if balance_sheet is not None and not isinstance(
        balance_sheet,
        pd.DataFrame,
    ):
        raise EdgarQuarterlyBuildError("balance_sheet must be a pandas DataFrame when supplied.")

    metric_values = {
        "revenue": _extract_statement_flow(
            income_statement,
            canonical_name="revenue",
            symbol=symbol,
            accession_number=sec_accession_number,
            period_end=normalized_period_end,
            fiscal_quarter=fiscal_quarter,
        ),
        "gross_profit": _extract_statement_flow(
            income_statement,
            canonical_name="gross_profit",
            symbol=symbol,
            accession_number=sec_accession_number,
            period_end=normalized_period_end,
            fiscal_quarter=fiscal_quarter,
        ),
        "operating_income": _extract_statement_flow(
            income_statement,
            canonical_name="operating_income",
            symbol=symbol,
            accession_number=sec_accession_number,
            period_end=normalized_period_end,
            fiscal_quarter=fiscal_quarter,
        ),
        "net_income": _extract_statement_flow(
            income_statement,
            canonical_name="net_income",
            symbol=symbol,
            accession_number=sec_accession_number,
            period_end=normalized_period_end,
            fiscal_quarter=fiscal_quarter,
        ),
        "operating_cash_flow": _extract_statement_flow(
            cash_flow_statement,
            canonical_name="operating_cash_flow",
            symbol=symbol,
            accession_number=sec_accession_number,
            period_end=normalized_period_end,
            fiscal_quarter=fiscal_quarter,
        ),
    }

    row = {column: None for column in QUARTERLY_FUNDAMENTAL_COLUMNS}

    metadata_values = {
        "symbol": symbol,
        "provider_symbol": provider_symbol,
        "period_end": normalized_period_end,
        "available_at": available_at,
        "retrieved_at": normalized_retrieved_at,
        "filing_date": normalized_filing_date,
        "sec_form_type": sec_form_type,
        "sec_accession_number": sec_accession_number,
        "availability_source": availability_source,
        "currency": currency,
        "source": source,
    }

    for name, value in metadata_values.items():
        if name in row:
            row[name] = value

    for name, value in metric_values.items():
        if name in row:
            row[name] = value

    frame = pd.DataFrame(
        [row],
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )

    try:
        validate_quarterly_fundamental_frame(frame)
    except ValueError as exc:
        raise EdgarQuarterlyBuildError(
            "Constructed EDGAR quarterly row failed "
            f"canonical validation for {symbol}, "
            f"accession={sec_accession_number}: {exc}"
        ) from exc

    return frame
