from __future__ import annotations

from typing import Any

import pandas as pd

from stock_agent.data.fundamentals.edgar_concepts import (
    resolve_statement_concept,
)
from stock_agent.data.fundamentals.edgar_periods import (
    derive_quarter_from_ytd,
    find_period_columns,
    get_direct_quarter_value,
    get_ytd_value,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)


class EdgarQuarterlyBuildError(ValueError):
    """Raised when an EDGAR quarterly row cannot be assembled safely."""


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


def _resolve_available_at(
    *,
    filing_date: pd.Timestamp,
    accepted_at: Any | None,
) -> tuple[pd.Timestamp, str]:
    """
    Determine when the filing became usable.

    Prefer the SEC acceptance timestamp when available.

    When it is unavailable, use filing date + one day.
    The fallback is deliberately conservative so that a
    historical model does not accidentally see a filing
    before it was realistically available.
    """

    if accepted_at is not None:
        available_at = _to_utc_timestamp(
            accepted_at,
            name="accepted_at",
        )

        return (
            available_at,
            "sec_acceptance_datetime",
        )

    return (
        filing_date + pd.Timedelta(days=1),
        "sec_filing_date_plus_1d",
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


def _find_previous_ytd_value(
    row: pd.Series,
    *,
    period_end: pd.Timestamp,
) -> float | None:
    """
    Find the most recent YTD value before period_end.

    This avoids assuming every company's fiscal periods
    are separated by exactly three calendar months.
    """

    row_frame = row.to_frame().T

    candidates = [
        period
        for period in find_period_columns(row_frame)
        if (period.period_label == "YTD" and period.period_end < period_end)
    ]

    if not candidates:
        return None

    previous_period = max(
        candidates,
        key=lambda period: period.period_end,
    )

    return get_ytd_value(
        row,
        previous_period.period_end,
    )


def _extract_quarterly_flow(
    row: pd.Series | None,
    *,
    period_end: pd.Timestamp,
    fiscal_quarter: int,
) -> float | None:
    """
    Extract one quarterly flow metric.

    Priority:
        1. Direct standalone-quarter value.
        2. Derive Q1/Q2/Q3 from YTD data.
        3. Return missing rather than guess.

    Q4 YTD/FY derivation is deliberately deferred until
    the period layer explicitly supports FY minus Q3 YTD.
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

    current_ytd = get_ytd_value(
        row,
        period_end=period_end,
    )

    if current_ytd is None:
        return None

    if fiscal_quarter == 1:
        return derive_quarter_from_ytd(
            current_ytd=current_ytd,
            previous_ytd=None,
            fiscal_quarter=1,
        )

    if fiscal_quarter in {2, 3}:
        previous_ytd = _find_previous_ytd_value(
            row,
            period_end=period_end,
        )

        if previous_ytd is None:
            return None

        return derive_quarter_from_ytd(
            current_ytd=current_ytd,
            previous_ytd=previous_ytd,
            fiscal_quarter=fiscal_quarter,
        )

    # Q4 fallback is intentionally unsupported for now.
    return None


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
