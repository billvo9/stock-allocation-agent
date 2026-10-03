from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from edgar import find

from stock_agent.data.fundamentals.edgar_fiscal import (
    EdgarFiscalIdentityError,
    infer_edgar_fiscal_identity,
)
from stock_agent.data.fundamentals.edgar_periods import (
    find_period_columns,
)
from stock_agent.data.fundamentals.edgar_quarterly import (
    EdgarQuarterlyBuildError,
    build_edgar_quarter,
)
from stock_agent.data.fundamentals.edgar_source import (
    list_financial_filings,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)

DIAGNOSTIC_COLUMNS = [
    "symbol",
    "provider_symbol",
    "accession_number",
    "form",
    "period_end",
    "filing_date",
    "accepted_at",
    "status",
    "reason",
    "detail",
]


@dataclass(frozen=True)
class EdgarHistoryResult:
    frame: pd.DataFrame
    diagnostics: pd.DataFrame


FilingLister = Callable[[str], pd.DataFrame]
FilingLoader = Callable[[str], Any]
Clock = Callable[[], pd.Timestamp]


class EdgarHistoryBuildError(ValueError):
    """Raised when EDGAR filing history cannot be assembled safely."""


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalize_timestamp(
    value: object,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.normalize()


def _infer_fiscal_quarter(
    *,
    form: str,
    period_end: object,
    statement_frames: list[pd.DataFrame | None],
) -> int:
    normalized_form = form.strip().upper()

    if normalized_form in {"10-K", "10-K/A"}:
        return 4

    if normalized_form not in {"10-Q", "10-Q/A"}:
        raise EdgarHistoryBuildError(f"Unsupported SEC form for quarterly history: {form!r}.")

    target_period_end = _normalize_timestamp(
        period_end,
    )

    matching_quarters: set[int] = set()

    for frame in statement_frames:
        if frame is None or frame.empty:
            continue

        for period in find_period_columns(frame):
            if period.period_end != target_period_end:
                continue

            if period.period_label not in {
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            }:
                continue

            matching_quarters.add(int(period.period_label[-1]))

    if not matching_quarters:
        raise EdgarHistoryBuildError(
            "Unable to infer fiscal quarter from "
            f"EDGAR statements for period_end={target_period_end}."
        )

    if len(matching_quarters) > 1:
        raise EdgarHistoryBuildError(
            "Conflicting fiscal-quarter labels were found "
            f"for period_end={target_period_end}: "
            f"{sorted(matching_quarters)}."
        )

    return matching_quarters.pop()


def _statement_frame(
    statements: Any,
    method_name: str,
) -> pd.DataFrame | None:
    getter = getattr(
        statements,
        method_name,
        None,
    )

    if getter is None:
        return None

    statement = getter()

    if statement is None:
        return None

    frame = statement.to_dataframe()

    if not isinstance(frame, pd.DataFrame):
        raise EdgarHistoryBuildError(f"{method_name} did not return a pandas DataFrame.")

    return frame


def _diagnostic_row(
    *,
    symbol: str,
    provider_symbol: str,
    filing: pd.Series,
    status: str,
    reason: str,
    detail: str = "",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "provider_symbol": provider_symbol,
        "accession_number": filing.get("accession_number"),
        "form": filing.get("form"),
        "period_end": filing.get("period_end"),
        "filing_date": filing.get("filing_date"),
        "accepted_at": filing.get("accepted_at"),
        "status": status,
        "reason": reason,
        "detail": detail,
    }


def build_edgar_history(
    *,
    symbol: str,
    provider_symbol: str,
    filing_lister: FilingLister = list_financial_filings,
    filing_loader: FilingLoader = find,
    clock: Clock = _utc_now,
    currency: str = "USD",
    strict: bool = False,
) -> EdgarHistoryResult:
    symbol = symbol.strip()
    provider_symbol = provider_symbol.strip()

    if not symbol:
        raise EdgarHistoryBuildError("symbol cannot be empty.")

    if not provider_symbol:
        raise EdgarHistoryBuildError("provider_symbol cannot be empty.")

    filings = filing_lister(provider_symbol)

    if not isinstance(filings, pd.DataFrame):
        raise EdgarHistoryBuildError("filing_lister must return a pandas DataFrame.")

    empty_history = pd.DataFrame(columns=QUARTERLY_FUNDAMENTAL_COLUMNS)

    empty_diagnostics = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    if filings.empty:
        return EdgarHistoryResult(
            frame=empty_history,
            diagnostics=empty_diagnostics,
        )

    required_columns = {
        "form",
        "period_end",
        "filing_date",
        "accepted_at",
        "accession_number",
        "is_xbrl",
    }

    missing_columns = sorted(required_columns - set(filings.columns))

    if missing_columns:
        raise EdgarHistoryBuildError(
            f"EDGAR filing metadata are missing required columns: {missing_columns}"
        )

    retrieved_at = pd.Timestamp(clock())

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize("UTC")
    else:
        retrieved_at = retrieved_at.tz_convert("UTC")

    quarterly_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    for _, filing_row in filings.iterrows():
        accession_number = str(filing_row["accession_number"])

        if not bool(filing_row["is_xbrl"]):
            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="skipped",
                    reason="no_xbrl",
                )
            )
            continue

        try:
            filing = filing_loader(accession_number)

            if filing is None:
                raise EdgarHistoryBuildError("Filing loader returned None.")

            xbrl = filing.xbrl()

            if xbrl is None:
                diagnostic_rows.append(
                    _diagnostic_row(
                        symbol=symbol,
                        provider_symbol=provider_symbol,
                        filing=filing_row,
                        status="skipped",
                        reason="no_xbrl_payload",
                    )
                )
                continue

            fiscal_identity = infer_edgar_fiscal_identity(
                form=str(filing_row["form"]),
                period_end=filing_row["period_end"],
                reporting_periods=xbrl.reporting_periods,
            )

            statements = xbrl.statements

            income_statement = _statement_frame(
                statements,
                "income_statement",
            )

            balance_sheet = _statement_frame(
                statements,
                "balance_sheet",
            )

            cash_flow_statement = _statement_frame(
                statements,
                "cash_flow_statement",
            )

            statement_frames = [
                income_statement,
                balance_sheet,
                cash_flow_statement,
            ]

            has_usable_statement = any(
                statement is not None and not statement.empty for statement in statement_frames
            )

            if not has_usable_statement:
                raise EdgarHistoryBuildError(
                    "No usable EDGAR financial statements were found "
                    f"for {symbol}, accession={accession_number}."
                )

            fiscal_quarter = fiscal_identity.fiscal_quarter

            accepted_at = filing_row["accepted_at"]

            if pd.isna(accepted_at):
                accepted_at = None

            quarter = build_edgar_quarter(
                symbol=symbol,
                provider_symbol=provider_symbol,
                period_end=filing_row["period_end"],
                fiscal_quarter=fiscal_quarter,
                filing_date=filing_row["filing_date"],
                retrieved_at=retrieved_at,
                sec_form_type=str(filing_row["form"]),
                sec_accession_number=(accession_number),
                income_statement=income_statement,
                cash_flow_statement=(cash_flow_statement),
                balance_sheet=balance_sheet,
                accepted_at=accepted_at,
                currency=currency,
                source="edgar",
            )

            quarterly_frames.append(quarter)

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="success",
                    reason="ok",
                )
            )

        except EdgarFiscalIdentityError as exc:
            if strict:
                raise

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="skipped",
                    reason="fiscal_identity_error",
                    detail=str(exc),
                )
            )

        except (
            EdgarHistoryBuildError,
            EdgarQuarterlyBuildError,
        ) as exc:
            if strict:
                raise

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="skipped",
                    reason="quarter_build_error",
                    detail=str(exc),
                )
            )

        except Exception as exc:
            if strict:
                raise

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="error",
                    reason="unexpected_error",
                    detail=(f"{type(exc).__name__}: {exc}"),
                )
            )

    if quarterly_frames:
        history = pd.concat(
            quarterly_frames,
            ignore_index=True,
        )

        history = history.sort_values(
            [
                "available_at",
                "period_end",
                "sec_accession_number",
            ]
        ).reset_index(drop=True)

        validate_quarterly_fundamental_frame(history)
    else:
        history = empty_history

    diagnostics = pd.DataFrame(
        diagnostic_rows,
        columns=DIAGNOSTIC_COLUMNS,
    )

    return EdgarHistoryResult(
        frame=history,
        diagnostics=diagnostics,
    )
