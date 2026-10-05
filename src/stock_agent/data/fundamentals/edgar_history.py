from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    FLOW_METRIC_STATEMENTS,
    EdgarQuarterlyBuildError,
    build_edgar_quarter,
    extract_fiscal_flow_observations,
    resolve_edgar_available_at,
)
from stock_agent.data.fundamentals.edgar_reconciliation import (
    FiscalFlowObservation,
    FiscalFlowReconciliation,
    PriorSelection,
    find_superseding_prior_observations,
    reconcile_fiscal_flow_from_candidates,
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


# Reconciliation lineage: one row per (symbol, sec_accession_number,
# metric_name). (symbol, sec_accession_number) maps back to the canonical
# filing row. Kept outside the canonical quarterly schema on purpose.
RECONCILIATION_COLUMNS = [
    "symbol",
    "sec_accession_number",
    "sec_form_type",
    "metric_name",
    "fiscal_year",
    "fiscal_quarter",
    "period_end",
    "available_at",
    "value",
    "method",
    "reason",
    "prior_accession_number",
    "prior_available_at",
    "prior_value_source",
    "skipped_prior_accessions",
    "equivalent_prior_accessions",
    "conflicting_prior_accessions",
]

_SUBTRACTION_METHODS = {
    "ytd_minus_prior_ytd",
    "fy_minus_q3_ytd",
}


def _empty_reconciliation() -> pd.DataFrame:
    return pd.DataFrame(columns=RECONCILIATION_COLUMNS)


@dataclass(frozen=True)
class EdgarHistoryResult:
    frame: pd.DataFrame
    diagnostics: pd.DataFrame
    reconciliation: pd.DataFrame = field(default_factory=_empty_reconciliation)


@dataclass(frozen=True)
class _BuiltFiling:
    accession_number: str
    filing_row: pd.Series
    quarter: pd.DataFrame
    observations: list[FiscalFlowObservation]


@dataclass(frozen=True)
class _ReconciledFlow:
    current: FiscalFlowObservation
    result: FiscalFlowReconciliation
    selection: PriorSelection | None
    superseding: tuple[FiscalFlowObservation, ...]


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


def _reconcile_observation(
    current: FiscalFlowObservation,
    prior_candidates: list[FiscalFlowObservation],
) -> _ReconciledFlow:
    outcome = reconcile_fiscal_flow_from_candidates(
        current=current,
        candidates=prior_candidates,
    )

    superseding: tuple[FiscalFlowObservation, ...] = ()

    if outcome.result.method in _SUBTRACTION_METHODS and outcome.selection is not None:
        superseding = find_superseding_prior_observations(
            current=current,
            prior=outcome.selection.prior,
            candidates=prior_candidates,
        )

    return _ReconciledFlow(
        current=current,
        result=outcome.result,
        selection=outcome.selection,
        superseding=superseding,
    )


def _reconcile_flows(
    built_filings: list[_BuiltFiling],
) -> list[_ReconciledFlow]:
    observations_by_quarter: dict[tuple[str, int, int], list[FiscalFlowObservation]] = {}

    for built in built_filings:
        for observation in built.observations:
            key = (
                observation.metric_name,
                observation.fiscal_year,
                observation.fiscal_quarter,
            )
            observations_by_quarter.setdefault(key, []).append(observation)

    reconciled: list[_ReconciledFlow] = []

    for built in built_filings:
        for observation in built.observations:
            prior_candidates = (
                observations_by_quarter.get(
                    (
                        observation.metric_name,
                        observation.fiscal_year,
                        observation.fiscal_quarter - 1,
                    ),
                    [],
                )
                if observation.fiscal_quarter > 1
                else []
            )

            reconciled.append(
                _reconcile_observation(
                    observation,
                    prior_candidates,
                )
            )

    return reconciled


def _reconciliation_frame(
    reconciled: list[_ReconciledFlow],
    built_filings: list[_BuiltFiling],
) -> pd.DataFrame:
    if not reconciled:
        return _empty_reconciliation()

    forms = {built.accession_number: str(built.filing_row["form"]) for built in built_filings}

    rows: list[dict[str, object]] = []

    for flow in reconciled:
        current = flow.current
        selection = flow.selection
        prior = None if selection is None else selection.prior

        rows.append(
            {
                "symbol": current.symbol,
                "sec_accession_number": current.accession_number,
                "sec_form_type": forms[current.accession_number],
                "metric_name": current.metric_name,
                "fiscal_year": current.fiscal_year,
                "fiscal_quarter": current.fiscal_quarter,
                "period_end": current.period_end,
                "available_at": current.available_at,
                "value": flow.result.value,
                "method": flow.result.method,
                "reason": flow.result.reason,
                "prior_accession_number": flow.result.prior_accession_number,
                "prior_available_at": None if prior is None else prior.available_at,
                "prior_value_source": flow.result.prior_value_source,
                "skipped_prior_accessions": (
                    () if selection is None else selection.skipped_prior_accessions
                ),
                "equivalent_prior_accessions": (
                    () if selection is None else selection.equivalent_prior_accessions
                ),
                "conflicting_prior_accessions": (
                    () if selection is None else selection.conflicting_prior_accessions
                ),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=RECONCILIATION_COLUMNS,
    )

    # Explicit dtypes keep any subset of the lineage comparable with any
    # other (e.g. history built from fewer filings).
    frame["fiscal_year"] = frame["fiscal_year"].astype("int64")
    frame["fiscal_quarter"] = frame["fiscal_quarter"].astype("int64")
    frame["value"] = pd.to_numeric(frame["value"]).astype("float64")

    # Nullable text columns use pandas' missing-value-aware string dtype,
    # so missing stays missing (never the text "None") and the dtype does
    # not depend on whether a subset happens to be all-missing.
    for column in ("prior_accession_number", "prior_value_source"):
        frame[column] = frame[column].astype("string")

    for column in ("period_end", "available_at", "prior_available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")

    return frame.sort_values(
        [
            "available_at",
            "sec_accession_number",
            "metric_name",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _superseded_input_diagnostics(
    reconciled: list[_ReconciledFlow],
    built_filings: list[_BuiltFiling],
    *,
    symbol: str,
    provider_symbol: str,
) -> list[dict[str, object]]:
    """
    Flag derived quarters whose prior input was later superseded.

    The diagnostic is attached to the SUPERSEDING filing, with that
    filing's timestamps: the fact "an earlier derived quarter used an
    input that has since been restated" only becomes knowable when the
    superseding filing does. Attaching it to the derived quarter's filing
    would date it before the amendment existed.

    The derived value itself is NOT rewritten: rewriting would backdate
    the later amendment. Until amendment-aware derived versions exist,
    the history is prefix-stable but not as-of complete for these cells,
    and this diagnostic is how they are identified.
    """

    filing_rows = {built.accession_number: built.filing_row for built in built_filings}

    flagged: list[tuple[pd.Timestamp, str, str, str, dict[str, object]]] = []

    for flow in reconciled:
        current = flow.current

        for superseding in flow.superseding:
            detail = (
                f"metric={current.metric_name}; "
                f"derived_accession={current.accession_number}; "
                f"used_prior={flow.result.prior_accession_number}"
            )

            flagged.append(
                (
                    pd.Timestamp(superseding.available_at),
                    superseding.accession_number,
                    current.metric_name,
                    current.accession_number,
                    _diagnostic_row(
                        symbol=symbol,
                        provider_symbol=provider_symbol,
                        filing=filing_rows[superseding.accession_number],
                        status="info",
                        reason="derived_quarter_input_superseded",
                        detail=detail,
                    ),
                )
            )

    flagged.sort(key=lambda item: item[:4])

    return [item[4] for item in flagged]


def _normalize_accession(
    value: object,
) -> str | None:
    """Return a stripped accession number, or None when missing/empty."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip()

    return normalized or None


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

    # Accession numbers identify filings everywhere downstream (lineage,
    # prior candidates, diagnostics), so normalize them once, up front.
    filings = filings.copy()
    filings["accession_number"] = filings["accession_number"].map(_normalize_accession)

    accession_counts = filings["accession_number"].value_counts(dropna=True)
    duplicate_accessions = set(accession_counts[accession_counts > 1].index)

    retrieved_at = pd.Timestamp(clock())

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize("UTC")
    else:
        retrieved_at = retrieved_at.tz_convert("UTC")

    built_filings: list[_BuiltFiling] = []
    diagnostic_rows: list[dict[str, object]] = []

    for _, filing_row in filings.iterrows():
        accession_number = filing_row["accession_number"]

        if not isinstance(accession_number, str):
            if strict:
                raise EdgarHistoryBuildError(
                    "EDGAR filing listing contains a missing or empty accession number."
                )

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="skipped",
                    reason="invalid_accession",
                    detail="Missing or empty accession number.",
                )
            )
            continue

        # Conflicting rows cannot be told apart by accession, and keeping
        # the first one would depend on listing order. Skip every copy.
        if accession_number in duplicate_accessions:
            if strict:
                raise EdgarHistoryBuildError(
                    f"Duplicate accession number in EDGAR filing listing: {accession_number}."
                )

            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=filing_row,
                    status="skipped",
                    reason="duplicate_accession",
                    detail="Accession number appears more than once in the listing.",
                )
            )
            continue

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
            # A filing accepted after hours is not usable until its deferred
            # availability. A run inside that window is an expected data
            # condition, not an error, in both strict and non-strict modes;
            # the filing is picked up by the first run at or after it.
            available_at, _ = resolve_edgar_available_at(
                filing_date=filing_row["filing_date"],
                accepted_at=filing_row["accepted_at"],
            )

            if available_at > retrieved_at:
                diagnostic_rows.append(
                    _diagnostic_row(
                        symbol=symbol,
                        provider_symbol=provider_symbol,
                        filing=filing_row,
                        status="skipped",
                        reason="not_yet_available",
                        detail=(
                            f"available_at={available_at.isoformat()} is after "
                            f"retrieved_at={retrieved_at.isoformat()}"
                        ),
                    )
                )
                continue

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

            observations = extract_fiscal_flow_observations(
                symbol=symbol,
                fiscal_year=fiscal_identity.fiscal_year,
                fiscal_quarter=fiscal_quarter,
                period_end=filing_row["period_end"],
                available_at=quarter.iloc[0]["available_at"],
                accession_number=accession_number,
                income_statement=income_statement,
                cash_flow_statement=cash_flow_statement,
            )

            built_filings.append(
                _BuiltFiling(
                    accession_number=accession_number,
                    filing_row=filing_row,
                    quarter=quarter,
                    observations=observations,
                )
            )

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

    # Cross-filing reconciliation runs only after every filing has been
    # collected, so the result never depends on filing iteration order.
    # Point-in-time safety comes from prior selection, which only admits
    # priors available at or before each current filing.
    reconciled = _reconcile_flows(built_filings)

    reconciliation = _reconciliation_frame(
        reconciled,
        built_filings,
    )

    diagnostic_rows.extend(
        _superseded_input_diagnostics(
            reconciled,
            built_filings,
            symbol=symbol,
            provider_symbol=provider_symbol,
        )
    )

    if built_filings:
        history = pd.concat(
            [built.quarter for built in built_filings],
            ignore_index=True,
        )

        flow_values = {
            (flow.current.accession_number, flow.current.metric_name): flow.result.value
            for flow in reconciled
        }

        # Within the history, reconcile_fiscal_flow is the single
        # authority for standalone-quarter flow values.
        for metric_name in FLOW_METRIC_STATEMENTS:
            history[metric_name] = pd.to_numeric(
                history["sec_accession_number"].map(
                    lambda accession, metric_name=metric_name: flow_values.get(
                        (accession, metric_name)
                    )
                ),
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
        reconciliation=reconciliation,
    )
