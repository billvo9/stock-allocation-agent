from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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
    select_current_fiscal_observation,
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


# Reconciliation lineage: one row per (emitted version, metric), keyed by
# (symbol, sec_accession_number, available_at, metric_name).
# (symbol, sec_accession_number, available_at) maps back to the canonical
# version row; sec_accession_number is the version's ANCHOR filing.
# source_accession_number / source_available_at identify the raw filing
# whose current-side value was used; version_trigger_accessions lists the
# filings (own period and prior quarter) available exactly at the
# version's available_at. Kept outside the canonical quarterly schema.
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
    "source_accession_number",
    "source_available_at",
    "version_trigger_accessions",
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
    observations: tuple[FiscalFlowObservation, ...]
    fiscal_year: int
    fiscal_quarter: int
    period_end: pd.Timestamp
    available_at: pd.Timestamp

    @property
    def fiscal_period(self) -> tuple[int, int]:
        return (self.fiscal_year, self.fiscal_quarter)


# Raw SEC observations keyed by (metric_name, fiscal_year, fiscal_quarter).
# Built once, read-only; versioned outputs are never fed back into it.
_CandidateIndex = Mapping[tuple[str, int, int], tuple[FiscalFlowObservation, ...]]


@dataclass(frozen=True)
class _MetricVersion:
    metric_name: str
    value: float | None
    method: str
    reason: str
    source: FiscalFlowObservation | None
    result: FiscalFlowReconciliation | None
    selection: PriorSelection | None

    def outcome(self) -> tuple[float | None, str, str]:
        return (self.value, self.method, self.reason)


@dataclass(frozen=True)
class _Version:
    available_at: pd.Timestamp
    anchor: _BuiltFiling
    availability_source: str
    trigger_accessions: tuple[str, ...]
    metrics: tuple[_MetricVersion, ...]


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


def _period_end_rejections(
    built_filings: list[_BuiltFiling],
) -> dict[str, pd.Timestamp | None]:
    """
    Return rejected accession -> reference period_end of its fiscal period
    (None when rejected because the earliest filings disagreed).

    For each fiscal period the reference period_end is that of the
    earliest-available own-period filing. If the earliest filings (same
    available_at) disagree among themselves, all of them are rejected and
    the next-earliest remaining filing time is tried. Any later filing
    whose period_end differs from the reference is rejected.

    Only earlier-or-equal filings decide a filing's fate, so rejection
    never changes earlier results (prefix stability), and nothing depends
    on filing order.
    """

    by_period: dict[tuple[int, int], dict[pd.Timestamp, list[_BuiltFiling]]] = {}

    for built in built_filings:
        by_period.setdefault(built.fiscal_period, {}).setdefault(built.available_at, []).append(
            built
        )

    rejected: dict[str, pd.Timestamp | None] = {}

    for by_time in by_period.values():
        reference: pd.Timestamp | None = None

        for available_at in sorted(by_time):
            filings = by_time[available_at]

            if reference is None:
                period_ends = {built.period_end for built in filings}

                if len(period_ends) == 1:
                    reference = period_ends.pop()
                else:
                    for built in filings:
                        rejected[built.accession_number] = None
                continue

            for built in filings:
                if built.period_end != reference:
                    rejected[built.accession_number] = reference

    return rejected


def _build_candidate_index(
    built_filings: list[_BuiltFiling],
) -> _CandidateIndex:
    """
    Index the raw SEC observations once, before any version is produced.

    Tuples are sorted by accession so nothing depends on filing order.
    """

    grouped: dict[tuple[str, int, int], list[FiscalFlowObservation]] = {}

    for built in built_filings:
        for observation in built.observations:
            key = (
                observation.metric_name,
                observation.fiscal_year,
                observation.fiscal_quarter,
            )
            grouped.setdefault(key, []).append(observation)

    return MappingProxyType(
        {
            key: tuple(sorted(observations, key=lambda item: item.accession_number))
            for key, observations in grouped.items()
        }
    )


def _prior_candidates(
    index: _CandidateIndex,
    metric_name: str,
    fiscal_year: int,
    fiscal_quarter: int,
) -> tuple[FiscalFlowObservation, ...]:
    if fiscal_quarter <= 1:
        return ()

    return index.get((metric_name, fiscal_year, fiscal_quarter - 1), ())


def _reconcile_flows(
    built_filings: list[_BuiltFiling],
    index: _CandidateIndex,
) -> list[_ReconciledFlow]:
    """
    Reconcile every raw filing observation at its own available_at.

    Used only for the derived_quarter_input_superseded diagnostic; the
    versioned history itself is produced by _period_versions.
    """

    reconciled: list[_ReconciledFlow] = []

    for built in built_filings:
        for observation in built.observations:
            reconciled.append(
                _reconcile_observation(
                    observation,
                    list(
                        _prior_candidates(
                            index,
                            observation.metric_name,
                            observation.fiscal_year,
                            observation.fiscal_quarter,
                        )
                    ),
                )
            )

    return reconciled


def _evaluate_metric(
    index: _CandidateIndex,
    *,
    metric_name: str,
    fiscal_year: int,
    fiscal_quarter: int,
    as_of: pd.Timestamp,
) -> _MetricVersion:
    """
    Reconcile one metric of one fiscal period from raw inputs known at as_of.

    Only raw observations available at or before as_of are passed in, so
    nothing about a later filing (not even its existence, via a reason
    code such as "prior_available_after_current") reaches this version.
    """

    def known(
        observations: tuple[FiscalFlowObservation, ...],
    ) -> tuple[FiscalFlowObservation, ...]:
        return tuple(
            observation
            for observation in observations
            if pd.Timestamp(observation.available_at) <= as_of
        )

    selection = select_current_fiscal_observation(
        candidates=known(index.get((metric_name, fiscal_year, fiscal_quarter), ())),
        as_of=as_of,
    )

    if selection.current is None:
        return _MetricVersion(
            metric_name=metric_name,
            value=None,
            method="unavailable",
            reason=selection.reason,
            source=None,
            result=None,
            selection=None,
        )

    outcome = reconcile_fiscal_flow_from_candidates(
        current=selection.current,
        candidates=known(_prior_candidates(index, metric_name, fiscal_year, fiscal_quarter)),
        as_of=as_of,
    )

    return _MetricVersion(
        metric_name=metric_name,
        value=outcome.result.value,
        method=outcome.result.method,
        reason=outcome.result.reason,
        source=selection.current,
        result=outcome.result,
        selection=outcome.selection,
    )


def _period_versions(
    own_filings: list[_BuiltFiling],
    prior_filings: list[_BuiltFiling],
    index: _CandidateIndex,
) -> list[_Version]:
    """
    Produce every version of one fiscal period, in knowledge-time order.

    Events are the distinct available_at values of the period's own
    filings and of the immediately preceding fiscal quarter's filings, at
    or after the period's first own filing. Each event time t is evaluated
    once, from all raw filings available at or before t (same-instant
    filings are coalesced).

    A version is emitted at t when an own filing becomes available at t
    (filings always stay visible), or when any metric's
    (value, method, reason) differs from the previously emitted version.
    """

    own_sorted = sorted(own_filings, key=lambda item: (item.available_at, item.accession_number))
    first_own_at = own_sorted[0].available_at
    fiscal_year, fiscal_quarter = own_sorted[0].fiscal_period

    events = sorted(
        {built.available_at for built in own_filings}
        | {built.available_at for built in prior_filings if built.available_at >= first_own_at}
    )

    versions: list[_Version] = []
    previous: tuple[tuple[float | None, str, str], ...] | None = None

    for event_at in events:
        visible = [built for built in own_sorted if built.available_at <= event_at]
        latest_at = visible[-1].available_at
        anchor = min(
            (built for built in visible if built.available_at == latest_at),
            key=lambda item: item.accession_number,
        )

        metrics = tuple(
            _evaluate_metric(
                index,
                metric_name=metric_name,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                as_of=event_at,
            )
            for metric_name in FLOW_METRIC_STATEMENTS
        )

        outcomes = tuple(metric.outcome() for metric in metrics)
        own_filing_at_event = any(built.available_at == event_at for built in own_filings)

        if not own_filing_at_event and outcomes == previous:
            continue

        triggers = sorted(
            (built for built in (*own_filings, *prior_filings) if built.available_at == event_at),
            key=lambda item: item.accession_number,
        )

        source_filing = anchor if anchor.available_at == event_at else triggers[0]

        versions.append(
            _Version(
                available_at=event_at,
                anchor=anchor,
                availability_source=str(source_filing.quarter.iloc[0]["availability_source"]),
                trigger_accessions=tuple(built.accession_number for built in triggers),
                metrics=metrics,
            )
        )
        previous = outcomes

    return versions


def _reconciliation_frame(
    versions: list[_Version],
) -> pd.DataFrame:
    if not versions:
        return _empty_reconciliation()

    rows: list[dict[str, object]] = []

    for version in versions:
        anchor = version.anchor

        for metric in version.metrics:
            selection = metric.selection
            prior = None if selection is None else selection.prior
            result = metric.result
            source = metric.source

            rows.append(
                {
                    "symbol": str(anchor.quarter.iloc[0]["symbol"]),
                    "sec_accession_number": anchor.accession_number,
                    "sec_form_type": str(anchor.filing_row["form"]),
                    "metric_name": metric.metric_name,
                    "fiscal_year": anchor.fiscal_year,
                    "fiscal_quarter": anchor.fiscal_quarter,
                    "period_end": anchor.period_end,
                    "available_at": version.available_at,
                    "value": metric.value,
                    "method": metric.method,
                    "reason": metric.reason,
                    "prior_accession_number": (
                        None if result is None else result.prior_accession_number
                    ),
                    "prior_available_at": None if prior is None else prior.available_at,
                    "prior_value_source": None if result is None else result.prior_value_source,
                    "skipped_prior_accessions": (
                        () if selection is None else selection.skipped_prior_accessions
                    ),
                    "equivalent_prior_accessions": (
                        () if selection is None else selection.equivalent_prior_accessions
                    ),
                    "conflicting_prior_accessions": (
                        () if selection is None else selection.conflicting_prior_accessions
                    ),
                    "source_accession_number": None if source is None else source.accession_number,
                    "source_available_at": None if source is None else source.available_at,
                    "version_trigger_accessions": version.trigger_accessions,
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
    for column in ("prior_accession_number", "prior_value_source", "source_accession_number"):
        frame[column] = frame[column].astype("string")

    for column in ("period_end", "available_at", "prior_available_at", "source_available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")

    frame = frame.sort_values(
        [
            "available_at",
            "sec_accession_number",
            "metric_name",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if frame.duplicated(
        subset=["symbol", "sec_accession_number", "available_at", "metric_name"]
    ).any():
        raise EdgarHistoryBuildError(
            "Internal invariant violated: duplicate reconciliation lineage key "
            "(symbol, sec_accession_number, available_at, metric_name)."
        )

    return frame


def _history_frame(
    versions: list[_Version],
) -> pd.DataFrame:
    """One canonical row per version: the anchor's row at the version's time."""

    rows: list[pd.DataFrame] = []
    flow_values: dict[str, list[float | None]] = {name: [] for name in FLOW_METRIC_STATEMENTS}

    for version in versions:
        row = version.anchor.quarter.copy()
        row["available_at"] = version.available_at
        row["availability_source"] = version.availability_source
        rows.append(row)

        for metric in version.metrics:
            flow_values[metric.metric_name].append(metric.value)

    history = pd.concat(rows, ignore_index=True)

    # Within the history, reconcile_fiscal_flow is the single
    # authority for standalone-quarter flow values.
    for metric_name, values in flow_values.items():
        history[metric_name] = pd.to_numeric(pd.Series(values, index=history.index, dtype=object))

    if history.duplicated(subset=["symbol", "period_end", "available_at"]).any():
        raise EdgarHistoryBuildError(
            "Internal invariant violated: duplicate (symbol, period_end, available_at) "
            "in EDGAR history."
        )

    return history.sort_values(
        [
            "available_at",
            "period_end",
            "sec_accession_number",
        ]
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

    The earlier derived value itself is NOT rewritten: rewriting would
    backdate the later amendment. A re-derived version of the derived
    quarter is emitted at the superseding filing's time only when it
    changes some metric's (value, method, reason); see _period_versions.

    This diagnostic is an informational audit record computed from each
    derived filing's OWN-time evaluation: used_prior is the prior chosen
    when that filing was reconciled at its own available_at, which may
    differ from the prior recorded in a later version's lineage.
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
                    observations=tuple(observations),
                    fiscal_year=fiscal_identity.fiscal_year,
                    fiscal_quarter=fiscal_quarter,
                    period_end=pd.Timestamp(quarter.iloc[0]["period_end"]),
                    available_at=pd.Timestamp(quarter.iloc[0]["available_at"]),
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

    # A filing whose period_end disagrees with its fiscal period's
    # reference is rejected BEFORE the candidate index is built, so it is
    # never a current, prior, anchor, or trigger.
    rejections = _period_end_rejections(built_filings)

    if rejections:
        rejected_filings = sorted(
            (built for built in built_filings if built.accession_number in rejections),
            key=lambda item: item.accession_number,
        )

        if strict:
            raise EdgarHistoryBuildError(
                "Conflicting period_end values within one fiscal period for accessions "
                f"{[built.accession_number for built in rejected_filings]}."
            )

        # The rejection replaces the filing's success diagnostic.
        diagnostic_rows = [
            row
            for row in diagnostic_rows
            if not (row["accession_number"] in rejections and row["status"] == "success")
        ]

        for built in rejected_filings:
            reference = rejections[built.accession_number]
            diagnostic_rows.append(
                _diagnostic_row(
                    symbol=symbol,
                    provider_symbol=provider_symbol,
                    filing=built.filing_row,
                    status="skipped",
                    reason="conflicting_period_end",
                    detail=(
                        f"fiscal_year={built.fiscal_year}; "
                        f"fiscal_quarter={built.fiscal_quarter}; "
                        f"period_end={built.period_end.date()}; "
                        + (
                            "reference_period_end=none (earliest filings disagree)"
                            if reference is None
                            else f"reference_period_end={reference.date()}"
                        )
                    ),
                )
            )

        built_filings = [
            built for built in built_filings if built.accession_number not in rejections
        ]

    # Cross-filing reconciliation runs only after every filing has been
    # collected, so the result never depends on filing iteration order.
    # The raw candidate index is the ONLY reconciliation input; versions
    # are a pure function of it. Point-in-time safety: each version only
    # receives raw observations available at or before its own time.
    index = _build_candidate_index(built_filings)

    filings_by_period: dict[tuple[int, int], list[_BuiltFiling]] = {}

    for built in built_filings:
        filings_by_period.setdefault(built.fiscal_period, []).append(built)

    versions: list[_Version] = []

    for period in sorted(filings_by_period):
        own_filings = filings_by_period[period]
        fiscal_year, fiscal_quarter = period
        prior_filings = (
            filings_by_period.get((fiscal_year, fiscal_quarter - 1), [])
            if fiscal_quarter > 1
            else []
        )

        versions.extend(_period_versions(own_filings, prior_filings, index))

    reconciliation = _reconciliation_frame(versions)

    reconciled = _reconcile_flows(built_filings, index)

    diagnostic_rows.extend(
        _superseded_input_diagnostics(
            reconciled,
            built_filings,
            symbol=symbol,
            provider_symbol=provider_symbol,
        )
    )

    if versions:
        history = _history_frame(versions)

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
