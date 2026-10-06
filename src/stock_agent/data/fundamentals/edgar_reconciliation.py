from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


class EdgarReconciliationError(ValueError):
    """Raised when fiscal-flow reconciliation inputs are structurally invalid."""


@dataclass(frozen=True)
class FiscalFlowObservation:
    """
    One point-in-time fiscal flow observation from an SEC filing.

    An observation describes what one filing tells us about one metric.

    Examples:
        MU FY2026 Q3 operating_cash_flow:
            direct_value = None
            ytd_value = 13.2B

        MU FY2025 annual revenue:
            direct_value = None
            fy_value = 37.4B

    Cross-filing arithmetic is intentionally NOT performed here.
    """

    symbol: str
    metric_name: str
    fiscal_year: int
    fiscal_quarter: int
    period_end: pd.Timestamp
    available_at: pd.Timestamp
    accession_number: str

    direct_value: float | None = None
    ytd_value: float | None = None
    fy_value: float | None = None


@dataclass(frozen=True)
class FiscalFlowReconciliation:
    """
    Result of reconciling one fiscal flow metric.

    The method and accession numbers preserve lineage so downstream
    systems can explain exactly how a standalone quarterly value was
    obtained.
    """

    symbol: str
    metric_name: str
    fiscal_year: int
    fiscal_quarter: int

    value: float | None
    method: str
    reason: str

    current_accession_number: str
    prior_accession_number: str | None = None
    prior_value_source: str | None = None


@dataclass(frozen=True)
class PriorSelection:
    """
    Result of choosing one prior observation from several candidates.

    prior is None whenever reason != "ok". The accession tuples are
    sorted so the selection is a pure function of the candidate set.

    skipped_prior_accessions:
        eligible candidates that carry no usable prior input value
        (for example an exhibit-only 10-Q/A).

    equivalent_prior_accessions:
        latest-available candidates with exactly equal input values;
        prior is the smallest accession among them.

    conflicting_prior_accessions:
        latest-available candidates whose input values differ.
    """

    prior: FiscalFlowObservation | None
    reason: str
    skipped_prior_accessions: tuple[str, ...] = ()
    equivalent_prior_accessions: tuple[str, ...] = ()
    conflicting_prior_accessions: tuple[str, ...] = ()


def _require_nonempty_string(
    value: str,
    *,
    name: str,
) -> str:
    normalized = str(value).strip()

    if not normalized:
        raise EdgarReconciliationError(
            f"{name} cannot be empty.",
        )

    return normalized


def _normalize_timestamp(
    value: object,
    *,
    name: str,
    normalize: bool,
) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EdgarReconciliationError(
            f"{name} must be a valid timestamp.",
        ) from exc

    if pd.isna(timestamp):
        raise EdgarReconciliationError(
            f"{name} cannot be missing.",
        )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    if normalize:
        timestamp = timestamp.normalize()

    return timestamp


def _optional_float(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EdgarReconciliationError(
            f"{name} must be numeric when supplied.",
        ) from exc

    if not math.isfinite(numeric):
        raise EdgarReconciliationError(
            f"{name} must be finite when supplied.",
        )

    return numeric


def _validate_observation(
    observation: FiscalFlowObservation,
) -> None:
    _require_nonempty_string(
        observation.symbol,
        name="symbol",
    )

    _require_nonempty_string(
        observation.metric_name,
        name="metric_name",
    )

    _require_nonempty_string(
        observation.accession_number,
        name="accession_number",
    )

    if observation.fiscal_year <= 0:
        raise EdgarReconciliationError(
            "fiscal_year must be positive.",
        )

    if observation.fiscal_quarter not in {1, 2, 3, 4}:
        raise EdgarReconciliationError(
            "fiscal_quarter must be between 1 and 4.",
        )

    _normalize_timestamp(
        observation.period_end,
        name="period_end",
        normalize=True,
    )

    _normalize_timestamp(
        observation.available_at,
        name="available_at",
        normalize=False,
    )

    _optional_float(
        observation.direct_value,
        name="direct_value",
    )

    _optional_float(
        observation.ytd_value,
        name="ytd_value",
    )

    _optional_float(
        observation.fy_value,
        name="fy_value",
    )


def _resolve_as_of(
    *,
    current: FiscalFlowObservation,
    as_of: object | None,
) -> pd.Timestamp:
    """
    Return the availability time of the derived observation.

    "current" is the derived observation being produced. Its
    availability is as_of, which defaults to current.available_at and may
    never be earlier: a derived value is available no earlier than the
    latest availability of every input (AGENTS.md).
    """

    current_available_at = _normalize_timestamp(
        current.available_at,
        name="current.available_at",
        normalize=False,
    )

    if as_of is None:
        return current_available_at

    resolved = _normalize_timestamp(
        as_of,
        name="as_of",
        normalize=False,
    )

    if resolved < current_available_at:
        raise EdgarReconciliationError(
            "as_of cannot be earlier than current.available_at.",
        )

    return resolved


def _prior_validation_reason(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation,
    as_of: pd.Timestamp | None = None,
) -> str:
    """
    Return 'ok' when prior is eligible for cross-filing arithmetic.

    This function contains the deterministic leakage and identity
    guardrails used before any subtraction occurs.

    "current" means the derived observation being produced, whose
    availability is as_of (default: current.available_at). A prior is
    point-in-time eligible when prior.available_at <= as_of, consistent
    with AGENTS.md: every derived value is available no earlier than the
    latest availability of every input.
    """

    resolved_as_of = _resolve_as_of(current=current, as_of=as_of)

    current_symbol = _require_nonempty_string(
        current.symbol,
        name="current.symbol",
    )

    prior_symbol = _require_nonempty_string(
        prior.symbol,
        name="prior.symbol",
    )

    if current_symbol != prior_symbol:
        return "different_symbol"

    current_metric = _require_nonempty_string(
        current.metric_name,
        name="current.metric_name",
    )

    prior_metric = _require_nonempty_string(
        prior.metric_name,
        name="prior.metric_name",
    )

    if current_metric != prior_metric:
        return "different_metric"

    if prior.fiscal_year != current.fiscal_year:
        return "different_fiscal_year"

    if current.fiscal_quarter not in {2, 3, 4}:
        return "no_prior_expected"

    expected_prior_quarter = current.fiscal_quarter - 1

    if prior.fiscal_quarter != expected_prior_quarter:
        return "wrong_prior_fiscal_quarter"

    current_accession = _require_nonempty_string(
        current.accession_number,
        name="current.accession_number",
    )

    prior_accession = _require_nonempty_string(
        prior.accession_number,
        name="prior.accession_number",
    )

    if prior_accession == current_accession:
        return "same_accession_number"

    current_period_end = _normalize_timestamp(
        current.period_end,
        name="current.period_end",
        normalize=True,
    )

    prior_period_end = _normalize_timestamp(
        prior.period_end,
        name="prior.period_end",
        normalize=True,
    )

    if prior_period_end >= current_period_end:
        return "prior_period_not_before_current"

    prior_available_at = _normalize_timestamp(
        prior.available_at,
        name="prior.available_at",
        normalize=False,
    )

    if prior_available_at > resolved_as_of:
        return "prior_available_after_current"

    return "ok"


def is_valid_prior_fiscal_observation(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation,
) -> bool:
    """
    Return whether prior may safely participate in reconciliation.

    Safety requires:

    - same symbol
    - same metric
    - same fiscal year
    - immediately preceding fiscal quarter
    - different SEC accession
    - earlier fiscal period
    - no point-in-time lookahead
    """

    _validate_observation(current)
    _validate_observation(prior)

    return (
        _prior_validation_reason(
            current=current,
            prior=prior,
        )
        == "ok"
    )


@dataclass(frozen=True)
class _PriorInput:
    value: float | None
    source: str | None
    reason: str


def _prior_input_value(
    prior: FiscalFlowObservation,
) -> _PriorInput:
    """
    Return the cumulative value a prior contributes to subtraction.

    YTD is always preferred. Only for an authoritative fiscal-Q1
    prior may the direct value stand in for a missing YTD value,
    because standalone Q1 and Q1 YTD cover the same duration.
    Standalone Q2 or Q3 never equals Q2 or Q3 YTD, so no other
    quarter gets this fallback.
    """

    ytd_value = _optional_float(
        prior.ytd_value,
        name="prior.ytd_value",
    )

    if prior.fiscal_quarter != 1:
        if ytd_value is None:
            return _PriorInput(None, None, "missing_prior_ytd")

        return _PriorInput(ytd_value, "ytd", "ok")

    direct_value = _optional_float(
        prior.direct_value,
        name="prior.direct_value",
    )

    if ytd_value is not None and direct_value is not None and ytd_value != direct_value:
        return _PriorInput(None, None, "prior_q1_direct_ytd_conflict")

    if ytd_value is not None:
        return _PriorInput(ytd_value, "ytd", "ok")

    if direct_value is not None:
        return _PriorInput(direct_value, "q1_direct", "ok")

    return _PriorInput(None, None, "missing_prior_ytd")


def select_prior_fiscal_observation(
    *,
    current: FiscalFlowObservation,
    candidates: Iterable[FiscalFlowObservation],
    as_of: pd.Timestamp | None = None,
) -> PriorSelection:
    """
    Choose the prior observation used to reconcile current.

    as_of is the availability of the derived observation being produced
    (default: current.available_at; never earlier).

    Rule:
        1. Keep candidates that pass the prior safety checks,
           including available_at <= as_of.
           Filtering happens before ranking, so a later amendment
           can never win and then be rejected.
        2. Keep candidates with a usable prior input value.
        3. Take the candidates with the latest available_at.
        4. Equal values: choose the smallest accession number.
           Different values: ambiguous, nothing is chosen.

    The result does not depend on candidate order.
    """

    _validate_observation(current)

    resolved_as_of = _resolve_as_of(current=current, as_of=as_of)

    candidate_list = tuple(candidates)

    seen_accessions: set[str] = set()

    for candidate in candidate_list:
        _validate_observation(candidate)

        accession = candidate.accession_number.strip()

        if accession in seen_accessions:
            raise EdgarReconciliationError(
                f"Duplicate prior candidate accession: {accession}.",
            )

        seen_accessions.add(accession)

    if current.fiscal_quarter == 1:
        return PriorSelection(
            prior=None,
            reason="no_prior_expected",
        )

    eligible: list[FiscalFlowObservation] = []
    rejected_only_for_availability = False

    for candidate in candidate_list:
        reason = _prior_validation_reason(
            current=current,
            prior=candidate,
            as_of=resolved_as_of,
        )

        if reason == "ok":
            eligible.append(candidate)
        elif reason == "prior_available_after_current":
            rejected_only_for_availability = True

    valued: list[tuple[FiscalFlowObservation, float]] = []
    skipped: list[str] = []
    unusable_reasons: set[str] = set()

    for candidate in eligible:
        prior_input = _prior_input_value(candidate)

        if prior_input.value is None:
            skipped.append(candidate.accession_number.strip())
            unusable_reasons.add(prior_input.reason)
        else:
            valued.append((candidate, prior_input.value))

    skipped_accessions = tuple(sorted(skipped))

    if not valued:
        if rejected_only_for_availability:
            reason = "prior_available_after_current"
        elif unusable_reasons == {"prior_q1_direct_ytd_conflict"}:
            reason = "prior_q1_direct_ytd_conflict"
        elif eligible:
            reason = "missing_prior_ytd"
        else:
            reason = "missing_prior_fiscal_observation"

        return PriorSelection(
            prior=None,
            reason=reason,
            skipped_prior_accessions=skipped_accessions,
        )

    def _available_at(observation: FiscalFlowObservation) -> pd.Timestamp:
        return _normalize_timestamp(
            observation.available_at,
            name="available_at",
            normalize=False,
        )

    latest_available_at = max(_available_at(candidate) for candidate, _ in valued)

    latest = sorted(
        (
            (candidate.accession_number.strip(), candidate, value)
            for candidate, value in valued
            if _available_at(candidate) == latest_available_at
        ),
        key=lambda item: item[0],
    )

    latest_accessions = tuple(accession for accession, _, _ in latest)

    if len({value for _, _, value in latest}) > 1:
        return PriorSelection(
            prior=None,
            reason="ambiguous_prior_observation",
            skipped_prior_accessions=skipped_accessions,
            conflicting_prior_accessions=latest_accessions,
        )

    return PriorSelection(
        prior=latest[0][1],
        reason="ok",
        skipped_prior_accessions=skipped_accessions,
        equivalent_prior_accessions=latest_accessions,
    )


def find_superseding_prior_observations(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation,
    candidates: Iterable[FiscalFlowObservation],
) -> tuple[FiscalFlowObservation, ...]:
    """
    Return priors that became available AFTER current and would have
    changed the subtraction input that was actually used.

    These are never used to rewrite the derived value: that would
    backdate later information. They exist so the history can flag
    derived quarters whose inputs were later superseded.

    A candidate qualifies when it fails the prior checks only because
    it became available after current, and its prior input value is
    usable and differs from the used prior's input value.

    Sorted by (available_at, accession_number).
    """

    used_value = _prior_input_value(prior).value

    superseding: list[tuple[pd.Timestamp, str, FiscalFlowObservation]] = []

    for candidate in candidates:
        _validate_observation(candidate)

        reason = _prior_validation_reason(
            current=current,
            prior=candidate,
        )

        if reason != "prior_available_after_current":
            continue

        candidate_value = _prior_input_value(candidate).value

        if candidate_value is None or candidate_value == used_value:
            continue

        superseding.append(
            (
                _normalize_timestamp(
                    candidate.available_at,
                    name="available_at",
                    normalize=False,
                ),
                candidate.accession_number.strip(),
                candidate,
            )
        )

    superseding.sort(key=lambda item: (item[0], item[1]))

    return tuple(candidate for _, _, candidate in superseding)


def _result(
    *,
    current: FiscalFlowObservation,
    value: float | None,
    method: str,
    reason: str,
    prior: FiscalFlowObservation | None = None,
    prior_value_source: str | None = None,
) -> FiscalFlowReconciliation:
    return FiscalFlowReconciliation(
        symbol=_require_nonempty_string(
            current.symbol,
            name="symbol",
        ),
        metric_name=_require_nonempty_string(
            current.metric_name,
            name="metric_name",
        ),
        fiscal_year=current.fiscal_year,
        fiscal_quarter=current.fiscal_quarter,
        value=value,
        method=method,
        reason=reason,
        current_accession_number=_require_nonempty_string(
            current.accession_number,
            name="accession_number",
        ),
        prior_accession_number=(
            None
            if prior is None
            else _require_nonempty_string(
                prior.accession_number,
                name="prior.accession_number",
            )
        ),
        prior_value_source=prior_value_source,
    )


def reconcile_fiscal_flow(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation | None = None,
    as_of: pd.Timestamp | None = None,
) -> FiscalFlowReconciliation:
    """
    Convert fiscal flow information into one standalone-quarter value.

    as_of is the availability of the derived value (default:
    current.available_at; never earlier). The prior must be available
    no later than as_of.

    Accounting hierarchy:

        Direct value:
            use the standalone quarterly value exactly as reported.

        Q1:
            Q1 = Q1 YTD

        Q2:
            Q2 = Q2 YTD - Q1 YTD

        Q3:
            Q3 = Q3 YTD - Q2 YTD

        Q4:
            Q4 = FY - Q3 YTD

    Cross-filing subtraction is allowed only when the prior observation
    passes fiscal-identity and point-in-time safety checks.

    Missing information returns a structured unavailable result rather
    than guessing or silently selecting another observation.
    """

    _validate_observation(current)

    resolved_as_of = _resolve_as_of(current=current, as_of=as_of)

    direct_value = _optional_float(
        current.direct_value,
        name="current.direct_value",
    )

    if direct_value is not None:
        return _result(
            current=current,
            value=direct_value,
            method="direct_quarter",
            reason="ok",
        )

    if current.fiscal_quarter == 1:
        current_ytd = _optional_float(
            current.ytd_value,
            name="current.ytd_value",
        )

        if current_ytd is None:
            return _result(
                current=current,
                value=None,
                method="unavailable",
                reason="missing_current_ytd",
            )

        return _result(
            current=current,
            value=current_ytd,
            method="q1_ytd",
            reason="ok",
        )

    # The current filing's own input is checked before any prior, so a
    # missing current value is never reported as a prior problem.
    if current.fiscal_quarter in {2, 3}:
        current_ytd = _optional_float(
            current.ytd_value,
            name="current.ytd_value",
        )

        if current_ytd is None:
            return _result(
                current=current,
                value=None,
                method="unavailable",
                reason="missing_current_ytd",
            )

    if current.fiscal_quarter == 4:
        current_fy = _optional_float(
            current.fy_value,
            name="current.fy_value",
        )

        if current_fy is None:
            return _result(
                current=current,
                value=None,
                method="unavailable",
                reason="missing_current_fy",
            )

    if prior is None:
        return _result(
            current=current,
            value=None,
            method="unavailable",
            reason="missing_prior_fiscal_observation",
        )

    _validate_observation(prior)

    prior_reason = _prior_validation_reason(
        current=current,
        prior=prior,
        as_of=resolved_as_of,
    )

    if prior_reason != "ok":
        return _result(
            current=current,
            prior=prior,
            value=None,
            method="unavailable",
            reason=prior_reason,
        )

    prior_input = _prior_input_value(prior)

    if prior_input.value is None:
        return _result(
            current=current,
            prior=prior,
            value=None,
            method="unavailable",
            reason=prior_input.reason,
        )

    prior_ytd = prior_input.value

    if current.fiscal_quarter in {2, 3}:
        return _result(
            current=current,
            prior=prior,
            value=current_ytd - prior_ytd,
            method="ytd_minus_prior_ytd",
            reason="ok",
            prior_value_source=prior_input.source,
        )

    if current.fiscal_quarter == 4:
        return _result(
            current=current,
            prior=prior,
            value=current_fy - prior_ytd,
            method="fy_minus_q3_ytd",
            reason="ok",
            prior_value_source=prior_input.source,
        )

    raise EdgarReconciliationError(
        "Unsupported fiscal quarter.",
    )


@dataclass(frozen=True)
class CandidateReconciliation:
    """
    Reconciliation of one observation against its prior candidates.

    selection is None when no prior was needed (direct value, fiscal
    Q1, nothing reported, or a missing current input).
    """

    result: FiscalFlowReconciliation
    selection: PriorSelection | None


def reconcile_fiscal_flow_from_candidates(
    *,
    current: FiscalFlowObservation,
    candidates: Iterable[FiscalFlowObservation],
    as_of: pd.Timestamp | None = None,
) -> CandidateReconciliation:
    """
    Reconcile current, selecting a prior only when subtraction is needed.

    as_of is the availability of the derived value (default:
    current.available_at; never earlier). Only priors available by
    as_of may be selected.

    Order:
        1. Nothing reported for this metric -> no_reported_value.
        2. Direct standalone value, or fiscal Q1 -> no prior needed.
        3. Missing current YTD (Q2/Q3) or FY (Q4) -> unavailable,
           reported as the current filing's problem.
        4. Otherwise select a prior; if none qualifies, the selection
           reason is the unavailable reason.
    """

    _validate_observation(current)

    resolved_as_of = _resolve_as_of(current=current, as_of=as_of)

    reported = (
        current.direct_value,
        current.ytd_value,
        current.fy_value,
    )

    if all(_optional_float(value, name="value") is None for value in reported):
        return CandidateReconciliation(
            result=_result(
                current=current,
                value=None,
                method="unavailable",
                reason="no_reported_value",
            ),
            selection=None,
        )

    without_prior = reconcile_fiscal_flow(current=current, as_of=resolved_as_of)

    if without_prior.reason != "missing_prior_fiscal_observation":
        return CandidateReconciliation(
            result=without_prior,
            selection=None,
        )

    selection = select_prior_fiscal_observation(
        current=current,
        candidates=candidates,
        as_of=resolved_as_of,
    )

    if selection.prior is None:
        return CandidateReconciliation(
            result=_result(
                current=current,
                value=None,
                method="unavailable",
                reason=selection.reason,
            ),
            selection=selection,
        )

    return CandidateReconciliation(
        result=reconcile_fiscal_flow(
            current=current,
            prior=selection.prior,
            as_of=resolved_as_of,
        ),
        selection=selection,
    )


@dataclass(frozen=True)
class CurrentSelection:
    """
    Result of choosing the current-side observation for one fiscal period.

    current is None whenever reason != "ok". Accession tuples are sorted
    so the selection is a pure function of the candidate set.

    skipped_current_accessions:
        candidates available by as_of that carry no usable current-side
        input (for example a partial amendment omitting this metric).

    equivalent_current_accessions:
        latest-available usable candidates with equal inputs; current is
        the smallest accession among them.

    conflicting_current_accessions:
        latest-available usable candidates whose inputs differ.
    """

    current: FiscalFlowObservation | None
    reason: str
    skipped_current_accessions: tuple[str, ...] = ()
    equivalent_current_accessions: tuple[str, ...] = ()
    conflicting_current_accessions: tuple[str, ...] = ()


_USABLE_CURRENT_REASONS = frozenset({"ok", "missing_prior_fiscal_observation"})


def _current_input_reason(
    observation: FiscalFlowObservation,
) -> str:
    """
    Reason reconciliation gives for observation's own current-side input.

    "ok" or "missing_prior_fiscal_observation" means the current input
    is usable (direct value; else YTD for Q1-Q3; else FY for Q4), exactly
    as reconcile_fiscal_flow decides. Nothing reported at all is
    "no_reported_value", as in reconcile_fiscal_flow_from_candidates.
    """

    reported = (
        observation.direct_value,
        observation.ytd_value,
        observation.fy_value,
    )

    if all(_optional_float(value, name="value") is None for value in reported):
        return "no_reported_value"

    return reconcile_fiscal_flow(current=observation).reason


def _current_inputs(
    observation: FiscalFlowObservation,
) -> tuple[float | None, float | None, float | None]:
    # _optional_float maps None and NaN to None, so comparison is NaN-safe.
    return (
        _optional_float(observation.direct_value, name="direct_value"),
        _optional_float(observation.ytd_value, name="ytd_value"),
        _optional_float(observation.fy_value, name="fy_value"),
    )


def select_current_fiscal_observation(
    *,
    candidates: Iterable[FiscalFlowObservation],
    as_of: pd.Timestamp,
) -> CurrentSelection:
    """
    Choose the current-side observation for one fiscal period at as_of.

    candidates are raw observations for one (symbol, metric, fiscal_year,
    fiscal_quarter); mixed identity or duplicate accessions raise.

    Rule:
        1. Keep candidates with available_at <= as_of.
        2. Skip candidates without a usable current-side input, so a
           partial amendment never erases an earlier valid value.
        3. Among usable candidates take the latest available_at.
           Equal inputs: smallest accession. Different inputs:
           "ambiguous_current_observation", nothing is chosen.
        4. No usable candidate: the reason reconciliation gives for the
           latest available candidate (tie: smallest accession), or
           "missing_current_observation" when nothing is available.

    The result does not depend on candidate order.
    """

    resolved_as_of = _normalize_timestamp(
        as_of,
        name="as_of",
        normalize=False,
    )

    candidate_list = tuple(candidates)

    seen_accessions: set[str] = set()
    identities: set[tuple[str, str, int, int]] = set()

    for candidate in candidate_list:
        _validate_observation(candidate)

        accession = candidate.accession_number.strip()

        if accession in seen_accessions:
            raise EdgarReconciliationError(
                f"Duplicate current candidate accession: {accession}.",
            )

        seen_accessions.add(accession)

        identities.add(
            (
                candidate.symbol.strip(),
                candidate.metric_name.strip(),
                candidate.fiscal_year,
                candidate.fiscal_quarter,
            )
        )

    if len(identities) > 1:
        raise EdgarReconciliationError(
            "Current candidates must share one (symbol, metric, fiscal_year, "
            "fiscal_quarter) identity.",
        )

    def _available_at(observation: FiscalFlowObservation) -> pd.Timestamp:
        return _normalize_timestamp(
            observation.available_at,
            name="available_at",
            normalize=False,
        )

    available = sorted(
        (
            (_available_at(candidate), candidate.accession_number.strip(), candidate)
            for candidate in candidate_list
            if _available_at(candidate) <= resolved_as_of
        ),
        key=lambda item: (item[0], item[1]),
    )

    if not available:
        return CurrentSelection(
            current=None,
            reason="missing_current_observation",
        )

    usable: list[tuple[pd.Timestamp, str, FiscalFlowObservation]] = []
    skipped: list[str] = []
    unusable_reasons: dict[str, str] = {}

    for available_at, accession, candidate in available:
        reason = _current_input_reason(candidate)

        if reason in _USABLE_CURRENT_REASONS:
            usable.append((available_at, accession, candidate))
        else:
            skipped.append(accession)
            unusable_reasons[accession] = reason

    skipped_accessions = tuple(sorted(skipped))

    if not usable:
        latest_available_at = available[-1][0]
        latest_accession = min(
            accession
            for available_at, accession, _ in available
            if available_at == latest_available_at
        )

        return CurrentSelection(
            current=None,
            reason=unusable_reasons[latest_accession],
            skipped_current_accessions=skipped_accessions,
        )

    latest_available_at = usable[-1][0]

    latest = [
        (accession, candidate)
        for available_at, accession, candidate in usable
        if available_at == latest_available_at
    ]

    latest_accessions = tuple(accession for accession, _ in latest)

    if len({_current_inputs(candidate) for _, candidate in latest}) > 1:
        return CurrentSelection(
            current=None,
            reason="ambiguous_current_observation",
            skipped_current_accessions=skipped_accessions,
            conflicting_current_accessions=latest_accessions,
        )

    return CurrentSelection(
        current=latest[0][1],
        reason="ok",
        skipped_current_accessions=skipped_accessions,
        equivalent_current_accessions=latest_accessions,
    )
