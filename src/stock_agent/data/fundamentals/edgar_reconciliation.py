from __future__ import annotations

import math
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


def _prior_validation_reason(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation,
) -> str:
    """
    Return 'ok' when prior is eligible for cross-filing arithmetic.

    This function contains the deterministic leakage and identity
    guardrails used before any subtraction occurs.
    """

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

    current_available_at = _normalize_timestamp(
        current.available_at,
        name="current.available_at",
        normalize=False,
    )

    prior_available_at = _normalize_timestamp(
        prior.available_at,
        name="prior.available_at",
        normalize=False,
    )

    if prior_available_at > current_available_at:
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


def _result(
    *,
    current: FiscalFlowObservation,
    value: float | None,
    method: str,
    reason: str,
    prior: FiscalFlowObservation | None = None,
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
    )


def reconcile_fiscal_flow(
    *,
    current: FiscalFlowObservation,
    prior: FiscalFlowObservation | None = None,
) -> FiscalFlowReconciliation:
    """
    Convert fiscal flow information into one standalone-quarter value.

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
    )

    if prior_reason != "ok":
        return _result(
            current=current,
            prior=prior,
            value=None,
            method="unavailable",
            reason=prior_reason,
        )

    prior_ytd = _optional_float(
        prior.ytd_value,
        name="prior.ytd_value",
    )

    if prior_ytd is None:
        return _result(
            current=current,
            prior=prior,
            value=None,
            method="unavailable",
            reason="missing_prior_ytd",
        )

    if current.fiscal_quarter in {2, 3}:
        current_ytd = _optional_float(
            current.ytd_value,
            name="current.ytd_value",
        )

        if current_ytd is None:
            return _result(
                current=current,
                prior=prior,
                value=None,
                method="unavailable",
                reason="missing_current_ytd",
            )

        return _result(
            current=current,
            prior=prior,
            value=current_ytd - prior_ytd,
            method="ytd_minus_prior_ytd",
            reason="ok",
        )

    if current.fiscal_quarter == 4:
        current_fy = _optional_float(
            current.fy_value,
            name="current.fy_value",
        )

        if current_fy is None:
            return _result(
                current=current,
                prior=prior,
                value=None,
                method="unavailable",
                reason="missing_current_fy",
            )

        return _result(
            current=current,
            prior=prior,
            value=current_fy - prior_ytd,
            method="fy_minus_q3_ytd",
            reason="ok",
        )

    raise EdgarReconciliationError(
        "Unsupported fiscal quarter.",
    )
