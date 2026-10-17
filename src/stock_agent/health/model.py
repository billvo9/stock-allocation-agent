"""
The health vocabulary: five states and their precedence, one check, one
alert, and the versioned summary with its strict, fail-closed reader.

States, worst first. The overall state is the first state in this order that
any check has:

    failed                 measured, and a hard limit or contract is broken
    unavailable            a measurement that should exist could not be taken
                           (an input is unreadable or a tool did not run); it
                           can hide a failure, so it outranks degraded
    degraded               measured, outside a warning threshold, still usable
    insufficient_evidence  nothing to judge yet: the input does not exist, or
                           there are too few observations
    healthy                measured, within its threshold

A metric that was never collected is never healthy: a healthy check must have
both `measured_at` and `value`, and a summary without checks is
insufficient_evidence.

Alerts are informational only. Every alert carries `automatic_action: "none"`
and the summary carries `alert_policy: "informational_only"`; the reader
refuses anything else. Nothing in this package trades, retrains, promotes,
deletes or changes configuration.

Standard library only, so the Model Health page loads this module without
pandas or any collector.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

FORMAT = "stock_agent.health_summary"
SCHEMA_VERSION = "1.0"
SUPPORTED_VERSIONS = frozenset({SCHEMA_VERSION})
ALERT_POLICY = "informational_only"
AUTOMATIC_ACTION = "none"
RUNBOOK = "docs/operations/runbook.md"


class HealthState(StrEnum):
    HEALTHY = "healthy"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


# Worst first (see the module docstring for the reasons).
PRECEDENCE = (
    HealthState.FAILED,
    HealthState.UNAVAILABLE,
    HealthState.DEGRADED,
    HealthState.INSUFFICIENT_EVIDENCE,
    HealthState.HEALTHY,
)
_RANK = {state: rank for rank, state in enumerate(PRECEDENCE)}


def worse(first: HealthState, second: HealthState) -> bool:
    """True when `first` is strictly worse than `second`."""

    return _RANK[first] < _RANK[second]


def worst(states: Iterable[HealthState]) -> HealthState:
    """The overall state: the worst state present; insufficient_evidence for none."""

    present = list(states)
    if not present:
        return HealthState.INSUFFICIENT_EVIDENCE
    return min(present, key=_RANK.__getitem__)


class Condition(StrEnum):
    """
    What a check's state and reason mean to an operator, worst first. It is
    derived from the stored state and reason, never stored, so every reader of a
    summary derives the same condition and the summary format is unchanged:

        failed         the state failed
        unavailable    the state unavailable
        degraded       degraded for any reason but staleness
        stale          degraded because the data or evidence is older than its
                       limit (STALE_REASONS)
        no_evidence    insufficient_evidence: no input yet, or too few observations
        not_activated  insufficient_evidence because the producer is not
                       authorized to run yet (NOT_ACTIVATED_REASONS)
        healthy        the state healthy
    """

    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    STALE = "stale"
    NO_EVIDENCE = "no_evidence"
    NOT_ACTIVATED = "not_activated"
    HEALTHY = "healthy"


# Worst first; it keeps the state precedence (every condition of a worse state is worse).
CONDITION_PRECEDENCE = tuple(Condition)
# Degraded reasons that mean "measured, but older than its limit" (tested against
# every `stale_*` reason the checks emit).
STALE_REASONS = frozenset({"stale_data", "stale_evidence", "stale_prices", "daily_run_overdue"})
NOT_ACTIVATED_REASONS = frozenset({"not_authorized"})


def condition(state: HealthState, reason: str | None) -> Condition:
    """The operator condition of a state and its reason (see `Condition`)."""

    state = HealthState(state)
    if state is HealthState.DEGRADED and reason in STALE_REASONS:
        return Condition.STALE
    if state is HealthState.INSUFFICIENT_EVIDENCE:
        if reason in NOT_ACTIVATED_REASONS:
            return Condition.NOT_ACTIVATED
        return Condition.NO_EVIDENCE
    return Condition(state.value)


class Area(StrEnum):
    """The prefix of every check id in that area."""

    CI = "ci"
    DEPS = "deps"
    CONTRACT = "contract"
    SMOKE = "smoke"
    DATA = "data"
    TRAINING = "training"
    FORECAST = "forecast"
    REGISTRY = "registry"
    PORTFOLIO = "portfolio"


class Scope(StrEnum):
    # Built from synthetic fixtures and repository code only: safe to publish.
    SYNTHETIC = "synthetic"
    # Built on the owner's machine from private stores: never leaves it.
    PRIVATE = "private"


class Better(StrEnum):
    """Which direction of a numeric value is an improvement (for deterioration)."""

    HIGHER = "higher"
    LOWER = "lower"


class AlertKind(StrEnum):
    FAILURE = "failure"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNAVAILABLE_DATA = "unavailable_data"
    DEGRADED = "degraded"
    DETERIORATION = "deterioration"


# A breach of a stored limit (for example a portfolio cap) uses this reason, so
# its alert is a constraint violation rather than a generic failure.
CONSTRAINT_BREACHED = "constraint_breached"

# Run-context keys a summary may carry (never a token or any other variable).
RUN_CONTEXT_KEYS = frozenset(
    {
        "environment",
        "workflow",
        "run_id",
        "run_number",
        "run_attempt",
        "event",
        "ref",
        "sha",
        "repository",
        "run_url",
        "runner_os",
    }
)

_TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")
_CHECK_ID_PATTERN = re.compile(r"([a-z]+)\.[a-z][a-z0-9_]{0,47}")
_REASON_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_ANCHOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class SummaryRefusal(StrEnum):
    NOT_FOUND = "not_found"
    INVALID_JSON = "invalid_json"
    NOT_A_SUMMARY = "not_a_summary"
    UNSUPPORTED_VERSION = "unsupported_version"
    FIELD_SET = "field_set"
    INVALID_FIELD = "invalid_field"
    DUPLICATE_CHECK = "duplicate_check"
    STATE_MISMATCH = "state_mismatch"
    POLICY_VIOLATION = "policy_violation"


class SummaryRefused(Exception):
    """A health summary was refused; `reason` is machine-readable."""

    def __init__(self, reason: SummaryRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def _refuse(reason: SummaryRefusal, detail: str) -> SummaryRefused:
    return SummaryRefused(reason, detail)


# --- timestamps and strict JSON ----------------------------------------------


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"timestamp {value!r} is not UTC")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp(label: str, value: object) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_PATTERN.fullmatch(value):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} {value!r} is not a UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} {value!r} is not a date") from error


def _reject_constant(token: str) -> Any:
    raise _refuse(SummaryRefusal.INVALID_JSON, f"non-finite number {token}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _refuse(SummaryRefusal.INVALID_JSON, f"duplicate key {key!r}")
        result[key] = value
    return result


def loads_strict(text: str) -> Any:
    """JSON with no NaN or Infinity and no duplicate keys."""

    try:
        return json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except json.JSONDecodeError as error:
        raise _refuse(SummaryRefusal.INVALID_JSON, str(error)) from error


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"


# --- field checks --------------------------------------------------------------


def _text(label: str, value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or _CONTROL.search(value):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} must be non-empty plain text")
    return value


def _enum(label: str, enum_type: type[StrEnum], value: object) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} {value!r} is not allowed") from error


def _number(label: str, value: object, *, optional: bool = True) -> float | int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} must be finite")
    return value


def _value(label: str, value: object) -> float | int | str | None:
    if isinstance(value, str):
        return _text(label, value)
    return _number(label, value)


def runbook_for(check_id: str) -> str:
    """The runbook anchor of a check: `ci.verify` -> `...runbook.md#ci-verify`."""

    return f"{RUNBOOK}#{check_id.replace('.', '-').replace('_', '-')}"


def _runbook(label: str, value: object) -> str:
    text = _text(label, value)
    path, _, anchor = text.partition("#")
    if path != RUNBOOK or not _ANCHOR_PATTERN.fullmatch(anchor):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} {text!r} is not a runbook anchor")
    return text


# --- records -------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """
    One measured (or unmeasurable) health check.

    `summary` is plain words for anyone; `detail` is technical. `reason` is a
    machine-readable code for any state other than healthy. `evidence` points
    at what was measured: a path relative to the summary file, an absolute
    path (private summaries only), or a `fixture:` / `module:` reference.
    `better` and `tolerance` let a later summary flag deterioration of a
    numeric value that is still inside its threshold.
    """

    check_id: str
    area: Area
    title: str
    state: HealthState
    measured_at: datetime | None
    value: float | int | str | None
    unit: str | None
    threshold: str | None
    summary: str
    detail: str
    evidence: str | None
    reason: str | None = None
    better: Better | None = None
    tolerance: float | None = None
    runbook: str = ""

    def __post_init__(self) -> None:
        match = (
            _CHECK_ID_PATTERN.fullmatch(self.check_id) if isinstance(self.check_id, str) else None
        )
        if match is None:
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"check_id {self.check_id!r}")
        area = _enum("area", Area, self.area)
        if match.group(1) != area.value:
            raise _refuse(
                SummaryRefusal.INVALID_FIELD, f"{self.check_id} is not in area {area.value}"
            )
        object.__setattr__(self, "area", area)
        state = _enum("state", HealthState, self.state)
        object.__setattr__(self, "state", state)
        if self.measured_at is not None:
            format_timestamp(self.measured_at)
        _value("value", self.value)
        for label in ("unit", "threshold", "evidence"):
            _text(label, getattr(self, label), optional=True)
        _text("title", self.title)
        _text("summary", self.summary)
        _text("detail", self.detail)
        if self.reason is not None and not (
            isinstance(self.reason, str) and _REASON_PATTERN.fullmatch(self.reason)
        ):
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"reason {self.reason!r}")
        if self.better is not None:
            object.__setattr__(self, "better", _enum("better", Better, self.better))
        _number("tolerance", self.tolerance)
        if not self.runbook:
            object.__setattr__(self, "runbook", runbook_for(self.check_id))
        _runbook("runbook", self.runbook)
        if state is HealthState.HEALTHY:
            if self.measured_at is None or self.value is None:
                raise _refuse(
                    SummaryRefusal.POLICY_VIOLATION,
                    f"{self.check_id} is healthy without a measurement",
                )
        elif self.reason is None:
            raise _refuse(
                SummaryRefusal.INVALID_FIELD, f"{self.check_id} is {state} without a reason"
            )

    @property
    def condition(self) -> Condition:
        """The operator condition of this check's state and reason (derived, not stored)."""

        return condition(self.state, self.reason)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "area": self.area.value,
            "title": self.title,
            "state": self.state.value,
            "reason": self.reason,
            "measured_at": None if self.measured_at is None else format_timestamp(self.measured_at),
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "better": None if self.better is None else self.better.value,
            "tolerance": self.tolerance,
            "summary": self.summary,
            "detail": self.detail,
            "runbook": self.runbook,
            "evidence": self.evidence,
        }

    @classmethod
    def from_json_dict(cls, data: object) -> Check:
        fields = _require_fields("check", data, CHECK_FIELDS)
        measured_at = fields["measured_at"]
        return cls(
            check_id=fields["check_id"],
            area=fields["area"],
            title=fields["title"],
            state=fields["state"],
            reason=fields["reason"],
            measured_at=None
            if measured_at is None
            else parse_timestamp("measured_at", measured_at),
            value=fields["value"],
            unit=fields["unit"],
            threshold=fields["threshold"],
            better=fields["better"],
            tolerance=fields["tolerance"],
            summary=fields["summary"],
            detail=fields["detail"],
            runbook=fields["runbook"],
            evidence=fields["evidence"],
        )


CHECK_FIELDS = frozenset(
    {
        "check_id",
        "area",
        "title",
        "state",
        "reason",
        "measured_at",
        "value",
        "unit",
        "threshold",
        "better",
        "tolerance",
        "summary",
        "detail",
        "runbook",
        "evidence",
    }
)


@dataclass(frozen=True)
class Alert:
    """An informational alert about one check. It never triggers an action."""

    kind: AlertKind
    check_id: str
    state: HealthState
    message: str
    raised_at: datetime
    automatic_action: str = AUTOMATIC_ACTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum("kind", AlertKind, self.kind))
        object.__setattr__(self, "state", _enum("state", HealthState, self.state))
        if not _CHECK_ID_PATTERN.fullmatch(str(self.check_id)):
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"alert check_id {self.check_id!r}")
        _text("message", self.message)
        format_timestamp(self.raised_at)
        if self.automatic_action != AUTOMATIC_ACTION:
            raise _refuse(
                SummaryRefusal.POLICY_VIOLATION,
                f"alerts are informational; automatic_action {self.automatic_action!r}",
            )

    @property
    def alert_id(self) -> str:
        return f"{self.kind.value}:{self.check_id}"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "kind": self.kind.value,
            "check_id": self.check_id,
            "state": self.state.value,
            "message": self.message,
            "raised_at": format_timestamp(self.raised_at),
            "automatic_action": self.automatic_action,
        }

    @classmethod
    def from_json_dict(cls, data: object) -> Alert:
        fields = _require_fields("alert", data, ALERT_FIELDS)
        alert = cls(
            kind=fields["kind"],
            check_id=fields["check_id"],
            state=fields["state"],
            message=fields["message"],
            raised_at=parse_timestamp("raised_at", fields["raised_at"]),
            automatic_action=fields["automatic_action"],
        )
        if fields["alert_id"] != alert.alert_id:
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"alert_id {fields['alert_id']!r}")
        return alert


ALERT_FIELDS = frozenset(
    {"alert_id", "kind", "check_id", "state", "message", "raised_at", "automatic_action"}
)


@dataclass(frozen=True)
class NotApplicable:
    """
    A check that does not apply to what was measured (for example interval
    coverage for a point forecast). It is listed and shown, has no state, and
    never counts towards the overall state.
    """

    check_id: str
    area: Area
    title: str
    reason: str
    detail: str

    def __post_init__(self) -> None:
        match = _CHECK_ID_PATTERN.fullmatch(str(self.check_id))
        area = _enum("area", Area, self.area)
        if match is None or match.group(1) != area.value:
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"not-applicable {self.check_id!r}")
        object.__setattr__(self, "area", area)
        _text("title", self.title)
        _text("detail", self.detail)
        if not (isinstance(self.reason, str) and _REASON_PATTERN.fullmatch(self.reason)):
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"reason {self.reason!r}")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "area": self.area.value,
            "title": self.title,
            "reason": self.reason,
            "detail": self.detail,
        }

    @classmethod
    def from_json_dict(cls, data: object) -> NotApplicable:
        fields = _require_fields("not_applicable entry", data, NOT_APPLICABLE_FIELDS)
        return cls(**fields)


NOT_APPLICABLE_FIELDS = frozenset({"check_id", "area", "title", "reason", "detail"})


@dataclass(frozen=True)
class HealthSummary:
    """
    One versioned health summary. `overall_state` and the state counts are
    derived from the checks; the reader recomputes and compares them.
    `code_sha` and `code_dirty` are the producer's claims (None when git was
    not available), not authenticated provenance.
    """

    generated_at: datetime
    code_sha: str | None
    code_dirty: bool | None
    scope: Scope
    run_context: Mapping[str, str | None]
    checks: tuple[Check, ...]
    alerts: tuple[Alert, ...] = ()
    not_applicable: tuple[NotApplicable, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        format_timestamp(self.generated_at)
        if self.code_sha is not None and not _SHA_PATTERN.fullmatch(str(self.code_sha)):
            raise _refuse(SummaryRefusal.INVALID_FIELD, f"code_sha {self.code_sha!r}")
        if self.code_dirty is not None and not isinstance(self.code_dirty, bool):
            raise _refuse(SummaryRefusal.INVALID_FIELD, "code_dirty must be a boolean or null")
        object.__setattr__(self, "scope", _enum("scope", Scope, self.scope))
        context = dict(self.run_context)
        if not set(context) <= RUN_CONTEXT_KEYS or "environment" not in context:
            raise _refuse(
                SummaryRefusal.POLICY_VIOLATION,
                f"run_context keys {sorted(context)} (allowed {sorted(RUN_CONTEXT_KEYS)})",
            )
        for key, value in context.items():
            _text(f"run_context.{key}", value, optional=key != "environment")
        object.__setattr__(self, "run_context", context)
        if self.schema_version not in SUPPORTED_VERSIONS:
            raise _refuse(SummaryRefusal.UNSUPPORTED_VERSION, repr(self.schema_version))
        ids = [check.check_id for check in self.checks]
        ids_all = ids + [item.check_id for item in self.not_applicable]
        duplicates = sorted({check_id for check_id in ids_all if ids_all.count(check_id) > 1})
        if duplicates:
            raise _refuse(SummaryRefusal.DUPLICATE_CHECK, f"checks repeat: {duplicates}")
        for alert in self.alerts:
            if alert.check_id not in ids:
                raise _refuse(SummaryRefusal.INVALID_FIELD, f"alert for unknown {alert.check_id}")

    @property
    def overall_state(self) -> HealthState:
        return worst(check.state for check in self.checks)

    def state_counts(self) -> dict[str, int]:
        return {
            state.value: sum(check.state is state for check in self.checks) for state in PRECEDENCE
        }

    def check(self, check_id: str) -> Check | None:
        return next((check for check in self.checks if check.check_id == check_id), None)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "schema_version": self.schema_version,
            "generated_at": format_timestamp(self.generated_at),
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
            "scope": self.scope.value,
            "run_context": dict(self.run_context),
            "precedence": [state.value for state in PRECEDENCE],
            "overall_state": self.overall_state.value,
            "state_counts": self.state_counts(),
            "alert_policy": ALERT_POLICY,
            "checks": [check.to_json_dict() for check in self.checks],
            "not_applicable": [item.to_json_dict() for item in self.not_applicable],
            "alerts": [alert.to_json_dict() for alert in self.alerts],
        }

    @classmethod
    def from_json_dict(cls, data: object) -> HealthSummary:
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise _refuse(SummaryRefusal.NOT_A_SUMMARY, f"format is not {FORMAT!r}")
        version = data.get("schema_version")
        if version not in SUPPORTED_VERSIONS:
            raise _refuse(SummaryRefusal.UNSUPPORTED_VERSION, repr(version))
        fields = _require_fields("summary", data, SUMMARY_FIELDS)
        if fields["alert_policy"] != ALERT_POLICY:
            raise _refuse(
                SummaryRefusal.POLICY_VIOLATION, f"alert_policy {fields['alert_policy']!r}"
            )
        if fields["precedence"] != [state.value for state in PRECEDENCE]:
            raise _refuse(SummaryRefusal.INVALID_FIELD, "precedence differs from this version's")
        for name in ("checks", "not_applicable", "alerts"):
            if not isinstance(fields[name], list):
                raise _refuse(SummaryRefusal.INVALID_FIELD, f"{name} must be a list")
        if not isinstance(fields["run_context"], dict):
            raise _refuse(SummaryRefusal.INVALID_FIELD, "run_context must be an object")
        summary = cls(
            schema_version=version,
            generated_at=parse_timestamp("generated_at", fields["generated_at"]),
            code_sha=fields["code_sha"],
            code_dirty=fields["code_dirty"],
            scope=fields["scope"],
            run_context=fields["run_context"],
            checks=tuple(Check.from_json_dict(item) for item in fields["checks"]),
            alerts=tuple(Alert.from_json_dict(item) for item in fields["alerts"]),
            not_applicable=tuple(
                NotApplicable.from_json_dict(item) for item in fields["not_applicable"]
            ),
        )
        if fields["overall_state"] != summary.overall_state.value:
            raise _refuse(
                SummaryRefusal.STATE_MISMATCH,
                f"overall_state {fields['overall_state']!r}, checks give "
                f"{summary.overall_state.value!r}",
            )
        if fields["state_counts"] != summary.state_counts():
            raise _refuse(SummaryRefusal.STATE_MISMATCH, "state_counts differ from the checks")
        return summary


SUMMARY_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "generated_at",
        "code_sha",
        "code_dirty",
        "scope",
        "run_context",
        "precedence",
        "overall_state",
        "state_counts",
        "alert_policy",
        "checks",
        "not_applicable",
        "alerts",
    }
)


def _require_fields(label: str, value: object, expected: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _refuse(SummaryRefusal.INVALID_FIELD, f"{label} is not a JSON object")
    if set(value) != expected:
        missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
        raise _refuse(SummaryRefusal.FIELD_SET, f"{label}: missing {missing}, unexpected {extra}")
    return value


def read_summary(path: str | Path) -> HealthSummary:
    """Read and verify one summary file; any defect refuses the whole file."""

    file = Path(path)
    if not file.is_file():
        raise _refuse(SummaryRefusal.NOT_FOUND, str(file))
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise _refuse(SummaryRefusal.INVALID_JSON, f"{file.name} is not UTF-8") from error
    return HealthSummary.from_json_dict(loads_strict(text))


__all__ = [
    "CONDITION_PRECEDENCE",
    "PRECEDENCE",
    "Alert",
    "AlertKind",
    "Area",
    "Better",
    "Check",
    "Condition",
    "HealthState",
    "HealthSummary",
    "NotApplicable",
    "Scope",
    "SummaryRefused",
    "condition",
    "read_summary",
    "worst",
]
