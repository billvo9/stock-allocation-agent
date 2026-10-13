"""
The publication envelope and the primitives shared by the writer and the
reader: closed enums, structured refusals, strict JSON, the one timestamp
format, publication ids, the lockbox check and the interim knowledge-time
rule.

Nothing here reads a clock, the network or a file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

SCHEMA_VERSION = "1.0"
SUPPORTED_VERSIONS = (SCHEMA_VERSION,)

_VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)")
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_CODE_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
# Every pattern is applied with fullmatch: `$` alone would accept a trailing newline.
PUBLICATION_ID_PATTERN = re.compile(r"\d{8}T\d{12}Z-[0-9a-f]{12}")

MANIFEST_FILE = "manifest.json"
ENVELOPE_FILE = "envelope.json"
ROWS_FILE = "rows.parquet"


class EvidenceStatus(StrEnum):
    DESCRIPTIVE = "descriptive"
    RULES_ILLUSTRATION = "rules_illustration"


class Kind(StrEnum):
    SYNTHETIC_DESCRIPTOR = "synthetic_descriptor"
    SYNTHETIC_RULE_OUTPUT = "synthetic_rule_output"


class ClaimType(StrEnum):
    DESCRIPTIVE_FACT = "descriptive_fact"
    RULE_ILLUSTRATION = "rule_illustration"


class KnowledgeTimeRule(StrEnum):
    # Synthetic dates only (owner decision 2026-10-13); not a production rule.
    INTERIM_SYNTHETIC_SESSION_END = "interim_synthetic_session_end"


class Limitation(StrEnum):
    SYNTHETIC_DATA = "synthetic_data"
    SYNTHETIC_CALENDAR = "synthetic_calendar"


class RowStatus(StrEnum):
    OK = "ok"
    UNAVAILABLE = "unavailable"


class RowReason(StrEnum):
    MISSING_INPUT = "missing_input"


class RefusalReason(StrEnum):
    NOT_A_PUBLICATION = "not_a_publication"
    SYMLINK = "symlink"
    FILE_SET = "file_set"
    INVALID_JSON = "invalid_json"
    MALFORMED_VERSION = "malformed_version"
    UNSUPPORTED_VERSION = "unsupported_version"
    FIELD_SET = "field_set"
    INVALID_FIELD = "invalid_field"
    INVALID_TIMESTAMP = "invalid_timestamp"
    UNSUPPORTED_KIND = "unsupported_kind"
    UNKNOWN_STATUS = "unknown_status"
    STATUS_NOT_ALLOWED = "status_not_allowed"
    KNOWLEDGE_RULE_NOT_PERMITTED = "knowledge_rule_not_permitted"
    MISSING_LIMITATION = "missing_limitation"
    HASH_MISMATCH = "hash_mismatch"
    ID_MISMATCH = "id_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ROW = "invalid_row"
    INVALID_ASSET_ID = "invalid_asset_id"
    DIRTY_TREE = "dirty_tree"
    LOCKBOX_VIOLATION = "lockbox_violation"
    KNOWLEDGE_TIME_MISMATCH = "knowledge_time_mismatch"
    KNOWLEDGE_AFTER_AS_OF = "knowledge_after_as_of"
    AS_OF_AFTER_CREATED_AT = "as_of_after_created_at"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    INVALID_SPEC = "invalid_spec"
    ALREADY_EXISTS = "already_exists"


class PublicationRefused(Exception):
    """A publication was refused; `reason` is machine-readable."""

    def __init__(self, reason: RefusalReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def refuse(reason: RefusalReason, detail: str) -> PublicationRefused:
    return PublicationRefused(reason, detail)


# --- strict JSON ---------------------------------------------------------


def _reject_constant(token: str) -> Any:
    raise refuse(RefusalReason.INVALID_JSON, f"non-finite number token {token!r}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise refuse(RefusalReason.INVALID_JSON, f"non-finite number {text!r}")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise refuse(RefusalReason.INVALID_JSON, f"duplicate key {key!r}")
        result[key] = value
    return result


def loads_strict(text: str) -> Any:
    """Parse JSON, refusing NaN/Infinity tokens and duplicate keys."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except PublicationRefused:
        raise
    except (TypeError, ValueError) as error:
        raise refuse(RefusalReason.INVALID_JSON, str(error)) from error


def dumps_canonical(value: Any) -> str:
    """Compact, key-sorted JSON; non-finite numbers are refused."""

    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
    except (TypeError, ValueError) as error:
        raise refuse(RefusalReason.INVALID_JSON, str(error)) from error


def dumps_file(value: Any) -> str:
    """The on-disk form: indented, key-sorted JSON with a trailing newline."""

    try:
        return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise refuse(RefusalReason.INVALID_JSON, str(error)) from error


# --- versions and timestamps ----------------------------------------------


def check_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.MALFORMED_VERSION, f"schema version {value!r}")
    if value not in SUPPORTED_VERSIONS:
        raise refuse(RefusalReason.UNSUPPORTED_VERSION, f"schema version {value!r}")
    return value


def _require_utc(label: str, value: object) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or value is pd.NaT:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} must be a pandas Timestamp")
    if value.tzinfo is None or str(value.tz) != "UTC":
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} must be timezone-aware UTC")
    if value.nanosecond:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} has sub-microsecond precision")
    return value


def format_timestamp(label: str, value: object) -> str:
    """The one timestamp format: YYYY-MM-DDTHH:MM:SS.ffffffZ, UTC only."""

    timestamp = _require_utc(label, value)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timestamp(label: str, value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not _TIMESTAMP_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} {value!r} is not in the format")
    try:
        return pd.Timestamp(value).as_unit("us")
    except ValueError as error:
        raise refuse(RefusalReason.INVALID_TIMESTAMP, f"{label} {value!r}") from error


# --- lockbox and knowledge time -------------------------------------------


def check_lockbox(label: str, value: pd.Timestamp) -> None:
    """The one lockbox check: no information dated on or after the holdout start."""

    if value >= MODEL_HOLDOUT_START:
        raise refuse(
            RefusalReason.LOCKBOX_VIOLATION,
            f"{label} {value.isoformat()} is on or after the holdout start",
        )


def synthetic_session_end(session_date: pd.Timestamp) -> pd.Timestamp:
    """
    Interim knowledge-time rule for synthetic dates only (owner decision
    2026-10-13): a synthetic session date d, stored at 00:00 UTC, is known at
    d 23:59:59.999999Z. Not a production market-data timestamp rule.
    """

    return (session_date + timedelta(days=1) - timedelta(microseconds=1)).as_unit("us")


# --- envelope -------------------------------------------------------------

ENVELOPE_FIELDS = (
    "publication_id",
    "kind",
    "schema_version",
    "as_of",
    "knowledge_time",
    "knowledge_time_rule",
    "created_at",
    "code_sha",
    "code_dirty",
    "input_hashes",
    "universe_id",
    "method_id",
    "method_version",
    "parameters",
    "evidence_status",
    "limitations",
)


@dataclass(frozen=True)
class CodeState:
    sha: str
    dirty: bool


@dataclass(frozen=True, eq=False)
class Draft:
    """Everything a producer supplies; the writer derives the rest."""

    kind: Kind
    evidence_status: EvidenceStatus
    knowledge_time_rule: KnowledgeTimeRule
    universe_id: str
    method_id: str
    method_version: str
    parameters: Mapping[str, Any]
    limitations: tuple[Limitation, ...]
    input_hashes: tuple[Any, ...]
    rows: pd.DataFrame


@dataclass(frozen=True)
class Envelope:
    publication_id: str
    kind: Kind
    schema_version: str
    as_of: pd.Timestamp
    knowledge_time: pd.Timestamp
    knowledge_time_rule: KnowledgeTimeRule
    created_at: pd.Timestamp
    code_sha: str
    code_dirty: bool
    input_hashes: tuple[Any, ...]
    universe_id: str
    method_id: str
    method_version: str
    parameters: Mapping[str, Any] = field(hash=False)
    evidence_status: EvidenceStatus
    limitations: tuple[Limitation, ...]

    def to_json_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "publication_id": self.publication_id,
            "kind": self.kind.value,
            "schema_version": self.schema_version,
            "as_of": format_timestamp("as_of", self.as_of),
            "knowledge_time": format_timestamp("knowledge_time", self.knowledge_time),
            "knowledge_time_rule": self.knowledge_time_rule.value,
            "created_at": format_timestamp("created_at", self.created_at),
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
            "input_hashes": list(self.input_hashes),
            "universe_id": self.universe_id,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "parameters": dict(self.parameters),
            "evidence_status": self.evidence_status.value,
            "limitations": [limitation.value for limitation in self.limitations],
        }
        if not include_id:
            del data["publication_id"]
        return data

    @classmethod
    def from_json_dict(cls, data: object) -> Envelope:
        if not isinstance(data, dict):
            raise refuse(RefusalReason.INVALID_FIELD, "the envelope is not a JSON object")
        if set(data) != set(ENVELOPE_FIELDS):
            missing = sorted(set(ENVELOPE_FIELDS) - set(data))
            extra = sorted(set(data) - set(ENVELOPE_FIELDS))
            raise refuse(RefusalReason.FIELD_SET, f"missing {missing}, unexpected {extra}")
        for name in ("publication_id", "universe_id", "method_id", "method_version"):
            _require_text(name, data[name])
        if not isinstance(data["code_dirty"], bool):
            raise refuse(RefusalReason.INVALID_FIELD, "code_dirty must be a JSON boolean")
        if not isinstance(data["input_hashes"], list):
            raise refuse(RefusalReason.INVALID_FIELD, "input_hashes must be a list")
        parameters = data["parameters"]
        if not isinstance(parameters, dict):
            raise refuse(RefusalReason.INVALID_FIELD, "parameters must be a JSON object")
        return cls(
            publication_id=data["publication_id"],
            kind=_enum(Kind, data["kind"], RefusalReason.UNSUPPORTED_KIND),
            schema_version=check_version(data["schema_version"]),
            as_of=parse_timestamp("as_of", data["as_of"]),
            knowledge_time=parse_timestamp("knowledge_time", data["knowledge_time"]),
            knowledge_time_rule=_enum(
                KnowledgeTimeRule,
                data["knowledge_time_rule"],
                RefusalReason.KNOWLEDGE_RULE_NOT_PERMITTED,
            ),
            created_at=parse_timestamp("created_at", data["created_at"]),
            code_sha=check_code_sha(data["code_sha"]),
            code_dirty=data["code_dirty"],
            input_hashes=tuple(data["input_hashes"]),
            universe_id=data["universe_id"],
            method_id=data["method_id"],
            method_version=data["method_version"],
            parameters=MappingProxyType(dict(parameters)),
            evidence_status=_enum(
                EvidenceStatus, data["evidence_status"], RefusalReason.UNKNOWN_STATUS
            ),
            limitations=parse_limitations(data["limitations"]),
        )


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise refuse(RefusalReason.INVALID_FIELD, f"{name} must be a non-empty string")
    return value


def _enum(enum_type: type[StrEnum], value: object, reason: RefusalReason) -> Any:
    if not isinstance(value, str):
        raise refuse(reason, f"{enum_type.__name__} value {value!r}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise refuse(reason, f"{enum_type.__name__} value {value!r}") from error


def check_code_sha(value: object) -> str:
    if not isinstance(value, str) or not _CODE_SHA_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_FIELD, f"code_sha {value!r} is not a 40-hex SHA")
    return value


def parse_limitations(value: object) -> tuple[Limitation, ...]:
    if not isinstance(value, list):
        raise refuse(RefusalReason.INVALID_FIELD, "limitations must be a list")
    parsed = tuple(_enum(Limitation, item, RefusalReason.INVALID_FIELD) for item in value)
    if list(parsed) != sorted(set(parsed)):
        raise refuse(RefusalReason.INVALID_FIELD, "limitations must be sorted and unique")
    return parsed


def derive_publication_id(envelope_without_id: Mapping[str, Any], rows_sha256: str) -> str:
    """`<created_at stamp>-<12 hex of the envelope (without id) and rows hash>`."""

    created = parse_timestamp("created_at", envelope_without_id["created_at"])
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    payload = dumps_canonical({"envelope": dict(envelope_without_id), "rows": rows_sha256})
    try:
        encoded = payload.encode("utf-8")
    except UnicodeError as error:
        raise refuse(RefusalReason.INVALID_JSON, "the envelope is not valid Unicode") from error
    return f"{stamp}-{hashlib.sha256(encoded).hexdigest()[:12]}"
