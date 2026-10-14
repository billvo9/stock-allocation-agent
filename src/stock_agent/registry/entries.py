"""
Registry entries: immutable, content-hashed records of one model version.

An entry records what was fitted and how it was evaluated: the model id
and version, its research scope, the parent version it descends from, the
configuration (and its hash), fingerprints of the training data and fold
geometry, summary metrics, the artifacts it was evaluated from (each with a
SHA-256), the evidence derived from those artifacts, and the code state.

There is deliberately no stage field. A stage is computed from the recorded
evidence by registry.stages, and the evidence itself is re-derived from the
verified run artifact when an entry is assessed (registry.evidence), so
editing an entry cannot promote a model.

Every entry carries `entry_sha256`, the SHA-256 of the canonical JSON of all
other fields; loading an entry whose hash does not match raises.

Pure: no file I/O here (see store.py).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from stock_agent.model_diagnostics.record import canonical_json, sha256_json

REGISTRY_SCHEMA_VERSION = "1.0"

# Path-safe identifiers: entries live at entries/<model_id>/v<version>.json.
MODEL_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,99}")
SCOPE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:/=-]{0,199}")
ARTIFACT_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,49}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# What the evaluated data can support. Only "research_panel" data (a
# registered point-in-time universe) can carry development evidence;
# synthetic and sandbox runs verify the pipeline only.
DATA_CLASSES = ("synthetic", "engineering_sandbox", "research_panel")


class RegistryError(ValueError):
    """A registry entry, event or store violates the registry contract."""


def _jsonable(value: object, name: str) -> object:
    """A plain-JSON copy of `value` (canonical round trip), or RegistryError."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise RegistryError(f"{name} must be JSON-serializable: {error}") from None


@dataclass(frozen=True, order=True)
class EntryRef:
    """One model version: `model_id@v<version>`."""

    model_id: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise RegistryError(
                f"model_id {self.model_id!r} must match {MODEL_ID_PATTERN.pattern} "
                "(lower case, digits, '_', '.', '-')."
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise RegistryError(f"version must be an integer; got {self.version!r}.")
        if self.version < 1:
            raise RegistryError(f"version must be >= 1; got {self.version}.")

    def __str__(self) -> str:
        return f"{self.model_id}@v{self.version}"

    @classmethod
    def parse(cls, text: str) -> EntryRef:
        model_id, separator, version = str(text).rpartition("@v")
        if not separator or not version.isdigit():
            raise RegistryError(f"{text!r} is not a reference of the form model_id@v<version>.")
        return cls(model_id, int(version))


@dataclass(frozen=True)
class ArtifactRef:
    """
    A file or directory an entry was evaluated from.

    `path` is POSIX and relative to the artifacts root the registry is
    opened with; `sha256` is the SHA-256 of the artifact's manifest file
    (for a directory) or of the file itself.
    """

    kind: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not ARTIFACT_KIND_PATTERN.fullmatch(self.kind):
            raise RegistryError(f"Artifact kind {self.kind!r} is not a valid kind.")
        path = str(self.path)
        parts = path.split("/")
        if (
            not path
            or path.startswith("/")
            or "\\" in path
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise RegistryError(
                f"Artifact path {path!r} must be a relative POSIX path without '.' or '..'."
            )
        if not isinstance(self.sha256, str) or not SHA256_PATTERN.fullmatch(self.sha256):
            raise RegistryError(f"Artifact sha256 {self.sha256!r} is not a SHA-256 hex digest.")

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}


def _utc_iso(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RegistryError("Timestamps must be timezone-aware.")
    return stamp.tz_convert(UTC).isoformat()


@dataclass(frozen=True)
class RegistryEntry:
    """
    One immutable model version. Mapping fields are plain-JSON copies.

    Build entries with `build_entry` (which computes the hashes) or load
    them with `entry_from_json` (which verifies them).
    """

    ref: EntryRef
    research_scope: str
    parent: EntryRef | None
    created_at: str
    config: Mapping[str, object]
    config_sha256: str
    training_data: Mapping[str, object]
    metrics: Mapping[str, float | None]
    artifacts: tuple[ArtifactRef, ...]
    evidence: Mapping[str, object]
    code: Mapping[str, object]
    entry_sha256: str

    @property
    def model_id(self) -> str:
        return self.ref.model_id

    @property
    def version(self) -> int:
        return self.ref.version

    @property
    def data_class(self) -> str | None:
        value = self.training_data.get("data_class")
        return value if isinstance(value, str) else None

    def to_json(self) -> dict[str, object]:
        return {**_content(self), "entry_sha256": self.entry_sha256}


def _content(entry: RegistryEntry) -> dict[str, object]:
    """Every hashed field of an entry (everything except entry_sha256)."""

    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "model_id": entry.ref.model_id,
        "version": entry.ref.version,
        "research_scope": entry.research_scope,
        "parent": None if entry.parent is None else str(entry.parent),
        "created_at": entry.created_at,
        "config": entry.config,
        "config_sha256": entry.config_sha256,
        "training_data": entry.training_data,
        "metrics": entry.metrics,
        "artifacts": [artifact.to_json() for artifact in entry.artifacts],
        "evidence": entry.evidence,
        "code": entry.code,
    }


def _metrics(metrics: Mapping[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise RegistryError("Metric names must be non-empty strings.")
        if value is None:
            result[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                number = float(value)  # numpy scalars
            except (TypeError, ValueError):
                raise RegistryError(f"Metric {name!r} must be a number or None.") from None
        else:
            number = float(value)
        result[name] = number if math.isfinite(number) else None
    return dict(sorted(result.items()))


def _validate_scope(scope: str) -> str:
    if not isinstance(scope, str) or not SCOPE_PATTERN.fullmatch(scope):
        raise RegistryError(f"research_scope {scope!r} must match {SCOPE_PATTERN.pattern}.")
    return scope


def build_entry(
    *,
    ref: EntryRef,
    research_scope: str,
    parent: EntryRef | None,
    created_at: datetime,
    config: Mapping[str, object],
    training_data: Mapping[str, object],
    metrics: Mapping[str, object],
    artifacts: Sequence[ArtifactRef],
    evidence: Mapping[str, object],
    code: Mapping[str, object],
) -> RegistryEntry:
    """A validated entry with its config and entry hashes computed."""

    _validate_scope(research_scope)
    if parent is not None and parent.model_id == ref.model_id and parent.version >= ref.version:
        raise RegistryError(f"Parent {parent} must be an earlier version than {ref}.")
    data = _jsonable(dict(training_data), "training_data")
    if data.get("data_class") not in DATA_CLASSES:
        raise RegistryError(f"training_data.data_class must be one of {DATA_CLASSES}.")
    if not artifacts:
        raise RegistryError("An entry needs at least one artifact it was evaluated from.")
    config_json = _jsonable(dict(config), "config")
    draft = RegistryEntry(
        ref=ref,
        research_scope=research_scope,
        parent=parent,
        created_at=_utc_iso(created_at),
        config=config_json,
        config_sha256=sha256_json(config_json),
        training_data=data,
        metrics=_metrics(metrics),
        artifacts=tuple(artifacts),
        evidence=_jsonable(dict(evidence), "evidence"),
        code=_jsonable(dict(code), "code"),
        entry_sha256="",
    )
    return _with_hash(draft)


def _with_hash(entry: RegistryEntry) -> RegistryEntry:
    return RegistryEntry(**{**entry.__dict__, "entry_sha256": sha256_json(_content(entry))})


def entry_from_json(payload: Mapping[str, object]) -> RegistryEntry:
    """Load an entry and verify its schema version and both hashes."""

    expected = set(_content_keys()) | {"entry_sha256"}
    keys = set(payload)
    if keys != expected:
        raise RegistryError(
            f"Entry fields {sorted(keys ^ expected)} are missing or unexpected; "
            "an entry holds exactly the registry contract's fields."
        )
    if payload["registry_schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f"Entry schema {payload['registry_schema_version']!r} is not readable by registry "
            f"contract {REGISTRY_SCHEMA_VERSION}."
        )
    parent = payload["parent"]
    entry = RegistryEntry(
        ref=EntryRef(str(payload["model_id"]), payload["version"]),
        research_scope=_validate_scope(payload["research_scope"]),
        parent=None if parent is None else EntryRef.parse(str(parent)),
        created_at=str(payload["created_at"]),
        config=_jsonable(payload["config"], "config"),
        config_sha256=str(payload["config_sha256"]),
        training_data=_jsonable(payload["training_data"], "training_data"),
        metrics=_metrics(payload["metrics"]),
        artifacts=tuple(ArtifactRef(**item) for item in payload["artifacts"]),
        evidence=_jsonable(payload["evidence"], "evidence"),
        code=_jsonable(payload["code"], "code"),
        entry_sha256=str(payload["entry_sha256"]),
    )
    if sha256_json(entry.config) != entry.config_sha256:
        raise RegistryError(f"{entry.ref}: config does not match config_sha256.")
    if sha256_json(_content(entry)) != entry.entry_sha256:
        raise RegistryError(f"{entry.ref}: content does not match entry_sha256 (edited entry).")
    return entry


def _content_keys() -> tuple[str, ...]:
    return (
        "registry_schema_version",
        "model_id",
        "version",
        "research_scope",
        "parent",
        "created_at",
        "config",
        "config_sha256",
        "training_data",
        "metrics",
        "artifacts",
        "evidence",
        "code",
    )
