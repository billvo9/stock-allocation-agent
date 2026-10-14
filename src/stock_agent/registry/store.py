"""
The registry store: write-once entry files and an append-only event log.

Layout under a registry root (default reports/ml/registry/, ignored by Git):

    entries/<model_id>/v0001.json   one immutable entry per model version,
                                    created with exclusive create (never
                                    overwritten, never edited in place)
    events.jsonl                    activation and rollback events, one JSON
                                    line each, hash-chained (each line holds
                                    the previous line's SHA-256)

Artifact paths in entries are relative to an artifacts root (default: the
registry root's parent, e.g. reports/ml/), where experiment runs and
forecast artifacts live.

Reading (RegistryReader) never writes. Every load verifies the entry's
hashes and the event chain; a stage is computed on demand from the entry's
evidence, after re-verifying its artifacts and re-deriving its evidence from
the verified run record (registry.stages, registry.evidence).

"Active" is a research bookkeeping pointer: the version currently used as
the reference for one research scope. Activating or rolling back never
changes a stage, and an active version is not a deployed one.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.model_diagnostics.record import canonical_json, sha256_json
from stock_agent.model_validation.audit import file_sha256
from stock_agent.registry.entries import (
    ArtifactRef,
    EntryRef,
    RegistryEntry,
    RegistryError,
    build_entry,
    entry_from_json,
)
from stock_agent.registry.evidence import derive_evidence
from stock_agent.registry.stages import StageAssessment, assess

ENTRIES_DIRECTORY = "entries"
EVENTS_FILE = "events.jsonl"
MANIFEST_FILE = "manifest.json"
RECORD_FILE = "record.json"
EXPERIMENT_RUN_KIND = "experiment_run"
EVENT_KINDS = ("activate", "rollback")
_VERSION_FILE = re.compile(r"v(\d{4,})\.json")
_GENESIS = "0" * 64


@dataclass(frozen=True)
class ActivationEvent:
    """One change of a scope's active version."""

    sequence: int
    event: str
    research_scope: str
    ref: EntryRef
    previous: EntryRef | None
    reason: str
    at: str
    event_sha256: str


@dataclass(frozen=True)
class Comparison:
    """
    Two versions of one research scope, metric by metric (b minus a).

    Descriptive only: a credible difference needs paired predictions on
    identical rows, which the experiment runner reports with uncertainty.
    `same_training_data` and `same_folds` say whether the two were even
    evaluated on the same rows.
    """

    a: EntryRef
    b: EntryRef
    research_scope: str
    same_training_data: bool
    same_folds: bool
    table: pd.DataFrame


def _now() -> datetime:
    return datetime.now(UTC)


def _inside(root: Path, path: Path) -> bool:
    """True when `path` resolves inside `root` and no component is a symlink."""

    current = path
    while current != root and current != current.parent:
        if current.is_symlink():
            return False
        current = current.parent
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_manifest_directory(directory: Path, manifest_sha256: str) -> bool:
    """
    A directory artifact verifies when its manifest.json has the recorded
    SHA-256 and every file the manifest lists exists inside the directory
    and matches its own recorded file_sha256.
    """

    manifest = directory / MANIFEST_FILE
    if not manifest.is_file() or manifest.is_symlink():
        return False
    if file_sha256(manifest) != manifest_sha256:
        return False
    try:
        listed = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    except (ValueError, KeyError, TypeError):
        return False
    if not isinstance(listed, Mapping) or not listed:
        return False
    for name, entry in listed.items():
        path = directory / str(name)
        if (
            "/" in str(name)
            or not path.is_file()
            or not _inside(directory, path)
            or not isinstance(entry, Mapping)
            or file_sha256(path) != entry.get("file_sha256")
        ):
            return False
    return True


class RegistryReader:
    """Read-only access to a registry (the API other packages may use)."""

    def __init__(self, root: str | Path, artifacts_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.artifacts_root = Path(artifacts_root) if artifacts_root else self.root.parent

    # --- entries ---

    def _entry_path(self, ref: EntryRef) -> Path:
        return self.root / ENTRIES_DIRECTORY / ref.model_id / f"v{ref.version:04d}.json"

    def _load(self, path: Path, ref: EntryRef) -> RegistryEntry:
        if path.is_symlink() or not path.is_file():
            raise RegistryError(f"{path} is not a regular entry file.")
        entry = entry_from_json(json.loads(path.read_text(encoding="utf-8")))
        if entry.ref != ref:
            raise RegistryError(f"{path} holds {entry.ref}, not {ref}.")
        return entry

    def refs(self) -> tuple[EntryRef, ...]:
        """Every registered version, sorted by model id and version."""

        directory = self.root / ENTRIES_DIRECTORY
        if not directory.exists():
            return ()
        refs = []
        for model_dir in sorted(directory.iterdir()):
            if model_dir.name.startswith("."):
                continue
            if model_dir.is_symlink() or not model_dir.is_dir():
                raise RegistryError(f"{model_dir} is not a model directory.")
            for path in sorted(model_dir.iterdir()):
                match = _VERSION_FILE.fullmatch(path.name)
                if match is None:
                    raise RegistryError(f"Unexpected file {path} in the registry.")
                refs.append(EntryRef(model_dir.name, int(match.group(1))))
        return tuple(sorted(refs))

    def entry(self, ref: EntryRef | str) -> RegistryEntry:
        ref = EntryRef.parse(ref) if isinstance(ref, str) else ref
        path = self._entry_path(ref)
        if not path.exists():
            raise RegistryError(f"{ref} is not registered.")
        return self._load(path, ref)

    def entries(self, research_scope: str | None = None) -> tuple[RegistryEntry, ...]:
        loaded = (self.entry(ref) for ref in self.refs())
        return tuple(e for e in loaded if research_scope in (None, e.research_scope))

    def versions(self, model_id: str) -> tuple[EntryRef, ...]:
        return tuple(ref for ref in self.refs() if ref.model_id == model_id)

    # --- artifacts and evidence ---

    def _artifact_path(self, artifact: ArtifactRef) -> Path | None:
        path = self.artifacts_root / artifact.path
        return path if path.exists() and _inside(self.artifacts_root, path) else None

    def artifacts_verified(self, entry: RegistryEntry) -> bool:
        """Every artifact exists under the artifacts root and matches its SHA-256."""

        for artifact in entry.artifacts:
            path = self._artifact_path(artifact)
            if path is None:
                return False
            if path.is_dir():
                if not verify_manifest_directory(path, artifact.sha256):
                    return False
            elif file_sha256(path) != artifact.sha256:
                return False
        return True

    def run_record(self, entry: RegistryEntry) -> Mapping[str, object] | None:
        """The verified record.json of the entry's experiment run, if any."""

        runs = [a for a in entry.artifacts if a.kind == EXPERIMENT_RUN_KIND]
        if len(runs) != 1:
            return None
        path = self._artifact_path(runs[0])
        if path is None or not path.is_dir() or not verify_manifest_directory(path, runs[0].sha256):
            return None
        record_path = path / RECORD_FILE
        if not record_path.is_file():
            return None
        return json.loads(record_path.read_text(encoding="utf-8"))

    def rederived_evidence(self, entry: RegistryEntry) -> dict[str, object] | None:
        record = self.run_record(entry)
        model = entry.evidence.get("model")
        if record is None or not isinstance(model, str):
            return None
        try:
            return derive_evidence(record, model)
        except RegistryError:
            return None

    def assessment(self, ref: EntryRef | str) -> StageAssessment:
        """The entry's computed stage and every criterion behind it."""

        entry = self.entry(ref)
        return assess(
            entry,
            artifacts_verified=self.artifacts_verified(entry),
            rederived_evidence=self.rederived_evidence(entry),
        )

    def stage(self, ref: EntryRef | str) -> str:
        return self.assessment(ref).stage

    # --- activation ---

    def events(self) -> tuple[ActivationEvent, ...]:
        """Every activation event, after verifying the hash chain."""

        path = self.root / EVENTS_FILE
        if not path.exists():
            return ()
        if path.is_symlink():
            raise RegistryError(f"{path} must be a regular file.")
        events = []
        previous_sha = _GENESIS
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            payload = json.loads(line)
            claimed = payload.pop("event_sha256", None)
            if (
                payload.get("prev_event_sha256") != previous_sha
                or payload.get("sequence") != number
            ):
                raise RegistryError(f"{path} line {number + 1} breaks the event chain.")
            if sha256_json(payload) != claimed:
                raise RegistryError(f"{path} line {number + 1} was edited (hash mismatch).")
            if payload.get("event") not in EVENT_KINDS:
                raise RegistryError(f"{path} line {number + 1} has an unknown event.")
            previous = payload.get("previous")
            events.append(
                ActivationEvent(
                    sequence=number,
                    event=payload["event"],
                    research_scope=payload["research_scope"],
                    ref=EntryRef.parse(payload["ref"]),
                    previous=None if previous is None else EntryRef.parse(previous),
                    reason=payload["reason"],
                    at=payload["at"],
                    event_sha256=claimed,
                )
            )
            previous_sha = claimed
        return tuple(events)

    def activation_history(self, research_scope: str) -> tuple[ActivationEvent, ...]:
        return tuple(e for e in self.events() if e.research_scope == research_scope)

    def active(self, research_scope: str) -> EntryRef | None:
        """The scope's active version (its latest activation or rollback)."""

        history = self.activation_history(research_scope)
        return history[-1].ref if history else None

    # --- comparison and summaries ---

    def compare(self, a: EntryRef | str, b: EntryRef | str) -> Comparison:
        """Metric differences of two versions of one research scope."""

        first, second = self.entry(a), self.entry(b)
        if first.research_scope != second.research_scope:
            raise RegistryError(
                f"{first.ref} ({first.research_scope}) and {second.ref} "
                f"({second.research_scope}) belong to different research scopes."
            )
        names = sorted(set(first.metrics) | set(second.metrics))
        rows = []
        for name in names:
            left, right = first.metrics.get(name), second.metrics.get(name)
            rows.append(
                {
                    "metric": name,
                    "a": left,
                    "b": right,
                    "b_minus_a": None if left is None or right is None else right - left,
                }
            )
        table = pd.DataFrame(rows, columns=["metric", "a", "b", "b_minus_a"]).astype(
            {"a": "float64", "b": "float64", "b_minus_a": "float64"}
        )
        data_a, data_b = first.training_data, second.training_data
        return Comparison(
            a=first.ref,
            b=second.ref,
            research_scope=first.research_scope,
            same_training_data=data_a.get("labeled_frame_sha256") is not None
            and data_a.get("labeled_frame_sha256") == data_b.get("labeled_frame_sha256"),
            same_folds=data_a.get("fold_table_sha256") is not None
            and data_a.get("fold_table_sha256") == data_b.get("fold_table_sha256"),
            table=table,
        )

    def summary(self) -> pd.DataFrame:
        """One row per version: identity, scope, lineage, data class, stage, active."""

        rows = []
        actives: dict[str, EntryRef | None] = {}
        for entry in self.entries():
            scope = entry.research_scope
            if scope not in actives:
                actives[scope] = self.active(scope)
            assessment = self.assessment(entry.ref)
            rows.append(
                {
                    "ref": str(entry.ref),
                    "model_id": entry.model_id,
                    "version": entry.version,
                    "research_scope": scope,
                    "parent": None if entry.parent is None else str(entry.parent),
                    "created_at": entry.created_at,
                    "data_class": entry.data_class,
                    "run_id": entry.evidence.get("run_id"),
                    "stage": assessment.stage,
                    "next_stage": assessment.next_stage,
                    "n_blockers": len(assessment.blockers),
                    "active": actives[scope] == entry.ref,
                }
            )
        columns = [
            "ref",
            "model_id",
            "version",
            "research_scope",
            "parent",
            "created_at",
            "data_class",
            "run_id",
            "stage",
            "next_stage",
            "n_blockers",
            "active",
        ]
        return pd.DataFrame(rows, columns=columns)

    def metrics_frame(self) -> pd.DataFrame:
        """Long table of every version's recorded metrics (ref, metric, value)."""

        rows = [
            {"ref": str(entry.ref), "metric": name, "value": value}
            for entry in self.entries()
            for name, value in entry.metrics.items()
        ]
        return pd.DataFrame(rows, columns=["ref", "metric", "value"]).astype({"value": "float64"})


class Registry(RegistryReader):
    """Registry with write access: register versions, activate, roll back."""

    def __init__(
        self,
        root: str | Path,
        artifacts_root: str | Path | None = None,
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        super().__init__(root, artifacts_root)
        self.clock = clock

    def next_version(self, model_id: str) -> EntryRef:
        """The reference the next registration of `model_id` will receive."""

        existing = self.versions(model_id)
        return EntryRef(model_id, existing[-1].version + 1 if existing else 1)

    def register(
        self,
        *,
        model_id: str,
        research_scope: str,
        config: Mapping[str, object],
        training_data: Mapping[str, object],
        metrics: Mapping[str, object],
        artifacts: Sequence[ArtifactRef],
        evidence: Mapping[str, object],
        code: Mapping[str, object],
        parent: EntryRef | str | None = None,
        expected_version: int | None = None,
    ) -> RegistryEntry:
        """
        Write the next version of `model_id` (write-once).

        Refused unless every artifact verifies and, when the entry has an
        experiment run, its evidence equals the evidence re-derived from that
        run's verified record. The parent must be registered.
        `expected_version` (from next_version, when an artifact had to name
        the version before registration) must still be the next version.
        """

        parent_ref = EntryRef.parse(parent) if isinstance(parent, str) else parent
        if parent_ref is not None:
            self.entry(parent_ref)  # raises if unregistered
        ref = self.next_version(model_id)
        if expected_version is not None and ref.version != expected_version:
            raise RegistryError(
                f"{model_id}: expected to register v{expected_version}, but the next version "
                f"is v{ref.version} (another registration happened in between)."
            )
        entry = build_entry(
            ref=ref,
            research_scope=research_scope,
            parent=parent_ref,
            created_at=self.clock(),
            config=config,
            training_data=training_data,
            metrics=metrics,
            artifacts=artifacts,
            evidence=evidence,
            code=code,
        )
        if not self.artifacts_verified(entry):
            raise RegistryError(f"{ref}: an artifact is missing or does not match its SHA-256.")
        has_run = any(a.kind == EXPERIMENT_RUN_KIND for a in entry.artifacts)
        if has_run and self.rederived_evidence(entry) != dict(entry.evidence):
            raise RegistryError(
                f"{ref}: the evidence does not follow from the experiment run's record."
            )
        path = self._entry_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(json.loads(canonical_json(entry.to_json())), indent=2, sort_keys=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        except FileExistsError:
            raise RegistryError(f"{ref} already exists; registration raced, retry.") from None
        return entry

    def _append_event(self, event: str, entry: RegistryEntry, reason: str) -> ActivationEvent:
        if not isinstance(reason, str) or not reason.strip():
            raise RegistryError("An activation needs a stated reason.")
        events = self.events()
        previous = self.active(entry.research_scope)
        payload = {
            "sequence": len(events),
            "event": event,
            "research_scope": entry.research_scope,
            "ref": str(entry.ref),
            "previous": None if previous is None else str(previous),
            "reason": reason,
            "at": self.clock().astimezone(UTC).isoformat(),
            "prev_event_sha256": events[-1].event_sha256 if events else _GENESIS,
        }
        digest = sha256_json(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(canonical_json({**payload, "event_sha256": digest}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.events()[-1]

    def activate(self, ref: EntryRef | str, *, reason: str) -> ActivationEvent:
        """Make a registered version the active one of its research scope."""

        entry = self.entry(ref)
        if self.active(entry.research_scope) == entry.ref:
            raise RegistryError(f"{entry.ref} is already active in {entry.research_scope}.")
        return self._append_event("activate", entry, reason)

    def rollback(self, research_scope: str, to: EntryRef | str, *, reason: str) -> ActivationEvent:
        """Re-activate a version that was active earlier in this scope."""

        entry = self.entry(to)
        if entry.research_scope != research_scope:
            raise RegistryError(f"{entry.ref} does not belong to {research_scope}.")
        history = self.activation_history(research_scope)
        if not history:
            raise RegistryError(f"{research_scope} has no active version to roll back from.")
        if history[-1].ref == entry.ref:
            raise RegistryError(f"{entry.ref} is already active in {research_scope}.")
        if entry.ref not in {event.ref for event in history}:
            raise RegistryError(
                f"{entry.ref} was never active in {research_scope}; activate it instead."
            )
        return self._append_event("rollback", entry, reason)


def open_registry(root: str | Path, artifacts_root: str | Path | None = None) -> RegistryReader:
    """Read-only registry access (entries, stages, active versions, comparisons)."""

    return RegistryReader(root, artifacts_root)
