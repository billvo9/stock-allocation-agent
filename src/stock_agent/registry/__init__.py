"""
Model registry: versioned, content-hashed model entries with stages computed
from recorded evidence (docs/research/ml_registry.md).

Read API (stable; used by the Model Health Center):

    open_registry(root, artifacts_root=None) -> RegistryReader
        .refs()                       every EntryRef, sorted
        .entries(research_scope=None) RegistryEntry objects (hash-verified)
        .entry(ref)                   one entry ("model_id@v3" or EntryRef)
        .assessment(ref)              StageAssessment: stage, criteria, blockers
        .stage(ref)                   the computed stage name
        .active(research_scope)       the scope's active EntryRef, or None
        .activation_history(scope)    ActivationEvent tuple (append-only)
        .compare(a, b)                Comparison of two versions of one scope
        .summary()                    one row per version (DataFrame)
        .metrics_frame()              long table: ref, metric, value

Write API (the experiment runner): Registry(root, artifacts_root, clock=...)
with register(), activate() and rollback().

Modules:
    entries   entry and reference types, canonical hashing
    evidence  derive_evidence: the only way evidence enters an entry
    stages    stage rules (fail closed above development_evaluated)
    store     write-once files, hash-chained activation events, reader
"""

from stock_agent.registry.entries import (
    DATA_CLASSES,
    REGISTRY_SCHEMA_VERSION,
    ArtifactRef,
    EntryRef,
    RegistryEntry,
    RegistryError,
)
from stock_agent.registry.evidence import derive_evidence
from stock_agent.registry.stages import RULES, STAGES, Criterion, StageAssessment, assess
from stock_agent.registry.store import (
    ActivationEvent,
    Comparison,
    Registry,
    RegistryReader,
    open_registry,
)

__all__ = [
    "DATA_CLASSES",
    "REGISTRY_SCHEMA_VERSION",
    "RULES",
    "STAGES",
    "ActivationEvent",
    "ArtifactRef",
    "Comparison",
    "Criterion",
    "EntryRef",
    "Registry",
    "RegistryEntry",
    "RegistryError",
    "RegistryReader",
    "StageAssessment",
    "assess",
    "derive_evidence",
    "open_registry",
]
