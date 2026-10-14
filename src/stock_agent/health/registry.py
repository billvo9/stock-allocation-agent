"""
Registry health from the ML track's model registry
(`stock_agent.registry`, docs/research/ml_registry.md), through its read API
only (`open_registry`). Every call of that API re-verifies what it reads.

registry.entries            every entry loads and hash-verifies, the activation
                            event chain verifies, and every entry's artifacts still
                            match their SHA-256. A registry the reader refuses fails
                            (`registry_refused`); an entry whose artifacts no longer
                            verify fails (`artifacts_unverified`)
registry.stages             each version's computed stage. The value is the number
                            of versions above the floor stage, so a stage that drops
                            (an artifact altered or deleted) raises a deterioration
                            alert. The upper stages are fail-closed pending owner
                            rules, so the floor is the expected stage today
registry.selection_history  activation and rollback events (which version is each
                            scope's reference). The active version of every scope
                            must still verify (`active_unverified` fails)
registry.experiments        multiple-testing counts. Fails when an entry records no
                            candidate count; degraded when an entry's
                            `prior_variants_in_scope` is below the number of
                            versions registered earlier in its scope, which would
                            understate how many variants were tried

Forecast coverage and calibration come from the registry's export
(`health.forecast`), not from here.

In the private scope a registry whose versions are all synthetic is at most
insufficient_evidence (`synthetic_input`); a failure still shows. Health
never writes to the registry and changes no stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from stock_agent.health.model import Area, Better, Check, HealthState
from stock_agent.registry import STAGES, RegistryError, open_registry

FLOOR = STAGES[0]
CHECKS = {
    "registry.entries": (Area.REGISTRY, "Registry entries and artifacts"),
    "registry.stages": (Area.REGISTRY, "Computed stages"),
    "registry.selection_history": (Area.REGISTRY, "Reference-version history"),
    "registry.experiments": (Area.REGISTRY, "Experiment counts"),
}


def _make(check_id: str, **fields) -> Check:
    area, title = CHECKS[check_id]
    return Check(check_id=check_id, area=area, title=title, **fields)


def _not_read(state: HealthState, reason: str, detail: str, evidence: str | None) -> list[Check]:
    summary = {
        HealthState.INSUFFICIENT_EVIDENCE: "The model registry has not been written yet.",
        HealthState.UNAVAILABLE: "The model registry could not be read, so this is not judged.",
    }[state]
    return [
        _make(
            check_id,
            state=state,
            reason=reason,
            measured_at=None,
            value=None,
            unit=None,
            threshold=None,
            summary=summary,
            detail=detail,
            evidence=evidence,
        )
        for check_id in CHECKS
    ]


def _entries_check(entries, unverified: list[str], **common) -> Check:
    if not entries:
        return _make(
            "registry.entries",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_entries",
            value=0,
            unit="registered versions",
            threshold="every entry and its artifacts verify",
            summary="The registry holds no model version yet.",
            detail="no entries",
            **common,
        )
    if unverified:
        state, reason = HealthState.FAILED, "artifacts_unverified"
        summary = f"{len(unverified)} registered versions have artifacts that no longer verify."
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"All {len(entries)} registered versions and their artifacts verify."
    scopes = Counter(entry.research_scope for entry in entries)
    detail = "; ".join(f"{scope}: {count} versions" for scope, count in sorted(scopes.items()))
    if unverified:
        detail += f"; artifacts not verified: {', '.join(unverified[:5])}"
    return _make(
        "registry.entries",
        state=state,
        reason=reason,
        value=len(entries),
        unit="registered versions",
        threshold="every entry and its artifacts verify",
        summary=summary,
        detail=detail,
        **common,
    )


def _stages_check(assessments, **common) -> Check:
    if not assessments:
        return _make(
            "registry.stages",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_entries",
            value=None,
            unit=f"versions above {FLOOR}",
            threshold="computed from verified evidence",
            summary="No version to assess yet.",
            detail="no entries",
            **common,
        )
    stages = Counter(assessment.stage for assessment in assessments)
    blockers = Counter(
        blocker.reason for assessment in assessments for blocker in assessment.blockers
    )
    above = sum(count for stage, count in stages.items() if stage != FLOOR)
    detail = (
        "by stage: "
        + ", ".join(f"{stage} {stages[stage]}" for stage in STAGES if stages[stage])
        + "; blockers to the next stage: "
        + (", ".join(f"{reason} {n}" for reason, n in blockers.most_common(5)) or "none")
    )
    return _make(
        "registry.stages",
        state=HealthState.HEALTHY,
        value=above,
        unit=f"versions above {FLOOR}",
        threshold="computed from verified evidence; a drop is a deterioration",
        better=Better.HIGHER,
        tolerance=0,
        summary=f"Stages were computed for all {len(assessments)} versions; {above} are above "
        f"{FLOOR} (upper stages stay closed until the owner approves their rules).",
        detail=detail,
        **common,
    )


def _history_check(reader, events, verified: dict[str, bool], **common) -> Check:
    if not events:
        return _make(
            "registry.selection_history",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_selection_events",
            value=0,
            unit="activation events",
            threshold="every scope's active version verifies",
            summary="No reference version has been activated yet.",
            detail="no activation or rollback events",
            **common,
        )
    scopes = sorted({event.research_scope for event in events})
    actives = {scope: reader.active(scope) for scope in scopes}
    broken = [
        f"{scope} -> {ref}"
        for scope, ref in actives.items()
        if ref is not None and not verified.get(str(ref), False)
    ]
    rollbacks = sum(event.event == "rollback" for event in events)
    detail = (
        f"{len(events)} events ({rollbacks} rollbacks) across {len(scopes)} scopes; active: "
        + ", ".join(f"{scope} -> {ref}" for scope, ref in actives.items())
    )
    if broken:
        state, reason = HealthState.FAILED, "active_unverified"
        summary = f"{len(broken)} active reference versions no longer verify."
        detail += f"; not verified: {', '.join(broken)}"
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"{len(events)} recorded activations; every active version verifies."
    return _make(
        "registry.selection_history",
        state=state,
        reason=reason,
        value=len(events),
        unit="activation events",
        threshold="every scope's active version verifies",
        summary=summary,
        detail=detail,
        **common,
    )


def _experiments_check(entries, **common) -> Check:
    if not entries:
        return _make(
            "registry.experiments",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_entries",
            value=None,
            unit="registered variants",
            threshold="every entry records its multiple-testing counts",
            summary="No experiment has been registered yet.",
            detail="no entries",
            **common,
        )
    missing, below = [], []
    ordered = sorted(entries, key=lambda entry: (entry.created_at, str(entry.ref)))
    seen: Counter[str] = Counter()
    for entry in ordered:
        testing = entry.evidence.get("multiple_testing") or {}
        if (
            not isinstance(testing.get("candidates_in_run"), int)
            or testing["candidates_in_run"] < 1
        ):
            missing.append(str(entry.ref))
        prior = testing.get("prior_variants_in_scope")
        if isinstance(prior, int) and prior < seen[entry.research_scope]:
            below.append(f"{entry.ref} records {prior}, {seen[entry.research_scope]} registered")
        seen[entry.research_scope] += 1
    detail = (
        "registered variants per scope: "
        + ", ".join(f"{scope} {count}" for scope, count in sorted(seen.items()))
        + ". Every variant tried counts toward multiple-testing corrections."
    )
    if missing:
        state, reason = HealthState.FAILED, "experiment_count_missing"
        summary = f"{len(missing)} versions record no multiple-testing count."
        detail += f" No count: {', '.join(missing[:5])}."
    elif below:
        state, reason = HealthState.DEGRADED, "experiment_count_below_registry"
        summary = (
            f"{len(below)} versions record fewer prior variants than the registry holds, "
            "which would understate multiple testing."
        )
        detail += f" Below: {'; '.join(below[:5])}."
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"{len(entries)} registered variants, each with its multiple-testing counts."
    return _make(
        "registry.experiments",
        state=state,
        reason=reason,
        value=len(entries),
        unit="registered variants",
        threshold="every entry records its counts, and prior variants cover earlier versions",
        summary=summary,
        detail=detail,
        **common,
    )


def _cap_synthetic(checks: list[Check], entries) -> list[Check]:
    """In the private scope, a registry of synthetic versions only is never healthy."""

    if not entries or any(entry.data_class != "synthetic" for entry in entries):
        return checks
    return [
        replace(
            check,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="synthetic_input",
            summary="Every registered version is synthetic, so the registry says nothing "
            "about real models.",
        )
        if check.state is HealthState.HEALTHY
        else check
        for check in checks
    ]


def registry_checks(
    root: Path | None,
    *,
    artifacts_root: Path | None = None,
    now: Callable[[], datetime],
    allow_synthetic: bool = False,
) -> list[Check]:
    evidence = None if root is None else str(root)
    if root is None:
        return _not_read(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            "No --registry was given to this run.",
            evidence,
        )
    if not root.is_dir():
        return _not_read(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_missing",
            f"{root} does not exist yet; the experiment runner has not registered anything.",
            evidence,
        )
    measured_at = now()
    try:
        reader = open_registry(root, artifacts_root)
        entries = reader.entries()
        events = reader.events()
        verified = {str(entry.ref): reader.artifacts_verified(entry) for entry in entries}
        assessments = [reader.assessment(entry.ref) for entry in entries]
    except RegistryError as error:
        refused = _not_read(
            HealthState.UNAVAILABLE, "registry_refused", f"{root}: {error}", evidence
        )
        refused[0] = _make(
            "registry.entries",
            state=HealthState.FAILED,
            reason="registry_refused",
            measured_at=measured_at,
            value=None,
            unit=None,
            threshold="every entry and its artifacts verify",
            summary="The registry reader refused the registry (an edited or broken record).",
            detail=f"{root}: {error}",
            evidence=evidence,
        )
        return refused
    except OSError as error:
        return _not_read(
            HealthState.UNAVAILABLE, "registry_unreadable", f"{root}: {error}", evidence
        )

    common = {"measured_at": measured_at, "evidence": evidence}
    unverified = [ref for ref, ok in verified.items() if not ok]
    checks = [
        _entries_check(entries, unverified, **common),
        _stages_check(assessments, **common),
        _history_check(reader, events, verified, **common),
        _experiments_check(entries, **common),
    ]
    return checks if allow_synthetic else _cap_synthetic(checks, entries)
