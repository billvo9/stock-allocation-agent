"""Model registry: write-once entries, computed stages, activation and rollback."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.model_diagnostics.record import canonical_json, sha256_json
from stock_agent.model_validation.audit import file_sha256
from stock_agent.registry import (
    RULES,
    STAGES,
    ArtifactRef,
    EntryRef,
    Registry,
    RegistryError,
    derive_evidence,
    open_registry,
    stages,
)

HOLDOUT = "2025-01-01T00:00:00+00:00"


def passing_record(run_id: str = "20261014T000000Z-aaaaaaaaaaaa") -> dict:
    """A run record whose evidence meets every development_evaluated criterion."""

    return {
        "run_id": run_id,
        "data": {"data_class": "research_panel"},
        "lockbox_evidence": {
            "mode": "development",
            "holdout_start": HOLDOUT,
            "max_date": "2024-12-31T00:00:00+00:00",
            "max_target_end_date": "2024-12-30T00:00:00+00:00",
            "holdout_values_used": False,
        },
        "checks_summary": {
            "families": {
                "leakage": {"total": 5, "passed": 5, "failed": 0, "undetermined": 0},
                "instrument": {"total": 4, "passed": 4, "failed": 0, "undetermined": 0},
            }
        },
        "decision_method": "fold_block_t",
        "nulls": {"models": ["zero", "random_noise"], "permutation_draws": 199},
        "multiple_testing": {
            "candidates_in_run": 2,
            "grid_points_in_run": 12,
            "prior_variants_in_scope": 0,
        },
        "universe": {"point_in_time": True, "survivorship_free": True},
        "models": {
            "ridge": {"role": "candidate", "variant_id": "v-ridge"},
            "zero": {"role": "null", "variant_id": "v-zero"},
        },
    }


def synthetic_record(run_id: str = "20261014T000000Z-bbbbbbbbbbbb") -> dict:
    record = passing_record(run_id)
    record["data"]["data_class"] = "synthetic"
    record["universe"] = {"point_in_time": True, "survivorship_free": False}
    return record


def write_run(artifacts_root: Path, record: dict) -> ArtifactRef:
    """A minimal experiment-run directory: record.json plus a manifest."""

    directory = artifacts_root / "runs" / record["run_id"]
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(canonical_json(record), encoding="utf-8")
    manifest = {"files": {"record.json": {"file_sha256": file_sha256(directory / "record.json")}}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ArtifactRef(
        "experiment_run", f"runs/{record['run_id']}", file_sha256(directory / "manifest.json")
    )


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry", tmp_path, clock=Clock())


def register(registry: Registry, record: dict, *, model: str = "ridge", **changes):
    artifact = write_run(registry.artifacts_root, record)
    arguments = {
        "model_id": "equity_xs.ridge",
        "research_scope": "equity_xs:synthetic:h20",
        "config": {"model": model, "grid": [1, 2]},
        "training_data": {
            "data_class": record["data"]["data_class"],
            "labeled_frame_sha256": "a" * 64,
            "fold_table_sha256": "b" * 64,
        },
        "metrics": {"mean_rank_ic": 0.05, "oos_r2_vs_zero": -0.001},
        "artifacts": [artifact],
        "evidence": derive_evidence(record, model),
        "code": {"git_sha": "c" * 40, "git_dirty": False},
        **changes,
    }
    return registry.register(**arguments)


def set_path(record: dict, path: str, value: object) -> dict:
    result = copy.deepcopy(record)
    target = result
    *parents, last = path.split(".")
    for key in parents:
        target = target[key]
    target[last] = value
    return result


# --- entries ---


def test_versions_are_sequential_and_entries_are_write_once(registry):
    first = register(registry, synthetic_record("r1"))
    second = register(registry, synthetic_record("r2"), parent=first.ref)
    assert (first.ref, second.ref) == (
        EntryRef("equity_xs.ridge", 1),
        EntryRef("equity_xs.ridge", 2),
    )
    assert registry.entry("equity_xs.ridge@v2").parent == first.ref
    path = registry.root / "entries" / "equity_xs.ridge" / "v0002.json"
    with pytest.raises(FileExistsError):
        path.open("x")


def test_an_edited_entry_fails_to_load(registry):
    entry = register(registry, synthetic_record())
    path = registry.root / "entries" / "equity_xs.ridge" / "v0001.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["mean_rank_ic"] = 0.5
    path.write_text(json.dumps(payload))
    with pytest.raises(RegistryError, match="edited entry"):
        registry.entry(entry.ref)


def test_an_entry_has_no_stage_field_and_refuses_one(registry):
    entry = register(registry, synthetic_record())
    path = registry.root / "entries" / "equity_xs.ridge" / "v0001.json"
    payload = json.loads(path.read_text())
    assert "stage" not in payload
    payload["stage"] = "approved_for_scope"
    path.write_text(json.dumps(payload))
    with pytest.raises(RegistryError, match="unexpected"):
        registry.entry(entry.ref)


def test_registration_refuses_evidence_that_does_not_follow_from_the_run(registry):
    record = synthetic_record()
    claimed = derive_evidence(passing_record(record["run_id"]), "ridge")
    with pytest.raises(RegistryError, match="does not follow"):
        register(registry, record, evidence=claimed)


def test_registration_refuses_unverifiable_artifacts_and_unknown_parents(registry, tmp_path):
    record = synthetic_record()
    artifact = write_run(registry.artifacts_root, record)
    (tmp_path / "runs" / record["run_id"] / "record.json").write_text("{}")
    with pytest.raises(RegistryError, match="artifact"):
        register(registry, synthetic_record("other"), artifacts=[artifact])
    with pytest.raises(RegistryError, match="not registered"):
        register(registry, synthetic_record("r3"), parent="equity_xs.ridge@v7")


def test_artifact_paths_must_stay_inside_the_artifacts_root():
    for path in ("../outside", "/abs/path", "runs/../x", "runs\\x", ""):
        with pytest.raises(RegistryError):
            ArtifactRef("experiment_run", path, "0" * 64)


def test_a_symlinked_artifact_does_not_verify(registry, tmp_path):
    entry = register(registry, synthetic_record())
    real = tmp_path / "runs" / entry.evidence["run_id"]
    moved = tmp_path / "elsewhere"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    assert registry.artifacts_verified(entry) is False


def test_expected_version_must_still_be_next(registry):
    register(registry, synthetic_record("r1"))
    with pytest.raises(RegistryError, match="expected to register v1"):
        register(registry, synthetic_record("r2"), expected_version=1)
    assert register(registry, synthetic_record("r3"), expected_version=2).version == 2


# --- stages ---


def test_synthetic_runs_stay_engineering_only_and_say_why(registry):
    entry = register(registry, synthetic_record())
    assessment = registry.assessment(entry.ref)
    assert assessment.stage == "engineering_only"
    assert assessment.next_stage == "development_evaluated"
    reasons = {c.reason for c in assessment.blockers}
    assert "data_class_engineering_only" in reasons
    assert "universe_not_point_in_time" in reasons


def test_a_complete_research_run_reaches_development_evaluated_and_no_further(registry):
    entry = register(registry, passing_record())
    assessment = registry.assessment(entry.ref)
    assert assessment.stage == "development_evaluated"
    assert {c.criterion for c in assessment.blockers} == {"owner_approved_evidence_rule"}
    upper = [c for c in assessment.criteria if c.stage in STAGES[2:]]
    assert len(upper) == 3 and all(c.reason == "rule_not_approved" for c in upper)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("data.data_class", "engineering_sandbox", "data_class_engineering_only"),
        ("universe.survivorship_free", False, "universe_not_point_in_time"),
        ("lockbox_evidence.holdout_values_used", True, "lockbox_not_proven_closed"),
        ("lockbox_evidence.max_target_end_date", HOLDOUT, "lockbox_not_proven_closed"),
        ("lockbox_evidence.mode", "holdout", "lockbox_not_proven_closed"),
        ("checks_summary.families.leakage.failed", 1, "leakage_checks"),
        ("checks_summary.families.leakage.undetermined", 1, "leakage_checks"),
        ("checks_summary.families.instrument.failed", 1, "instrument_checks"),
        ("decision_method", "newey_west", "decision_method"),
        ("nulls.permutation_draws", 49, "nulls_insufficient"),
        ("nulls.models", [], "nulls_insufficient"),
        ("multiple_testing.candidates_in_run", 0, "multiple_testing_not_recorded"),
        ("models.ridge.role", "control", "not_candidate"),
    ],
)
def test_each_development_criterion_blocks_on_its_own(registry, path, value, reason):
    entry = register(registry, set_path(passing_record(), path, value))
    assessment = registry.assessment(entry.ref)
    assert assessment.stage == "engineering_only"
    assert reason in {c.reason for c in assessment.blockers}


def test_tampering_with_the_run_record_after_registration_demotes_the_entry(registry, tmp_path):
    entry = register(registry, passing_record())
    record_path = tmp_path / "runs" / entry.evidence["run_id"] / "record.json"
    record_path.write_text(canonical_json(passing_record()) + " ")
    assessment = registry.assessment(entry.ref)
    assert assessment.stage == "engineering_only"
    assert {"artifacts_unverified", "evidence_not_reproducible"} <= {
        c.reason for c in assessment.blockers
    }


def test_missing_artifacts_demote_the_entry(registry, tmp_path):
    entry = register(registry, passing_record())
    run = tmp_path / "runs" / entry.evidence["run_id"]
    run.rename(tmp_path / "moved")
    assert registry.stage(entry.ref) == "engineering_only"


def test_upper_stages_fail_closed_even_if_their_rule_is_marked_approved(registry, monkeypatch):
    for stage in STAGES[2:]:
        assert RULES[stage].approved is False
    entry = register(registry, passing_record())
    approved = {**RULES, "historically_checked": stages.Rule(True, "test")}
    monkeypatch.setattr(stages, "RULES", approved)
    assessment = registry.assessment(entry.ref)
    assert assessment.stage == "development_evaluated"
    assert [c.reason for c in assessment.blockers] == ["criteria_not_implemented"]


def test_evidence_claims_beyond_the_rules_are_ignored(registry):
    record = passing_record()
    record["holdout_checked"] = True
    record["prospective_confirmation"] = {"passed": True}
    entry = register(registry, record)
    assert registry.stage(entry.ref) == "development_evaluated"


# --- activation and rollback ---


def test_activation_and_rollback_keep_an_append_only_history(registry):
    first = register(registry, synthetic_record("r1"))
    second = register(registry, synthetic_record("r2"), parent=first.ref)
    scope = first.research_scope
    assert registry.active(scope) is None
    registry.activate(first.ref, reason="baseline")
    registry.activate(second.ref, reason="new version")
    assert registry.active(scope) == second.ref
    event = registry.rollback(scope, first.ref, reason="regression in v2")
    assert event.event == "rollback" and event.previous == second.ref
    assert registry.active(scope) == first.ref
    assert [e.ref for e in registry.activation_history(scope)] == [
        first.ref,
        second.ref,
        first.ref,
    ]
    assert registry.stage(first.ref) == registry.stage(second.ref) == "engineering_only"


def test_rollback_needs_a_previously_active_version_of_the_same_scope(registry):
    first = register(registry, synthetic_record("r1"))
    second = register(registry, synthetic_record("r2"))
    other = register(
        registry, synthetic_record("r3"), model_id="etf.vol", research_scope="etf:synthetic:h20"
    )
    scope = first.research_scope
    with pytest.raises(RegistryError, match="no active version"):
        registry.rollback(scope, first.ref, reason="x")
    registry.activate(first.ref, reason="baseline")
    with pytest.raises(RegistryError, match="never active"):
        registry.rollback(scope, second.ref, reason="x")
    with pytest.raises(RegistryError, match="does not belong"):
        registry.rollback(scope, other.ref, reason="x")
    with pytest.raises(RegistryError, match="already active"):
        registry.activate(first.ref, reason="again")
    with pytest.raises(RegistryError, match="reason"):
        registry.activate(second.ref, reason=" ")


def test_an_edited_or_reordered_event_log_is_refused(registry):
    first = register(registry, synthetic_record("r1"))
    second = register(registry, synthetic_record("r2"))
    registry.activate(first.ref, reason="a")
    registry.activate(second.ref, reason="b")
    path = registry.root / "events.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0].replace('"a"', '"z"'), lines[1]]) + "\n")
    with pytest.raises(RegistryError, match="edited"):
        registry.events()
    path.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(RegistryError, match="chain"):
        registry.events()


def _rewrite_events(path: Path, payloads: list[dict], *, relink: bool) -> None:
    """Re-hash each line; with relink, also repair every prev_event_sha256 pointer."""

    previous = "0" * 64
    lines = []
    for payload in payloads:
        payload = {k: v for k, v in payload.items() if k != "event_sha256"}
        if relink:
            payload["prev_event_sha256"] = previous
        digest = sha256_json(payload)
        lines.append(canonical_json({**payload, "event_sha256": digest}))
        previous = digest
    path.write_text("\n".join(lines) + "\n")


def test_a_truncated_or_renumbered_event_log_is_refused(registry):
    refs = [register(registry, synthetic_record(f"r{i}")).ref for i in range(3)]
    for ref in refs:
        registry.activate(ref, reason="step")
    path = registry.root / "events.jsonl"
    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    truncated = [{**p, "sequence": i} for i, p in enumerate(payloads[1:])]
    _rewrite_events(path, truncated, relink=False)  # first event dropped, hashes redone
    with pytest.raises(RegistryError, match="chain"):
        registry.events()
    renumbered = [{**p, "sequence": i + 1} for i, p in enumerate(payloads)]
    _rewrite_events(path, renumbered, relink=True)  # a consistent chain, wrong numbering
    with pytest.raises(RegistryError, match="chain"):
        registry.events()


# --- comparison and read API ---


def test_compare_reports_differences_within_one_scope_only(registry):
    first = register(registry, synthetic_record("r1"))
    second = register(
        registry,
        synthetic_record("r2"),
        metrics={"mean_rank_ic": 0.08, "coverage": 0.9},
        training_data={
            "data_class": "synthetic",
            "labeled_frame_sha256": "a" * 64,
            "fold_table_sha256": "d" * 64,
        },
    )
    comparison = registry.compare(first.ref, "equity_xs.ridge@v2")
    table = comparison.table.set_index("metric")
    assert table.loc["mean_rank_ic", "b_minus_a"] == pytest.approx(0.03)
    assert pd.isna(table.loc["coverage", "a"]) and pd.isna(table.loc["coverage", "b_minus_a"])
    assert comparison.same_training_data and not comparison.same_folds
    other = register(
        registry, synthetic_record("r3"), model_id="etf.vol", research_scope="etf:synthetic:h20"
    )
    with pytest.raises(RegistryError, match="different research scopes"):
        registry.compare(second.ref, other.ref)


def test_the_read_api_summarizes_without_writing(registry, tmp_path):
    reader = open_registry(tmp_path / "registry")
    assert reader.summary().empty and reader.summary().columns[0] == "ref"
    first = register(registry, synthetic_record("r1"))
    register(registry, passing_record("r2"))
    registry.activate(first.ref, reason="baseline")
    before = sorted(p.name for p in (tmp_path / "registry").rglob("*"))
    summary = reader.summary().set_index("ref")
    assert summary.loc["equity_xs.ridge@v1", "stage"] == "engineering_only"
    assert summary.loc["equity_xs.ridge@v2", "stage"] == "development_evaluated"
    assert summary["active"].tolist() == [True, False]
    metrics = reader.metrics_frame()
    assert set(metrics["metric"]) == {"mean_rank_ic", "oos_r2_vs_zero"}
    assert sorted(p.name for p in (tmp_path / "registry").rglob("*")) == before
    assert not hasattr(reader, "register")


def test_parents_must_be_earlier_versions_of_the_same_model():
    from stock_agent.registry.entries import build_entry

    with pytest.raises(RegistryError, match="earlier version"):
        build_entry(
            ref=EntryRef("m", 2),
            research_scope="s",
            parent=EntryRef("m", 2),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            config={},
            training_data={"data_class": "synthetic"},
            metrics={},
            artifacts=[ArtifactRef("experiment_run", "runs/x", "0" * 64)],
            evidence={},
            code={},
        )


def test_references_round_trip_and_reject_unsafe_ids():
    assert EntryRef.parse("etf.vol_har@v12") == EntryRef("etf.vol_har", 12)
    for text in ("Upper@v1", "a/b@v1", "x@v0", "x@1", "../x@v1"):
        with pytest.raises(RegistryError):
            EntryRef.parse(text)
