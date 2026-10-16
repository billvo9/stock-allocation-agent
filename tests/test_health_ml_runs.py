"""
The latest ML experiment run (the runner's attempt ledger) in every state,
and each registered version's evidence stage with the reasons that block its
next stage, as the registry's own assessment states them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stock_agent.health import fixtures, ml_runs, registry
from stock_agent.health.ml_runs import ml_runs_check
from stock_agent.health.model import HealthState
from stock_agent.model_diagnostics import artifacts
from stock_agent.registry import open_registry

INSUFFICIENT = HealthState.INSUFFICIENT_EVIDENCE


def _attempt(started: str, status: str = "completed", **extra) -> dict:
    record = {
        "started_at": started,
        "experiment": "etf_return_risk_null",
        "config_sha256": "a" * 64,
        "mode": "development",
        "status": status,
    }
    if status == "completed":
        record |= {
            "run_id": f"run-{started[:10]}",
            "results_sha256": "b" * 64,
            "registered": ["etf.ridge@v1"],
            "git_sha": "c" * 40,
            "git_dirty": False,
        }
    else:
        record["error"] = "ValueError: planted failure"
    return record | extra


def _ledger(root: Path, *records, raw_lines=()) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records] + list(raw_lines)
    (root / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_the_ledger_file_name_is_the_runners():
    assert ml_runs.LEDGER_FILE == artifacts.LEDGER_FILE


def test_no_root_or_no_ledger_is_insufficient_evidence(tmp_path):
    for root, reason in ((None, "input_not_configured"), (tmp_path, "input_missing")):
        check = ml_runs_check(root)
        assert (check.state, check.reason, check.value) == (INSUFFICIENT, reason, None)


def test_a_completed_real_run_from_a_clean_tree_is_healthy(tmp_path):
    root = _ledger(tmp_path, _attempt("2026-10-14T10:00:00+00:00"))
    check = ml_runs_check(root, data_classes={"etf.ridge@v1": "research"})
    assert (check.state, check.value) == (HealthState.HEALTHY, "completed")
    assert "registered: etf.ridge@v1" in check.detail
    assert "attempts in the ledger: 1 (1 completed, 0 failed)" in check.detail


def test_the_newest_attempt_decides_and_a_failure_fails(tmp_path):
    root = _ledger(
        tmp_path,
        _attempt("2026-10-15T10:00:00+00:00", "failed"),
        _attempt("2026-10-14T10:00:00+00:00"),
    )
    check = ml_runs_check(root)
    assert (check.state, check.reason, check.value) == (
        HealthState.FAILED,
        "ml_run_failed",
        "failed",
    )
    assert "planted failure" in check.detail


def test_a_run_from_a_dirty_tree_is_degraded(tmp_path):
    root = _ledger(tmp_path, _attempt("2026-10-14T10:00:00+00:00", git_dirty=True))
    check = ml_runs_check(root, data_classes={"etf.ridge@v1": "research"})
    assert (check.state, check.reason) == (HealthState.DEGRADED, "dirty_tree")


def test_a_run_of_synthetic_versions_is_capped_in_the_private_scope(tmp_path):
    root = _ledger(tmp_path, _attempt("2026-10-14T10:00:00+00:00"))
    classes = {"etf.ridge@v1": "synthetic"}
    assert ml_runs_check(root, data_classes=classes).reason == "synthetic_input"
    assert ml_runs_check(root, data_classes=classes, allow_synthetic=True).state is (
        HealthState.HEALTHY
    )


def test_unreadable_lines_degrade_and_an_unreadable_ledger_is_unavailable(tmp_path):
    mixed = _ledger(tmp_path / "a", _attempt("2026-10-14T10:00:00+00:00"), raw_lines=["{oops"])
    check = ml_runs_check(mixed, data_classes={"etf.ridge@v1": "research"})
    assert (check.state, check.reason) == (HealthState.DEGRADED, "unreadable_inputs")
    broken = _ledger(tmp_path / "b", raw_lines=["{oops", '{"status": "unknown"}'])
    check = ml_runs_check(broken)
    assert (check.state, check.reason, check.value) == (
        HealthState.UNAVAILABLE,
        "unrecognized_format",
        None,
    )


@pytest.fixture
def registry_root(tmp_path) -> Path:
    return fixtures.build_registry(tmp_path / "ml-fixture")


def test_every_version_is_listed_with_its_stage_and_blockers(registry_root):
    checks = {
        c.check_id: c
        for c in registry.registry_checks(
            registry_root, now=lambda: datetime(2026, 10, 15, tzinfo=UTC), allow_synthetic=True
        )
    }
    stages = checks["registry.stages"]
    reader = open_registry(registry_root)
    for entry in reader.entries():
        assessment = reader.assessment(entry.ref)
        line = next(row for row in stages.detail.splitlines() if row.startswith(str(entry.ref)))
        assert f"({entry.research_scope}, {entry.data_class}): {assessment.stage}" in line
        assert f"{assessment.next_stage} blocked by: " in line
        for blocker in assessment.blockers:
            assert blocker.reason in line
    assert "data_class_engineering_only" in stages.detail


def test_registry_data_classes_feed_the_ml_runs_check(registry_root, tmp_path):
    classes = registry.data_classes(registry_root)
    assert set(classes.values()) == {"synthetic"} and len(classes) == 2
    assert registry.data_classes(None) == {} and registry.data_classes(tmp_path / "no") == {}
