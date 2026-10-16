"""
Policy safety and diagnostics availability from backtest summaries and RL
evaluation reports, in every state: constraint violations, fallback
decisions, and which of risk, turnover and drawdown are reported. Missing
inputs are never healthy; nothing is recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_agent.health import fixtures
from stock_agent.health.model import CONSTRAINT_BREACHED, HealthState
from stock_agent.health.policy import load_evaluation, policy_checks

INSUFFICIENT = HealthState.INSUFFICIENT_EVIDENCE


def _backtest(inputs: Path, **changes) -> Path:
    document = fixtures.backtest_summary()
    measures = changes.pop("measures", {})
    document |= {"synthetic": False, "transition_status": "approved", **changes}
    document["measures"] |= measures
    fixtures.write_input(inputs, document)
    return inputs / "backtest_summary"


def _method(kind="baseline", decisions=100, invalid=0, fallbacks=0, reasons=None):
    return {
        "kind": kind,
        "group": None,
        "episodes": 4,
        "decisions": decisions,
        "trades": 10,
        "all_finite": True,
        "invalid_targets": invalid,
        "fallback_decisions": fallbacks,
        "fallback_reasons": reasons or {},
    }


def _evaluation(root: Path, *, label="engineering_only_unapproved_transition", **methods) -> Path:
    report = {
        "schema": "rl_engineering_report",
        "schema_version": "1.0.0",
        "label": label,
        "run_id": "ppo-run-1",
        "evaluated_at": "2026-10-15T12:00:00.000000Z",
        "fairness": "passed",
        "methods": methods
        or {
            "equal_weight": _method(),
            "guarded:ppo_seed_0": _method(
                "ppo_guarded", fallbacks=3, reasons={"out_of_distribution": 3}
            ),
        },
        "unavailable": {},
    }
    path = root / "ppo-run-1" / "evaluation_test_1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return root


def _checks(backtest_dir=None, rl_runs=None, **kwargs):
    return {c.check_id: c for c in policy_checks(backtest_dir, rl_runs, **kwargs)}


def test_with_no_input_every_policy_check_is_insufficient_evidence(tmp_path):
    for checks in (_checks(), _checks(tmp_path / "none", tmp_path / "nothing")):
        assert list(checks) == [
            "portfolio.violations",
            "portfolio.fallbacks",
            "portfolio.diagnostics",
        ]
        for check in checks.values():
            assert (check.state, check.value) == (INSUFFICIENT, None)


def test_a_clean_approved_backtest_is_healthy(tmp_path):
    directory = _backtest(
        tmp_path,
        constraint_violations=0,
        fallback_events=0,
        measures={"volatility": 0.11},
    )
    checks = _checks(directory)
    assert {c.state for c in checks.values()} == {HealthState.HEALTHY}
    assert checks["portfolio.diagnostics"].value == 3
    assert "risk (volatility): 0.11" in checks["portfolio.diagnostics"].detail


def test_a_constraint_violation_fails_with_the_constraint_reason(tmp_path):
    checks = _checks(_backtest(tmp_path, constraint_violations=2, fallback_events=0))
    violations = checks["portfolio.violations"]
    assert (violations.state, violations.reason, violations.value) == (
        HealthState.FAILED,
        CONSTRAINT_BREACHED,
        2,
    )
    assert "nothing acts on this alert" in violations.summary


def test_fallback_decisions_are_degraded_with_their_reasons(tmp_path):
    directory = _backtest(
        tmp_path, fallback_events=4, fallback_reasons={"stale_inputs": 1, "model_unhealthy": 3}
    )
    fallbacks = _checks(directory)["portfolio.fallbacks"]
    assert (fallbacks.state, fallbacks.reason, fallbacks.value) == (
        HealthState.DEGRADED,
        "fallbacks_recorded",
        4,
    )
    assert "model_unhealthy 3, stale_inputs 1" in fallbacks.detail


def test_a_backtest_that_does_not_report_events_is_insufficient_evidence(tmp_path):
    checks = _checks(_backtest(tmp_path))
    for check_id in ("portfolio.violations", "portfolio.fallbacks"):
        assert (checks[check_id].state, checks[check_id].reason) == (INSUFFICIENT, "not_reported")
        assert "not reported" in checks[check_id].detail


def test_risk_not_reported_leaves_diagnostics_insufficient(tmp_path):
    diagnostics = _checks(_backtest(tmp_path))["portfolio.diagnostics"]
    assert (diagnostics.state, diagnostics.reason, diagnostics.value) == (
        INSUFFICIENT,
        "diagnostics_missing",
        2,
    )
    assert "risk (volatility): not reported" in diagnostics.detail


def test_an_unapproved_transition_is_a_visible_limitation(tmp_path):
    directory = _backtest(
        tmp_path,
        transition_status="engineering_only_unapproved_transition",
        constraint_violations=0,
        fallback_events=0,
        measures={"volatility": 0.1},
    )
    for check in _checks(directory).values():
        assert (check.state, check.reason) == (HealthState.DEGRADED, "unapproved_transition")


def test_rl_evaluation_counts_are_read_but_synthetic_in_the_private_scope(tmp_path):
    rl_runs = _evaluation(tmp_path / "rl")
    checks = _checks(None, rl_runs)
    fallbacks = checks["portfolio.fallbacks"]
    assert fallbacks.value == 3
    assert "guarded:ppo_seed_0 3 of 100" in fallbacks.detail
    assert "out_of_distribution 3" in fallbacks.detail
    # Fallbacks (degraded) and an unapproved transition (degraded) outrank the cap.
    assert fallbacks.state is HealthState.DEGRADED
    assert checks["portfolio.violations"].value == 0
    allowed = _checks(
        None,
        _evaluation(tmp_path / "rl2", label="approved_transition", equal_weight=_method()),
        allow_synthetic=True,
    )
    assert allowed["portfolio.violations"].state is HealthState.HEALTHY


def test_a_clean_synthetic_evaluation_is_capped_at_insufficient_evidence(tmp_path):
    rl_runs = _evaluation(tmp_path / "rl", label="approved_transition", equal_weight=_method())
    violations = _checks(None, rl_runs)["portfolio.violations"]
    assert (violations.state, violations.reason) == (INSUFFICIENT, "synthetic_input")


def test_invalid_targets_in_an_rl_evaluation_fail(tmp_path):
    rl_runs = _evaluation(tmp_path / "rl", ppo_seed_0=_method("ppo", invalid=5))
    violations = _checks(None, rl_runs)["portfolio.violations"]
    assert (violations.state, violations.value) == (HealthState.FAILED, 5)
    assert "ppo_seed_0 5" in violations.detail


def test_an_unreadable_rl_report_is_unavailable_never_ignored(tmp_path):
    root = tmp_path / "rl"
    (root / "run").mkdir(parents=True)
    (root / "run" / "evaluation_test_1.json").write_text("{not json", encoding="utf-8")
    assert load_evaluation(root).state is HealthState.UNAVAILABLE
    directory = _backtest(tmp_path / "inputs", constraint_violations=0, fallback_events=0)
    checks = _checks(directory, root)
    for check_id in ("portfolio.violations", "portfolio.fallbacks"):
        assert (checks[check_id].state, checks[check_id].reason) == (
            HealthState.UNAVAILABLE,
            "unrecognized_format",
        )


def test_the_newest_rl_report_is_used(tmp_path):
    root = _evaluation(tmp_path / "rl", ppo_seed_0=_method("ppo", invalid=5))
    newer = json.loads((root / "ppo-run-1" / "evaluation_test_1.json").read_text("utf-8"))
    newer |= {"evaluated_at": "2026-10-16T12:00:00.000000Z", "methods": {"a": _method()}}
    (root / "ppo-run-1" / "evaluation_test_2.json").write_text(json.dumps(newer), "utf-8")
    assert load_evaluation(root).path.name == "evaluation_test_2.json"


@pytest.mark.parametrize("bad", [{"constraint_violations": -1}, {"fallback_events": "two"}])
def test_a_malformed_optional_field_makes_the_backtest_unreadable(tmp_path, bad):
    checks = _checks(_backtest(tmp_path, **bad))
    assert checks["portfolio.diagnostics"].state is HealthState.UNAVAILABLE
