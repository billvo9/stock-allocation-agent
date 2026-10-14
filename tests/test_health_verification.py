"""
CI and test health: parsing verify.py's real output format, the states of
`ci.verify` and `ci.tests`, staleness, and the latest CI run on main from a
`gh run list` file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from stock_agent.health.model import HealthState
from stock_agent.health.verification import (
    STALE_AFTER,
    main_runs_check,
    parse_verify_output,
    verify_checks,
)

NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)


def verify_text(
    first="1958 passed in 73.68s (0:01:13)",
    second="1958 passed in 81.11s (0:01:21)",
    gates=("PASS", "PASS", "PASS", "PASS"),
) -> str:
    """The shape of a real scripts/verify.py output (2026-10-14)."""

    names = (
        "pytest -q",
        "python -m pytest -q",
        "ruff format --check src tests scripts .claude/hooks",
        "ruff check src tests scripts .claude/hooks",
    )
    codes = ["0" if gate == "PASS" else "1" for gate in gates]
    summary = "\n".join(
        f"  {gate}  {name}  (exit {code})" for gate, name, code in zip(gates, names, codes)
    )
    return f"""verify: repo /repo
verify: interpreter /repo/.venv/bin/python (Python 3.12)
pytest 9.1.1
ruff 0.16.2

==> .venv/bin/pytest -q
........................................................................ [  3%]
{first}
<== exit {codes[0]}

==> .venv/bin/python -m pytest -q
........................................................................ [  3%]
{second}
<== exit {codes[1]}

==> .venv/bin/ruff format --check src tests scripts .claude/hooks
257 files already formatted
<== exit {codes[2]}

==> .venv/bin/ruff check src tests scripts .claude/hooks
All checks passed!
<== exit {codes[3]}

verify summary:
{summary}
verify: {"all gates passed" if all(g == "PASS" for g in gates) else "FAILED"}
"""


def checks(text, measured_at=NOW - timedelta(minutes=5)):
    return verify_checks(text, measured_at=measured_at, now=NOW, evidence="evidence/verify.log")


def test_parses_gates_and_both_pytest_passes():
    result = parse_verify_output(verify_text())
    assert [gate.name for gate in result.gates][:2] == ["pytest -q", "python -m pytest -q"]
    assert all(gate.passed for gate in result.gates)
    assert [run.passed for run in result.pytest_runs] == [1958, 1958]
    assert result.pytest_runs[1].seconds == pytest.approx(81.11)


def test_a_green_run_is_healthy_with_the_actual_counts():
    gates, tests = checks(verify_text())
    assert (gates.state, gates.value, gates.unit) == (HealthState.HEALTHY, 4, "of 4 gates passed")
    assert (tests.state, tests.value) == (HealthState.HEALTHY, 1958)
    assert "1,958 tests passed" in tests.summary
    assert tests.evidence == "evidence/verify.log"


def test_failures_and_errors_are_counted():
    text = verify_text(
        first="2 failed, 1955 passed, 1 error in 70.00s",
        second="2 failed, 1955 passed, 1 error in 70.00s",
        gates=("FAIL", "FAIL", "PASS", "PASS"),
    )
    result = parse_verify_output(text)
    assert (result.pytest_runs[0].failed, result.pytest_runs[0].errors) == (2, 1)
    gates, tests = checks(text)
    assert (gates.state, gates.reason, gates.value) == (HealthState.FAILED, "gate_failed", 2)
    assert (tests.state, tests.reason) == (HealthState.FAILED, "tests_failed")
    assert tests.summary == "3 tests failed or errored."


def test_a_lint_failure_fails_the_gates_but_not_the_tests():
    gates, tests = checks(verify_text(gates=("PASS", "PASS", "FAIL", "PASS")))
    assert gates.state is HealthState.FAILED
    assert tests.state is HealthState.HEALTHY


def test_disagreeing_passes_are_degraded():
    _, tests = checks(verify_text(second="1957 passed, 1 skipped in 70.00s"))
    assert (tests.state, tests.reason) == (HealthState.DEGRADED, "passes_disagree")


def test_an_old_green_result_is_degraded_not_healthy():
    gates, tests = checks(verify_text(), measured_at=NOW - STALE_AFTER - timedelta(hours=1))
    assert (gates.state, gates.reason) == (HealthState.DEGRADED, "stale_evidence")
    assert (tests.state, tests.reason) == (HealthState.DEGRADED, "stale_evidence")


def test_no_output_is_insufficient_evidence_never_healthy():
    gates, tests = verify_checks(None, measured_at=None, now=NOW, evidence=None)
    for check in (gates, tests):
        assert (check.state, check.reason, check.value) == (
            HealthState.INSUFFICIENT_EVIDENCE,
            "not_collected",
            None,
        )


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("something else entirely\n", "unrecognized_output"),
        ("verify summary:\n", "unrecognized_output"),
        (
            "verify: project virtual environment not usable: /r/.venv/bin is missing python.\n",
            "verify_environment_unusable",
        ),
        (
            "verify: .venv runs Python 3.11; this project requires 3.12.\n",
            "verify_environment_unusable",
        ),
    ],
)
def test_unreadable_output_is_unavailable(text, reason):
    gates, tests = checks(text)
    assert (gates.state, gates.reason) == (HealthState.UNAVAILABLE, reason)
    assert tests.state is HealthState.UNAVAILABLE


def test_gates_without_a_pytest_line_leave_the_tests_unavailable():
    text = verify_text().replace("1958 passed in 73.68s (0:01:13)", "INTERNALERROR")
    text = text.replace("1958 passed in 81.11s (0:01:21)", "INTERNALERROR")
    gates, tests = checks(text)
    assert gates.state is HealthState.HEALTHY
    assert (tests.state, tests.reason) == (HealthState.UNAVAILABLE, "no_test_counts")


def test_a_pass_that_ran_nothing_fails():
    _, tests = checks(verify_text(first="no tests ran in 0.01s", second="0 passed in 0.01s"))
    assert tests.state is HealthState.FAILED


# --- CI runs on main ---------------------------------------------------------------


def run(run_id, conclusion, created, status="completed"):
    return {
        "databaseId": run_id,
        "status": status,
        "conclusion": conclusion,
        "headSha": "6b33c1b215ef89acc600d72021dd013ba6fd56a2",
        "createdAt": created,
        "url": f"https://github.com/o/r/actions/runs/{run_id}",
        "event": "push",
    }


def main_runs(runs, measured_at=NOW - timedelta(hours=1)):
    return main_runs_check(
        json.dumps(runs), measured_at=measured_at, now=NOW, evidence="/private/ci_runs.json"
    )


def test_the_latest_completed_run_decides():
    check = main_runs(
        [
            run(3, None, "2026-10-14T17:00:00Z", status="in_progress"),
            run(2, "success", "2026-10-14T12:00:00Z"),
            run(1, "failure", "2026-10-13T12:00:00Z"),
        ]
    )
    assert (check.state, check.value) == (HealthState.HEALTHY, "success")
    assert "run 2 on 6b33c1b215ef" in check.detail
    assert "1 of the last 2 completed runs failed" in check.detail


@pytest.mark.parametrize(
    ("conclusion", "state", "reason"),
    [
        ("failure", HealthState.FAILED, "ci_failed"),
        ("startup_failure", HealthState.FAILED, "ci_failed"),
        ("cancelled", HealthState.DEGRADED, "ci_inconclusive"),
    ],
)
def test_a_failed_or_inconclusive_latest_run(conclusion, state, reason):
    check = main_runs([run(9, conclusion, "2026-10-14T12:00:00Z")])
    assert (check.state, check.reason) == (state, reason)


def test_run_list_states_without_a_judgeable_run():
    assert main_runs([]).reason == "no_completed_runs"
    stale = main_runs(
        [run(1, "success", "2026-10-01T12:00:00Z")], measured_at=NOW - timedelta(days=9)
    )
    assert (stale.state, stale.reason) == (HealthState.DEGRADED, "stale_evidence")
    garbage = main_runs_check("{}", measured_at=NOW, now=NOW, evidence="x")
    assert (garbage.state, garbage.reason) == (HealthState.UNAVAILABLE, "unrecognized_format")
    absent = main_runs_check(None, measured_at=None, now=NOW, evidence=None)
    assert (absent.state, absent.reason) == (HealthState.INSUFFICIENT_EVIDENCE, "not_collected")
