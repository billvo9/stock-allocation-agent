"""
The GitHub run list for the local monitor: `gh run list` per workflow through
an injected runner (no test runs gh), the daily smoke-test check in every
state, and the public CI run never calling gh.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stock_agent.health import collect, github
from stock_agent.health.model import HealthState
from stock_agent.health.verification import (
    DAILY_OVERDUE_AFTER,
    RunList,
    daily_health_check,
    run_list_checks,
)

NOW = datetime(2026, 10, 15, 18, tzinfo=UTC)
SHA = "f98668a5b2c3" + "0" * 28


def _run(workflow: str, conclusion: str | None, created: datetime, run_id: int, **extra):
    return {
        "databaseId": run_id,
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion or "",
        "headSha": SHA,
        "headBranch": extra.get("branch", "main"),
        "workflowName": workflow,
        "createdAt": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": f"https://github.com/o/r/actions/runs/{run_id}",
        "event": "schedule" if workflow == "Model health" else "push",
    }


class FakeGh:
    def __init__(self, lists=None, error=None, code=0, stderr=""):
        self.lists = lists or {}
        self.error, self.code, self.stderr = error, code, stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if self.error is not None:
            raise self.error
        workflow = argv[argv.index("--workflow") + 1]
        out = json.dumps(self.lists.get(workflow, []))
        return subprocess.CompletedProcess(argv, self.code, out, self.stderr)


LISTS = {
    "ci.yml": [_run("CI", "success", NOW - timedelta(hours=2), 2)],
    "health.yml": [_run("Model health", "success", NOW - timedelta(hours=12), 3)],
}


def test_gh_is_called_once_per_workflow_and_the_list_kept_as_evidence(tmp_path):
    gh = FakeGh(LISTS)
    runs = github.collect_runs(gh, now=lambda: NOW, evidence_dir=tmp_path)
    assert [call[call.index("--workflow") + 1] for call in gh.calls] == ["ci.yml", "health.yml"]
    assert all(call[:3] == ["gh", "run", "list"] for call in gh.calls)
    assert all(github.REPOSITORY in call and "main" in call for call in gh.calls)
    assert runs.measured_at == NOW and runs.evidence == str(tmp_path / "ci_runs.json")
    assert len(json.loads((tmp_path / "ci_runs.json").read_text(encoding="utf-8"))) == 2
    main, daily = run_list_checks(runs, now=NOW)
    assert (main.check_id, main.state, main.value) == (
        "ci.main_runs",
        HealthState.HEALTHY,
        "success",
    )
    assert (daily.check_id, daily.state) == ("ci.daily_health", HealthState.HEALTHY)


@pytest.mark.parametrize(
    ("gh", "detail"),
    [
        (FakeGh(error=FileNotFoundError("gh")), "not installed"),
        (FakeGh(error=subprocess.TimeoutExpired("gh", 60)), "TimeoutExpired"),
        (
            FakeGh(code=4, stderr="To get started with GitHub CLI, please run: gh auth login"),
            "exited 4",
        ),
    ],
)
def test_without_a_working_gh_the_ci_checks_are_insufficient_evidence(tmp_path, gh, detail):
    runs = github.collect_runs(gh, now=lambda: NOW, evidence_dir=tmp_path)
    assert runs.text is None and not (tmp_path / "ci_runs.json").exists()
    for check in run_list_checks(runs, now=NOW):
        assert (check.state, check.reason, check.value) == (
            HealthState.INSUFFICIENT_EVIDENCE,
            "gh_unavailable",
            None,
        )
        assert detail in check.detail


def test_output_that_is_not_a_run_list_is_unavailable(tmp_path):
    class Garbage(FakeGh):
        def __call__(self, argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "<html>rate limited</html>", "")

    runs = github.collect_runs(Garbage(), now=lambda: NOW, evidence_dir=tmp_path)
    for check in run_list_checks(runs, now=NOW):
        assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "unrecognized_format")


def _daily(runs):
    return daily_health_check(json.dumps(runs), measured_at=NOW, now=NOW, evidence="runs.json")


def test_the_daily_check_reads_only_the_health_workflow_on_main():
    check = _daily(
        [
            _run("CI", "failure", NOW - timedelta(hours=1), 9),
            _run("Model health", "success", NOW - timedelta(hours=5), 8, branch="feature"),
            _run("Model health", "success", NOW - timedelta(hours=20), 7),
        ]
    )
    assert check.state is HealthState.HEALTHY
    assert (
        "run 7" in check.detail and "2 runs of other workflows or branches ignored" in check.detail
    )


def test_a_failed_daily_run_fails():
    check = _daily([_run("Model health", "failure", NOW - timedelta(hours=3), 4)])
    assert (check.state, check.reason) == (HealthState.FAILED, "ci_failed")


def test_a_daily_run_that_is_overdue_is_degraded():
    old = NOW - DAILY_OVERDUE_AFTER - timedelta(minutes=1)
    check = _daily([_run("Model health", "success", old, 4)])
    assert (check.state, check.reason) == (HealthState.DEGRADED, "daily_run_overdue")


def test_a_cancelled_daily_run_is_degraded_and_one_in_progress_is_not_judged():
    cancelled = _daily([_run("Model health", "cancelled", NOW - timedelta(hours=1), 5)])
    assert (cancelled.state, cancelled.reason) == (HealthState.DEGRADED, "ci_inconclusive")
    running = _daily([_run("Model health", None, NOW - timedelta(minutes=5), 6)])
    assert (running.state, running.reason) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "no_completed_runs",
    )


def test_no_run_list_is_insufficient_evidence_never_healthy():
    for check in run_list_checks(RunList(None, None, None), now=NOW):
        assert (check.state, check.value) == (HealthState.INSUFFICIENT_EVIDENCE, None)
        assert check.reason == "not_collected" and "--github" in check.detail


def _environment(root, runner):
    return collect.Environment(
        root=root,
        env={},
        now=lambda: NOW,
        runner=runner,
        installed=lambda name: None,
        python="/repo/.venv/bin/python",
    )


class Machine(FakeGh):
    def __call__(self, argv, **kwargs):
        argv = [str(part) for part in argv]
        if argv[0] == "gh":
            return super().__call__(argv, **kwargs)
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_only_the_local_run_with_github_asked_for_calls_gh(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    machine = Machine(LISTS)
    collect.run_ci(_environment(root, machine), tmp_path / "ci", verify=None)
    collect.run_local(
        _environment(root, machine), tmp_path / "a", verify=None, inputs=collect.PrivateInputs()
    )
    assert not [call for call in machine.calls if call[0] == "gh"]
    summary, _ = collect.run_local(
        _environment(root, machine),
        tmp_path / "b",
        verify=None,
        inputs=collect.PrivateInputs(github=True),
    )
    assert len([call for call in machine.calls if call[0] == "gh"]) == 2
    assert summary.check("ci.daily_health").state is HealthState.HEALTHY
    evidence = Path(summary.check("ci.main_runs").evidence)
    assert evidence.name == "ci_runs.json" and evidence.is_file()
