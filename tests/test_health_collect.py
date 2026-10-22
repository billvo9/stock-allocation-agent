"""
The two health runs with a fake machine (clock, environment, subprocess
runner, installed versions): the public-safe CI run, the private local run
and its guards, history and deterioration, the run context's allowlist (no
secrets), the code state, and scripts/health_check.py's exit codes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.data.funds.adapter import SYNTHETIC_FUND_IDS, SyntheticFundAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.health import collect, fixtures, forecast
from stock_agent.health import stock_lab as health_stock_lab
from stock_agent.health.context import code_state, run_context
from stock_agent.health.model import AlertKind, Condition, HealthState, Scope, read_summary

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)
HOME = Path("/Users/someone")
SHA = "6b33c1b215ef89acc600d72021dd013ba6fd56a2"
VERIFY_OK = """verify: repo /repo
==> .venv/bin/pytest -q
1958 passed in 73.68s (0:01:13)
<== exit 0
==> .venv/bin/python -m pytest -q
1958 passed in 81.11s (0:01:21)
<== exit 0

verify summary:
  PASS  pytest -q  (exit 0)
  PASS  python -m pytest -q  (exit 0)
  PASS  ruff format --check src tests scripts .claude/hooks  (exit 0)
  PASS  ruff check src tests scripts .claude/hooks  (exit 0)
verify: all gates passed
"""
PUBLIC_CHECKS = [
    "ci.verify",
    "ci.tests",
    "deps.pins",
    "deps.pip_check",
    "contract.publication",
    "contract.fund_data",
    "contract.generators",
    "smoke.data_monitors",
    "smoke.input_readers",
]
PRIVATE_CHECKS = [
    "ci.main_runs",
    "ci.daily_health",
    "data.freshness",
    "data.coverage",
    "data.missing_sessions",
    "data.vendor_discrepancy",
    "data.anchors",
    "data.store_batches",
    "data.window_coverage",
    "training.health",
    "training.ml_runs",
    "registry.entries",
    "registry.stages",
    "registry.selection_history",
    "registry.experiments",
    "forecast.coverage",
    "forecast.calibration",
    "forecast.ledger",
    "forecast.ledger_backup",
    "forecast.ledger_sessions",
    "forecast.ledger_scores",
    "forecast.ledger_trial",
    "portfolio.turnover",
    "portfolio.costs",
    "portfolio.concentration",
    "portfolio.drawdown",
    "data.stock_lab_freshness",
    "data.stock_lab_tickers",
    "forecast.stock_lab_job",
    "forecast.stock_lab_scores",
    "forecast.stock_lab_drift",
    "forecast.stock_lab_ledger",
    "portfolio.violations",
    "portfolio.fallbacks",
    "portfolio.diagnostics",
]


class FakeMachine:
    def __init__(self, *, verify=VERIFY_OK, pip_code=0, ignored=True, git_ok=True):
        self.verify, self.pip_code, self.ignored, self.git_ok = verify, pip_code, ignored, git_ok
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        argv = [str(part) for part in argv]
        self.calls.append(argv)
        if argv[0] == "git":
            if not self.git_ok:
                raise FileNotFoundError("git")
            if argv[1] == "rev-parse":
                return subprocess.CompletedProcess(argv, 0, SHA + "\n", "")
            if argv[1] == "status":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[1] == "check-ignore":
                return subprocess.CompletedProcess(argv, 0 if self.ignored else 1)
        if argv[1:4] == ["-m", "pip", "check"]:
            out = "No broken requirements found.\n" if self.pip_code == 0 else "a 1 needs b.\n"
            return subprocess.CompletedProcess(argv, self.pip_code, out, "")
        if argv[-1].endswith("verify.py"):
            return subprocess.CompletedProcess(argv, 0, self.verify, None)
        raise AssertionError(f"unexpected command {argv}")


@pytest.fixture
def root(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("pandas\npytest==9.1.1\n", encoding="utf-8")
    return repo


def environment(root, machine=None, env=None, now=NOW) -> collect.Environment:
    clock = iter(now + timedelta(seconds=index) for index in range(10_000))
    return collect.Environment(
        root=root,
        env=env or {},
        now=lambda: next(clock),
        runner=machine or FakeMachine(),
        installed={"pytest": "9.1.1"}.get,
        python="/repo/.venv/bin/python",
        home=HOME,
    )


def ids(summary) -> list[str]:
    return [check.check_id for check in summary.checks]


# --- run context and code state ------------------------------------------------------

ACTIONS = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_WORKFLOW": "Model health",
    "GITHUB_RUN_ID": "123456789",
    "GITHUB_RUN_NUMBER": "42",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_EVENT_NAME": "schedule",
    "GITHUB_REF_NAME": "main",
    "GITHUB_SHA": SHA,
    "GITHUB_REPOSITORY": "billvo9/stock-allocation-agent",
    "GITHUB_SERVER_URL": "https://github.com",
    "RUNNER_OS": "Linux",
    "GITHUB_TOKEN": "ghs_secretvalue",
    "ACTIONS_RUNTIME_TOKEN": "secret-runtime",
    "TIINGO_API_KEY": "secret-key",
}


def test_run_context_copies_only_allowlisted_actions_metadata():
    context = run_context(ACTIONS)
    assert context == {
        "environment": "github_actions",
        "workflow": "Model health",
        "run_id": "123456789",
        "run_number": "42",
        "run_attempt": "1",
        "event": "schedule",
        "ref": "main",
        "sha": SHA,
        "repository": "billvo9/stock-allocation-agent",
        "runner_os": "Linux",
        "run_url": "https://github.com/billvo9/stock-allocation-agent/actions/runs/123456789",
    }
    assert "secret" not in json.dumps(context)


def test_run_context_drops_values_that_fail_their_pattern():
    context = run_context({**ACTIONS, "GITHUB_RUN_ID": "1; curl x", "GITHUB_REF_NAME": "a\nb"})
    assert (context["run_id"], context["ref"], context["run_url"]) == (None, None, None)
    assert run_context({"GITHUB_TOKEN": "x"}) == {"environment": "local"}


def test_code_state_without_git_is_unknown(root):
    assert code_state(root, FakeMachine()).sha == SHA
    unknown = code_state(root, FakeMachine(git_ok=False))
    assert (unknown.sha, unknown.dirty) == (None, None)


@pytest.mark.parametrize(
    ("status", "dirty"),
    [
        ("", False),
        ("?? .venv\n", False),  # a session worktree's symlinked virtual environment
        ("?? src/stock_agent/health/new.py\n", True),  # uncommitted code runs too
        (" M README.md\n?? .venv\n", True),
    ],
)
def test_untracked_files_other_than_the_venv_make_the_tree_dirty(root, status, dirty):
    class Status(FakeMachine):
        def __call__(self, argv, **kwargs):
            if list(argv[:2]) == ["git", "status"]:
                return subprocess.CompletedProcess(argv, 0, status, "")
            return super().__call__(argv, **kwargs)

    assert code_state(root, Status()).dirty is dirty


# --- the CI run ----------------------------------------------------------------------


def test_the_ci_run_writes_a_public_safe_summary(root, tmp_path):
    output = tmp_path / "out"
    env = environment(root, env=ACTIONS)
    verify = collect.run_verify(env, public=True)
    summary, path = collect.run_ci(env, output, verify=verify)

    assert path == output / "health_summary.json"
    assert read_summary(path) == summary
    assert ids(summary) == PUBLIC_CHECKS
    assert summary.scope is Scope.SYNTHETIC
    assert summary.overall_state is HealthState.HEALTHY
    assert (summary.code_sha, summary.code_dirty) == (SHA, False)
    assert summary.check("ci.tests").value == 1958
    assert summary.check("ci.verify").evidence == "evidence/verify.log"
    assert summary.check("deps.pip_check").evidence == "evidence/pip_check.txt"
    log = (output / "evidence" / "verify.log").read_text()
    assert log.startswith(collect.VERIFY_STAMP) and log.endswith(VERIFY_OK)
    for check in summary.checks:  # every evidence pointer names a file that exists
        if check.evidence and check.evidence.startswith("evidence/"):
            assert (output / check.evidence).is_file(), check.check_id
    text = path.read_text(encoding="utf-8")
    assert "secret" not in text and "ghs_" not in text
    assert str(tmp_path) not in text  # no absolute local paths in a public summary


def test_a_ci_run_without_verify_is_insufficient_evidence(root, tmp_path):
    summary, _ = collect.run_ci(environment(root), tmp_path / "out", verify=None)
    assert summary.overall_state is HealthState.INSUFFICIENT_EVIDENCE
    assert summary.check("ci.tests").reason == "not_collected"


def test_verify_that_cannot_start_is_recorded_as_unavailable(root, tmp_path):
    class NoVerify(FakeMachine):
        def __call__(self, argv, **kwargs):
            if str(argv[-1]).endswith("verify.py"):
                raise FileNotFoundError("python")
            return super().__call__(argv, **kwargs)

    env = environment(root, NoVerify())
    verify = collect.run_verify(env, public=True)
    summary, _ = collect.run_ci(env, tmp_path / "out", verify=verify)
    assert summary.check("ci.verify").state is HealthState.UNAVAILABLE


# --- the private local run ----------------------------------------------------------------


def test_the_local_run_refuses_to_run_in_github_actions(root, tmp_path):
    with pytest.raises(collect.LocalRunRefused, match="GitHub Actions cannot see"):
        collect.run_local(
            environment(root, env={"GITHUB_ACTIONS": "true"}),
            tmp_path / "private",
            verify=None,
            inputs=collect.PrivateInputs(),
        )


def test_the_local_run_refuses_an_output_git_would_track(root):
    with pytest.raises(collect.LocalRunRefused, match="not ignored by Git"):
        collect.run_local(
            environment(root, FakeMachine(ignored=False)),
            root / "docs" / "health",
            verify=None,
            inputs=collect.PrivateInputs(),
        )


def test_the_local_run_accepts_an_ignored_path_inside_the_repository(root):
    machine = FakeMachine(ignored=True)
    _, latest = collect.run_local(
        environment(root, machine),
        root / "reports" / "health",
        verify=None,
        inputs=collect.PrivateInputs(),
    )
    assert latest == (root / "reports" / "health" / "latest.json").resolve()
    assert any(call[:2] == ["git", "check-ignore"] for call in machine.calls)


def test_a_local_run_without_private_inputs_is_never_healthy(root, tmp_path):
    summary, _ = collect.run_local(
        environment(root), tmp_path / "private", verify=None, inputs=collect.PrivateInputs()
    )
    assert ids(summary) == PUBLIC_CHECKS + PRIVATE_CHECKS
    assert summary.scope is Scope.PRIVATE
    for check_id in PRIVATE_CHECKS:
        check = summary.check(check_id)
        assert (check.state, check.value) == (HealthState.INSUFFICIENT_EVIDENCE, None), check_id
    assert summary.overall_state is HealthState.INSUFFICIENT_EVIDENCE


def _inputs(tmp_path) -> collect.PrivateInputs:
    store = tmp_path / "store"
    store.mkdir()
    adapter = SyntheticFundAdapter(retrieved_at=pd.Timestamp("2025-01-02T12:00:00", tz="UTC"))
    ingest_batch(
        store,
        adapter.fetch(
            list(SYNTHETIC_FUND_IDS),
            pd.Timestamp("2024-10-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        ),
    )
    health_inputs = tmp_path / "health_inputs"
    for document in (
        fixtures.training_health(),
        fixtures.backtest_summary(),
        fixtures.registry_export(),
    ):
        fixtures.write_input(health_inputs, document)
    runs = tmp_path / "ci_runs.json"
    runs.write_text(
        json.dumps(
            [
                {
                    "databaseId": 7,
                    "status": "completed",
                    "conclusion": "success",
                    "headSha": SHA,
                    "headBranch": "main",
                    "workflowName": "CI",
                    "createdAt": "2026-10-14T12:00:00Z",
                    "url": "https://github.com/o/r/actions/runs/7",
                    "event": "push",
                }
            ]
        ),
        encoding="utf-8",
    )
    return collect.PrivateInputs(
        ci_runs=runs,
        fund_store=store,
        health_inputs=health_inputs,
        registry=fixtures.build_registry(tmp_path / "ml-fixture"),
    )


def test_a_local_run_reads_every_private_input(root, tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    runs_written = datetime.fromtimestamp(inputs.ci_runs.stat().st_mtime, tz=UTC)
    summary, latest = collect.run_local(
        environment(root, now=runs_written + timedelta(hours=1)),
        tmp_path / "private",
        verify=collect.VerifyOutput(VERIFY_OK, runs_written),
        inputs=inputs,
    )
    states = {check.check_id: (check.state, check.reason) for check in summary.checks}
    unapproved = (HealthState.DEGRADED, "unapproved_transition")
    economics = ["training.health", *(c for c in states if c.startswith("portfolio."))]
    assert {states[check_id] for check_id in economics} == {unapproved}
    synthetic = [c for c in states if c.startswith("registry.") or c in forecast.CHECKS]
    assert {states[check_id] for check_id in synthetic} == {
        (HealthState.INSUFFICIENT_EVIDENCE, "synthetic_input")  # private scope: never healthy
    }
    others = {
        state for check_id, (state, _) in states.items() if check_id not in economics + synthetic
    }
    assert others == {HealthState.HEALTHY, HealthState.INSUFFICIENT_EVIDENCE}
    assert summary.overall_state is HealthState.DEGRADED  # shown, never acted on
    for check in summary.checks:  # every evidence file pointer exists
        if check.evidence and "/evidence/" in check.evidence:
            assert Path(check.evidence).is_file(), check.check_id
    states = {check_id: state for check_id, (state, _) in states.items()}
    assert states["data.vendor_discrepancy"] is HealthState.INSUFFICIENT_EVIDENCE  # one source
    assert summary.check("data.coverage").value == 1.0
    assert summary.check("ci.main_runs").value == "success"
    assert summary.check("data.freshness").evidence == str(inputs.fund_store)
    history = list((tmp_path / "private" / "history").glob("*.json"))
    assert len(history) == 1 and read_summary(history[0]) == read_summary(latest)


def test_a_second_local_run_compares_with_the_first(root, tmp_path):
    private = tmp_path / "private"
    first, _ = collect.run_local(
        environment(root),
        private,
        verify=collect.VerifyOutput(VERIFY_OK, NOW),
        inputs=collect.PrivateInputs(),
    )
    fewer = VERIFY_OK.replace("1958 passed", "1950 passed")
    later = NOW + timedelta(days=1)
    second, _ = collect.run_local(
        environment(root, now=later),
        private,
        verify=collect.VerifyOutput(fewer, later),
        inputs=collect.PrivateInputs(),
    )
    assert first.alerts == ()
    deteriorated = [(a.kind, a.check_id) for a in second.alerts]
    assert deteriorated == [(AlertKind.DETERIORATION, "ci.tests")]
    assert "from 1958 to 1950" in second.alerts[0].message
    assert len(list((private / "history").glob("*.json"))) == 2


def test_an_unreadable_fund_store_is_unavailable(root, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / ".DS_Store").write_text("x")
    summary, _ = collect.run_local(
        environment(root),
        tmp_path / "private",
        verify=None,
        inputs=collect.PrivateInputs(fund_store=store),
    )
    check = summary.check("data.coverage")
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "store_refused")
    missing, _ = collect.run_local(
        environment(root, now=NOW + timedelta(hours=1)),
        tmp_path / "private",
        verify=None,
        inputs=collect.PrivateInputs(fund_store=tmp_path / "absent"),
    )
    assert missing.check("data.coverage").reason == "store_missing"


# --- scripts/health_check.py -----------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("health_check", REPO / "scripts/health_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_exit_codes(root, tmp_path, capsys):
    script = _script()
    ok = script.main(["ci", "--output-dir", str(tmp_path / "a")], environment(root))
    assert ok == 0
    assert "overall insufficient_evidence" in capsys.readouterr().out

    broken = environment(root, FakeMachine(pip_code=1))
    assert script.main(["ci", "--output-dir", str(tmp_path / "b")], broken) == 1

    actions = environment(root, env={"GITHUB_ACTIONS": "true"})
    assert script.main(["local", "--output-dir", str(tmp_path / "c")], actions) == 2
    assert "refused" in capsys.readouterr().err


def test_the_script_reads_a_recorded_verify_log_dated_by_its_stamp(root, tmp_path):
    log = tmp_path / "verify.log"
    log.write_text(f"{collect.VERIFY_STAMP}2026-10-14T17:00:00.000000Z\n{VERIFY_OK}", "utf-8")
    script = _script()
    out = tmp_path / "out"
    assert (
        script.main(["ci", "--output-dir", str(out), "--verify-log", str(log)], environment(root))
        == 0
    )
    summary = read_summary(out / "health_summary.json")
    assert summary.check("ci.tests").value == 1958
    assert summary.check("ci.tests").measured_at == datetime(2026, 10, 14, 17, tzinfo=UTC)
    assert (out / summary.check("ci.verify").evidence).is_file()  # the pointer exists
    missing = ["ci", "--output-dir", str(out), "--verify-log", str(tmp_path / "absent.log")]
    assert script.main(missing, environment(root)) == 2


def test_a_copied_log_without_a_stamp_is_not_dated_by_its_file_time(root, tmp_path):
    log = tmp_path / "verify.log"
    log.write_text(VERIFY_OK, encoding="utf-8")  # fresh mtime, no recorded run time
    assert collect.read_verify_log(log).measured_at is None
    summary, _ = collect.run_ci(
        environment(root), tmp_path / "out", verify=collect.read_verify_log(log)
    )
    assert summary.check("ci.tests").reason == "verify_time_unknown"


def test_an_empty_ci_run_list_is_unavailable_not_missing(root, tmp_path):
    runs = tmp_path / "ci_runs.json"
    runs.write_text("", encoding="utf-8")  # what `gh ... > file` leaves when gh fails
    summary, _ = collect.run_local(
        environment(root, now=datetime.now(UTC)),
        tmp_path / "private",
        verify=None,
        inputs=collect.PrivateInputs(ci_runs=runs),
    )
    check = summary.check("ci.main_runs")
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "unrecognized_format")


# --- the public artifact holds nothing private -------------------------------------------


def test_ci_evidence_and_summary_show_local_paths_redacted(root, tmp_path):
    verify = VERIFY_OK.replace(
        "verify: repo /repo", f"verify: repo {root}\nverify: interpreter {HOME}/x/bin/python"
    )
    env = collect.Environment(
        root=root,
        env={},
        now=lambda: NOW,
        runner=FakeMachine(verify=verify),
        installed={"pytest": "9.1.1"}.get,
        python=str(root / ".venv" / "bin" / "python"),
        home=HOME,
    )
    output = tmp_path / "out"
    summary, path = collect.run_ci(env, output, verify=collect.run_verify(env, public=True))
    log = (output / "evidence" / "verify.log").read_text(encoding="utf-8")
    assert "verify: repo ." in log and "interpreter ~/x/bin/python" in log
    assert str(root) not in log and str(HOME) not in log
    text = path.read_text(encoding="utf-8")
    assert "interpreter .venv/bin/python" in summary.check("deps.pins").detail
    assert str(root) not in text and str(HOME) not in text


def test_a_ci_summary_that_would_publish_an_email_is_refused(root, tmp_path):
    class Leaky(FakeMachine):
        def __call__(self, argv, **kwargs):
            if list(argv[1:4]) == ["-m", "pip", "check"]:
                out = "a 1 needs b; reported by someone@example.com\n"
                return subprocess.CompletedProcess(argv, 1, out, "")
            return super().__call__(argv, **kwargs)

    with pytest.raises(collect.PublicSummaryRefused, match="an email address"):
        collect.run_ci(environment(root, Leaky()), tmp_path / "out", verify=None)
    assert not (tmp_path / "out").exists()  # nothing at all for the workflow to upload
    script = _script()
    assert script.main(["ci", "--output-dir", str(tmp_path / "b")], environment(root, Leaky())) == 2


def test_a_refused_verify_log_leaves_no_evidence_to_upload(root, tmp_path):
    leaky = VERIFY_OK.replace("verify: repo /repo", "verify: run by someone@example.com")
    env = environment(root, FakeMachine(verify=leaky))
    output = tmp_path / "upload"
    with pytest.raises(collect.PublicSummaryRefused, match="evidence/verify.log"):
        collect.run_ci(env, output, verify=collect.run_verify(env, public=True))
    assert not output.exists() or not any(output.rglob("*"))


def test_a_local_run_keeps_its_private_paths(root, tmp_path):
    summary, _ = collect.run_local(
        environment(root), tmp_path / "private", verify=None, inputs=collect.PrivateInputs()
    )
    assert "/repo/.venv/bin/python" in summary.check("deps.pins").detail


def test_a_point_only_export_is_listed_as_not_applicable(root, tmp_path):
    export = fixtures.registry_export()
    export["forecast_evaluation"] = export["forecast_evaluation"][1:]
    fixtures.write_input(tmp_path / "inputs", export)
    summary, latest = collect.run_local(
        environment(root),
        tmp_path / "private",
        verify=None,
        inputs=collect.PrivateInputs(health_inputs=tmp_path / "inputs"),
    )
    assert summary.check("forecast.coverage") is None
    assert [item.check_id for item in summary.not_applicable] == [
        "forecast.coverage",
        "forecast.calibration",
    ]
    assert read_summary(latest).not_applicable == summary.not_applicable


def test_the_local_script_wires_the_private_inputs(root, tmp_path):
    script = _script()
    out = tmp_path / "out"
    (tmp_path / "kept.txt").write_text("0" * 64 + "\n", encoding="utf-8")  # the genesis head
    arguments = [
        "local",
        "--output-dir",
        str(out),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--ledger-head-backup",
        str(tmp_path / "kept.txt"),
        "--anchors",
        str(tmp_path / "anchors.csv"),
        "--prospective-store",
        str(tmp_path / "prospective"),
        "--rl-runs",
        str(tmp_path / "rl"),
        "--ml-root",
        str(tmp_path / "ml"),
        "--health-inputs",
        str(tmp_path / "inputs"),
        "--registry",
        str(tmp_path / "registry"),
    ]
    assert script.main(arguments, environment(root)) == 0
    summary = read_summary(out / "latest.json")
    reasons = {check.check_id: check.reason for check in summary.checks}
    assert reasons["forecast.ledger"] == "input_missing"
    assert reasons["forecast.ledger_backup"] == "input_missing"  # genesis head, no ledger yet
    assert reasons["data.anchors"] == "anchors_missing"
    assert reasons["data.store_batches"] == "store_missing"
    assert reasons["data.window_coverage"] == "store_missing"
    assert reasons["training.ml_runs"] == "input_missing"
    assert reasons["portfolio.violations"] == "not_reported"
    assert reasons["ci.daily_health"] == "not_collected"
    assert summary.overall_state is HealthState.INSUFFICIENT_EVIDENCE


def _stock_lab_file(inputs: Path, run_id: str) -> None:
    document = fixtures.stock_lab_monitor()
    document["run_id"] = run_id
    fixtures.write_input(inputs, document)


def test_the_local_script_reads_the_stock_labs_shared_inputs_once_its_job_writes_there(
    root, tmp_path
):
    # The evening runs from a runner clone outside ~/Desktop and writes its monitor files to
    # ~/stock-agent-private/health_inputs, while Model Health runs from another checkout.
    script = _script()
    home = tmp_path / "home"
    checkout = tmp_path / "inputs"  # this checkout's reports/health_inputs: older files
    _stock_lab_file(checkout, "job-from-the-desktop-checkout")
    machine = dataclasses.replace(environment(root), home=home)
    runs = iter(range(10))

    def evidence(*extra: str) -> dict[str, str | None]:
        out = tmp_path / f"out-{next(runs)}"
        arguments = ["local", "--output-dir", str(out), "--health-inputs", str(checkout)]
        assert script.main([*arguments, *extra], machine) in (0, 1)
        summary = read_summary(out / "latest.json")
        return {check.check_id: check.evidence for check in summary.checks}

    before = evidence()
    assert before["forecast.stock_lab_job"] == str(checkout / "stock_lab_monitor")
    shared = home / "stock-agent-private" / "health_inputs"
    (shared / "stock_lab_monitor").mkdir(parents=True)  # empty: still the checkout's files
    assert evidence()["forecast.stock_lab_job"] == str(checkout / "stock_lab_monitor")
    _stock_lab_file(shared, "job-from-the-runner")
    after = evidence()
    for check_id in health_stock_lab.TITLES:
        assert after[check_id] == str(shared / "stock_lab_monitor"), check_id
    # The other producers' inputs are still read from --health-inputs.
    assert after["training.health"] == before["training.health"]
    other = tmp_path / "elsewhere"
    given = evidence("--stock-lab-inputs", str(other))
    assert given["forecast.stock_lab_job"] == str(other / "stock_lab_monitor")


def test_only_the_stock_lab_monitor_files_move_to_their_own_root(tmp_path):
    inputs = collect.PrivateInputs(health_inputs=tmp_path / "a", stock_lab_inputs=tmp_path / "b")
    assert inputs.artifact_directory("stock_lab_monitor") == tmp_path / "b" / "stock_lab_monitor"
    assert inputs.artifact_directory("rl_training_health") == tmp_path / "a" / "rl_training_health"
    unset = collect.PrivateInputs(health_inputs=tmp_path / "a")
    assert unset.artifact_directory("stock_lab_monitor") == tmp_path / "a" / "stock_lab_monitor"
    assert collect.PrivateInputs().artifact_directory("stock_lab_monitor") is None


def test_the_script_refuses_both_github_and_a_run_list(root, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        _script().main(
            ["local", "--github", "--ci-runs", str(tmp_path / "runs.json")], environment(root)
        )
    assert stopped.value.code == 2


# --- missing inputs and the public boundary ------------------------------------------


def test_a_local_run_whose_private_inputs_are_all_missing_is_never_healthy(root, tmp_path):
    gone = tmp_path / "gone"  # every input is named, and none of them exists
    inputs = collect.PrivateInputs(
        ci_runs=gone / "ci_runs.json",
        fund_store=gone / "fund_store",
        health_inputs=gone / "health_inputs",
        registry=gone / "registry",
        ml_artifacts=gone / "artifacts",
        ml_root=gone / "ml",
        rl_runs=gone / "rl",
        ledger=gone / "ledger.jsonl",
        ledger_head_backup=gone / "ledger_head.txt",
    )
    summary, _ = collect.run_local(
        environment(root), tmp_path / "private", verify=None, inputs=inputs
    )
    assert ids(summary) == PUBLIC_CHECKS + PRIVATE_CHECKS
    for check_id in PRIVATE_CHECKS:
        check = summary.check(check_id)
        # Missing is "no evidence" or "unavailable": never healthy, stale or failed.
        assert check.condition in {Condition.NO_EVIDENCE, Condition.UNAVAILABLE}, check_id
        assert check.value is None or check.state is not HealthState.HEALTHY, check_id
    assert summary.overall_state is not HealthState.HEALTHY


# Where private data lives: the local monitor's defaults and the jobs' private stores.
PRIVATE_DEFAULTS = (
    "data/raw",
    "reports/health_inputs",
    "reports/ml",
    "reports/health",
    "models/rl",
)
AUDITED = {"open", "os.listdir", "os.scandir", "sqlite3.connect", "glob.glob", "os.chdir"}
_opened: list[tuple[str, str]] | None = None


def _audit(event: str, args: tuple) -> None:
    if _opened is not None and event in AUDITED and args:
        target = args[0]
        if isinstance(target, str | bytes | os.PathLike):
            _opened.append((event, os.fsdecode(target)))


sys.addaudithook(_audit)  # records only while `_opened` is a list (the hook cannot be removed)


def test_the_ci_run_never_reads_a_private_path(root, tmp_path, monkeypatch):
    global _opened
    home = tmp_path / "home"
    private = home / "stock-agent-private"
    planted = [
        private / "health" / "latest.json",
        private / "stock_lab" / "stock_lab.sqlite3",
        private / "stock_lab" / "stock_lab_ledger_head",
        private / "health_inputs" / "stock_lab_monitor" / "planted.json",
        private / "ledger_head.txt",
        private / "fund_store" / "batches.json",
        private / "prospective_store" / "batches.json",
        private / "anchors.csv",
        *(root / directory / "planted.json" for directory in PRIVATE_DEFAULTS),
    ]
    for path in planted:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"private": true}\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_HOME", str(private / "stock_lab"))
    monkeypatch.setenv("STOCK_AGENT_ANCHORS", str(private / "anchors.csv"))
    monkeypatch.setenv("STOCK_AGENT_PROSPECTIVE_LEDGER", str(root / "data/raw/planted.json"))
    env = collect.Environment(
        root=root,
        env=dict(os.environ),
        now=lambda: NOW,
        runner=FakeMachine(),
        installed={"pytest": "9.1.1"}.get,
        python="/repo/.venv/bin/python",
        home=home,
    )
    output = tmp_path / "out"
    script = _script()  # loaded before the hook records: its own files are the repository's
    _opened = []
    try:
        summary, _ = collect.run_ci(env, output, verify=collect.run_verify(env, public=True))
        # The script's ci command too: it must never resolve a private input either.
        assert script.main(["ci", "--output-dir", str(tmp_path / "script-out")], env) in (0, 1)
    finally:
        touched, _opened = _opened, None
    assert summary.scope is Scope.SYNTHETIC and ids(summary) == PUBLIC_CHECKS
    forbidden = [
        private,
        Path.home() / "stock-agent-private",  # the real defaults, whatever HOME says
        *(root / directory for directory in PRIVATE_DEFAULTS),
    ]
    assert touched  # the hook saw the run's own reads and writes
    read = [
        (event, path)
        for event, path in touched
        if any(Path(path).resolve().is_relative_to(place.resolve()) for place in forbidden)
    ]
    assert read == []
    assert all(path.read_text(encoding="utf-8") == '{"private": true}\n' for path in planted)


def test_the_ci_run_takes_no_private_input():
    import inspect

    assert list(inspect.signature(collect.run_ci).parameters) == [
        "environment",
        "output_dir",
        "verify",
    ]


def test_the_local_run_passes_the_readiness_report_only_with_a_store(
    prospective_frozen, prospective_store_root, tmp_path
):
    from forecast_prospective_helpers import register

    from stock_agent.calendars import nyse_rules
    from stock_agent.data.prospective.inference import load_inference_window
    from stock_agent.forecasting.prospective import ledger as job_ledger
    from stock_agent.health.ledger import RulesCalendar
    from stock_agent.health.prospective import ProspectiveSources

    calendar = RulesCalendar(nyse_rules.sessions, nyse_rules.CALENDAR_ID)

    def sources(store, lookback):
        return ProspectiveSources(
            store_root=store,
            assets=("not", "the", "record's"),  # the report uses the registration's funds
            window_start=pd.Timestamp("2026-07-16", tz="UTC"),
            h_end=pd.Timestamp("2026-10-14", tz="UTC"),
            calendar=calendar,
            load_window=load_inference_window,
            lookback=lookback,
        )

    assert collect._readiness(collect.PrivateInputs()) is None
    without_store = collect.PrivateInputs(calendar=calendar, prospective=sources(None, 63))
    assert collect._readiness(without_store) is None
    without_lookback = collect.PrivateInputs(
        calendar=calendar, prospective=sources(prospective_store_root, None)
    )
    assert collect._readiness(without_lookback) is None
    with_store = collect.PrivateInputs(
        calendar=calendar, prospective=sources(prospective_store_root, 63)
    )
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    rows = collect._readiness(with_store)(
        job_ledger.read_ledger(path), pd.Timestamp("2026-10-16T22:00:00Z")
    )
    assert [(row.day, row.status) for row in rows] == [
        ("2026-10-14", "not_recorded"),
        ("2026-10-15", "not_recorded"),
        ("2026-10-16", "pending"),
    ]


def test_the_local_run_hands_the_readiness_report_to_the_ledger_checks(root, tmp_path, monkeypatch):
    seen = {}
    real = collect.ledger.ledger_checks

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    sentinel = object()
    monkeypatch.setattr(collect.ledger, "ledger_checks", spy)
    monkeypatch.setattr(collect, "_readiness", lambda inputs: sentinel)
    collect.run_local(
        environment(root), tmp_path / "private", verify=None, inputs=collect.PrivateInputs()
    )
    assert seen["readiness"] is sentinel
