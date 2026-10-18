"""
scripts/forecast_prospective.py: the report, the registration record and
entry, and forecasts and scores through the real loader and calendar. Git is
replaced by a fake local repository (`_offline`); the real Git helpers are
tested on a temporary repository in test_prospective_provenance.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import UTC, datetime

import pandas as pd
import pytest
from forecast_prospective_helpers import (
    AS_OF,
    CODE_TREES,
    CREATED,
    DOCUMENT_COMMIT,
    RECORD_PATH,
    REGISTERED_RECORDED,
    REGISTRATION_COMMIT,
    REPO,
    RUN_CODE,
    RUN_ENVIRONMENT,
    github_pull_request,
    register,
    registration,
)

from stock_agent.calendars import nyse_rules
from stock_agent.forecasting.prospective import forecast, ledger, registration_record, zero_vol

SCRIPT = REPO / "scripts" / "forecast_prospective.py"
MERGED = "2026-10-14T14:00:00Z"  # GitHub's merge time in the fixtures (REGISTRATION_TIME)


def _script():
    spec = importlib.util.spec_from_file_location("forecast_prospective", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline(
    monkeypatch,
    script,
    *,
    running=None,
    files=None,
    trees=None,
    committed_at=MERGED,
    ancestor=True,
    on_main=True,
    pull_request=None,
    parent=True,
):
    """
    A fake local repository and GitHub: `files` maps (commit, path) to
    committed bytes, `trees` maps a commit to its registered paths' tree ids
    (default: the fixture's everywhere), the merge commit's time, whether the
    registration document is in its history (`ancestor`) and whether it is in
    origin/main (`on_main`), and the pull request GitHub returns.
    """

    def committed(commit, path):
        if (commit, path) not in files:
            raise forecast.ForecastRefused("record_not_committed", f"{path} not in {commit}")
        return files[(commit, path)]

    monkeypatch.setattr(script, "_code", lambda: RUN_CODE)
    monkeypatch.setattr(
        script, "_environment", lambda: RUN_ENVIRONMENT if running is None else running
    )
    monkeypatch.setattr(script, "_code_trees", lambda commit: (trees or {}).get(commit, CODE_TREES))
    monkeypatch.setattr(script, "_committed_file", committed)
    monkeypatch.setattr(script, "_commit_time", lambda commit: pd.Timestamp(committed_at))
    monkeypatch.setattr(
        script,
        "_is_ancestor",
        lambda older, newer: on_main if newer == "origin/main" else ancestor,
    )
    fetched = []

    def fetch(number):
        fetched.append(number)
        return github_pull_request() if pull_request is None else pull_request

    monkeypatch.setattr(script, "_fetch_pull_request", fetch)
    monkeypatch.setattr(script, "_has_commit", lambda revision: parent)
    return fetched


def _fixture_files(models):
    text = json.dumps(registration(models), indent=2, sort_keys=True) + "\n"
    return {(REGISTRATION_COMMIT, RECORD_PATH): text.encode()}


def test_the_report_on_an_empty_ledger_says_so(tmp_path, capsys):
    assert _script().main(["--ledger", str(tmp_path / "ledger.jsonl"), "report"]) == 0
    out = capsys.readouterr().out
    assert "0 forecasts" in out and "no scores yet" in out and "not registered" in out


def test_the_report_names_the_registration(prospective_frozen, tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    assert _script().main(["--ledger", str(path), "report"]) == 0
    out = capsys.readouterr().out
    assert f"registered 2026-10-14T14:00:00+00:00 at {REGISTRATION_COMMIT}" in out


def test_the_report_refuses_a_ledger_without_the_kept_head(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(ledger.LedgerError, match="not in the ledger"):
        _script().main(["--ledger", str(path), "report", "--expect-head", "a" * 64])


def _forecast_argv(prospective_frozen, store, path, as_of=AS_OF):
    return [
        "--ledger",
        str(path),
        "forecast",
        "--bundle",
        str(prospective_frozen.result.bundle.path),
        "--store-root",
        str(store),
        "--forecast-close",
        as_of,
    ]


def test_the_script_forecasts_through_the_real_loader_and_calendar(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    _offline(monkeypatch, script, files=_fixture_files(prospective_frozen.models()))
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    assert script.main(_forecast_argv(prospective_frozen, prospective_store_root, path)) == 0
    (_, entry) = ledger.read_ledger(path)
    assert entry.kind == "forecast"
    assert entry.payload["data"]["calendar"] == nyse_rules.CALENDAR_ID
    assert entry.payload["data"]["retrieved_by"] == CREATED.isoformat()
    assert script.exchange_calendar.calendar_id == nyse_rules.CALENDAR_ID


def test_the_defaults_are_private_paths():
    script = _script()
    assert script.DEFAULT_LEDGER.relative_to(REPO).parts[:2] == ("data", "raw")
    assert script.DEFAULT_OUTPUT.relative_to(REPO).parts[0] == "reports"
    assert script.DEFAULT_CONFIG.name == "etf_prospective_v1.yaml"


def test_the_script_scores_under_the_rule_in_force_only(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    # The job's CLI cannot select the research option B2: its scores record the rule in
    # force, and the forecast records the bundle's training rule (docs/research/zero_vol_floor.md).
    script = _script()
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    _offline(monkeypatch, script, files=_fixture_files(prospective_frozen.models()))
    assert script.main(_forecast_argv(prospective_frozen, prospective_store_root, path)) == 0
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-23T23:00:00Z"))
    score_argv = ["--store-root", str(prospective_store_root), "--score-close", "2026-10-23"]
    assert script.main(["--ledger", str(path), "score", *score_argv]) == 0
    _, forecast_entry, score_entry = ledger.read_ledger(path)
    assert forecast_entry.payload["bundle"]["zero_volatility_rule"] == zero_vol.CURRENT
    volatility = [
        s for s in score_entry.payload["scores"] if s["target"] == "log_realized_volatility"
    ]
    assert volatility and all(s["zero_vol_rule"] == zero_vol.CURRENT for s in volatility)
    assert all(s["floored"] is False and s["sigma_floor"] is None for s in volatility)


@pytest.mark.parametrize("command", ["forecast", "score"])
def test_the_job_refuses_a_ledger_record_other_than_the_committed_one(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch, command
):
    # A consistent rewrite of the whole ledger could change its record; the committed file
    # in the registration commit cannot be rewritten that way.
    script = _script()
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    other = registration(prospective_frozen.models())
    other["code"]["trees"] = {**CODE_TREES, "src": "f" * 40}
    files = {(REGISTRATION_COMMIT, RECORD_PATH): json.dumps(other).encode()}
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    _offline(monkeypatch, script, files=files)
    if command == "forecast":
        argv = _forecast_argv(prospective_frozen, prospective_store_root, path)
    else:
        argv = ["--ledger", str(path), "score", "--store-root", str(prospective_store_root)]
        argv += ["--score-close", "2026-10-23"]
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "registration_tampered"
    assert [e.kind for e in ledger.read_ledger(path)] == ["registration"]


def test_the_forecast_script_refuses_before_registration(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    _offline(monkeypatch, script, files={})
    argv = _forecast_argv(prospective_frozen, prospective_store_root, tmp_path / "ledger.jsonl")
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "not_registered"


# --- the registration record and entry ------------------------------------------------------


def _repository(tmp_path, monkeypatch, script):
    root = tmp_path / "repo"
    monkeypatch.setattr(script, "PROJECT_ROOT", root)
    monkeypatch.setattr(script, "_code_trees", lambda commit: CODE_TREES)
    return root


def test_the_script_writes_the_record_file_a_registration_pull_request_adds(
    prospective_frozen, tmp_path, monkeypatch, capsys
):
    script = _script()
    root = _repository(tmp_path, monkeypatch, script)
    bundle = prospective_frozen.result.bundle
    argv = ["registration-record", "--bundle", str(bundle.path)]
    argv += ["--registration-doc-commit", DOCUMENT_COMMIT, "--registrations", str(root / "r")]
    assert script.main(argv) == 0
    path = root / "r" / f"{bundle.bundle_id}.json"
    written = registration_record.read_record(path)
    assert written == registration(prospective_frozen.models())
    out = capsys.readouterr().out
    assert registration_record.record_sha256(written) in out
    assert "nothing is registered" in out


def test_a_record_file_is_never_overwritten(prospective_frozen, tmp_path, monkeypatch):
    script = _script()
    root = _repository(tmp_path, monkeypatch, script)
    argv = ["registration-record", "--bundle", str(prospective_frozen.result.bundle.path)]
    argv += ["--registration-doc-commit", DOCUMENT_COMMIT, "--registrations", str(root / "r")]
    assert script.main(argv) == 0
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "record_exists"


def _record_file(prospective_frozen, root):
    path = root / "docs" / "research" / "registrations" / "fixture.json"
    return registration_record.write_record(path, registration(prospective_frozen.models()))


def _register_argv(
    prospective_frozen,
    ledger_path,
    record_path,
    registered_at=MERGED,
    proof=("--pull-request", "120"),
):
    return [
        "--ledger",
        str(ledger_path),
        "register",
        "--bundle",
        str(prospective_frozen.result.bundle.path),
        "--record",
        str(record_path),
        "--registration-commit",
        REGISTRATION_COMMIT,
        "--registered-at",
        registered_at,
        *proof,
    ]


@pytest.fixture
def registering(prospective_frozen, tmp_path, monkeypatch):
    """The script, a repository holding the fixture record, and a ledger path."""

    script = _script()
    root = _repository(tmp_path, monkeypatch, script)
    record_path = _record_file(prospective_frozen, root)
    monkeypatch.setattr(script, "_now", lambda: REGISTERED_RECORDED)
    files = {(REGISTRATION_COMMIT, RECORD_PATH): record_path.read_bytes()}
    return script, record_path, files, tmp_path / "private" / "ledger.jsonl"


def test_the_script_registers_a_committed_record_as_the_ledgers_first_entry(
    prospective_frozen, registering, monkeypatch, capsys
):
    script, record_path, files, ledger_path = registering
    _offline(monkeypatch, script, files=files)
    assert script.main(_register_argv(prospective_frozen, ledger_path, record_path)) == 0
    (entry,) = ledger.read_ledger(ledger_path)
    assert entry.kind == "registration"
    assert entry.payload["record_path"] == RECORD_PATH
    assert entry.payload["registered_at"] == "2026-10-14T14:00:00+00:00"
    out = capsys.readouterr().out
    assert entry.entry_sha256 in out and "outside the ledger" in out


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ({"files": {}}, "record_not_committed"),
        ({"files": {(REGISTRATION_COMMIT, RECORD_PATH): b"{}\n"}}, "record_not_committed"),
        # The record's tree ids are not those of the bundle's commit.
        ({"trees": {"0" * 40: {**CODE_TREES, "src": "f" * 40}}}, "registration_mismatch"),
        # The merge commit changed a registered code path after the freeze.
        (
            {"trees": {REGISTRATION_COMMIT: {**CODE_TREES, "src": "f" * 40}}},
            "registration_mismatch",
        ),
        # The typed merge time is earlier than the merge commit itself.
        ({"committed_at": "2026-10-14T14:00:01Z"}, "registered_before_commit"),
        # The registration document's commit is not in the registration commit's history.
        ({"ancestor": False}, "registration_invalid"),
        # The merge commit is not in the local origin/main history.
        ({"on_main": False}, "registration_unverified"),
        # GitHub reports another merge time, merge commit or base.
        (
            {"pull_request": github_pull_request(merged_at="2026-10-14T14:00:01Z")},
            "registration_unverified",
        ),
        (
            {"pull_request": github_pull_request(merge_commit_sha="f" * 40)},
            "registration_unverified",
        ),
        ({"pull_request": github_pull_request(merged=False)}, "registration_unverified"),
        # GitHub answers about another pull request than the one asked for.
        ({"pull_request": github_pull_request(number=121)}, "registration_unverified"),
        # The merge commit's parent is not in the local repository (a shallow clone): it
        # cannot be shown that this merge added the record.
        ({"parent": False}, "registration_unverified"),
    ],
)
def test_the_script_refuses_a_registration_that_does_not_match_the_repository(
    prospective_frozen, registering, monkeypatch, setup, reason
):
    script, record_path, files, ledger_path = registering
    _offline(monkeypatch, script, **{"files": files, **setup})
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(_register_argv(prospective_frozen, ledger_path, record_path))
    assert refused.value.reason == reason
    assert not ledger_path.exists()


def test_the_script_refuses_a_record_outside_the_repository(
    prospective_frozen, tmp_path, monkeypatch
):
    script = _script()
    _repository(tmp_path, monkeypatch, script)
    elsewhere = registration_record.write_record(
        tmp_path / "elsewhere.json", registration(prospective_frozen.models())
    )
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(_register_argv(prospective_frozen, tmp_path / "l.jsonl", elsewhere))
    assert refused.value.reason == "registration_invalid"


@pytest.mark.parametrize("typed", ["2026-10-14 14:00", "yesterday afternoon"])
def test_the_script_takes_the_merge_time_with_its_timezone(
    prospective_frozen, registering, monkeypatch, typed
):
    script, record_path, files, ledger_path = registering
    _offline(monkeypatch, script, files=files)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(_register_argv(prospective_frozen, ledger_path, record_path, typed))
    assert refused.value.reason == "registration_invalid"


def test_the_script_offers_no_zero_volatility_rule_anywhere():
    script = _script()
    parser = script.build_parser()
    commands = next(a for a in parser._actions if a.dest == "command").choices
    assert set(commands) == {
        "freeze",
        "registration-record",
        "register",
        "forecast",
        "score",
        "report",
        "readiness",
        "preflight",
        "environment",
    }
    for name, command in commands.items():
        for action in command._actions:
            for option in action.option_strings:
                assert not any(word in option for word in ("rule", "zero", "b2", "floor")), (
                    name,
                    option,
                )


# --- the running code ---------------------------------------------------------------------


def _git_seams(monkeypatch, script, calls):
    monkeypatch.setattr(
        script, "code_provenance", lambda root: {"git_sha": "a" * 40, "git_dirty": False}
    )
    monkeypatch.setattr(
        script,
        "code_trees",
        lambda root, commit, paths: calls.append((commit, paths)) or CODE_TREES,
    )
    monkeypatch.setattr(script, "paths_dirty", lambda root, paths: False)


def test_the_job_reads_the_running_codes_tree_ids(monkeypatch):
    script = _script()
    calls = []
    _git_seams(monkeypatch, script, calls)
    code = script._code()
    assert code == {
        "git_sha": "a" * 40,
        "git_dirty": False,
        "trees": CODE_TREES,
        "trees_dirty": False,
    }
    assert calls == [("HEAD", registration_record.CODE_PATHS)]


def test_the_job_refuses_a_package_imported_from_another_checkout(tmp_path, monkeypatch):
    # A worktree whose interpreter imports another checkout's package would check one tree
    # and run another.
    script = _script()
    _git_seams(monkeypatch, script, [])
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path / "another_checkout")
    with pytest.raises(forecast.ForecastRefused) as refused:
        script._code()
    assert refused.value.reason == "registration_mismatch"


def test_a_git_failure_is_a_structured_refusal(monkeypatch):
    import subprocess

    script = _script()
    _git_seams(monkeypatch, script, [])

    def failing(root, commit, paths):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse"])

    monkeypatch.setattr(script, "code_trees", failing)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script._code()
    assert refused.value.reason == "registration_mismatch"


def test_freezing_refuses_a_package_imported_from_another_checkout(tmp_path, monkeypatch):
    # The bundle's code commit must describe the code that trained it.
    script = _script()
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path / "another_checkout")
    argv = ["freeze", "--research-store", str(tmp_path / "never_read")]
    argv += ["--output-root", str(tmp_path / "out")]
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "registration_mismatch"
    assert not (tmp_path / "out").exists()


def test_the_script_registers_with_githubs_pull_request_as_evidence(
    prospective_frozen, registering, monkeypatch
):
    script, record_path, files, ledger_path = registering
    fetched = _offline(monkeypatch, script, files=files)
    assert script.main(_register_argv(prospective_frozen, ledger_path, record_path)) == 0
    assert fetched == [120]
    (entry,) = ledger.read_ledger(ledger_path)
    kept = entry.payload["registration_evidence"]
    assert kept["source"] == "github_api"
    assert kept["fetched_at"] == REGISTERED_RECORDED.isoformat()
    assert kept["pull_request"]["merged_at"] == "2026-10-14T14:00:00Z"


def test_the_script_registers_with_the_owners_evidence_file(
    prospective_frozen, registering, monkeypatch, tmp_path
):
    script, record_path, files, ledger_path = registering
    fetched = _offline(monkeypatch, script, files=files)
    supplied = tmp_path / "pull_request_120.json"
    supplied.write_text(json.dumps(github_pull_request()), encoding="utf-8")
    proof = ("--evidence-file", str(supplied))
    argv = _register_argv(prospective_frozen, ledger_path, record_path, proof=proof)
    assert script.main(argv) == 0
    assert fetched == []  # nothing fetched
    (entry,) = ledger.read_ledger(ledger_path)
    assert entry.payload["registration_evidence"]["source"] == "owner_evidence_file"


def test_the_script_refuses_a_registration_time_without_evidence(
    prospective_frozen, registering, monkeypatch
):
    script, record_path, files, ledger_path = registering
    _offline(monkeypatch, script, files=files)
    argv = _register_argv(prospective_frozen, ledger_path, record_path, proof=())
    with pytest.raises(SystemExit):
        script.main(argv)
    assert not ledger_path.exists()


def test_the_pull_request_is_fetched_with_gh_and_the_owners_login(monkeypatch):
    # No token passes through the job: `gh` uses the owner's existing login.
    script = _script()
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(github_pull_request()), "")

    monkeypatch.setattr(script.subprocess, "run", run)
    assert script._fetch_pull_request(120)["merged_at"] == "2026-10-14T14:00:00Z"
    ((argv, kwargs),) = calls
    assert argv == [
        "gh",
        "api",
        "--hostname",
        "github.com",
        "repos/billvo9/stock-allocation-agent/pulls/120",
    ]
    assert "env" not in kwargs and kwargs["capture_output"] is True


@pytest.mark.parametrize(
    "outcome",
    [
        (1, "", "HTTP 401: Bad credentials (secret-ish)"),
        (0, "<html>not json, secret-ish</html>", ""),
        FileNotFoundError("gh"),
    ],
    ids=["gh_failed", "not_json", "gh_missing"],
)
def test_a_failed_fetch_is_a_refusal_that_shows_no_output(monkeypatch, capsys, outcome):
    script = _script()

    def run(argv, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return subprocess.CompletedProcess(argv, *outcome)

    monkeypatch.setattr(script.subprocess, "run", run)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script._fetch_pull_request(120)
    assert refused.value.reason == "registration_unverified"
    assert "secret-ish" not in str(refused.value)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


# --- launch readiness (read-only) ---------------------------------------------------------


def _config_file(prospective_frozen, tmp_path):
    from stock_agent.model_diagnostics.record import canonical_json

    path = tmp_path / "config.yaml"
    path.write_text(canonical_json(prospective_frozen.config.to_json()), encoding="utf-8")
    return path  # JSON is YAML: the fixture's registered variant


def test_the_readiness_report_lists_each_close_since_the_registration(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch, capsys
):
    script = _script()
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-16T22:00:00Z"))
    argv = ["--ledger", str(path), "readiness", "--store-root", str(prospective_store_root)]
    argv += ["--config", str(_config_file(prospective_frozen, tmp_path))]
    assert script.main(argv) == 0
    out = capsys.readouterr().out
    assert "close=2026-10-14 status=not_recorded reason=operator_missed" in out
    assert "close=2026-10-16 status=pending" in out
    assert "the session-1 rule is unchanged" in out
    assert "proposed ETF-R1 (not in force): session 1 would be 2026-10-14" in out
    assert "missed under ETF-R1: 2026-10-14, 2026-10-15" in out
    assert [e.kind for e in ledger.read_ledger(path)] == ["registration"]  # nothing written


def test_the_readiness_report_refuses_another_config_than_the_registered_one(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    argv = ["--ledger", str(path), "readiness", "--store-root", str(prospective_store_root)]
    with pytest.raises(forecast.ForecastRefused) as refused:  # the default, registered config
        script.main(argv)
    assert refused.value.reason == "registration_mismatch"


@pytest.mark.parametrize(
    ("now", "code"), [("2026-10-15T23:00:00Z", 0), ("2026-10-14T22:10:00Z", 1)]
)
def test_before_the_merge_the_report_says_whether_the_window_is_ready(
    prospective_store_root, tmp_path, monkeypatch, capsys, now, code
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp(now))
    argv = ["--ledger", str(tmp_path / "none.jsonl"), "readiness", "--pre-merge"]
    argv += ["--store-root", str(prospective_store_root)]
    assert script.main(argv) == code
    out = capsys.readouterr().out
    assert ("ready" if code == 0 else "NOT ready") in out


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_an_unusable_evidence_file_is_refused(
    prospective_frozen, registering, monkeypatch, tmp_path, content
):
    script, record_path, files, ledger_path = registering
    _offline(monkeypatch, script, files=files)
    supplied = tmp_path / "evidence.json"
    supplied.write_text(content, encoding="utf-8")
    argv = _register_argv(
        prospective_frozen, ledger_path, record_path, proof=("--evidence-file", str(supplied))
    )
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "registration_unverified"
    assert not ledger_path.exists()


def test_the_registration_pull_request_is_the_one_that_added_the_record(
    prospective_frozen, registering, monkeypatch
):
    # A later pull request whose merge commit also holds the record (a docs-only one, say)
    # must not stand in for the registration: the record must be new in the merge commit.
    script, record_path, files, ledger_path = registering
    earlier = {**files, (f"{REGISTRATION_COMMIT}^1", RECORD_PATH): record_path.read_bytes()}
    _offline(monkeypatch, script, files=earlier)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(_register_argv(prospective_frozen, ledger_path, record_path))
    assert refused.value.reason == "registration_unverified"
    assert not ledger_path.exists()


# --- the registration preflight (read-only) -------------------------------------------------


def _health_summary(path, generated):
    from stock_agent.health.context import CodeState
    from stock_agent.health.ledger import ledger_checks
    from stock_agent.health.model import Scope
    from stock_agent.health.summary import build_summary, write_summary

    checks = ledger_checks(None, backup=None, calendar=None, now=generated)
    summary = build_summary(
        checks,
        generated_at=generated,
        code=CodeState(None, None),
        scope=Scope.PRIVATE,
        run_context={"environment": "local"},
    )
    return write_summary(path, summary)


def _preflight_argv(prospective_frozen, prospective_store_root, tmp_path, **changes):
    kept = tmp_path / "private" / "ledger_head.txt"  # section 6: a 700 directory, two 600 files
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.parent.chmod(0o700)
    kept.write_text("0" * 64 + "\n", encoding="utf-8")
    kept.chmod(0o600)
    history = kept.parent / "ledger_head_history.log"
    history.write_text("", encoding="utf-8")
    history.chmod(0o600)
    record = registration_record.write_record(
        tmp_path / "record.json", registration(prospective_frozen.models())
    )
    health = _health_summary(
        tmp_path / "health" / "latest.json", datetime(2026, 10, 15, 11, tzinfo=UTC)
    )
    options = {
        "--bundle": prospective_frozen.result.bundle.path,
        "--research-store": prospective_frozen.store,
        "--store-root": prospective_store_root,
        "--record": record,
        "--health-summary": health,
        "--ledger-head-backup": kept,
        "--out": tmp_path / "preflight.json",
        **changes,
    }
    argv = ["--ledger", str(tmp_path / "ledger.jsonl"), "preflight"]
    for name, value in options.items():
        if value is not None:
            argv += [name, str(value)]
    return argv, options["--out"]


def test_the_preflight_reports_every_check_with_its_hash(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch, capsys
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-15T23:00:00Z"))
    _offline(monkeypatch, script, files={})
    argv, out = _preflight_argv(prospective_frozen, prospective_store_root, tmp_path)
    code = script.main(argv)
    document = json.loads(out.read_text(encoding="utf-8"))
    from stock_agent.forecasting.prospective import preflight

    assert preflight.verify_report(document) == document
    statuses = {f"{i['area']}.{i['check']}": i["status"] for i in document["items"]}
    assert statuses == {
        "data.window": "pass",
        "data.fingerprint": "pass",
        "software.code_identity": "pass",
        "software.record": "pass",
        "software.environment": "pass",
        "monitoring.model_health": "pass",
        # The rules calendar ends 2027-12-31: C24 is out of reach until it is extended.
        "monitoring.calendar": "fail",
        "backups.head_backup": "pass",
        "backups.second_medium": "owner_attestation_required",
        "operations.forecast_job_schedule": "not_built",
        "analysis.checkpoint_analysis": "not_built",
    }
    assert code == 1 and document["passed"] is False
    assert document["subject"] == {
        "bundle_id": prospective_frozen.result.bundle.bundle_id,
        "record_sha256": registration_record.record_sha256(
            registration(prospective_frozen.models())
        ),
    }
    printed = capsys.readouterr().out
    assert document["sha256"] in printed and "nothing is registered" in printed
    assert not (tmp_path / "ledger.jsonl").exists()  # read-only


def test_a_failing_preflight_names_each_failure(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-15T23:00:00Z"))
    _offline(monkeypatch, script, files={}, trees={"0" * 40: {**CODE_TREES, "src": "f" * 40}})
    argv, out = _preflight_argv(
        prospective_frozen,
        prospective_store_root,
        tmp_path,
        **{"--health-summary": tmp_path / "absent.json", "--ledger-head-backup": None},
    )
    assert script.main(argv) == 1
    document = json.loads(out.read_text(encoding="utf-8"))
    failed = {
        f"{i['area']}.{i['check']}": i["reason"] for i in document["items"] if i["status"] == "fail"
    }
    assert failed["software.code_identity"] == "code_differs_from_freeze"
    assert failed["monitoring.model_health"] == "health_unreadable"
    assert failed["backups.head_backup"] == "not_configured"
    assert document["passed"] is False


def test_a_preflight_report_is_never_overwritten(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-15T23:00:00Z"))
    _offline(monkeypatch, script, files={})
    argv, _ = _preflight_argv(prospective_frozen, prospective_store_root, tmp_path)
    script.main(argv)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(argv)
    assert refused.value.reason == "preflight_exists"


@pytest.mark.parametrize("failure", ["unreadable_summary", "leaking_research_store"])
def test_the_preflight_reports_input_failures_as_failed_checks(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch, failure
):
    from stock_agent.model_validation.audit import LeakageError

    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-15T23:00:00Z"))
    _offline(monkeypatch, script, files={})
    argv, out = _preflight_argv(prospective_frozen, prospective_store_root, tmp_path)
    if failure == "unreadable_summary":
        monkeypatch.setattr(
            "stock_agent.health.model.read_summary",
            lambda path: (_ for _ in ()).throw(PermissionError("denied")),
        )
        expected = ("monitoring.model_health", "health_unreadable")
    else:

        def leaking(store_root, config):
            raise LeakageError("a session after 2024-12-31")

        monkeypatch.setattr(script.training, "load_training_data", leaking)
        expected = ("data.fingerprint", "research_store_unreadable")
    assert script.main(argv) == 1
    document = json.loads(out.read_text(encoding="utf-8"))
    failed = {
        f"{i['area']}.{i['check']}": i["reason"] for i in document["items"] if i["status"] == "fail"
    }
    assert failed[expected[0]] == expected[1]


def test_the_job_reads_the_environment_of_its_own_modules(monkeypatch):
    script = _script()
    seen = []

    def manifest(root, modules, required=()):
        seen.append((root, tuple(modules), tuple(required)))
        return RUN_ENVIRONMENT

    monkeypatch.setattr(script, "environment_manifest", manifest)
    assert script._environment() == RUN_ENVIRONMENT
    ((root, modules, required),) = seen
    assert root == script.PROJECT_ROOT
    from stock_agent.forecasting.prospective import environment as job_environment

    assert tuple(required) == job_environment.KEY_DISTRIBUTIONS
    assert {
        "stock_agent.forecasting.prospective.forecast",
        "stock_agent.forecasting.prospective.freeze",
        "stock_agent.data.prospective.inference",
    } <= set(modules)


def test_an_unreadable_environment_is_a_refusal(monkeypatch):
    script = _script()

    def failing(root, modules, required=()):
        raise subprocess.CalledProcessError(1, ["python"])

    monkeypatch.setattr(script, "environment_manifest", failing)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script._environment()
    assert refused.value.reason == "environment_invalid"


def test_the_bundles_environment_prints_as_pip_constraints(prospective_frozen, capsys):
    script = _script()
    argv = ["environment", "--bundle", str(prospective_frozen.result.bundle.path)]
    assert script.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert "numpy==2.5.1" in out and "scikit-learn==1.9.1" in out
    assert out[0].startswith("# python 3.12.2 (cpython)")
    assert all(line.startswith("#") or "==" in line for line in out)


def _other_minor():
    import copy

    other = copy.deepcopy(RUN_ENVIRONMENT)
    other["python"] = {**other["python"], "minor": "3.13", "version": "3.13.0"}
    return other


def test_the_forecast_script_refuses_another_environment(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    _offline(
        monkeypatch,
        script,
        files=_fixture_files(prospective_frozen.models()),
        running=_other_minor(),
    )
    path = tmp_path / "ledger.jsonl"
    register(prospective_frozen.models(), path)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(_forecast_argv(prospective_frozen, prospective_store_root, path))
    assert refused.value.reason == "environment_mismatch"


def test_the_preflight_fails_another_environment(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "_now", lambda: pd.Timestamp("2026-10-15T23:00:00Z"))
    _offline(monkeypatch, script, files={}, running=_other_minor())
    argv, out = _preflight_argv(prospective_frozen, prospective_store_root, tmp_path)
    assert script.main(argv) == 1
    document = json.loads(out.read_text(encoding="utf-8"))
    item = next(i for i in document["items"] if i["check"] == "environment")
    assert (item["status"], item["reason"]) == ("fail", "environment_mismatch")


@pytest.mark.parametrize(
    "error", [ImportError("no such distribution"), KeyError("Name")], ids=["missing", "metadata"]
)
def test_an_unreadable_environment_of_any_kind_is_a_refusal(monkeypatch, error):
    script = _script()

    def failing(root, modules, required=()):
        raise error

    monkeypatch.setattr(script, "environment_manifest", failing)
    with pytest.raises(forecast.ForecastRefused) as refused:
        script._environment()
    assert refused.value.reason == "environment_invalid"


def test_the_environment_command_refuses_a_bundle_without_a_manifest(
    prospective_frozen, tmp_path, monkeypatch
):
    import dataclasses

    script = _script()
    bundle = prospective_frozen.result.bundle
    document = {k: v for k, v in bundle.document.items() if k != "environment_manifest"}
    monkeypatch.setattr(
        script.frozen, "read_bundle", lambda path: dataclasses.replace(bundle, document=document)
    )
    with pytest.raises(forecast.ForecastRefused) as refused:
        script.main(["environment", "--bundle", str(bundle.path)])
    assert refused.value.reason == "environment_unregistered"
