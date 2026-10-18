"""
scripts/forecast_prospective.py: the report, the registration record and
entry, and forecasts and scores through the real loader and calendar. Git is
replaced by a fake local repository (`_offline`); the real Git helpers are
tested on a temporary repository in test_prospective_provenance.py.
"""

from __future__ import annotations

import importlib.util
import json

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


def _offline(monkeypatch, script, *, files=None, trees=None, committed_at=MERGED, ancestor=True):
    """
    A fake local repository: `files` maps (commit, path) to committed bytes
    (default: the fixture record at RECORD_PATH in the registration commit),
    `trees` maps a commit to its registered paths' tree ids (default: the
    fixture's everywhere), and the merge commit's time and ancestry are given.
    """

    def committed(commit, path):
        if (commit, path) not in files:
            raise forecast.ForecastRefused("record_not_committed", f"{path} not in {commit}")
        return files[(commit, path)]

    monkeypatch.setattr(script, "_code", lambda: RUN_CODE)
    monkeypatch.setattr(script, "_code_trees", lambda commit: (trees or {}).get(commit, CODE_TREES))
    monkeypatch.setattr(script, "_committed_file", committed)
    monkeypatch.setattr(script, "_commit_time", lambda commit: pd.Timestamp(committed_at))
    monkeypatch.setattr(script, "_is_ancestor", lambda older, newer: ancestor)
    return committed


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


def _register_argv(prospective_frozen, ledger_path, record_path, registered_at=MERGED):
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
