"""
scripts/setup_health_schedule.py on a fake Mac: the dry run writes nothing,
--write creates run.sh and the plist only when absent (identical is left
alone, anything else is refused and nothing is written), no mode ever
loads, starts or removes a launchd job, and --status tells loaded, fresh,
stale and missing apart. Every path is under tmp_path; no real launchctl,
gh or home directory is touched.
"""

from __future__ import annotations

import importlib.util
import io
import os
import plistlib
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stock_agent.health.context import CodeState
from stock_agent.health.model import Area, Check, HealthState, Scope
from stock_agent.health.summary import build_summary, write_summary

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 17, 15, tzinfo=UTC)
UID = 501
SERVICE = f"gui/{UID}/local.stock-agent.health"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


setup = _load("setup_health_schedule")


class Launchctl:
    """A fake `launchctl print`: loaded or not; records every call."""

    def __init__(self, *, loaded=False, text="", missing=False):
        self.loaded, self.text, self.missing = loaded, text, missing
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append([str(part) for part in argv])
        if self.missing:
            raise FileNotFoundError("launchctl")
        if self.loaded:
            return subprocess.CompletedProcess(argv, 0, self.text, "")
        return subprocess.CompletedProcess(argv, 113, "", "Could not find service")


def fake(
    tmp_path,
    *,
    worktree=False,
    version="3.12.2",
    venv=True,
    gh="/opt/homebrew/bin/gh",
    env=None,
    platform="darwin",
    launchctl=None,
    now=NOW,
):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if venv:
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
        (repo / ".venv" / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")
    if worktree:
        main = tmp_path / "main"
        (repo / ".git").write_text(f"gitdir: {main}/.git/worktrees/x\n", encoding="utf-8")
    else:
        (repo / ".git").mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    runner = launchctl or Launchctl()
    machine = setup.Machine(
        repo=repo,
        home=home,
        env=env or {},
        uid=UID,
        platform=platform,
        which={"gh": gh, "git": "/usr/bin/git"}.get,
        runner=runner,
        now=lambda: now,
    )
    return machine, runner


def run(machine, *argv):
    out = io.StringIO()
    code = setup.main(list(argv), machine=machine, out=out)
    return code, out.getvalue()


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def plan_of(machine, *argv):
    return setup.detect(machine, setup._parser().parse_args(list(argv)))


# --- the dry run ------------------------------------------------------------------------


def test_the_dry_run_writes_nothing_and_shows_both_files_and_the_owner_steps(tmp_path):
    machine, launchctl = fake(tmp_path)
    before = tree(tmp_path)
    code, out = run(machine)
    assert code == 0
    assert tree(tmp_path) == before  # nothing written, no directory created
    assert not (machine.home / "stock-agent-private").exists()
    assert launchctl.calls == []  # a dry run never calls launchctl
    plan = plan_of(machine)
    assert setup.render_run_sh(plan) in out
    assert setup.render_plist(plan).decode() in out
    assert "dry run (nothing is written)" in out and "would be written with --write" in out
    for _, command in setup.owner_commands(plan):
        assert command in out
    assert f"launchctl bootstrap gui/{UID} " in out and f"launchctl bootout {SERVICE}" in out
    assert "every day at 07:30 local time" in out


def test_found_inputs_are_passed_and_missing_ones_say_why(tmp_path):
    machine, _ = fake(tmp_path)
    private = machine.home / "stock-agent-private"
    (private / "fund_store").mkdir(parents=True)
    (private / "ledger_head.txt").write_text("0" * 64 + "\n", encoding="utf-8")
    plan = plan_of(machine)
    options = {option.flag: option for option in plan.options}
    assert options["--fund-store"].enabled and options["--fund-store"].value == str(
        private / "fund_store"
    )
    assert options["--ledger-head-backup"].enabled
    prospective = options["--prospective-store"]
    assert not prospective.enabled and prospective.value is None
    assert f"not found at {private / 'prospective_store'}" in prospective.reason
    assert "insufficient evidence" in prospective.reason
    script = setup.render_run_sh(plan)
    assert f"--fund-store {private / 'fund_store'}" in script
    assert f"--ledger-head-backup {private / 'ledger_head.txt'}" in script
    assert "\n  --prospective-store" not in script  # left out, and said so in a comment
    assert f"#   --prospective-store: not found at {private / 'prospective_store'}" in script


def test_a_given_path_is_passed_even_before_it_exists(tmp_path):
    machine, _ = fake(tmp_path)
    missing = tmp_path / "elsewhere" / "prospective store"
    option = next(
        o
        for o in plan_of(machine, "--prospective-store", str(missing)).options
        if o.flag == "--prospective-store"
    )
    assert (option.enabled, option.value) == (True, str(missing))
    assert "does not exist yet" in option.reason
    script = setup.render_run_sh(plan_of(machine, "--prospective-store", str(missing)))
    assert f"--prospective-store '{missing}'" in script  # quoted: the path has a space


def test_variables_launchd_will_not_pass_are_made_explicit(tmp_path):
    anchors = tmp_path / "private" / "anchors.csv"
    machine, _ = fake(
        tmp_path,
        env={
            "STOCK_AGENT_ANCHORS": str(anchors),
            "STOCK_AGENT_PROSPECTIVE_LEDGER": "~/ledgers/etf.jsonl",
            "STOCK_AGENT_STOCK_LAB_HOME": str(tmp_path / "lab"),
        },
    )
    plan = plan_of(machine)
    options = {option.flag: option for option in plan.options}
    assert (options["--anchors"].enabled, options["--anchors"].value) == (True, str(anchors))
    assert "launchd jobs do not inherit" in options["--anchors"].reason
    assert options["--ledger"].value == str(machine.home / "ledgers" / "etf.jsonl")
    assert any("STOCK_AGENT_STOCK_LAB_HOME is set here" in warning for warning in plan.warnings)
    lab = {item.name: item for item in plan.stock_lab}
    assert lab["Stock Lab store"].path == tmp_path / "lab" / "stock_lab.sqlite3"
    names = {item.name for item in plan.defaults}
    assert "anchors file" not in names and "ETF forecast ledger" not in names  # passed instead


def test_without_gh_or_with_the_opt_outs_the_flags_are_left_out(tmp_path):
    machine, _ = fake(tmp_path, gh=None)
    options = {option.flag: option for option in plan_of(machine).options}
    assert not options["--github"].enabled
    assert options["--github"].reason.startswith("gh is not on PATH")
    machine, _ = fake(tmp_path)
    plan = plan_of(machine, "--no-github", "--no-verify")
    options = {option.flag: option for option in plan.options}
    assert not options["--github"].enabled and not options["--run-verify"].enabled
    assert "/opt/homebrew/bin" not in plan.path_dirs  # gh's directory only when gh is used
    script = setup.render_run_sh(plan)
    assert "\n  --github" not in script and "\n  --run-verify" not in script


def test_the_run_script_sets_the_environment_launchd_does_not_give(tmp_path):
    machine, _ = fake(tmp_path)
    plan = plan_of(machine)
    script = setup.render_run_sh(plan)
    lines = script.splitlines()
    assert lines[0] == "#!/bin/zsh"
    assert "set -u" in lines and "umask 077" in lines
    assert "export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" in lines
    assert f"export PYTHONPATH={machine.repo / 'src'}" in lines  # this checkout's code
    assert f"cd {machine.repo} || exit 2" in lines
    health = machine.home / "stock-agent-private" / "health"
    assert (
        f"exec {machine.repo / '.venv/bin/python'} scripts/health_check.py local \\\n"
        f"  --output-dir {health} \\\n  --run-verify \\\n  --github\n"
    ) in script
    assert "launchctl" not in script


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_the_run_script_is_valid_zsh(tmp_path):
    machine, _ = fake(tmp_path)
    path = tmp_path / "run.sh"
    path.write_text(
        setup.render_run_sh(plan_of(machine, "--fund-store", "/a b/c'd")), encoding="utf-8"
    )
    assert subprocess.run(["zsh", "-n", str(path)], check=False).returncode == 0


def test_the_plist_is_valid_and_holds_only_the_reviewed_keys(tmp_path):
    machine, _ = fake(tmp_path)
    plan = plan_of(machine, "--at", "06:05")
    document = plistlib.loads(setup.render_plist(plan))
    health = machine.home / "stock-agent-private" / "health"
    assert document == {
        "Label": "local.stock-agent.health",
        "ProgramArguments": ["/bin/zsh", str(health / "run.sh")],
        "StartCalendarInterval": {"Hour": 6, "Minute": 5},
        "RunAtLoad": False,
        "Umask": 0o077,
        "StandardOutPath": str(health / "launchd.out.log"),
        "StandardErrorPath": str(health / "launchd.err.log"),
    }
    assert setup.render_plist(plan) == setup.render_plist(plan_of(machine, "--at", "06:05"))


@pytest.mark.parametrize("at", ["24:00", "7:5", "07:60", "noon"])
def test_a_bad_time_is_refused(tmp_path, at):
    machine, _ = fake(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        run(machine, "--at", at)
    assert stopped.value.code == 2


# --- --write ----------------------------------------------------------------------------


def test_write_creates_both_files_privately_and_never_runs_launchctl(tmp_path):
    machine, launchctl = fake(tmp_path)
    code, out = run(machine, "--write")
    assert code == 0
    where = setup.places(machine)
    plan = plan_of(machine)
    assert where.run_sh.read_text(encoding="utf-8") == setup.render_run_sh(plan)
    assert where.plist.read_bytes() == setup.render_plist(plan)
    assert where.run_sh.stat().st_mode & 0o777 == 0o700
    assert where.plist.stat().st_mode & 0o777 == 0o644
    for directory in (where.private, where.health):
        assert directory.stat().st_mode & 0o777 == 0o700
    assert f"wrote {where.run_sh} (mode 700)" in out and f"wrote {where.plist} (mode 644)" in out
    assert f"launchctl bootstrap gui/{UID} {where.plist}" in out
    assert launchctl.calls == []  # installing is the owner's step
    assert not where.run_sh.is_relative_to(machine.repo)  # private files stay out of the repo


def test_write_again_leaves_identical_files_alone(tmp_path):
    machine, _ = fake(tmp_path)
    assert run(machine, "--write")[0] == 0
    where = setup.places(machine)
    stamps = [path.stat().st_mtime_ns for path in (where.run_sh, where.plist)]
    code, out = run(machine, "--write")
    assert code == 0
    assert f"unchanged {where.run_sh}" in out and f"unchanged {where.plist}" in out
    assert [path.stat().st_mtime_ns for path in (where.run_sh, where.plist)] == stamps


def test_write_never_overwrites_a_different_file_and_then_writes_neither(tmp_path):
    machine, _ = fake(tmp_path)
    where = setup.places(machine)
    where.health.mkdir(parents=True)
    where.run_sh.write_text("#!/bin/zsh\necho the owner's own version\n", encoding="utf-8")
    code, out = run(machine, "--write")
    assert code == 1
    assert where.run_sh.read_text(encoding="utf-8") == "#!/bin/zsh\necho the owner's own version\n"
    assert not where.plist.exists()  # nothing is written when anything would be replaced
    assert "-echo the owner's own version" in out  # the difference is shown
    assert f"not written: {where.run_sh} differs from the plan: never overwritten" in out
    assert "Nothing was written." in out


def test_write_never_follows_or_replaces_a_symlink(tmp_path):
    machine, _ = fake(tmp_path)
    where = setup.places(machine)
    target = tmp_path / "elsewhere.plist"
    target.write_text("keep me", encoding="utf-8")
    where.plist.parent.mkdir(parents=True)
    where.plist.symlink_to(target)
    code, out = run(machine, "--write")
    assert code == 1
    assert target.read_text(encoding="utf-8") == "keep me" and where.plist.is_symlink()
    assert not where.run_sh.exists()
    assert "is a symlink: never followed or replaced" in out


@pytest.mark.parametrize(
    "setting, refusal",
    [
        ({"worktree": True}, "is a linked Git worktree"),
        ({"venv": False}, "does not exist: create the project venv first"),
        ({"version": "3.11.9"}, "the venv is Python 3.11.9, not 3.12"),
        ({"platform": "linux"}, "launchd is macOS only"),
    ],
)
def test_write_refuses_a_place_the_job_could_not_run_from(tmp_path, capsys, setting, refusal):
    machine, launchctl = fake(tmp_path, **setting)
    before = tree(tmp_path)
    code, out = run(machine, "--write")
    assert code == 2 and "wrote" not in out
    assert refusal in capsys.readouterr().err
    assert tree(tmp_path) == before and launchctl.calls == []
    dry_code, dry = run(machine)  # the dry run still shows the plan and the refusal
    assert dry_code == 0 and refusal in dry


def test_a_worktree_names_its_main_checkout(tmp_path):
    machine, _ = fake(tmp_path, worktree=True)
    assert setup.main_checkout(machine.repo) == tmp_path / "main"
    assert f"main checkout ({tmp_path / 'main'})" in plan_of(machine).refusals[0]


def test_the_only_launchctl_call_in_the_script_is_print():
    source = (REPO / "scripts" / "setup_health_schedule.py").read_text(encoding="utf-8")
    calls = re.findall(r'\["launchctl", "([a-z]+)"', source)
    assert calls == ["print"]
    for verb in ("bootstrap", "kickstart", "bootout", "load", "unload", "enable"):
        assert f'"{verb}"' not in source, verb


# --- --status ---------------------------------------------------------------------------

PRINTED = """gui/501/local.stock-agent.health = {
\tactive count = 0
\tpath = /Users/someone/Library/LaunchAgents/local.stock-agent.health.plist
\ttype = LaunchAgent
\tstate = not running
\truns = 3
\tlast exit code = 0
}
"""


def _summary(path: Path, generated_at: datetime) -> None:
    checks = [
        Check(
            check_id="ci.tests",
            area=Area.CI,
            title="Tests",
            state=HealthState.HEALTHY,
            measured_at=generated_at,
            value=10,
            unit="tests passed",
            threshold="t",
            summary="ok.",
            detail="d",
            evidence=None,
        ),
        Check(
            check_id="data.freshness",
            area=Area.DATA,
            title="Freshness",
            state=HealthState.DEGRADED,
            reason="stale_data",
            measured_at=generated_at,
            value=3,
            unit="sessions",
            threshold="t",
            summary="old.",
            detail="d",
            evidence=None,
        ),
    ]
    summary = build_summary(
        checks,
        generated_at=generated_at,
        code=CodeState("a" * 40, False),
        scope=Scope.PRIVATE,
        run_context={"environment": "local"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(path, summary)


def test_status_of_a_job_never_installed_is_not_loaded_and_missing(tmp_path):
    machine, launchctl = fake(tmp_path)
    code, out = run(machine, "--status")
    assert code == 1
    assert launchctl.calls == [["launchctl", "print", SERVICE]]
    assert "local.stock-agent.health: not loaded" in out
    assert "summary   missing" in out and "run.sh" in out and "absent" in out


def test_status_of_a_loaded_job_with_a_fresh_summary(tmp_path):
    machine, launchctl = fake(tmp_path, launchctl=Launchctl(loaded=True, text=PRINTED))
    where = setup.places(machine)
    _summary(where.latest, NOW - timedelta(hours=3))
    where.err_log.write_text("", encoding="utf-8")
    code, out = run(machine, "--status")
    assert code == 0
    assert "loaded (not running)" in out and "runs      3" in out and "last exit 0" in out
    assert "3.0 hours ago" in out and "summary   fresh" in out
    assert "overall   degraded: 1 stale, 1 healthy" in out  # stale is named, not "degraded"
    assert "launchd.err.log: empty" in out
    assert launchctl.calls == [["launchctl", "print", SERVICE]]


def test_status_names_a_stale_summary_and_the_last_error(tmp_path):
    machine, _ = fake(tmp_path, launchctl=Launchctl(loaded=True, text=PRINTED))
    where = setup.places(machine)
    _summary(where.latest, NOW - timedelta(hours=setup.FRESH_HOURS + 4))
    where.err_log.write_text("health_check: refused: something\n", encoding="utf-8")
    code, out = run(machine, "--status")
    assert code == 1
    assert "summary   stale" in out and "30.0 hours ago" in out
    assert "last line: health_check: refused: something" in out


def test_status_shows_a_refused_summary_and_a_missing_launchctl(tmp_path):
    machine, _ = fake(tmp_path, launchctl=Launchctl(missing=True))
    where = setup.places(machine)
    where.health.mkdir(parents=True)
    where.latest.write_text('{"format": "something else"}', encoding="utf-8")
    code, out = run(machine, "--status")
    assert code == 1
    assert "launchctl is not available here" in out
    assert "summary   refused (" in out


# --- names kept in step with their sources ----------------------------------------------


def test_the_scripts_copies_of_other_modules_names_match_their_sources():
    from stock_agent.app import forecast_records
    from stock_agent.data.vendors import anchors
    from stock_agent.health import collect

    assert setup.ANCHORS_ENV == anchors.ANCHORS_ENV
    assert setup.ANCHORS_DEFAULT == anchors.DEFAULT_RELATIVE_PATH
    assert setup.PROSPECTIVE_LEDGER_ENV == forecast_records.LEDGER_ENV
    assert setup.STOCK_LAB_LABEL == _load("stock_lab_schedule").LABEL
    assert setup.STOCK_LAB_KEPT_HEAD == _load("stock_lab_daily").KEPT_HEAD_NAME
    real = setup.places(setup.this_machine())
    assert real.health == collect.DEFAULT_PRIVATE_DIR
    assert real.run_sh.parent == real.health and real.plist.parent.name == "LaunchAgents"
    assert os.fspath(setup.PROJECT) == os.fspath(REPO)
