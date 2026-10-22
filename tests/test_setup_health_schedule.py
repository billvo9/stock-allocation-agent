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
import inspect
import io
import os
import plistlib
import re
import shlex
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
    branch="main",
    git="/usr/bin/git",
    place="repo",
):
    repo = tmp_path / place
    repo.mkdir(parents=True, exist_ok=True)
    if venv:
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
        (repo / ".venv" / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")
    if worktree:
        main = tmp_path / "main"
        (repo / ".git").write_text(f"gitdir: {main}/.git/worktrees/x\n", encoding="utf-8")
    else:
        (repo / ".git").mkdir(exist_ok=True)
        head = f"ref: refs/heads/{branch}" if branch else "1" * 40
        (repo / ".git" / "HEAD").write_text(head + "\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    runner = launchctl or Launchctl()
    machine = setup.Machine(
        repo=repo,
        home=home,
        env=env or {},
        uid=UID,
        platform=platform,
        which={"gh": gh, "git": git}.get,
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


def test_the_stock_labs_monitor_files_are_listed_where_model_health_reads_them(tmp_path):
    machine, _ = fake(tmp_path)
    inputs = machine.repo / "reports" / "health_inputs"
    before = {item.name: item for item in plan_of(machine).defaults}
    assert before["stock_lab_monitor inputs"].path == inputs / "stock_lab_monitor"
    # Once the daily job writes to the shared folder (from a runner clone), it is listed there.
    shared = machine.home / "stock-agent-private" / "health_inputs" / "stock_lab_monitor"
    shared.mkdir(parents=True)
    (shared / "job-1.json").write_text("{}", encoding="utf-8")
    after = {item.name: item for item in plan_of(machine).defaults}
    listed = after["stock_lab_monitor inputs"]
    assert listed.path == shared and listed.present and listed.note.endswith("; 1 files")
    # The other producers' inputs stay in the checkout.
    assert after["rl_training_health inputs"].path == inputs / "rl_training_health"


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
    assert "holds only the variables run.sh names" in options["--anchors"].reason
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


def test_the_run_script_starts_clean_and_passes_only_named_variables(tmp_path):
    machine, _ = fake(tmp_path)
    plan = plan_of(machine)
    script = setup.render_run_sh(plan)
    lines = script.splitlines()
    assert lines[0] == "#!/bin/zsh -f"  # none of the user's zsh startup files
    assert "set -u" in lines and "umask 077" in lines
    assert f"repo={machine.repo}" in lines and "git=/usr/bin/git" in lines
    health = machine.home / "stock-agent-private" / "health"
    # One stripped environment, for git and for the monitor.
    assert (
        'clean=(/usr/bin/env -i "HOME=${HOME:-}" "USER=${USER:-}" "LOGNAME=${LOGNAME:-}" '
        '"TMPDIR=${TMPDIR:-}" PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin '
        f"PYTHONPATH={machine.repo / 'src'})"
    ) in lines
    assert (
        f'exec "${{clean[@]}}" {machine.repo / ".venv/bin/python"} scripts/health_check.py local'
        f" \\\n  --output-dir {health} \\\n  --run-verify \\\n  --github\n"
    ) in script
    commands = [line for line in lines if not line.startswith("#")]
    assert not [line for line in commands if "export " in line]  # nothing exported to the job
    assert not [line for line in commands if "launchctl" in line]
    # The guard comes before anything runs from the checkout.
    assert script.index("refuse() {") < script.index('cd "$repo"') < script.index("exec ")
    assert '[[ "$branch" == main ]] || refuse' in script
    # git never runs in launchd's environment: both calls are stripped.
    assert script.count('"$git" -C') == script.count('"${clean[@]}" "$git" -C "$repo" ') == 2
    assert "status --porcelain --untracked-files=all \\\n  --ignore-submodules=none" in script


def test_relative_path_entries_never_reach_the_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    machine, _ = fake(tmp_path, gh="tools/gh", git="bin/git")  # found through relative PATH
    plan = plan_of(machine)
    assert plan.gh == str(tmp_path / "tools" / "gh") and plan.git == str(tmp_path / "bin" / "git")
    assert all(os.path.isabs(entry) for entry in plan.path_dirs)
    assert plan.path_dirs[:2] == (str(tmp_path / "tools"), str(tmp_path / "bin"))
    script = setup.render_run_sh(plan)
    path = re.search(r"PATH=(\S+) PYTHONPATH", script).group(1)
    assert all(entry.startswith("/") for entry in path.split(":"))


def test_the_dry_run_warns_when_the_checkout_is_not_main(tmp_path):
    machine, _ = fake(tmp_path, branch="feature/x")
    plan = plan_of(machine)
    assert any("is on feature/x, not main: the job refuses" in w for w in plan.warnings)
    assert plan.refusals == ()  # the owner may switch to main after writing the files
    (tmp_path / "detached").mkdir()
    machine, _ = fake(tmp_path / "detached", branch=None)
    assert any("a detached HEAD" in w for w in plan_of(machine).warnings)
    (tmp_path / "on-main").mkdir()
    machine, _ = fake(tmp_path / "on-main")
    assert not any("not main" in w for w in plan_of(machine).warnings)


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
        "ProgramArguments": ["/bin/zsh", "-f", str(health / "run.sh")],
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
        ({"place": "home/Desktop/OMSA/repo"}, "is inside ~/Desktop. macOS privacy protection"),
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


def test_a_worktree_of_a_checkout_launchd_cannot_read_points_at_the_monitor_clone(tmp_path):
    machine, _ = fake(tmp_path, worktree=True)
    main = machine.home / "Desktop" / "OMSA" / "stock-allocation-agent"
    (machine.repo / ".git").write_text(f"gitdir: {main}/.git/worktrees/x\n", encoding="utf-8")
    (refusal,) = plan_of(machine).refusals
    assert refusal.endswith("schedule the monitor from a clone such as ~/stock-agent-monitor")
    assert str(main) not in refusal


# --- checkouts launchd jobs cannot read --------------------------------------------------


@pytest.mark.parametrize(
    "place, folder",
    [
        ("home/Desktop/OMSA/stock-allocation-agent", "Desktop"),
        ("home/Documents/repo", "Documents"),
        ("home/Downloads/repo", "Downloads"),
        ("home/desktop/repo", "Desktop"),  # macOS's default file system ignores case
        ("home/stock-agent-monitor", None),
        ("home/Desktop-runner/repo", None),
        ("home/Projects/Desktop/repo", None),  # only the home folder's own Desktop
        ("Desktop/repo", None),  # another folder's Desktop
    ],
)
def test_a_checkout_launchd_jobs_cannot_read_is_refused_for_write(tmp_path, place, folder):
    machine, _ = fake(tmp_path, place=place)
    refusals = plan_of(machine).refusals
    if folder is None:
        assert refusals == ()
        return
    (refusal,) = refusals
    assert refusal.startswith(f"this checkout ({machine.repo}) is inside ~/{folder}. ")
    assert "this job would be refused every time it runs (git: Operation not permitted)" in refusal
    assert "for example ~/stock-agent-monitor" in refusal and "Activate the scheduler" in refusal


def test_the_dry_run_from_a_checkout_launchd_cannot_read_says_why_and_writes_nothing(tmp_path):
    machine, launchctl = fake(tmp_path, place="home/Desktop/OMSA/stock-allocation-agent")
    before = tree(tmp_path)
    code, out = run(machine)
    assert code == 0 and tree(tmp_path) == before and launchctl.calls == []
    assert f"! --write would refuse: this checkout ({machine.repo}) is inside ~/Desktop." in out


def test_a_home_folder_given_through_a_link_is_recognized_too(tmp_path):
    real = tmp_path / "real home"
    real.mkdir()
    machine, _ = fake(tmp_path, place="real home/Desktop/repo")
    link = tmp_path / "home link"
    link.symlink_to(real)
    machine = setup.Machine(**{**vars(machine), "repo": machine.repo.resolve(), "home": link})
    assert setup.protected_folder(machine.repo, link) == "Desktop"
    assert any("is inside ~/Desktop" in refusal for refusal in plan_of(machine).refusals)


def test_the_protected_folders_are_the_stock_labs_word_for_word():
    # A copy, since this script imports no other script: the two never drift apart.
    schedule = _load("stock_lab_schedule")
    assert setup.PROTECTED_FOLDERS == schedule.PROTECTED_FOLDERS
    assert inspect.getsource(setup.protected_folder) == inspect.getsource(schedule.protected_folder)


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
    _summary(where.latest, NOW - setup.STALE_AFTER - timedelta(hours=4))
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


# --- the generated job, run as launchd starts it ----------------------------------------

GIT = shutil.which("git")
needs_shell = pytest.mark.skipif(
    not Path("/bin/zsh").exists() or GIT is None, reason="needs /bin/zsh and git"
)
# Stands in for the venv's Python: prints the environment it was given and its arguments.
FAKE_PYTHON = '#!/bin/sh\n/usr/bin/env\necho "argv: $*"\n'
# Stands in for git (as in tests/test_stock_lab_schedule.py): the real git, except for the git
# command named in $HOME/git-fails (symbolic-ref or status), which prints $HOME/git-message on
# stderr and exits with $HOME/git-exit, as git does when it cannot read the checkout; an exit
# of 0 prints the message, then runs the real git (a warning on success).
FAKE_GIT = """#!/bin/sh
if [ -f "$HOME/git-fails" ] && [ "$3" = "$(cat "$HOME/git-fails")" ]; then
  cat "$HOME/git-message" >&2
  status=$(cat "$HOME/git-exit")
  [ "$status" = 0 ] || exit "$status"
fi
exec {git} "$@"
"""
GIT_PROGRAM = "git's folder/git"  # the fake, in the test's home: its quoting is run too
# What git printed on 2026-10-21, when launchd ran the Stock Lab's evening from ~/Desktop.
PRIVACY_REFUSAL = "fatal: Unable to read current working directory: Operation not permitted"
PRIVACY_CAUSE = (
    "macOS privacy protection blocks launchd jobs from ~/Desktop, ~/Documents and ~/Downloads: "
    "set the job up from a clone outside them"
)
DIRTY = "has uncommitted or untracked changes, or git warned (git status --porcelain: "
STAMP = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ health job refused: "
SECRET = "planted-secret-value"
# Variables a POSIX shell sets for itself when the fake Python starts.
SHELL_OWN = {"PWD", "OLDPWD", "SHLVL", "_"}


def _git(repo: Path, *args: str) -> None:
    identity = {"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test"}
    identity |= {"GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test"}
    environment = {"PATH": os.environ["PATH"], "HOME": str(repo.parent), **identity}
    environment |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run([GIT, "-C", str(repo), *args], env=environment, capture_output=True, check=True)


@pytest.fixture
def job(tmp_path):
    """A real checkout on a clean main with a fake venv Python, and its job installed."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".venv\n", encoding="utf-8")
    (repo / "README").write_text("health\n", encoding="utf-8")
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(FAKE_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    (repo / ".venv" / "pyvenv.cfg").write_text("version = 3.12.2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "start")
    home = tmp_path / "home"
    home.mkdir()
    # A profile that would leak a variable and leave a trace if zsh read it.
    marker = home / "zshenv-was-read"
    (home / ".zshenv").write_text(
        f"touch {marker}\nexport FROM_ZSHENV={SECRET}\n", encoding="utf-8"
    )
    (home / GIT_PROGRAM).parent.mkdir()
    (home / GIT_PROGRAM).write_text(FAKE_GIT.format(git=shlex.quote(GIT)), encoding="utf-8")
    (home / GIT_PROGRAM).chmod(0o755)
    return repo, home, _install(repo, home), marker


def _install(repo: Path, home: Path) -> list[str]:
    """
    Write the job's run.sh for `repo`, reading the checkout with the fake git (the real one
    unless told); the plist's command.
    """

    machine = setup.Machine(
        repo=repo,
        home=home,
        env={},
        uid=UID,
        platform="darwin",
        which={"gh": None, "git": str(home / GIT_PROGRAM)}.get,
        runner=Launchctl(),
        now=lambda: NOW,
    )
    plan = plan_of(machine)
    where = setup.places(machine)
    where.health.mkdir(parents=True, exist_ok=True)
    where.run_sh.write_text(setup.render_run_sh(plan), encoding="utf-8")
    where.run_sh.chmod(0o700)
    return plistlib.loads(setup.render_plist(plan))["ProgramArguments"]


def _launch(arguments, home: Path, **extra: str) -> subprocess.CompletedProcess:
    """The plist's command with a launchd-like environment and a secret set for the job."""

    environment = extra | {
        "HOME": str(home),
        "USER": "someone",
        "LOGNAME": "someone",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(home),
        "TIINGO_API_KEY": SECRET,  # as if set with launchctl setenv
    }
    return subprocess.run(
        arguments, env=environment, capture_output=True, text=True, timeout=60, check=False
    )


@needs_shell
def test_the_job_reads_no_profile_and_passes_only_the_named_variables(job):
    repo, home, arguments, marker = job
    assert arguments[:2] == ["/bin/zsh", "-f"]
    done = _launch(arguments, home)
    assert done.returncode == 0, done.stderr
    assert not marker.exists()  # ~/.zshenv was never read
    assert SECRET not in done.stdout and "TIINGO_API_KEY" not in done.stdout
    lines = done.stdout.splitlines()
    names = {line.split("=", 1)[0] for line in lines if "=" in line and not line.startswith("argv")}
    assert names - SHELL_OWN == {"HOME", "USER", "LOGNAME", "TMPDIR", "PATH", "PYTHONPATH"}
    assert f"PYTHONPATH={repo / 'src'}" in lines and f"HOME={home}" in lines
    health = home / "stock-agent-private" / "health"
    assert f"argv: scripts/health_check.py local --output-dir {health} --run-verify" in lines


@needs_shell
@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda repo: _git(repo, "checkout", "-q", "-b", "feature/x"), "is on branch feature/x"),
        (lambda repo: _git(repo, "checkout", "-q", "--detach"), "has a detached HEAD"),
        (
            lambda repo: (repo / "conftest.py").write_text("print('hi')\n", encoding="utf-8"),
            DIRTY + "?? conftest.py)",
        ),
        (
            lambda repo: (repo / "README").write_text("edited\n", encoding="utf-8"),
            DIRTY + " M README)",
        ),
    ],
    ids=["feature-branch", "detached", "untracked-file", "modified-file"],
)
def test_the_job_refuses_anything_but_a_clean_main_and_runs_nothing(job, change, reason):
    repo, home, arguments, _ = job
    change(repo)
    done = _launch(arguments, home)
    assert done.returncode == setup.REFUSED_EXIT == 3
    assert "argv:" not in done.stdout  # the monitor never started
    (line,) = done.stderr.strip().splitlines()
    assert re.fullmatch(
        rf"\d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ health job refused: {re.escape(str(repo))} "
        rf"{re.escape(reason)}.*; nothing was run",
        line,
    ), line


def _git_fails(home: Path, command: str, message: str, status: int) -> None:
    """Make the fake git fail `command` with `message` on stderr and exit `status`."""

    (home / "git-fails").write_text(f"{command}\n", encoding="utf-8")
    (home / "git-message").write_text(message, encoding="utf-8")
    (home / "git-exit").write_text(f"{status}\n", encoding="utf-8")


def _refusal(done: subprocess.CompletedProcess) -> str:
    """The job's one dated stderr line, once it is known to have refused and run nothing."""

    assert done.returncode == setup.REFUSED_EXIT, done.stderr
    assert "argv:" not in done.stdout  # the monitor never started
    (line,) = done.stderr.strip().splitlines()
    assert re.match(STAMP, line), line
    return line


@needs_shell
@pytest.mark.parametrize("command", ["symbolic-ref", "status"])
def test_a_checkout_launchd_cannot_read_is_refused_with_gits_message_and_its_cause(job, command):
    # As the Stock Lab's evening was on 2026-10-21, when its run.sh said "detached HEAD".
    repo, home, arguments, _ = job
    _git_fails(home, command, PRIVACY_REFUSAL + "\n", 128)
    line = _refusal(_launch(arguments, home))
    assert line.endswith(
        f"refused: git cannot read the checkout {repo} (exit 128: {PRIVACY_REFUSAL}); "
        f"{PRIVACY_CAUSE}; nothing was run"
    ), line
    assert "detached" not in line and "uncommitted" not in line


@needs_shell
@pytest.mark.parametrize(
    "command, message, status",
    [
        # /usr/bin/git without the developer tools exits 1, like a detached HEAD, but says so.
        ("symbolic-ref", "xcrun: error: invalid active developer path\nmore detail\n", 1),
        ("symbolic-ref", "", 128),
        ("status", "fatal: index file corrupt\n", 128),
    ],
    ids=["exit-1-with-a-message", "no-message", "status-error"],
)
def test_any_other_git_error_is_reported_as_gits_own_never_as_a_detached_head(
    job, command, message, status
):
    repo, home, arguments, _ = job
    _git_fails(home, command, message, status)
    line = _refusal(_launch(arguments, home))  # only git's first line, on one dated line
    first = message.splitlines()[0] if message else "no message"
    assert line.endswith(
        f"refused: git cannot read the checkout {repo} (exit {status}: {first}); see git's "
        "message; nothing was run"
    ), line
    assert "detached" not in line and "privacy" not in line


@needs_shell
@pytest.mark.parametrize("present", [True, False], ids=["plain-folder", "moved-away"])
def test_a_folder_that_is_not_a_checkout_is_named_with_the_real_gits_message(
    job, tmp_path, present
):
    _, home, _, _ = job
    folder = tmp_path / "not a checkout"
    if present:
        folder.mkdir()
    line = _refusal(_launch(_install(folder, home), home))
    assert f"refused: git cannot read the checkout {folder} (exit 128: fatal: " in line, line
    if present:
        assert "not a git repository" in line and "; the folder is not a Git checkout;" in line
    else:
        assert "cannot change to" in line and "; see git's message;" in line
    assert "detached" not in line


@needs_shell
@pytest.mark.parametrize("command", ["symbolic-ref", "status"])
def test_a_warning_from_git_on_success_still_keeps_the_job_from_starting(job, command):
    repo, home, arguments, _ = job
    _git_fails(home, command, "warning: unable to access a configuration file\n", 0)
    line = _refusal(_launch(arguments, home))  # one dated line, git's warning on it
    if command == "symbolic-ref":
        assert "(exit 0: warning: unable to access a configuration file)" in line, line
    else:  # anything git status prints is not a clean tree; its first line is shown
        assert line.endswith(
            f"refused: {repo} has uncommitted or untracked changes, or git warned (git status "
            "--porcelain: warning: unable to access a configuration file); nothing was run"
        ), line
    assert "detached" not in line


@needs_shell
@pytest.mark.parametrize("hidden_by", ["git-variables", "git-configuration"])
def test_git_variables_and_configuration_cannot_hide_a_dirty_checkout(job, tmp_path, hidden_by):
    repo, home, arguments, _ = job
    (repo / "src").mkdir()
    (repo / "src" / "sitecustomize.py").write_text("print('runs first')\n", encoding="utf-8")
    extra = {}
    if hidden_by == "git-variables":  # launchd's environment points git at a clean main
        other = tmp_path / "other"
        other.mkdir()
        _git(other, "init", "-q", "-b", "main")
        (other / "README").write_text("clean\n", encoding="utf-8")
        _git(other, "add", ".")
        _git(other, "commit", "-q", "-m", "start")
        extra = {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other)}
    else:
        _git(repo, "config", "status.showUntrackedFiles", "no")  # would hide untracked files
    line = _refusal(_launch(arguments, home, **extra))
    # Every untracked file is listed, not only its folder (src/).
    assert line.endswith("(git status --porcelain: ?? src/sitecustomize.py); nothing was run"), line


@needs_shell
def test_control_characters_in_gits_message_never_reach_the_log(job):
    _, home, arguments, _ = job
    _git_fails(home, "symbolic-ref", "fatal: \x1b[31mred\x07 text\n", 128)
    done = _launch(arguments, home)
    assert "(exit 128: fatal: ?[31mred? text)" in _refusal(done)
    assert "\x1b" not in done.stderr and "\x07" not in done.stderr


def test_status_explains_a_refused_run(tmp_path):
    printed = PRINTED.replace("last exit code = 0", "last exit code = 3")
    machine, _ = fake(tmp_path, launchctl=Launchctl(loaded=True, text=printed))
    where = setup.places(machine)
    _summary(where.latest, NOW - timedelta(hours=30))
    where.err_log.write_text(
        "2026-10-17T14:30:00Z health job refused: /repo is on branch x, not main; nothing was "
        "run\n",
        encoding="utf-8",
    )
    code, out = run(machine, "--status")
    assert code == 1
    assert (
        "last exit 3: refused: the checkout is not a clean main, or git cannot read it (see the "
        "stderr log)"
    ) in out
    assert "summary   stale" in out and "is on branch x, not main" in out
