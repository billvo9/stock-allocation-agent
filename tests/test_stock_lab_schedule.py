"""
The printed launchd agents, the evening agent's run.sh and the email dry run
(scripts/stock_lab_schedule.py): the evening agent starts run.sh at 20:30 New
York on weekdays and the briefing agent the briefing script's local preview at
08:30, both in any zone that keeps New York's clock changes, both with a
private umask. A zone that would drift is refused (for the briefing, to 09:30;
for the evening, an hour or more either way). run.sh runs only a clean main
checkout, updates prices with the key from the Keychain and records the
evening only after that succeeds, each step in a stripped environment; it is
run here under the real /bin/zsh -f with a fake venv Python. The optional wakes
are notes only; the dry run describes an email adapter and its approvals without
an address, a credential or any code that could connect. Nothing is installed,
loaded, connected or sent.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import plistlib
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stock_lab_schedule.py"
NEW_YORK = ZoneInfo("America/New_York")


def _script():
    spec = importlib.util.spec_from_file_location("stock_lab_schedule_printer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedule = _script()


def run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = schedule.main(list(args), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_the_briefing_agent_runs_the_local_preview_at_0830_new_york_on_weekdays():
    code, printed, note = run("--briefing")
    assert code == 0
    agent = plistlib.loads(printed.encode())
    assert agent["Label"] == schedule.BRIEFING_LABEL == "local.stock-lab.briefing"
    python, script, *flags = agent["ProgramArguments"]
    assert python.endswith(".venv/bin/python") and script.endswith("scripts/stock_lab_briefing.py")
    assert flags == ["--preview"]  # the local preview only: no email, no --at, no closed days
    times = agent["StartCalendarInterval"]
    assert [t["Weekday"] for t in times] == [1, 2, 3, 4, 5]
    assert {(t["Hour"], t["Minute"]) for t in times} == {(8, 30)}
    assert agent["StandardOutPath"].endswith("stock-agent-private/stock_lab/logs/briefing.out.log")
    assert agent["EnvironmentVariables"] == {"PYTHONPATH": str(schedule.PROJECT / "src")}
    assert agent["WorkingDirectory"] == str(schedule.PROJECT)
    assert not {"RunAtLoad", "KeepAlive", "StartInterval"} & set(agent)  # never runs on load
    assert agent["Umask"] == 0o077  # its files and logs are the owner's only
    assert "sudo pmset repeat wakeorpoweron MTWRF 08:25:00" in note
    assert "pmset -g sched" in note and "may replace" in note
    assert "<plist" not in note and "pmset" not in printed  # the note never enters the file


@pytest.mark.parametrize(
    ("zone", "local"),
    [
        ("America/New_York", (8, 30)),
        ("America/Toronto", (8, 30)),
        ("America/Chicago", (7, 30)),
        ("America/Denver", (6, 30)),
        ("America/Los_Angeles", (5, 30)),
    ],
)
def test_a_zone_that_keeps_new_yorks_clock_changes_fires_at_0830_new_york_all_year(zone, local):
    agent = plistlib.loads(schedule.briefing_plist(zone).encode())
    assert {(t["Hour"], t["Minute"]) for t in agent["StartCalendarInterval"]} == {local}
    # Either side of both clock changes, the printed local time is 08:30 in New York.
    for day in ("2026-03-06", "2026-03-09", "2026-10-30", "2026-11-02", "2026-12-21"):
        moment = datetime.fromisoformat(f"{day}T{local[0]:02d}:{local[1]:02d}")
        there = moment.replace(tzinfo=ZoneInfo(zone)).astimezone(NEW_YORK)
        assert (there.hour, there.minute) == (8, 30), (zone, day)


@pytest.mark.parametrize(
    "zone", ["Europe/London", "Europe/Paris", "Asia/Tokyo", "America/Phoenix", "UTC"]
)
def test_a_zone_that_would_drift_to_the_open_is_refused_and_prints_nothing(zone):
    code, printed, message = run("--briefing", "--zone", zone)
    assert code == 2 and printed == ""
    assert message.startswith("refused:") and "an hour off" in message


def test_the_evening_agent_starts_run_sh_at_2030_new_york_with_a_private_umask():
    code, printed, note = run()
    assert code == 0 and printed == schedule.plist()
    agent = plistlib.loads(schedule.plist("America/Los_Angeles", home=Path("/h")).encode())
    # Only zsh -f and the private run.sh: no variable, working folder or key in the file.
    assert agent == {
        "Label": "local.stock-lab.daily",
        "ProgramArguments": ["/bin/zsh", "-f", "/h/stock-agent-private/stock_lab/run.sh"],
        "StartCalendarInterval": [
            {"Weekday": day, "Hour": 17, "Minute": 30} for day in (1, 2, 3, 4, 5)
        ],
        "StandardOutPath": "/h/stock-agent-private/stock_lab/logs/daily.out.log",
        "StandardErrorPath": "/h/stock-agent-private/stock_lab/logs/daily.err.log",
        "Umask": 0o077,  # what it creates is the owner's only, as for the briefing's agent
    }
    assert "sudo pmset repeat wakeorpoweron MTWRF 20:25:00" in note  # printed, never run
    assert "pmset -g sched" in note and "replaces" in note and "08:25 wake" in note
    assert "09:30 New York" in note and "logged in" in note
    assert "<plist" not in note and "pmset" not in printed  # the note never enters the file


def test_the_evening_wake_is_given_in_the_macs_zone():
    note = schedule.evening_wake_note("America/Los_Angeles")
    assert "sudo pmset repeat wakeorpoweron MTWRF 17:25:00" in note
    with pytest.raises(ValueError):
        schedule.evening_wake_note("Europe/London")  # would drift, like the agent itself


def _launchd_weekday(day: date) -> int:
    return day.isoweekday() % 7  # launchd: Sunday 0, Monday 1, ... Saturday 6


def test_every_zone_is_refused_or_runs_the_daily_job_at_2030_new_york_on_each_weekday():
    # Read back from the printed file: each entry's local weekday and time, on every such
    # day of 2026 in that zone, is 20:30 on a New York weekday, never before the close
    # counts as available (stock_lab_daily.AVAILABILITY, 20:00).
    accepted, refused = set(), set()
    for zone in sorted(available_timezones()):
        try:
            agent = plistlib.loads(schedule.plist(zone).encode())
        except ValueError:
            refused.add(zone)
            continue
        accepted.add(zone)
        there, fired = ZoneInfo(zone), set()
        for entry in agent["StartCalendarInterval"]:
            for offset in range(365):
                day = date(2026, 1, 1) + timedelta(days=offset)
                if _launchd_weekday(day) != entry["Weekday"]:
                    continue
                hour, minute = entry["Hour"], entry["Minute"]
                local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=there)
                new_york = local.astimezone(NEW_YORK)
                assert (new_york.hour, new_york.minute) == (20, 30), (zone, day)
                fired.add(_launchd_weekday(new_york.date()))
        assert fired == {1, 2, 3, 4, 5}, zone
    assert {"America/New_York", "America/Toronto", "America/Los_Angeles"} <= accepted
    assert {"Europe/London", "Asia/Tokyo", "Australia/Sydney", "UTC"} <= refused


@pytest.mark.parametrize(
    "zone",
    [
        "Europe/London",  # 21:30 New York in the weeks between the two countries' changes
        "Asia/Tokyo",  # 21:30 New York all summer
        "America/Phoenix",
        "UTC",
        "Australia/Sydney",  # 22:30 New York in the northern summer
        "Antarctica/Troll",  # 19:30 New York in summer, before the close counts as available
    ],
)
def test_the_daily_agent_refuses_a_zone_that_would_drift_and_prints_nothing(zone):
    code, printed, message = run("--zone", zone)
    assert code == 2 and printed == ""
    assert message.startswith("refused:") and "20:30 New York" in message


def test_a_time_past_local_midnight_moves_the_weekdays_with_it(monkeypatch):
    # No zone that keeps New York's clock changes is that far east; the shift stays correct
    # for one that would be.
    monkeypatch.setattr(schedule, "fixed_local_time", lambda zone, when: (0, 30, 1))
    times = plistlib.loads(schedule.plist().encode())["StartCalendarInterval"]
    assert [t["Weekday"] for t in times] == [2, 3, 4, 5, 6]  # Tuesday-Saturday there
    assert {(t["Hour"], t["Minute"]) for t in times} == {(0, 30)}


def test_the_install_steps_create_every_private_folder_with_mode_700():
    # mkdir -p -m 700 sets the mode of each folder it names only: one it creates on the way
    # gets the default 0755, and the store refuses a stock_lab folder others can read.
    steps = next(line.split() for line in schedule.__doc__.splitlines() if "mkdir" in line)
    private = "~/stock-agent-private"
    assert steps == ["mkdir", "-p", "-m", "700", private, f"{private}/stock_lab",
                     f"{private}/stock_lab/logs"]  # fmt: skip


@pytest.mark.parametrize("args", [["--zone", "Not/AZone"], ["--briefing", "--zone", "../etc"]])
def test_a_zone_that_is_not_one_is_refused_without_a_traceback(args):
    code, printed, message = run(*args)
    assert code == 2 and printed == "" and message.startswith("refused:")


def test_the_options_are_exclusive_and_never_abbreviated():
    for args in (["--briefing", "--email-dry-run"], ["--brief"], ["--email"],
                 ["--run-sh", "--briefing", "--model", MODEL], ["--run", "--model", MODEL]):  # fmt: skip
        with pytest.raises(SystemExit) as stopped:
            run(*args)
        assert stopped.value.code == 2


def test_the_email_dry_run_describes_the_adapter_and_its_approvals_only():
    code, printed, note = run("--email-dry-run")
    assert code == 0 and note == ""
    assert printed.startswith("EMAIL DELIVERY OF THE PREMARKET BRIEFING: DRY RUN")
    for fact in (
        "Nothing is connected, no credential is read and nothing is sent.",
        "Provider: not chosen",
        "Recipient: not set",
        "macOS Keychain",
        "at or after 09:30 New York on the market date nothing is sent",
        "never --at",
        "At most once per market date",
        "The schedule never emails a synthetic demo.",
        "Public CI never holds it",
        "Data licence",
        "Nothing above has run.",
        "could read it",
        "needs its own approval",
    ):
        assert fact in " ".join(printed.split()), fact
    assert "@" not in printed  # no address, real or made up
    # The credential is typed at a prompt: nothing follows -w in the add command.
    add = next(line for line in printed.splitlines() if "add-generic-password" in line)
    assert add.rstrip().endswith("-U -w")
    assert f"-a {schedule.KEYCHAIN_ACCOUNT} " in add  # a generic label, never the address


MODEL = "2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7"


def test_run_sh_updates_prices_then_records_the_activated_model_and_never_holds_the_key():
    code, printed, note = run("--run-sh", "--model", MODEL)
    assert code == 0 and note == "" and printed == schedule.render_run_sh(MODEL)
    lines = printed.splitlines()
    assert lines[0] == "#!/bin/zsh -f"  # none of the user's zsh startup files
    assert "set -u" in lines and "umask 077" in lines
    assert f"repo='{schedule.PROJECT}'" in lines and f"model={MODEL}" in lines
    code_lines = [line for line in lines if not line.lstrip().startswith("#")]
    (fetch,) = [i for i, line in enumerate(lines) if line in code_lines and "fetch_recent" in line]
    (record,) = [i for i, line in enumerate(lines) if line in code_lines and "--record" in line]
    assert lines[fetch] == (
        '"${clean[@]}" "$python" scripts/stock_lab_fetch_recent.py --authorized --store --keychain'
    )
    assert lines[record] == (
        'exec "${clean[@]}" "$python" scripts/stock_lab_daily.py --record-real --model "$model"'
    )
    # The fetch's status is checked before the record step, which runs only after success.
    assert fetch < lines.index("fetched=$?") < lines.index("if (( fetched != 0 )); then") < record
    assert lines[lines.index("if (( fetched != 0 )); then") + 2] == "  exit $fetched"
    (clean,) = [line for line in lines if line.startswith("clean=(")]
    assert clean.startswith("clean=(/usr/bin/env -i ")
    assert 'PATH=/usr/bin:/bin:/usr/sbin:/sbin "PYTHONPATH=$repo/src")' in clean
    body = "\n".join(code_lines)
    for absent in ("TIINGO_API_KEY", "security", "launchctl", "pmset", "export ", "setenv",
                   "--backfill", "--at "):  # fmt: skip
        assert absent not in body, absent


@pytest.mark.parametrize(
    "args",
    [
        ["--run-sh"],
        ["--model", MODEL],
        ["--briefing", "--model", MODEL],
        ["--run-sh", "--model", MODEL[:16]],  # a prefix is not enough for an unattended job
        ["--run-sh", "--model", MODEL.upper()],
        ["--run-sh", "--model", MODEL[:-1] + "g"],
        ["--run-sh", "--model", f"{MODEL[:-1]}'"],
    ],
)
def test_run_sh_needs_one_full_model_id_and_prints_nothing_otherwise(args):
    code, printed, message = run(*args)
    assert code == 2 and printed == "" and message.startswith("refused:")


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
def test_run_sh_is_valid_zsh(tmp_path):
    path = tmp_path / "run.sh"
    path.write_text(schedule.render_run_sh(MODEL, project=Path("/a b/it's here")), encoding="utf-8")
    assert subprocess.run(["zsh", "-n", str(path)], check=False).returncode == 0


def test_the_printer_imports_the_standard_library_only_and_runs_nothing():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert imported <= {"__future__", "argparse", "sys", "datetime", "pathlib", "typing", "xml",
                        "zoneinfo"}, imported  # fmt: skip
    for forbidden in ("environ", "getenv", "system", "popen", "run", "open", "write_text",
                      "mkdir", "call", "check_output"):  # fmt: skip
        assert forbidden not in names - {"write"}, forbidden


# --- the evening agent's run.sh, run as launchd starts it --------------------------------

GIT = shutil.which("git")
# run.sh uses /bin/zsh and /usr/bin/git, both part of macOS: there these tests never skip.
needs_shell = pytest.mark.skipif(
    sys.platform != "darwin" and not (Path("/bin/zsh").exists() and Path("/usr/bin/git").exists()),
    reason="the evening's run.sh needs /bin/zsh and /usr/bin/git (always on macOS)",
)
# Stands in for the venv's Python: records each step's arguments and environment, and exits
# with .venv/fetch-exit (the price update) or .venv/record-exit (the record step) when that
# file exists (the job runs in the checkout, where .venv is ignored: never dirty).
FAKE_PYTHON = """#!/bin/sh
step=$(basename "$1")
/usr/bin/env > "$HOME/env-$step"
echo "$*" >> "$HOME/steps"
case "$step" in
  stock_lab_fetch_recent.py) exits=.venv/fetch-exit ;;
  *) exits=.venv/record-exit ;;
esac
[ -f "$exits" ] && exit "$(cat "$exits")"
exit 0
"""
SECRET = "planted-secret-value"
SHELL_OWN = {"PWD", "OLDPWD", "SHLVL", "_"}  # set by the fake Python's own shell
STAMP = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ stock lab evening "


def _git(repo: Path, *args: str) -> None:
    identity = {"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test"}
    identity |= {"GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test"}
    environment = {"PATH": os.environ["PATH"], "HOME": str(repo.parent), **identity}
    environment |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run([GIT, "-C", str(repo), *args], env=environment, capture_output=True, check=True)


@pytest.fixture
def evening(tmp_path):
    """A clean main checkout with a fake venv Python, its run.sh and the agent's command."""

    repo = tmp_path / "the repo's $HOME `x`"  # quoting is tested by running, not only parsing
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".venv\n", encoding="utf-8")
    (repo / "README").write_text("stock lab\n", encoding="utf-8")
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(FAKE_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "start")
    home = tmp_path / "home"
    folder = schedule.private_folder(home)
    folder.mkdir(parents=True)
    # A profile that would leak a variable and leave a trace if zsh read it.
    marker = home / "zshenv-was-read"
    (home / ".zshenv").write_text(f"touch {marker}\nexport FROM_ZSHENV={SECRET}\n")
    (folder / schedule.RUN_SH).write_text(schedule.render_run_sh(MODEL, project=repo))
    (folder / schedule.RUN_SH).chmod(0o700)
    arguments = plistlib.loads(schedule.plist(home=home).encode())["ProgramArguments"]
    return repo, home, arguments, marker


def _launch(arguments, home: Path, **extra: str) -> subprocess.CompletedProcess:
    """The plist's command with a launchd-like environment and the key set for the job."""

    environment = extra | {
        "HOME": str(home),
        "USER": "someone",
        "LOGNAME": "someone",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(home),
        "TIINGO_API_KEY": SECRET,  # as if set with launchctl setenv: it must not pass
    }
    return subprocess.run(
        arguments, env=environment, capture_output=True, text=True, timeout=60, check=False
    )


def _steps(home: Path) -> list[str]:
    path = home / "steps"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@needs_shell
def test_the_evening_updates_prices_then_records_with_only_the_named_variables(evening):
    repo, home, arguments, marker = evening
    done = _launch(arguments, home)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    assert _steps(home) == [
        "scripts/stock_lab_fetch_recent.py --authorized --store --keychain",
        f"scripts/stock_lab_daily.py --record-real --model {MODEL}",
    ]
    assert not marker.exists()  # ~/.zshenv was never read
    for step in ("stock_lab_fetch_recent.py", "stock_lab_daily.py"):
        lines = (home / f"env-{step}").read_text(encoding="utf-8").splitlines()
        assert not [line for line in lines if SECRET in line or "TIINGO" in line], step
        names = {line.split("=", 1)[0] for line in lines if "=" in line}
        assert names - SHELL_OWN == {"HOME", "USER", "LOGNAME", "TMPDIR", "PATH", "PYTHONPATH"}
        assert f"PYTHONPATH={repo / 'src'}" in lines and f"HOME={home}" in lines
        assert "PATH=/usr/bin:/bin:/usr/sbin:/sbin" in lines


@needs_shell
@pytest.mark.parametrize("status", [1, 2, 127])
def test_a_failed_price_update_stops_the_evening_before_any_forecast(evening, status):
    repo, home, arguments, _ = evening
    (repo / ".venv" / "fetch-exit").write_text(f"{status}\n", encoding="utf-8")
    done = _launch(arguments, home)
    assert done.returncode == status
    assert _steps(home) == ["scripts/stock_lab_fetch_recent.py --authorized --store --keychain"]
    (line,) = done.stderr.strip().splitlines()
    assert re.fullmatch(
        STAMP + rf"stopped: the price update failed \(exit {status}\); no forecast was recorded "
        r"tonight",
        line,
    ), line


@needs_shell
@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda repo: _git(repo, "checkout", "-q", "-b", "feature/x"), "is on branch feature/x"),
        (lambda repo: _git(repo, "checkout", "-q", "--detach"), "has a detached HEAD"),
        (
            lambda repo: (repo / "conftest.py").write_text("print('hi')\n", encoding="utf-8"),
            "has uncommitted or untracked changes",
        ),
        (
            lambda repo: (repo / "README").write_text("edited\n", encoding="utf-8"),
            "has uncommitted or untracked changes",
        ),
    ],
    ids=["feature-branch", "detached", "untracked-file", "modified-file"],
)
def test_the_evening_refuses_anything_but_a_clean_main_and_runs_nothing(evening, change, reason):
    repo, home, arguments, _ = evening
    change(repo)
    done = _launch(arguments, home)
    assert done.returncode == schedule.REFUSED_EXIT == 3
    assert _steps(home) == []  # neither step started, so the key was never read
    (line,) = done.stderr.strip().splitlines()
    assert re.fullmatch(
        STAMP + rf"refused: {re.escape(str(repo))} {re.escape(reason)}.*; nothing was run", line
    ), line


@needs_shell
@pytest.mark.parametrize("status", [0, 1, 2])
def test_the_record_steps_status_is_the_evenings(evening, status):
    repo, home, arguments, _ = evening
    (repo / ".venv" / "record-exit").write_text(f"{status}\n", encoding="utf-8")
    done = _launch(arguments, home)
    assert done.returncode == status and done.stderr == ""
    assert len(_steps(home)) == 2


@needs_shell
def test_git_variables_and_configuration_cannot_hide_a_dirty_checkout(evening, tmp_path):
    repo, home, arguments, _ = evening
    other = tmp_path / "other"  # a clean main elsewhere, which GIT_DIR would point at
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")
    (other / "README").write_text("clean\n", encoding="utf-8")
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "start")
    _git(repo, "config", "status.showUntrackedFiles", "no")  # would hide untracked files
    (repo / "src").mkdir()
    (repo / "src" / "sitecustomize.py").write_text("print('runs first')\n", encoding="utf-8")
    done = _launch(arguments, home, GIT_DIR=str(other / ".git"), GIT_WORK_TREE=str(other))
    assert done.returncode == 3 and _steps(home) == []
    assert "has uncommitted or untracked changes" in done.stderr


@needs_shell
def test_the_evening_refuses_a_linked_worktree(evening):
    repo, home, arguments, _ = evening
    worktree = repo.parent / "worktree"
    _git(repo, "worktree", "add", "-q", "-b", "main-copy", str(worktree))
    run_sh = schedule.private_folder(home) / schedule.RUN_SH
    run_sh.write_text(schedule.render_run_sh(MODEL, project=worktree), encoding="utf-8")
    done = _launch(arguments, home)
    assert done.returncode == 3 and _steps(home) == []
    assert "is not the main checkout" in done.stderr
