"""
The printed 08:30 premarket-briefing agent and the email dry run
(scripts/stock_lab_schedule.py): the agent runs the briefing script's local
preview at 08:30 New York on weekdays in any zone that keeps New York's clock
changes, refuses a zone that would drift to 09:30, and prints the optional
wake as a note only; the dry run describes an email adapter and its
approvals without an address, a credential or any code that could connect.
Nothing is installed, loaded, connected or sent.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import plistlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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


def test_the_daily_agent_is_unchanged_by_the_briefing_option():
    code, printed, note = run()
    assert code == 0 and note == "" and printed == schedule.plist()
    agent = plistlib.loads(schedule.plist("America/Los_Angeles", home=Path("/h")).encode())
    project = schedule.PROJECT
    assert agent == {
        "Label": "local.stock-lab.daily",
        "ProgramArguments": [
            str(project / ".venv" / "bin" / "python"),
            str(project / "scripts" / "stock_lab_daily.py"),
        ],
        "WorkingDirectory": str(project),
        "EnvironmentVariables": {"PYTHONPATH": str(project / "src")},
        "StartCalendarInterval": [
            {"Weekday": day, "Hour": 17, "Minute": 30} for day in (1, 2, 3, 4, 5)
        ],
        "StandardOutPath": "/h/stock-agent-private/stock_lab/logs/daily.out.log",
        "StandardErrorPath": "/h/stock-agent-private/stock_lab/logs/daily.err.log",
    }


@pytest.mark.parametrize("args", [["--zone", "Not/AZone"], ["--briefing", "--zone", "../etc"]])
def test_a_zone_that_is_not_one_is_refused_without_a_traceback(args):
    code, printed, message = run(*args)
    assert code == 2 and printed == "" and message.startswith("refused:")


def test_the_options_are_exclusive_and_never_abbreviated():
    for args in (["--briefing", "--email-dry-run"], ["--brief"], ["--email"]):
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
