"""
The printed launchd agents and the email dry run
(scripts/stock_lab_schedule.py): the daily agent runs the daily job at 20:30
New York on weekdays and the briefing agent the briefing script's local
preview at 08:30, both in any zone that keeps New York's clock changes, both
with a private umask. A zone that would drift is refused (for the briefing,
to 09:30; for the daily job, an hour or more either way). The optional wake is
a note only; the dry run describes an email adapter and its approvals without an
address, a credential or any code that could connect. Nothing is installed,
loaded, connected or sent.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import plistlib
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


def test_the_daily_agent_runs_the_daily_job_at_2030_new_york_with_a_private_umask():
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
        "Umask": 0o077,  # what it creates is the owner's only, as for the briefing's agent
    }


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
