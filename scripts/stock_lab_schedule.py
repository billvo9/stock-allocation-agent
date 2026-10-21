"""
Print a macOS launchd agent for the Stock Lab, or describe an email delivery
as a dry run (owner steps; nothing here installs, loads, connects or sends
anything, and no credential is read).

Modes
    (default)        the unattended evening agent: 20:30 New York time on
                     weekdays, `/bin/zsh -f ~/stock-agent-private/stock_lab/run.sh`
                     (print that file with --run-sh). An optional wake five
                     minutes earlier is printed as a note on standard error,
                     never run.
    --run-sh --model ID
                     the evening agent's run.sh, for the activated model ID (its
                     full 64-character id). It runs only from a clean checkout
                     of main (otherwise exit 3 and a dated reason on stderr,
                     nothing run), then each step under `env -i` with only HOME,
                     USER, LOGNAME, TMPDIR, a system PATH and PYTHONPATH:
                       1. scripts/stock_lab_fetch_recent.py --authorized --store
                          --keychain: the recent prices, with the Tiingo key
                          read from the macOS Keychain inside that process;
                       2. only when step 1 succeeds, scripts/stock_lab_daily.py
                          --record-real --model ID: that evening's forecasts.
                     A failed step 1 stops the run with its exit status: no
                     forecast is recorded that evening and none is backfilled
                     later. The job takes its session from the wall clock and
                     skips holidays from the NYSE calendar, so a run on a
                     holiday records nothing new rather than inventing a session.
    --briefing       the premarket briefing: 08:30 New York time on weekdays,
                     scripts/stock_lab_briefing.py --preview, which writes the
                     briefing's files into the private briefing folder once
                     per market date (the local preview channel). It skips
                     weekends and holidays itself. A run that starts at or
                     after 09:30 New York (a Mac that woke late) writes a
                     labelled late preview, never that day's delivery. An
                     optional wake five minutes earlier is printed as a note
                     on standard error, never run.
    --email-dry-run  what an email adapter for the briefing would do, and what
                     the owner must approve first. No email code exists.

launchd uses the Mac's local time zone, so the printed hour is converted from
New York to the local zone given (America/New_York by default). The zone must
keep the same local time all year: a zone whose clock changes on other dates
than New York's (or never) would fire an hour or more off for part of the
year, at 09:30 for an 08:30 briefing, and is refused for both agents.

The agents run the code of the checkout they were printed from: print them
from the clean main checkout, never from a working branch or worktree.

To install, after reading the printed files and approving the installation
(owner, once per agent; the Keychain item first, docs/operations/runbook.md,
"Stock Lab unattended evening"):
    mkdir -p -m 700 ~/stock-agent-private ~/stock-agent-private/stock_lab ~/stock-agent-private/stock_lab/logs
    .venv/bin/python scripts/stock_lab_schedule.py --run-sh --model ID > ~/stock-agent-private/stock_lab/run.sh.new && mv ~/stock-agent-private/stock_lab/run.sh.new ~/stock-agent-private/stock_lab/run.sh
    chmod 700 ~/stock-agent-private/stock_lab/run.sh
    .venv/bin/python scripts/stock_lab_schedule.py > ~/Library/LaunchAgents/local.stock-lab.daily.plist
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.stock-lab.daily.plist
    .venv/bin/python scripts/stock_lab_schedule.py --briefing > ~/Library/LaunchAgents/local.stock-lab.briefing.plist
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.stock-lab.briefing.plist
To remove one: launchctl bootout gui/$(id -u)/<label>, then delete its file.

Logs go to ~/stock-agent-private/stock_lab/logs/ (private, outside Git) and
keep each run's printed text; both agents write with a private umask (files
0600). An agent installed from an earlier printout keeps its old settings
until it is printed again, booted out and bootstrapped. When the Mac is asleep
at the scheduled time, launchd runs the job when it next wakes, so a sleeping
Mac delivers late or not at all (docs/research/stock_probability_lab.md,
section 9, has the limits and the recommended hosted option).

Usage:
    .venv/bin/python scripts/stock_lab_schedule.py [--zone America/Los_Angeles]
    .venv/bin/python scripts/stock_lab_schedule.py --run-sh --model ID
    .venv/bin/python scripts/stock_lab_schedule.py --briefing [--zone America/Chicago]
    .venv/bin/python scripts/stock_lab_schedule.py --email-dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")
LABEL = "local.stock-lab.daily"
NEW_YORK_TIME = (20, 30)
EVENING_WAKE = (20, 25)  # an optional wake, five minutes before the evening agent
RUN_SH = "run.sh"  # the evening agent's script, in ~/stock-agent-private/stock_lab/
MAIN_BRANCH = "main"
REFUSED_EXIT = 3  # run.sh's exit status when the checkout is not a clean main
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
# What each step's environment holds (`env -i`): nothing else from launchd or a shell.
KEPT_VARIABLES = ("HOME", "USER", "LOGNAME", "TMPDIR")
HEX = frozenset("0123456789abcdef")
BRIEFING_LABEL = "local.stock-lab.briefing"
BRIEFING_TIME = (8, 30)  # the briefing's target (stock_lab.briefing.timing.TARGET_TIME)
WAKE_TIME = (8, 25)  # an optional wake, five minutes before
PMSET_DAYS = "MTWRFSU"  # pmset's weekday letters, Monday first
KEYCHAIN_SERVICE = "stock-agent-briefing-email"
KEYCHAIN_ACCOUNT = "stock-lab-briefing"  # a generic label, never the address
PRIVATE_UMASK = 0o077  # both agents' files and logs: readable by the owner only


def fixed_local_time(zone: str, new_york_time: tuple[int, int]) -> tuple[int, int, int]:
    """
    `new_york_time` in `zone` as (hour, minute, day shift) when it is the same
    on every day of a year; otherwise refused (ValueError), since launchd and
    pmset take a local time with no time zone. A zone east of New York may
    reach the next calendar day, so its weekdays shift by one.
    """

    target = ZoneInfo(zone)
    seen = set()
    for offset in range(366):
        day = date(2026, 1, 1) + timedelta(days=offset)
        new_york = datetime(day.year, day.month, day.day, *new_york_time, tzinfo=NEW_YORK)
        local = new_york.astimezone(target)
        seen.add((local.hour, local.minute, (local.date() - day).days))
    if len(seen) != 1:
        hour, minute = new_york_time
        raise ValueError(
            f"{hour:02d}:{minute:02d} New York is not one fixed local time in {zone} all year "
            "(its clock changes on other dates than New York's, or never), so a fixed launchd "
            "time would run an hour off for part of the year; set the Mac to a zone that follows "
            "New York's clock changes, or use a host on New York time"
        )
    return seen.pop()


def private_folder(home: Path | None = None) -> Path:
    """~/stock-agent-private/stock_lab: the store, the evening agent's run.sh and the logs."""

    return (home or Path.home()) / "stock-agent-private" / "stock_lab"


def _plist(
    label: str,
    when: tuple[int, int, int],
    program: list[str],
    log: str,
    home: Path | None,
    *,
    python_job: bool = True,
) -> str:
    logs = private_folder(home) / "logs"
    hour, minute, shift = when
    arguments = "".join(f"\n    <string>{escape(str(part))}</string>" for part in program)
    # A Python job gets this checkout and its src; run.sh sets both itself.
    settings = (
        f"""
  <key>WorkingDirectory</key><string>{escape(str(PROJECT))}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key><string>{escape(str(PROJECT / "src"))}</string>
  </dict>"""
        if python_job
        else ""
    )
    days = "".join(
        f"""
    <dict>
      <key>Weekday</key><integer>{(day + shift) % 7}</integer>
      <key>Hour</key><integer>{hour}</integer>
      <key>Minute</key><integer>{minute}</integer>
    </dict>"""
        for day in range(1, 6)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>{arguments}
  </array>{settings}
  <key>StartCalendarInterval</key>
  <array>{days}
  </array>
  <key>StandardOutPath</key><string>{escape(str(logs / f"{log}.out.log"))}</string>
  <key>StandardErrorPath</key><string>{escape(str(logs / f"{log}.err.log"))}</string>
  <key>Umask</key><integer>{PRIVATE_UMASK}</integer>
</dict>
</plist>
"""


def plist(zone: str = "America/New_York", home: Path | None = None) -> str:
    """The evening agent: 20:30 New York on weekdays, `/bin/zsh -f run.sh` (render_run_sh)."""

    program = ["/bin/zsh", "-f", str(private_folder(home) / RUN_SH)]
    when = fixed_local_time(zone, NEW_YORK_TIME)
    return _plist(LABEL, when, program, "daily", home, python_job=False)


def briefing_plist(zone: str = "America/New_York", home: Path | None = None) -> str:
    """The premarket briefing's agent: 08:30 New York on weekdays, writing the local preview."""

    python = PROJECT / ".venv" / "bin" / "python"
    script = PROJECT / "scripts" / "stock_lab_briefing.py"
    when = fixed_local_time(zone, BRIEFING_TIME)
    return _plist(BRIEFING_LABEL, when, [str(python), str(script), "--preview"], "briefing", home)


def _pmset(zone: str, wake: tuple[int, int]) -> str:
    hour, minute, shift = fixed_local_time(zone, wake)
    days = "".join(PMSET_DAYS[(day + shift) % 7] for day in range(5))
    return f"sudo pmset repeat wakeorpoweron {days} {hour:02d}:{minute:02d}:00"


def wake_note(zone: str = "America/New_York") -> str:
    """The optional pmset wake before the briefing, as text for the owner (never run)."""

    return (
        "note: launchd runs a job missed while the Mac sleeps when it next wakes, which may be "
        "after 09:30 New York (then only a late preview is written). An optional wake, five "
        "minutes before (an administrator setting; pmset keeps one repeating pair, a power-on "
        "and a power-off event, so this may replace a repeating schedule already set, the "
        "evening agent's wake included: keep that one, since a late briefing can still be read "
        f"but a missed evening cannot be recorded): {_pmset(zone, WAKE_TIME)}. Check it before "
        "and after with: pmset -g sched\n"
    )


def evening_wake_note(zone: str = "America/New_York") -> str:
    """The optional pmset wake before the evening agent, as text for the owner (never run)."""

    return (
        "note: launchd runs this agent only while you are logged in (asleep is fine, shut down "
        "or logged out is not) and the Mac is awake. A run missed while the Mac sleeps starts "
        "when it next wakes, and still records that evening if it starts before the next "
        "09:30 New York open; after that, the evening is a permanent gap in the prospective "
        "record. An optional wake, five minutes before (an administrator setting; pmset keeps "
        "one repeating pair, a power-on and a power-off event, so this replaces a repeating "
        "schedule already set, the briefing's 08:25 wake included: keep this one, since a late "
        f"briefing can still be read but a missed evening cannot be recorded): "
        f"{_pmset(zone, EVENING_WAKE)}. Check it before and after with: pmset -g sched. The Mac "
        "must stay awake until 20:30 New York: a laptop should stay on power, and check the "
        "first evening's log.\n"
    )


def _quoted(text: str) -> str:
    """`text` as one zsh word (single quotes; a single quote inside is closed and escaped)."""

    return "'" + text.replace("'", "'\\''") + "'"


def render_run_sh(model: str, project: Path = PROJECT) -> str:
    """
    The evening agent's run.sh for the activated `model` (its full 64-hex id). ValueError for
    anything else; whether the model is activated is checked when the job runs.
    """

    if len(model) != 64 or not set(model) <= HEX:
        raise ValueError("--model takes the activated model's full id (64 lowercase hex)")
    kept = " ".join(f'"{name}=${{{name}:-}}"' for name in KEPT_VARIABLES)
    stamp = "$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)"
    return f"""#!/bin/zsh -f
# Stock Lab: the unattended evening, started by launchd at 20:30 New York on weekdays
# ({LABEL}). Printed by scripts/stock_lab_schedule.py --run-sh: to change it, print it
# again.
#   1. The recent prices: scripts/stock_lab_fetch_recent.py --authorized --store --keychain.
#      The Tiingo key is read from the macOS Keychain inside that process: never an
#      environment variable, a command-line argument, this file or a log.
#   2. Only when step 1 succeeds, the evening's real forecasts for the activated model:
#      scripts/stock_lab_daily.py --record-real --model <id>.
# A failed step 1 stops the run with its exit status and a dated reason on stderr: no
# forecast is recorded that evening, and none is backfilled later. Model Health shows a
# stopped job (job_overdue) and, after the next evening, the missed one (evening_missed).
# zsh -f reads none of the user's startup files, and env -i passes each step only the
# variables named below, so nothing exported in a profile or set in launchd's own
# environment reaches either step.
set -u
umask 077
repo={_quoted(str(project))}
git=/usr/bin/git
python="$repo/.venv/bin/python"
model={model}
say() {{
  print -u2 -r -- "{stamp} stock lab evening $1"
}}
refuse() {{
  say "refused: $1; nothing was run"
  exit {REFUSED_EXIT}
}}
clean=(/usr/bin/env -i {kept} PATH={SYSTEM_PATH} "PYTHONPATH=$repo/src")
# Only a clean main checkout runs: the job is unattended and reads the Tiingo key, so it
# never executes a worktree, a feature branch or uncommitted code. Git runs in the same
# stripped environment (no GIT_DIR or GIT_WORK_TREE from launchd), and lists every
# untracked file whatever the user's git configuration hides.
[[ -d "$repo/.git" ]] || refuse "$repo is not the main checkout (.git is not a folder)"
branch=$("${{clean[@]}}" "$git" -C "$repo" symbolic-ref --quiet --short HEAD) \\
  || refuse "$repo has a detached HEAD, not {MAIN_BRANCH}"
[[ "$branch" == {MAIN_BRANCH} ]] || refuse "$repo is on branch $branch, not {MAIN_BRANCH}"
changes=$("${{clean[@]}}" "$git" -C "$repo" status --porcelain --untracked-files=all \\
  --ignore-submodules=none) || refuse "git status failed in $repo"
[[ -z "$changes" ]] || refuse "$repo has uncommitted or untracked changes (git status --porcelain)"
cd "$repo" || refuse "cannot enter $repo"
# Keep the Mac from idle sleep until this script, and the step it becomes, has exited.
if [[ -x /usr/bin/caffeinate ]]; then
  /usr/bin/caffeinate -i -w $$ </dev/null >/dev/null 2>&1 &!
fi
"${{clean[@]}}" "$python" scripts/stock_lab_fetch_recent.py --authorized --store --keychain
fetched=$?
if (( fetched != 0 )); then
  say "stopped: the price update failed (exit $fetched); no forecast was recorded tonight"
  exit $fetched
fi
exec "${{clean[@]}}" "$python" scripts/stock_lab_daily.py --record-real --model "$model"
"""


EMAIL_DRY_RUN = f"""\
EMAIL DELIVERY OF THE PREMARKET BRIEFING: DRY RUN
Nothing is connected, no credential is read and nothing is sent. No email
adapter exists yet: this describes the one a later, separately approved change
would add, behind the existing DeliveryAdapter interface.

Waiting on the owner, each a separate approval
  1. Provider: not chosen (an SMTP submission server over TLS, or an email
     API; each needs its own review).
  2. Recipient: not set. One address, the owner's own, kept outside Git.
  3. Credential: a macOS Keychain item, not created.
  4. Installation: the 08:30 launchd agent (--briefing), not installed, and
     optionally the 08:25 wake (pmset, an administrator setting), not set.
  5. Data licence: confirm that emailing derived numbers to oneself is
     private use under the market-data vendor's terms.

What it would send
  - The same rendered briefing the local preview writes: its subject line,
    plain text and HTML in one message. No attachment, image, link, script
    or tracking.
  - Real briefings only. The schedule never emails a synthetic demo.

When
  - Only a premarket briefing: prepared on a session morning before 09:30
    New York.
  - The wall clock (never --at, which prepares a past moment) is checked
    again just before connecting: at or after 09:30 New York on the market
    date nothing is sent, and the attempt is recorded as late. A Mac that
    wakes late writes a labelled late preview only.
  - At most once per market date: the existing duplicate-send log, on its
    own "email" channel, claimed before connecting. An attempt left
    unfinished blocks every later one until the owner checks the log.
  - Weekends and market holidays are skipped by the NYSE calendar.

The credential, at send time only
  - Read from the macOS Keychain just before connecting, with
        security find-generic-password -s {KEYCHAIN_SERVICE} -a {KEYCHAIN_ACCOUNT} -w
    run without a shell; its output stays in memory for that one connection.
    The account is a generic label, never the address (command lines are
    visible to other processes).
  - Never in Git or the repository's files, an environment variable, a
    command-line argument, a log, the delivery log, a preview file or an
    error message. A failure records only the error's type.
  - Added by the owner once, typed at a prompt so it never reaches the
    shell history (nothing follows -w):
        security add-generic-password -s {KEYCHAIN_SERVICE} -a {KEYCHAIN_ACCOUNT} -U -w
  - An item the security tool adds can be read again by that tool without a
    prompt, so any program running as this user, an agent's shell included,
    could read it. Before an adapter exists, the owner decides how to narrow
    that (for example a keychain of its own, or a command guard that refuses
    "security find-generic-password" and "dump-keychain" to agents); a guard
    is a guardrail change and needs its own approval.
  - Public CI never holds it: the repository is public.

What the adapter must never do
  - Print or record the address, the password or the server's replies.
  - Send twice, send after 09:30 New York, or send to more than one address.
  - Send anything but the rendered briefing.

Offline tests it needs first (a fake transport, no network)
  - the secret scan over every output, the 09:30 refusal on the wall clock,
    the duplicate-send log on the email channel, and no network or mail
    module imported anywhere else in the briefing code.

Nothing above has run.
"""


def main(
    argv: list[str] | None = None, *, out: TextIO = sys.stdout, err: TextIO = sys.stderr
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--briefing", action="store_true")
    choice.add_argument("--email-dry-run", action="store_true")
    choice.add_argument("--run-sh", action="store_true")
    parser.add_argument("--zone", default="America/New_York")
    parser.add_argument("--model", metavar="ID")
    args = parser.parse_args(argv)
    if (args.model is not None) != args.run_sh:
        print("refused: --run-sh needs --model ID, and --model is for --run-sh only", file=err)
        return 2
    if args.email_dry_run:
        out.write(EMAIL_DRY_RUN)
        return 0
    try:
        if args.run_sh:
            out.write(render_run_sh(args.model))
            return 0
        if args.briefing:
            printed, note = briefing_plist(args.zone), wake_note(args.zone)
        else:
            printed, note = plist(args.zone), evening_wake_note(args.zone)
    except ZoneInfoNotFoundError:
        print(f"refused: {args.zone!r} is not an IANA time zone", file=err)
        return 2
    except ValueError as error:
        print(f"refused: {error}", file=err)
        return 2
    out.write(printed)
    err.write(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
