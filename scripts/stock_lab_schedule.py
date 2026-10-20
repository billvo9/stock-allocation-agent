"""
Print a macOS launchd agent for the Stock Lab, or describe an email delivery
as a dry run (owner steps; nothing here installs, loads, connects or sends
anything, and no credential is read).

Modes
    (default)        the daily job: 20:30 New York time on weekdays,
                     scripts/stock_lab_daily.py. The job itself decides which
                     session is expected and skips holidays from the NYSE
                     calendar, so a run on a holiday records "nothing new"
                     rather than inventing a session.
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

The agent runs the code of the checkout it was printed from: print it from
a clean checkout of main, never from a working branch or worktree.

To install, after reading the printed file and approving the installation
(owner, once per agent):
    mkdir -p -m 700 ~/stock-agent-private ~/stock-agent-private/stock_lab ~/stock-agent-private/stock_lab/logs
    .venv/bin/python scripts/stock_lab_schedule.py > ~/Library/LaunchAgents/local.stock-lab.daily.plist
    launchctl load ~/Library/LaunchAgents/local.stock-lab.daily.plist
    .venv/bin/python scripts/stock_lab_schedule.py --briefing > ~/Library/LaunchAgents/local.stock-lab.briefing.plist
    launchctl load ~/Library/LaunchAgents/local.stock-lab.briefing.plist
To remove one: launchctl unload the same file, then delete it.

Logs go to ~/stock-agent-private/stock_lab/logs/ (private, outside Git) and
keep each run's printed text; both agents write with a private umask (files
0600). An agent installed from an earlier printout keeps its old settings
until it is printed again, unloaded and loaded. When the Mac is asleep at the
scheduled time, launchd runs the job when it next wakes, so a sleeping Mac
delivers late or not at all (docs/research/stock_probability_lab.md, section
9, has the limits and the recommended hosted option).

Usage:
    .venv/bin/python scripts/stock_lab_schedule.py [--zone America/Los_Angeles]
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


def _plist(
    label: str,
    when: tuple[int, int, int],
    arguments: list[str],
    log: str,
    home: Path | None,
) -> str:
    home = home or Path.home()
    logs = home / "stock-agent-private" / "stock_lab" / "logs"
    hour, minute, shift = when
    python = PROJECT / ".venv" / "bin" / "python"
    program = "".join(
        f"\n    <string>{escape(str(part))}</string>" for part in [python, *arguments]
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
  <array>{program}
  </array>
  <key>WorkingDirectory</key><string>{escape(str(PROJECT))}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key><string>{escape(str(PROJECT / "src"))}</string>
  </dict>
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
    """The daily job's agent: 20:30 New York on weekdays."""

    script = PROJECT / "scripts" / "stock_lab_daily.py"
    return _plist(LABEL, fixed_local_time(zone, NEW_YORK_TIME), [str(script)], "daily", home)


def briefing_plist(zone: str = "America/New_York", home: Path | None = None) -> str:
    """The premarket briefing's agent: 08:30 New York on weekdays, writing the local preview."""

    script = PROJECT / "scripts" / "stock_lab_briefing.py"
    when = fixed_local_time(zone, BRIEFING_TIME)
    arguments = [str(script), "--preview"]
    return _plist(BRIEFING_LABEL, when, arguments, "briefing", home)


def wake_note(zone: str = "America/New_York") -> str:
    """The optional pmset wake before the briefing, as text for the owner (never run)."""

    hour, minute, shift = fixed_local_time(zone, WAKE_TIME)
    days = "".join(PMSET_DAYS[(day + shift) % 7] for day in range(5))
    return (
        "note: launchd runs a job missed while the Mac sleeps when it next wakes, which may be "
        "after 09:30 New York (then only a late preview is written). An optional wake, five "
        "minutes before (an administrator setting; pmset keeps one repeating pair, a power-on "
        "and a power-off event, so this may replace a repeating schedule already set): sudo "
        f"pmset repeat wakeorpoweron {days} {hour:02d}:{minute:02d}:00. Check it before and "
        "after with: pmset -g sched\n"
    )


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
    parser.add_argument("--zone", default="America/New_York")
    args = parser.parse_args(argv)
    if args.email_dry_run:
        out.write(EMAIL_DRY_RUN)
        return 0
    try:
        if not args.briefing:
            out.write(plist(args.zone))
            return 0
        printed, note = briefing_plist(args.zone), wake_note(args.zone)
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
