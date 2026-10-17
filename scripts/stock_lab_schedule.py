"""
Print a macOS launchd agent that runs the Stock Lab's daily job once a day
after the close (owner steps; nothing here installs or loads anything).

The job runs at 20:30 New York time on weekdays (launchd uses the Mac's
local time zone, so the printed hour is converted from New York to the
local zone given, America/New_York by default). The job itself decides which
session is expected and skips holidays from the NYSE calendar, so a run on a
holiday records "nothing new" rather than inventing a session.

To install, after reading the printed file (owner, once):
    mkdir -p ~/stock-agent-private/stock_lab/logs
    .venv/bin/python scripts/stock_lab_schedule.py > ~/Library/LaunchAgents/local.stock-lab.daily.plist
    launchctl load ~/Library/LaunchAgents/local.stock-lab.daily.plist
To remove: launchctl unload the same file, then delete it.

Logs go to ~/stock-agent-private/stock_lab/logs/ (private, outside Git).

Usage:
    .venv/bin/python scripts/stock_lab_schedule.py [--zone America/Los_Angeles]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
LABEL = "local.stock-lab.daily"
NEW_YORK_TIME = (20, 30)


def local_time(zone: str) -> tuple[int, int, int]:
    """
    20:30 New York time in `zone` as (hour, minute, day shift), on a winter
    weekday (launchd has no time zones): a zone east of New York may reach the
    next calendar day, so its weekdays shift by one.
    """

    new_york = datetime(2026, 1, 6, *NEW_YORK_TIME, tzinfo=ZoneInfo("America/New_York"))
    local = new_york.astimezone(ZoneInfo(zone))
    return local.hour, local.minute, (local.date() - new_york.date()).days


def plist(zone: str = "America/New_York", home: Path | None = None) -> str:
    home = home or Path.home()
    logs = home / "stock-agent-private" / "stock_lab" / "logs"
    hour, minute, shift = local_time(zone)
    python = PROJECT / ".venv" / "bin" / "python"
    script = PROJECT / "scripts" / "stock_lab_daily.py"
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
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{escape(str(python))}</string>
    <string>{escape(str(script))}</string>
  </array>
  <key>WorkingDirectory</key><string>{escape(str(PROJECT))}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key><string>{escape(str(PROJECT / "src"))}</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>{days}
  </array>
  <key>StandardOutPath</key><string>{escape(str(logs / "daily.out.log"))}</string>
  <key>StandardErrorPath</key><string>{escape(str(logs / "daily.err.log"))}</string>
</dict>
</plist>
"""


def main(argv: list[str] | None = None, *, out: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--zone", default="America/New_York")
    args = parser.parse_args(argv)
    out.write(plist(args.zone))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
