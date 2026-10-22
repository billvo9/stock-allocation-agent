"""
The Stock Lab's morning check (owner-run, or the optional launchd agent; local,
read-only): a local notification when last evening is not recorded yet.

On an NYSE session morning it reads, read-only, which evenings the Stock Lab
store has real forecasts for (stock_lab.morning). When the prior session's
evening has none, it shows one local macOS notification (stock_lab.notify):
before the 09:30 New York open the evening can still be recorded (the runbook's
"Recovering a failed evening"); after it, the evening is a permanent gap (from
20:00 it is still that evening, never tonight's, whose run is still to come).
On a weekend or a market holiday it checks nothing and stays silent. It never
writes to the store, runs
the evening job, reads a credential or sends anything off this Mac, and it
prints dates only, never a price or a forecast value.

scripts/stock_lab_schedule.py --morning-check prints its launchd agent (08:00
New York on weekdays by default); the owner installs it. Nothing here installs
anything.

Options
    --test-notification   show the fixed test notice and exit (the owner's
                          one-time check that notifications reach this Mac);
                          nothing is read

Exit codes: 0 nothing to report (recorded, or nothing to check today); 1 last
evening is not recorded (notified); 2 the check could not run (notified). With
--test-notification: 0 shown, 1 not shown.

Usage:
    .venv/bin/python scripts/stock_lab_morning_check.py
    .venv/bin/python scripts/stock_lab_morning_check.py --test-notification
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TextIO

from stock_agent.calendars.nyse_rules import sessions
from stock_agent.stock_lab import morning, notify
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.store import StoreRefused

EXIT = {
    morning.RECORDED: 0,
    morning.NOT_DUE: 0,
    morning.NOT_RECORDED: 1,
    morning.MISSED: 1,
    morning.FAILED: 2,
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def _check(store_factory: Callable, now: datetime, calendar) -> morning.MorningCheck:
    """The check; any error becomes a failed check, so a crash still notifies."""

    try:
        return morning.morning_check(store_factory(), now=now, calendar=calendar)
    except StoreRefused as error:  # e.g. the store's folder is not private
        detail = f"the check could not run: the record ({error})"
    except Exception as error:  # noqa: BLE001 - never a silent morning; the type only
        detail = f"the check could not run: {type(error).__name__}"
    return morning.MorningCheck(morning.FAILED, "error", None, detail)


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable = ReadOnlyRecords,
    clock: Callable[[], datetime] = utc_now,
    notifier: Callable[[str], bool] = notify.notify,
    out: TextIO = sys.stdout,
    calendar=sessions,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--test-notification", action="store_true")
    args = parser.parse_args(argv)
    now = clock()
    stamp = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.test_notification:
        shown = notifier("test")
        words = (
            "shown"
            if shown
            else "not shown (osascript failed: allow notifications for Script Editor in System "
            "Settings > Notifications)"
        )
        print(f"{stamp} stock lab morning check: the test notification was {words}", file=out)
        return 0 if shown else 1
    found = _check(store_factory, now, calendar)
    line = f"{stamp} stock lab morning check: {found.detail}"
    if found.notice is not None:
        shown = notifier(found.notice)
        line += "; notified" if shown else "; the notification could not be shown"
    print(line, file=out)
    return EXIT[found.status]


if __name__ == "__main__":
    raise SystemExit(main())
