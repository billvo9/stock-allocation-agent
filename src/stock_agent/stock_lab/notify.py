"""
Local macOS notifications for the live Stock Lab (standard library only).

A notification tells the owner that something needs a look, so the logs are
read only after one appears. It is shown on this Mac by `/usr/bin/osascript`
(AppleScript's `display notification`) and is never sent anywhere else.

The notices are a closed set (NOTICES): fixed, short, plain ASCII sentences.
A caller names a notice and cannot pass text of its own, so no secret, price,
probability or forecast value can reach a notification. The text is passed
to osascript as an argument of its run handler, never spliced into the
script.

`notify` never raises for a notification that cannot be shown (osascript
missing, refused, timed out or failing): it returns False, and the caller
carries on, because a notification must never fail the job it reports on.
The runner is injectable, so tests use a fake and nothing is shown.

The evening agent's run.sh shows its own notices the same way, from zsh
(scripts/stock_lab_schedule.py, which imports nothing from this package; a
test keeps the two AppleScript commands identical).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence

OSASCRIPT = "/usr/bin/osascript"
TITLE = "Stock Lab"
# The AppleScript lines, each passed with -e; the notice is the run handler's one argument.
APPLESCRIPT = (
    "on run argv",
    f'display notification (item 1 of argv) with title "{TITLE}"',
    "end run",
)
TIMEOUT = 10  # seconds; displaying a notification returns at once
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
KEPT_VARIABLES = ("HOME", "USER", "LOGNAME", "TMPDIR")  # nothing else reaches osascript

NOTICES = {
    # The morning check (stock_lab.morning).
    "evening_not_recorded": "last evening was not recorded; it can still be recorded before 09:30",
    "evening_missed": "last evening was not recorded, and after the 09:30 open it can no longer be",
    "check_failed": "the morning check could not run - see its log before 09:30",
    # The evening's price update (scripts/stock_lab_fetch_recent.py --unattended): run.sh's
    # caffeinate -s holds the Mac awake on AC power only.
    "on_battery": "running on battery - the Mac may sleep; plug in for reliable evenings",
    # scripts/stock_lab_morning_check.py --test-notification (the owner's one-time setup).
    "test": "test notification - alerts from the Stock Lab reach this Mac",
}

Runner = Callable[..., subprocess.CompletedProcess]


def command(text: str, program: str = OSASCRIPT) -> list[str]:
    """The osascript command that shows `text` under the title TITLE."""

    arguments: list[str] = [program]
    for line in APPLESCRIPT:
        arguments += ["-e", line]
    return [*arguments, text]


def _environment(names: Sequence[str] = KEPT_VARIABLES) -> dict[str, str]:
    kept = {name: os.environ[name] for name in names if name in os.environ}
    return kept | {"PATH": SYSTEM_PATH}


def notify(notice: str, *, runner: Runner = subprocess.run, program: str = OSASCRIPT) -> bool:
    """
    Show the notice named `notice` (a NOTICES key; ValueError for anything else). True when
    osascript reported success; False, never an exception, when it could not be shown.
    """

    if notice not in NOTICES:
        raise ValueError(f"no such notice: {notice!r}")
    try:
        done = runner(
            command(NOTICES[notice], program),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TIMEOUT,
            check=False,
            env=_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0
