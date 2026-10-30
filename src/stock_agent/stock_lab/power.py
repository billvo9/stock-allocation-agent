"""
The Mac's power source, read-only, for the unattended evening
(scripts/stock_lab_fetch_recent.py --unattended).

The evening's `caffeinate -s` (run.sh) keeps the Mac from sleeping only while
it runs on AC power. On battery, macOS may still put it to sleep in the middle
of a run. On 2026-10-27 the scheduled wake on battery was only a dark wake,
and the Mac fell back asleep between the price update and the forecasts. On
battery, the evening warns and notifies, then carries on.

`source()` runs `/usr/bin/pmset -g batt` without a shell, in a stripped
environment, and reads its first line: "Now drawing from 'AC Power'" or
"Now drawing from 'Battery Power'". It reads and never changes a setting.
Anything else is UNKNOWN, which never warns: no pmset, a failure, a timeout,
another source (a UPS).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

PMSET = "/usr/bin/pmset"
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
TIMEOUT = 10  # seconds; pmset answers at once
AC = "ac"
BATTERY = "battery"
UNKNOWN = "unknown"
_DRAWING = {"Now drawing from 'AC Power'": AC, "Now drawing from 'Battery Power'": BATTERY}

Runner = Callable[..., subprocess.CompletedProcess]


def source(*, runner: Runner = subprocess.run, program: str = PMSET) -> str:
    """AC, BATTERY or UNKNOWN (module docstring); never raises."""

    try:
        done = runner(
            [program, "-g", "batt"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
            env={"PATH": SYSTEM_PATH},
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if done.returncode != 0:
        return UNKNOWN
    first = (done.stdout or "").partition("\n")[0].strip()
    return _DRAWING.get(first, UNKNOWN)
