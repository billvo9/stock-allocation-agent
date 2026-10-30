"""
The Mac's power source for the unattended evening (stock_agent.stock_lab.power):
read-only, from `/usr/bin/pmset -g batt`, through an injected runner. No pmset is
run here and no setting is read or changed.
"""

from __future__ import annotations

import subprocess

import pytest

from stock_agent.stock_lab import power

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining\n"
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t83%; discharging;\n"
DESKTOP = "Now drawing from 'AC Power'\n"  # a Mac without a battery


class Runner:
    def __init__(self, stdout="", returncode=0, error=None):
        self.stdout, self.returncode, self.error, self.calls = stdout, returncode, error, []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "")


@pytest.mark.parametrize(
    "stdout, source",
    [(AC, power.AC), (DESKTOP, power.AC), (BATTERY, power.BATTERY)],
    ids=["laptop-on-ac", "desktop", "laptop-on-battery"],
)
def test_the_source_is_read_from_pmsets_first_line(stdout, source):
    runner = Runner(stdout)
    assert power.source(runner=runner) == source
    ((command, kwargs),) = runner.calls
    assert command == ["/usr/bin/pmset", "-g", "batt"]  # reads; never sets anything
    assert kwargs["timeout"] == power.TIMEOUT and kwargs["check"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["capture_output"] is True
    assert kwargs["env"] == {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}  # nothing of the caller's


@pytest.mark.parametrize(
    "runner",
    [
        Runner(error=FileNotFoundError("pmset")),
        Runner(error=subprocess.TimeoutExpired("pmset", 10)),
        Runner(error=PermissionError("pmset")),
        Runner(BATTERY, returncode=1),
        Runner("Now drawing from 'UPS Power'\n"),
        Runner(""),
        Runner(" -InternalBattery-0 'Battery Power'\nNow drawing from 'AC Power'\n"),
    ],
    ids=["missing", "timeout", "denied", "failed", "ups", "empty", "not-the-first-line"],
)
def test_anything_else_is_unknown_and_never_raises(runner):
    assert power.source(runner=runner) == power.UNKNOWN
