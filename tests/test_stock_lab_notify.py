"""
The Stock Lab's local notifications (stock_lab.notify): a closed set of short,
fixed notices, each shown with `/usr/bin/osascript` through an injectable
runner. A notice holds no value, price, forecast or key, and a notification
that cannot be shown never raises. A fake runner stands in for osascript, so
nothing is shown here.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from stock_agent.stock_lab import notify


class Runner:
    """Records each command and its keyword arguments; fails as told."""

    def __init__(self, fail: BaseException | None = None, returncode: int = 0):
        self.calls: list[tuple[list[str], dict]] = []
        self.fail, self.returncode = fail, returncode

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if self.fail is not None:
            raise self.fail
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def test_a_notice_is_shown_by_osascript_with_the_text_as_an_argument_never_as_code():
    runner = Runner()
    assert notify.notify("evening_not_recorded", runner=runner) is True
    ((command, kwargs),) = runner.calls
    assert command == [
        "/usr/bin/osascript",
        "-e",
        "on run argv",
        "-e",
        'display notification (item 1 of argv) with title "Stock Lab"',
        "-e",
        "end run",
        notify.NOTICES["evening_not_recorded"],
    ]
    assert command == notify.command(notify.NOTICES["evening_not_recorded"])
    # Bounded, silent, never interactive, and with none of the caller's variables.
    assert kwargs["timeout"] == notify.TIMEOUT and kwargs["check"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["capture_output"] is True
    assert set(kwargs["env"]) <= {"HOME", "USER", "LOGNAME", "TMPDIR", "PATH"}
    assert kwargs["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


def test_the_callers_variables_never_reach_the_notifier(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "planted-secret-value")
    monkeypatch.setenv("HOME", "/h")
    runner = Runner()
    notify.notify("check_failed", runner=runner)
    ((command, kwargs),) = runner.calls
    assert "TIINGO_API_KEY" not in kwargs["env"] and kwargs["env"]["HOME"] == "/h"
    assert not [part for part in command if "planted-secret-value" in part]


@pytest.mark.parametrize(
    "runner",
    [
        Runner(fail=FileNotFoundError("/usr/bin/osascript")),
        Runner(fail=PermissionError("denied")),
        Runner(fail=subprocess.TimeoutExpired("osascript", notify.TIMEOUT)),
        Runner(fail=subprocess.SubprocessError("broken")),
        Runner(returncode=1),
    ],
    ids=["missing", "not-permitted", "timed-out", "subprocess-error", "non-zero"],
)
def test_a_notification_that_cannot_be_shown_returns_false_and_never_raises(runner):
    assert notify.notify("evening_missed", runner=runner) is False
    assert len(runner.calls) == 1


def test_only_a_named_notice_can_be_shown_so_no_free_text_reaches_it():
    runner = Runner()
    with pytest.raises(ValueError, match="no such notice"):
        notify.notify("tonight AMD closed at 123.45", runner=runner)
    assert runner.calls == []


def test_every_notice_is_short_plain_ascii_and_holds_no_number_but_the_open():
    assert set(notify.NOTICES) == {
        "evening_not_recorded",
        "evening_missed",
        "check_failed",
        "on_battery",
        "test",
    }
    for key, text in notify.NOTICES.items():
        assert text.isascii() and text.isprintable() and len(text) <= 90, key
        # No price, probability, count, date or key: the only number is the 09:30 open.
        assert not re.search(r"\d", text.replace("09:30", "")), key
        assert not text.startswith("Stock Lab"), key  # the title says it once
    assert "before 09:30" in notify.NOTICES["evening_not_recorded"]
    assert "can still be recorded" in notify.NOTICES["evening_not_recorded"]
