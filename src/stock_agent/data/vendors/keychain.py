"""
A vendor key read from the macOS Keychain at run time, for unattended owner
jobs (the Stock Lab's scheduled evening run).

The key is read just before it is needed, with

    /usr/bin/security find-generic-password -s <service> -a <account> -w

run without a shell through an injectable runner (tests use a fake; nothing
here runs at import). Only the item's service and account labels, which are
not secrets, are on the command line. The key reaches this process through
the tool's standard output only: never an environment variable, a
command-line argument, a file, a log or a message.

Fail closed. Anything but one usable value refuses with a `KeychainRefused`
whose message is built from fixed text, the item's labels and the tool's exit
status, never from its output: a missing item, a locked keychain or an item
that would need a prompt (an unattended job has no one to answer it), a
missing tool (not macOS), a timeout, an empty or unprintable value. The
refusal is raised outside the except block, so no chained exception carries
the tool's output.

Who else can read it. An item the `security` tool adds can be read again by
that tool without a prompt, so any program running as the owner can read it.
Narrowing that is an owner decision (docs/research/stock_probability_lab.md).

Tiingo's item: service TIINGO_SERVICE, account TIINGO_ACCOUNT (a generic label).
The owner adds it once, typed at a prompt so it never reaches the shell history
(nothing follows -w):

    security add-generic-password -s stock-agent-tiingo -a stock-lab -U -w
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from enum import StrEnum

SECURITY = "/usr/bin/security"
TIINGO_SERVICE = "stock-agent-tiingo"
TIINGO_ACCOUNT = "stock-lab"
TIMEOUT_SECONDS = 30.0
_VALUE = re.compile(r"[\x21-\x7e]{1,256}")  # printable ASCII, no spaces or line breaks
# security's exit status is the low byte of the Security framework's error code.
ITEM_NOT_FOUND = 44  # errSecItemNotFound (-25300)
INTERACTION_NOT_ALLOWED = 36  # errSecInteractionNotAllowed (-25308): locked, or needs a prompt
AUTH_FAILED = 51  # errSecAuthFailed (-25293)

Runner = Callable[..., subprocess.CompletedProcess]


class KeychainFailure(StrEnum):
    ITEM_NOT_FOUND = "item_not_found"
    NOT_ALLOWED = "interaction_not_allowed"
    TOOL_MISSING = "tool_missing"
    TOOL_FAILED = "tool_failed"
    TIMED_OUT = "timed_out"
    EMPTY = "empty"
    UNUSABLE = "unusable_value"


DETAILS = {
    KeychainFailure.ITEM_NOT_FOUND: "no such item in the keychain search list",
    KeychainFailure.NOT_ALLOWED: (
        "the keychain is locked, or the item would need a prompt that an unattended job "
        "cannot answer"
    ),
    KeychainFailure.TOOL_MISSING: f"{SECURITY} could not be started (macOS only)",
    KeychainFailure.TOOL_FAILED: "the security tool failed",
    KeychainFailure.TIMED_OUT: (
        "the security tool did not answer in time (a locked keychain or an access prompt may be "
        "waiting on screen for an answer nobody gives)"
    ),
    KeychainFailure.EMPTY: "the item holds no value",
    KeychainFailure.UNUSABLE: "the value is not printable ASCII without spaces",
}


def _value(output: object) -> tuple[str, KeychainFailure | None]:
    """The tool's output without its line break, or why it is unusable (never raises)."""

    if not isinstance(output, bytes):
        return "", KeychainFailure.EMPTY
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError:
        return "", KeychainFailure.UNUSABLE
    value = text.removesuffix("\n")  # the tool ends the value with one line break
    if not value:
        return "", KeychainFailure.EMPTY
    if not _VALUE.fullmatch(value):
        return "", KeychainFailure.UNUSABLE
    return value, None


class KeychainRefused(RuntimeError):
    """The key could not be read; the message never holds the tool's output."""

    def __init__(
        self, reason: KeychainFailure, *, service: str, account: str, status: int | None = None
    ) -> None:
        exit_text = "" if status is None else f", exit {status}"
        super().__init__(
            f"{reason.value}: {DETAILS[reason]} (service {service}, account {account}{exit_text})"
        )
        self.reason = reason
        self.service = service
        self.account = account
        self.status = status


def add_command(service: str = TIINGO_SERVICE, account: str = TIINGO_ACCOUNT) -> str:
    """The owner's command to add or replace the item, typed at a prompt (nothing after -w)."""

    return f"security add-generic-password -s {service} -a {account} -U -w"


def read_key(
    service: str = TIINGO_SERVICE,
    account: str = TIINGO_ACCOUNT,
    *,
    runner: Runner = subprocess.run,
    timeout: float = TIMEOUT_SECONDS,
) -> str:
    """The item's value, or KeychainRefused (module docstring)."""

    failure: KeychainFailure | None = None
    status: int | None = None
    completed = None
    try:
        completed = runner(
            [SECURITY, "find-generic-password", "-s", service, "-a", account, "-w"],
            stdin=subprocess.DEVNULL,
            capture_output=True,  # bytes, decoded as ASCII by _value
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        failure = KeychainFailure.TIMED_OUT
    except OSError:
        failure = KeychainFailure.TOOL_MISSING
    if failure is None:
        status = completed.returncode
        if status != 0:
            failure = {
                ITEM_NOT_FOUND: KeychainFailure.ITEM_NOT_FOUND,
                INTERACTION_NOT_ALLOWED: KeychainFailure.NOT_ALLOWED,
                AUTH_FAILED: KeychainFailure.NOT_ALLOWED,
            }.get(status, KeychainFailure.TOOL_FAILED)
    value = ""
    if failure is None:
        value, failure = _value(completed.stdout)  # "" with any failure
        status = None
    del completed  # the tool's output stays out of the locals of a refusal's traceback
    if failure is not None:  # raised outside the except blocks: no chained cause
        raise KeychainRefused(failure, service=service, account=account, status=status)
    return value
