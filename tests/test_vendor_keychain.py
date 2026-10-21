"""
The run-time key reader (stock_agent.data.vendors.keychain) on a fake runner:
it runs only `/usr/bin/security find-generic-password -s ... -a ... -w`
without a shell, returns the value without its line break, and refuses every
other outcome with a reason built from fixed text, never from the tool's
output, and with no chained exception. No test runs the real tool.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from stock_agent.data.vendors import keychain
from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused, read_key

MODULE = Path(keychain.__file__)
VALUE = "fixture-key-not-real-0123456789abcdef"
PLANTED = "planted-output-never-shown"


class FakeSecurity:
    """Stands in for subprocess.run: records each call and answers as configured."""

    def __init__(self, *, code=0, stdout=f"{VALUE}\n", stderr="", raises=None):
        self.code, self.stdout, self.stderr, self.raises = code, stdout, stderr, raises
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.raises is not None:
            raise self.raises
        out, err = (
            text.encode() if isinstance(text, str) else text for text in (self.stdout, self.stderr)
        )
        return subprocess.CompletedProcess(argv, self.code, out, err)


def test_it_reads_the_named_item_without_a_shell_and_strips_one_line_break():
    fake = FakeSecurity()
    assert read_key(runner=fake) == VALUE
    ((argv, kwargs),) = fake.calls
    assert argv == ["/usr/bin/security", "find-generic-password", "-s", "stock-agent-tiingo",
                    "-a", "stock-lab", "-w"]  # fmt: skip
    assert "shell" not in kwargs and kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["capture_output"] is True and kwargs["check"] is False
    assert "text" not in kwargs and "encoding" not in kwargs  # bytes, decoded as ASCII here
    assert kwargs["timeout"] == keychain.TIMEOUT_SECONDS
    assert "env" not in kwargs  # the caller's (already stripped) environment, nothing added
    assert read_key("svc", "acct", runner=FakeSecurity(stdout=VALUE)) == VALUE  # no line break


@pytest.mark.parametrize(
    ("fake", "reason", "status"),
    [
        (FakeSecurity(code=44, stdout=PLANTED, stderr=PLANTED), KeychainFailure.ITEM_NOT_FOUND, 44),
        (FakeSecurity(code=36, stderr=PLANTED), KeychainFailure.NOT_ALLOWED, 36),
        (FakeSecurity(code=51), KeychainFailure.NOT_ALLOWED, 51),
        (FakeSecurity(code=1, stdout=f"{PLANTED}\n"), KeychainFailure.TOOL_FAILED, 1),
        (FakeSecurity(raises=FileNotFoundError(PLANTED)), KeychainFailure.TOOL_MISSING, None),
        (FakeSecurity(raises=PermissionError(PLANTED)), KeychainFailure.TOOL_MISSING, None),
        (
            FakeSecurity(raises=subprocess.TimeoutExpired(["security"], 30, output=PLANTED)),
            KeychainFailure.TIMED_OUT,
            None,
        ),
        (FakeSecurity(stdout=""), KeychainFailure.EMPTY, None),
        (FakeSecurity(stdout="\n"), KeychainFailure.EMPTY, None),
        (FakeSecurity(stdout=None), KeychainFailure.EMPTY, None),
        (FakeSecurity(stdout=f"{PLANTED} with spaces\n"), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout=f"{PLANTED}\n{PLANTED}\n"), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout=f"{PLANTED}\n\n"), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout="k" * 257), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout="clé"), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout=b"\xff\xfe" + PLANTED.encode()), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout=f"{PLANTED}\r\n"), KeychainFailure.UNUSABLE, None),
        (FakeSecurity(stdout=f"{PLANTED} \n"), KeychainFailure.UNUSABLE, None),
    ],
    ids=lambda value: value.value if isinstance(value, KeychainFailure) else None,
)
def test_anything_but_one_usable_value_fails_closed_without_the_output(fake, reason, status):
    with pytest.raises(KeychainRefused) as refused:
        read_key(runner=fake)
    error = refused.value
    assert (error.reason, error.status) == (reason, status)
    assert (error.service, error.account) == ("stock-agent-tiingo", "stock-lab")
    message = str(error)
    assert message.startswith(f"{reason.value}: ")
    assert "service stock-agent-tiingo, account stock-lab" in message
    assert PLANTED not in message and PLANTED not in repr(error)
    # Raised outside the except block: no chained exception carries the tool's output.
    assert error.__cause__ is None and error.__context__ is None
    # Nor does any frame of its traceback: a key with a stray space or CR is close to the real
    # value, and a locals-capturing traceback (a debugger, a crash reporter) would show it.
    traceback = error.__traceback__
    while traceback is not None:
        assert PLANTED not in repr(traceback.tb_frame.f_locals), traceback.tb_frame.f_code
        traceback = traceback.tb_next


def test_the_add_command_is_typed_at_a_prompt_and_names_only_the_labels():
    command = keychain.add_command()
    assert command == "security add-generic-password -s stock-agent-tiingo -a stock-lab -U -w"
    assert command.endswith("-U -w")  # nothing follows -w: never in the shell history
    assert keychain.add_command() in MODULE.read_text(encoding="utf-8")  # the docstring agrees


def test_the_module_uses_only_the_standard_library_and_runs_nothing_at_import():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names), roots
    assert not roots & {"urllib", "http", "socket", "os"}  # no network, no environment
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert calls == []  # nothing runs at import: only definitions
