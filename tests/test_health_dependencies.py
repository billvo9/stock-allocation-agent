"""
Dependency health: requirement parsing, pins installed at their pin (with
unpinned requirements listed, never judged) and `pip check` through a fake
runner.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.health.dependencies import (
    marker_applies,
    parse_requirements,
    pins_check,
    pip_check,
)
from stock_agent.health.model import HealthState

NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
TEXT = """# comment
pandas
Scikit_Learn==1.9.1
pytest==9.1.1  # pinned for the gates
pyarrow>=25
-e .
extras[fast]==2.0 ; python_version >= "3.12"
"""


def test_parse_requirements_normalizes_names_and_reads_pins():
    parsed = {item.name: item.pin for item in parse_requirements(TEXT)}
    assert parsed == {
        "pandas": None,
        "scikit-learn": "1.9.1",
        "pytest": "9.1.1",
        "pyarrow": None,
        "extras": "2.0",
    }


def test_the_repository_requirements_parse():
    parsed = parse_requirements(REQUIREMENTS.read_text(encoding="utf-8"))
    assert {item.name for item in parsed if item.pin} >= {"pytest", "ruff"}


def pins(installed, text=TEXT):
    return pins_check(
        text,
        installed=installed.get,
        interpreter="/repo/.venv/bin/python",
        measured_at=NOW,
        evidence="requirements.txt",
    )


def test_every_pin_installed_is_healthy_and_unpinned_are_listed():
    check = pins({"scikit-learn": "1.9.1", "pytest": "9.1.1", "extras": "2.0"})
    assert (check.state, check.value, check.unit) == (
        HealthState.HEALTHY,
        3,
        "of 3 pins installed at the pinned version",
    )
    assert "unpinned (not judged): pandas, pyarrow" in check.detail


def test_a_wrong_or_missing_version_fails():
    check = pins({"scikit-learn": "1.9.2", "pytest": "9.1.1"})
    assert (check.state, check.reason, check.value) == (HealthState.FAILED, "pin_mismatch", 1)
    assert "scikit-learn, extras" in check.summary
    assert "scikit-learn==1.9.1 (installed 1.9.2)" in check.detail
    assert "extras==2.0 (installed missing)" in check.detail


def test_no_file_or_no_pins_is_never_healthy():
    assert pins({}, text=None).state is HealthState.UNAVAILABLE
    assert pins({}, text="pandas\nnumpy\n").state is HealthState.INSUFFICIENT_EVIDENCE


def runner(code, stdout="", raises=None):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if raises:
            raise raises
        return subprocess.CompletedProcess(argv, code, stdout, "")

    run.calls = calls
    return run


def test_pip_check_clean_and_broken():
    clean = runner(0, "No broken requirements found.\n")
    check, output = pip_check(python="py", runner=clean, measured_at=NOW, evidence="e")
    assert (check.state, check.value) == (HealthState.HEALTHY, 0)
    assert clean.calls == [["py", "-m", "pip", "check"]]
    assert output == "No broken requirements found.\n"

    broken = runner(1, "a 1.0 has requirement b<2, but you have b 2.1.\nc 3 requires d.\n")
    check, _ = pip_check(python="py", runner=broken, measured_at=NOW, evidence="e")
    assert (check.state, check.reason, check.value) == (
        HealthState.FAILED,
        "broken_requirements",
        2,
    )


def test_pip_that_cannot_run_is_unavailable():
    failing = runner(0, raises=FileNotFoundError("no python"))
    check, output = pip_check(python="py", runner=failing, measured_at=NOW, evidence="e")
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "pip_unavailable")
    assert "FileNotFoundError" in output


LINUX = {
    "sys_platform": "linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "platform_system": "Linux",
    "python_version": "3.12",
}
MAC = LINUX | {"sys_platform": "darwin", "platform_machine": "arm64", "platform_system": "Darwin"}


def repository_pins(environment):
    text = REQUIREMENTS.read_text(encoding="utf-8")
    parsed = parse_requirements(text, environment)
    return text, {item.name: item.pin for item in parsed if item.applies and item.pin}


def test_the_repository_torch_pin_follows_the_platform():
    # The scheduled health run of 2026-10-15 failed on Linux CI: the macOS line
    # torch==2.14.0 was applied there, while CI installs 2.14.0+cpu by URL.
    _, linux = repository_pins(LINUX)
    _, mac = repository_pins(MAC)
    assert linux["torch"] == "2.14.0+cpu" and mac["torch"] == "2.14.0"


def test_the_repository_pins_are_healthy_on_each_supported_platform():
    for environment in (LINUX, MAC):
        text, pinned = repository_pins(environment)
        check = pins_check(
            text,
            installed=pinned.get,
            interpreter="/repo/.venv/bin/python",
            measured_at=NOW,
            evidence="requirements.txt",
            environment=environment,
        )
        assert check.state is HealthState.HEALTHY, (environment["sys_platform"], check.summary)


def test_a_cuda_or_plain_torch_on_linux_still_fails_its_url_pin():
    text, pinned = repository_pins(LINUX)
    check = pins_check(
        text,
        installed=(pinned | {"torch": "2.14.0"}).get,
        interpreter="/repo/.venv/bin/python",
        measured_at=NOW,
        evidence="requirements.txt",
        environment=LINUX,
    )
    assert (check.state, check.reason) == (HealthState.FAILED, "pin_mismatch")
    assert "torch==2.14.0+cpu (installed 2.14.0)" in check.detail


def test_markers_in_the_supported_subset_are_evaluated():
    assert marker_applies('python_version >= "3.12"', LINUX) is True
    assert marker_applies('python_version >= "3.12"', LINUX | {"python_version": "3.11"}) is False
    linux_x86 = 'sys_platform == "linux" and platform_machine == "x86_64"'
    assert marker_applies(linux_x86, LINUX) is True
    assert marker_applies(linux_x86, LINUX | {"platform_machine": "aarch64"}) is False
    assert marker_applies('sys_platform == "darwin" or sys_platform == "linux"', LINUX) is True
    # `and` binds tighter than `or`.
    mixed = 'sys_platform == "win32" and os_name == "nt" or sys_platform == "linux"'
    assert marker_applies(mixed, LINUX) is True


def test_markers_outside_the_subset_are_unjudged_never_passed_or_failed():
    assert marker_applies('(sys_platform == "linux")', LINUX) is None
    assert marker_applies('implementation_name == "cpython"', LINUX) is None
    assert marker_applies('sys_platform > "linux"', LINUX) is None
    text = 'odd==1.0 ; implementation_name == "cpython"\nplain==2.0\n'
    check = pins_check(
        text,
        installed={"plain": "2.0"}.get,
        interpreter="/repo/.venv/bin/python",
        measured_at=NOW,
        evidence="requirements.txt",
        environment=LINUX,
    )
    assert (check.state, check.value) == (HealthState.HEALTHY, 1)
    assert "marker not evaluated (not judged): odd" in check.detail


def test_a_url_that_is_not_a_matching_wheel_is_unpinned():
    text = (
        "torch @ https://example.org/other-1.0-py3-none-any.whl\n"
        "torch2 @ https://example.org/archive.tar.gz\n"
    )
    assert {item.name: item.pin for item in parse_requirements(text, LINUX)} == {
        "torch": None,
        "torch2": None,
    }


def test_a_url_fragment_is_not_a_comment_but_a_spaced_hash_is():
    text = (
        'pkg @ https://example.org/pkg-1.0-py3-none-any.whl#sha256=abc ; sys_platform == "darwin"\n'
        "other==2.0 # a trailing comment\n"
        "#whole-line comment==9\n"
    )
    parsed = {item.name: (item.pin, item.applies) for item in parse_requirements(text, LINUX)}
    assert parsed == {"pkg": ("1.0", False), "other": ("2.0", True)}
