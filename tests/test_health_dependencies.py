"""
Dependency health: requirement parsing, pins installed at their pin (with
unpinned requirements listed, never judged) and `pip check` through a fake
runner.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.health.dependencies import parse_requirements, pins_check, pip_check
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
