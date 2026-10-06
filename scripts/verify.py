#!/usr/bin/env python3
"""
Canonical quality gates for this repository, used locally and in CI.

    python3 scripts/verify.py

Every gate runs with the tools of the project virtual environment
(`<repo>/.venv/bin`), whichever Python launched this script, so an unrelated
interpreter on PATH (for example Anaconda) can never stand in for the project
environment. A missing `.venv` or the wrong Python version fails loudly; there
is no fallback to another interpreter.

Gates (AGENTS.md), all run even when an earlier one fails:

    pytest -q
    python -m pytest -q
    ruff format --check src tests scripts .claude/hooks
    ruff check src tests scripts .claude/hooks

Exit status: 0 every gate passed, 1 a gate failed, 2 unusable environment.

Read-only: never formats, fixes, installs, or edits anything.
Standard library only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

REQUIRED_PYTHON = (3, 12)

LINT_PATHS = ("src", "tests", "scripts", ".claude/hooks")

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_BAD_ENVIRONMENT = 2

Runner = Callable[..., int]
VersionProbe = Callable[[Path], tuple[int, int]]


@dataclass(frozen=True)
class Gate:
    name: str
    argv: tuple[str, ...]


def repo_root() -> Path:
    """The repository root: this file lives in `<root>/scripts/`."""

    return Path(__file__).resolve().parent.parent


def venv_bin(root: Path) -> Path:
    return root / ".venv" / "bin"


def missing_tools(bin_dir: Path) -> list[str]:
    """Names of required venv executables that are absent or not executable."""

    return [
        name
        for name in ("python", "pytest", "ruff")
        if not os.access(bin_dir / name, os.X_OK) or not (bin_dir / name).is_file()
    ]


def build_gates(bin_dir: Path) -> list[Gate]:
    python = str(bin_dir / "python")
    pytest = str(bin_dir / "pytest")
    ruff = str(bin_dir / "ruff")

    return [
        Gate("pytest -q", (pytest, "-q")),
        Gate("python -m pytest -q", (python, "-m", "pytest", "-q")),
        Gate(
            "ruff format --check " + " ".join(LINT_PATHS),
            (ruff, "format", "--check", *LINT_PATHS),
        ),
        Gate("ruff check " + " ".join(LINT_PATHS), (ruff, "check", *LINT_PATHS)),
    ]


def gate_environment(base: Mapping[str, str], bin_dir: Path) -> dict[str, str]:
    """
    Environment equivalent to an activated venv.

    Subprocesses spawned by the gates (and by tests) resolve `python`,
    `pytest`, and `ruff` to the venv first. PYTHONHOME is dropped because it
    would redirect the venv interpreter to another installation.
    """

    env = dict(base)
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(bin_dir.parent)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")]).rstrip(os.pathsep)
    return env


def probe_python_version(python: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        [str(python), "-c", "import sys; print(*sys.version_info[:2])"],
        text=True,
    )
    major, minor = output.split()
    return int(major), int(minor)


def _display(argv: Sequence[str], root: Path) -> str:
    """Show venv executables relative to the repo root for readable logs."""

    shown = []
    for arg in argv:
        path = Path(arg)
        if path.is_absolute() and path.is_relative_to(root):
            shown.append(str(path.relative_to(root)))
        else:
            shown.append(arg)
    return " ".join(shown)


def run(
    *,
    root: Path,
    runner: Runner = subprocess.call,
    probe: VersionProbe = probe_python_version,
    base_env: Mapping[str, str] | None = None,
    out: TextIO = sys.stdout,
) -> int:
    bin_dir = venv_bin(root)

    missing = missing_tools(bin_dir)
    if missing:
        print(
            f"verify: project virtual environment not usable: {bin_dir} is missing "
            f"{', '.join(missing)}.\n"
            "verify: create it with:\n"
            "    python3.12 -m venv .venv\n"
            "    .venv/bin/python -m pip install -r requirements.txt\n"
            "    .venv/bin/python -m pip install -e .\n"
            "verify: refusing to fall back to another interpreter.",
            file=out,
        )
        return EXIT_BAD_ENVIRONMENT

    version = probe(bin_dir / "python")
    if version[:2] != REQUIRED_PYTHON:
        required = ".".join(map(str, REQUIRED_PYTHON))
        found = ".".join(map(str, version))
        print(f"verify: .venv runs Python {found}; this project requires {required}.", file=out)
        return EXIT_BAD_ENVIRONMENT

    env = gate_environment(os.environ if base_env is None else base_env, bin_dir)

    print(f"verify: repo {root}", file=out)
    print(
        f"verify: interpreter {bin_dir / 'python'} (Python {'.'.join(map(str, version))})",
        file=out,
    )
    out.flush()
    for tool in (("pytest", "--version"), ("ruff", "--version")):
        runner([str(bin_dir / tool[0]), tool[1]], cwd=root, env=env)

    results: list[tuple[Gate, int]] = []

    for gate in build_gates(bin_dir):
        print(f"\n==> {_display(gate.argv, root)}", file=out)
        out.flush()
        code = runner(list(gate.argv), cwd=root, env=env)
        print(f"<== exit {code}", file=out)
        results.append((gate, code))

    print("\nverify summary:", file=out)
    for gate, code in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {gate.name}  (exit {code})", file=out)

    if all(code == 0 for _, code in results):
        print("verify: all gates passed", file=out)
        return EXIT_OK

    print("verify: FAILED", file=out)
    return EXIT_GATE_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args:
        print("usage: python3 scripts/verify.py   (takes no arguments)", file=sys.stderr)
        return EXIT_BAD_ENVIRONMENT

    return run(root=repo_root())


if __name__ == "__main__":
    sys.exit(main())
