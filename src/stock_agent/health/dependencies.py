"""
Dependency health: every `==` pin in requirements.txt is installed at exactly
that version, and `pip check` finds no broken requirement.

Unpinned requirements are listed as facts, not judged: several are unpinned
by owner decision (for example pyarrow, "Allow 25.0.1, no pin"), and pinning
them is an owner approval gate. The installed-version lookup and the command
runner are injected, so tests never inspect the real environment.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from stock_agent.health.model import Area, Better, Check, HealthState

_LINE = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)")
_PIN = re.compile(r"==\s*([A-Za-z0-9._+!-]+)")


def normalize(name: str) -> str:
    """PEP 503 name normalization (`Scikit_Learn` -> `scikit-learn`)."""

    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Requirement:
    name: str
    pin: str | None
    line: str


def parse_requirements(text: str) -> tuple[Requirement, ...]:
    requirements = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            continue
        specifier = match.group(3).split(";", 1)[0].strip()
        pin = _PIN.fullmatch(specifier)
        requirements.append(
            Requirement(normalize(match.group(1)), pin.group(1) if pin else None, line)
        )
    return tuple(requirements)


VersionLookup = Callable[[str], str | None]


def pins_check(
    requirements_text: str | None,
    *,
    installed: VersionLookup,
    interpreter: str,
    measured_at: datetime,
    evidence: str,
) -> Check:
    """`deps.pins`: every pinned requirement is installed at its pin."""

    common = {
        "check_id": "deps.pins",
        "area": Area.DEPS,
        "title": "Pinned dependencies",
        "measured_at": measured_at,
        "threshold": "every == pin installed at exactly its version",
        "evidence": evidence,
    }
    if requirements_text is None:
        return Check(
            state=HealthState.UNAVAILABLE,
            reason="no_requirements",
            value=None,
            unit=None,
            summary="requirements.txt could not be read.",
            detail=f"No requirements file at {evidence}.",
            **common,
        )
    requirements = parse_requirements(requirements_text)
    pinned = [item for item in requirements if item.pin is not None]
    unpinned = sorted(item.name for item in requirements if item.pin is None)
    if not pinned:
        return Check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_pins",
            value=None,
            unit=None,
            summary="requirements.txt pins nothing, so pins cannot be checked.",
            detail=f"unpinned: {', '.join(unpinned) or 'none'}",
            **common,
        )
    found = {item.name: installed(item.name) for item in pinned}
    wrong = [item for item in pinned if found[item.name] != item.pin]
    listing = "; ".join(
        f"{item.name}=={item.pin} (installed {found[item.name] or 'missing'})" for item in pinned
    )
    detail = (
        f"interpreter {interpreter}; pinned: {listing}; "
        f"unpinned (not judged): {', '.join(unpinned) or 'none'}"
    )
    if wrong:
        names = ", ".join(item.name for item in wrong)
        state, reason = HealthState.FAILED, "pin_mismatch"
        summary = f"{len(wrong)} pinned packages are not installed at their pin: {names}."
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"All {len(pinned)} pinned packages are installed at their pinned versions."
    return Check(
        state=state,
        reason=reason,
        value=len(pinned) - len(wrong),
        unit=f"of {len(pinned)} pins installed at the pinned version",
        better=Better.HIGHER,
        tolerance=0,
        summary=summary,
        detail=detail,
        **common,
    )


Runner = Callable[..., subprocess.CompletedProcess]


def pip_check(
    *,
    python: str,
    display: str | None = None,
    runner: Runner = subprocess.run,
    measured_at: datetime,
    evidence: str,
) -> tuple[Check, str]:
    """
    `deps.pip_check` and the command output (written as evidence by the
    caller). `display` is how the interpreter is named in the detail.
    """

    command = [python, "-m", "pip", "check"]
    shown = " ".join([display or python, *command[1:]])
    common = {
        "check_id": "deps.pip_check",
        "area": Area.DEPS,
        "title": "Dependency consistency (pip check)",
        "measured_at": measured_at,
        "unit": "broken requirements reported",
        "threshold": "0",
        "evidence": evidence,
    }
    try:
        completed = runner(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        output = f"{shown}: {type(error).__name__}: {error}"
        return (
            Check(
                state=HealthState.UNAVAILABLE,
                reason="pip_unavailable",
                value=None,
                summary="pip check could not be run.",
                detail=output,
                **common,
            ),
            output,
        )
    output = (completed.stdout or "") + (completed.stderr or "")
    problems = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if completed.returncode == 0:
        check = Check(
            state=HealthState.HEALTHY,
            value=0,
            better=Better.LOWER,
            tolerance=0,
            summary="pip check found no broken requirements.",
            detail=f"{shown} exited 0: {output.strip() or 'no output'}",
            **common,
        )
    else:
        check = Check(
            state=HealthState.FAILED,
            reason="broken_requirements",
            value=len(problems),
            better=Better.LOWER,
            tolerance=0,
            summary=f"pip check reported {len(problems)} broken requirements.",
            detail=f"{shown} exited {completed.returncode}: {output.strip()[:2000]}",
            **common,
        )
    return check, output
