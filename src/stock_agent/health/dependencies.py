"""
Dependency health: every `==` pin in requirements.txt is installed at exactly
that version, and `pip check` finds no broken requirement.

Unpinned requirements are listed as facts, not judged: several are unpinned
by owner decision (for example pyarrow, "Allow 25.0.1, no pin"), and pinning
them is an owner approval gate. The installed-version lookup and the command
runner are injected, so tests never inspect the real environment.

Environment markers decide which lines apply on this platform. Only the PEP
508 subset the requirements use is evaluated: `and`/`or` of comparisons on
`sys_platform`, `platform_machine`, `os_name`, `platform_system` (== and !=)
and `python_version` (any comparison). A line whose marker is outside that
subset is listed as unjudged, never failed or passed. A direct-URL wheel line
(`name @ https://.../name-1.2.3+cpu-....whl#sha256=...`) is pinned by URL, so
its pin is the version in the wheel's file name.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote, urlsplit

from stock_agent.health.model import Area, Better, Check, HealthState

_LINE = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)")
_PIN = re.compile(r"==\s*([A-Za-z0-9._+!-]+)")
_URL = re.compile(r"@\s*(\S+)")
# pip: a comment starts at a line-initial "#" or one preceded by whitespace, so
# a URL fragment such as "#sha256=..." is not a comment.
_COMMENT = re.compile(r"(^|\s)#.*$")
_CLAUSE = re.compile(r"\s*([a-z_]+)\s*(==|!=|<=|>=|<|>)\s*(['\"])([^'\"]*)\3\s*")
_STRING_MARKERS = ("sys_platform", "platform_machine", "os_name", "platform_system")
_VERSION_MARKERS = ("python_version",)


def normalize(name: str) -> str:
    """PEP 503 name normalization (`Scikit_Learn` -> `scikit-learn`)."""

    return re.sub(r"[-_.]+", "-", name).lower()


def current_environment() -> dict[str, str]:
    """The marker values of the running interpreter."""

    return {
        "sys_platform": sys.platform,
        "platform_machine": platform.machine(),
        "os_name": os.name,
        "platform_system": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _version(text: str) -> tuple[int, ...] | None:
    parts = text.split(".")
    return tuple(int(part) for part in parts) if all(p.isdigit() for p in parts) else None


def _clause(text: str, environment: Mapping[str, str]) -> bool | None:
    match = _CLAUSE.fullmatch(text)
    if match is None or match.group(1) not in environment:
        return None
    name, op, value = match.group(1), match.group(2), match.group(4)
    actual = environment[name]
    if name in _STRING_MARKERS and op in ("==", "!="):
        return (actual == value) == (op == "==")
    if name in _VERSION_MARKERS:
        left, right = _version(actual), _version(value)
        if left is None or right is None:
            return None
        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[op]
    return None


def marker_applies(marker: str, environment: Mapping[str, str]) -> bool | None:
    """
    Whether a requirement's marker holds in `environment`: True or False, or
    None when the marker is outside the supported subset (parentheses, other
    variables, string ordering).
    """

    if "(" in marker or ")" in marker:
        return None
    alternatives = []
    for alternative in re.split(r"\s+or\s+", marker.strip()):
        results = [_clause(part, environment) for part in re.split(r"\s+and\s+", alternative)]
        if any(result is None for result in results):
            return None
        alternatives.append(all(results))
    return any(alternatives)


def _wheel_version(url: str, name: str) -> str | None:
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    parts = filename.removesuffix(".whl").split("-")
    if not filename.endswith(".whl") or len(parts) < 5 or normalize(parts[0]) != name:
        return None
    return parts[1]


@dataclass(frozen=True)
class Requirement:
    name: str
    pin: str | None
    line: str
    applies: bool | None = True  # None: the marker is outside the supported subset


def parse_requirements(
    text: str, environment: Mapping[str, str] | None = None
) -> tuple[Requirement, ...]:
    """Every requirement line, with its pin and whether its marker applies here."""

    environment = current_environment() if environment is None else environment
    requirements = []
    for raw in text.splitlines():
        line = _COMMENT.sub("", raw).strip()
        if not line or line.startswith("-"):
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            continue
        specifier, _, marker = match.group(3).partition(";")
        specifier = specifier.strip()
        name = normalize(match.group(1))
        url = _URL.fullmatch(specifier)
        if url is not None:
            pin = _wheel_version(url.group(1), name)
        else:
            exact = _PIN.fullmatch(specifier)
            pin = exact.group(1) if exact else None
        applies = marker_applies(marker, environment) if marker.strip() else True
        requirements.append(Requirement(name, pin, line, applies))
    return tuple(requirements)


VersionLookup = Callable[[str], str | None]


def pins_check(
    requirements_text: str | None,
    *,
    installed: VersionLookup,
    interpreter: str,
    measured_at: datetime,
    evidence: str,
    environment: Mapping[str, str] | None = None,
) -> Check:
    """
    `deps.pins`: every pinned requirement whose marker applies here is
    installed at its pin. Lines for other platforms are skipped; lines whose
    marker cannot be evaluated are listed as unjudged.
    """

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
    parsed = parse_requirements(requirements_text, environment)
    requirements = [item for item in parsed if item.applies is True]
    unjudged = sorted(item.name for item in parsed if item.applies is None)
    pinned = [item for item in requirements if item.pin is not None]
    unpinned = sorted(item.name for item in requirements if item.pin is None)
    if not pinned:
        return Check(
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_pins",
            value=None,
            unit=None,
            summary="requirements.txt pins nothing, so pins cannot be checked.",
            detail=(
                f"unpinned: {', '.join(unpinned) or 'none'}; "
                f"marker not evaluated: {', '.join(unjudged) or 'none'}"
            ),
            **common,
        )
    found = {item.name: installed(item.name) for item in pinned}
    wrong = [item for item in pinned if found[item.name] != item.pin]
    listing = "; ".join(
        f"{item.name}=={item.pin} (installed {found[item.name] or 'missing'})" for item in pinned
    )
    detail = (
        f"interpreter {interpreter}; pinned: {listing}; "
        f"unpinned (not judged): {', '.join(unpinned) or 'none'}; "
        f"marker not evaluated (not judged): {', '.join(unjudged) or 'none'}"
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
