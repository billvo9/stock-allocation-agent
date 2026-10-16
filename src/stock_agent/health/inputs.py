"""
Reading another track's health input (agreed formats v1.0, 2026-10-14):
the common envelope, the newest file per artifact, transition status, and
the worst-of rule that turns several findings into one check state.

Location. A producer writes `<inputs>/<artifact>/<run_id>.json` (locally
`reports/health_inputs/`, ignored by Git). Health reads the newest valid file
per artifact by `generated_at` (ties: the larger `run_id`), or, with
`load_latest_by`, the newest per value of a field (backtest summaries: one
per `policy`, so a run writing several policies is read whole). CI reads only
the synthetic fixtures in `stock_agent.health.fixtures`.

Envelope (every file is one strict-JSON object; extra fields are ignored):

    artifact        the artifact name, equal to its directory name
    schema_version  "1.0"
    generated_at    "YYYY-MM-DDTHH:MM:SS.ffffffZ" (UTC)
    code_sha        a 40-hex git SHA, or null
    synthetic       true for fixture data, false for real data
    run_id          text, equal to the file name without ".json"

Outcomes:

    no directory configured, or no file yet   insufficient_evidence
                                              (input_not_configured / input_missing)
    no file readable                          unavailable (unrecognized_format)
    some files unreadable                     the newest readable file is used, and
                                              the check is at least degraded
                                              (unreadable_inputs)

Transition status. An artifact that depends on portfolio economics carries
`transition_status`: "approved" or "engineering_only_unapproved_transition".
The second is a visible limitation, shown as degraded (`unapproved_transition`),
never as a failure of the run itself.

Synthetic inputs. In the private (real) scope a `synthetic: true` input says
nothing about real training, forecasts or portfolios, so it is capped at
insufficient_evidence (`synthetic_input`): never healthy, while a worse
finding still shows. The CI smoke checks read the synthetic fixtures with
`allow_synthetic=True`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_agent.health.model import (
    PRECEDENCE,
    HealthState,
    SummaryRefused,
    loads_strict,
    parse_timestamp,
)

INPUT_VERSIONS = frozenset({"1.0"})
TRANSITION_APPROVED = "approved"
TRANSITION_UNAPPROVED = "engineering_only_unapproved_transition"
TRANSITION_STATUSES = frozenset({TRANSITION_APPROVED, TRANSITION_UNAPPROVED})
_SHA = re.compile(r"[0-9a-f]{40}")


class FormatError(ValueError):
    """A producer file does not follow its documented format."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    document: dict[str, Any]
    generated_at: datetime
    code_sha: str | None
    synthetic: bool
    run_id: str
    unreadable: tuple[str, ...] = ()

    def describe(self) -> str:
        code = self.code_sha[:12] if self.code_sha else "unknown code"
        kind = "synthetic input" if self.synthetic else "real input"
        text = (
            f"{self.path.name}, run {self.run_id}, generated {self.generated_at:%Y-%m-%d %H:%M} "
            f"UTC from {code}, {kind}"
        )
        if self.unreadable:
            text += f"; unreadable files ignored: {', '.join(self.unreadable[:5])}"
        return text


@dataclass(frozen=True)
class NotRead:
    state: HealthState
    reason: str
    detail: str


def _load(path: Path, artifact: str) -> Artifact:
    try:
        document = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SummaryRefused) as error:
        raise FormatError(str(error)) from error
    if not isinstance(document, dict):
        raise FormatError("the file is not a JSON object")
    if document.get("artifact") != artifact:
        raise FormatError(f"artifact is {document.get('artifact')!r}, expected {artifact!r}")
    if document.get("schema_version") not in INPUT_VERSIONS:
        raise FormatError(f"schema_version {document.get('schema_version')!r} not supported")
    try:
        generated_at = parse_timestamp("generated_at", document.get("generated_at"))
    except SummaryRefused as error:
        raise FormatError(error.detail) from error
    code_sha = document.get("code_sha")
    if code_sha is not None and not (isinstance(code_sha, str) and _SHA.fullmatch(code_sha)):
        raise FormatError(f"code_sha {code_sha!r}")
    synthetic = document.get("synthetic")
    if not isinstance(synthetic, bool):
        raise FormatError("synthetic must be true or false")
    run_id = require(document, "run_id", "text")
    if path.stem != run_id:
        raise FormatError(f"the file name {path.name} differs from run_id {run_id!r}")
    return Artifact(path, document, generated_at, code_sha, synthetic, run_id)


def load_latest(directory: Path | None, artifact: str) -> Artifact | NotRead:
    """The newest readable `<directory>/*.json` of `artifact`, by `generated_at`."""

    if directory is None:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            f"No health-inputs directory was configured for {artifact}.",
        )
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not files:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_missing",
            f"No {artifact} file in {directory} yet; its producer has not written one.",
        )
    readable, unreadable = [], []
    for path in files:
        try:
            readable.append(_load(path, artifact))
        except FormatError as error:
            unreadable.append(f"{path.name} ({error})")
    if not readable:
        return NotRead(
            HealthState.UNAVAILABLE,
            "unrecognized_format",
            f"No {artifact} file in {directory} could be read: {'; '.join(unreadable[:3])}",
        )
    newest = max(readable, key=lambda item: (item.generated_at, item.run_id))
    return Artifact(**{**newest.__dict__, "unreadable": tuple(unreadable)})


def load_latest_by(
    directory: Path | None,
    artifact: str,
    key: str,
    parse: Callable[[Artifact], Any],
) -> dict[str, tuple[Artifact, Any]] | NotRead:
    """
    The newest readable file per value of the text field `key` (for example
    each backtest's `policy`), by `generated_at` then `run_id`, with what
    `parse` returned for it; sorted by that value. A file whose envelope,
    `key` or `parse` fails is unreadable: it is named, every group is at least
    degraded, and when no file is readable the result is unavailable. So
    several producers writing in one run (several policies sharing one
    `generated_at`) are all kept, never reduced to one.
    """

    if directory is None:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            f"No health-inputs directory was configured for {artifact}.",
        )
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not files:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_missing",
            f"No {artifact} file in {directory} yet; its producer has not written one.",
        )
    readable: list[tuple[str, Artifact, Any]] = []
    unreadable: list[str] = []
    for path in files:
        try:
            loaded = _load(path, artifact)
            group = require(loaded.document, key, "text")
            readable.append((group, loaded, parse(loaded)))
        except FormatError as error:
            unreadable.append(f"{path.name} ({error})")
    if not readable:
        return NotRead(
            HealthState.UNAVAILABLE,
            "unrecognized_format",
            f"No {artifact} file in {directory} could be read: {'; '.join(unreadable[:3])}",
        )
    newest: dict[str, tuple[Artifact, Any]] = {}
    for group, loaded, parsed in readable:
        current = newest.get(group)
        if current is None or (loaded.generated_at, loaded.run_id) > (
            current[0].generated_at,
            current[0].run_id,
        ):
            newest[group] = (loaded, parsed)
    return {
        group: (Artifact(**{**loaded.__dict__, "unreadable": tuple(unreadable)}), parsed)
        for group, (loaded, parsed) in sorted(newest.items())
    }


def require(document: dict[str, Any], key: str, kind: str, *, optional: bool = False) -> Any:
    """A typed field: text, int, positive, number, nonnegative, fraction, list or object."""

    if key not in document:
        raise FormatError(f"missing field {key!r}")
    value = document[key]
    if value is None and optional:
        return None
    integer = isinstance(value, int) and not isinstance(value, bool)
    number = isinstance(value, int | float) and not isinstance(value, bool)
    valid = {
        "text": isinstance(value, str) and bool(value.strip()),
        "int": integer and value >= 0,
        "positive": integer and value >= 1,
        "number": number and math.isfinite(value),
        "nonnegative": number and math.isfinite(value) and value >= 0,
        "fraction": number and 0 <= value <= 1,
        "list": isinstance(value, list),
        "object": isinstance(value, dict),
    }[kind]
    if not valid:
        raise FormatError(f"field {key!r} is not a valid {kind}: {value!r}")
    return value


def choice(document: dict[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = require(document, key, "text")
    if value not in allowed:
        raise FormatError(f"field {key!r} is {value!r}, expected one of {sorted(allowed)}")
    return value


@dataclass(frozen=True)
class Finding:
    """One reason a check is in a state; the check takes the worst finding."""

    state: HealthState
    reason: str | None
    summary: str


def worst_finding(findings: Sequence[Finding]) -> Finding:
    """The worst finding; among equals, the first listed."""

    return min(findings, key=lambda finding: PRECEDENCE.index(finding.state))


SYNTHETIC_FINDING = Finding(
    HealthState.INSUFFICIENT_EVIDENCE,
    "synthetic_input",
    "The input is synthetic fixture data, so it says nothing about the real system.",
)


def common_findings(artifact: Artifact, *, allow_synthetic: bool) -> list[Finding]:
    """Findings every input shares: synthetic data, transition and unreadable files."""

    findings = []
    if artifact.synthetic and not allow_synthetic:
        findings.append(SYNTHETIC_FINDING)
    if artifact.document.get("transition_status") == TRANSITION_UNAPPROVED:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "unapproved_transition",
                "It uses an engineering-only portfolio transition that is not yet approved, "
                "so its results are a visible limitation, not evidence.",
            )
        )
    if artifact.unreadable:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "unreadable_inputs",
                f"{len(artifact.unreadable)} input files could not be read; the newest readable "
                "one was used.",
            )
        )
    return findings
