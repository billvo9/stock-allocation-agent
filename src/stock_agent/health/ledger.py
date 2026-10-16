"""
Prospective forecast ledger health (local monitor only), read with the
forecast job's own verifying reader (`forecasting.prospective.ledger`): the
same chain rule, so health and the job can never disagree on what is intact.

forecast.ledger
    The hash chain verifies. Value: the number of entries (forecasts, scores
    and refusals); a smaller number than the previous summary raises a
    deterioration alert, which is how a removed tail line shows up even
    without a kept head. The detail names the head hash, the as-of sessions
    recorded and the data class. A refused ledger (an edited, inserted,
    removed or reordered line, an incomplete last line, a symlink) is failed;
    an unreadable file is unavailable; no file or no entry yet is
    insufficient_evidence (recording a real forecast awaits the owner's
    activation decision).
forecast.ledger_backup
    The head hash the owner keeps outside the ledger (`--ledger-head-backup`,
    a file whose first line is the 64-hex hash the job printed). Equal to the
    head is healthy; an earlier entry's hash is degraded (copy the new head);
    a hash the ledger no longer holds is failed (lines were removed or the
    file was rewritten, which the chain alone cannot show). Not configured is
    insufficient_evidence, never healthy.
forecast.ledger_sessions
    Missing expected dates: rules-calendar sessions from the first recorded
    as-of session to the last session whose recording deadline (09:30
    America/New_York on the next weekday, the job's convention) has passed,
    with no forecast or refusal record. No cadence is pre-registered yet, so
    every session is expected (an operator default, not owner policy): gaps
    are degraded and listed. The calendar is injected by the private script,
    because research code may not import the prospective side.

In the private scope a ledger whose records are all synthetic is at most
insufficient_evidence (`synthetic_input`); a failure still shows. Health
never writes to the ledger.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.forecasting.prospective.ledger import (
    LedgerEntry,
    LedgerError,
    head,
    read_ledger,
)
from stock_agent.forecasting.prospective.window import local_instant, next_weekday
from stock_agent.health.inputs import SYNTHETIC_FINDING, Finding, worst_finding
from stock_agent.health.model import Area, Better, Check, HealthState

TITLES = {
    "forecast.ledger": "Forecast ledger integrity",
    "forecast.ledger_backup": "Kept copy of the ledger head",
    "forecast.ledger_sessions": "Sessions with a forecast record",
}
RECORD_KINDS = ("forecast", "refusal")
KINDS = ("forecast", "score", "refusal")
DEADLINE_LOCAL = "09:30"
TIMEZONE = "America/New_York"
SHOWN_DATES = 20
_HASH = re.compile(r"[0-9a-f]{64}")

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]


@dataclass(frozen=True)
class RulesCalendar:
    """The rules-calendar session list and its id, passed in by the private script."""

    sessions: Sessions
    calendar_id: str


@dataclass(frozen=True)
class _Read:
    entries: tuple[LedgerEntry, ...] | None
    state: HealthState | None = None
    reason: str | None = None
    detail: str = ""


def _read(path: Path | None) -> _Read:
    if path is None:
        return _Read(
            None,
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            "No ledger path was given to this run.",
        )
    if not path.exists() and not path.is_symlink():
        return _Read(
            None,
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_missing",
            f"{path} does not exist: no forecast has been recorded. Recording a real forecast "
            "awaits the owner's activation decision.",
        )
    try:
        return _Read(read_ledger(path))
    except LedgerError as error:
        return _Read(None, HealthState.FAILED, "ledger_refused", f"{path}: {error}")
    except (OSError, UnicodeDecodeError) as error:
        return _Read(
            None, HealthState.UNAVAILABLE, "ledger_unreadable", f"{path}: {type(error).__name__}"
        )


def _records(entries: Sequence[LedgerEntry]) -> list[LedgerEntry]:
    return [entry for entry in entries if entry.kind in RECORD_KINDS]


def _as_of(entry: LedgerEntry) -> str | None:
    value = entry.payload.get("as_of_session")
    return value if isinstance(value, str) else None


def _data_classes(records: Sequence[LedgerEntry]) -> set[str]:
    classes = set()
    for entry in records:
        data = entry.payload.get("data")
        value = data.get("data_class") if isinstance(data, dict) else None
        classes.add(value if isinstance(value, str) else "unknown")
    return classes


def _synthetic(records: Sequence[LedgerEntry]) -> bool:
    return bool(records) and _data_classes(records) == {"synthetic"}


def _check(check_id: str, *, evidence: str | None, **fields) -> Check:
    return Check(
        check_id=check_id, area=Area.FORECAST, title=TITLES[check_id], evidence=evidence, **fields
    )


def _not_judged(check_id: str, read: _Read, *, evidence: str | None, summary: str) -> Check:
    return _check(
        check_id,
        state=read.state,
        reason=read.reason,
        measured_at=None,
        value=None,
        unit=None,
        threshold=None,
        summary=summary,
        detail=read.detail,
        evidence=evidence,
    )


def _finish(findings: list[Finding], records, allow_synthetic: bool) -> Finding:
    if _synthetic(records) and not allow_synthetic:
        findings.append(SYNTHETIC_FINDING)
    return worst_finding(findings)


def _integrity(read: _Read, *, now: datetime, evidence: str | None, allow_synthetic: bool) -> Check:
    check_id = "forecast.ledger"
    if read.entries is None:
        summary = {
            HealthState.FAILED: "The forecast ledger failed its integrity check, so none of its "
            "records can be trusted.",
            HealthState.UNAVAILABLE: "The forecast ledger could not be read.",
        }.get(read.state, "No forecast has been recorded yet.")
        if read.state is HealthState.FAILED:
            return _check(
                check_id,
                state=read.state,
                reason=read.reason,
                measured_at=now,
                value=None,
                unit=None,
                threshold="the hash chain verifies",
                summary=summary,
                detail=read.detail,
                evidence=evidence,
            )
        return _not_judged(check_id, read, evidence=evidence, summary=summary)
    entries = read.entries
    if not entries:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_entries",
            measured_at=now,
            value=0,
            unit="ledger entries",
            threshold="the hash chain verifies",
            summary="The forecast ledger exists but holds no entry yet.",
            detail="0 entries; head is the genesis hash.",
            evidence=evidence,
        )
    records = _records(entries)
    kinds = {kind: sum(entry.kind == kind for entry in entries) for kind in KINDS}
    sessions = sorted(filter(None, (_as_of(entry) for entry in records)))
    last = entries[-1]
    recorded = (
        f"{len(set(sessions))}, first {sessions[0]}, last {sessions[-1]}" if sessions else "none"
    )
    lines = [
        (
            f"chain verified: {len(entries)} entries ({kinds['forecast']} forecasts, "
            f"{kinds['score']} scores, {kinds['refusal']} refusals)"
        ),
        (
            f"head hash {head(entries)} (entry {last.sequence}, {last.kind}, recorded "
            f"{last.recorded_at})"
        ),
        f"as-of sessions recorded: {recorded}",
        f"data class of the records: {', '.join(sorted(_data_classes(records))) or 'none'}",
    ]
    healthy = Finding(
        HealthState.HEALTHY,
        None,
        f"The ledger's hash chain verifies: {len(entries)} entries"
        + (f", the last forecast record as of {sessions[-1]}." if sessions else "."),
    )
    finding = _finish([healthy], records, allow_synthetic)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=len(entries),
        unit="ledger entries",
        threshold="the hash chain verifies; fewer entries than the previous summary is a "
        "deterioration",
        better=Better.HIGHER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        evidence=evidence,
    )


def _kept_hash(path: Path) -> str | _Read:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return _Read(
            None,
            HealthState.UNAVAILABLE,
            "backup_unreadable",
            f"The kept head file {path} could not be read ({type(error).__name__}).",
        )
    first = text.splitlines()[0].strip() if text.strip() else ""
    if not _HASH.fullmatch(first):
        return _Read(
            None,
            HealthState.UNAVAILABLE,
            "backup_unrecognized",
            f"The first line of {path} is not a 64-character lower-case hex hash.",
        )
    return first


def _backup(
    read: _Read,
    backup: Path | None,
    *,
    now: datetime,
    evidence: str | None,
    allow_synthetic: bool,
) -> Check:
    check_id = "forecast.ledger_backup"
    threshold = "the kept copy equals the ledger head"
    unit = "entries recorded after the kept head"
    if backup is None:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="backup_not_configured",
            measured_at=None,
            value=None,
            unit=None,
            threshold=threshold,
            summary="No copy of the ledger head is configured, so a removed last line would "
            "go unnoticed.",
            detail="Keep the head hash the forecast job prints in a file outside the ledger and "
            "pass it with --ledger-head-backup.",
            evidence=evidence,
        )
    shown = f"{evidence}; kept head {backup}"
    if read.entries is None:
        not_read = _Read(
            None,
            HealthState.UNAVAILABLE if read.state is HealthState.FAILED else read.state,
            read.reason,
            read.detail,
        )
        return _not_judged(
            check_id,
            not_read,
            evidence=shown,
            summary="The kept head cannot be compared while the ledger is not readable.",
        )
    kept = _kept_hash(backup)
    if isinstance(kept, _Read):
        return _not_judged(
            check_id, kept, evidence=shown, summary="The kept copy of the ledger head is unusable."
        )
    entries = read.entries
    hashes = [entry.entry_sha256 for entry in entries]
    records = _records(entries)
    if kept not in hashes:
        finding = Finding(
            HealthState.FAILED,
            "head_not_in_ledger",
            "The ledger no longer holds the kept head hash: lines were removed or the file was "
            "rewritten.",
        )
        value = None
    else:
        behind = len(hashes) - 1 - hashes.index(kept)
        value = behind
        finding = (
            Finding(HealthState.HEALTHY, None, "The kept copy matches the ledger head.")
            if behind == 0
            else Finding(
                HealthState.DEGRADED,
                "backup_behind",
                f"The kept copy is {behind} entries behind the head; copy the new head hash.",
            )
        )
    finding = _finish([finding], records, allow_synthetic)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=value,
        unit=unit if value is not None else None,
        threshold=threshold,
        summary=finding.summary,
        detail=f"kept {kept}\nhead {head(entries)}\n{len(entries)} entries in the ledger",
        evidence=shown,
    )


def _deadline(session: pd.Timestamp) -> pd.Timestamp:
    return local_instant(next_weekday(session), DEADLINE_LOCAL, TIMEZONE)


def _sessions(
    read: _Read,
    calendar: RulesCalendar | None,
    *,
    now: datetime,
    evidence: str | None,
    allow_synthetic: bool,
) -> Check:
    check_id = "forecast.ledger_sessions"
    threshold = (
        "a forecast or refusal record for every rules-calendar session from the first record to "
        "the last one whose 09:30 New York deadline has passed (operator default: no cadence is "
        "pre-registered)"
    )
    if read.entries is None:
        not_read = _Read(
            None,
            HealthState.UNAVAILABLE if read.state is HealthState.FAILED else read.state,
            read.reason,
            read.detail,
        )
        return _not_judged(
            check_id,
            not_read,
            evidence=evidence,
            summary="Recorded sessions cannot be counted while the ledger is not readable."
            if read.state is HealthState.FAILED
            else "No forecast has been recorded yet.",
        )
    records = _records(read.entries)
    recorded = {value for value in (_as_of(entry) for entry in records) if value}
    common = {"threshold": threshold, "evidence": evidence}
    if not recorded:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_forecast_recorded",
            measured_at=now,
            value=None,
            unit=None,
            summary="No forecast or refusal has been recorded yet.",
            detail=f"{len(read.entries)} entries, none a forecast or refusal record.",
            **common,
        )
    if calendar is None:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="calendar_not_configured",
            measured_at=None,
            value=None,
            unit=None,
            summary="No exchange calendar was given, so missing sessions are not counted.",
            detail="The private script passes the rules calendar; this run had none.",
            **common,
        )
    first = pd.Timestamp(min(recorded), tz="UTC")
    stamp = pd.Timestamp(now)
    try:
        candidates = calendar.sessions(first, stamp.tz_convert("UTC").normalize())
    except (CalendarRefused, ValueError) as error:
        return _check(
            check_id,
            state=HealthState.UNAVAILABLE,
            reason="calendar_refused",
            measured_at=now,
            value=None,
            unit=None,
            summary="The exchange calendar refused the date range, so sessions are not counted.",
            detail=f"{calendar.calendar_id}: {error}",
            **common,
        )
    due = [session for session in candidates if _deadline(session) <= stamp]
    expected = {f"{session:%Y-%m-%d}" for session in due}
    missing = sorted(expected - recorded)
    span = f"{due[0]:%Y-%m-%d} to {due[-1]:%Y-%m-%d}" if due else "none due yet"
    shown = ", ".join(missing[:SHOWN_DATES]) or "none"
    if len(missing) > SHOWN_DATES:
        shown += f" and {len(missing) - SHOWN_DATES} more"
    lines = [
        f"calendar {calendar.calendar_id}; sessions due: {len(expected)} ({span})",
        f"sessions recorded: {len(recorded)}",
        f"sessions without a record: {shown}",
    ]
    if missing:
        finding = Finding(
            HealthState.DEGRADED,
            "sessions_without_record",
            f"{len(missing)} sessions since the first record have no forecast or refusal "
            "record. No cadence is pre-registered, so every session is expected.",
        )
    else:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"Every session due since the first record ({len(expected)}) has a forecast or "
            "refusal record.",
        )
    finding = _finish([finding], records, allow_synthetic)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=len(missing),
        unit="sessions without a record",
        better=Better.LOWER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        **common,
    )


def ledger_checks(
    path: Path | None,
    *,
    backup: Path | None,
    calendar: RulesCalendar | None,
    now: datetime,
    allow_synthetic: bool = False,
) -> list[Check]:
    """`forecast.ledger`, `forecast.ledger_backup` and `forecast.ledger_sessions`."""

    read = _read(path)
    evidence = None if path is None else str(path)
    common = {"now": now, "evidence": evidence, "allow_synthetic": allow_synthetic}
    return [
        _integrity(read, **common),
        _backup(read, backup, **common),
        _sessions(read, calendar, **common),
    ]


__all__ = ["TITLES", "RulesCalendar", "ledger_checks"]
