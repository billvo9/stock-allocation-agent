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
    head is healthy (the genesis hash for an empty ledger); an earlier entry's
    hash is degraded (copy the new head); a hash the ledger no longer holds is
    failed (lines were removed or the file was rewritten, which the chain
    alone cannot show). A missing ledger with a kept head past genesis was
    deleted or moved: `forecast.ledger` and this check fail
    (`ledger_deleted`, `head_not_in_ledger`). Not configured is
    insufficient_evidence, never healthy.
forecast.ledger_sessions
    Missing expected dates: rules-calendar sessions from the first recorded
    as-of session to the last session whose recording deadline (09:30
    America/New_York on the next weekday, the job's convention) has passed,
    with no forecast or refusal record. In a registered ledger the first
    record is session 1 and every later session is expected (the registered
    cadence, section 3 of docs/research/prospective_registration.md); in a
    ledger without a registration every session is expected as an operator
    default. Gaps are degraded and listed. The calendar is injected by the
    private script, because research code may not import the prospective
    side.
forecast.ledger_scores
    Matured forecast rows without a score: a row (asset, horizon, target of
    a forecast entry) matures at its exit session, the entry_lag + horizon-th
    rules-calendar session after its as-of session, and is due for scoring
    once that session's recording deadline (09:30 America/New_York on the
    next weekday, as above; an operator default) has passed. A due row with
    no score entry is degraded (`scores_overdue`: the score step did not run,
    or its outcome inputs are missing; nothing is filled); no row due yet is
    insufficient_evidence (`none_matured`); every due row scored is healthy.
    Only the identities in the ledger are read: never a forecast value.
forecast.ledger_trial
    The registered trial and section 5's conditions, from the job's own
    cadence (forecasting.prospective.cadence): session 1, the latest due
    session's number and phase (the trial is sessions 1-20, the confirmatory
    phase starts at 21), and every condition as one `key=value` line
    (condition, phase, session, number, fund): a chain break, a failed or
    unresolved kept head, a missed, refused or out-of-window session record,
    a record before the registration, and a fund without forecast rows for
    more than 2 consecutive expected sessions. A condition during the trial
    fails (`trial_void_condition`); one after the trial is degraded
    (`condition_after_trial`: it counts against C3). Health reports these and
    changes nothing: voiding and re-registering are the owner's decisions.
    Records in a ledger that does not open with a registration fail
    (`records_without_registration`); a registration without a record yet is
    insufficient_evidence (`awaiting_session_1`), and so is a trial whose kept
    head is not configured (`kept_head_not_judged`).

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
from stock_agent.forecasting.prospective import cadence, registration_record
from stock_agent.forecasting.prospective.ledger import (
    GENESIS,
    LedgerEntry,
    LedgerError,
    head,
    read_ledger,
)
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.forecasting.prospective.window import local_instant, next_weekday
from stock_agent.health.inputs import SYNTHETIC_FINDING, Finding, worst_finding
from stock_agent.health.model import Area, Better, Check, HealthState

TITLES = {
    "forecast.ledger": "Forecast ledger integrity",
    "forecast.ledger_backup": "Kept copy of the ledger head",
    "forecast.ledger_sessions": "Sessions with a forecast record",
    "forecast.ledger_scores": "Matured forecasts with a score",
    "forecast.ledger_trial": "Registered trial and its void conditions",
}
ROW_FIELDS = ("asset", "horizon", "target", "entry_lag", "status")
RECORD_KINDS = ("forecast", "refusal")
KINDS = ("registration", "forecast", "score", "refusal")
DEADLINE_LOCAL = "09:30"
TIMEZONE = "America/New_York"
SHOWN_DATES = 20
DELETED = "ledger_deleted"
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
    summary: str | None = None


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
        summary = read.summary or {
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
            f"chain verified: {len(entries)} entries ({kinds['registration']} registrations, "
            f"{kinds['forecast']} forecasts, {kinds['score']} scores, {kinds['refusal']} refusals)"
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
    kept: str | _Read | None,
    *,
    now: datetime,
    evidence: str | None,
    allow_synthetic: bool,
) -> Check:
    check_id = "forecast.ledger_backup"
    threshold = "the kept copy equals the ledger head (the genesis hash for an empty ledger)"
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
    if isinstance(kept, _Read):
        return _not_judged(
            check_id, kept, evidence=shown, summary="The kept copy of the ledger head is unusable."
        )
    if read.reason == DELETED:
        return _check(
            check_id,
            state=HealthState.FAILED,
            reason="head_not_in_ledger",
            measured_at=now,
            value=None,
            unit=None,
            threshold=threshold,
            summary="The ledger is gone, but its kept head shows it held records: it was "
            "deleted or moved.",
            detail=f"kept {kept}\nledger: {read.detail}",
            evidence=shown,
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
            evidence=shown,
            summary="The kept head cannot be compared while the ledger is not readable."
            if read.state is HealthState.FAILED
            else "No ledger yet; the kept head is the empty ledger's genesis hash.",
        )
    entries = read.entries
    # The genesis hash is the head of an empty ledger and the start of every chain.
    hashes = [GENESIS, *(entry.entry_sha256 for entry in entries)]
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
    registered = read.entries is not None and _registered(read.entries)
    threshold = (
        "a forecast or refusal record for every rules-calendar session from session 1, the "
        "first record after the registration, to the last one whose 09:30 New York deadline has "
        "passed (the registered cadence, section 3)"
        if registered
        else "a forecast or refusal record for every rules-calendar session from the first "
        "record to the last one whose 09:30 New York deadline has passed (operator default: no "
        "cadence is pre-registered)"
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
            f"{len(missing)} sessions since session 1 have no forecast or refusal record. "
            "Every session from session 1 is expected (section 3), and none is backfilled."
            if registered
            else f"{len(missing)} sessions since the first record have no forecast or refusal "
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


def _registered(entries: Sequence[LedgerEntry]) -> bool:
    try:
        return registration_record.current(entries) is not None
    except ForecastRefused:
        return False


def _trial(
    read: _Read,
    kept: str | None,
    calendar: RulesCalendar | None,
    *,
    now: datetime,
    evidence: str | None,
    allow_synthetic: bool,
) -> Check:
    check_id = "forecast.ledger_trial"
    unit = "section 5 conditions"
    threshold = (
        "no section 5 condition: the chain and the kept head verify, every due session from "
        "session 1 has a record inside its window and is not refused, and no fund lacks forecast "
        "rows for more than 2 consecutive sessions"
    )
    common = {"threshold": threshold, "evidence": evidence}
    if read.entries is None:
        if read.state is HealthState.FAILED:
            # A deleted ledger whose kept head shows it held records lost its kept head; any
            # other refusal is a chain break. The phase cannot be read from a refused ledger.
            kind = "kept_head_failed" if read.reason == DELETED else "chain_break"
            what = "its kept head is lost" if kind == "kept_head_failed" else "a chain break"
            line = cadence.Condition(kind, "-", None, None, None, read.detail).line()
            return _check(
                check_id,
                state=HealthState.FAILED,
                reason=kind,
                measured_at=now,
                value=None,
                unit=None,
                summary=f"The forecast ledger fails its integrity check: {what}. That is a void "
                "condition during the trial (section 5) and a C3 failure after it (section 11); "
                "the phase cannot be read from a refused ledger. Health reports it and changes "
                "nothing; re-registration is the owner's decision.",
                detail=line,
                **common,
            )
        return _not_judged(
            check_id, read, evidence=evidence, summary="No forecast has been recorded yet."
        )
    entries = read.entries
    if not entries:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="not_registered",
            measured_at=now,
            value=None,
            unit=None,
            summary="The ledger holds no registration yet, so no trial has started.",
            detail="0 entries.",
            **common,
        )
    try:
        found = registration_record.current(entries)
    except ForecastRefused as refused:
        reason = (
            "records_without_registration"
            if refused.reason == "not_registered"
            else (refused.reason)
        )
        return _check(
            check_id,
            state=HealthState.FAILED,
            reason=reason,
            measured_at=now,
            value=None,
            unit=None,
            summary="The ledger's records are not under a valid registration: none of them can "
            "count toward the trial or a checkpoint.",
            detail=f"{refused.reason}: {refused.detail}",
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
            summary="No exchange calendar was given, so the trial's sessions are not counted.",
            detail="The private script passes the rules calendar; this run had none.",
            **common,
        )
    try:
        report = cadence.cadence(entries, calendar=calendar.sessions, now=now, kept=kept)
    except (CalendarRefused, ValueError) as error:
        return _check(
            check_id,
            state=HealthState.UNAVAILABLE,
            reason="calendar_refused",
            measured_at=now,
            value=None,
            unit=None,
            summary="The exchange calendar refused the date range, so the trial is not judged.",
            detail=f"{calendar.calendar_id}: {error}",
            **common,
        )
    record = found.record
    lines = [
        (
            f"registered {found.registered_at.isoformat()} at {found.registration_commit} "
            f"(entry {found.entry.sequence}; bundle {record['bundle']['bundle_id']}; config "
            f"{record['config_sha256'][:12]}; rule {record['zero_volatility_rule']})"
        )
    ]
    if report.first_session is None:
        lines.append("session 1: not recorded yet")
    else:
        latest = (
            f"latest due session {report.number} ({report.phase})"
            if report.number is not None
            else "no session due yet"
        )
        lines += [
            f"session 1 {report.first_session}; {latest}",
            (
                f"trial: sessions 1-{cadence.TRIAL_SESSIONS}, to "
                f"{report.trial_last_session or '?'}; confirmatory phase from session "
                f"{cadence.TRIAL_SESSIONS + 1}, {report.confirmatory_first_session or '?'}"
            ),
            f"sessions due: {len(report.due)}; sessions recorded: {len(report.recorded)}",
        ]
    lines.append(
        "kept head: judged" if kept is not None else "kept head: not configured or unusable"
    )
    lines += [condition.line() for condition in report.conditions]
    trial, later = report.trial_conditions, report.later_conditions
    if trial:
        kinds = ", ".join(sorted({c.condition for c in trial}))
        finding = Finding(
            HealthState.FAILED,
            "trial_void_condition",
            f"{len(trial)} section 5 conditions during the trial ({kinds}): each voids the "
            "registration under section 5. Health reports them and changes nothing; "
            "re-registering is the owner's decision.",
        )
    elif later:
        kinds = ", ".join(sorted({c.condition for c in later}))
        finding = Finding(
            HealthState.DEGRADED,
            "condition_after_trial",
            f"{len(later)} conditions after the trial ({kinds}). They count against C3 "
            "(section 11), not as a void; nothing is changed.",
        )
    elif report.first_session is None:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "awaiting_session_1",
            "Registered; no record yet. Session 1 is the first eligible close (section 4).",
        )
    elif kept is None:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "kept_head_not_judged",
            "No usable kept head, so its void conditions cannot be checked.",
        )
    else:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"No section 5 condition through session {report.number or 0} "
            f"({report.phase or 'none due'}).",
        )
    finding = _finish([finding], _records(entries), allow_synthetic)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=len(report.conditions),
        unit=unit,
        better=Better.LOWER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        **common,
    )


def _scored(entries: Sequence[LedgerEntry]) -> set[tuple]:
    """(forecast entry hash, asset, horizon, target) of every scored row."""

    keys = set()
    for entry in entries:
        scores = entry.payload.get("scores") if entry.kind == "score" else None
        for score in scores if isinstance(scores, list) else ():
            if not isinstance(score, dict):
                continue
            try:
                horizon = int(score.get("horizon"))
            except (TypeError, ValueError):
                continue
            keys.add(
                (
                    score.get("forecast_entry_sha256"),
                    score.get("asset"),
                    horizon,
                    score.get("target"),
                )
            )
    return keys


def _forecast_rows(entries: Sequence[LedgerEntry]) -> tuple[list[tuple[LedgerEntry, dict]], int]:
    """The forecast rows that were made (status ok), and how many rows are unreadable."""

    rows, unreadable = [], 0
    for entry in entries:
        if entry.kind != "forecast":
            continue
        payload_rows = entry.payload.get("rows")
        if not isinstance(payload_rows, list) or _as_of(entry) is None:
            unreadable += 1
            continue
        for row in payload_rows:
            if not isinstance(row, dict) or any(field not in row for field in ROW_FIELDS):
                unreadable += 1
            elif row["status"] == "ok":
                rows.append((entry, row))
    return rows, unreadable


def _scores(
    read: _Read,
    calendar: RulesCalendar | None,
    *,
    now: datetime,
    evidence: str | None,
    allow_synthetic: bool,
) -> Check:
    check_id = "forecast.ledger_scores"
    threshold = (
        "every forecast row whose exit session's 09:30 New York deadline (next weekday) has "
        "passed has a score (operator default)"
    )
    common = {"threshold": threshold, "evidence": evidence}
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
            summary="Scores cannot be counted while the ledger is not readable."
            if read.state is HealthState.FAILED
            else "No forecast has been recorded yet.",
        )
    rows, unreadable = _forecast_rows(read.entries)
    if not rows and not unreadable:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_forecast_recorded",
            measured_at=now,
            value=None,
            unit=None,
            summary="No forecast row has been recorded yet, so nothing can be scored.",
            detail=f"{len(read.entries)} entries, none a forecast with a made row.",
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
            summary="No exchange calendar was given, so matured forecasts are not counted.",
            detail="The private script passes the rules calendar; this run had none.",
            **common,
        )
    stamp = pd.Timestamp(now)
    first = min((pd.Timestamp(_as_of(entry), tz="UTC") for entry, _ in rows), default=None)
    try:
        sessions = (
            calendar.sessions(first, stamp.tz_convert("UTC").normalize())
            if first is not None
            else pd.DatetimeIndex([])
        )
    except (CalendarRefused, ValueError) as error:
        return _check(
            check_id,
            state=HealthState.UNAVAILABLE,
            reason="calendar_refused",
            measured_at=now,
            value=None,
            unit=None,
            summary="The exchange calendar refused the date range, so scores are not counted.",
            detail=f"{calendar.calendar_id}: {error}",
            **common,
        )
    position = {f"{session:%Y-%m-%d}": index for index, session in enumerate(sessions)}
    scored = _scored(read.entries)
    done, pending, overdue, unplaced = 0, 0, [], 0
    for entry, row in rows:
        try:
            horizon, lag = int(row["horizon"]), int(row["entry_lag"])
        except (TypeError, ValueError):
            unplaced += 1
            continue
        if (entry.entry_sha256, row["asset"], horizon, row["target"]) in scored:
            done += 1
            continue
        start = position.get(_as_of(entry))
        if start is None:
            unplaced += 1
        elif start + lag + horizon < len(sessions) and (
            _deadline(sessions[start + lag + horizon]) <= stamp
        ):
            overdue.append((sessions[start + lag + horizon], _as_of(entry), row))
        else:
            pending += 1
    overdue.sort(key=lambda item: item[0])
    shown = "; ".join(
        f"{as_of} {row['asset']} h{row['horizon']} {row['target']} (exit {exit:%Y-%m-%d})"
        for exit, as_of, row in overdue[:SHOWN_DATES]
    )
    if len(overdue) > SHOWN_DATES:
        shown += f" and {len(overdue) - SHOWN_DATES} more"
    lines = [
        f"calendar {calendar.calendar_id}; forecast rows made: {len(rows)}",
        f"scored: {done}; not due yet: {pending}; due without a score: {len(overdue)}",
        f"due without a score: {shown or 'none'}",
        f"rows not placed on the calendar: {unplaced}; unreadable rows: {unreadable}",
    ]
    findings = []
    if unreadable or unplaced:
        findings.append(
            Finding(
                HealthState.UNAVAILABLE,
                "unrecognized_rows",
                f"{unreadable + unplaced} forecast rows could not be read or placed on the "
                "calendar, so their scoring cannot be judged.",
            )
        )
    if overdue:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "scores_overdue",
                f"{len(overdue)} matured forecast rows have no score past their deadline (oldest "
                f"exit session {overdue[0][0]:%Y-%m-%d}): the score step did not run or its "
                "outcome is missing. Nothing is filled.",
            )
        )
    elif done == 0:
        findings.append(
            Finding(
                HealthState.INSUFFICIENT_EVIDENCE,
                "none_matured",
                f"No forecast row is due for scoring yet ({pending} rows waiting for their exit "
                "session).",
            )
        )
    else:
        findings.append(
            Finding(
                HealthState.HEALTHY,
                None,
                f"Every forecast row due for scoring has a score ({done} scored, {pending} not "
                "due yet).",
            )
        )
    finding = _finish(findings, _records(read.entries), allow_synthetic)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=len(overdue),
        unit="matured rows without a score",
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
    """
    `forecast.ledger`, `forecast.ledger_backup`, `forecast.ledger_sessions`,
    `forecast.ledger_scores` and `forecast.ledger_trial`.
    """

    read = _read(path)
    kept = None if backup is None else _kept_hash(backup)
    if read.reason == "input_missing" and isinstance(kept, str) and kept != GENESIS:
        # A kept head past genesis proves the ledger held records: a missing file is a
        # deletion, as a removed tail line is a truncation.
        read = _Read(
            None,
            HealthState.FAILED,
            DELETED,
            f"{path} does not exist, but the kept head {kept[:12]}... shows it held records.",
            "The forecast ledger is missing, but its kept head shows it held records: it was "
            "deleted or moved.",
        )
    evidence = None if path is None else str(path)
    common = {"now": now, "evidence": evidence, "allow_synthetic": allow_synthetic}
    return [
        _integrity(read, **common),
        _backup(read, backup, kept, **common),
        _sessions(read, calendar, **common),
        _scores(read, calendar, **common),
        _trial(read, kept if isinstance(kept, str) else None, calendar, **common),
    ]


__all__ = ["TITLES", "RulesCalendar", "ledger_checks"]
