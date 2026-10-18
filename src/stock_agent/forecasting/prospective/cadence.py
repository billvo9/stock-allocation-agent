"""
The registered cadence of a prospective ledger and section 5's conditions
(docs/research/prospective_registration.md, sections 3 to 6), as structured
diagnostics. Pure: it reads verified ledger entries, an injected exchange
calendar, the time and the owner's kept head hash; it never decides whether a
registration is void and never writes or alters a record.

Sessions. Session 1 is the as-of session of the ledger's first forecast or
refusal record after its registration (the job records nothing before the
first eligible close). Session n is the n-th exchange session from session 1,
recorded or not: every session from session 1 is expected (section 3).
Sessions 1 to TRIAL_SESSIONS (20) are the trial; the confirmatory phase
starts at session 21. A session is due once its recording deadline, 09:30
New York on the next weekday, has passed.

Conditions (each a `Condition`: condition, phase, session, number, fund,
detail). During the trial they are section 5's void conditions; after it
they count against C3 (section 11). Which is which is reported, never acted
on:

    session_not_recorded        a due session without a forecast or refusal
                                record (a missed run, or a session the job
                                refused as a whole: timing, models, stage or
                                registration)
    session_refused             a refusal record (every fund refused)
    record_outside_window       a record appended before 18:00 New York on its
                                session or at or after its deadline
    record_before_registration  a record whose close (18:00 New York) is not
                                after the registration time
    record_off_calendar         a record whose as-of date is not an exchange
                                session
    data_gap                    a fund without forecast rows for more than
                                MAX_GAP (2) consecutive expected sessions,
                                from single-fund refusals, refused or missed
                                sessions; reported at the session where the
                                run exceeds 2, once per run, in the phase
                                of the run's first session
    kept_head_failed            the kept head hash is not in the ledger
    kept_head_unresolved        the kept head is behind the ledger head, and
                                the first recording deadline after the first
                                entry it does not hold has passed (section 6:
                                then a degraded check counts as failed)

A chain break (the ledger fails its integrity check) is the remaining
condition; it is found before any entry can be read, by the caller.

Phases. A session's condition takes that session's phase; a data gap takes
the phase of its first session (a gap that starts in the trial is a trial
condition). An unresolved kept head takes the phase of the session whose
deadline passed, a failed kept head the phase of the latest due session; both
are trial conditions before session 1. Health judges the kept head when it
runs: a lapse repaired after its deadline but before the next run is not seen
(the head history file is not read).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from stock_agent.forecasting.prospective import ledger, registration, registration_record
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.forecasting.prospective.window import local_instant, next_weekday

TRIAL_SESSIONS = registration.TRIAL_SESSIONS
MAX_GAP = 2  # consecutive expected sessions a fund may lack forecast rows (section 5)
CLOSE_AVAILABLE_LOCAL = "18:00"  # the registered config's conventions
DEADLINE_LOCAL = "09:30"
TIMEZONE = "America/New_York"
RECORD_KINDS = ("forecast", "refusal")
TRIAL, CONFIRMATORY = "trial", "confirmatory"

Calendar = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]


@dataclass(frozen=True)
class Condition:
    """One section 5 condition, as a structured diagnostic."""

    condition: str
    phase: str
    session: str | None
    number: int | None
    asset: str | None
    detail: str

    def to_json(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "phase": self.phase,
            "session": self.session,
            "number": self.number,
            "asset": self.asset,
            "detail": self.detail,
        }

    def line(self) -> str:
        """One `key=value` line: machine-readable fields, then the detail."""

        fields = ("condition", "phase", "session", "number", "asset")
        head = " ".join(f"{name}={getattr(self, name) or '-'}" for name in fields)
        return f"{head}: {self.detail}"


@dataclass(frozen=True)
class Cadence:
    registration: registration_record.Registration
    first_session: str | None  # session 1
    due: tuple[str, ...]  # expected sessions whose deadline has passed
    recorded: tuple[str, ...]  # as-of sessions with a forecast or refusal record
    number: int | None  # the latest due session's number
    trial_last_session: str | None  # session 20
    confirmatory_first_session: str | None  # session 21
    conditions: tuple[Condition, ...]

    @property
    def phase(self) -> str | None:
        return None if self.number is None else phase_of(self.number)

    @property
    def trial_conditions(self) -> tuple[Condition, ...]:
        return tuple(c for c in self.conditions if c.phase == TRIAL)

    @property
    def later_conditions(self) -> tuple[Condition, ...]:
        return tuple(c for c in self.conditions if c.phase != TRIAL)


def phase_of(number: int) -> str:
    return TRIAL if number <= TRIAL_SESSIONS else CONFIRMATORY


def _day(stamp: pd.Timestamp) -> str:
    return f"{stamp:%Y-%m-%d}"


def opens(session: str) -> pd.Timestamp:
    """When a session's close counts as available: its recording window opens."""

    return local_instant(session, CLOSE_AVAILABLE_LOCAL, TIMEZONE)


def deadline(session: str | pd.Timestamp) -> pd.Timestamp:
    """09:30 New York on the first weekday after the session."""

    return local_instant(next_weekday(session), DEADLINE_LOCAL, TIMEZONE)


def _as_of(entry: ledger.LedgerEntry) -> str | None:
    value = entry.payload.get("as_of_session")
    return value if isinstance(value, str) else None


def _funds_with_rows(entry: ledger.LedgerEntry) -> set[str]:
    rows = entry.payload.get("rows") if entry.kind == "forecast" else None
    return {
        row.get("asset")
        for row in (rows if isinstance(rows, list) else ())
        if isinstance(row, Mapping) and row.get("status") == "ok"
    }


def _next_deadline(after: pd.Timestamp, calendar: Calendar) -> tuple[pd.Timestamp, str] | None:
    """The first recording deadline after `after`, and the session it belongs to."""

    day = after.tz_convert("UTC").normalize()
    try:
        sessions = calendar(day - pd.Timedelta(days=10), day + pd.Timedelta(days=20))
    except ValueError:
        return None
    later = sorted((deadline(s), _day(s)) for s in sessions if deadline(s) > after)
    return later[0] if later else None


def _kept_conditions(
    entries: Sequence[ledger.LedgerEntry],
    kept: str,
    calendar: Calendar,
    now: pd.Timestamp,
    phase: str,
    phase_at: Callable[[str], str],
) -> list[Condition]:
    hashes = [ledger.GENESIS, *(entry.entry_sha256 for entry in entries)]
    if kept not in hashes:
        return [
            Condition(
                "kept_head_failed",
                phase,
                None,
                None,
                None,
                f"the kept head {kept[:12]}... is not in the ledger: lines were removed or the "
                "file was rewritten",
            )
        ]
    behind = len(hashes) - 1 - hashes.index(kept)
    if behind == 0:
        return []
    uncovered = entries[hashes.index(kept)]  # the first entry the kept head does not hold
    appended = pd.Timestamp(uncovered.recorded_at)
    found = _next_deadline(appended, calendar)
    if found is None or now < found[0]:
        return []
    limit, missed = found
    # The phase of the session whose deadline passed, not of when health looked.
    return [
        Condition(
            "kept_head_unresolved",
            phase_at(missed),
            None,
            None,
            None,
            f"the kept head is {behind} entries behind; entry {uncovered.sequence} was appended "
            f"{appended.isoformat()} and the next recording deadline {limit.isoformat()} passed",
        )
    ]


def _gaps(
    order: Sequence[str], rows: Mapping[str, set[str]], funds: Sequence[str], numbers, phase
) -> list[Condition]:
    found = []
    for fund in funds:
        run: list[str] = []
        for session in [*order, None]:
            if session is not None and fund not in rows.get(session, set()):
                run.append(session)
                continue
            if len(run) > MAX_GAP:
                at = run[MAX_GAP]
                # A gap that starts in the trial takes the trial's phase, even if it first
                # exceeds MAX_GAP sessions after session 20: the conservative reading.
                found.append(
                    Condition(
                        "data_gap",
                        phase(numbers[run[0]]),
                        at,
                        numbers[at],
                        fund,
                        f"{fund} has no forecast rows for {len(run)} consecutive expected "
                        f"sessions, {run[0]} to {run[-1]} (more than {MAX_GAP})",
                    )
                )
            run = []
    return found


def cadence(
    entries: Sequence[ledger.LedgerEntry],
    *,
    calendar: Calendar,
    now: pd.Timestamp,
    kept: str | None = None,
) -> Cadence:
    """
    The cadence of verified `entries` at `now`. The ledger must open with its
    registration (registration_record.current refuses otherwise). `kept` is
    the owner's kept head hash, or None when none is configured (then no
    kept-head condition is judged).
    """

    found = registration_record.current(entries)
    if found is None:
        raise ForecastRefused("not_registered", "the ledger holds no entry")
    stamp = pd.Timestamp(now).tz_convert("UTC")
    records = [entry for entry in entries if entry.kind in RECORD_KINDS]
    sessions_recorded = sorted({s for s in map(_as_of, records) if s})
    conditions: list[Condition] = []
    if not sessions_recorded:
        if kept is not None:
            conditions += _kept_conditions(
                entries, kept, calendar, stamp, TRIAL, lambda session: TRIAL
            )
        return Cadence(found, None, (), (), None, None, None, tuple(conditions))
    first = sessions_recorded[0]
    last = max(_day(stamp), sessions_recorded[-1])
    calendar_days = [
        _day(s) for s in calendar(pd.Timestamp(first, tz="UTC"), pd.Timestamp(last, tz="UTC"))
    ]
    numbers = {session: index + 1 for index, session in enumerate(calendar_days)}
    due = [session for session in calendar_days if deadline(session) <= stamp]
    number = numbers[due[-1]] if due else None

    def phase(at: int | None) -> str:
        return TRIAL if at is None else phase_of(at)

    rows: dict[str, set[str]] = {}
    for entry in records:
        session = _as_of(entry)
        if session is None or session not in numbers:
            conditions.append(
                Condition(
                    "record_off_calendar",
                    phase(number),
                    session,
                    None,
                    None,
                    f"entry {entry.sequence} names {session!r}, not an exchange session",
                )
            )
            continue
        at = numbers[session]
        rows.setdefault(session, set()).update(_funds_with_rows(entry))
        recorded_at = pd.Timestamp(entry.recorded_at)
        if not opens(session) <= recorded_at < deadline(session):
            conditions.append(
                Condition(
                    "record_outside_window",
                    phase(at),
                    session,
                    at,
                    None,
                    f"entry {entry.sequence} was appended {recorded_at.isoformat()}, outside "
                    f"[{opens(session).isoformat()}, {deadline(session).isoformat()})",
                )
            )
        if opens(session) <= found.registered_at:
            conditions.append(
                Condition(
                    "record_before_registration",
                    phase(at),
                    session,
                    at,
                    None,
                    f"the {session} close is not after the registration time "
                    f"{found.registered_at.isoformat()}",
                )
            )
        if entry.kind == "refusal":
            conditions.append(
                Condition(
                    "session_refused",
                    phase(at),
                    session,
                    at,
                    None,
                    f"entry {entry.sequence}: every fund was refused "
                    f"({len(entry.payload.get('refusals') or [])} refusals)",
                )
            )
    recorded = set(rows) | {s for s in sessions_recorded if s not in numbers}
    for session in due:
        if session not in recorded:
            conditions.append(
                Condition(
                    "session_not_recorded",
                    phase(numbers[session]),
                    session,
                    numbers[session],
                    None,
                    f"no forecast or refusal record before {deadline(session).isoformat()}",
                )
            )
    order = [session for session in calendar_days if session in due or session in rows]
    funds = [str(fund) for fund in found.record["assets"]]
    conditions += _gaps(order, rows, funds, numbers, phase)
    if kept is not None:
        conditions += _kept_conditions(
            entries, kept, calendar, stamp, phase(number), lambda s: phase(numbers.get(s))
        )
    conditions.sort(key=lambda c: (c.number or 0, c.session or "", c.condition, c.asset or ""))
    trial_end = None
    try:
        ahead = calendar(
            pd.Timestamp(first, tz="UTC"), pd.Timestamp(first, tz="UTC") + pd.Timedelta(days=60)
        )
        trial_end = (_day(ahead[TRIAL_SESSIONS - 1]), _day(ahead[TRIAL_SESSIONS]))
    except (ValueError, IndexError):
        pass
    return Cadence(
        found,
        first,
        tuple(due),
        tuple(sessions_recorded),
        number,
        trial_end[0] if trial_end else None,
        trial_end[1] if trial_end else None,
        tuple(conditions),
    )


__all__ = [
    "CONFIRMATORY",
    "MAX_GAP",
    "TRIAL",
    "TRIAL_SESSIONS",
    "Cadence",
    "Condition",
    "cadence",
    "deadline",
    "opens",
    "phase_of",
]
