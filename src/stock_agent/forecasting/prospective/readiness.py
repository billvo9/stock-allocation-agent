"""
Launch readiness: which closes since the registration could have been
session 1, and why they were or were not recorded. Read-only reporting: it
never records, refuses or changes the session-1 rule (section 4 of
docs/research/prospective_registration.md: session 1 is the first record
after the registration). The store's reader and the exchange calendar are
injected, as everywhere on the research side.

`closes(entries, ...)`: every weekday from the registration time's New York
date through the first forecast or refusal record (or today), one `Close`:

    recorded        the session has a record (the first one ends the report)
    refused         no record, and the job would have refused it: the close
                    is not after the registration (`before_registration`),
                    some fund's window is incomplete or a batch records no
                    spot check (`not_first_eligible_close`), or the store
                    refused the window (`window_refused`, with its reason)
    not_recorded    no record, though the close was eligible: its complete,
                    spot-checked window was retrieved, per its batch records,
                    before its recording deadline (`operator_missed`; the store
                    keeps no separate write time)
    market_closed   a weekday that is not an exchange session
    pending         no record yet, and the recording deadline has not passed

Eligibility is judged as the job would have judged it inside the recording
window: from 18:00 New York on the session (or the registration entry's
append time, if later; a window that closes before the entry existed is
`before_registration`) to one second before the deadline (09:30 New York on
the next weekday). The store is read with retrieved_by at the deadline, at
the window's start and at each batch retrieval time inside it, because a
later batch replaces an earlier one; the close is eligible if any of those
readings is. The store is validated as it is today: a batch written after
the window that the reader refuses makes the close `window_refused`.

`pre_merge(...)`: before the registration pull request is merged, whether the
store already holds the complete, spot-checked window ending at the latest
available close (18:00 New York), read with retrieved_by = now.

`fixed_start(...)`: the proposed ETF-R1 view, NOT the rule in force. Session 1
would be the first exchange session whose 18:00 New York availability is
after the (GitHub-verified) registration time, whatever the job did; from it,
every session is expected, and a session without a forecast or refusal
recorded inside its window is missed. The start never moves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.schema import FundDataRefused
from stock_agent.forecasting.prospective import ledger, registration_record, window
from stock_agent.forecasting.prospective.cadence import RECORD_KINDS, TIMEZONE, deadline, opens

RECORDED, REFUSED, NOT_RECORDED, MISSED = "recorded", "refused", "not_recorded", "missed"
MARKET_CLOSED, PENDING = "market_closed", "pending"
STATUSES = (RECORDED, REFUSED, NOT_RECORDED, MARKET_CLOSED, PENDING)
LAST_MOMENT = pd.Timedelta(seconds=1)  # read the store as of just before the deadline

Loader = Callable[..., object]


@dataclass(frozen=True)
class Close:
    """One weekday's classification, with its machine-readable evidence."""

    day: str
    status: str
    reason: str | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "day": self.day,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }

    def line(self) -> str:
        shown = "; ".join(f"{key} {value}" for key, value in self.evidence.items())
        return f"close={self.day} status={self.status} reason={self.reason or '-'}: {shown}"


@dataclass(frozen=True)
class FixedStart:
    """The proposed ETF-R1 view (not in force): a fixed session 1 and every session after it."""

    session_1: str | None
    rows: tuple[Close, ...]

    @property
    def missed(self) -> tuple[str, ...]:
        return tuple(row.day for row in self.rows if row.status == MISSED)


@dataclass(frozen=True)
class PreMerge:
    close: str | None
    ready: bool
    evidence: Mapping[str, object]


window_gaps = window.session_one_gaps  # the job's own session-1 test


def _window(load_window, store_root, assets, close, *, lookback, calendar, retrieved_by):
    """The coerced window, or the store's refusal reason (a leak is never swallowed)."""

    try:
        result = load_window(
            store_root, list(assets), close, lookback=lookback, retrieved_by=retrieved_by
        )
        return window.coerce_window(
            result,
            assets=assets,
            forecast_close=close,
            lookback=lookback,
            calendar=calendar,
            retrieved_by=retrieved_by,
        ), None
    except FundDataRefused as error:
        return None, str(getattr(error.reason, "value", error.reason))
    except window.WindowRefused as error:
        return None, f"window_contract: {error}"


def _read(load_window, store_root, assets, close, *, lookback, calendar, retrieved_by):
    """Section 4, condition 2, at one decision time: (ready, evidence, batch retrieval times)."""

    win, refusal = _window(
        load_window,
        store_root,
        assets,
        close,
        lookback=lookback,
        calendar=calendar,
        retrieved_by=retrieved_by,
    )
    base = {"retrieved_by": retrieved_by.isoformat()}
    if win is None:
        return False, {**base, "refusal": refusal}, []
    gaps = window.session_one_gaps(win, assets)
    batches = sorted(str(batch) for batch in win.batches.get("batch_id", []))
    times = [pd.Timestamp(t) for t in win.batches.get("retrieved_at", [])]
    return window.session_one_ready(gaps), {**base, **gaps, "batches": batches}, times


def _eligibility(load_window, store_root, assets, close, *, lookback, calendar, start, end):
    """
    Whether the job could have recorded `close` at some decision time in
    [start, end]: the store is read at `end`, then at `start`, and at each
    batch retrieval time seen inside the interval and one second before it,
    since a later batch replaces an earlier one. The first time found
    eligible is the evidence; otherwise the reading at `end`, with every time
    checked.
    """

    common = {"lookback": lookback, "calendar": calendar}
    pending, checked, first = [end, start], [], None
    while pending:
        at = pending.pop(0)
        if at in checked:
            continue
        checked.append(at)
        ready, evidence, times = _read(
            load_window, store_root, assets, close, retrieved_by=at, **common
        )
        if ready:
            return True, evidence
        first = evidence if first is None else first
        # A batch retrieved inside the window replaced what was there before: read at its
        # retrieval time and one second earlier.
        later = {t for t in times if start < t <= end} | {
            t - LAST_MOMENT for t in times if start < t - LAST_MOMENT <= end
        }
        pending += sorted(t for t in later if t not in checked)
    shown = sorted(t.isoformat() for t in checked)
    return False, {**first, "checked_at": shown} if len(shown) > 1 else first


def _day(stamp: pd.Timestamp) -> str:
    return f"{stamp:%Y-%m-%d}"


def closes(
    entries: Sequence[ledger.LedgerEntry],
    *,
    calendar: window.Calendar,
    now: pd.Timestamp,
    load_window: Loader,
    store_root: str | Path | None,
    lookback: int,
    assets: Sequence[str] | None = None,
) -> tuple[Close, ...]:
    """
    Every weekday from the registration through the first record (module
    docstring). The funds are the registration record's unless given.
    """

    found = registration_record.current(entries)
    if found is None:
        return ()
    assets = list(found.record["assets"]) if assets is None else list(assets)
    stamp = pd.Timestamp(now).tz_convert("UTC")
    records = {}
    for entry in entries:
        session = entry.payload.get("as_of_session") if entry.kind in RECORD_KINDS else None
        if isinstance(session, str):
            records.setdefault(session, entry)
    start = found.registered_at.tz_convert(TIMEZONE).normalize().tz_localize(None)
    end = (
        pd.Timestamp(min(records))
        if records
        else stamp.tz_convert(TIMEZONE).normalize().tz_localize(None)
    )
    weekdays = pd.bdate_range(start, max(start, end))
    sessions = {
        _day(s) for s in calendar(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    }
    rows = []
    for weekday in weekdays:
        day = _day(weekday)
        if day not in sessions:
            rows.append(Close(day, MARKET_CLOSED))
            continue
        if day in records:
            entry = records[day]
            rows.append(Close(day, RECORDED, None, {"entry": entry.sequence, "kind": entry.kind}))
            break
        if opens(day) <= found.registered_at:
            rows.append(
                Close(
                    day,
                    REFUSED,
                    "before_registration",
                    {
                        "registered_at": found.registered_at.isoformat(),
                        "close_available_at": opens(day).isoformat(),
                    },
                )
            )
            continue
        limit = deadline(day)
        appended = pd.Timestamp(found.entry.recorded_at)
        if appended >= limit:  # the job refuses a record made before its registration entry
            rows.append(
                Close(
                    day,
                    REFUSED,
                    "before_registration",
                    {
                        "registration_entry_appended": appended.isoformat(),
                        "record_deadline": limit.isoformat(),
                    },
                )
            )
            continue
        if stamp < limit:
            rows.append(Close(day, PENDING))
            continue
        ready, evidence = _eligibility(
            load_window,
            store_root,
            assets,
            pd.Timestamp(day, tz="UTC"),
            lookback=lookback,
            calendar=calendar,
            start=max(opens(day), appended),
            end=limit - LAST_MOMENT,
        )
        if ready:
            rows.append(Close(day, NOT_RECORDED, "operator_missed", evidence))
        elif "refusal" in evidence:
            rows.append(Close(day, REFUSED, "window_refused", evidence))
        else:
            rows.append(Close(day, REFUSED, "not_first_eligible_close", evidence))
    return tuple(rows)


def fixed_start(
    entries: Sequence[ledger.LedgerEntry], *, calendar: window.Calendar, now: pd.Timestamp
) -> FixedStart:
    """The proposed ETF-R1 view (module docstring); reports only, never decides."""

    found = registration_record.current(entries)
    if found is None:
        return FixedStart(None, ())
    stamp = pd.Timestamp(now).tz_convert("UTC")
    start = pd.Timestamp(found.registered_at.tz_convert(TIMEZONE).date(), tz="UTC")
    today = pd.Timestamp(stamp.tz_convert(TIMEZONE).date(), tz="UTC")
    # Session 1 is known at the registration, before its date arrives.
    ahead = calendar(start, max(start + pd.Timedelta(days=14), today))
    eligible = [_day(s) for s in ahead if opens(_day(s)) > found.registered_at]
    if not eligible:
        return FixedStart(None, ())
    session_1 = eligible[0]
    eligible = [day for day in eligible if pd.Timestamp(day, tz="UTC") <= today]
    records: dict[str, list[ledger.LedgerEntry]] = {}
    for entry in entries:
        session = entry.payload.get("as_of_session") if entry.kind in RECORD_KINDS else None
        if isinstance(session, str):
            records.setdefault(session, []).append(entry)
    rows = []
    for number, day in enumerate(eligible, start=1):
        evidence = {"number": number}
        inside = [
            entry
            for entry in records.get(day, [])
            if opens(day) <= pd.Timestamp(entry.recorded_at) < deadline(day)
        ]
        if inside:
            entry = inside[0]
            rows.append(
                Close(
                    day, RECORDED, None, {**evidence, "entry": entry.sequence, "kind": entry.kind}
                )
            )
        elif stamp < deadline(day):
            rows.append(Close(day, PENDING, None, evidence))
        elif records.get(day):
            rows.append(Close(day, MISSED, "record_outside_window", evidence))
        else:
            rows.append(Close(day, MISSED, "no_record", evidence))
    return FixedStart(session_1, tuple(rows))


def pre_merge(
    *,
    load_window: Loader,
    store_root: str | Path | None,
    assets: Sequence[str],
    calendar: window.Calendar,
    now: pd.Timestamp,
    lookback: int,
) -> PreMerge:
    """Whether the store already holds the verified window ending at the latest available close."""

    stamp = pd.Timestamp(now).tz_convert("UTC")
    today = stamp.tz_convert(TIMEZONE).normalize().tz_localize(None)
    try:
        recent = calendar(
            pd.Timestamp(today - pd.Timedelta(days=10), tz="UTC"), pd.Timestamp(today, tz="UTC")
        )
    except ValueError:  # CalendarRefused: the date is outside the calendar's coverage
        return PreMerge(None, False, {"refusal": "calendar_refused"})
    available = [s for s in recent if opens(_day(s)) <= stamp]
    if not available:
        return PreMerge(None, False, {"refusal": "no_available_close"})
    close = pd.Timestamp(available[-1]).tz_convert("UTC").normalize()
    ready, evidence, _ = _read(
        load_window,
        store_root,
        assets,
        close,
        lookback=lookback,
        calendar=calendar,
        retrieved_by=stamp,
    )
    return PreMerge(_day(close), ready, evidence)


__all__ = [
    "MARKET_CLOSED",
    "MISSED",
    "NOT_RECORDED",
    "PENDING",
    "RECORDED",
    "REFUSED",
    "STATUSES",
    "Close",
    "FixedStart",
    "PreMerge",
    "closes",
    "fixed_start",
    "pre_merge",
    "window_gaps",
]
