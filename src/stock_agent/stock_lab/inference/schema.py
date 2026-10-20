"""
The inference store's contract, shared by its writer (`store`) and its two
readers (`window`, `recent`). Standard library, numpy and pandas only: no
calendar is imported (callers pass one) and no clock is read.

Location (private, never in Git)
    The file `stock_lab_inference.sqlite3` in the directory named by
    STOCK_AGENT_STOCK_LAB_INFERENCE_HOME, else
    ~/stock-agent-private/stock_lab_inference/. A directory inside a Git
    working tree is refused (stricter than the research store: not even an
    ignored data/raw/), and so is a directory holding the research store's
    file: the two stores never share a directory. The file must have this
    name, so code that never names it cannot open it.

Identity
    The file header's `application_id` is APPLICATION_ID, and the `meta`
    table holds exactly the store kind and the schema version. The research
    store refuses any file with a non-zero application id, so it can neither
    read this file nor add its tables to it; this store refuses any file
    without its id.

Tables (write-once: UPDATE and DELETE abort, an insert onto an existing key
aborts, and a batch cannot gain rows once its count is reached)
    meta     store_kind and schema_version, written once at creation
    batches  one retrieval of one ticker's bars: batch_id, ticker, source,
             retrieved_at (when the provider served it, UTC), recorded_at
             (when the store wrote it, by the store's clock), request_start
             and request_end (the requested sessions, both inclusive),
             calendar_id (the exchange calendar its rows follow), row_count,
             content_sha256, data_class, vintage_note
    bars     the batch's rows, prices.BAR_COLUMNS: exactly one per calendar
             session of the request, so a session the provider did not serve
             is a row without prices (a gap stays a gap), and every session
             a batch covers is a row of it

Timing conventions (assumed and recorded here; not vendor publication times)
    CLOSE_AVAILABLE  20:00 America/New_York on the session, the daily job's
                     convention: the earliest time a close counts as final. A
                     batch retrieved earlier than its last session's close was
                     available is refused (it could hold a partial session).
    SESSION_OPEN     09:30 America/New_York (`forecast.SESSION_OPEN`): a
                     decision for an as-of close must be made before the next
                     session opens.
    RECORD_WITHIN    a batch is recorded when it is retrieved: its retrieved_at
                     is at most one hour before the store's clock and never
                     after it, so no batch can claim an earlier retrieval than
                     it had.

Vintage rule (both readers)
    For each session, the latest-retrieved batch whose request covers it
    (ties: the larger batch id); never a fallback to an older batch. A
    session whose values differ between covering batches is reported as
    revised; nothing is overwritten.
"""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import date, datetime, time
from enum import StrEnum
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.contracts import DEVELOPMENT_END
from stock_agent.stock_lab.prices import BAR_COLUMNS
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME

INFERENCE_SESSIONS = 252  # decision A2: the trailing input window, in sessions (closes)
PROTECTED_START = date.fromordinal(DEVELOPMENT_END.toordinal() + 1)  # 2025-01-01
HOME_ENV = "STOCK_AGENT_STOCK_LAB_INFERENCE_HOME"
DIRECTORY = "stock_lab_inference"
DB_NAME = "stock_lab_inference.sqlite3"
APPLICATION_ID = int.from_bytes(b"SLIW", "big")  # "Stock Lab inference window"
STORE_KIND = "stock_lab_inference"
SCHEMA_VERSION = "1"
NEW_YORK = "America/New_York"
CLOSE_AVAILABLE = time(20, 0)
SESSION_OPEN = time(9, 30)
RECORD_WITHIN = pd.Timedelta(hours=1)

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]


class Refusal(StrEnum):
    """Why the inference store refused a request or a batch (machine-readable)."""

    INVALID_REQUEST = "invalid_request"
    WRONG_LOOKBACK = "wrong_lookback"
    BEFORE_PROTECTED_PERIOD = "before_protected_period"
    CLOSE_NOT_AVAILABLE = "close_not_available"
    AFTER_NEXT_OPEN = "after_next_open"
    NO_STORE = "no_store"
    NOT_INFERENCE_STORE = "not_inference_store"
    SHORT_WINDOW = "short_window"
    AS_OF_CLOSE_MISSING = "as_of_close_missing"
    CALENDAR_MISMATCH = "calendar_mismatch"
    INTEGRITY = "integrity"
    LOCATION = "location"
    NOT_ON_CALENDAR = "not_on_calendar"
    RETRIEVAL_TIME = "retrieval_time"
    DATA_CLASS = "data_class"
    INVALID_BARS = "invalid_bars"
    DATABASE_ERROR = "database_error"


class InferenceRefused(ValueError):
    """A refusal with a machine-readable `reason` (a Refusal) and plain words."""

    def __init__(self, reason: Refusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class InferenceBatch:
    """One stored retrieval (the `batches` row); every time is a UTC stamp string."""

    batch_id: str
    ticker: str
    source: str
    retrieved_at: str
    recorded_at: str
    request_start: str
    request_end: str
    calendar_id: str
    row_count: int
    content_sha256: str
    data_class: str
    vintage_note: str

    def covers(self, session: pd.Timestamp) -> bool:
        return self.request_start <= session.strftime("%Y-%m-%d") <= self.request_end


BATCH_FIELDS = tuple(field.name for field in fields(InferenceBatch))


# --- location and identity ---


def default_path() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).expanduser() / DB_NAME
    return Path.home() / "stock-agent-private" / DIRECTORY / DB_NAME


def check_location(path: Path) -> Path:
    """The resolved path, after the location rules in the module docstring."""

    resolved = Path(path).expanduser().resolve()
    if any((folder / ".git").exists() for folder in resolved.parents):
        raise InferenceRefused(
            Refusal.LOCATION, "the inference store must be outside every Git working tree"
        )
    if resolved.name != DB_NAME:
        raise InferenceRefused(Refusal.LOCATION, f"the inference store's file is named {DB_NAME}")
    if (resolved.parent / RESEARCH_DB_NAME).exists():
        raise InferenceRefused(
            Refusal.LOCATION,
            "this directory holds the research store; the inference store needs its own",
        )
    return resolved


def check_identity(connection: sqlite3.Connection) -> None:
    """Refuses (not_inference_store) a database that is not this store at this version."""

    try:
        (application,) = connection.execute("PRAGMA application_id").fetchone()
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError:
        application, meta = None, None
    if application != APPLICATION_ID or meta != {
        "store_kind": STORE_KIND,
        "schema_version": SCHEMA_VERSION,
    }:
        raise InferenceRefused(
            Refusal.NOT_INFERENCE_STORE, "the file is not a Stock Lab inference store (version 1)"
        )


@contextmanager
def read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """
    A read-only, query-only connection to the store at `path` (identity
    checked), or None when the file does not exist. Opening it creates and
    changes nothing. Any SQLite error while it is open (a locked or damaged
    file) is refused as `database_error`, never raised as it is.
    """

    resolved = check_location(path)
    if not resolved.is_file():
        yield None
        return
    try:
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error as error:
        raise InferenceRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
    try:
        connection.execute("PRAGMA query_only = ON")
        check_identity(connection)
        yield connection
    except sqlite3.Error as error:
        raise InferenceRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
    finally:
        connection.close()


# --- values ---


def stamp(value: pd.Timestamp) -> str:
    """UTC with microseconds and a Z (the research store's format)."""

    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_time(value: object, label: str) -> pd.Timestamp:
    """A timezone-aware time, in UTC; refuses a naive or unparseable one."""

    if not isinstance(value, datetime):
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} must be a timezone-aware time")
    when = pd.Timestamp(value)
    if when.tzinfo is None:
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} must be timezone-aware")
    return when.tz_convert("UTC")


def session_day(value: object, label: str) -> pd.Timestamp:
    """A session date as 00:00 UTC: YYYY-MM-DD, a date, or a naive or UTC midnight."""

    try:
        day = pd.Timestamp(value) if isinstance(value, str | date) else pd.NaT
    except (TypeError, ValueError):
        day = pd.NaT
    if day is pd.NaT:
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} is not a date")
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    elif str(day.tz) != "UTC":
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} must be naive or UTC")
    if day != day.normalize():
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} must be a date (00:00)")
    return day.as_unit("us")


def new_york(session: pd.Timestamp, clock_time: time) -> pd.Timestamp:
    """`clock_time` in New York on the session's date, in UTC."""

    local = pd.Timestamp(datetime.combine(session.date(), clock_time)).tz_localize(NEW_YORK)
    return local.tz_convert("UTC")


def close_available_at(session: pd.Timestamp) -> pd.Timestamp:
    return new_york(session, CLOSE_AVAILABLE)


def opens_at(session: pd.Timestamp) -> pd.Timestamp:
    return new_york(session, SESSION_OPEN)


def calendar(sessions: Sessions, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """The calendar's sessions in [start, end], checked: UTC midnights, increasing, unique."""

    try:
        found = pd.DatetimeIndex(sessions(start, end))
    except (ValueError, TypeError, KeyError) as error:
        raise InferenceRefused(
            Refusal.INVALID_REQUEST,
            f"the calendar refused {start.date()} to {end.date()} ({type(error).__name__})",
        ) from None
    if found.tz is None or str(found.tz) != "UTC":
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the calendar must give UTC sessions")
    if not (found == found.normalize()).all() or not found.is_monotonic_increasing:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the calendar's sessions are not dates")
    if found.has_duplicates:
        raise InferenceRefused(Refusal.INVALID_REQUEST, "the calendar repeats a session")
    return found.as_unit("us")


def bars_sha256(bars: pd.DataFrame) -> str:
    """The content hash of bars (BAR_COLUMNS, sessions as dates), as the research store hashes."""

    buffer = io.StringIO()
    frame = bars.loc[:, list(BAR_COLUMNS)].copy()
    frame["session"] = pd.DatetimeIndex(frame["session"]).strftime("%Y-%m-%d")
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


# --- reading batches ---


def batches(connection: sqlite3.Connection, ticker: str) -> tuple[InferenceBatch, ...]:
    """The ticker's batches, oldest retrieval first."""

    rows = connection.execute(
        f"SELECT {', '.join(BATCH_FIELDS)} FROM batches WHERE ticker = ? "
        "ORDER BY retrieved_at, batch_id",
        (ticker,),
    ).fetchall()
    return tuple(InferenceBatch(*row) for row in rows)


def bars_of(connection: sqlite3.Connection, chosen: Sequence[InferenceBatch]) -> pd.DataFrame:
    """
    The rows of `chosen` (batch_id plus BAR_COLUMNS, sessions as UTC
    timestamps), each batch checked against its recorded count, content hash
    and request range (integrity).
    """

    columns = ["batch_id", *BAR_COLUMNS]
    if not chosen:
        return pd.DataFrame(columns=columns)
    placeholders = ", ".join("?" for _ in chosen)
    rows = connection.execute(
        f"SELECT {', '.join(columns)} FROM bars WHERE batch_id IN ({placeholders})",
        [batch.batch_id for batch in chosen],
    ).fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    try:
        frame["session"] = pd.to_datetime(frame["session"], utc=True).dt.as_unit("us")
        for column in BAR_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column]).astype("float64")
    except (TypeError, ValueError):
        raise InferenceRefused(
            Refusal.INTEGRITY, "a stored value is not a date or a number"
        ) from None
    for batch in chosen:
        own = frame[frame["batch_id"] == batch.batch_id].sort_values("session")
        if len(own) != batch.row_count or own.empty:
            raise InferenceRefused(
                Refusal.INTEGRITY,
                f"batch {batch.batch_id} has {len(own)} rows, not {batch.row_count}",
            )
        if bars_sha256(own) != batch.content_sha256:
            raise InferenceRefused(
                Refusal.INTEGRITY, f"batch {batch.batch_id} does not match its content hash"
            )
        first, last = (own["session"].iloc[i].strftime("%Y-%m-%d") for i in (0, -1))
        if (first, last) != (batch.request_start, batch.request_end):
            raise InferenceRefused(
                Refusal.INTEGRITY, f"batch {batch.batch_id} does not cover its request"
            )
    return frame.sort_values(["session", "batch_id"]).reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class Vintage:
    """Per session, the selected batch's row; and the sessions revised between batches."""

    bars: pd.DataFrame  # session plus BAR_COLUMNS[1:] plus batch_id, one row per session
    revised: tuple[str, ...]
    used: tuple[InferenceBatch, ...]


def select_vintage(chosen: Sequence[InferenceBatch], frame: pd.DataFrame) -> Vintage:
    """The vintage rule in the module docstring, over the rows of `chosen`."""

    order = sorted(chosen, key=lambda batch: (batch.retrieved_at, batch.batch_id))
    rank = {batch.batch_id: position for position, batch in enumerate(order)}
    ranked = frame.assign(rank=frame["batch_id"].map(rank))
    values = list(BAR_COLUMNS[1:])
    variants = ranked.groupby("session")[values].nunique(dropna=False)
    revised = tuple(
        session.strftime("%Y-%m-%d") for session in variants.index[(variants > 1).any(axis=1)]
    )
    latest = ranked.sort_values(["session", "rank"]).groupby("session").tail(1)
    latest = latest.sort_values("session").reset_index(drop=True)
    used = set(latest["batch_id"])
    return Vintage(
        bars=latest[["session", *values, "batch_id"]],
        revised=revised,
        used=tuple(batch for batch in order if batch.batch_id in used),
    )


def eligible_batches(
    stored: Sequence[InferenceBatch], decision: pd.Timestamp, calendar_id: str
) -> list[InferenceBatch]:
    """
    The batches retrieved AND recorded by `decision` (both readers' cut-off), each re-checked:
    on `calendar_id` (calendar_mismatch otherwise), retrieved no earlier than its last close
    was available and never after it was recorded (integrity otherwise).
    """

    eligible = []
    for batch in stored:
        if (
            pd.Timestamp(batch.recorded_at) > decision
            or pd.Timestamp(batch.retrieved_at) > decision
        ):
            continue
        if batch.calendar_id != calendar_id:
            raise InferenceRefused(
                Refusal.CALENDAR_MISMATCH,
                f"batch {batch.batch_id} follows {batch.calendar_id}, not {calendar_id}",
            )
        end = session_day(batch.request_end, "request_end")
        if pd.Timestamp(batch.retrieved_at) < close_available_at(end):
            raise InferenceRefused(
                Refusal.INTEGRITY,
                f"batch {batch.batch_id} claims a retrieval before its last close was available",
            )
        if pd.Timestamp(batch.retrieved_at) > pd.Timestamp(batch.recorded_at):
            raise InferenceRefused(
                Refusal.INTEGRITY, f"batch {batch.batch_id} was recorded before it was retrieved"
            )
        eligible.append(batch)
    return eligible


def data_class_of(chosen: Sequence[InferenceBatch]) -> str:
    """The one data class of `chosen`; mixing real and synthetic batches is refused."""

    classes = {batch.data_class for batch in chosen}
    if len(classes) != 1:
        raise InferenceRefused(Refusal.INTEGRITY, "a ticker has both real and synthetic batches")
    return classes.pop()
