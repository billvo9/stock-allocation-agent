"""
The private holdout store: the protected-period bars fetched for the one
sealed evaluation, and that evaluation's record. Write-once, and spent by its
one run: once the evaluation is recorded, the bars are never handed out
again and nothing can be added.

Location (private, never in Git)
    The file `stock_lab_holdout.sqlite3` in the directory named by
    STOCK_AGENT_STOCK_LAB_HOLDOUT_HOME, else ~/stock-agent-private/
    stock_lab_holdout/. Refused: a directory inside a Git working tree, a
    file of another name, and a directory holding any other SQLite database
    (the research or the inference store's, for instance): the holdout store
    has a directory of its own.

Identity
    The file header's `application_id` is APPLICATION_ID ("SLHO"), and the
    `meta` table holds exactly the store kind and the schema version. The
    research store refuses any file with a non-zero application id, so it can
    neither read this file nor add its tables to it; this store refuses any
    file without its id.

Tables (write-once: UPDATE and DELETE abort, and an insert onto an existing
key aborts)
    meta        store_kind and schema_version, written once at creation
    batches     one retrieval of one stock's bars, at most one per stock
                ever (a second fetch is refused: already_fetched):
                batch_id, ticker, source, retrieved_at, recorded_at,
                request_start and request_end (the protocol's fetched
                sessions), calendar_id, row_count, content_sha256,
                protocol_sha256, vintage_note
    bars        the batch's rows (prices.BAR_COLUMNS): exactly one per
                calendar session of the request, so a session the provider
                did not serve is a row without prices
    evaluation  at most one row, ever: model_id, protocol_sha256, record_id,
                recorded_at and the record (canonical JSON). Once it exists,
                no batch or bar can be added (spent)

How the bars are read: only by `evaluate_once`, which, inside one write
transaction, refuses when an evaluation exists, reads and checks every
stock's bars, hands them to the run, and records the run's result before the
transaction ends. A run that fails records nothing (it can be repeated, since
nobody saw a result); a run that succeeds spends the period. `batches` gives
lineage only and `evaluation` the stored result; neither returns a price.

What write-once means here: the triggers and checks prevent accidental
reuse through this code (a second fetch, a second run, a changed bar). They
do not stop someone who edits the file by hand, deletes it, or points the
environment variable at a new directory; the record's id and the batches'
content hashes make such a change detectable when they are kept elsewhere.

Timing: a batch is recorded when it is retrieved (at most RECORD_WITHIN
before the store's clock and never after it), and no earlier than the last
session's close was available (20:00 America/New_York).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.frozen_model import canonical
from stock_agent.stock_lab.holdout.protocol import (
    PROTOCOL,
    HoldoutRefused,
    Protocol,
    Refusal,
    Sessions,
    record_id,
)
from stock_agent.stock_lab.prices import BAR_COLUMNS, BarsRefused, validate_bars
from stock_agent.stock_lab.store import stamp, utc_now

HOME_ENV = "STOCK_AGENT_STOCK_LAB_HOLDOUT_HOME"
DIRECTORY = "stock_lab_holdout"
DB_NAME = "stock_lab_holdout.sqlite3"
APPLICATION_ID = int.from_bytes(b"SLHO", "big")  # "Stock Lab holdout"
STORE_KIND = "stock_lab_holdout"
SCHEMA_VERSION = "1"
NEW_YORK = "America/New_York"
CLOSE_AVAILABLE = time(20, 0)
RECORD_WITHIN = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class HoldoutBatch:
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
    protocol_sha256: str
    vintage_note: str

    def lineage(self) -> dict:
        return {name.name: getattr(self, name.name) for name in fields(self)}


BATCH_FIELDS = tuple(field.name for field in fields(HoldoutBatch))

STATEMENTS = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE batches (
        batch_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        request_start TEXT NOT NULL,
        request_end TEXT NOT NULL,
        calendar_id TEXT NOT NULL,
        row_count INTEGER NOT NULL CHECK (row_count > 0),
        content_sha256 TEXT NOT NULL,
        protocol_sha256 TEXT NOT NULL,
        vintage_note TEXT NOT NULL,
        CHECK (request_start <= request_end),
        CHECK (retrieved_at <= recorded_at)
    )""",
    """CREATE TABLE bars (
        batch_id TEXT NOT NULL REFERENCES batches (batch_id),
        session TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        div_cash REAL NOT NULL,
        split_factor REAL NOT NULL,
        PRIMARY KEY (batch_id, session)
    )""",
    """CREATE TABLE evaluation (
        slot INTEGER PRIMARY KEY CHECK (slot = 1),
        model_id TEXT NOT NULL UNIQUE,
        protocol_sha256 TEXT NOT NULL,
        record_id TEXT NOT NULL UNIQUE,
        recorded_at TEXT NOT NULL,
        record TEXT NOT NULL
    )""",
    *(
        f"CREATE TRIGGER {table}_no_{action} BEFORE {action.upper()} ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'write-once table'); END"
        for table in ("meta", "batches", "bars", "evaluation")
        for action in ("update", "delete")
    ),
    *(
        f"CREATE TRIGGER {table}_no_replace BEFORE INSERT ON {table} "
        f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {key}) "
        "BEGIN SELECT RAISE(ABORT, 'write-once table'); END"
        for table, key in (
            ("meta", "key = NEW.key"),
            ("batches", "batch_id = NEW.batch_id OR ticker = NEW.ticker"),
            ("bars", "batch_id = NEW.batch_id AND session = NEW.session"),
            ("evaluation", "1"),
        )
    ),
    (
        "CREATE TRIGGER bars_complete BEFORE INSERT ON bars "
        "WHEN (SELECT COUNT(*) FROM bars WHERE batch_id = NEW.batch_id) "
        ">= (SELECT row_count FROM batches WHERE batch_id = NEW.batch_id) "
        "BEGIN SELECT RAISE(ABORT, 'write-once table: the batch is complete'); END"
    ),
    *(
        f"CREATE TRIGGER {table}_spent BEFORE INSERT ON {table} "
        "WHEN EXISTS (SELECT 1 FROM evaluation) "
        "BEGIN SELECT RAISE(ABORT, 'the holdout is spent'); END"
        for table in ("batches", "bars")
    ),
)


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
        raise HoldoutRefused(
            Refusal.LOCATION, "the holdout store must be outside every Git working tree"
        )
    if resolved.name != DB_NAME:
        raise HoldoutRefused(Refusal.LOCATION, f"the holdout store's file is named {DB_NAME}")
    if resolved.parent.is_dir() and any(
        other.name != DB_NAME for other in resolved.parent.glob("*.sqlite3")
    ):
        raise HoldoutRefused(
            Refusal.LOCATION,
            "this directory holds another database; the holdout store needs its own",
        )
    return resolved


def check_identity(connection: sqlite3.Connection) -> None:
    """Refuses (not_holdout_store) a database that is not this store at this version."""

    try:
        (application,) = connection.execute("PRAGMA application_id").fetchone()
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError:
        application, meta = None, None
    if application != APPLICATION_ID or meta != {
        "store_kind": STORE_KIND,
        "schema_version": SCHEMA_VERSION,
    }:
        raise HoldoutRefused(
            Refusal.NOT_HOLDOUT_STORE, "the file is not a Stock Lab holdout store (version 1)"
        )


# --- values ---


def _utc(value: object, label: str) -> pd.Timestamp:
    if not isinstance(value, datetime) or pd.Timestamp(value).tzinfo is None:
        raise HoldoutRefused(Refusal.INVALID_REQUEST, f"{label} must be a timezone-aware time")
    return pd.Timestamp(value).tz_convert("UTC")


def _text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HoldoutRefused(Refusal.INVALID_REQUEST, f"{label} must be non-empty text")
    return value


def close_available_at(session: pd.Timestamp) -> pd.Timestamp:
    local = pd.Timestamp(datetime.combine(session.date(), CLOSE_AVAILABLE)).tz_localize(NEW_YORK)
    return local.tz_convert("UTC")


def bars_sha256(bars: pd.DataFrame) -> str:
    """The content hash of bars (BAR_COLUMNS, sessions as dates), as the other stores hash."""

    buffer = io.StringIO()
    frame = bars.loc[:, list(BAR_COLUMNS)].copy()
    frame["session"] = pd.DatetimeIndex(frame["session"]).strftime("%Y-%m-%d")
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def _batches(connection: sqlite3.Connection) -> tuple[HoldoutBatch, ...]:
    rows = connection.execute(
        f"SELECT {', '.join(BATCH_FIELDS)} FROM batches ORDER BY ticker"
    ).fetchall()
    return tuple(HoldoutBatch(*row) for row in rows)


def _bars_of(connection: sqlite3.Connection, batch: HoldoutBatch) -> pd.DataFrame:
    """The batch's rows (BAR_COLUMNS, sessions as UTC timestamps), checked (integrity)."""

    rows = connection.execute(
        f"SELECT {', '.join(BAR_COLUMNS)} FROM bars WHERE batch_id = ? ORDER BY session",
        (batch.batch_id,),
    ).fetchall()
    frame = pd.DataFrame(rows, columns=list(BAR_COLUMNS))
    try:
        frame["session"] = pd.to_datetime(frame["session"], utc=True).dt.as_unit("us")
        for column in BAR_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column]).astype("float64")
    except (TypeError, ValueError):
        raise HoldoutRefused(
            Refusal.INTEGRITY, "a stored value is not a date or a number"
        ) from None
    if len(frame) != batch.row_count or frame.empty:
        raise HoldoutRefused(
            Refusal.INTEGRITY,
            f"batch {batch.batch_id} has {len(frame)} rows, not {batch.row_count}",
        )
    if bars_sha256(frame) != batch.content_sha256:
        raise HoldoutRefused(
            Refusal.INTEGRITY, f"batch {batch.batch_id} does not match its content hash"
        )
    first, last = (frame["session"].iloc[i].strftime("%Y-%m-%d") for i in (0, -1))
    if (first, last) != (batch.request_start, batch.request_end):
        raise HoldoutRefused(
            Refusal.INTEGRITY, f"batch {batch.batch_id} does not cover its request"
        )
    return frame


def _checked_record(row: Sequence) -> dict:
    """The stored record, refused (integrity) unless it is exactly the sealed record stored."""

    model_id, protocol_sha256, stored_id, _, text = row
    try:
        record = json.loads(text)
        sealed = (
            isinstance(record, dict)
            and isinstance(record.get("model"), dict)
            and canonical(record) == text
            and record.get("record_id") == stored_id == record_id(record)
            and record["model"].get("model_id") == model_id
            and record.get("protocol_sha256") == protocol_sha256
        )
    except (TypeError, ValueError):  # not JSON, or not canonical JSON
        sealed = False
    if not sealed:
        raise HoldoutRefused(Refusal.INTEGRITY, "the evaluation record does not match its id")
    return record


class HoldoutStore:
    """The holdout store at `path` (default `default_path()`), for `protocol`."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        protocol: Protocol = PROTOCOL,
        clock: Callable[[], pd.Timestamp] = utc_now,
    ) -> None:
        self.path = check_location(path or default_path())
        self.protocol = protocol
        self._clock = clock

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection | None]:
        resolved = check_location(self.path)
        if not resolved.is_file():
            yield None
            return
        try:
            connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=10.0)
        except sqlite3.Error as error:
            raise HoldoutRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
        try:
            connection.execute("PRAGMA query_only = ON")
            check_identity(connection)
            yield connection
        except sqlite3.Error as error:
            raise HoldoutRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        check_location(self.path)
        directory = self.path.parent
        for folder in reversed((directory, *directory.parents)):
            if not folder.exists():  # every missing folder, parents too, private
                folder.mkdir(mode=0o700)
        if directory.stat().st_mode & 0o077:
            raise HoldoutRefused(
                Refusal.LOCATION, f"{directory} is open to other users; restrict it (chmod 700)"
            )
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            pass  # created earlier, or by another writer at the same moment
        if self.path.stat().st_mode & 0o077:
            raise HoldoutRefused(
                Refusal.LOCATION, f"{self.path} is open to other users; restrict it (chmod 600)"
            )
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            try:
                (application,) = connection.execute("PRAGMA application_id").fetchone()
                (objects,) = connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            except sqlite3.DatabaseError:
                application, objects = None, None
            if application == 0 and objects == 0:  # an empty file this writer just created
                connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                for statement in STATEMENTS:
                    connection.execute(statement)
                connection.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    (("store_kind", STORE_KIND), ("schema_version", SCHEMA_VERSION)),
                )
            check_identity(connection)
            yield connection
        except BaseException as error:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):  # a locked, damaged or foreign file
                raise HoldoutRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
            raise
        else:
            try:
                connection.execute("COMMIT")
            except sqlite3.Error as error:
                raise HoldoutRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
        finally:
            connection.close()

    # --- reading (no prices) ---

    def batches(self) -> tuple[HoldoutBatch, ...]:
        """The stored batches (lineage only, no bars), by ticker."""

        with self._read() as connection:
            return () if connection is None else _batches(connection)

    def evaluation(self) -> dict | None:
        """The one evaluation record, re-checked against its id, or None before the run."""

        with self._read() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT model_id, protocol_sha256, record_id, recorded_at, record FROM evaluation"
            ).fetchone()
        return None if row is None else _checked_record(row)

    # --- writing ---

    def record_batch(
        self,
        ticker: str,
        bars: pd.DataFrame,
        *,
        source: str,
        retrieved_at: pd.Timestamp,
        sessions: Sessions,
        calendar_id: str,
        vintage_note: str,
    ) -> HoldoutBatch:
        """Store one stock's fetched bars (once per stock, ever); see the module docstring."""

        protocol = self.protocol
        if ticker not in protocol.tickers:
            raise HoldoutRefused(Refusal.INVALID_REQUEST, f"{ticker!r} is not a protocol stock")
        texts = {
            label: _text(label, value)
            for label, value in (
                ("source", source),
                ("calendar_id", calendar_id),
                ("vintage_note", vintage_note),
            )
        }
        expected = protocol.fetch_sessions(sessions)
        try:
            checked = validate_bars(bars)
        except BarsRefused as error:
            raise HoldoutRefused(Refusal.INVALID_BARS, str(error)) from None
        if not pd.DatetimeIndex(checked["session"]).as_unit("us").equals(expected):
            raise HoldoutRefused(
                Refusal.INVALID_BARS,
                f"{len(checked)} rows for the protocol's {len(expected)} sessions: a batch holds "
                "exactly one row per session (a gap is a row without prices)",
            )
        when = _utc(retrieved_at, "retrieved_at")
        now = _utc(self._clock(), "the store's clock")
        if when > now or when < now - RECORD_WITHIN:
            raise HoldoutRefused(
                Refusal.RETRIEVAL_TIME,
                f"a batch is recorded when it is retrieved (within {RECORD_WITHIN}, never later)",
            )
        if when < close_available_at(expected[-1]):
            raise HoldoutRefused(
                Refusal.RETRIEVAL_TIME,
                f"retrieved before the {expected[-1].date()} close was available "
                "(20:00 America/New_York)",
            )
        digest = bars_sha256(checked)
        retrieved = stamp(when)
        compact = retrieved.replace("-", "").replace(":", "").replace(".", "")
        batch = HoldoutBatch(
            batch_id=f"{ticker}-{compact}-{digest[:12]}",
            ticker=ticker,
            source=texts["source"],
            retrieved_at=retrieved,
            recorded_at=stamp(now),
            request_start=expected[0].strftime("%Y-%m-%d"),
            request_end=expected[-1].strftime("%Y-%m-%d"),
            calendar_id=texts["calendar_id"],
            row_count=len(checked),
            content_sha256=digest,
            protocol_sha256=protocol.sha256(),
            vintage_note=texts["vintage_note"],
        )
        with self._write() as connection:
            if connection.execute("SELECT 1 FROM evaluation").fetchone() is not None:
                raise HoldoutRefused(Refusal.SPENT, "the holdout is spent: nothing can be added")
            if connection.execute("SELECT 1 FROM batches WHERE ticker = ?", (ticker,)).fetchone():
                raise HoldoutRefused(
                    Refusal.ALREADY_FETCHED, f"{ticker} is already stored: it is fetched once"
                )
            connection.execute(
                f"INSERT INTO batches ({', '.join(BATCH_FIELDS)}) "
                f"VALUES ({', '.join('?' for _ in BATCH_FIELDS)})",
                tuple(getattr(batch, name) for name in BATCH_FIELDS),
            )
            connection.executemany(
                "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        batch.batch_id,
                        row.session.strftime("%Y-%m-%d"),
                        *(
                            None if pd.isna(value) else float(value)
                            for value in (row.open, row.high, row.low, row.close, row.volume)
                        ),
                        float(row.div_cash),
                        float(row.split_factor),
                    )
                    for row in checked.itertuples()
                ],
            )
        return batch

    def evaluate_once(
        self,
        run: Callable[[dict[str, pd.DataFrame], tuple[HoldoutBatch, ...]], dict],
        *,
        calendar_id: str,
    ) -> dict:
        """
        The one evaluation (module docstring): `run(histories, batches)` gets every
        protocol stock's bars and returns the record, which is stored before this
        returns. Refused when the period is spent, a stock is missing, or a batch
        follows another protocol or calendar.
        """

        protocol = self.protocol
        with self._write() as connection:
            if connection.execute("SELECT 1 FROM evaluation").fetchone() is not None:
                raise HoldoutRefused(Refusal.SPENT, "the holdout is spent: it was evaluated once")
            stored = _batches(connection)
            missing = sorted(set(protocol.tickers) - {batch.ticker for batch in stored})
            if missing:
                raise HoldoutRefused(Refusal.INCOMPLETE, f"not fetched yet: {', '.join(missing)}")
            for batch in stored:
                if batch.protocol_sha256 != protocol.sha256():
                    raise HoldoutRefused(
                        Refusal.PROTOCOL, f"batch {batch.batch_id} follows another protocol"
                    )
                if batch.calendar_id != calendar_id:
                    raise HoldoutRefused(
                        Refusal.PROTOCOL, f"batch {batch.batch_id} follows {batch.calendar_id}"
                    )
            histories = {batch.ticker: _bars_of(connection, batch) for batch in stored}
            record = run(histories, stored)
            if (
                not isinstance(record, dict)
                or record.get("record_id") != record_id(record)
                or record.get("protocol_sha256") != protocol.sha256()
                or not isinstance(record.get("model"), dict)
            ):
                raise HoldoutRefused(Refusal.INTEGRITY, "the run returned an unsealed record")
            now = stamp(_utc(self._clock(), "the store's clock"))
            connection.execute(
                "INSERT INTO evaluation (slot, model_id, protocol_sha256, record_id, recorded_at, "
                "record) VALUES (1, ?, ?, ?, ?, ?)",
                (
                    record["model"]["model_id"],
                    record["protocol_sha256"],
                    record["record_id"],
                    now,
                    canonical(record),
                ),
            )
        return record
