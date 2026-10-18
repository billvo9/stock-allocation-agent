"""
The inference store's write-once writer: the only module that writes into it
(its contract is in `schema`). Only the owner-run inference fetch script uses
it, and only with `--authorized --store`.

What a batch must be before anything is written (each refusal names its
reason, and nothing is created by a refused write):
    - a real ticker in the provider's form, or a demo identifier (with "_")
      for synthetic bars, and never the other way round;
    - a request inside the protected period (from 2025-01-01) that starts and
      ends on sessions of the calendar passed in, with exactly one row per
      calendar session of the request (a session the provider did not serve
      is a row without prices), valid bars (`prices.validate_bars`);
    - retrieved no earlier than its last session's close was available
      (20:00 America/New_York), no later than the store's clock, and no more
      than RECORD_WITHIN before it: a batch is recorded when it is retrieved,
      so its retrieval time can be trusted as the time it became known.

A retrieval identical to the ticker's newest batch (same sessions, content,
calendar and data class) is not stored again: that batch is returned.

The file is created only when a batch is written: its directory (and any
missing parent) with mode 0o700 (a directory open to other users is
refused), the file with 0o600,
then, in one transaction, the header id, the tables, the write-once triggers
and the meta rows. An existing file is written only when it is this store.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.inference.schema import (
    APPLICATION_ID,
    BATCH_FIELDS,
    PROTECTED_START,
    RECORD_WITHIN,
    SCHEMA_VERSION,
    STORE_KIND,
    InferenceBatch,
    InferenceRefused,
    Refusal,
    Sessions,
    bars_sha256,
    batches,
    calendar,
    check_identity,
    check_location,
    close_available_at,
    default_path,
    read_only,
    session_day,
    stamp,
    utc_time,
)
from stock_agent.stock_lab.prices import BarsRefused, validate_bars
from stock_agent.stock_lab.store import utc_now

_TICKER = re.compile(r"[A-Z]{1,5}(-[A-Z]{1,2})?")  # a US ticker in the provider's form
_DEMO = re.compile(r"[A-Z0-9]+(_[A-Z0-9]+)+")  # demo identifiers contain "_"
_PROTECTED = PROTECTED_START.isoformat()

STATEMENTS = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    f"""CREATE TABLE batches (
        batch_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        source TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        request_start TEXT NOT NULL,
        request_end TEXT NOT NULL,
        calendar_id TEXT NOT NULL,
        row_count INTEGER NOT NULL CHECK (row_count > 0),
        content_sha256 TEXT NOT NULL,
        data_class TEXT NOT NULL CHECK (data_class IN ('real', 'synthetic')),
        vintage_note TEXT NOT NULL,
        CHECK (request_start >= '{_PROTECTED}' AND request_start <= request_end),
        CHECK (retrieved_at <= recorded_at)
    )""",
    "CREATE INDEX batches_ticker ON batches (ticker, retrieved_at)",
    f"""CREATE TABLE bars (
        batch_id TEXT NOT NULL REFERENCES batches (batch_id),
        session TEXT NOT NULL CHECK (session >= '{_PROTECTED}'),
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        div_cash REAL NOT NULL,
        split_factor REAL NOT NULL,
        PRIMARY KEY (batch_id, session)
    )""",
    *(
        f"CREATE TRIGGER {table}_no_{action} BEFORE {action.upper()} ON {table} "
        "BEGIN SELECT RAISE(ABORT, 'write-once table'); END"
        for table in ("meta", "batches", "bars")
        for action in ("update", "delete")
    ),
    *(
        f"CREATE TRIGGER {table}_no_replace BEFORE INSERT ON {table} "
        f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {key}) "
        "BEGIN SELECT RAISE(ABORT, 'write-once table'); END"
        for table, key in (
            ("meta", "key = NEW.key"),
            ("batches", "batch_id = NEW.batch_id"),
            ("bars", "batch_id = NEW.batch_id AND session = NEW.session"),
        )
    ),
    (
        "CREATE TRIGGER bars_complete BEFORE INSERT ON bars "
        "WHEN (SELECT COUNT(*) FROM bars WHERE batch_id = NEW.batch_id) "
        ">= (SELECT row_count FROM batches WHERE batch_id = NEW.batch_id) "
        "BEGIN SELECT RAISE(ABORT, 'write-once table: the batch is complete'); END"
    ),
)


def _kind(data_class: object) -> DataClass:
    try:
        return DataClass(data_class)
    except ValueError:
        raise InferenceRefused(Refusal.DATA_CLASS, f"unknown data class {data_class!r}") from None


def _check_ticker(ticker: object, kind: DataClass) -> str:
    if not isinstance(ticker, str) or not (_TICKER.fullmatch(ticker) or _DEMO.fullmatch(ticker)):
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"not a ticker: {ticker!r}")
    demo = "_" in ticker
    if kind is DataClass.SYNTHETIC and not demo:
        raise InferenceRefused(Refusal.DATA_CLASS, "synthetic bars only under a demo identifier")
    if kind is DataClass.REAL and demo:
        raise InferenceRefused(Refusal.DATA_CLASS, "real bars never under a demo identifier")
    return ticker


def _text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceRefused(Refusal.INVALID_REQUEST, f"{label} must be non-empty text")
    return value


class InferenceStore:
    """The writer for the store at `path` (default `schema.default_path()`)."""

    def __init__(
        self, path: Path | None = None, *, clock: Callable[[], pd.Timestamp] = utc_now
    ) -> None:
        self.path = check_location(path or default_path())
        self._clock = clock

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        check_location(self.path)
        directory = self.path.parent
        for folder in reversed((directory, *directory.parents)):
            if not folder.exists():  # every missing folder, parents too, private
                folder.mkdir(mode=0o700)
        if directory.stat().st_mode & 0o077:
            raise InferenceRefused(
                Refusal.LOCATION, f"{directory} is open to other users; restrict it (chmod 700)"
            )
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            pass  # created earlier, or by another writer at the same moment
        if self.path.stat().st_mode & 0o077:
            raise InferenceRefused(
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
                raise InferenceRefused(Refusal.DATABASE_ERROR, type(error).__name__) from None
            raise
        else:
            connection.execute("COMMIT")
        finally:
            connection.close()

    def batches(self, ticker: str) -> tuple[InferenceBatch, ...]:
        """The ticker's stored batches (lineage only, no bars), oldest retrieval first."""

        with read_only(self.path) as connection:
            return () if connection is None else batches(connection, ticker)

    def record_batch(
        self,
        ticker: str,
        bars: pd.DataFrame,
        *,
        source: str,
        retrieved_at: pd.Timestamp,
        request_start: object,
        request_end: object,
        sessions: Sessions,
        calendar_id: str,
        data_class: DataClass | str,
        vintage_note: str,
    ) -> InferenceBatch:
        """Store one retrieval of `ticker`'s bars (write-once); see the module docstring."""

        kind = _kind(data_class)
        _check_ticker(ticker, kind)
        texts = {
            label: _text(label, value)
            for label, value in (
                ("source", source),
                ("calendar_id", calendar_id),
                ("vintage_note", vintage_note),
            )
        }
        start = session_day(request_start, "request_start")
        end = session_day(request_end, "request_end")
        if start > end:
            raise InferenceRefused(Refusal.INVALID_REQUEST, "request_start is after request_end")
        if start.date() < PROTECTED_START:
            raise InferenceRefused(
                Refusal.BEFORE_PROTECTED_PERIOD,
                f"the request starts {start.date()}: sessions through {PROTECTED_START} minus "
                "one day are development data and belong to the research store",
            )
        expected = calendar(sessions, start, end)
        if len(expected) == 0 or expected[0] != start or expected[-1] != end:
            raise InferenceRefused(
                Refusal.INVALID_REQUEST, "a request must start and end on calendar sessions"
            )
        try:
            checked = validate_bars(bars)
        except BarsRefused as error:
            raise InferenceRefused(Refusal.INVALID_BARS, str(error)) from None
        if not pd.DatetimeIndex(checked["session"]).as_unit("us").equals(expected):
            raise InferenceRefused(
                Refusal.NOT_ON_CALENDAR,
                f"{len(checked)} rows for {len(expected)} calendar sessions: a batch holds exactly "
                "one row per session of its request (a gap is a row without prices)",
            )
        when = utc_time(retrieved_at, "retrieved_at")
        now = utc_time(self._clock(), "the store's clock")
        if when > now:
            raise InferenceRefused(
                Refusal.RETRIEVAL_TIME, "retrieved_at is after the store's clock"
            )
        if when < now - RECORD_WITHIN:
            raise InferenceRefused(
                Refusal.RETRIEVAL_TIME,
                "a batch is recorded when it is retrieved: retrieved_at is more than "
                f"{RECORD_WITHIN} before the store's clock",
            )
        if when < close_available_at(end):
            raise InferenceRefused(
                Refusal.RETRIEVAL_TIME,
                f"retrieved before the {end.date()} close was available (20:00 America/New_York)",
            )
        digest = bars_sha256(checked)
        retrieved = stamp(when)
        compact = retrieved.replace("-", "").replace(":", "").replace(".", "")
        batch = InferenceBatch(
            batch_id=f"{ticker}-{compact}-{digest[:12]}",
            ticker=ticker,
            source=texts["source"],
            retrieved_at=retrieved,
            recorded_at=stamp(now),
            request_start=start.strftime("%Y-%m-%d"),
            request_end=end.strftime("%Y-%m-%d"),
            calendar_id=texts["calendar_id"],
            row_count=len(checked),
            content_sha256=digest,
            data_class=kind.value,
            vintage_note=texts["vintage_note"],
        )
        with self._write() as connection:
            newest = connection.execute(
                f"SELECT {', '.join(BATCH_FIELDS)} FROM batches WHERE ticker = ? "
                "ORDER BY retrieved_at DESC, batch_id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if newest is not None:
                previous = InferenceBatch(*newest)
                same = ("request_start", "request_end", "content_sha256", "calendar_id")
                if all(getattr(previous, name) == getattr(batch, name) for name in same):
                    return previous
                if previous.data_class != batch.data_class:
                    raise InferenceRefused(Refusal.DATA_CLASS, f"{ticker} has other batches")
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
