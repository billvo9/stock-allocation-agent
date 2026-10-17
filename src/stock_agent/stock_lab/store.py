"""
The Stock Lab's private local store: one SQLite file (standard library).

Why SQLite: the watchlist must survive app restarts (Streamlit's Session
State does not), the page and the daily job may both write, and the vendor
data must be write-once. SQLite gives atomic transactions, concurrent
readers with one writer (WAL), and triggers that make tables write-once,
with no server and no new dependency.

Location (private, never in Git): the file `stock_lab.sqlite3` in the
directory named by STOCK_AGENT_STOCK_LAB_HOME, else
~/stock-agent-private/stock_lab/. A directory inside this repository is
refused unless it is under the ignored data/raw/. The directory is created
with mode 0o700 and the file with 0o600, and only when something is first
written: opening a store that does not exist reads as empty.

Tables
    watchlist_events   append-only: every add and remove, with the text as
                       entered and the time. The watchlist is derived: a
                       ticker is watched when its latest event is an add.
                       Removing a ticker keeps its history.
    listing_snapshots  one provider listing per retrieval (write-once),
    listing_rows       with its rows.
    symbol_checks      append-only: each classification of a ticker against a
                       listing snapshot (symbols.classify), made by the
                       owner-run update script, which has the exchange
                       calendar; the page shows the latest check per ticker.
    price_batches      one retrieval of one ticker's bars (write-once), with
    price_bars         its rows: a vintage. Later retrievals never overwrite
                       earlier ones; reads choose, per session, the latest
                       batch retrieved at or before a given time, and report
                       sessions whose values changed between vintages.

UPDATE and DELETE on every table except `meta` abort (triggers), as does an
insert onto an existing key (so INSERT OR REPLACE cannot overwrite either,
whatever the connection's settings), and a recorded
batch or listing cannot gain rows after its count is reached, so a recorded
event, listing or price can never be rewritten in place. Reads also check
each batch's row count and content hash.
"""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.prices import BAR_COLUMNS, validate_bars
from stock_agent.stock_lab.symbols import Classification, ListingRow, ParsedTicker

SCHEMA_VERSION = "1"
HOME_ENV = "STOCK_AGENT_STOCK_LAB_HOME"
DB_NAME = "stock_lab.sqlite3"
WRITE_ONCE_TABLES = (
    "watchlist_events",
    "symbol_checks",
    "listing_snapshots",
    "listing_rows",
    "price_batches",
    "price_bars",
)

SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('add', 'remove')),
    entered_as TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS watchlist_events_ticker ON watchlist_events (ticker, seq);
CREATE TABLE IF NOT EXISTS symbol_checks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES listing_snapshots (snapshot_id)
);
CREATE TABLE IF NOT EXISTS listing_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    data_class TEXT NOT NULL CHECK (data_class IN ('real', 'synthetic'))
);
CREATE TABLE IF NOT EXISTS listing_rows (
    snapshot_id TEXT NOT NULL REFERENCES listing_snapshots (snapshot_id),
    ticker TEXT NOT NULL,
    exchange TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    price_currency TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT
);
CREATE INDEX IF NOT EXISTS listing_rows_ticker ON listing_rows (snapshot_id, ticker);
CREATE TABLE IF NOT EXISTS price_batches (
    batch_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    first_session TEXT NOT NULL,
    last_session TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    data_class TEXT NOT NULL CHECK (data_class IN ('real', 'synthetic')),
    vintage_note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS price_batches_ticker ON price_batches (ticker, retrieved_at);
CREATE TABLE IF NOT EXISTS price_bars (
    batch_id TEXT NOT NULL REFERENCES price_batches (batch_id),
    session TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    div_cash REAL NOT NULL,
    split_factor REAL NOT NULL,
    PRIMARY KEY (batch_id, session)
);
"""
    + "".join(
        f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
"""
        for table in WRITE_ONCE_TABLES
    )
    + """
CREATE TRIGGER IF NOT EXISTS watchlist_events_no_replace BEFORE INSERT ON watchlist_events
WHEN EXISTS (SELECT 1 FROM watchlist_events WHERE seq = NEW.seq)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS symbol_checks_no_replace BEFORE INSERT ON symbol_checks
WHEN EXISTS (SELECT 1 FROM symbol_checks WHERE seq = NEW.seq)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS listing_snapshots_no_replace BEFORE INSERT ON listing_snapshots
WHEN EXISTS (SELECT 1 FROM listing_snapshots WHERE snapshot_id = NEW.snapshot_id)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS listing_rows_no_replace BEFORE INSERT ON listing_rows
WHEN EXISTS (SELECT 1 FROM listing_rows WHERE rowid = NEW.rowid)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS price_batches_no_replace BEFORE INSERT ON price_batches
WHEN EXISTS (SELECT 1 FROM price_batches WHERE batch_id = NEW.batch_id)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS price_bars_no_replace BEFORE INSERT ON price_bars
WHEN EXISTS (
    SELECT 1 FROM price_bars WHERE batch_id = NEW.batch_id AND session = NEW.session
)
BEGIN SELECT RAISE(ABORT, 'write-once table'); END;
CREATE TRIGGER IF NOT EXISTS price_bars_complete BEFORE INSERT ON price_bars
WHEN (SELECT COUNT(*) FROM price_bars WHERE batch_id = NEW.batch_id)
     >= (SELECT row_count FROM price_batches WHERE batch_id = NEW.batch_id)
BEGIN SELECT RAISE(ABORT, 'write-once table: the batch is complete'); END;
CREATE TRIGGER IF NOT EXISTS listing_rows_complete BEFORE INSERT ON listing_rows
WHEN (SELECT COUNT(*) FROM listing_rows WHERE snapshot_id = NEW.snapshot_id)
     >= (SELECT row_count FROM listing_snapshots WHERE snapshot_id = NEW.snapshot_id)
BEGIN SELECT RAISE(ABORT, 'write-once table: the snapshot is complete'); END;
"""
)


class StoreRefused(ValueError):
    """A store location or a write that the rules refuse."""


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("us")


def stamp(value: pd.Timestamp) -> str:
    """UTC with microseconds and a Z: 2026-10-16T21:30:00.000000Z."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise StoreRefused("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def default_path() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    home = Path(configured).expanduser() if configured else Path.home() / "stock-agent-private"
    if not configured:
        home = home / "stock_lab"
    return home / DB_NAME


def _working_tree(path: Path) -> Path | None:
    """The nearest enclosing directory with a .git entry (a Git working tree), if any."""

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def check_private(path: Path) -> Path:
    """
    The resolved path; refuses one inside any Git working tree (this
    repository included, however the package was imported) unless it is
    under that tree's ignored data/raw/.
    """

    resolved = path.expanduser().resolve()
    root = _working_tree(resolved.parent)
    if root is None:
        return resolved
    if resolved.relative_to(root).parts[:2] != ("data", "raw"):
        raise StoreRefused(
            "the Stock Lab store must be outside any Git working tree (or under its ignored "
            "data/raw/)"
        )
    return resolved


@dataclass(frozen=True)
class WatchItem:
    ticker: str
    entered_as: str
    added_at: str


@dataclass(frozen=True)
class AddResult:
    ticker: str
    added: bool  # False when it was already watched


@dataclass(frozen=True)
class Batch:
    batch_id: str
    ticker: str
    source: str
    retrieved_at: str
    first_session: str
    last_session: str
    row_count: int
    content_sha256: str
    data_class: str
    vintage_note: str


@dataclass(frozen=True)
class StoredBars:
    """A ticker's bars as of a retrieval time, with their lineage."""

    ticker: str
    bars: pd.DataFrame  # prices.BAR_COLUMNS plus batch_id
    batches: tuple[Batch, ...]  # the batches that supplied at least one session
    revised_sessions: tuple[str, ...]  # sessions whose values differ between vintages
    data_class: str


def _bars_sha256(bars: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame = bars.loc[:, list(BAR_COLUMNS)].copy()
    frame["session"] = frame["session"].dt.strftime("%Y-%m-%d")
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[ListingRow]) -> str:
    text = "\n".join(
        "|".join(
            (
                row.ticker,
                row.exchange,
                row.asset_type,
                row.price_currency,
                row.start_date.isoformat() if row.start_date else "",
                row.end_date.isoformat() if row.end_date else "",
            )
        )
        for row in rows
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _date(text: str | None) -> date | None:
    return None if text in (None, "") else date.fromisoformat(text)


class StockLabStore:
    """The store at `path` (default `default_path()`); see the module docstring."""

    def __init__(
        self, path: Path | None = None, *, clock: Callable[[], pd.Timestamp] = utc_now
    ) -> None:
        self.path = check_private(path or default_path())
        self._clock = clock

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    # --- connections ---

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        directory = self.path.parent
        if not directory.exists():
            directory.mkdir(mode=0o700, parents=True)
        if directory.stat().st_mode & 0o077:
            raise StoreRefused(
                f"{directory} is readable by other users; restrict it (chmod 700) first"
            )
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            pass  # created earlier, or by another writer at the same moment
        if self.path.stat().st_mode & 0o077:
            raise StoreRefused(f"{self.path} is readable by other users; restrict it (chmod 600)")
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA recursive_triggers = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self._check_version(connection)
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection | None]:
        if not self.exists:
            yield None
            return
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=10.0)
        try:
            self._check_version(connection)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _check_version(connection: sqlite3.Connection) -> None:
        row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or row[0] != SCHEMA_VERSION:
            raise StoreRefused(f"unsupported store schema version {row[0] if row else None}")

    # --- the watchlist ---

    def watchlist(self) -> tuple[WatchItem, ...]:
        """Watched tickers in alphabetical order (a fixed order; nothing is ranked)."""

        with self._read() as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                """
                SELECT e.ticker, e.entered_as, e.recorded_at FROM watchlist_events e
                JOIN (SELECT ticker, MAX(seq) AS seq FROM watchlist_events GROUP BY ticker) last
                  ON e.ticker = last.ticker AND e.seq = last.seq
                WHERE e.action = 'add' ORDER BY e.ticker
                """
            ).fetchall()
        return tuple(WatchItem(*row) for row in rows)

    def watched(self) -> tuple[str, ...]:
        return tuple(item.ticker for item in self.watchlist())

    def add(self, parsed: Iterable[ParsedTicker]) -> tuple[AddResult, ...]:
        """Record an add for each valid entry not already watched (one transaction)."""

        entries = [entry for entry in parsed if entry.ok]
        if not entries:
            return ()
        now = stamp(self._clock())
        results = []
        with self._write() as connection:
            watched = set(self._watched(connection))
            for entry in entries:
                if entry.ticker in watched:
                    results.append(AddResult(entry.ticker, added=False))
                    continue
                connection.execute(
                    "INSERT INTO watchlist_events (ticker, action, entered_as, recorded_at) "
                    "VALUES (?, 'add', ?, ?)",
                    (entry.ticker, entry.entered.strip(), now),
                )
                watched.add(entry.ticker)
                results.append(AddResult(entry.ticker, added=True))
        return tuple(results)

    def remove(self, tickers: Iterable[str]) -> tuple[str, ...]:
        """Record a remove for each watched ticker; returns those removed."""

        wanted = list(dict.fromkeys(tickers))
        if not wanted or not self.exists:
            return ()
        now = stamp(self._clock())
        removed = []
        with self._write() as connection:
            watched = set(self._watched(connection))
            for ticker in wanted:
                if ticker in watched:
                    connection.execute(
                        "INSERT INTO watchlist_events (ticker, action, entered_as, recorded_at) "
                        "VALUES (?, 'remove', ?, ?)",
                        (ticker, ticker, now),
                    )
                    removed.append(ticker)
        return tuple(removed)

    @staticmethod
    def _watched(connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            """
            SELECT e.ticker FROM watchlist_events e
            JOIN (SELECT ticker, MAX(seq) AS seq FROM watchlist_events GROUP BY ticker) last
              ON e.ticker = last.ticker AND e.seq = last.seq
            WHERE e.action = 'add'
            """
        ).fetchall()
        return [row[0] for row in rows]

    def events(self) -> pd.DataFrame:
        """Every watchlist event, oldest first."""

        columns = ["seq", "ticker", "action", "entered_as", "recorded_at"]
        with self._read() as connection:
            if connection is None:
                return pd.DataFrame(columns=columns)
            rows = connection.execute(
                f"SELECT {', '.join(columns)} FROM watchlist_events ORDER BY seq"
            ).fetchall()
        return pd.DataFrame(rows, columns=columns)

    # --- listings ---

    def record_listing(
        self,
        rows: Sequence[ListingRow],
        *,
        source: str,
        retrieved_at: pd.Timestamp,
        data_class: DataClass,
    ) -> str:
        """Store one listing snapshot (write-once); returns its id."""

        if not rows:
            raise StoreRefused("a listing snapshot needs at least one row")
        digest = _rows_sha256(rows)
        when = stamp(retrieved_at)
        snapshot_id = f"L{when.replace('-', '').replace(':', '').replace('.', '')}-{digest[:12]}"
        with self._write() as connection:
            connection.execute(
                "INSERT INTO listing_snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (snapshot_id, source, when, digest, len(rows), DataClass(data_class).value),
            )
            connection.executemany(
                "INSERT INTO listing_rows VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        snapshot_id,
                        row.ticker,
                        row.exchange,
                        row.asset_type,
                        row.price_currency,
                        row.start_date.isoformat() if row.start_date else None,
                        row.end_date.isoformat() if row.end_date else None,
                    )
                    for row in rows
                ],
            )
        return snapshot_id

    def latest_listing(self) -> tuple[dict, tuple[ListingRow, ...]] | None:
        """The newest listing snapshot's metadata and rows, or None."""

        with self._read() as connection:
            if connection is None:
                return None
            snapshot = connection.execute(
                "SELECT snapshot_id, source, retrieved_at, content_sha256, row_count, data_class "
                "FROM listing_snapshots ORDER BY retrieved_at DESC, snapshot_id DESC LIMIT 1"
            ).fetchone()
            if snapshot is None:
                return None
            rows = connection.execute(
                "SELECT ticker, exchange, asset_type, price_currency, start_date, end_date "
                "FROM listing_rows WHERE snapshot_id = ?",
                (snapshot[0],),
            ).fetchall()
        keys = ("snapshot_id", "source", "retrieved_at", "content_sha256", "row_count")
        meta = dict(zip((*keys, "data_class"), snapshot, strict=True))
        listing = tuple(
            ListingRow(ticker, exchange, asset, currency, _date(start), _date(end))
            for ticker, exchange, asset, currency, start, end in rows
        )
        return meta, listing

    def record_checks(self, classifications: Iterable[Classification], *, snapshot_id: str) -> int:
        """Append one check per classification against `snapshot_id`; returns the count."""

        rows = [
            (c.ticker, c.status.value, c.reason.value, c.detail, stamp(self._clock()), snapshot_id)
            for c in classifications
        ]
        if not rows:
            return 0
        with self._write() as connection:
            connection.executemany(
                "INSERT INTO symbol_checks "
                "(ticker, status, reason, detail, checked_at, snapshot_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def latest_checks(self) -> dict[str, Classification]:
        """The latest check of every checked ticker."""

        with self._read() as connection:
            if connection is None:
                return {}
            rows = connection.execute(
                """
                SELECT c.ticker, c.status, c.reason, c.detail FROM symbol_checks c
                JOIN (SELECT ticker, MAX(seq) AS seq FROM symbol_checks GROUP BY ticker) last
                  ON c.ticker = last.ticker AND c.seq = last.seq
                """
            ).fetchall()
        return {
            ticker: Classification(ticker, SymbolStatus(status), SymbolReason(reason), detail)
            for ticker, status, reason, detail in rows
        }

    def listing_rows_for(self, ticker: str) -> tuple[ListingRow, ...] | None:
        """The newest snapshot's rows for one ticker; None without any snapshot."""

        latest = self.latest_listing()
        if latest is None:
            return None
        return tuple(row for row in latest[1] if row.ticker.strip().upper() == ticker)

    # --- prices ---

    def record_prices(
        self,
        ticker: str,
        bars: pd.DataFrame,
        *,
        source: str,
        retrieved_at: pd.Timestamp,
        data_class: DataClass,
        vintage_note: str,
    ) -> Batch:
        """
        Store one retrieval of `ticker`'s bars as a new batch (write-once). An
        identical retrieval of the same sessions as the ticker's newest batch is
        not stored again: that batch is returned.
        """

        kind = DataClass(data_class)
        demo = "_" in ticker  # demo identifiers contain "_", which no US ticker can
        if kind is DataClass.SYNTHETIC and not demo:
            raise StoreRefused("synthetic bars may be stored only under a demo identifier")
        if kind is DataClass.REAL and demo:
            raise StoreRefused("real bars may not be stored under a demo identifier")
        checked = validate_bars(bars)
        if checked.empty:
            raise StoreRefused("a price batch needs at least one bar")
        digest = _bars_sha256(checked)
        when = stamp(retrieved_at)
        first = checked["session"].iloc[0].strftime("%Y-%m-%d")
        last = checked["session"].iloc[-1].strftime("%Y-%m-%d")
        newest = self._newest_batch(ticker)
        if newest and (newest.content_sha256, newest.first_session, newest.last_session) == (
            digest,
            first,
            last,
        ):
            return newest
        compact = when.replace("-", "").replace(":", "").replace(".", "")
        batch = Batch(
            batch_id=f"{ticker}-{compact}-{digest[:12]}",
            ticker=ticker,
            source=source,
            retrieved_at=when,
            first_session=first,
            last_session=last,
            row_count=len(checked),
            content_sha256=digest,
            data_class=kind.value,
            vintage_note=vintage_note,
        )
        with self._write() as connection:
            connection.execute(
                "INSERT INTO price_batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch.batch_id,
                    batch.ticker,
                    batch.source,
                    batch.retrieved_at,
                    batch.first_session,
                    batch.last_session,
                    batch.row_count,
                    batch.content_sha256,
                    batch.data_class,
                    batch.vintage_note,
                ),
            )
            connection.executemany(
                "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def batches(self, ticker: str) -> tuple[Batch, ...]:
        with self._read() as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                "SELECT * FROM price_batches WHERE ticker = ? ORDER BY retrieved_at, batch_id",
                (ticker,),
            ).fetchall()
        return tuple(Batch(*row) for row in rows)

    @staticmethod
    def _check_batches(frame: pd.DataFrame, batches: Sequence[Batch]) -> None:
        """Each batch's stored rows must match its recorded count and content hash."""

        for batch in batches:
            rows = frame[frame["batch_id"] == batch.batch_id].sort_values("session")
            if len(rows) != batch.row_count:
                raise StoreRefused(
                    f"batch {batch.batch_id} has {len(rows)} rows, not {batch.row_count}"
                )
            stored = rows.drop(columns=["batch_id"]).reset_index(drop=True)
            stored["session"] = pd.to_datetime(stored["session"], utc=True)
            if _bars_sha256(stored) != batch.content_sha256:
                raise StoreRefused(f"batch {batch.batch_id} does not match its content hash")

    def _newest_batch(self, ticker: str) -> Batch | None:
        found = self.batches(ticker)
        return found[-1] if found else None

    def bars(self, ticker: str, *, as_of: pd.Timestamp | None = None) -> StoredBars | None:
        """
        The ticker's bars from the batches retrieved at or before `as_of` (all
        when None): per session, the latest such batch. None without a batch.
        """

        usable = [
            batch
            for batch in self.batches(ticker)
            if as_of is None or batch.retrieved_at <= stamp(as_of)
        ]
        if not usable:
            return None
        classes = {batch.data_class for batch in usable}
        if len(classes) > 1:
            raise StoreRefused(f"{ticker} has both real and synthetic batches")
        order = {batch.batch_id: rank for rank, batch in enumerate(usable)}
        with self._read() as connection:
            placeholders = ", ".join("?" for _ in usable)
            rows = connection.execute(
                f"SELECT * FROM price_bars WHERE batch_id IN ({placeholders})",
                [batch.batch_id for batch in usable],
            ).fetchall()
        frame = pd.DataFrame(rows, columns=["batch_id", *BAR_COLUMNS])
        self._check_batches(frame, usable)
        frame["rank"] = frame["batch_id"].map(order)
        values = list(BAR_COLUMNS[1:])
        variants = frame.groupby("session")[values].nunique(dropna=False)
        revised = tuple(sorted(variants.index[(variants > 1).any(axis=1)]))
        latest = frame.sort_values(["session", "rank"]).groupby("session").tail(1)
        latest = latest.sort_values("session").reset_index(drop=True)
        latest["session"] = pd.to_datetime(latest["session"], utc=True).dt.as_unit("us")
        used = set(latest["batch_id"])
        return StoredBars(
            ticker=ticker,
            bars=latest[["batch_id", *BAR_COLUMNS]],
            batches=tuple(batch for batch in usable if batch.batch_id in used),
            revised_sessions=revised,
            data_class=classes.pop(),
        )
