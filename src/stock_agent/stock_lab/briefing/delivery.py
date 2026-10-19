"""
Delivering a rendered briefing (standard library only): the adapter
interface, the local preview adapter and the duplicate-send log.

Nothing here sends anything over a network. The only adapter,
`PreviewAdapter`, writes the briefing's HTML, text and JSON into a private
local directory. An email adapter is a later step that needs the owner's
separate approval; this module reads, asks for and stores no credential.

Where previews go
    STOCK_AGENT_STOCK_LAB_BRIEFING_HOME, or ~/stock-agent-private/
    stock_lab_briefings/. The directory is refused inside any Git working
    tree (this repository is public) and when it holds the research
    store's database; it is created with mode 0o700, and each file with
    0o600, never overwritten.

Duplicate-send protection
    `DeliveryLog` is an append-only SQLite file in that directory (its own
    file, not the Stock Lab's store or ledger). One delivery per market
    date, channel and data class:

        claimed   an attempt has started (written before the adapter runs)
        sent      the adapter finished; no further attempt is allowed
        failed    the adapter raised; a new attempt may be claimed

    The claim is taken inside an immediate transaction, so two runs at the
    same moment cannot both send. An attempt left "claimed" (a run that died
    mid-delivery) blocks further attempts until the owner looks: a briefing
    is delivered at most once, never twice. A failure records only the
    error's type, never its message, which could echo an address or a
    secret.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import default_path as research_store_path
from stock_agent.stock_lab.store import stamp, utc_now

HOME_ENV = "STOCK_AGENT_STOCK_LAB_BRIEFING_HOME"
DIRECTORY = "stock_lab_briefings"
LOG_NAME = "briefing_deliveries.sqlite3"
CLAIMED = "claimed"
SENT = "sent"
FAILED = "failed"
DUPLICATE = "duplicate"
IN_PROGRESS = "in_progress"

SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    market_date TEXT NOT NULL,
    channel TEXT NOT NULL,
    data_class TEXT NOT NULL CHECK (data_class IN ('real', 'synthetic')),
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('claimed', 'sent', 'failed')),
    at TEXT NOT NULL,
    detail TEXT
);
CREATE TRIGGER IF NOT EXISTS deliveries_no_update BEFORE UPDATE ON deliveries
BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER IF NOT EXISTS deliveries_no_delete BEFORE DELETE ON deliveries
BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
"""


class DeliveryRefused(ValueError):
    """A location or delivery the rules refuse."""


def default_directory() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "stock-agent-private" / DIRECTORY


def research_home() -> Path:
    """The research store's directory (from its own environment variable or default)."""

    return research_store_path().parent


def _working_tree(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def private_directory(path: Path | None = None) -> Path:
    """
    The resolved preview directory, created with mode 0o700 if missing.
    Refuses one inside any Git working tree, the research store's directory
    (or any holding its database), one owned by another user and one other
    users can read.
    """

    resolved = (path or default_directory()).expanduser().resolve()
    if _working_tree(resolved) is not None:
        raise DeliveryRefused(
            "the briefing directory must be outside any Git working tree (the repository is public)"
        )
    if (resolved / RESEARCH_DB_NAME).exists() or resolved == research_home().resolve():
        raise DeliveryRefused("the briefing directory must not hold the research store")
    if not resolved.exists():
        resolved.mkdir(mode=0o700, parents=True)
    if resolved.stat().st_uid != os.getuid():
        raise DeliveryRefused(f"{resolved} belongs to another user")
    if resolved.stat().st_mode & 0o077:
        raise DeliveryRefused(
            f"{resolved} is readable by other users; restrict it (chmod 700) first"
        )
    return resolved


@dataclass(frozen=True)
class RenderedBriefing:
    market_date: str
    data_class: str
    subject: str
    text: str
    html: str
    document: str  # the briefing as strict JSON
    label: str = "premarket"  # the file names' kind: "premarket", or a preview's state and time


class DeliveryAdapter(Protocol):
    """Delivers one rendered briefing; returns a short receipt that holds no secret."""

    channel: str

    def deliver(self, briefing: RenderedBriefing) -> str: ...


class PreviewAdapter:
    """Writes the briefing's HTML, text and JSON into the private directory (never overwrites)."""

    channel = "preview"

    def __init__(self, directory: Path) -> None:
        self.directory = private_directory(directory)

    def paths(self, briefing: RenderedBriefing) -> dict[str, Path]:
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", briefing.label):
            raise DeliveryRefused("a briefing label is lower-case letters, digits, - and _")
        stem = f"{briefing.market_date}_{briefing.data_class}_{briefing.label}_briefing"
        return {kind: self.directory / f"{stem}.{kind}" for kind in ("html", "txt", "json")}

    def deliver(self, briefing: RenderedBriefing) -> str:
        """
        Writes all three files or none: each is written under a temporary name,
        then linked to its final name, which never replaces an existing file.
        """

        targets = self.paths(briefing)
        contents = {"html": briefing.html, "txt": briefing.text, "json": briefing.document}
        partial = {kind: path.with_name(f".{path.name}.partial") for kind, path in targets.items()}
        linked: list[Path] = []
        try:
            for kind, path in partial.items():
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(contents[kind])
            for kind, path in targets.items():
                os.link(partial[kind], path)
                linked.append(path)
        except BaseException:
            for path in linked:  # only the files this attempt created
                path.unlink(missing_ok=True)
            raise
        finally:
            for path in partial.values():
                path.unlink(missing_ok=True)
        return ", ".join(path.name for path in targets.values())


_RECEIPT = re.compile(r"[A-Za-z0-9._, -]{1,400}")


def safe_receipt(receipt: object) -> str:
    """The adapter's receipt when it is short plain text (file names); otherwise "sent"."""

    text = receipt if isinstance(receipt, str) else ""
    return text if _RECEIPT.fullmatch(text) else "sent"


@dataclass(frozen=True)
class DeliveryResult:
    status: str  # sent, failed, duplicate or in_progress
    channel: str
    market_date: str
    attempt: int | None
    detail: str


class DeliveryLog:
    """The append-only delivery log in `directory` (module docstring)."""

    def __init__(self, directory: Path, *, clock: Callable[[], pd.Timestamp] = utc_now) -> None:
        self.directory = private_directory(directory)
        self.path = self.directory / LOG_NAME
        self._clock = clock

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            pass  # created earlier, or by another run at the same moment
        info = os.lstat(self.path)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise DeliveryRefused("the delivery log must be a regular file owned by this user")
        if info.st_mode & 0o077:
            raise DeliveryRefused(
                f"{self.path} is readable by other users; restrict it (chmod 600)"
            )
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            connection.executescript(SCHEMA)
            yield connection
        finally:
            connection.close()

    def claim(self, market_date: str, channel: str, data_class: str) -> tuple[str, int | None]:
        """(CLAIMED, attempt), or (DUPLICATE | IN_PROGRESS, None) when no attempt may start."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT attempt, status FROM deliveries WHERE market_date = ? AND channel = ? "
                    "AND data_class = ? ORDER BY seq",
                    (market_date, channel, data_class),
                ).fetchall()
                latest: dict[int, str] = {}
                for attempt, status in rows:
                    latest[attempt] = status
                if SENT in latest.values():
                    connection.execute("ROLLBACK")
                    return DUPLICATE, None
                if CLAIMED in latest.values():
                    connection.execute("ROLLBACK")
                    return IN_PROGRESS, None
                attempt = max(latest, default=0) + 1
                connection.execute(
                    "INSERT INTO deliveries (market_date, channel, data_class, attempt, status, "
                    "at, detail) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (market_date, channel, data_class, attempt, CLAIMED, stamp(self._clock())),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return CLAIMED, attempt

    def finish(
        self,
        market_date: str,
        channel: str,
        data_class: str,
        attempt: int,
        status: str,
        detail: str,
    ) -> None:
        if status not in (SENT, FAILED):
            raise DeliveryRefused("an attempt finishes as sent or failed")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO deliveries (market_date, channel, data_class, attempt, status, at, "
                "detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (market_date, channel, data_class, attempt, status, stamp(self._clock()), detail),
            )

    def history(self) -> tuple[dict, ...]:
        if not self.path.exists():
            return ()
        columns = (
            "seq",
            "market_date",
            "channel",
            "data_class",
            "attempt",
            "status",
            "at",
            "detail",
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(columns)} FROM deliveries ORDER BY seq"
            ).fetchall()
        return tuple(dict(zip(columns, row, strict=True)) for row in rows)


def deliver_once(
    briefing: RenderedBriefing, adapter: DeliveryAdapter, log: DeliveryLog
) -> DeliveryResult:
    """Deliver `briefing` through `adapter` at most once per market date (module docstring)."""

    key = (briefing.market_date, adapter.channel, briefing.data_class)
    state, attempt = log.claim(*key)
    if state == DUPLICATE:
        return DeliveryResult(
            DUPLICATE, adapter.channel, briefing.market_date, None,
            "already delivered for this market date; nothing was sent again",
        )  # fmt: skip
    if state == IN_PROGRESS:
        return DeliveryResult(
            IN_PROGRESS, adapter.channel, briefing.market_date, None,
            "an earlier attempt never finished; check the log before delivering again",
        )  # fmt: skip
    try:
        receipt = adapter.deliver(briefing)
    except BaseException as error:
        # Any failure is recorded, so the attempt never stays "claimed" by mistake. Only the
        # error's type is kept: its message could echo an address or a secret, so it is never
        # stored, printed or chained into another traceback.
        detail = f"failed: {type(error).__name__}"
        try:
            log.finish(*key, attempt, FAILED, detail)
        except (sqlite3.Error, OSError, DeliveryRefused) as problem:
            raise DeliveryRefused(
                f"the failed attempt could not be recorded ({type(problem).__name__})"
            ) from None
        if not isinstance(error, Exception):  # an interrupt still stops the run
            raise
        return DeliveryResult(FAILED, adapter.channel, briefing.market_date, attempt, detail)
    kept = safe_receipt(receipt)
    log.finish(*key, attempt, SENT, kept)
    return DeliveryResult(SENT, adapter.channel, briefing.market_date, attempt, kept)
