"""
The briefings already written, read back without writing anything (standard
library and the briefing's report and renderer only).

The briefing script's previews live in one private directory (`delivery`
writes them there): STOCK_AGENT_STOCK_LAB_BRIEFING_HOME, or
~/stock-agent-private/stock_lab_briefings/. This module finds the briefings
in it, reads one back from its JSON copy, and says what the duplicate-send
log recorded for its market date. The Portfolio Lab's Premarket Briefing page
reads only through here and never loads `delivery`, the writers.

Nothing here writes
    A missing directory or log is not created. Each file is opened without
    following a link and without waiting (a named pipe is refused, never
    read), and read only when it is a regular file owned by this user that no
    other user can open, at most MAX_BYTES long, in a directory with the same
    rules, outside any Git working tree (the repository is public). The log
    is opened with SQLite `mode=ro`, and only when it uses the rollback
    journal: a read-only reader of a write-ahead-log database would create
    its -wal and -shm files, so such a log is refused.
    Another process running as this user could still swap a file between
    the checks and the read; this module guards against other users, not
    against the owner.

What a written briefing is
    The JSON copy `<market date>_<data class>_<label>_briefing.json`, read
    field by field (`report.briefing_from_document`). Its name must agree
    with its contents: the market date, the data class, and the label
    ("premarket", or a preview's state and time). It must also render (the
    plain-text rendering is prepared once and discarded), so a page can show
    it. Anything else is unreadable, with a short reason that never repeats
    the file's contents at length, and is never skipped in silence. A file
    whose name is not a briefing's (another extension, an impossible date) is
    not a briefing and is not listed.

Which briefing is the latest
    The newest market date first; among one date's briefings, the one
    prepared last. Whole market dates are read, newest first, until at least
    MAX_LISTED files are read; files of older dates are counted, never
    opened.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from pathlib import Path

from stock_agent.stock_lab.briefing.render import facts, render_text
from stock_agent.stock_lab.briefing.report import Briefing, briefing_from_document
from stock_agent.stock_lab.briefing.timing import PREMARKET

HOME_ENV = "STOCK_AGENT_STOCK_LAB_BRIEFING_HOME"
DIRECTORY = "stock_lab_briefings"
LOG_NAME = "briefing_deliveries.sqlite3"
DATA_CLASSES = ("real", "synthetic")
MAX_LISTED = 30
MAX_BYTES = 2_000_000  # a four-stock briefing's JSON copy is about 40 kB
MAX_LOG_BYTES = 50_000_000  # the log gains a few hundred bytes a day
MAX_REASON = 200
NAME = re.compile(
    r"(?P<market>\d{4}-\d{2}-\d{2})_(?P<kind>real|synthetic)_(?P<label>[a-z0-9_-]{1,40})"
    r"_briefing\.json",
    re.ASCII,
)
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", re.ASCII)  # store.stamp
DETAIL = re.compile(r"[A-Za-z0-9._, :-]{1,400}", re.ASCII)  # as delivery.safe_receipt keeps
HIDDEN_DETAIL = "a detail this page does not show"
LOG_STATUSES = ("claimed", "sent", "failed")
SQLITE_HEADER = b"SQLite format 3\x00"
OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC

NONE = "none"  # no directory, or no briefing of the data class in it
REFUSED = "refused"  # the directory breaks a rule, or cannot be read: nothing in it is read
LISTED = "listed"

SENT = "sent"
FAILED = "failed"
IN_PROGRESS = "in_progress"
NOT_RECORDED = "not_recorded"
UNREADABLE = "unreadable"


class BriefingUnreadable(ValueError):
    """A file or directory this module will not read, with the reason."""


def short(text: str, limit: int = MAX_REASON) -> str:
    """A reason cut to `limit` characters, so a file's contents are never repeated at length."""

    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def default_directory() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "stock-agent-private" / DIRECTORY


def working_tree(path: Path) -> Path | None:
    """The nearest enclosing directory with a .git entry (a Git working tree), if any."""

    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _private(info: os.stat_result, what: str, mode: str) -> None:
    if info.st_uid != os.getuid():
        raise BriefingUnreadable(f"the {what} belongs to another user")
    if info.st_mode & 0o077:
        raise BriefingUnreadable(
            f"the {what} is open to other users; restrict it (chmod {mode}) first"
        )


def readable_directory(path: Path | None = None) -> Path | None:
    """
    The resolved briefing directory, or None when it does not exist (it is
    never created). Refuses one inside a Git working tree, one that is not a
    directory, and one another user owns or can open.
    """

    try:
        resolved = (path or default_directory()).expanduser().resolve()
        if working_tree(resolved) is not None:
            raise BriefingUnreadable(
                "the briefing directory is inside a Git working tree (the repository is public)"
            )
        if not resolved.exists():
            return None
        info = resolved.stat()
    except OSError as error:
        raise BriefingUnreadable(
            f"the briefing directory could not be checked ({type(error).__name__})"
        ) from None
    if not stat.S_ISDIR(info.st_mode):
        raise BriefingUnreadable("the briefing location is not a directory")
    _private(info, "briefing directory", "700")
    return resolved


def _open_private(path: Path, what: str, limit: int | None = None) -> tuple[int, os.stat_result]:
    """A descriptor and its status for a private regular file, opened without following a link."""

    limit = MAX_BYTES if limit is None else limit

    try:
        descriptor = os.open(path, OPEN_FLAGS)
    except OSError as error:
        raise BriefingUnreadable(
            f"the {what} could not be opened ({type(error).__name__})"
        ) from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise BriefingUnreadable(f"the {what} is not a regular file")
        _private(info, what, "600")
        if info.st_size > limit:
            raise BriefingUnreadable(f"the {what} is larger than {limit:,} bytes")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, info


def _read_private(path: Path, what: str) -> bytes:
    descriptor, _ = _open_private(path, what)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read(MAX_BYTES + 1)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise BriefingUnreadable("the JSON repeats a field")
    return dict(pairs)


def _no_constant(name: str) -> object:
    raise BriefingUnreadable("the JSON holds NaN or Infinity, which strict JSON does not allow")


def _parse(raw: bytes) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_no_constant
        )
    except BriefingUnreadable:
        raise
    except (ValueError, RecursionError):  # also an over-long integer or nesting too deep
        raise BriefingUnreadable("the file is not valid JSON") from None


def _showable(briefing: Briefing) -> None:
    """Refuses a briefing whose dates, times or states the renderer cannot write."""

    try:
        date.fromisoformat(briefing.market_date)
        date.fromisoformat(briefing.prior_session)
        render_text(briefing)
        facts(briefing)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, IndexError) as error:
        raise BriefingUnreadable(f"the briefing cannot be shown ({type(error).__name__})") from None


def read_briefing(path: Path) -> Briefing:
    """
    One written briefing, read back from its JSON copy. Refused
    (BriefingUnreadable, with a short reason) unless the file is private,
    strict JSON of this briefing schema and version that renders, and its
    name agrees with its contents.
    """

    found = NAME.fullmatch(path.name)
    if found is None:
        raise BriefingUnreadable("the file name is not a written briefing's")
    document = _parse(_read_private(path, "briefing file"))
    try:
        briefing = briefing_from_document(document)
    except (TypeError, ValueError, RecursionError) as error:
        raise BriefingUnreadable(short(f"the briefing does not read back: {error}")) from None
    if briefing.market_date != found["market"] or briefing.data_class != found["kind"]:
        raise BriefingUnreadable("the file name and its contents name different briefings")
    expected = "premarket" if briefing.state == PREMARKET else briefing.state.replace("_", "-")
    if not (found["label"] == expected or found["label"].startswith(f"{expected}-preview-")):
        raise BriefingUnreadable("the file name and its contents give different states")
    _showable(briefing)
    return briefing


@dataclass(frozen=True)
class StoredBriefing:
    name: str  # the JSON copy's file name (no directory)
    market_date: str
    data_class: str
    label: str
    briefing: Briefing | None  # None when it could not be read
    problem: str | None  # why it could not be read (short; never the file's contents)


@dataclass(frozen=True)
class Listing:
    state: str  # none, refused or listed
    reason: str | None  # why the directory was refused
    briefings: tuple[StoredBriefing, ...]  # one data class, the latest first
    older: int  # files of that data class on older market dates, not opened


def _valid_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _names(directory: Path, data_class: str) -> list[re.Match]:
    """The data class's briefing file names, the newest market date first."""

    with os.scandir(directory) as entries:
        found = [NAME.fullmatch(entry.name) for entry in entries]
    return sorted(
        (
            match
            for match in found
            if match is not None and match["kind"] == data_class and _valid_date(match["market"])
        ),
        key=lambda match: (match["market"], match.string),
        reverse=True,
    )


def _newest_dates(names: list[re.Match]) -> tuple[list[re.Match], int]:
    """Whole market dates, newest first, until at least MAX_LISTED files; and the rest's count."""

    chosen: list[re.Match] = []
    for _, group in groupby(names, key=lambda match: match["market"]):
        if len(chosen) >= MAX_LISTED:
            break
        chosen.extend(group)
    return chosen, len(names) - len(chosen)


def _latest_first(entries: Iterable[StoredBriefing]) -> tuple[StoredBriefing, ...]:
    entries = list(entries)
    readable = [entry for entry in entries if entry.briefing is not None]
    unreadable = [entry for entry in entries if entry.briefing is None]
    readable.sort(key=lambda e: (e.market_date, e.briefing.generated_at, e.name), reverse=True)
    unreadable.sort(key=lambda e: (e.market_date, e.name), reverse=True)
    return (*readable, *unreadable)


def list_briefings(data_class: str, directory: Path | None = None) -> Listing:
    """The written briefings of one data class, the latest first (module docstring)."""

    if data_class not in DATA_CLASSES:
        raise ValueError(f"unknown data class {data_class!r}")
    try:
        resolved = readable_directory(directory)
        if resolved is None:
            return Listing(NONE, None, (), 0)
        names = _names(resolved, data_class)
    except BriefingUnreadable as error:
        return Listing(REFUSED, str(error), (), 0)
    except OSError as error:
        reason = f"the briefing directory could not be listed ({type(error).__name__})"
        return Listing(REFUSED, reason, (), 0)
    chosen, older = _newest_dates(names)
    entries = []
    for match in chosen:
        try:
            briefing, problem = read_briefing(resolved / match.string), None
        except BriefingUnreadable as error:
            briefing, problem = None, short(str(error))
        entries.append(
            StoredBriefing(
                match.string, match["market"], match["kind"], match["label"], briefing, problem
            )
        )
    return Listing(LISTED if entries else NONE, None, _latest_first(entries), older)


@dataclass(frozen=True)
class DeliveryStatus:
    channel: str | None
    status: str  # sent, failed, in_progress, not_recorded or unreadable
    attempts: int
    at: str | None  # the latest entry's time (UTC stamp, as the log writes it)
    detail: str | None


def _unreadable(reason: str) -> tuple[DeliveryStatus, ...]:
    return (DeliveryStatus(None, UNREADABLE, 0, None, short(reason)),)


def _valid_row(row: tuple) -> bool:
    channel, attempt, status, at, detail = row
    return (
        isinstance(channel, str)
        and re.fullmatch(r"[a-z0-9_-]{1,40}", channel, re.ASCII) is not None
        and type(attempt) is int
        and attempt >= 1
        and status in LOG_STATUSES
        and isinstance(at, str)
        and STAMP.fullmatch(at) is not None
        and (detail is None or isinstance(detail, str))
    )


def _detail(detail: str | None) -> str | None:
    """A log entry's detail when it is short plain text, as the writer keeps it."""

    if detail is None:
        return None
    return detail if DETAIL.fullmatch(detail) else HIDDEN_DETAIL


def _status(channel: str, rows: list[tuple[int, str, str, str | None]]) -> DeliveryStatus:
    latest: dict[int, tuple[str, str, str | None]] = {}
    for attempt, status, at, detail in rows:
        latest[attempt] = (status, at, detail)
    attempts = len(latest)
    for wanted, word in (("sent", SENT), ("claimed", IN_PROGRESS), ("failed", FAILED)):
        found = [entry for entry in latest.values() if entry[0] == wanted]
        if found:
            _, at, detail = max(found, key=lambda entry: entry[1])
            return DeliveryStatus(channel, word, attempts, at, _detail(detail))
    return DeliveryStatus(channel, NOT_RECORDED, attempts, None, None)


def _log_rows(path: Path, market_date: str, data_class: str) -> list[tuple]:
    """The log's rows for one market date and class, read only (module docstring)."""

    descriptor, info = _open_private(path, "delivery log", MAX_LOG_BYTES)
    with os.fdopen(descriptor, "rb") as handle:
        header = handle.read(100)
    if len(header) < 100 or not header.startswith(SQLITE_HEADER):
        raise BriefingUnreadable("the delivery log is not an SQLite database")
    if header[18] != 1 or header[19] != 1:  # 2 is write-ahead logging
        raise BriefingUnreadable(
            "the delivery log is not in rollback-journal mode, so it cannot be read without "
            "creating files"
        )
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        rows = connection.execute(
            "SELECT channel, attempt, status, at, detail FROM deliveries "
            "WHERE market_date = ? AND data_class = ? ORDER BY seq",
            (market_date, data_class),
        ).fetchall()
    finally:
        connection.close()
    after = os.lstat(path)
    if (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
        raise BriefingUnreadable("the delivery log was replaced while it was read")
    return rows


def delivery_status(
    market_date: str, data_class: str, directory: Path | None = None
) -> tuple[DeliveryStatus, ...]:
    """
    What the duplicate-send log recorded for one market date and data class,
    per channel: sent, failed, or an attempt with no recorded end. One
    NOT_RECORDED entry when the log holds nothing for them (or does not
    exist); one UNREADABLE entry, with a short reason, when it cannot be
    read or holds an entry this module cannot read.
    """

    try:
        resolved = readable_directory(directory)
    except BriefingUnreadable as error:
        return _unreadable(str(error))
    path = None if resolved is None else resolved / LOG_NAME
    if path is None or not os.path.lexists(path):
        return (DeliveryStatus(None, NOT_RECORDED, 0, None, None),)
    try:
        rows = _log_rows(path, market_date, data_class)
    except BriefingUnreadable as error:
        return _unreadable(str(error))
    except (OSError, sqlite3.Error) as error:
        return _unreadable(f"the delivery log could not be read ({type(error).__name__})")
    if not all(_valid_row(row) for row in rows):
        return _unreadable("the delivery log holds an entry that is not one the log writes")
    channels: dict[str, list] = {}
    for channel, attempt, status, at, detail in rows:
        channels.setdefault(channel, []).append((attempt, status, at, detail))
    if not channels:
        return (DeliveryStatus(None, NOT_RECORDED, 0, None, None),)
    return tuple(_status(channel, found) for channel, found in sorted(channels.items()))
