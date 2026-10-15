"""
The prospective ledger: write-once, append-only and hash-chained.

One JSON line per entry, in a private, Git-ignored file (default
data/raw/prospective_forecasts/ledger.jsonl):

    {"ledger_schema_version", "sequence", "kind", "recorded_at", "payload",
     "prev_entry_sha256", "entry_sha256"}

`entry_sha256` is the SHA-256 of the canonical JSON of every other field,
and `prev_entry_sha256` is the previous line's (zeros for the first), so an
edited, inserted, removed or reordered line breaks the chain and the whole
ledger is refused on read. Kinds: "forecast" (one forecast run: every
asset, horizon and target at one as-of session) and "score" (outcomes of
matured forecast rows). Nothing is ever rewritten; a correction would be a
new entry.

Appending takes an exclusive lock, re-reads and verifies the whole chain,
runs the caller's guard on the verified entries (so a duplicate check
cannot race), then writes one line and fsyncs it.

Integrity, not provenance. The chain detects an edit inside the ledger. It
cannot detect a truncated tail (dropping the last lines leaves a valid
chain) or a consistent rewrite of the whole file: removing the last line is
undetectable from the file alone. `head()` returns the last entry's hash.
The job prints it after every append, and the owner keeps a copy outside
the ledger; `check_head` (the job's `report --expect-head`) then refuses a
ledger that no longer holds that hash, so a later truncation or rewrite is
detected.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.model_diagnostics.record import canonical_json, sha256_json

LEDGER_SCHEMA_VERSION = "1.0"
KINDS = ("forecast", "score")
GENESIS = "0" * 64
DEFAULT_LEDGER = Path("data") / "raw" / "prospective_forecasts" / "ledger.jsonl"
_FIELDS = {
    "ledger_schema_version",
    "sequence",
    "kind",
    "recorded_at",
    "payload",
    "prev_entry_sha256",
    "entry_sha256",
}


class LedgerError(ValueError):
    """The ledger is edited, malformed or a write was refused."""


@dataclass(frozen=True)
class LedgerEntry:
    sequence: int
    kind: str
    recorded_at: str
    payload: Mapping[str, object]
    prev_entry_sha256: str
    entry_sha256: str


def _refuse_constant(token: str) -> None:
    raise LedgerError(f"The ledger holds a non-finite number ({token}).")


def _parse(line: str, number: int, previous: str) -> LedgerEntry:
    try:
        raw = json.loads(line, parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        raise LedgerError(f"Line {number + 1} is not JSON ({error.msg}).") from None
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise LedgerError(f"Line {number + 1} does not hold exactly the ledger fields.")
    claimed = raw.pop("entry_sha256")
    if raw["ledger_schema_version"] != LEDGER_SCHEMA_VERSION:
        raise LedgerError(f"Line {number + 1}: ledger schema {raw['ledger_schema_version']!r}.")
    if raw["sequence"] != number or raw["prev_entry_sha256"] != previous:
        raise LedgerError(f"Line {number + 1} breaks the chain (removed, inserted or reordered).")
    if sha256_json(raw) != claimed:
        raise LedgerError(f"Line {number + 1} was edited (its hash does not match).")
    if raw["kind"] not in KINDS or not isinstance(raw["payload"], dict):
        raise LedgerError(f"Line {number + 1} has an unknown kind or payload.")
    if canonical_json({**raw, "entry_sha256": claimed}) != line:
        raise LedgerError(f"Line {number + 1} is not in canonical form (edited).")
    return LedgerEntry(
        sequence=number,
        kind=raw["kind"],
        recorded_at=raw["recorded_at"],
        payload=raw["payload"],
        prev_entry_sha256=raw["prev_entry_sha256"],
        entry_sha256=claimed,
    )


def _entries(text: str) -> tuple[LedgerEntry, ...]:
    if text and not text.endswith("\n"):
        raise LedgerError("The last line is incomplete (an interrupted write); nothing is read.")
    entries, previous = [], GENESIS
    for number, line in enumerate(text.splitlines()):
        entry = _parse(line, number, previous)
        entries.append(entry)
        previous = entry.entry_sha256
    return tuple(entries)


def _check_path(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise LedgerError(f"{path} must be a regular file.")


def read_ledger(path: str | Path) -> tuple[LedgerEntry, ...]:
    """Every entry after verifying the chain; () when the ledger does not exist."""

    path = Path(path)
    _check_path(path)
    if not path.exists():
        return ()
    return _entries(path.read_text(encoding="utf-8"))


def head(entries: Sequence[LedgerEntry]) -> str:
    """The last entry's hash (GENESIS for an empty ledger)."""

    return entries[-1].entry_sha256 if entries else GENESIS


def check_head(entries: Sequence[LedgerEntry], expected: str) -> None:
    """Refuse a ledger that no longer holds a head hash kept outside it."""

    if expected not in {entry.entry_sha256 for entry in entries}:
        raise LedgerError(
            f"The kept head {expected[:12]}... is not in the ledger: lines were removed or the "
            "file was rewritten."
        )


def append_entry(
    path: str | Path,
    kind: str,
    payload: Mapping[str, object],
    *,
    recorded_at: pd.Timestamp,
    guard: Callable[[tuple[LedgerEntry, ...]], None] | None = None,
) -> LedgerEntry:
    """Append one entry under an exclusive lock, after verifying the chain."""

    if kind not in KINDS:
        raise LedgerError(f"kind must be one of {KINDS}.")
    stamp = pd.Timestamp(recorded_at)
    if stamp.tzinfo is None:
        raise LedgerError("recorded_at must be timezone-aware.")
    path = Path(path)
    _check_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            entries = _entries(handle.read())
            if guard is not None:
                guard(entries)
            body = {
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "sequence": len(entries),
                "kind": kind,
                "recorded_at": stamp.tz_convert("UTC").isoformat(),
                "payload": json.loads(canonical_json(payload)),
                "prev_entry_sha256": head(entries),
            }
            digest = sha256_json(body)
            handle.write(canonical_json({**body, "entry_sha256": digest}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return LedgerEntry(
        sequence=body["sequence"],
        kind=kind,
        recorded_at=body["recorded_at"],
        payload=body["payload"],
        prev_entry_sha256=body["prev_entry_sha256"],
        entry_sha256=digest,
    )
