"""
The fail-closed fund-data reader. It never imports the ingest writer.

`read_batch` checks, in order: the directory and symlinks, the exact file set,
strict JSON and the schema version, every file hash, the batch record and its
recomputed id, the raw payload hash, the table hashes and row counts, then the
full batch validation (the same code ingest ran), and finally that the stored
missing-session report equals a fresh recomputation. Any failure raises
`FundDataRefused`; nothing is repaired, skipped or returned partially. Any
extra file, including an operating-system file such as `.DS_Store`, refuses
the batch.

Integrity, not provenance. The hashes detect an inconsistent edit (a changed
file whose manifest entry was not updated). A consistent rewrite that
recomputes every hash and the batch id passes them, and passes validation too
if the new rows obey the contract. Nothing here proves who wrote a batch.

Follow-up (not implemented): re-parse `raw_payload.bin` with the adapter's
parser on read and require the stored tables to match, so the tables are tied
to the raw bytes and not only to the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.schema import (
    BATCH_FILE,
    BATCH_ID_PATTERN,
    DATA_TABLES,
    DISTRIBUTIONS,
    MANIFEST_FILE,
    MISSING_SESSIONS,
    PAYLOAD_FILE,
    PRICES,
    SCHEMA_VERSION,
    SPLITS,
    STAGING_PREFIX,
    TABLE_FILES,
    BatchRecord,
    FundDataRefused,
    RefusalReason,
    TableSpec,
    batches_frame,
    conform,
    derive_batch_id,
    empty,
    loads_strict,
    refuse,
)
from stock_agent.data.funds.validation import validate_batch
from stock_agent.model_validation.audit import file_sha256, table_sha256

FILE_SET = frozenset({BATCH_FILE, PAYLOAD_FILE, MANIFEST_FILE, *TABLE_FILES.values()})
_MANIFEST_FIELDS = {"schema_version", "batch_id", "files"}
_PLAIN_ENTRY = {"file_sha256"}
_TABLE_ENTRY = {"file_sha256", "table_sha256", "rows"}


@dataclass(frozen=True, eq=False)
class StoredBatch:
    path: Path
    record: BatchRecord
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame
    missing_sessions: pd.DataFrame


@dataclass(frozen=True, eq=False)
class FundStore:
    """Every batch under a root. Keys are unique per batch; batches are separate vintages."""

    batches: pd.DataFrame
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame
    missing_sessions: pd.DataFrame


def _fields(label: str, value: object, expected: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise refuse(RefusalReason.FILE_SET, f"{label} has {found}, expected {sorted(expected)}")
    return value


def _check_directory(directory: Path) -> None:
    if directory.is_symlink():
        raise refuse(RefusalReason.SYMLINK, f"{directory.name} is a symlink")
    if not directory.is_dir() or not BATCH_ID_PATTERN.fullmatch(directory.name):
        raise refuse(RefusalReason.NOT_A_BATCH, str(directory))
    entries = list(directory.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise refuse(RefusalReason.SYMLINK, f"{entry.name} is a symlink")
    names = {entry.name for entry in entries}
    if names != FILE_SET or not all(entry.is_file() for entry in entries):
        raise refuse(RefusalReason.FILE_SET, f"files {sorted(names)}, expected {sorted(FILE_SET)}")


def _read_json(path: Path) -> object:
    try:
        return loads_strict(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise refuse(RefusalReason.INVALID_JSON, f"{path.name} is not UTF-8") from error


def _read_table(directory: Path, spec: TableSpec, entry: dict) -> pd.DataFrame:
    filename = TABLE_FILES[spec.name]
    try:
        frame = pd.read_parquet(directory / filename)
    except Exception as error:  # untrusted bytes: any parser failure refuses
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{filename} is unreadable") from error
    table = conform(spec, frame)
    if type(entry["rows"]) is not int or entry["rows"] != len(table):
        raise refuse(RefusalReason.HASH_MISMATCH, f"{filename} row count differs")
    if entry["table_sha256"] != table_sha256(table):
        raise refuse(RefusalReason.HASH_MISMATCH, f"{filename} rows differ from the manifest")
    return table


def read_batch(path: str | Path) -> StoredBatch:
    """Read and verify one batch directory."""

    directory = Path(path)
    _check_directory(directory)

    manifest = _fields("manifest", _read_json(directory / MANIFEST_FILE), _MANIFEST_FIELDS)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise refuse(RefusalReason.INVALID_FIELD, f"schema {manifest['schema_version']!r}")
    if manifest["batch_id"] != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "manifest id differs from the directory name")
    files = _fields("manifest files", manifest["files"], set(FILE_SET - {MANIFEST_FILE}))
    for name, entry in files.items():
        expected = _TABLE_ENTRY if name in TABLE_FILES.values() else _PLAIN_ENTRY
        _fields(f"manifest entry {name}", entry, expected)
        if entry["file_sha256"] != file_sha256(directory / name):
            raise refuse(RefusalReason.HASH_MISMATCH, f"{name} does not match the manifest")

    record = BatchRecord.from_json_dict(_read_json(directory / BATCH_FILE))
    if record.batch_id != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "record id differs from the directory name")
    if derive_batch_id(record.to_json_dict(include_id=False)) != record.batch_id:
        raise refuse(RefusalReason.ID_MISMATCH, "the batch id does not match the record")
    if file_sha256(directory / PAYLOAD_FILE) != record.raw_payload_sha256:
        raise refuse(RefusalReason.HASH_MISMATCH, "the raw payload differs from the record")

    stored = {
        spec.name: _read_table(directory, spec, files[TABLE_FILES[spec.name]])
        for spec in (*DATA_TABLES, MISSING_SESSIONS)
    }
    tables, missing = validate_batch(
        record, stored[PRICES.name], stored[DISTRIBUTIONS.name], stored[SPLITS.name]
    )
    if table_sha256(missing) != table_sha256(stored[MISSING_SESSIONS.name]):
        raise refuse(RefusalReason.DIAGNOSTICS_MISMATCH, "the missing-session report is stale")
    return StoredBatch(
        path=directory,
        record=record,
        prices=tables[PRICES.name],
        distributions=tables[DISTRIBUTIONS.name],
        splits=tables[SPLITS.name],
        missing_sessions=missing,
    )


def _stack(spec: TableSpec, frames: list[pd.DataFrame]) -> pd.DataFrame:
    present = [frame for frame in frames if len(frame)]
    if not present:
        return empty(spec)
    stacked = pd.concat(present, ignore_index=True)
    return stacked.sort_values(["batch_id", *spec.key], kind="stable").reset_index(drop=True)


def read_store(root: str | Path) -> FundStore:
    """
    Read and verify every batch under `root`. Only a crashed write's staging
    directory (the `.staging-` prefix) is skipped. Every other entry must be a
    valid batch, or the whole store is refused: a hidden or renamed batch can
    never silently drop out, and an operating-system file such as `.DS_Store`
    in the root refuses the store.
    """

    base = Path(root)
    if base.is_symlink() or not base.is_dir():
        raise refuse(RefusalReason.NOT_A_BATCH, f"root {base} is not a directory")
    batches = [
        read_batch(entry)
        for entry in sorted(base.iterdir())
        if not entry.name.startswith(STAGING_PREFIX)
    ]
    frame = batches_frame([batch.record for batch in batches])
    store = FundStore(
        batches=frame,
        prices=_stack(PRICES, [batch.prices for batch in batches]),
        distributions=_stack(DISTRIBUTIONS, [batch.distributions for batch in batches]),
        splits=_stack(SPLITS, [batch.splits for batch in batches]),
        missing_sessions=_stack(MISSING_SESSIONS, [b.missing_sessions for b in batches]),
    )
    known = set(frame["batch_id"])
    for name in ("prices", "distributions", "splits", "missing_sessions"):
        unknown = sorted(set(getattr(store, name)["batch_id"]) - known)
        if unknown:
            raise refuse(RefusalReason.UNKNOWN_BATCH, f"{name} rows name batch {unknown[:3]}")
    return store


__all__ = ["FundDataRefused", "FundStore", "StoredBatch", "read_batch", "read_store"]
