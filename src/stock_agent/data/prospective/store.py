"""
The prospective store's write-once writer: the only module that writes into
a prospective root. Mirrors F1's `ingest` (staging directory, tables first,
manifest last, atomic rename, cleanup on failure) with the prospective
contract (`validation`).

Private root (caller-given; there is no default)
    The collection script checks it with `private_paths.private_root`: outside
    every repository, or strictly inside an ignored `data/raw/`.
    `store_root_refusal` checks a root without creating anything, so the
    collection can refuse a bad root before any request. `prepare_store`
    then creates a missing root (and missing parents) with mode 0o700 and
    writes the marker, only when a verified batch is about to be written; it
    writes the marker into an existing root only when that root is empty and
    grants no group or other access, and never changes an existing
    directory's mode. A non-empty root without this store's marker (an F1
    store, for example) is refused.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.schema import (
    MISSING_SESSIONS,
    PAYLOAD_FILE,
    TABLE_FILES,
    FundDataRefused,
    Instrument,
    RefusalReason,
    dumps_canonical,
    dumps_file,
    loads_strict,
    refuse,
    utc_timestamp,
)
from stock_agent.data.private_paths import make_private_directories, root_refusal
from stock_agent.data.prospective.validation import (
    MANIFEST_FILE,
    MARKER_FILE,
    RECORD_FILE,
    SCHEMA_VERSION,
    STAGING_PREFIX,
    ProspectiveRecord,
    check_marker,
    derive_batch_id,
    marker_document,
    validate_batch,
)
from stock_agent.model_validation.audit import file_sha256, table_sha256


@dataclass(frozen=True, eq=False)
class ProspectiveBatch:
    """
    One prospective retrieval as the adapter built it, after its own checks.
    The tables carry every F1 column except `batch_id`, which the writer
    assigns. `start` and `end` are the requested sessions, both inclusive.
    """

    adapter_id: str
    adapter_version: str
    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    instruments: tuple[Instrument, ...]
    retrieved_at: pd.Timestamp
    raw_payload: bytes
    anchors_sha256: str | None
    terms_note: str
    vintage_note: str
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame


@dataclass(frozen=True, eq=False)
class ProspectiveIngestResult:
    path: Path
    record: ProspectiveRecord
    missing_sessions: pd.DataFrame


def store_root_refusal(root: Path) -> str | None:
    """
    Read-only: why `root` cannot be a prospective store, or None. A missing
    root or an empty private directory is acceptable; an existing prospective
    store is accepted as it is. Nothing is created or changed.
    """

    refusal = root_refusal(root)
    if refusal is not None or not root.exists() or not any(root.iterdir()):
        return refusal
    try:
        check_marker(root)
    except FundDataRefused as error:
        return (
            f"--root {root} is neither empty nor a prospective store ({error}); use a separate "
            "directory, never the research store"
        )
    return None


def prepare_store(root: Path) -> str | None:
    """
    Make `root` a prospective store: create it (and missing parents, mode
    0o700) with the marker, or write the marker into an existing empty private
    directory, or accept an existing prospective store. Returns the refusal
    message, or None.
    """

    refusal = store_root_refusal(root)
    if refusal is not None:
        return refusal
    make_private_directories(root)
    if not any(root.iterdir()):
        marker = root / MARKER_FILE
        marker.write_text(dumps_file(marker_document()), encoding="utf-8")
        os.chmod(marker, 0o600)
    return None


def build_record(batch: ProspectiveBatch) -> ProspectiveRecord:
    """The batch record with its derived id, after a JSON round trip (the reader's view)."""

    if not isinstance(batch, ProspectiveBatch):
        raise refuse(RefusalReason.INVALID_FIELD, f"expected a ProspectiveBatch, got {type(batch)}")
    if not isinstance(batch.raw_payload, bytes):
        raise refuse(RefusalReason.INVALID_FIELD, "raw_payload must be bytes")
    instruments = batch.instruments
    if not isinstance(instruments, tuple) or not all(
        isinstance(item, Instrument) for item in instruments
    ):
        raise refuse(RefusalReason.INVALID_FIELD, "instruments must be a tuple of Instrument")
    draft = ProspectiveRecord(
        batch_id="",
        adapter_id=batch.adapter_id,
        adapter_version=batch.adapter_version,
        fund_ids=tuple(batch.fund_ids),
        start=utc_timestamp("start", batch.start),
        end=utc_timestamp("end", batch.end),
        instruments=tuple(sorted(instruments, key=lambda item: item.fund_id)),
        retrieved_at=utc_timestamp("retrieved_at", batch.retrieved_at),
        raw_payload_sha256=hashlib.sha256(batch.raw_payload).hexdigest(),
        anchors_sha256=batch.anchors_sha256,
        terms_note=batch.terms_note,
        vintage_note=batch.vintage_note,
    )
    fields = loads_strict(dumps_canonical(draft.to_json_dict(include_id=False)))
    return ProspectiveRecord.from_json_dict({"batch_id": derive_batch_id(fields), **fields})


def _with_batch_id(name: str, frame: object, batch_id: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{name} is not a DataFrame")
    if "batch_id" in frame.columns:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{name} may not set batch_id")
    return frame.assign(batch_id=batch_id)


def ingest_prospective(root: str | Path, batch: ProspectiveBatch) -> ProspectiveIngestResult:
    """Validate `batch`, then write it once under the prospective `root`."""

    base = check_marker(root)
    record = build_record(batch)
    frames = [
        _with_batch_id(name, getattr(batch, name), record.batch_id)
        for name in ("prices", "distributions", "splits")
    ]
    tables, missing = validate_batch(record, *frames)
    tables[MISSING_SESSIONS.name] = missing

    final = base / record.batch_id
    if final.exists() or final.is_symlink():
        raise refuse(RefusalReason.ALREADY_EXISTS, f"batch {final.name} already exists")
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=base))
    written: list[Path] = []
    try:
        written.append(staging / PAYLOAD_FILE)
        written[-1].write_bytes(batch.raw_payload)
        written.append(staging / RECORD_FILE)
        written[-1].write_text(dumps_file(record.to_json_dict()), encoding="utf-8")
        files: dict[str, dict[str, object]] = {
            RECORD_FILE: {"file_sha256": file_sha256(staging / RECORD_FILE)},
            PAYLOAD_FILE: {"file_sha256": file_sha256(staging / PAYLOAD_FILE)},
        }
        for name, filename in TABLE_FILES.items():
            written.append(staging / filename)
            tables[name].to_parquet(written[-1], index=False)
            files[filename] = {
                "file_sha256": file_sha256(written[-1]),
                "table_sha256": table_sha256(tables[name]),
                "rows": len(tables[name]),
            }
        written.append(staging / MANIFEST_FILE)
        manifest = {
            "prospective_schema_version": SCHEMA_VERSION,
            "batch_id": record.batch_id,
            "files": files,
        }
        written[-1].write_text(dumps_file(manifest), encoding="utf-8")
        try:
            os.rename(staging, final)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise refuse(RefusalReason.ALREADY_EXISTS, f"{final.name} appeared") from error
            raise
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            staging.rmdir()
        raise
    return ProspectiveIngestResult(path=final, record=record, missing_sessions=missing)


__all__ = [
    "ProspectiveBatch",
    "ProspectiveIngestResult",
    "build_record",
    "ingest_prospective",
    "prepare_store",
    "store_root_refusal",
]
