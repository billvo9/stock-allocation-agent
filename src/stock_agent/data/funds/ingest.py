"""
Validate one adapter batch and write it once to a private store: the only
module in this package that writes files.

Private root (caller-given; there is no default)
    The root must lie outside the repository, or under an ignored path such
    as `data/raw/`. Vendor data is never committed. Batch directories are
    created with mode 0o700.

Layout: `<root>/<batch_id>/`
    batch.json                 the batch record (request, adapter, retrieval time, notes)
    raw_payload.bin            the provider's bytes, exactly as retrieved
    prices.parquet, distributions.parquet, splits.parquet
    missing_sessions.parquet   the structured missing-session report
    manifest.json              file and table hashes, written last

`batch_id` is `<retrieved_at>-<12 hex>` from a hash of the record, so the same
retrieval stored twice has the same id and is refused (`already_exists`). A
batch is staged in a hidden directory inside the root and renamed into place;
on any failure the files this call wrote are removed and the error re-raised.

Lockbox cap
    The request's `end_exclusive` may not pass `MODEL_HOLDOUT_START`, and a
    single row dated on or after it, in any date column, refuses the whole
    batch. Nothing is dropped to make a batch fit.

Assumed availability
    No rule is approved (OD9), so `assumed_availability_rule` must be None;
    any value is refused (`pending_od9`) and no label is ever written.

The store keeps observations only. It never adjusts a close, fills a gap or
computes a return. The manifest hashes detect inconsistent edits; they are
integrity, not provenance: a consistent rewrite that recomputes every hash
passes them (see `reader`).
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

from stock_agent.data.funds.adapter import RawBatch
from stock_agent.data.funds.schema import (
    BATCH_FILE,
    MANIFEST_FILE,
    MISSING_SESSIONS,
    PAYLOAD_FILE,
    SCHEMA_VERSION,
    STAGING_PREFIX,
    TABLE_FILES,
    BatchRecord,
    Instrument,
    RefusalReason,
    check_availability_rule,
    derive_batch_id,
    dumps_canonical,
    dumps_file,
    loads_strict,
    refuse,
    utc_timestamp,
)
from stock_agent.data.funds.validation import validate_batch, validate_request
from stock_agent.model_validation.audit import file_sha256, table_sha256


@dataclass(frozen=True, eq=False)
class IngestResult:
    path: Path
    record: BatchRecord
    missing_sessions: pd.DataFrame


def _write_json(path: Path, value: object) -> None:
    path.write_text(dumps_file(value), encoding="utf-8")


def build_record(raw: RawBatch, *, assumed_availability_rule: object = None) -> BatchRecord:
    """The batch record for a raw batch, with its derived id."""

    if not isinstance(raw, RawBatch):
        raise refuse(RefusalReason.INVALID_FIELD, f"expected a RawBatch, got {type(raw)}")
    if not isinstance(raw.raw_payload, bytes):
        raise refuse(RefusalReason.INVALID_FIELD, "raw_payload must be bytes")
    check_availability_rule(assumed_availability_rule)
    fund_ids, start, end_exclusive = validate_request(raw.fund_ids, raw.start, raw.end_exclusive)
    instruments = raw.instruments
    if not isinstance(instruments, tuple) or not all(
        isinstance(item, Instrument) for item in instruments
    ):
        raise refuse(RefusalReason.INVALID_FIELD, "instruments must be a tuple of Instrument")
    draft = BatchRecord(
        batch_id="",
        adapter_id=raw.adapter_id,
        adapter_version=raw.adapter_version,
        fund_ids=fund_ids,
        start=start,
        end_exclusive=end_exclusive,
        instruments=tuple(sorted(instruments, key=lambda item: item.fund_id)),
        retrieved_at=utc_timestamp("retrieved_at", raw.retrieved_at),
        raw_payload_sha256=hashlib.sha256(raw.raw_payload).hexdigest(),
        terms_note=raw.terms_note,
        vintage_note=raw.vintage_note,
    )
    # A JSON round trip gives the writer exactly the reader's view, and parses
    # every field with the reader's checks.
    fields = loads_strict(dumps_canonical(draft.to_json_dict(include_id=False)))
    batch_id = derive_batch_id(fields)
    return BatchRecord.from_json_dict({"batch_id": batch_id, **fields})


def _with_batch_id(name: str, frame: object, batch_id: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{name} is not a DataFrame")
    if "batch_id" in frame.columns:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{name} may not set batch_id")
    return frame.assign(batch_id=batch_id)


def ingest_batch(
    root: str | Path,
    raw: RawBatch,
    *,
    assumed_availability_rule: object = None,
) -> IngestResult:
    """Validate `raw`, then write it once under `root`; return its directory and report."""

    base = Path(root)
    if base.is_symlink() or not base.is_dir():
        raise refuse(RefusalReason.INVALID_FIELD, f"root {base} is not a directory")
    record = build_record(raw, assumed_availability_rule=assumed_availability_rule)
    frames = [
        _with_batch_id(name, getattr(raw, name), record.batch_id)
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
        written[-1].write_bytes(raw.raw_payload)
        written.append(staging / BATCH_FILE)
        _write_json(written[-1], record.to_json_dict())
        files: dict[str, dict[str, object]] = {
            BATCH_FILE: {"file_sha256": file_sha256(staging / BATCH_FILE)},
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
        _write_json(
            written[-1],
            {"schema_version": SCHEMA_VERSION, "batch_id": record.batch_id, "files": files},
        )
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
    return IngestResult(path=final, record=record, missing_sessions=missing)


__all__ = ["IngestResult", "build_record", "ingest_batch"]
