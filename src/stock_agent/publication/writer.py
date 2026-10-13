"""
The write-once publication writer: the only module in this package that
writes files.

A publication is staged in a hidden directory inside `root` (same
filesystem, so the final rename is atomic), written rows first and
manifest last, then renamed to its id. An existing id is refused, even an
empty directory or a dangling symlink. On any failure the files this call
wrote are removed and the error is re-raised. Callers inject the root, the
clock and the code state; nothing here reads a clock or runs git.

`code_sha`, `code_dirty` and `created_at` are the producer's claims. The
manifest hashes protect them against accidental change, but nothing
authenticates them: anyone who can write the directory can edit and rehash.
Files are not fsynced; a torn write fails the reader's hash checks.
"""

from __future__ import annotations

import contextlib
import errno
import os
import tempfile
from pathlib import Path

import pandas as pd

from stock_agent.model_validation.audit import file_sha256, table_sha256
from stock_agent.publication.envelope import (
    ENVELOPE_FILE,
    MANIFEST_FILE,
    ROWS_FILE,
    SCHEMA_VERSION,
    CodeState,
    Draft,
    Envelope,
    RefusalReason,
    derive_publication_id,
    dumps_canonical,
    dumps_file,
    format_timestamp,
    loads_strict,
    refuse,
)
from stock_agent.publication.policy import (
    check_schema,
    knowledge_time_for,
    policy_for,
    validate,
)

STAGING_PREFIX = ".staging-"


def _write_json(path: Path, value: object) -> None:
    path.write_text(dumps_file(value), encoding="utf-8")


def build_envelope(
    draft: Draft, *, as_of: pd.Timestamp, created_at: pd.Timestamp, code_state: CodeState
) -> Envelope:
    """The complete envelope for a draft. Validation is separate (`policy.validate`)."""

    if not isinstance(draft, Draft):
        raise refuse(RefusalReason.INVALID_FIELD, f"expected a Draft, got {type(draft)}")
    if not isinstance(code_state, CodeState) or not isinstance(code_state.dirty, bool):
        raise refuse(RefusalReason.INVALID_FIELD, "code_state needs a sha and a boolean dirty")
    check_schema(policy_for(draft.kind), draft.rows)
    fields = {
        "kind": str(draft.kind),
        "schema_version": SCHEMA_VERSION,
        "as_of": format_timestamp("as_of", as_of),
        "knowledge_time": format_timestamp(
            "knowledge_time", knowledge_time_for(draft.kind, draft.rows)
        ),
        "knowledge_time_rule": str(draft.knowledge_time_rule),
        "created_at": format_timestamp("created_at", created_at),
        "code_sha": code_state.sha,
        "code_dirty": code_state.dirty,
        "input_hashes": list(draft.input_hashes),
        "universe_id": draft.universe_id,
        "method_id": draft.method_id,
        "method_version": draft.method_version,
        "parameters": dict(draft.parameters),
        "evidence_status": str(draft.evidence_status),
        "limitations": sorted(str(limitation) for limitation in draft.limitations),
    }
    # A JSON round trip makes the writer's id inputs exactly the reader's, and
    # refuses values JSON cannot represent (numpy scalars, non-string keys).
    fields = loads_strict(dumps_canonical(fields))
    publication_id = derive_publication_id(fields, table_sha256(draft.rows))
    return Envelope.from_json_dict({"publication_id": publication_id, **fields})


def write_publication(
    root: str | Path,
    draft: Draft,
    *,
    as_of: pd.Timestamp,
    created_at: pd.Timestamp,
    code_state: CodeState,
) -> Path:
    """Validate, then write one publication under `root`; return its directory."""

    base = Path(root)
    if not base.is_dir():
        raise refuse(RefusalReason.INVALID_FIELD, f"root {base} is not a directory")
    envelope = build_envelope(draft, as_of=as_of, created_at=created_at, code_state=code_state)
    rows = draft.rows
    validate(envelope, rows)
    final = base / envelope.publication_id
    if final.exists() or final.is_symlink():
        raise refuse(RefusalReason.ALREADY_EXISTS, f"{final.name} already exists")
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=base))
    written: list[Path] = []
    try:
        written.append(staging / ROWS_FILE)
        rows.to_parquet(written[-1], index=False)
        written.append(staging / ENVELOPE_FILE)
        _write_json(written[-1], envelope.to_json_dict())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "publication_id": envelope.publication_id,
            "files": {
                ENVELOPE_FILE: {"file_sha256": file_sha256(staging / ENVELOPE_FILE)},
                ROWS_FILE: {
                    "file_sha256": file_sha256(staging / ROWS_FILE),
                    "table_sha256": table_sha256(rows),
                    "rows": len(rows),
                },
            },
        }
        written.append(staging / MANIFEST_FILE)
        _write_json(written[-1], manifest)
        # The existence check above is not atomic with this rename. A
        # concurrent writer of the same id wrote the same content (the id binds
        # it, up to a 48-bit hash collision). POSIX rename may replace an empty
        # directory created in that window; a non-empty one is refused here.
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
    return final
