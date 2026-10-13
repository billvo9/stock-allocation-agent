"""
The fail-closed publication reader. It never imports the writer.

Checks run in a fixed order: path and symlinks, the exact file set,
strict JSON and the schema version, file and table hashes, the envelope's
fields, the recomputed id, then the shared `policy.validate` (whose
provenance check runs last), then the dirty-tree rule. Any failure raises
`PublicationRefused` with a machine-readable reason; nothing is repaired,
skipped or returned partially.

Hashes establish integrity, not trusted provenance: see the package
docstring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.model_validation.audit import file_sha256, table_sha256
from stock_agent.publication.envelope import (
    ENVELOPE_FILE,
    MANIFEST_FILE,
    PUBLICATION_ID_PATTERN,
    ROWS_FILE,
    Envelope,
    PublicationRefused,
    RefusalReason,
    check_version,
    derive_publication_id,
    loads_strict,
    refuse,
)
from stock_agent.publication.policy import Publication, policy_for, validate

FILE_SET = frozenset({MANIFEST_FILE, ENVELOPE_FILE, ROWS_FILE})
_MANIFEST_FIELDS = {"schema_version", "publication_id", "files"}
_FILE_ENTRY_FIELDS = {
    ENVELOPE_FILE: {"file_sha256"},
    ROWS_FILE: {"file_sha256", "table_sha256", "rows"},
}


def _require_fields(label: str, value: object, expected: set[str]) -> dict:
    if not isinstance(value, dict):
        raise refuse(RefusalReason.INVALID_FIELD, f"{label} is not a JSON object")
    if set(value) != expected:
        raise refuse(
            RefusalReason.FIELD_SET, f"{label} has {sorted(value)}, expected {sorted(expected)}"
        )
    return value


def _check_directory(directory: Path) -> None:
    if directory.is_symlink():
        raise refuse(RefusalReason.SYMLINK, f"{directory.name} is a symlink")
    if not directory.is_dir() or not PUBLICATION_ID_PATTERN.fullmatch(directory.name):
        raise refuse(RefusalReason.NOT_A_PUBLICATION, str(directory))
    entries = list(directory.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise refuse(RefusalReason.SYMLINK, f"{entry.name} is a symlink")
    names = {entry.name for entry in entries}
    if names != FILE_SET or not all(entry.is_file() for entry in entries):
        raise refuse(RefusalReason.FILE_SET, f"files {sorted(names)}, expected {sorted(FILE_SET)}")


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise refuse(RefusalReason.INVALID_JSON, f"{path.name} is not UTF-8") from error
    return loads_strict(text)


def read_publication(path: str | Path, *, allow_dirty: bool = False) -> Publication:
    """
    Read and verify one publication directory.

    `allow_dirty=True` is for Research mode only. It admits a dirty-tree
    publication with status `descriptive`; anything above `descriptive` from
    a dirty tree is refused regardless.
    """

    directory = Path(path)
    _check_directory(directory)

    manifest = _require_fields("manifest", _read_json(directory / MANIFEST_FILE), _MANIFEST_FIELDS)
    version = check_version(manifest["schema_version"])
    if manifest["publication_id"] != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "manifest id differs from the directory name")
    files = _require_fields("manifest files", manifest["files"], set(_FILE_ENTRY_FIELDS))
    for name, expected in _FILE_ENTRY_FIELDS.items():
        entry = _require_fields(f"manifest entry {name}", files[name], expected)
        if entry["file_sha256"] != file_sha256(directory / name):
            raise refuse(RefusalReason.HASH_MISMATCH, f"{name} does not match the manifest")

    envelope = Envelope.from_json_dict(_read_json(directory / ENVELOPE_FILE))
    if envelope.schema_version != version:
        raise refuse(RefusalReason.INVALID_FIELD, "manifest and envelope versions differ")
    if envelope.publication_id != directory.name:
        raise refuse(RefusalReason.ID_MISMATCH, "envelope id differs from the directory name")

    try:
        rows = pd.read_parquet(directory / ROWS_FILE)
    except Exception as error:  # untrusted bytes: any parser failure refuses
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{ROWS_FILE} is unreadable") from error
    rows_entry = files[ROWS_FILE]
    rows_sha256 = table_sha256(rows)
    if type(rows_entry["rows"]) is not int or rows_entry["rows"] != len(rows):
        raise refuse(RefusalReason.HASH_MISMATCH, "the row count differs from the manifest")
    if rows_entry["table_sha256"] != rows_sha256:
        raise refuse(RefusalReason.HASH_MISMATCH, "rows differ from the manifest")
    expected_id = derive_publication_id(envelope.to_json_dict(include_id=False), rows_sha256)
    if expected_id != envelope.publication_id:
        raise refuse(RefusalReason.ID_MISMATCH, "the id does not match the content")

    validate(envelope, rows)
    if envelope.code_dirty and not allow_dirty:
        raise refuse(RefusalReason.DIRTY_TREE, "dirty-tree publications need Research mode")
    return policy_for(envelope.kind).publication_type(envelope, rows.copy(), directory)


__all__ = ["PublicationRefused", "read_publication"]
