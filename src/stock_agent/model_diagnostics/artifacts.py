"""
The only file I/O of model_diagnostics: write and read run directories and
append to the run ledger.

Layout under an output root (default reports/, ignored by Git):

    runs/<run_id>/record.json
    runs/<run_id>/<table>.parquet
    runs/<run_id>/manifest.json      written last: completion marker
    ledger.jsonl                     one line per run attempt, append-only

A run is written to a temporary sibling directory and renamed into place, so
a reader never sees a half-written run. An existing run directory is never
overwritten. Readers check the manifest's schema version, every file's
SHA-256, and every table's contract.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from stock_agent.model_diagnostics import contract
from stock_agent.model_diagnostics.record import canonical_json
from stock_agent.model_diagnostics.runner import DiagnosticsRun
from stock_agent.model_validation.audit import file_sha256, table_sha256

RUNS_DIRECTORY = "runs"
LEDGER_FILE = "ledger.jsonl"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_run(
    run: DiagnosticsRun, root: str | Path, *, extra_record: Mapping | None = None
) -> Path:
    """Write a complete run directory under root/runs/ and return its path."""

    runs = Path(root) / RUNS_DIRECTORY
    runs.mkdir(parents=True, exist_ok=True)
    final = runs / run.run_id
    if final.exists():
        raise FileExistsError(f"Run directory {final} already exists; runs are never overwritten.")
    staging = runs / f".{run.run_id}.partial"
    staging.mkdir()
    files = {}
    for name, table in run.tables.items():
        contract.validate(name, table)
        path = staging / f"{name}.parquet"
        table.to_parquet(path, index=False)
        files[path.name] = {
            "table": name,
            "rows": len(table),
            "content_sha256": table_sha256(table),
        }
    record = {**run.record, **dict(extra_record or {})}
    _write_json(staging / "record.json", record)
    files["record.json"] = {}
    for filename, entry in files.items():
        entry["file_sha256"] = file_sha256(staging / filename)
    manifest = {
        "output_schema_version": contract.OUTPUT_SCHEMA_VERSION,
        "run_id": run.run_id,
        "files": files,
    }
    _write_json(staging / "manifest.json", manifest)
    os.replace(staging, final)
    return final


def append_ledger(root: str | Path, entry: Mapping[str, object]) -> None:
    """Append one JSON line per run attempt (successful or failed)."""

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    with (path / LEDGER_FILE).open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(entry) + "\n")


def read_ledger(root: str | Path) -> list[dict]:
    path = Path(root) / LEDGER_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_run(directory: str | Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Read and verify a run directory: (record, tables)."""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{directory} has no manifest.json; the run is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract.check_version(manifest["output_schema_version"])
    tables = {}
    for filename, entry in manifest["files"].items():
        path = directory / filename
        if file_sha256(path) != entry["file_sha256"]:
            raise ValueError(f"{path} does not match its manifest hash.")
        if "table" not in entry:
            continue
        table = pd.read_parquet(path)
        contract.validate(entry["table"], table)
        if table_sha256(table) != entry["content_sha256"]:
            raise ValueError(f"{path} content does not match its manifest hash.")
        tables[entry["table"]] = table
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    return record, tables
