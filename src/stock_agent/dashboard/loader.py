"""
Read-only access to saved T2 run directories (no Streamlit here).

The dashboard reads nothing but what `model_diagnostics.artifacts.read_run`
verifies: record.json and the contract tables listed in a run's manifest,
each checked against its SHA-256. It never touches raw market, macro or
fundamental data, never re-runs a model, and never recomputes inference.

A run is refused, with a machine-readable reason, when:

- it is not a development run (holdout runs stay locked until ticket T6);
- its record does not say holdout values were unused;
- any stored date (rows, label windows, curves, fold bounds) reaches the
  lockbox. The lockbox date is the EARLIEST of the record's two copies and
  the code's MODEL_HOLDOUT_START, so a run cannot move its own boundary;
- the manifest, a hash, or the table contract does not verify.

Refused runs stay visible in the run list with their reason; they are never
silently hidden.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.model_diagnostics.artifacts import RUNS_DIRECTORY, read_ledger, read_run
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

ROOT_ENVIRONMENT_VARIABLE = "STOCK_AGENT_DASHBOARD_ROOT"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

FOLD_DATE_COLUMNS = (
    "holdout_start",
    "train_start",
    "test_start",
    "test_end_exclusive",
    "knowledge_cutoff",
    "train_first_date",
    "train_last_date",
    "train_max_target_end_date",
    "test_first_date",
    "test_last_date",
    "score_available_at",
)
FOLD_JSON_COLUMNS = ("train_rows_by_symbol", "test_rows_by_symbol", "feature_columns")
# Fold columns that are exclusive bounds and may equal the lockbox date.
FOLD_BOUND_COLUMNS = ("holdout_start", "test_end_exclusive")
DECISION_METHODS = {
    "fold_block": "fold_block_t",
    "newey_west": "newey_west",
    "bootstrap": "circular_block_bootstrap",
}
UNSAFE_ROLE = "canary_unsafe_reference"
DATE_COLUMN_NAMES = ("date", "target_start_date", "target_end_date")


class RunRefused(Exception):
    """A run directory the dashboard will not display, with a reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    path: Path
    has_manifest: bool


@dataclass(frozen=True, eq=False)
class RunView:
    """One verified development run, as stored. Treat every field as read-only."""

    run_id: str
    path: Path
    record: dict
    tables: dict[str, pd.DataFrame]
    folds: pd.DataFrame
    holdout_start: pd.Timestamp
    decision_method: str | None
    models: pd.DataFrame


def default_root() -> Path:
    """Output root from the environment (set by the launcher), else <repo>/reports."""

    configured = os.environ.get(ROOT_ENVIRONMENT_VARIABLE)
    return Path(configured) if configured else PROJECT_ROOT / "reports"


def list_runs(root: str | Path) -> list[RunEntry]:
    """
    Run directories under root/runs, newest first. Only names shaped like a
    run id count; symlinks and in-progress (.partial) directories are skipped.
    """

    runs = Path(root) / RUNS_DIRECTORY
    if not runs.is_dir():
        return []
    entries = []
    for path in runs.iterdir():
        if path.is_symlink() or not path.is_dir() or not RUN_ID_PATTERN.match(path.name):
            continue
        entries.append(RunEntry(path.name, path, (path / "manifest.json").is_file()))
    return sorted(entries, key=lambda entry: entry.run_id, reverse=True)


def _timestamp(value: object, name: str) -> pd.Timestamp:
    if value is None:
        raise RunRefused("missing_lockbox_metadata", f"{name} is not recorded.")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _typed_folds(folds: pd.DataFrame) -> pd.DataFrame:
    typed = folds.copy()
    for column in FOLD_DATE_COLUMNS:
        if column in typed.columns:
            typed[column] = pd.to_datetime(typed[column], utc=True)
    for column in FOLD_JSON_COLUMNS:
        if column in typed.columns:
            typed[column] = typed[column].map(json.loads)
    return typed


def _lockbox_guard(record: dict, tables: dict[str, pd.DataFrame], folds: pd.DataFrame):
    try:
        lockbox = record["spec"]["lockbox"]
        evidence = record["lockbox_evidence"]
        recorded = (lockbox["holdout_start"], evidence["holdout_start"])
        modes = (lockbox["mode"], evidence["mode"])
        values_used = evidence["holdout_values_used"]
    except KeyError as missing:
        raise RunRefused("missing_lockbox_metadata", f"record lacks {missing}.") from None
    if any(mode != "development" for mode in modes) or (folds["mode"] != "development").any():
        raise RunRefused(
            "holdout_run_locked",
            "Only development runs are displayed; holdout results stay locked until T6.",
        )
    if values_used is not False:
        raise RunRefused(
            "holdout_values_used", "The record does not state holdout_values_used=False."
        )
    holdout = min(
        _timestamp(recorded[0], "spec.lockbox.holdout_start"),
        _timestamp(recorded[1], "lockbox_evidence.holdout_start"),
        MODEL_HOLDOUT_START,
    )
    for name in ("inputs", "predictions", "curves"):
        table = tables[name]
        for column in table.columns:
            dtype = table[column].dtype
            is_date = pd.api.types.is_datetime64_any_dtype(dtype) or column in DATE_COLUMN_NAMES
            if is_date:
                # Coerce naive or text dates to UTC (naive taken as UTC) so the
                # scan can never fail open on a loosely typed column.
                values = pd.to_datetime(table[column], utc=True, errors="raise", format="ISO8601")
                late = values.dropna() >= holdout
                if late.any():
                    raise RunRefused(
                        "lockbox_dates_present",
                        f"{name}.{column} has {int(late.sum())} values on or after {holdout.date()}.",
                    )
    if (folds["n_test_label_in_holdout"] != 0).any():
        raise RunRefused(
            "lockbox_dates_present", "A fold withholds labels maturing in the lockbox."
        )
    for column in FOLD_DATE_COLUMNS:
        values = folds[column].dropna() if column in folds.columns else pd.Series(dtype=object)
        too_late = values > holdout if column in FOLD_BOUND_COLUMNS else values >= holdout
        if too_late.any():
            raise RunRefused("lockbox_dates_present", f"folds.{column} reaches the lockbox.")
    return holdout


def decision_method(checks: pd.DataFrame) -> str | None:
    """The inference method the stored T2 checks decide on (from their metric names)."""

    found = set()
    for metric in checks["metric"].dropna():
        match = re.search(r"_t_(fold_block|newey_west|bootstrap)$", str(metric))
        if match:
            found.add(DECISION_METHODS[match.group(1)])
    return found.pop() if len(found) == 1 else None


def _models(record: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "name": model["name"],
            "role": model["role"],
            "output_kind": model["output_kind"],
            "estimator": model["estimator"].rsplit(".", 1)[-1],
            "params": json.dumps(model["params"], sort_keys=True),
            "feature_columns": ", ".join(model["feature_columns"]),
            "variant_id": model["variant_id"],
            "note": model.get("note", ""),
        }
        for model in record["spec"]["models"]
    ]
    known = {row["name"] for row in rows}
    for name in predictions.loc[predictions["role"] == UNSAFE_ROLE, "model"].unique():
        if name not in known:
            rows.append(
                {
                    "name": name,
                    "role": UNSAFE_ROLE,
                    "output_kind": "not recorded",
                    "estimator": "(memorizer without purging; outside the harness)",
                    "params": "",
                    "feature_columns": "",
                    "variant_id": "",
                    "note": "positive control: leaky by construction, never a strategy",
                }
            )
    return pd.DataFrame(rows)


def load_run(path: str | Path) -> RunView:
    """Verify and load one run directory, or raise RunRefused with a reason."""

    path = Path(path)
    if not RUN_ID_PATTERN.match(path.name):
        raise RunRefused("not_a_run_directory", f"{path.name} is not a run id.")
    try:
        record, tables = read_run(path)
    except FileNotFoundError as error:
        raise RunRefused("incomplete_run", str(error)) from None
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        raise RunRefused("integrity_check_failed", f"{type(error).__name__}: {error}") from None
    try:
        folds = _typed_folds(tables["folds"])
        holdout = _lockbox_guard(record, tables, folds)
        method = decision_method(tables["checks"])
        models = _models(record, tables["predictions"])
    except RunRefused:
        raise
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise RunRefused("malformed_run", f"{type(error).__name__}: {error}") from None
    return RunView(
        run_id=path.name,
        path=path,
        record=record,
        tables=tables,
        folds=folds,
        holdout_start=holdout,
        decision_method=method,
        models=models,
    )


def ledger_summary(root: str | Path, spec_sha256: str | None) -> dict[str, object]:
    """Run attempts recorded in the ledger (read live; never part of the cached run)."""

    try:
        entries = read_ledger(root)
    except (OSError, ValueError) as error:
        return {"status": "unreadable", "reason": f"{type(error).__name__}: {error}"}
    variants = {variant for entry in entries for variant in entry.get("variants") or []}
    return {
        "status": "ok",
        "attempts": len(entries),
        "failed": sum(1 for entry in entries if entry.get("status") == "failed"),
        "same_spec": sum(1 for entry in entries if entry.get("spec_sha256") == spec_sha256),
        "distinct_variants": len(variants),
    }
