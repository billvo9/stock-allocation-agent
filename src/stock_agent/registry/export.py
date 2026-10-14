"""
The registry's health export (artifact "model_registry", agreed input format
v1.0 of the Model Health Center; docs/research/ml_registry.md, "Health
export").

The registry stays the source of truth: the export is a derived, write-once
snapshot built only through the read API, so every stage in it is computed
from re-verified evidence at export time. It is written to
`<health_inputs>/model_registry/<run_id>.json` (locally
reports/health_inputs/, ignored by Git) as strict JSON (no NaN or Infinity).

Envelope: artifact, schema_version "1.0", generated_at (UTC, microseconds,
"Z"), code_sha (40-hex or null), synthetic (true only when every exported
entry is synthetic), run_id (the file name).

    models               the latest version of every model id in every
                         research scope, plus every scope's active version: model_id, version, stage,
                         stage_basis, registered_at (extra: research_scope,
                         data_class, next_stage, blockers)
    selection_events     every activation and rollback: event_id, decided_at,
                         candidates (versions registered in the scope by
                         then), selected (extra: event, research_scope,
                         previous, reason)
    experiments          variants_tried (registered versions: every
                         registration is a trial), runs (distinct experiment
                         runs that registered something) (extra:
                         distinct_variants; ledger_attempts and failed_runs
                         from the output root's ledger, null without one).
                         Every count covers one output root (registry and
                         ledger), not the whole machine
    forecast_evaluation  one row per forecast artifact of an exported model,
                         and per measured interval level for interval
                         artifacts: model_id, version, horizon, forecast_kind,
                         interval_level, observed_coverage, n,
                         calibration_status, calibration_detail

stage_basis lists the evidence records the computed stage rests on:
"entry:<ref>:<entry_sha256[:16]>" (the hash-verified entry), then
"artifact:<kind>:<sha256[:16]>" for every artifact that verified, then
"run:<run_id>" when the evidence re-derives from that run's verified record.

Calibration status (producer-owned monitoring status, not an evidence rule
and not used by any stage): for an interval level measured in the run, with
observed the fold-block coverage estimate (the mean of the folds' coverage
shares), gap = |observed - nominal| (compared with a 1e-9 tolerance) and p
the fold-block two-sided p-value of "coverage = nominal" (same estimand),
    fail     p < 0.01 and gap > 0.05
    warn     p < 0.05 or gap > 0.05
    pass     otherwise
    not_run  no test (p unavailable, e.g. fewer than 3 folds) and gap <= 0.05
The detail names the worst asset when a run reports per-asset coverage.
Point forecasts carry null interval fields and "not_run" ("not applicable:
point forecast"); an interval whose run metrics cannot be read verified is
"not_run" with the reason.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.model_diagnostics.record import canonical_json, sha256_json
from stock_agent.registry.entries import EntryRef, RegistryEntry, RegistryError
from stock_agent.registry.stages import STAGES
from stock_agent.registry.store import (
    EXPERIMENT_RUN_KIND,
    RegistryReader,
    verify_manifest_directory,
)

ARTIFACT = "model_registry"
SCHEMA_VERSION = "1.0"
FORECAST_KINDS = ("point", "interval", "probabilistic")
CALIBRATION_STATUSES = ("pass", "warn", "fail", "not_run")
FORECAST_ARTIFACT_KIND = "forecast_artifact"
POOLED = "__pooled__"
LEDGER_FILE = "ledger.jsonl"  # model_diagnostics.artifacts.LEDGER_FILE
GAP_WARN = 0.05
GAP_TOLERANCE = 1e-9  # 0.75 vs 0.80 is a gap of exactly 0.05, not more
P_WARN, P_FAIL = 0.05, 0.01
POINT_DETAIL = "not applicable: point forecast"
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_SHA = re.compile(r"[0-9a-f]{40}")
_LEVEL_METRIC = re.compile(r"interval_coverage_(\d+)")


def utc_stamp(value: object) -> str:
    """A timezone-aware time as YYYY-MM-DDTHH:MM:SS.ffffffZ (UTC)."""

    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RegistryError("Export timestamps must be timezone-aware.")
    return stamp.tz_convert(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stage_basis(reader: RegistryReader, entry: RegistryEntry) -> list[str]:
    basis = [f"entry:{entry.ref}:{entry.entry_sha256[:16]}"]
    for artifact in entry.artifacts:
        single = RegistryEntry(**{**entry.__dict__, "artifacts": (artifact,)})
        if reader.artifacts_verified(single):
            basis.append(f"artifact:{artifact.kind}:{artifact.sha256[:16]}")
    rederived = reader.rederived_evidence(entry)
    if rederived is not None and rederived == dict(entry.evidence):
        basis.append(f"run:{entry.evidence['run_id']}")
    return basis


def _exported_refs(reader: RegistryReader) -> list[EntryRef]:
    """The latest version of each model id in each research scope, plus active versions."""

    latest: dict[tuple[str, str], EntryRef] = {}
    for entry in reader.entries():
        key = (entry.model_id, entry.research_scope)
        latest[key] = max(latest.get(key, entry.ref), entry.ref)
    chosen = set(latest.values())
    for scope in {scope for _, scope in latest}:
        active = reader.active(scope)
        if active is not None:
            chosen.add(active)
    return sorted(chosen)


def _artifact_metadata(reader: RegistryReader, artifact) -> tuple[dict, int] | None:
    """A verified forecast artifact's metadata and row count, or None."""

    directory = reader.artifacts_root / artifact.path
    if not directory.is_dir() or not verify_manifest_directory(directory, artifact.sha256):
        return None
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = manifest["files"].get("forecasts.parquet", {}).get("rows")
    return metadata, int(rows) if isinstance(rows, int) else 0


def _run_metrics(reader: RegistryReader, entry: RegistryEntry) -> pd.DataFrame | None:
    runs = [a for a in entry.artifacts if a.kind == EXPERIMENT_RUN_KIND]
    if len(runs) != 1:
        return None
    directory = reader.artifacts_root / runs[0].path
    if not directory.is_dir() or not verify_manifest_directory(directory, runs[0].sha256):
        return None
    path = directory / "metrics.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _ledger_attempts(reader: RegistryReader) -> list[dict] | None:
    """
    Every recorded run attempt (completed or failed) in the artifacts root's
    ledger, or None when there is no ledger. Counts are per output root.
    """

    path = reader.artifacts_root / LEDGER_FILE
    if not path.is_file() or path.is_symlink():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def calibration_status(observed: float, nominal: float, p_value: float) -> str:
    """
    The producer's calibration status for one interval level. `observed` and
    `p_value` come from the same estimand (the run's fold-block row). Without
    a test (p is NaN, e.g. fewer than 3 folds) a gap beyond the band can
    still warn, but nothing can pass.
    """

    gap = abs(observed - nominal)
    wide = gap > GAP_WARN + GAP_TOLERANCE
    if not math.isfinite(p_value):
        return "warn" if wide else "not_run"
    if p_value < P_FAIL and wide:
        return "fail"
    if p_value < P_WARN or wide:
        return "warn"
    return "pass"


def _fold_block(rows: pd.DataFrame, metric: str):
    own = rows[(rows["metric"] == metric) & (rows["inference_method"] == "fold_block_t")]
    return None if own.empty else own.iloc[0]


def _worst_asset(metrics: pd.DataFrame, name: str, target: str, metric: str, nominal: float):
    """The asset whose fold-block coverage is furthest from nominal (per-asset tracks)."""

    if "asset" not in metrics.columns:
        return None
    rows = metrics[
        (metrics["model"] == name)
        & (metrics["target"] == target)
        & (metrics["asset"] != POOLED)
        & (metrics["metric"] == metric)
        & (metrics["inference_method"] == "fold_block_t")
    ]
    rows = rows[rows["estimate"].notna()]
    if rows.empty:
        return None
    return rows.loc[(rows["estimate"] - nominal).abs().idxmax()]


def _interval_rows(entry, metadata, n, metrics) -> list[dict]:
    base = {
        "model_id": entry.model_id,
        "version": entry.version,
        "horizon": int(metadata["horizon"]),
        "forecast_kind": "interval",
        "n": n,
    }
    if metrics is None:
        return [
            {
                **base,
                "interval_level": None,
                "observed_coverage": None,
                "calibration_status": "not_run",
                "calibration_detail": "the run's metrics could not be read verified",
            }
        ]
    name = entry.config.get("model", {}).get("name")
    target = metadata["target"]
    rows = metrics[(metrics["model"] == name) & metrics["metric"].str.match(_LEVEL_METRIC)]
    if "target" in rows.columns:
        rows = rows[(rows["target"] == target) & (rows["asset"] == POOLED)]
    result = []
    for metric in sorted(set(rows["metric"])):
        nominal = int(_LEVEL_METRIC.fullmatch(metric).group(1)) / 100.0
        test = _fold_block(rows, metric)
        if test is None:
            continue
        # One estimand for the coverage and its test: the fold-block mean of the
        # folds' coverage shares, and the fold-block t of that mean.
        observed = float(test["estimate"])
        p_value = float(test["p_value"])
        if not math.isfinite(observed):
            result.append(
                {
                    **base,
                    "interval_level": nominal,
                    "observed_coverage": None,
                    "calibration_status": "not_run",
                    "calibration_detail": f"coverage unavailable ({test['reason']})",
                }
            )
            continue
        status = calibration_status(observed, nominal, p_value)
        if math.isfinite(p_value) and test["se"] > 0:
            tested = (
                f"fold-block t {(observed - nominal) / test['se']:.2f}, p {p_value:.3f}, "
                f"df {test['df']:.0f}"
            )
        else:
            tested = f"no fold-block test ({test['reason']})"
        detail = (
            f"fold-block coverage {observed:.3f} vs nominal {nominal:.2f} "
            f"({int(test['n'])} dates, pooled over assets); {tested}"
        )
        worst = _worst_asset(metrics, name, target, metric, nominal)
        if worst is not None:
            worst_p = f", p {worst['p_value']:.3f}" if math.isfinite(worst["p_value"]) else ""
            detail += f"; worst asset {worst['asset']} {worst['estimate']:.3f}{worst_p}"
        detail += (
            f"; rule: fail if p < {P_FAIL} and gap > {GAP_WARN}, warn if p < {P_WARN} or "
            f"gap > {GAP_WARN}, not_run without a test unless the gap exceeds {GAP_WARN}"
        )
        result.append(
            {
                **base,
                "interval_level": nominal,
                "observed_coverage": round(observed, 6),
                "calibration_status": status,
                "calibration_detail": detail,
            }
        )
    return result or [
        {
            **base,
            "interval_level": None,
            "observed_coverage": None,
            "calibration_status": "not_run",
            "calibration_detail": "the run measured no interval coverage for this model",
        }
    ]


def _forecast_rows(reader: RegistryReader, entry: RegistryEntry) -> list[dict]:
    rows = []
    metrics = None
    for artifact in entry.artifacts:
        if artifact.kind != FORECAST_ARTIFACT_KIND:
            continue
        loaded = _artifact_metadata(reader, artifact)
        if loaded is None:
            continue  # an unverified artifact is reported by the registry checks
        metadata, n = loaded
        if metadata["forecast_kind"] == "point":
            rows.append(
                {
                    "model_id": entry.model_id,
                    "version": entry.version,
                    "horizon": int(metadata["horizon"]),
                    "forecast_kind": "point",
                    "interval_level": None,
                    "observed_coverage": None,
                    "n": n,
                    "calibration_status": "not_run",
                    "calibration_detail": POINT_DETAIL,
                }
            )
            continue
        if metrics is None:
            metrics = _run_metrics(reader, entry)
        rows += _interval_rows(entry, metadata, n, metrics)
    return rows


def build_export(
    reader: RegistryReader,
    *,
    generated_at: datetime,
    code_sha: str | None,
) -> dict[str, object]:
    """The export document (validated); nothing is written."""

    if code_sha is not None and not _SHA.fullmatch(str(code_sha)):
        raise RegistryError(f"code_sha {code_sha!r} is not a 40-hex Git SHA.")
    entries = reader.entries()
    models, forecast_evaluation = [], []
    for ref in _exported_refs(reader):
        entry = reader.entry(ref)
        assessment = reader.assessment(ref)
        models.append(
            {
                "model_id": entry.model_id,
                "version": entry.version,
                "stage": assessment.stage,
                "stage_basis": _stage_basis(reader, entry),
                "registered_at": utc_stamp(entry.created_at),
                "research_scope": entry.research_scope,
                "data_class": entry.data_class,
                "next_stage": assessment.next_stage,
                "blockers": sorted({c.reason for c in assessment.blockers if c.reason}),
            }
        )
        forecast_evaluation += _forecast_rows(reader, entry)
    created = {entry.ref: pd.Timestamp(entry.created_at) for entry in entries}
    selection_events = []
    for event in reader.events():
        decided = pd.Timestamp(event.at)
        candidates = sum(
            1
            for entry in entries
            if entry.research_scope == event.research_scope and created[entry.ref] <= decided
        )
        selection_events.append(
            {
                "event_id": f"{event.research_scope}#{event.sequence}:{event.event_sha256[:12]}",
                "decided_at": utc_stamp(decided),
                "candidates": candidates,
                "selected": str(event.ref),
                "event": event.event,
                "research_scope": event.research_scope,
                "previous": None if event.previous is None else str(event.previous),
                "reason": event.reason,
            }
        )
    runs = {entry.evidence.get("run_id") for entry in entries} - {None}
    variants = {entry.evidence.get("variant_id") for entry in entries} - {None}
    attempts = _ledger_attempts(reader)
    content = {
        "models": models,
        "selection_events": selection_events,
        "experiments": {
            "variants_tried": len(entries),
            "runs": len(runs),
            "distinct_variants": len(variants),
            "ledger_attempts": None if attempts is None else len(attempts),
            "failed_runs": None
            if attempts is None
            else sum(line.get("status") == "failed" for line in attempts),
        },
        "forecast_evaluation": forecast_evaluation,
    }
    stamp = pd.Timestamp(generated_at)
    document = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_stamp(stamp),
        "code_sha": code_sha,
        "synthetic": all(entry.data_class == "synthetic" for entry in entries),
        "run_id": f"{stamp.tz_convert(UTC):%Y%m%dT%H%M%S%fZ}-{sha256_json(content)[:12]}",
        **content,
    }
    document = json.loads(canonical_json(document))  # non-finite numbers -> null
    validate_export(document)
    return document


def _require(row: Mapping, key: str, check, what: str) -> None:
    if key not in row or not check(row[key]):
        raise RegistryError(f"Export field {key!r} must be {what}; got {row.get(key)!r}.")


def _int(minimum: int):
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= minimum


def _fraction_or_none(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_export(document: Mapping[str, object]) -> None:
    """Refuse a document that the agreed v1.0 format (or its readers) would refuse."""

    _require(document, "artifact", lambda v: v == ARTIFACT, f"{ARTIFACT!r}")
    _require(document, "schema_version", lambda v: v == SCHEMA_VERSION, f"{SCHEMA_VERSION!r}")
    _require(
        document,
        "generated_at",
        lambda v: isinstance(v, str) and _TIMESTAMP.fullmatch(v),
        "a UTC timestamp with microseconds and Z",
    )
    _require(
        document,
        "code_sha",
        lambda v: v is None or (isinstance(v, str) and _SHA.fullmatch(v)),
        "a 40-hex SHA or null",
    )
    _require(document, "synthetic", lambda v: isinstance(v, bool), "a boolean")
    _require(document, "run_id", _text, "text")
    for key in ("models", "selection_events", "forecast_evaluation"):
        _require(document, key, lambda v: isinstance(v, list), "a list")
    _require(document, "experiments", lambda v: isinstance(v, dict), "an object")
    for model in document["models"]:
        _require(model, "model_id", _text, "text")
        _require(model, "version", _int(1), "an integer >= 1")
        _require(model, "stage", lambda v: v in STAGES, f"one of {STAGES}")
        _require(
            model,
            "stage_basis",
            lambda v: isinstance(v, list) and all(_text(x) for x in v),
            "a list of evidence ids",
        )
        _require(
            model,
            "registered_at",
            lambda v: isinstance(v, str) and _TIMESTAMP.fullmatch(v),
            "a UTC timestamp",
        )
    for event in document["selection_events"]:
        _require(event, "event_id", _text, "text")
        _require(
            event,
            "decided_at",
            lambda v: isinstance(v, str) and _TIMESTAMP.fullmatch(v),
            "a UTC timestamp",
        )
        _require(event, "candidates", _int(0), "an integer >= 0")
        _require(event, "selected", lambda v: v is None or _text(v), "text or null")
    for key in ("variants_tried", "runs"):
        _require(document["experiments"], key, _int(0), "an integer >= 0")
    for row in document["forecast_evaluation"]:
        _require(row, "model_id", _text, "text")
        _require(row, "version", _int(1), "an integer >= 1")
        _require(row, "horizon", _int(1), "an integer >= 1")
        _require(row, "forecast_kind", lambda v: v in FORECAST_KINDS, f"one of {FORECAST_KINDS}")
        _require(row, "interval_level", _fraction_or_none, "a fraction or null")
        _require(row, "observed_coverage", _fraction_or_none, "a fraction or null")
        _require(row, "n", _int(0), "an integer >= 0")
        _require(
            row,
            "calibration_status",
            lambda v: v in CALIBRATION_STATUSES,
            f"one of {CALIBRATION_STATUSES}",
        )
        _require(row, "calibration_detail", lambda v: v is None or _text(v), "text or null")
        point = row["forecast_kind"] == "point"
        if point and (
            row["interval_level"] is not None
            or row["observed_coverage"] is not None
            or row["calibration_status"] != "not_run"
        ):
            raise RegistryError("A point forecast carries null interval fields and not_run.")
    json.dumps(document, allow_nan=False)  # strict JSON


def write_export(document: Mapping[str, object], health_inputs: str | Path) -> Path:
    """Write `<health_inputs>/model_registry/<run_id>.json` once (never overwritten)."""

    validate_export(document)
    directory = Path(health_inputs) / ARTIFACT
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document['run_id']}.json"
    text = json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path
