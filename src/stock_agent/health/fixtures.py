"""
Small synthetic examples of every producer input the health readers accept:
a training-health file, a backtest summary, a registry export and a Stock
Lab monitor file in the agreed formats v1.0 (written by `write_input` where producers write them),
and a model registry built through the ML track's own write API
(`stock_agent.registry.Registry`). They are the worked examples in
docs/operations/model_health.md, the inputs of the `smoke.input_readers`
check and test fixtures. Every value is invented and marked synthetic; none
is a measurement.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stock_agent.model_validation.audit import file_sha256
from stock_agent.registry import ArtifactRef, Registry, derive_evidence

GENERATED_AT = "2024-12-31T22:00:00.000000Z"
_ENVELOPE = {"schema_version": "1.0", "generated_at": GENERATED_AT, "code_sha": None}

_TRAINING = {
    "artifact": "rl_training_health",
    **_ENVELOPE,
    "synthetic": True,
    "run_id": "synthetic-ppo-run",
    "algorithm": "ppo",
    "status": "completed",
    "seeds": 4,
    "scenario_version": "synthetic-v1",
    "transition_status": "engineering_only_unapproved_transition",
    "checks": [
        {
            "check_id": "nan_or_inf",
            "status": "pass",
            "value": 0,
            "unit": "non-finite values",
            "threshold": "0",
            "detail": None,
        },
        {
            "check_id": "exploding_values",
            "status": "pass",
            "value": 0,
            "unit": "updates with exploding gradients",
            "threshold": "0",
            "detail": None,
        },
        {
            "check_id": "approx_kl",
            "status": "pass",
            "value": 0.012,
            "unit": "mean approximate KL per update",
            "threshold": "<= 0.05",
            "detail": None,
        },
        {
            "check_id": "clip_fraction",
            "status": "pass",
            "value": 0.11,
            "unit": "mean clip fraction",
            "threshold": "<= 0.3",
            "detail": None,
        },
        {
            "check_id": "entropy_collapse",
            "status": "pass",
            "value": 0.8,
            "unit": "final policy entropy",
            "threshold": ">= 0.1",
            "detail": None,
        },
        {
            "check_id": "value_loss",
            "status": "pass",
            "value": 0.04,
            "unit": "final value loss",
            "threshold": "finite and falling",
            "detail": None,
        },
        {
            "check_id": "reward_trend",
            "status": "pass",
            "value": 0.002,
            "unit": "mean episode-reward slope",
            "threshold": ">= 0",
            "detail": None,
        },
        {
            "check_id": "seed_spread",
            "status": "pass",
            "value": 0.15,
            "unit": "relative spread of final reward across seeds",
            "threshold": "<= 0.5",
            "detail": None,
        },
    ],
}

_BACKTEST = {
    "artifact": "backtest_summary",
    **_ENVELOPE,
    "synthetic": True,
    "run_id": "synthetic-backtest",
    "evidence_status": "rules_illustration",
    "policy": "equal_weight_monthly",
    "transition_status": "engineering_only_unapproved_transition",
    "constraints": {
        "max_turnover": 2.0,
        "max_cost_bps": 50.0,
        "max_weight": 0.4,
        "max_drawdown": 0.35,
    },
    "measures": {"turnover": 0.8, "costs_bps": 12.5, "max_weight": 0.34, "max_drawdown": 0.21},
    "units": {
        "turnover": "annualized one-way fraction of portfolio value",
        "costs_bps": "basis points of average portfolio value per year",
        "max_weight": "fraction of portfolio value",
        "max_drawdown": "fraction of peak value",
    },
}

_REGISTRY_EXPORT = {
    "artifact": "model_registry",
    **_ENVELOPE,
    "synthetic": True,
    "run_id": "synthetic-registry-export",
    "models": [
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "stage": "engineering_only",
            "stage_basis": ["20240628T210000Z-run0000002"],
            "registered_at": "2024-06-29T21:00:00.000000Z",
        }
    ],
    "selection_events": [],
    "experiments": {"variants_tried": 2, "runs": 2},
    "forecast_evaluation": [
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "horizon": 20,
            "forecast_kind": "interval",
            "interval_level": 0.9,
            "observed_coverage": 0.88,
            "n": 240,
            "calibration_status": "pass",
            "calibration_detail": None,
        },
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "horizon": 5,
            "forecast_kind": "point",
            "interval_level": None,
            "observed_coverage": None,
            "n": 240,
            "calibration_status": "not_run",
            "calibration_detail": None,
        },
    ],
}


_STOCK_LAB = {
    "artifact": "stock_lab_monitor",
    **_ENVELOPE,
    "synthetic": True,
    "run_id": "synthetic-stock-lab-day",
    "job": {
        "status": "ok",
        "started_at": "2024-12-31T21:10:00.000000Z",
        "finished_at": "2024-12-31T21:11:00.000000Z",
        "expected_session": "2024-12-31",
        "reason": None,
        "failed": [],
        "skipped": [],
    },
    "freshness": [
        {"ticker": "DEMO_CALM", "last_session": "2024-12-31", "sessions_behind": 0},
        {"ticker": "DEMO_SWING", "last_session": "2024-12-31", "sessions_behind": 0},
    ],
    "scores": {"n": 300, "rps": 0.205, "baseline_rps": 0.207, "brier": 0.61, "log_loss": 1.05},
    "inputs_drift": [{"feature": "ewma_vol", "psi": 0.08}, {"feature": "ret_20", "psi": 0.03}],
    "traps": {"resolved": 2, "failed": 1, "pending": 1},
    "ledger": {
        "entries": 450,
        "verified": True,
        "reason": None,
        "head": "c" * 64,
        "kept_head": {"state": "match", "kept": "b" * 64},
    },
    "lineage": {
        "estimator": "B1",
        "model_version": "synthetic",
        "feature_version": "stock_lab_features_v1",
        "evaluation_run_id": "S-synthetic",
    },
    "schedule": {"last_success_at": "2024-12-31T21:11:00.000000Z", "next_expected_at": None},
    "evenings": {
        "since": "2024-12-02",
        "due": 20,
        "recorded": 20,
        "missed": [],
        "recent": 10,
        "recent_missed": 0,
    },
    "symbols": {
        "tickers": [
            {"ticker": "DEMO_CALM", "status": "supported", "reason": "listed_us_stock"},
            {"ticker": "DEMO_SWING", "status": "supported", "reason": "listed_us_stock"},
        ]
    },
}


def stock_lab_monitor() -> dict[str, Any]:
    return copy.deepcopy(_STOCK_LAB)


def training_health() -> dict[str, Any]:
    return copy.deepcopy(_TRAINING)


def backtest_summary() -> dict[str, Any]:
    return copy.deepcopy(_BACKTEST)


def registry_export() -> dict[str, Any]:
    return copy.deepcopy(_REGISTRY_EXPORT)


def write_input(inputs: Path, document: dict[str, Any]) -> Path:
    """Write `document` where producers do: `<inputs>/<artifact>/<run_id>.json`."""

    path = inputs / document["artifact"] / f"{document['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --- a model registry, through the ML track's write API --------------------------

REGISTRY_SCOPE = "equity_xs:synthetic:h20"
REGISTRY_MODEL = "synthetic.ridge"


def synthetic_run_record(run_id: str, *, prior_variants: int) -> dict[str, Any]:
    """An experiment-run record in the runner's contract, on synthetic data."""

    return {
        "run_id": run_id,
        "data": {"data_class": "synthetic"},
        "lockbox_evidence": {
            "mode": "development",
            "holdout_start": "2025-01-01T00:00:00+00:00",
            "max_date": "2024-12-31T00:00:00+00:00",
            "max_target_end_date": "2024-12-30T00:00:00+00:00",
            "holdout_values_used": False,
        },
        "checks_summary": {
            "families": {
                "leakage": {"total": 5, "passed": 5, "failed": 0, "undetermined": 0},
                "instrument": {"total": 4, "passed": 4, "failed": 0, "undetermined": 0},
            }
        },
        "decision_method": "fold_block_t",
        "nulls": {"models": ["zero"], "permutation_draws": 199},
        "multiple_testing": {
            "candidates_in_run": 1,
            "grid_points_in_run": 4,
            "prior_variants_in_scope": prior_variants,
        },
        "universe": {"point_in_time": True, "survivorship_free": False},
        "models": {
            "ridge": {"role": "candidate", "variant_id": f"{run_id}-ridge"},
            "zero": {"role": "null", "variant_id": f"{run_id}-zero"},
        },
    }


def write_run_artifact(artifacts_root: Path, record: dict[str, Any]) -> ArtifactRef:
    directory = artifacts_root / "runs" / record["run_id"]
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    manifest = {"files": {"record.json": {"file_sha256": file_sha256(directory / "record.json")}}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ArtifactRef(
        "experiment_run", f"runs/{record['run_id']}", file_sha256(directory / "manifest.json")
    )


def build_registry(
    directory: Path, *, prior_variants: Sequence[int] = (0, 1), activate: bool = True
) -> Path:
    """
    A registry with one version per entry of `prior_variants` (each the run's
    recorded prior-variant count), the last one activated. Returns the registry
    root; its artifacts root is the root's parent, the registry's default.
    """

    artifacts_root = directory / "ml"
    times = iter(datetime(2024, 6, 28, 21, tzinfo=UTC) + timedelta(days=n) for n in range(100))
    registry = Registry(artifacts_root / "registry", artifacts_root, clock=lambda: next(times))
    parent = None
    for index, prior in enumerate(prior_variants, start=1):
        record = synthetic_run_record(f"20240628T210000Z-run{index:07d}", prior_variants=prior)
        entry = registry.register(
            model_id=REGISTRY_MODEL,
            research_scope=REGISTRY_SCOPE,
            config={"model": "ridge", "grid": [0.1, 1.0, 10.0, 100.0], "variant": index},
            training_data={
                "data_class": "synthetic",
                "labeled_frame_sha256": "a" * 64,
                "fold_table_sha256": "b" * 64,
            },
            metrics={"mean_rank_ic": 0.01 * index},
            artifacts=[write_run_artifact(artifacts_root, record)],
            evidence=derive_evidence(record, "ridge"),
            code={"git_sha": "c" * 40, "git_dirty": False},
            parent=parent,
        )
        parent = entry.ref
    if activate and parent is not None:
        registry.activate(parent, reason="synthetic fixture: reference version")
    return registry.root
