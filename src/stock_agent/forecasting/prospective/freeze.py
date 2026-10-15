"""
Freeze and register the prospective forecasters (one call, before the first
forecast).

    freeze(config, research_store_root, output_root, provenance, started_at)

1. Reads the F1 research store (training.load_training_data): sessions up
   to 2024-12-31 only.
2. Develops: nested purged out-of-fold predictions, bands and metrics
   (training.develop).
3. Refits each (asset, horizon, target) once on every training row
   (training.final_fit) and computes its band quantiles from every
   development score.
4. Writes, once each, the experiment directory (record.json, config.json and
   the development tables) and the frozen bundle (frozen.py).
5. Registers one registry version per (target, horizon): the ridge fits of
   all assets, with the experiment run and the bundle as artifacts, the data
   fingerprints, the code state and the config hash. The registry computes
   the stage from evidence; here it is at most `development_evaluated` (the
   higher stages' rules are unapproved), and with synthetic or sandbox data
   it is `engineering_only`. The entry's config carries the status
   "prospective record pending".

Outputs under one root (default reports/ml/prospective, ignored by Git):

    experiments/<run_id>/    record.json, config.json, predictions, bands,
                             metrics, checks, selection (parquet)
    frozen/<bundle_id>/      models.json, config.json, manifest.json
    registry/                the registry of these entries
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_RETURN_COLUMN
from stock_agent.forecasting import etf_data
from stock_agent.forecasting import metrics as em
from stock_agent.forecasting.experiment import write_directory
from stock_agent.forecasting.prospective import bands, frozen, training
from stock_agent.forecasting.prospective.config import TARGET_SHORT, TRACK, ProspectiveConfig
from stock_agent.model_diagnostics import linear
from stock_agent.model_diagnostics import runner as t2
from stock_agent.model_diagnostics.record import canonical_json, run_id, sha256_json
from stock_agent.model_diagnostics.seeds import SEED_RULE
from stock_agent.model_validation.audit import LeakageError, table_sha256
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.registry import ArtifactRef, Registry, RegistryEntry, derive_evidence

EXPERIMENT_SCHEMA_VERSION = "1.0"
EXPERIMENTS_DIRECTORY = "experiments"
FROZEN_DIRECTORY = "frozen"
REGISTRY_DIRECTORY = "registry"
DEFAULT_OUTPUT = Path("reports") / "ml" / "prospective"


@dataclass(frozen=True, eq=False)
class FreezeResult:
    run_id: str
    record: Mapping[str, object]
    experiment_directory: Path
    bundle: frozen.Bundle
    entries: tuple[RegistryEntry, ...]
    development: training.Development


def _json_bytes(value: object) -> bytes:
    return (json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True) + "\n").encode()


def model_key(target: str, horizon: int) -> str:
    return f"{TARGET_SHORT[target]}.h{horizon}.{training.CANDIDATE}"


def research_scope(config: ProspectiveConfig, data_class: str, target: str, horizon: int) -> str:
    return f"{TRACK}:{data_class}:{target}:h{horizon}:{config.universe_id}"


def _pick(metrics: pd.DataFrame, asset, horizon, target, metric, method):
    rows = metrics[
        (metrics["asset"] == asset)
        & (metrics["horizon"] == horizon)
        & (metrics["target"] == target)
        & (metrics["metric"] == metric)
        & (metrics["inference_method"] == method)
    ]
    return None if rows.empty or rows.iloc[0]["status"] != "ok" else rows.iloc[0]


def _estimate(row) -> float | None:
    return None if row is None else float(row["estimate"])


def _t(row) -> float | None:
    if row is None or not row["se"] > 0:
        return None
    return float((row["estimate"] - row["null_value"]) / row["se"])


def development_summary(dev: training.Development, config, asset, horizon, target) -> dict:
    """The development facts one fit carries (fold-block estimates; the bootstrap R^2)."""

    level = round(config.band.level * 100)
    metrics = dev.metrics
    fold = em.DECISION_METHOD
    summary = {
        "oos_r2_vs_historical_mean": _estimate(
            _pick(
                metrics,
                asset,
                horizon,
                target,
                "oos_r2_vs_historical_mean",
                "circular_block_bootstrap",
            )
        ),
        "mse_improvement_vs_historical_mean_t": _t(
            _pick(metrics, asset, horizon, target, "mse_improvement_vs_historical_mean", fold)
        ),
        f"band_coverage_{level}": _estimate(
            _pick(metrics, asset, horizon, target, f"band_coverage_{level}", fold)
        ),
    }
    if target == "log_realized_volatility":
        summary["qlike_improvement_vs_historical_mean_t"] = _t(
            _pick(metrics, asset, horizon, target, "qlike_improvement_vs_historical_mean", fold)
        )
    else:
        summary[f"plugin_coverage_{level}"] = _estimate(
            _pick(metrics, asset, horizon, target, f"plugin_coverage_{level}", fold)
        )
    banded = dev.bands.loc[
        (dev.bands["symbol"] == asset)
        & (dev.bands["horizon"] == horizon)
        & (dev.bands["target"] == target)
        & dev.bands["band_low"].notna()
    ]
    summary["band_rows_scored"] = len(banded)
    summary["band_folds_scored"] = int(banded["fold_id"].nunique())
    summary["band_share_inside"] = float(bands.inside(banded).mean()) if len(banded) else None
    return summary


def _fit_record(data, dev, config, asset, horizon, target, checks: list[dict]) -> dict:
    frame = training.target_frame(data.frames[(asset, horizon)], target)
    estimator, point, chosen, bounds, outer = training.final_fit(frame, config, target, horizon)
    scope = f"freeze:{target}:h{horizon}:{asset}"
    ends = pd.to_datetime(frame["target_end_date"], utc=True).iloc[outer.train_rows]
    checks.append(
        t2._check(
            "final_fit_labels_before_lockbox",
            scope,
            "leakage",
            "train_labels_in_sealed_period",
            int((ends >= MODEL_HOLDOUT_START).sum()),
            0,
            "==",
        )
    )
    checks.append(
        t2._check(
            "final_selection_inside_training",
            scope,
            "leakage",
            "inner_label_end_before_lockbox",
            int(pd.Timestamp(bounds["max_inner_label_end"]) < MODEL_HOLDOUT_START),
            1,
            "==",
        )
    )
    checks.append(
        t2._check(
            "final_fit_converged",
            scope,
            "instrument",
            "fits_not_converged",
            int(linear.has_converged(estimator) is False),
            0,
            "==",
        )
    )
    y = frame[TARGET_RETURN_COLUMN].to_numpy(np.float64)[outer.train_rows]
    rows = dev.bands.loc[
        (dev.bands["symbol"] == asset)
        & (dev.bands["horizon"] == horizon)
        & (dev.bands["target"] == target)
    ]
    if (pd.to_datetime(rows["target_end_date"], utc=True) >= MODEL_HOLDOUT_START).any():
        raise LeakageError(f"{scope}: a calibration score's label ends in the sealed period.")
    quantiles = bands.conformal_quantiles(
        bands.scores(rows), config.band.lower_quantile, config.band.upper_quantile
    )
    if quantiles is None or len(rows) < config.band.min_calibration:
        raise ValueError(
            f"{asset} h{horizon} {target}: {len(rows)} development scores cannot calibrate the band "
            f"(at least {config.band.min_calibration} are registered)."
        )
    train = frame.iloc[outer.train_rows]
    return {
        "asset": asset,
        "horizon": horizon,
        "target": target,
        "model": frozen.ridge_json(estimator, training.FEATURES),
        "selection": {
            "chosen": point,
            "grid": [
                p["penalty"]
                for p in training.ridge_spec(config, target, horizon).params["selection"]["grid"]
            ],
            "inner_scores": list(chosen.scores),
            "inner_dates": list(chosen.n_dates),
            "inner_test_start": pd.Timestamp(bounds["inner_test_start"]).isoformat(),
            "max_inner_label_end": pd.Timestamp(bounds["max_inner_label_end"]).isoformat(),
        },
        "null": {"model": training.NULL, "value": float(np.mean(y)), "n_train": len(y)},
        "band": {
            "method": bands.METHOD,
            "level": config.band.level,
            "lower_quantile": config.band.lower_quantile,
            "upper_quantile": config.band.upper_quantile,
            "lower": quantiles[0],
            "upper": quantiles[1],
            "n_calibration": len(rows),
            "scale": (
                "exp(volatility forecast) * sqrt(horizon)"
                if target == "total_log_return"
                else "1 (log volatility)"
            ),
        },
        "training": {
            "n_rows": len(train),
            "first_date": train["date"].min().isoformat(),
            "last_date": train["date"].max().isoformat(),
            "max_label_end": pd.Timestamp(ends.max()).isoformat(),
        },
        "development": {
            **development_summary(dev, config, asset, horizon, target),
            "zero_volatility_rows_dropped": data.zero_volatility_rows.get((asset, horizon), 0),
        },
    }


def freeze(
    config: ProspectiveConfig,
    *,
    research_store_root: str | Path,
    output_root: str | Path,
    provenance: Mapping[str, object],
    started_at: datetime,
    clock: Callable[[], datetime] | None = None,
) -> FreezeResult:
    """Develop, refit, write and register (see the module docstring)."""

    for key in ("code", "environment"):
        if key not in provenance:
            raise ValueError(f"provenance must include {key!r}.")
    data = training.load_training_data(research_store_root, config)
    dev = training.develop(data, config)
    checks = dev.checks.to_dict("records")
    fits = [
        _fit_record(data, dev, config, asset, horizon, target, checks)
        for horizon in config.horizons
        for target in config.targets
        for asset in config.assets
    ]
    checks_table = pd.DataFrame(checks)
    root = Path(output_root)
    registry = Registry(root / REGISTRY_DIRECTORY, root, **({"clock": clock} if clock else {}))
    pairs = [(target, horizon) for horizon in config.horizons for target in config.targets]
    scopes = {p: research_scope(config, data.data_class, *p) for p in pairs}
    model_ids = {p: f"{config.registry_prefix}.{model_key(*p)}" for p in pairs}
    versions = {p: registry.next_version(model_ids[p]) for p in pairs}
    prior = em.prior_trials(registry, scopes.values())
    spec_record = {
        "config_sha256": config.sha256,
        "code": dict(provenance["code"]),
        "data": dict(data.fingerprints),
        "batch_ids": list(data.batch_ids),
    }
    identifier = run_id(sha256_json(spec_record), started_at)
    specs = {p: training.ridge_spec(config, p[0], p[1]) for p in pairs}
    tables = {
        "predictions": dev.predictions,
        "bands": dev.bands,
        "metrics": dev.metrics,
        "checks": checks_table,
        "selection": dev.selection,
    }
    tables_sha = {name: table_sha256(t.reset_index(drop=True)) for name, t in tables.items()}
    bundle_id = f"{config.registry_prefix}-{identifier}"
    document = {
        "bundle_schema_version": frozen.BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "run_id": identifier,
        "status": frozen.STATUS,
        "labels": list(config.labels),
        "config": config.to_json(),
        "config_sha256": config.sha256,
        "code": dict(provenance["code"]),
        "data_class": data.data_class,
        "training_data": {
            **data.fingerprints,
            "batch_ids": list(data.batch_ids),
            "max_session": data.max_session.isoformat(),
            "training_end": config.training_end.isoformat(),
        },
        "registry": {
            f"{t}:h{h}": {"model_id": model_ids[(t, h)], "version": versions[(t, h)].version}
            for t, h in pairs
        },
        "fits": fits,
    }
    families = t2.summarize_checks(checks_table)
    summary = {
        model_key(t, h): {
            f"{name}.{fit['asset']}": value
            for fit in fits
            if (fit["target"], fit["horizon"]) == (t, h)
            for name, value in fit["development"].items()
        }
        for t, h in pairs
    }
    results = {"tables_sha256": tables_sha, "fits_sha256": sha256_json(fits)}
    record = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": identifier,
        "experiment": config.experiment,
        "track": TRACK,
        "description": config.description,
        "started_at": pd.Timestamp(started_at).isoformat(),
        "config": config.to_json(),
        "config_sha256": config.sha256,
        "code": dict(provenance["code"]),
        "environment": dict(provenance["environment"]),
        "data": {
            "source": config.data.source,
            "data_class": data.data_class,
            "batch_ids": list(data.batch_ids),
            "fingerprints": dict(data.fingerprints),
            "max_session": data.max_session.isoformat(),
            "note": training.ENGINEERING_NOTE[data.data_class],
        },
        "universe": data.universe.describe(),
        "features": {
            "names": list(training.FEATURES),
            "contracts": [etf_data.CONTRACTS_BY_NAME[n].describe() for n in training.FEATURES],
            "inputs": "daily log total returns from approved F5.1 (portfolio.descriptive)",
        },
        "label": {
            "targets": {f"{t}:h{h}": training.label_id(t, h, config.entry_lag) for t, h in pairs},
            "entry_lag": config.entry_lag,
            "availability": "known after the exit close (target_end_date), as for every label",
        },
        "folds": {
            **asdict(config.folds),
            "per_asset": True,
            "n_folds": dict(dev.n_folds),
        },
        "seeds": {"evaluation_seed": config.evaluation.seed, "seed_rule": SEED_RULE},
        "lockbox_evidence": {
            "mode": "development",
            "holdout_start": MODEL_HOLDOUT_START.isoformat(),
            "max_date": dev.lockbox["max_date"],
            "max_target_end_date": dev.lockbox["max_target_end_date"],
            "holdout_values_used": False,
        },
        "checks_summary": families,
        "decision_method": t2.DECISION_METHOD,
        "nulls": {"models": [training.NULL], "permutation_draws": 0},
        "band": {**asdict(config.band), "description": bands.DESCRIPTION},
        "multiple_testing": {
            "candidates_in_run": len(pairs),
            "grid_points_in_run": sum(len(specs[p].params["selection"]["grid"]) for p in pairs),
            "prior_variants_in_scope": prior,
        },
        "models": {
            model_key(*p): {"role": "candidate", "variant_id": specs[p].describe()["variant_id"]}
            for p in pairs
        },
        "prospective": {
            "status": frozen.STATUS,
            "h_end": config.h_end.isoformat(),
            "labels": list(config.labels),
            "bundle_id": bundle_id,
        },
        "registry": {"research_scopes": {f"{t}:h{h}": s for (t, h), s in scopes.items()}},
        "summary": summary,
        "outputs": {**results, "results_sha256": sha256_json(results)},
    }
    experiment_directory = root / EXPERIMENTS_DIRECTORY / identifier
    writers = {
        "record.json": lambda path: path.write_bytes(_json_bytes(record)),
        "config.json": lambda path: path.write_bytes(_json_bytes(config.to_json())),
    }
    for name, table in tables.items():
        writers[f"{name}.parquet"] = lambda path, table=table: table.to_parquet(path, index=False)
    run_digest = write_directory(
        experiment_directory,
        writers,
        {"experiment_schema_version": EXPERIMENT_SCHEMA_VERSION, "run_id": identifier},
    )
    bundle_directory = root / FROZEN_DIRECTORY / bundle_id
    bundle_digest = write_directory(
        bundle_directory,
        {
            frozen.MODELS_FILE: lambda path: path.write_bytes(_json_bytes(document)),
            frozen.CONFIG_FILE: lambda path: path.write_bytes(_json_bytes(config.to_json())),
        },
        {"bundle_schema_version": frozen.BUNDLE_SCHEMA_VERSION, "bundle_id": bundle_id},
    )
    run_ref = ArtifactRef("experiment_run", f"{EXPERIMENTS_DIRECTORY}/{identifier}", run_digest)
    bundle_ref = ArtifactRef(frozen.BUNDLE_KIND, f"{FROZEN_DIRECTORY}/{bundle_id}", bundle_digest)
    stored = json.loads(canonical_json(record))
    entries = []
    for pair in pairs:
        ref = versions[pair]
        previous = registry.versions(ref.model_id)
        target, horizon = pair
        entries.append(
            registry.register(
                model_id=ref.model_id,
                research_scope=scopes[pair],
                config={
                    "model": specs[pair].describe(),
                    "target": target,
                    "horizon": horizon,
                    "assets": list(config.assets),
                    "per_asset": True,
                    "experiment": config.experiment,
                    "experiment_config_sha256": config.sha256,
                    "band": asdict(config.band),
                    "prospective": {
                        "status": frozen.STATUS,
                        "h_end": config.h_end.isoformat(),
                        "labels": list(config.labels),
                        "bundle_id": bundle_id,
                    },
                },
                training_data={
                    "data_class": data.data_class,
                    **dict(data.fingerprints),
                    "batch_ids": list(data.batch_ids),
                    "fold_table_sha256": sha256_json(record["folds"]),
                    "labeled_frame_sha256": data.fingerprints["prices_sha256"],
                },
                metrics=summary[model_key(*pair)],
                artifacts=[run_ref, bundle_ref],
                evidence=derive_evidence(stored, model_key(*pair)),
                code=dict(provenance["code"]),
                parent=previous[-1] if previous else None,
                expected_version=ref.version,
            )
        )
    bundle = frozen.read_bundle(bundle_directory, bundle_digest)
    return FreezeResult(identifier, record, experiment_directory, bundle, tuple(entries), dev)
