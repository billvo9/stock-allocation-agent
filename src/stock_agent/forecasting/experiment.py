"""
The experiment runner: one config, one reproducible and versioned experiment
(docs/research/ml_pipeline.md).

Equity cross-sectional track (synthetic):

    data       seeded synthetic panel (forecasting.synthetic)
    universe   frozen eligible universe (features.frozen_universe)
    features   contracted families, validated (features.contracts)
    labels     prefix-invariant labels over the eligible members
    folds      purged expanding walk-forward folds, development mode
    models     T2 nulls, controls and canary; T4 nested-selected linear
               candidates and the OLS diagnostic
    run        model_diagnostics.runner.run_diagnostics (unchanged), so
               every model gets T2's metrics, inference, nulls and checks
    extras     coverage, coverage-weighted rank IC, paired increments over a
               linear reference, Holm-adjusted p (forecasting.metrics) and
               missingness diagnostics

Outputs under one root (default reports/ml, ignored by Git):

    runs/<run_id>/          the T2 contract run (write_run)
    experiments/<run_id>/   record.json, config.json, metrics.parquet,
                            missingness.parquet, manifest.json
    forecasts/<id>/         one out-of-fold forecast artifact per candidate
    registry/               one new registry version per candidate
    ledger.jsonl            one line per attempt, failures included

The record states the code SHA and dirty flag, the config hash, data and
universe fingerprints, the seeds, the label spec and the fold geometry.
Results are deterministic: the same config and code give the same
`results_sha256` under a new run id. Every run registers its candidates, so
every variant tried is counted for multiple testing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.features.contracts import FeatureContract, validate_feature_contracts
from stock_agent.features.families import build_feature_frame
from stock_agent.features.frozen_universe import (
    FrozenUniverse,
    UniverseRule,
    assert_membership_independent,
    freeze_universe,
)
from stock_agent.features.invariant_labels import frozen_universe_labels
from stock_agent.features.missingness import missingness_table
from stock_agent.features.training import LabelSpec
from stock_agent.forecasting import artifacts as fa
from stock_agent.forecasting import metrics as em
from stock_agent.forecasting.config import ConfigError, ExperimentConfig
from stock_agent.forecasting.synthetic import (
    SyntheticEquityPanel,
    SyntheticEquitySpec,
    synthetic_equity_panel,
)
from stock_agent.model_diagnostics import nested, scoring
from stock_agent.model_diagnostics import runner as t2
from stock_agent.model_diagnostics.artifacts import append_ledger, write_run
from stock_agent.model_diagnostics.record import canonical_json, sha256_json
from stock_agent.model_diagnostics.seeds import SEED_RULE, derive_seed
from stock_agent.model_validation.audit import (
    file_sha256,
    frame_content_sha256,
    table_sha256,
)
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    WalkForwardFold,
    make_expanding_folds,
    make_test_windows,
)
from stock_agent.registry import ArtifactRef, Registry, RegistryEntry, derive_evidence

EXPERIMENT_SCHEMA_VERSION = "1.0"
EXPERIMENTS_DIRECTORY = "experiments"
FORECASTS_DIRECTORY = "forecasts"
REGISTRY_DIRECTORY = "registry"
SCORE_FEATURE = "rel_momentum_20d"  # the T2 unfitted control; sign +1 pre-registered
BENCHMARK_MEMBERS = (
    "C_t = the frozen universe's eligible members on t (features.invariant_labels); "
    "T4's availability and fail-closed rules otherwise unchanged"
)
SYNTHETIC_NOTE = (
    "Synthetic data with a planted signal: verifies the pipeline; never evidence about markets."
)


@dataclass(frozen=True, eq=False)
class EquityDataset:
    panel: SyntheticEquityPanel
    universe: FrozenUniverse
    labeled: pd.DataFrame
    features: tuple[str, ...]
    contracts: dict[str, FeatureContract]
    label_spec: LabelSpec
    folds: tuple[WalkForwardFold, ...]
    fold_params: dict[str, object]
    fingerprints: dict[str, str]


@dataclass(frozen=True, eq=False)
class ExperimentResult:
    run_id: str
    record: dict[str, object]
    directory: Path
    diagnostics: t2.DiagnosticsRun
    tables: dict[str, pd.DataFrame]
    entries: tuple[RegistryEntry, ...]


def target_name(label_spec: LabelSpec) -> str:
    return "forward_return" if label_spec.benchmark is None else "equal_weight_excess_return"


def research_scope(config: ExperimentConfig, label_spec: LabelSpec) -> str:
    return (
        f"{config.track}:{config.data_class}:{target_name(label_spec)}:"
        f"h{label_spec.horizon}:{config.universe.id}"
    )


# --- data, universe, features, labels, folds ---


def build_equity_dataset(config: ExperimentConfig) -> EquityDataset:
    try:
        spec = SyntheticEquitySpec(**dict(config.data.params))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"data.params: {error}") from None
    panel = synthetic_equity_panel(spec)
    universe = freeze_universe(
        panel.prices,
        UniverseRule(config.universe.min_history_sessions),
        universe_id=config.universe.id,
        survivorship_free=panel.survivorship_free,
    )
    frame, contracts = build_feature_frame(
        panel.prices, universe, config.features.families, fundamentals=panel.fundamentals
    )
    features = tuple(contracts)
    validate_feature_contracts(frame, features, contracts)
    label_spec = LabelSpec(
        horizon=config.label.horizon,
        entry_lag=config.label.entry_lag,
        price_column=universe.rule.price_column,
        benchmark=config.label.benchmark,
    )
    labels = frozen_universe_labels(panel.prices, universe, label_spec)
    labeled = frame.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
    labeled = labeled.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert_membership_independent(labeled, universe)

    sessions = labeled["date"].drop_duplicates().sort_values()
    last_labeled = labeled.loc[labeled["target_return"].notna(), "date"].max()
    start = pd.Timestamp(config.folds.start, tz="UTC")
    end = last_labeled + pd.Timedelta(days=1)
    windows = make_test_windows(
        sessions, start=start, end=end, block_sessions=config.folds.block_sessions
    )
    folds = make_expanding_folds(labeled, windows, holdout_start=MODEL_HOLDOUT_START)
    fold_params = {
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "block_sessions": config.folds.block_sessions,
        "train_start": None,
        "mode": "development",
        "holdout_start": MODEL_HOLDOUT_START.isoformat(),
        "purge": "train iff target_end_date < test_start (global); no embargo (forward folds)",
    }
    fundamentals = panel.fundamentals.sort_values(["symbol", "available_at", "period_end"])
    fingerprints = {
        "prices_sha256": frame_content_sha256(panel.prices, ["close", "volume"]),
        "fundamentals_sha256": table_sha256(fundamentals.reset_index(drop=True)),
        "universe_sha256": universe.sha256,
        "features_sha256": frame_content_sha256(frame, list(features)),
        "labeled_frame_sha256": frame_content_sha256(
            labeled, [*features, "target_return", "target_start_date", "target_end_date"]
        ),
    }
    return EquityDataset(
        panel=panel,
        universe=universe,
        labeled=labeled,
        features=features,
        contracts=dict(contracts),
        label_spec=label_spec,
        folds=folds,
        fold_params=fold_params,
        fingerprints=fingerprints,
    )


# --- models ---


def equity_models(config: ExperimentConfig, dataset: EquityDataset) -> list[t2.ModelSpec]:
    """T2's nulls, controls and canary; the configured T4 candidates and OLS."""

    controls = [
        spec
        for spec in t2.default_models(score_feature=SCORE_FEATURE)
        if spec.name != SCORE_FEATURE or SCORE_FEATURE in dataset.features
    ]
    design = nested.InnerDesign(
        block_sessions=config.folds.inner_block_sessions,
        n_windows=config.folds.inner_windows,
        min_train_sessions=config.folds.inner_min_train_sessions,
    )
    linear = nested.linear_models(
        dataset.label_spec, dataset.features, design, include_ols=config.models.include_ols
    )
    chosen = [s for s in linear if s.name in config.models.candidates or s.name == "ols"]
    return [*controls, *chosen]


def diagnostics_config(config: ExperimentConfig, models, label_spec) -> t2.DiagnosticsConfig:
    names = {spec.name for spec in models}
    unknown = sorted(set(config.evaluation.permutation_models) - names)
    if unknown:
        raise ConfigError(f"evaluation.permutation_models names unknown models: {unknown}")
    stale_models = (
        (SCORE_FEATURE,) if config.evaluation.stale_lags and SCORE_FEATURE in names else ()
    )
    return t2.DiagnosticsConfig(
        seed=config.evaluation.seed,
        bootstrap_reps=config.evaluation.bootstrap_reps,
        permutation_draws=config.evaluation.permutation_draws,
        permutation_models=config.evaluation.permutation_models,
        stale_models=stale_models,
        stale_lags=config.evaluation.stale_lags or t2.DiagnosticsConfig().stale_lags,
        calibration_reps=config.evaluation.calibration_reps,
        calibration_bootstrap_reps=config.evaluation.calibration_bootstrap_reps,
        **nested.comparison_settings(label_spec, with_linear=config.models.include_ols),
    )


# --- experiment metrics ---


def experiment_metrics(
    config: ExperimentConfig, models, run: t2.DiagnosticsRun, settings: t2.DiagnosticsConfig
) -> pd.DataFrame:
    predictions = run.tables["predictions"]
    by_model = {name: frame for name, frame in predictions.groupby("model", sort=False)}
    reference = by_model[models[0].name]
    ctx = scoring.build_context(
        reference,
        block_length=settings.block_length,
        bootstrap_reps=settings.bootstrap_reps,
        seed=derive_seed(settings.seed, "bootstrap"),
        hac_lag=settings.hac_lag,
        ci_level=settings.ci_level,
        min_names=settings.min_names,
    )
    rows: list[dict] = []
    candidates = {}
    ref_name = config.models.increment_reference
    for spec in models:
        scored = by_model[spec.name]
        rows += em.coverage_metrics(spec.name, spec.role, scored, ctx)
        if spec.role in ("candidate", "diagnostic") and spec.name != ref_name:
            rows += em.increment_metrics(
                spec.name, spec.role, scored, ref_name, by_model[ref_name], ctx
            )
        if spec.role == "candidate":
            candidates[spec.name] = next(
                row
                for row in rows
                if row["model"] == spec.name
                and row["metric"] == "coverage_weighted_rank_ic"
                and row["inference_method"] == em.DECISION_METHOD
            )
            rows += em.rows(
                spec.name,
                spec.role,
                "interval_calibration",
                [scoring.point(float("nan"), n=0, status="unavailable", reason="point_forecast")],
            )
    roles = {spec.name: spec.role for spec in models}
    rows += em.holm_rows(candidates, roles)
    return em.table(rows)


def _metric(table: pd.DataFrame, model: str, metric: str, method: str, *, reference=None):
    rows = table[
        (table["model"] == model)
        & (table["metric"] == metric)
        & (table["inference_method"] == method)
        & table["fold_id"].isna()
    ]
    if reference is not None and "reference" in table.columns:
        rows = rows[rows["reference"] == reference]
    return None if rows.empty else rows.iloc[0]


def _t(row) -> float | None:
    if row is None or row["status"] != "ok" or not row["se"] > 0:
        return None
    return float((row["estimate"] - row["null_value"]) / row["se"])


def model_summary(
    name: str, diagnostics: pd.DataFrame, extra: pd.DataFrame, reference: str
) -> dict[str, float | None]:
    """Headline numbers of one model (decision method fold-block t unless stated)."""

    def value(row):
        return None if row is None else float(row["estimate"])

    fold = em.DECISION_METHOD
    ic = _metric(diagnostics, name, "mean_rank_ic", fold)
    weighted = _metric(extra, name, "coverage_weighted_rank_ic", fold)
    permutation = _metric(
        diagnostics, name, "mean_rank_ic_vs_block_permutation", "null_distribution"
    )
    summary = {
        "mean_rank_ic": value(ic),
        "mean_rank_ic_t": _t(ic),
        "rank_ic_coverage": value(_metric(extra, name, "rank_ic_coverage", "point")),
        "coverage_weighted_rank_ic": value(weighted),
        "coverage_weighted_rank_ic_t": _t(weighted),
        "mean_rank_ic_minimum_detectable": value(
            _metric(diagnostics, name, "mean_rank_ic_minimum_detectable", fold)
        ),
        "oos_r2_vs_zero": value(
            _metric(diagnostics, name, "oos_r2_vs_zero", "circular_block_bootstrap")
        ),
        "mse_improvement_vs_zero_t": _t(
            _metric(diagnostics, name, "mse_improvement_vs_zero", fold)
        ),
        "permutation_p_value": None if permutation is None else float(permutation["p_value"]),
        "holm_adjusted_p": value(
            _metric(extra, name, "holm_adjusted_p", "holm_one_sided_fold_block_t")
        ),
    }
    difference = _metric(extra, name, "rank_ic_difference", fold, reference=reference)
    if difference is not None:
        summary[f"rank_ic_difference_vs_{reference}"] = value(difference)
        summary[f"rank_ic_difference_vs_{reference}_t"] = _t(difference)
        r2 = _metric(
            extra, name, "oos_r2_vs_reference", "circular_block_bootstrap", reference=reference
        )
        summary[f"oos_r2_vs_{reference}"] = value(r2)
        mse = _metric(extra, name, "mse_improvement_vs_reference", fold, reference=reference)
        summary[f"mse_improvement_vs_{reference}_t"] = _t(mse)
    return summary


# --- writing ---


def _json_bytes(value: object) -> bytes:
    return (json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True) + "\n").encode()


def write_directory(
    directory: Path, writers: Mapping[str, Callable[[Path], None]], manifest: Mapping
) -> str:
    """Write files through a staging directory, then rename (write-once); returns the manifest SHA-256."""

    if directory.exists():
        raise FileExistsError(f"{directory} already exists; outputs are never overwritten.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.partial"
    staging.mkdir()
    written: list[Path] = []
    try:
        files = {}
        for name, write in writers.items():
            written.append(staging / name)
            write(staging / name)
            files[name] = {"file_sha256": file_sha256(staging / name)}
        written.append(staging / "manifest.json")
        (staging / "manifest.json").write_bytes(_json_bytes({**manifest, "files": files}))
        digest = file_sha256(staging / "manifest.json")
        os.replace(staging, directory)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return digest


def _model_ids(config: ExperimentConfig, models) -> dict[str, str]:
    return {
        spec.name: f"{config.registry.model_prefix}.{spec.name}"
        for spec in models
        if spec.role == "candidate"
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    output_root: str | Path,
    provenance: Mapping[str, object],
    started_at: datetime,
    clock: Callable[[], datetime] | None = None,
) -> ExperimentResult:
    """Run one experiment end to end and record the attempt in the ledger."""

    root = Path(output_root)
    ledger = {
        "started_at": pd.Timestamp(started_at).isoformat(),
        "experiment": config.experiment,
        "config_sha256": config.sha256,
        "mode": "development",
    }
    try:
        result = _run(config, root, provenance, started_at, clock)
    except Exception as error:
        append_ledger(
            root, {**ledger, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        raise
    append_ledger(
        root,
        {
            **ledger,
            "status": "completed",
            "run_id": result.run_id,
            "results_sha256": result.record["outputs"]["results_sha256"],
            "registered": [str(entry.ref) for entry in result.entries],
            "git_sha": provenance["code"].get("git_sha"),
            "git_dirty": provenance["code"].get("git_dirty"),
        },
    )
    return result


def _run(config, root, provenance, started_at, clock) -> ExperimentResult:
    if config.track != "equity_cross_sectional":
        raise ConfigError(f"Track {config.track!r} is not implemented by this runner.")
    for key in ("code", "environment"):
        if key not in provenance:
            raise ValueError(f"provenance must include {key!r}.")
    dataset = build_equity_dataset(config)
    models = equity_models(config, dataset)
    settings = diagnostics_config(config, models, dataset.label_spec)
    labeled = dataset.labeled
    run = t2.run_diagnostics(
        labeled,
        dataset.folds,
        universe=sorted(labeled["symbol"].unique()),
        label_spec=dataset.label_spec,
        diagnostic_features=list(dataset.features),
        provenance={
            "code": provenance["code"],
            "environment": provenance["environment"],
            "inputs": {**dataset.fingerprints, "config_sha256": config.sha256},
            "truncation": {
                "truncated_before": MODEL_HOLDOUT_START.isoformat(),
                "max_date_loaded": labeled["date"].max().isoformat(),
                "holdout_rows_loaded": False,
                "excluded_symbols": {},
            },
            "fold_params": dataset.fold_params,
            "universe_note": SYNTHETIC_NOTE,
        },
        started_at=started_at,
        models=models,
        config=settings,
    )
    extra = experiment_metrics(config, models, run, settings)
    missingness = missingness_table(labeled, dataset.contracts, folds=dataset.folds)
    registry = Registry(
        root / REGISTRY_DIRECTORY,
        root,
        **({"clock": clock} if clock is not None else {}),
    )
    scope = research_scope(config, dataset.label_spec)
    prior = {e.evidence.get("variant_id") for e in registry.entries(scope)}
    model_ids = _model_ids(config, models)
    versions = {name: registry.next_version(model_id) for name, model_id in model_ids.items()}

    diagnostics_dir = write_run(
        run, root, extra_record={"experiment": config.experiment, "status": "completed"}
    )
    diagnostics_ref = ArtifactRef(
        "diagnostics_run",
        f"runs/{run.run_id}",
        file_sha256(diagnostics_dir / "manifest.json"),
    )
    predictions = run.tables["predictions"]
    forecasts: dict[str, dict] = {}
    forecast_refs: dict[str, ArtifactRef] = {}
    specs = {spec.name: spec for spec in models}
    for name, ref in versions.items():
        metadata = fa.ForecastMetadata(
            model_id=ref.model_id,
            model_version=ref.version,
            variant_id=specs[name].describe()["variant_id"],
            run_id=run.run_id,
            track=config.track,
            target=target_name(dataset.label_spec),
            horizon=dataset.label_spec.horizon,
            label_id=dataset.label_spec.label_id,
            forecast_kind="point",
            data_class=config.data_class,
            knowledge_rule=fa.SYNTHETIC_END_OF_UTC_DATE,
            config_sha256=config.sha256,
            code=dict(provenance["code"]),
        )
        own = predictions.loc[predictions["model"] == name]
        table = fa.forecast_frame(own, labeled, dataset.folds, metadata=metadata)
        directory, digest = fa.write_forecast_artifact(table, metadata, root / FORECASTS_DIRECTORY)
        relative = f"{FORECASTS_DIRECTORY}/{directory.name}"
        forecast_refs[name] = ArtifactRef("forecast_artifact", relative, digest)
        forecasts[name] = {
            "path": relative,
            "manifest_sha256": digest,
            "model_id": ref.model_id,
            "model_version": ref.version,
            "content_sha256": table_sha256(table),
            # Everything but the registry version: equal across repeated runs.
            "values_sha256": table_sha256(table.drop(columns="model_version")),
        }

    reference = config.models.increment_reference
    summary = {
        spec.name: model_summary(spec.name, run.tables["metrics"], extra, reference)
        for spec in models
        if spec.role in ("candidate", "diagnostic")
    }
    tables_sha = {"metrics": table_sha256(extra), "missingness": table_sha256(missingness)}
    results = {
        "diagnostics_results_sha256": run.record["outputs"]["results_sha256"],
        "experiment_tables_sha256": tables_sha,
        "forecasts_values_sha256": {n: f["values_sha256"] for n, f in forecasts.items()},
    }
    nested_specs = [spec for spec in models if nested.is_nested(spec)]
    record = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": run.run_id,
        "experiment": config.experiment,
        "track": config.track,
        "description": config.description,
        "started_at": pd.Timestamp(started_at).isoformat(),
        "config": config.to_json(),
        "config_sha256": config.sha256,
        "code": dict(provenance["code"]),
        "environment": dict(provenance["environment"]),
        "data": {
            "source": config.data.source,
            "data_class": config.data_class,
            "params": dataset.panel.spec.describe(),
            "fingerprints": dataset.fingerprints,
            "note": SYNTHETIC_NOTE,
        },
        "universe": dataset.universe.describe(),
        "features": {
            "families": list(config.features.families),
            "contracts": [dataset.contracts[name].describe() for name in dataset.features],
        },
        "label": {**t2.label_record(dataset.label_spec), "benchmark_members": BENCHMARK_MEMBERS},
        "folds": {
            "params": dataset.fold_params,
            "n_folds": len(dataset.folds),
            "fold_table_sha256": run.record["spec"]["folds"]["fold_table_sha256"],
            "inner": {
                "block_sessions": config.folds.inner_block_sessions,
                "n_windows": config.folds.inner_windows,
                "min_train_sessions": config.folds.inner_min_train_sessions,
            },
        },
        "seeds": {
            "data_seed": dataset.panel.spec.seed,
            "evaluation_seed": config.evaluation.seed,
            "seed_rule": SEED_RULE,
        },
        "lockbox_evidence": {
            key: run.record["lockbox_evidence"][key]
            for key in (
                "mode",
                "holdout_start",
                "max_date",
                "max_target_end_date",
                "holdout_values_used",
            )
        },
        "checks_summary": run.record["checks_summary"],
        "decision_method": t2.DECISION_METHOD,
        "nulls": {
            "models": [s.name for s in models if s.role == "null"],
            "controls": [s.name for s in models if s.role == "control"],
            "canary": [s.name for s in models if s.role == "canary"],
            "permutation_draws": settings.permutation_draws,
            "permutation_models": list(settings.permutation_models),
        },
        "multiple_testing": {
            "candidates_in_run": len(model_ids),
            "grid_points_in_run": sum(len(s.params["selection"]["grid"]) for s in nested_specs),
            "prior_variants_in_scope": len(prior - {None}),
        },
        "models": {
            spec.name: {"role": spec.role, "variant_id": spec.describe()["variant_id"]}
            for spec in models
        },
        "diagnostics_run": {
            "path": diagnostics_ref.path,
            "manifest_sha256": diagnostics_ref.sha256,
        },
        "forecasts": forecasts,
        "registry": {"research_scope": scope, "model_ids": model_ids},
        "summary": summary,
        "outputs": {**results, "results_sha256": sha256_json(results)},
    }
    directory = root / EXPERIMENTS_DIRECTORY / run.run_id
    digest = write_directory(
        directory,
        {
            "record.json": lambda path: path.write_bytes(_json_bytes(record)),
            "config.json": lambda path: path.write_bytes(_json_bytes(config.to_json())),
            "metrics.parquet": lambda path: extra.to_parquet(path, index=False),
            "missingness.parquet": lambda path: missingness.to_parquet(path, index=False),
        },
        {"experiment_schema_version": EXPERIMENT_SCHEMA_VERSION, "run_id": run.run_id},
    )
    experiment_ref = ArtifactRef("experiment_run", f"{EXPERIMENTS_DIRECTORY}/{run.run_id}", digest)

    stored = json.loads(canonical_json(record))  # evidence from the record as written
    entries = []
    for name, ref in versions.items():
        previous = registry.versions(ref.model_id)
        entries.append(
            registry.register(
                model_id=ref.model_id,
                research_scope=scope,
                config={
                    "model": specs[name].describe(),
                    "experiment": config.experiment,
                    "experiment_config_sha256": config.sha256,
                },
                training_data={
                    "data_class": config.data_class,
                    "labeled_frame_sha256": dataset.fingerprints["labeled_frame_sha256"],
                    "fold_table_sha256": record["folds"]["fold_table_sha256"],
                    "universe_sha256": dataset.universe.sha256,
                },
                metrics=summary[name],
                artifacts=[experiment_ref, diagnostics_ref, forecast_refs[name]],
                evidence=derive_evidence(stored, name),
                code=dict(provenance["code"]),
                parent=previous[-1] if previous else None,
                expected_version=ref.version,
            )
        )
    return ExperimentResult(
        run_id=run.run_id,
        record=record,
        directory=directory,
        diagnostics=run,
        tables={"metrics": extra, "missingness": missingness},
        entries=tuple(entries),
    )
