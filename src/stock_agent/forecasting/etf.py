"""
The ETF return and risk track (synthetic), separate from the equity
cross-sectional track (docs/research/ml_etf_track.md).

Every asset gets its own models, its own nested selection and its own
results; nothing assumes one model or one target fits all assets. For each
target (etf_data.TARGETS) and each asset:

    folds    purged expanding walk-forward folds on that asset's rows only,
             the same outer windows for every asset (development mode)
    nulls    historical_mean (the asset's expanding training-label mean, the
             Campbell-Thompson benchmark), zero (returns only), persistence
             (volatility only: the trailing 20-session log realized vol)
    models   ols (diagnostic; nests historical_mean), ridge (T4 grid and
             nested selection), optionally the gradient-boosting challenger

Evaluation (one value per date before inference; the date's mean over
assets for pooled rows; fold-block t decides):

    oos_r2_vs_<null>, mse_improvement_vs_<null>   every model against every null
    clark_west_vs_historical_mean                 only the nested pair (ols)
    qlike_improvement_vs_<null>                   volatility target
    oos_r2_vs_reference, mse_improvement_vs_reference   challenger over linear
    interval_coverage_<level>                     return intervals built from
                                                  the volatility forecast:
                                                  mu_hat +- z sigma_hat sqrt(h)

Rank IC is not used here: with six heterogeneous assets a cross-sectional
ranking is not the question (the equity track answers that one).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stock_agent.features.contracts import validate_feature_contracts
from stock_agent.features.frozen_universe import (
    UniverseRule,
    eligible_keys,
    freeze_universe,
)
from stock_agent.features.missingness import missingness_table
from stock_agent.features.training import LabelSpec
from stock_agent.forecasting import artifacts as fa
from stock_agent.forecasting import etf_data, nonlinear
from stock_agent.forecasting import metrics as em
from stock_agent.forecasting.config import ConfigError, ExperimentConfig
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics import linear, nested, scoring
from stock_agent.model_diagnostics import runner as t2
from stock_agent.model_diagnostics.controls import PooledMeanForecast, ZeroForecast
from stock_agent.model_diagnostics.record import canonical_json, run_id, sha256_json
from stock_agent.model_diagnostics.seeds import SEED_RULE, derive_seed
from stock_agent.model_validation.audit import LeakageError, frame_content_sha256, table_sha256
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    WalkForwardFold,
    make_expanding_folds,
    make_test_windows,
)
from stock_agent.model_validation.harness import run_walk_forward
from stock_agent.registry import ArtifactRef, Registry, derive_evidence

POOLED = "__pooled__"
NULLS = {
    "forward_return": ("historical_mean", "zero"),
    "log_realized_volatility": ("historical_mean", "persistence"),
}
NESTED_PAIR = ("ols", "historical_mean")
INTERVAL_ARTIFACT_LEVEL = 0.8
METRIC_COLUMNS = (*em.COLUMNS, "target", "asset")
SYNTHETIC_NOTE = "Synthetic ETF-like series: verifies the pipeline; never evidence about funds."


class FeatureForecast:
    """
    An unfitted forecast equal to one feature's value (the persistence null);
    a missing value falls back to the training rows' median of that feature.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureForecast:
        column = np.asarray(X, dtype=np.float64)[:, 0]
        self.fallback_ = float(np.nanmedian(column))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        column = np.asarray(X, dtype=np.float64)[:, 0]
        return np.where(np.isfinite(column), column, self.fallback_)


@dataclass(frozen=True, eq=False)
class AssetTarget:
    target: str
    asset: str
    frame: pd.DataFrame
    folds: tuple[WalkForwardFold, ...]


def _validate(config: ExperimentConfig) -> None:
    if config.label.benchmark is not None:
        raise ConfigError(
            "The ETF track forecasts raw per-asset targets: label.benchmark must be null."
        )
    unknown = set(config.features.families) - {c.family for c in etf_data.CONTRACTS}
    if unknown:
        raise ConfigError(f"The ETF track has no feature family {sorted(unknown)}.")
    if config.evaluation.permutation_models or config.evaluation.stale_lags:
        raise ConfigError("The ETF track runs no permutation or stale-feature controls yet.")


def build_etf_dataset(config: ExperimentConfig):
    try:
        spec = etf_data.SyntheticEtfSpec(**dict(config.data.params))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"data.params: {error}") from None
    panel = etf_data.synthetic_etf_panel(spec)
    universe = freeze_universe(
        panel.prices,
        UniverseRule(config.universe.min_history_sessions),
        universe_id=config.universe.id,
        survivorship_free=panel.survivorship_free,
    )
    keys = eligible_keys(universe)
    features = keys.merge(etf_data.etf_features(panel.prices), on=["date", "symbol"], how="left")
    names = [c.name for c in etf_data.CONTRACTS if c.family in config.features.families]
    contracts = {name: etf_data.CONTRACTS_BY_NAME[name] for name in names}
    validate_feature_contracts(features, names, contracts)
    if etf_data.PERSISTENCE_FEATURE not in features.columns:
        raise ConfigError("The volatility persistence null needs log_rv_20d.")
    start = pd.Timestamp(config.folds.start, tz="UTC")
    pieces: list[AssetTarget] = []
    for target in etf_data.TARGETS:
        labels = etf_data.etf_labels(
            panel.prices,
            horizon=config.label.horizon,
            entry_lag=config.label.entry_lag,
            target=target,
        )
        labeled = features.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
        last = labeled.loc[labeled["target_return"].notna(), "date"].max()
        sessions = labeled["date"].drop_duplicates().sort_values()
        windows = make_test_windows(
            sessions,
            start=start,
            end=last + pd.Timedelta(days=1),
            block_sessions=config.folds.block_sessions,
        )
        for asset in sorted(labeled["symbol"].unique()):
            frame = labeled.loc[labeled["symbol"] == asset].reset_index(drop=True)
            folds = make_expanding_folds(frame, windows, holdout_start=MODEL_HOLDOUT_START)
            pieces.append(AssetTarget(target, asset, frame, folds))
    return panel, universe, features, names, contracts, pieces


def _label_spec(config: ExperimentConfig) -> LabelSpec:
    return LabelSpec(config.label.horizon, config.label.entry_lag, price_column="close")


def _label_id(config: ExperimentConfig, target: str) -> str:
    base = _label_spec(config).label_id
    return (
        base
        if target == "forward_return"
        else base.replace("forward_simple_return", "forward_log_realized_volatility")
    )


def model_specs(config: ExperimentConfig, target: str, features: list[str]):
    """name -> (role, ModelSpec or estimator factory, feature columns)."""

    design = nested.InnerDesign(
        block_sessions=config.folds.inner_block_sessions,
        n_windows=config.folds.inner_windows,
        min_train_sessions=config.folds.inner_min_train_sessions,
    )
    label_spec = _label_spec(config)
    linear_specs = {
        spec.name: replace(spec, params={**spec.params, "label_id": _label_id(config, target)})
        for spec in nested.linear_models(label_spec, features, design, include_ols=True)
    }
    specs: dict[str, tuple[str, object, list[str]]] = {
        "historical_mean": ("null", PooledMeanForecast, features[:1]),
    }
    if target == "forward_return":
        specs["zero"] = ("null", ZeroForecast, features[:1])
    else:
        specs["persistence"] = ("null", FeatureForecast, [etf_data.PERSISTENCE_FEATURE])
    if config.models.include_ols:
        specs["ols"] = ("diagnostic", linear_specs["ols"], features)
    for name in config.models.candidates:
        specs[name] = ("candidate", linear_specs[name], features)
    if nonlinear.NAME in config.models.challengers:
        challenger = nonlinear.challenger_spec(label_spec, features, design)
        challenger = replace(
            challenger, params={**challenger.params, "label_id": _label_id(config, target)}
        )
        specs[nonlinear.NAME] = ("candidate", challenger, features)
    return specs


def _predict(piece: AssetTarget, name: str, entry, checks: list[dict], selection: list[dict]):
    role, model, columns = entry
    if isinstance(model, t2.ModelSpec) and nested.is_nested(model):
        design = nested.InnerDesign(**model.params["selection"]["inner"])
        inner = {f.fold_id: nested.build_inner_folds(piece.frame, f, design) for f in piece.folds}
        result = nested.run_nested(
            piece.frame,
            piece.folds,
            model,
            inner,
            selector=linear.select_hyperparameters,
            lockbox_start=MODEL_HOLDOUT_START,
            record=True,
        )
        starts = {fold.fold_id: fold.test_start for fold in piece.folds}
        rows = [
            {**row, "target": piece.target, "asset": piece.asset} for row in result.selection_rows
        ]
        selection += rows
        verified = len(
            {r["fold_id"] for r in rows if r["max_inner_label_end"] < starts[r["fold_id"]]}
        )
        checks.append(
            t2._check(
                "nested_selection_inside_training",
                f"{piece.target}:{piece.asset}:{name}",
                "leakage",
                "outer_folds_with_verified_inner_folds",
                verified,
                len(piece.folds),
                "==",
            )
        )
        estimators = (*result.estimators, *result.inner_estimators)
        predictions = result.predictions
    else:
        factory = model if not isinstance(model, t2.ModelSpec) else None
        if factory is None:
            kwargs = nested.estimator_kwargs(model.params)

            def factory(model=model, kwargs=kwargs):
                return model.estimator(**kwargs)

        with threadpool_limits(limits=1):
            result = run_walk_forward(
                piece.frame,
                piece.folds,
                feature_columns=columns,
                make_estimator=factory,
                lockbox_start=MODEL_HOLDOUT_START,
            )
        estimators = result.estimators
        predictions = result.predictions
    if isinstance(model, t2.ModelSpec) and nested.is_linear(model):
        flags = [linear.has_converged(e) for e in estimators]
        checks.append(
            t2._check(
                "selection_fits_converged",
                f"{piece.target}:{piece.asset}:{name}",
                "instrument",
                "fits_not_converged",
                sum(flag is False for flag in flags),
                0,
                "==",
            )
        )
    expected = sum(len(fold.test_rows) for fold in piece.folds)
    checks.append(
        t2._check(
            "predictions_complete",
            f"{piece.target}:{piece.asset}:{name}",
            "leakage",
            "n_rows",
            len(predictions),
            expected,
            "==",
        )
    )
    return predictions.assign(model=name, role=role, target=piece.target)


def _qlike(label: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Patton's QLIKE loss on variances, with log-volatility inputs."""

    ratio = np.exp(2.0 * (label - forecast))
    return ratio - 2.0 * (label - forecast) - 1.0


def _paired_series(model: pd.DataFrame, baseline: pd.DataFrame, ctx, loss) -> np.ndarray:
    joined = scoring.aligned_pair(model, baseline, ctx.target_column)
    positions = scoring._positions(joined, ctx)
    label = joined[ctx.target_column].to_numpy(np.float64)
    difference = loss(label, joined["prediction_baseline"].to_numpy(np.float64)) - loss(
        label, joined["prediction"].to_numpy(np.float64)
    )
    return scoring.session_mean(difference, positions, ctx)


def _with(rows: list[dict], target: str, asset: str) -> list[dict]:
    return [{**row, "target": target, "asset": asset} for row in rows]


def evaluate(predictions: pd.DataFrame, config: ExperimentConfig, roles: Mapping[str, str]):
    """Every ETF metric row (pooled over assets and per asset)."""

    rows: list[dict] = []
    seed = config.evaluation.seed
    for target in etf_data.TARGETS:
        own = predictions.loc[predictions["target"] == target]
        by_model = {name: frame for name, frame in own.groupby("model", sort=False)}
        reference = by_model["historical_mean"]
        ctx = scoring.build_context(
            reference,
            block_length=inf.DEFAULT_BLOCK_LENGTH,
            bootstrap_reps=config.evaluation.bootstrap_reps,
            seed=derive_seed(seed, "bootstrap", target),
        )
        scopes = [(POOLED, None), *[(a, a) for a in sorted(own["symbol"].unique())]]
        compared = [m for m, role in roles.items() if role in ("candidate", "diagnostic")]
        for scope, asset in scopes:

            def pick(frame, asset=asset):
                return frame if asset is None else frame.loc[frame["symbol"] == asset]

            for name in compared:
                model = pick(by_model[name])
                for baseline in NULLS[target]:
                    base = pick(by_model[baseline])
                    r2, improvement, _ = scoring.oos_r2_vs(model, base, ctx)
                    rows += _with(
                        em.rows(name, roles[name], f"oos_r2_vs_{baseline}", [r2]), target, scope
                    )
                    rows += _with(
                        em.rows(name, roles[name], f"mse_improvement_vs_{baseline}", improvement),
                        target,
                        scope,
                    )
                    if target == "log_realized_volatility":
                        series = _paired_series(model, base, ctx, _qlike)
                        rows += _with(
                            em.rows(
                                name,
                                roles[name],
                                f"qlike_improvement_vs_{baseline}",
                                scoring.mean_inferences(series, ctx),
                            ),
                            target,
                            scope,
                        )
                if target == "forward_return" and name == NESTED_PAIR[0]:
                    cw = scoring.clark_west_vs(model, pick(by_model[NESTED_PAIR[1]]), ctx)
                    rows += _with(
                        em.rows(name, roles[name], f"clark_west_vs_{NESTED_PAIR[1]}", cw),
                        target,
                        scope,
                    )
                if name in config.models.challengers:
                    for reference_name in [*config.models.candidates, "ols"]:
                        if reference_name not in by_model:
                            continue
                        r2, improvement, _ = scoring.oos_r2_vs(
                            model, pick(by_model[reference_name]), ctx
                        )
                        rows += _with(
                            em.rows(
                                name,
                                roles[name],
                                "oos_r2_vs_reference",
                                [r2],
                                reference=reference_name,
                            ),
                            target,
                            scope,
                        )
                        rows += _with(
                            em.rows(
                                name,
                                roles[name],
                                "mse_improvement_vs_reference",
                                improvement,
                                reference=reference_name,
                            ),
                            target,
                            scope,
                        )
    rows += interval_rows(predictions, config)
    return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))


def return_intervals(predictions: pd.DataFrame, config: ExperimentConfig, level: float, model: str):
    """mu_hat +- z sigma_hat sqrt(h) from the same model's return and volatility forecasts."""

    keys = ["fold_id", "date", "symbol"]
    returns = predictions.query("target == 'forward_return' and model == @model")
    volatility = predictions.query("target == 'log_realized_volatility' and model == @model")
    joined = returns.merge(
        volatility[[*keys, "prediction"]], on=keys, how="inner", suffixes=("", "_vol")
    )
    z = inf.normal_ppf(0.5 + level / 2.0)
    spread = z * np.exp(joined["prediction_vol"]) * math.sqrt(config.label.horizon)
    return joined.assign(
        interval_low=joined["prediction"] - spread,
        interval_high=joined["prediction"] + spread,
        interval_level=level,
    )


def interval_rows(predictions: pd.DataFrame, config: ExperimentConfig) -> list[dict]:
    rows: list[dict] = []
    reference = config.models.increment_reference
    returns = predictions.query("target == 'forward_return' and model == @reference")
    ctx = scoring.build_context(
        returns,
        block_length=inf.DEFAULT_BLOCK_LENGTH,
        bootstrap_reps=config.evaluation.bootstrap_reps,
        seed=derive_seed(config.evaluation.seed, "bootstrap", "intervals"),
    )
    for level in config.evaluation.interval_levels:
        bands = return_intervals(predictions, config, level, reference)
        rows += _with(
            em.interval_coverage(reference, "candidate", bands, ctx, level=level),
            "forward_return",
            POOLED,
        )
        for asset, part in bands.groupby("symbol", sort=True):
            rows += _with(
                em.interval_coverage(reference, "candidate", part, ctx, level=level),
                "forward_return",
                asset,
            )
    return rows


def _pick(metrics: pd.DataFrame, model, target, metric, method, asset=POOLED, reference=None):
    rows = metrics[
        (metrics["model"] == model)
        & (metrics["target"] == target)
        & (metrics["metric"] == metric)
        & (metrics["inference_method"] == method)
        & (metrics["asset"] == asset)
    ]
    if reference is not None:
        rows = rows[rows["reference"] == reference]
    return None if rows.empty else rows.iloc[0]


def summary(metrics: pd.DataFrame, model: str, target: str, config) -> dict[str, float | None]:
    fold = em.DECISION_METHOD

    def est(row):
        return None if row is None or row["status"] != "ok" else float(row["estimate"])

    def t(row):
        if row is None or row["status"] != "ok" or not row["se"] > 0:
            return None
        return float((row["estimate"] - row["null_value"]) / row["se"])

    result: dict[str, float | None] = {}
    assets = sorted(set(metrics["asset"]) - {POOLED})
    for baseline in NULLS[target]:
        r2 = f"oos_r2_vs_{baseline}"
        result[r2] = est(_pick(metrics, model, target, r2, "circular_block_bootstrap"))
        mse = f"mse_improvement_vs_{baseline}"
        result[f"{mse}_t"] = t(_pick(metrics, model, target, mse, fold))
        for asset in assets:
            result[f"{r2}.{asset}"] = est(
                _pick(metrics, model, target, r2, "circular_block_bootstrap", asset)
            )
        if target == "log_realized_volatility":
            qlike = f"qlike_improvement_vs_{baseline}"
            result[f"{qlike}_t"] = t(_pick(metrics, model, target, qlike, fold))
    if model in config.models.challengers:
        for reference in [*config.models.candidates, "ols"]:
            row = _pick(
                metrics,
                model,
                target,
                "oos_r2_vs_reference",
                "circular_block_bootstrap",
                reference=reference,
            )
            result[f"oos_r2_vs_{reference}"] = est(row)
            row = _pick(
                metrics, model, target, "mse_improvement_vs_reference", fold, reference=reference
            )
            result[f"mse_improvement_vs_{reference}_t"] = t(row)
    if target == "forward_return" and model == config.models.increment_reference:
        for level in config.evaluation.interval_levels:
            metric = f"interval_coverage_{round(level * 100)}"
            result[metric] = est(_pick(metrics, model, target, metric, fold))
    return result


def run_etf(
    config, root: Path, provenance, started_at: datetime, clock, write_directory, json_bytes
):
    """The ETF track end to end; returns (run_id, record, directory, tables, entries)."""

    _validate(config)
    panel, universe, features, names, contracts, pieces = build_etf_dataset(config)
    checks: list[dict] = []
    selection: list[dict] = []
    frames: list[pd.DataFrame] = []
    evidence_dates, label_ends = [], []
    specs_by_target = {target: model_specs(config, target, names) for target in etf_data.TARGETS}
    for piece in pieces:
        lockbox = t2.lockbox_evidence(piece.frame, piece.folds, MODEL_HOLDOUT_START)  # raises
        late = t2.asof_violations(piece.frame, piece.folds, lockbox)
        if late:
            raise LeakageError(
                f"{piece.target}:{piece.asset}: folds {late} score unmatured labels."
            )
        scope = f"{piece.target}:{piece.asset}"
        checks.append(
            t2._check("lockbox_closed", scope, "leakage", "held_out_rows_total", 0, 0, "==")
        )
        checks.append(
            t2._check("evaluation_asof_matches_folds", scope, "leakage", "late_folds", 0, 0, "==")
        )
        evidence_dates.append(lockbox["max_date"])
        label_ends.append(lockbox["max_target_end_date"])
        for name, entry in specs_by_target[piece.target].items():
            frames.append(_predict(piece, name, entry, checks, selection))
    predictions = pd.concat(frames, ignore_index=True)
    roles = {name: entry[0] for name, entry in specs_by_target["forward_return"].items()}
    roles.update(
        {name: entry[0] for name, entry in specs_by_target["log_realized_volatility"].items()}
    )
    metrics = evaluate(predictions, config, roles)
    checks_table = pd.DataFrame(checks)
    selection_table = pd.DataFrame(selection)
    missingness = missingness_table(features, contracts)
    spec_record = {
        "config_sha256": config.sha256,
        "code": dict(provenance["code"]),
        "data": panel.spec.describe(),
        "universe_sha256": universe.sha256,
    }
    identifier = run_id(sha256_json(spec_record), started_at)
    tables = {
        "predictions": predictions,
        "metrics": metrics,
        "checks": checks_table,
        "selection": selection_table,
        "missingness": missingness,
    }
    tables_sha = {
        name: table_sha256(table.reset_index(drop=True)) for name, table in tables.items()
    }

    registry = Registry(root / "registry", root, **({"clock": clock} if clock else {}))
    registered = [
        (target, name)
        for target in etf_data.TARGETS
        for name, entry in specs_by_target[target].items()
        if entry[0] == "candidate"
    ]
    scopes = {
        target: f"{config.track}:{config.data_class}:{target}:h{config.label.horizon}:"
        f"{config.universe.id}"
        for target in etf_data.TARGETS
    }
    prior = em.prior_trials(registry, scopes.values())
    keys = {
        (target, name): f"{etf_data.TARGET_SHORT[target]}.{name}" for target, name in registered
    }
    versions = {
        key: registry.next_version(f"{config.registry.model_prefix}.{keys[key]}")
        for key in registered
    }

    forecasts, forecast_refs, values = {}, {}, {}
    for (target, name), ref in versions.items():
        spec = specs_by_target[target][name][1]
        kind = (
            "interval"
            if target == "forward_return" and name == config.models.increment_reference
            else "point"
        )
        metadata = fa.ForecastMetadata(
            model_id=ref.model_id,
            model_version=ref.version,
            variant_id=spec.describe()["variant_id"],
            run_id=identifier,
            track=config.track,
            target=target,
            horizon=config.label.horizon,
            label_id=_label_id(config, target),
            forecast_kind=kind,
            data_class=config.data_class,
            knowledge_rule=fa.SYNTHETIC_END_OF_UTC_DATE,
            config_sha256=config.sha256,
            code=dict(provenance["code"]),
        )
        own = predictions.query("target == @target and model == @name")
        intervals = (
            return_intervals(predictions, config, INTERVAL_ARTIFACT_LEVEL, name)
            if kind == "interval"
            else None
        )
        parts = []
        for piece in pieces:
            if piece.target != target:
                continue
            mine = own.loc[own["symbol"] == piece.asset]
            bounds = (
                None if intervals is None else intervals.loc[intervals["symbol"] == piece.asset]
            )
            parts.append(
                fa.forecast_frame(
                    mine, piece.frame, piece.folds, metadata=metadata, intervals=bounds
                )
            )
        table = fa.conform(pd.concat(parts, ignore_index=True))
        directory, digest = fa.write_forecast_artifact(table, metadata, root / "forecasts")
        relative = f"forecasts/{directory.name}"
        forecast_refs[(target, name)] = ArtifactRef("forecast_artifact", relative, digest)
        values[keys[(target, name)]] = table_sha256(table.drop(columns="model_version"))
        forecasts[keys[(target, name)]] = {
            "path": relative,
            "manifest_sha256": digest,
            "model_id": ref.model_id,
            "model_version": ref.version,
            "forecast_kind": kind,
        }

    results = {"tables_sha256": tables_sha, "forecasts_values_sha256": values}
    families = t2.summarize_checks(checks_table)
    record = {
        "experiment_schema_version": "1.0",
        "run_id": identifier,
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
            "params": panel.spec.describe(),
            "fingerprints": {
                "prices_sha256": frame_content_sha256(panel.prices, ["close"]),
                "universe_sha256": universe.sha256,
                "features_sha256": frame_content_sha256(features, names),
            },
            "note": SYNTHETIC_NOTE,
        },
        "universe": universe.describe(),
        "features": {
            "families": list(config.features.families),
            "contracts": [contracts[name].describe() for name in names],
        },
        "label": {
            "targets": {target: _label_id(config, target) for target in etf_data.TARGETS},
            "horizon": config.label.horizon,
            "entry_lag": config.label.entry_lag,
            "availability": "known after the exit close (target_end_date), as for every label",
        },
        "folds": {
            "start": pd.Timestamp(config.folds.start, tz="UTC").isoformat(),
            "block_sessions": config.folds.block_sessions,
            "per_asset": True,
            "n_folds": {f"{p.target}:{p.asset}": len(p.folds) for p in pieces},
            "fold_windows_sha256": sha256_json(
                [[str(f.test_start), str(f.test_end)] for f in pieces[0].folds]
            ),
        },
        "seeds": {
            "data_seed": panel.spec.seed,
            "evaluation_seed": config.evaluation.seed,
            "seed_rule": SEED_RULE,
        },
        "lockbox_evidence": {
            "mode": "development",
            "holdout_start": MODEL_HOLDOUT_START.isoformat(),
            "max_date": max(evidence_dates),
            "max_target_end_date": max(label_ends),
            "holdout_values_used": False,
        },
        "checks_summary": families,
        "decision_method": t2.DECISION_METHOD,
        "nulls": {
            "models": sorted({n for n, r in roles.items() if r == "null"}),
            "permutation_draws": 0,
            "nested_pair": list(NESTED_PAIR),
        },
        "multiple_testing": {
            "candidates_in_run": len(registered),
            "grid_points_in_run": sum(
                len(specs_by_target[t][n][1].params["selection"]["grid"]) for t, n in registered
            ),
            "prior_variants_in_scope": prior,
        },
        "models": {
            keys[(t, n)]: {
                "role": "candidate",
                "variant_id": specs_by_target[t][n][1].describe()["variant_id"],
            }
            for t, n in registered
        },
        "forecasts": forecasts,
        "registry": {"research_scopes": scopes},
        "summary": {keys[(t, n)]: summary(metrics, n, t, config) for t, n in registered},
        "outputs": {**results, "results_sha256": sha256_json(results)},
    }
    directory = root / "experiments" / identifier
    writers = {
        "record.json": lambda path: path.write_bytes(json_bytes(record)),
        "config.json": lambda path: path.write_bytes(json_bytes(config.to_json())),
    }
    for name, table in tables.items():
        writers[f"{name}.parquet"] = lambda path, table=table: table.to_parquet(path, index=False)
    digest = write_directory(
        directory, writers, {"experiment_schema_version": "1.0", "run_id": identifier}
    )
    experiment_ref = ArtifactRef("experiment_run", f"experiments/{identifier}", digest)
    stored = json.loads(canonical_json(record))
    entries = []
    for (target, name), ref in versions.items():
        previous = registry.versions(ref.model_id)
        spec = specs_by_target[target][name][1]
        entries.append(
            registry.register(
                model_id=ref.model_id,
                research_scope=scopes[target],
                config={
                    "model": spec.describe(),
                    "target": target,
                    "per_asset": True,
                    "experiment": config.experiment,
                    "experiment_config_sha256": config.sha256,
                },
                training_data={
                    "data_class": config.data_class,
                    "features_sha256": record["data"]["fingerprints"]["features_sha256"],
                    "fold_table_sha256": record["folds"]["fold_windows_sha256"],
                    "labeled_frame_sha256": record["data"]["fingerprints"]["prices_sha256"],
                    "universe_sha256": universe.sha256,
                },
                metrics=record["summary"][keys[(target, name)]],
                artifacts=[experiment_ref, forecast_refs[(target, name)]],
                evidence=derive_evidence(stored, keys[(target, name)]),
                code=dict(provenance["code"]),
                parent=previous[-1] if previous else None,
                expected_version=ref.version,
            )
        )
    return identifier, record, directory, tables, tuple(entries)
