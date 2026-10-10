"""
Pure orchestration of a development diagnostics run (no file I/O).

Order of a run, enforced by structure:

1. Lockbox gate: the frame must already be truncated before the holdout
   (no row dated on or after it, no finite label ending in it, no fold
   with held-out rows). A frame that was labeled before truncation is
   refused, because its late-2024 labels were computed from holdout prices.
2. Prediction stage: every model, placebo draw and timeliness control runs
   through model_validation.harness.run_walk_forward on the same folds; the
   unpurged canary reference runs outside it. All predictions are complete
   and their label-free hashes fixed before any scorer runs.
3. Scoring stage: metrics, curves, null comparisons, feature diagnostics.
   Nothing computed here is fed back into stage 2.
4. Run record: spec (hashed), lockbox evidence, output hashes, checks.

The caller injects provenance (git state, input file hashes, environment
versions, how the frame was truncated) and the start time; see
scripts/run_null_diagnostics.py.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    LabelSpec,
)
from stock_agent.model_diagnostics import contract, scoring
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics.canary import UNSAFE_ROLE, unpurged_reference_predictions
from stock_agent.model_diagnostics.controls import (
    CONTROL_COLUMNS,
    CONTROL_PREFIX,
    CONTROL_SESSION_INDEX,
    CONTROL_SYMBOL_CODE,
    FeatureScore,
    NearestKeyMemorizer,
    PerSymbolMeanForecast,
    PooledMeanForecast,
    RandomNoiseScore,
    ZeroForecast,
    add_control_columns,
    symbol_codes,
)
from stock_agent.model_diagnostics.feature_diagnostics import (
    evaluation_feature_drift,
    training_feature_diagnostics,
)
from stock_agent.model_diagnostics.placebo import block_permuted_features, stale_features
from stock_agent.model_diagnostics.record import (
    RECORD_SCHEMA_VERSION,
    run_id,
    sha256_json,
    variant_id,
)
from stock_agent.model_diagnostics.seeds import SEED_RULE, derive_seed
from stock_agent.model_validation.audit import (
    DECISION_CONVENTION,
    LeakageError,
    datetime_ns,
    describe_folds,
    frame_content_sha256,
    frame_keys_sha256,
    frame_rows_sha256,
    table_sha256,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, WalkForwardFold, is_labeled
from stock_agent.model_validation.harness import run_walk_forward

ROLES = ("null", "control", "canary", "candidate")
# Checks decide on the best-calibrated method in simulation (inference.py).
DECISION_METHOD = "fold_block_t"
PREDICTION_KEYS = ["fold_id", "date", "symbol"]
LABEL_COLUMNS = [TARGET_START_COLUMN, TARGET_END_COLUMN, TARGET_RETURN_COLUMN]


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """
    One predictor of a run. `seeded` estimators receive a per-fold seed
    derived from the run seed (see seeds.SEED_RULE). Candidate models may
    not use control_* columns, which encode symbol identity and time.
    """

    name: str
    role: str
    estimator: type
    feature_columns: tuple[str, ...]
    params: Mapping[str, object] = field(default_factory=dict)
    output_kind: str = "forecast"
    seeded: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ModelSpec.name must be a non-empty string.")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}; got {self.role!r}.")
        if self.output_kind not in scoring.OUTPUT_KINDS:
            raise ValueError(f"output_kind must be one of {scoring.OUTPUT_KINDS}.")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty.")
        if self.role == "candidate" and any(
            column.startswith(CONTROL_PREFIX) for column in self.feature_columns
        ):
            raise ValueError(
                "Candidate models may not use control_* columns: they encode symbol "
                "identity and time and would let a model fit a per-symbol effect."
            )

    def describe(self) -> dict[str, object]:
        description = {
            "name": self.name,
            "role": self.role,
            "estimator": f"{self.estimator.__module__}.{self.estimator.__qualname__}",
            "params": dict(self.params),
            "feature_columns": list(self.feature_columns),
            "output_kind": self.output_kind,
            "seeded": self.seeded,
            "preprocessing": None,
            "note": self.note,
        }
        description["variant_id"] = variant_id(description)
        return description


def default_models(*, score_feature: str = "momentum_20d") -> list[ModelSpec]:
    """The T2 null, control and canary set."""

    identity = (CONTROL_SYMBOL_CODE,)
    return [
        ModelSpec("zero", "null", ZeroForecast, identity, note="baseline; constant"),
        ModelSpec(
            "pooled_mean",
            "null",
            PooledMeanForecast,
            identity,
            note="expanding training-label mean; Campbell-Thompson R^2 baseline",
        ),
        ModelSpec(
            "random_noise",
            "null",
            RandomNoiseScore,
            identity,
            output_kind="score",
            seeded=True,
            note="iid noise; exercises rank and sign metrics under no skill",
        ),
        ModelSpec(
            "per_symbol_mean",
            "control",
            PerSymbolMeanForecast,
            identity,
            params={"min_history": 63},
            note="selection control: symbol identity only; not a null on this universe",
        ),
        ModelSpec(
            score_feature,
            "control",
            FeatureScore,
            (score_feature,),
            params={"sign": 1},
            output_kind="score",
            note="unfitted benchmark; sign +1 pre-registered (momentum), not tuned",
        ),
        ModelSpec(
            "memorizer",
            "canary",
            NearestKeyMemorizer,
            (CONTROL_SYMBOL_CODE, CONTROL_SESSION_INDEX),
            note="leakage canary; skill only through leakage",
        ),
    ]


@dataclass(frozen=True)
class DiagnosticsConfig:
    hac_lag: int = inf.DEFAULT_HAC_LAG
    block_length: int = inf.DEFAULT_BLOCK_LENGTH
    bootstrap_reps: int = inf.DEFAULT_BOOTSTRAP_REPS
    ci_level: float = inf.DEFAULT_CI_LEVEL
    seed: int = 0
    min_names: int = 3
    decile_bins: int = 10
    acf_max_lag: int = 40
    baselines: tuple[str, ...] = ("zero", "pooled_mean", "per_symbol_mean")
    canary_model: str = "memorizer"
    permutation_models: tuple[str, ...] = ("per_symbol_mean", "momentum_20d")
    permutation_draws: int = 49
    permutation_block_length: int = 63
    stale_models: tuple[str, ...] = ("momentum_20d",)
    stale_lags: tuple[int, ...] = (260, 290, 330)
    max_static_ordering_symbols: int = 6
    no_skill_abs_t: float = 3.0
    canary_unsafe_min_t: float = 2.0
    calibration_reps: int = 2000
    calibration_bootstrap_reps: int = 199
    calibration_signals: tuple[str, ...] = inf.CALIBRATION_SIGNALS
    # fold-block t is the decision method; 0.065 is ~3 Monte Carlo SEs above
    # 5% at 2000 replicates. Other methods' simulated sizes are recorded as info.
    max_false_positive_rate: float = 0.065
    research_minimum_hac_lag: int = 40


@dataclass(frozen=True, eq=False)
class DiagnosticsRun:
    run_id: str
    record: dict[str, object]
    tables: dict[str, pd.DataFrame]


def label_id(label_spec: LabelSpec) -> str:
    return (
        f"forward_simple_return:{label_spec.price_column}:"
        f"horizon={label_spec.horizon}:entry_lag={label_spec.entry_lag}"
    )


# --- stage 1: lockbox gate ---


def prepare_development_frame(
    panel: pd.DataFrame,
    *,
    symbols: Sequence[str],
    label_spec: LabelSpec,
    holdout_start: object = MODEL_HOLDOUT_START,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Restrict to the universe, truncate before the holdout, THEN label.

    Labeling after truncation means no label can use a holdout price: rows
    whose exit falls in the holdout get no label instead of a withheld one.
    Truncation keeps every development feature and matured label
    bit-identical (features look backward within a symbol; labels are
    same-symbol shifts). Symbols with no pre-holdout rows are excluded with
    reason no_pre_lockbox_rows. Returns (labeled frame, truncation facts).
    """

    cutoff = pd.Timestamp(holdout_start)
    wanted = list(symbols)
    present = set(panel["symbol"])
    missing = sorted(set(wanted) - present)
    if missing:
        raise ValueError(f"Requested symbols are absent from the panel: {missing}")
    universe_rows = panel.loc[panel["symbol"].isin(wanted)]
    before = universe_rows["date"] < cutoff
    kept = universe_rows.loc[before]
    dropped = universe_rows.loc[~before]
    excluded = {s: "no_pre_lockbox_rows" for s in sorted(set(wanted) - set(kept["symbol"]))}
    labeled = label_spec.apply(kept.copy())
    labeled = labeled.sort_values(["date", "symbol"]).reset_index(drop=True)
    unlabeled = ~is_labeled(labeled)
    facts = {
        "truncated_before": cutoff.isoformat(),
        "rows_dropped_by_symbol": {
            str(k): int(v) for k, v in dropped["symbol"].value_counts().sort_index().items()
        },
        "max_date_loaded": _iso(universe_rows["date"].max()),
        "excluded_symbols": excluded,
        "rows_unlabeled_after_truncation": int(unlabeled.sum()),
        "label_applied_after_truncation": True,
    }
    return labeled, facts


def lockbox_evidence(
    labeled: pd.DataFrame, folds: Sequence[WalkForwardFold], lockbox_start: object
) -> dict[str, object]:
    """
    Refuse holdout data in a development run; return the evidence.

    holdout_values_used is derived here from the data, never passed in.
    """

    lockbox = pd.Timestamp(lockbox_start)
    lockbox_ns = datetime_ns(pd.Series([lockbox]))[0]
    for fold in folds:
        if fold.mode != "development":
            raise LeakageError(
                f"Fold {fold.fold_id} is a {fold.mode} fold; T2 runs are development only."
            )
        if datetime_ns(pd.Series([fold.holdout_start]))[0] > lockbox_ns:
            raise LeakageError(f"Fold {fold.fold_id} has holdout_start after the lockbox.")
        if len(fold.held_out_rows):
            raise LeakageError(
                f"Fold {fold.fold_id} withholds {len(fold.held_out_rows)} rows whose labels mature "
                "in the lockbox: truncate the panel before labeling (prepare_development_frame)."
            )
    dates = datetime_ns(labeled["date"])
    if (dates >= lockbox_ns).any():
        raise LeakageError("The frame holds rows dated in the lockbox; truncate it first.")
    labeled_mask = is_labeled(labeled)
    for column in (TARGET_START_COLUMN, TARGET_END_COLUMN):
        values = datetime_ns(labeled[column])[labeled_mask]
        if (values >= lockbox_ns).any():
            raise LeakageError(
                f"Labels with {column} in the lockbox were computed from holdout prices; "
                "label after truncating."
            )
    return {
        "holdout_start": lockbox.isoformat(),
        "mode": "development",
        "max_date": _iso(labeled["date"].max()),
        "max_target_start_date": _iso(labeled.loc[labeled_mask, TARGET_START_COLUMN].max()),
        "max_target_end_date": _iso(labeled.loc[labeled_mask, TARGET_END_COLUMN].max()),
        "n_rows": len(labeled),
        "n_labeled": int(labeled_mask.sum()),
        "held_out_rows_total": 0,
        "holdout_values_used": False,
    }


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


# --- stage 2: predictions ---


def _factory(spec: ModelSpec, seed: int, tag: str):
    counter = itertools.count()

    def make():
        params = dict(spec.params)
        if spec.seeded:
            params["seed"] = derive_seed(seed, "model", spec.name, tag, "fold_call", next(counter))
        return spec.estimator(**params)

    return make


def _estimator_audit(estimators: Sequence[object]) -> dict[str, list]:
    audit: dict[str, list] = {}
    for attribute in ("fallback_predictions_", "n_train_"):
        values = [getattr(estimator, attribute, None) for estimator in estimators]
        if any(value is not None for value in values):
            audit[attribute.rstrip("_")] = values
    return audit


def _check_complete(name: str, predictions: pd.DataFrame, expected: pd.DataFrame) -> None:
    if len(predictions) != len(expected) or frame_keys_sha256(predictions) != frame_keys_sha256(
        expected
    ):
        raise ValueError(f"{name}: predictions do not cover exactly the folds' test rows.")
    if not np.isfinite(predictions["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name}: predictions must be finite.")


def _label_free_hash(predictions: pd.DataFrame) -> str:
    return table_sha256(predictions.loc[:, [*PREDICTION_KEYS, "prediction"]])


# --- stage 4: checks ---


def _check(check, model, severity, metric, observed, threshold, comparison, detail=None) -> dict:
    observed = float(observed) if observed is not None else math.nan
    if not math.isfinite(observed):
        passed = None
    elif comparison == "<":
        passed = observed < threshold
    elif comparison == "<=":
        passed = observed <= threshold
    elif comparison == ">":
        passed = observed > threshold
    elif comparison == ">=":
        passed = observed >= threshold
    else:
        passed = observed == threshold
    return {
        "check": check,
        "model": model,
        "severity": severity,
        "metric": metric,
        "observed": observed,
        "threshold": float(threshold),
        "comparison": comparison,
        "passed": passed,
        "detail": detail,
    }


def _abs_t(result: inf.Inference) -> float:
    if result.status != "ok" or not result.se > 0:
        return math.nan
    return abs(result.estimate - result.null_value) / result.se


def run_diagnostics(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    universe: Sequence[str],
    label_spec: LabelSpec,
    diagnostic_features: Sequence[str],
    provenance: Mapping[str, object],
    started_at: datetime,
    models: Sequence[ModelSpec] | None = None,
    config: DiagnosticsConfig | None = None,
    lockbox_start: object = MODEL_HOLDOUT_START,
) -> DiagnosticsRun:
    config = config or DiagnosticsConfig()
    models = list(models if models is not None else default_models())
    names = [spec.name for spec in models]
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique.")
    for key in ("code", "environment"):
        if key not in provenance:
            raise ValueError(f"provenance must include {key!r}.")
    if any(column in labeled.columns for column in CONTROL_COLUMNS):
        raise ValueError("Pass the frame without control columns; the runner adds them.")

    # 1. lockbox gate
    evidence = lockbox_evidence(labeled, folds, lockbox_start)
    if any(frame_rows_sha256(labeled) != fold.frame_rows_sha256 for fold in folds):
        raise LeakageError("Folds were built from a different frame or row order.")
    frame = add_control_columns(labeled, universe)
    expected = labeled.iloc[np.concatenate([fold.test_rows for fold in folds])]

    # 2. predictions (complete before any scoring)
    specs = {spec.name: spec for spec in models}
    predictions: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict] = {}
    for spec in models:
        result = run_walk_forward(
            frame,
            folds,
            feature_columns=list(spec.feature_columns),
            make_estimator=_factory(spec, config.seed, "main"),
            lockbox_start=lockbox_start,
        )
        _check_complete(spec.name, result.predictions, expected)
        predictions[spec.name] = result.predictions
        audit[spec.name] = _estimator_audit(result.estimators)

    canary_spec = specs.get(config.canary_model)
    unsafe = None
    if canary_spec is not None:
        unsafe = unpurged_reference_predictions(
            frame,
            folds,
            feature_columns=list(canary_spec.feature_columns),
            make_estimator=_factory(canary_spec, config.seed, "unsafe"),
            lockbox_start=lockbox_start,
        )
        _check_complete("canary_unsafe_reference", unsafe, expected)

    anchor = min(fold.test_start for fold in folds)
    permutation_seed = derive_seed(config.seed, "block_permutation")
    permuted: dict[str, list[pd.DataFrame]] = {}
    for name in config.permutation_models:
        if name not in specs:
            continue
        spec = specs[name]
        permuted[name] = []
        for draw in range(config.permutation_draws):
            placebo = block_permuted_features(
                frame,
                spec.feature_columns,
                seed=permutation_seed,
                draw=draw,
                block_length=config.permutation_block_length,
                anchor=anchor,
            )
            result = run_walk_forward(
                placebo,
                folds,
                feature_columns=list(spec.feature_columns),
                make_estimator=_factory(spec, config.seed, f"permutation_{draw}"),
                lockbox_start=lockbox_start,
            )
            _check_complete(f"{name} permutation {draw}", result.predictions, expected)
            permuted[name].append(result.predictions)

    stale: dict[tuple[str, int], pd.DataFrame | str] = {}
    test_rows = np.concatenate([fold.test_rows for fold in folds])
    for name in config.stale_models:
        if name not in specs:
            continue
        spec = specs[name]
        for lag in config.stale_lags:
            placebo = stale_features(frame, spec.feature_columns, lag_sessions=lag)
            values = placebo.iloc[test_rows][list(spec.feature_columns)].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                stale[(name, lag)] = "insufficient_history"
                continue
            result = run_walk_forward(
                placebo,
                folds,
                feature_columns=list(spec.feature_columns),
                make_estimator=_factory(spec, config.seed, f"stale_{lag}"),
                lockbox_start=lockbox_start,
            )
            _check_complete(f"{name} stale {lag}", result.predictions, expected)
            stale[(name, lag)] = result.predictions

    prediction_hashes = {name: _label_free_hash(frame_) for name, frame_ in predictions.items()}
    if unsafe is not None:
        prediction_hashes["canary_unsafe_reference"] = _label_free_hash(unsafe)

    # 3. scoring (labels used only from here on)
    reference = predictions[models[0].name]
    ctx = scoring.build_context(
        reference,
        block_length=config.block_length,
        bootstrap_reps=config.bootstrap_reps,
        seed=derive_seed(config.seed, "bootstrap"),
        hac_lag=config.hac_lag,
        ci_level=config.ci_level,
        min_names=config.min_names,
        decile_bins=config.decile_bins,
        acf_max_lag=config.acf_max_lag,
    )
    baselines = {
        name: predictions[name]
        for name in config.baselines
        if name in predictions and specs[name].output_kind == "forecast"
    }
    metrics: list[dict] = []
    curves: list[dict] = []
    null_draws: list[dict] = []
    scores: dict[str, scoring.ModelScore] = {}
    scored_models = [
        (spec.name, spec.role, spec.output_kind, predictions[spec.name]) for spec in models
    ]
    if unsafe is not None:
        scored_models.append(
            ("canary_unsafe_reference", UNSAFE_ROLE, canary_spec.output_kind, unsafe)
        )
    for name, role, kind, frame_ in scored_models:
        score = scoring.score_model(
            name, role, kind, frame_, ctx, baselines=baselines, horizon=label_spec.horizon
        )
        scores[name] = score
        metrics += score.metrics
        curves += score.curves

    def mean_ic(frame_: pd.DataFrame) -> float:
        by_date = scoring.rank_ic_by_date(frame_.sort_values(["date", "symbol"]), ctx)
        values = by_date["rank_ic"].dropna()
        return float(values.mean()) if len(values) else math.nan

    for name, draws in permuted.items():
        values = []
        for draw, frame_ in enumerate(draws):
            value = mean_ic(frame_)
            values.append(value)
            null_draws.append(
                {
                    "model": name,
                    "null_kind": "block_permutation",
                    "draw": draw,
                    "seed": permutation_seed,
                    "detail": f"block_length={config.permutation_block_length}",
                    "metric": "mean_rank_ic",
                    "value": value,
                }
            )
        observed = scores[name].mean_rank_ic_by("newey_west").estimate
        metrics += scoring.metric_rows(
            name,
            specs[name].role,
            "mean_rank_ic_vs_block_permutation",
            [scoring.compare_to_null(observed, values, "block_permutation")],
        )

    for (name, lag), frame_ in stale.items():
        metric = f"rank_ic_minus_stale_{lag}"
        if isinstance(frame_, str):
            metrics += scoring.metric_rows(
                name, specs[name].role, metric, scoring._unavailable_all(frame_)
            )
            continue
        stale_ic = scoring.ic_series(
            scoring.rank_ic_by_date(frame_.sort_values(["date", "symbol"]), ctx), ctx
        )
        model_ic = scoring.ic_series(scores[name].rank_ic_by_date, ctx)
        difference = model_ic - stale_ic
        metrics += scoring.metric_rows(
            name, specs[name].role, metric, scoring.mean_inferences(difference, ctx)
        )
        metrics += scoring.metric_rows(
            name,
            specs[name].role,
            f"mean_rank_ic_stale_{lag}",
            [scoring.point(float(np.nanmean(stale_ic)), n=int(np.isfinite(stale_ic).sum()))],
        )

    orderings, ordering_reason = scoring.static_ordering_ics(
        reference,
        ctx,
        max_symbols=config.max_static_ordering_symbols,
    )
    if ordering_reason is None:
        for row in orderings.itertuples(index=False):
            null_draws.append(
                {
                    "model": "__labels__",
                    "null_kind": "static_ordering",
                    "draw": int(row.draw),
                    "seed": None,
                    "detail": row.detail,
                    "metric": "mean_rank_ic",
                    "value": float(row.value),
                }
            )
    for name, score in scores.items():
        role = specs[name].role if name in specs else UNSAFE_ROLE
        if ordering_reason is None:
            observed = score.mean_rank_ic_by("newey_west").estimate
            comparison = scoring.compare_to_null(
                observed, orderings["value"].tolist(), "static_ordering", exhaustive=True
            )
            share = (
                float((orderings["value"] >= observed).mean())
                if math.isfinite(observed)
                else math.nan
            )
            metrics += scoring.metric_rows(
                name,
                role,
                "static_ordering_share_at_least",
                [scoring.point(share, n=len(orderings))],
            )
        else:
            comparison = inf.unavailable("null_enumeration", ordering_reason)
        metrics += scoring.metric_rows(name, role, "mean_rank_ic_vs_static_ordering", [comparison])

    feature_stats, feature_pairs = training_feature_diagnostics(frame, folds, diagnostic_features)
    drift = evaluation_feature_drift(frame, folds, diagnostic_features)
    feature_stats = pd.concat([feature_stats, drift], ignore_index=True)

    names_per_date = reference.groupby("date")["symbol"].size()
    calibration = {}
    for signal in config.calibration_signals:
        series = inf.simulate_null_ic_series(
            n_dates=len(ctx.sessions),
            n_names=int(names_per_date.mode().iloc[0]),
            horizon=label_spec.horizon,
            entry_lag=label_spec.entry_lag,
            signal=signal,
            reps=config.calibration_reps,
            seed=derive_seed(config.seed, "calibration", signal),
            block_length=config.block_length,
        )
        calibration[signal] = inf.null_rejection_rates(
            series,
            lag=config.hac_lag,
            block_length=config.block_length,
            bootstrap_reps=config.calibration_bootstrap_reps,
            seed=derive_seed(config.seed, "calibration_bootstrap", signal),
        )

    # 4. checks, tables, record
    checks = _checks(config, specs, scores, predictions, expected, evidence, calibration)
    folds_table = describe_folds(labeled, folds, feature_columns=list(diagnostic_features))
    prediction_frames = [
        predictions[spec.name].assign(model=spec.name, role=spec.role) for spec in models
    ]
    if unsafe is not None:
        prediction_frames.append(unsafe.assign(model="canary_unsafe_reference", role=UNSAFE_ROLE))
    input_columns = [
        "date",
        "symbol",
        *[c for c in diagnostic_features if c not in ("date", "symbol")],
        *LABEL_COLUMNS,
    ]
    tables = {
        "inputs": contract.conform("inputs", labeled.loc[:, input_columns]),
        "folds": contract.conform("folds", folds_table),
        "predictions": contract.conform(
            "predictions", pd.concat(prediction_frames, ignore_index=True)
        ),
        "metrics": contract.conform("metrics", pd.DataFrame(metrics)),
        "curves": contract.conform("curves", pd.DataFrame(curves)),
        "null_draws": contract.conform(
            "null_draws", pd.DataFrame(null_draws, columns=list(contract.SCHEMAS["null_draws"]))
        ),
        "feature_stats": contract.conform("feature_stats", feature_stats),
        "feature_pairs": contract.conform("feature_pairs", feature_pairs),
        "checks": contract.conform("checks", pd.DataFrame(checks)),
    }

    spec_record = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "output_schema_version": contract.OUTPUT_SCHEMA_VERSION,
        "code": provenance["code"],
        "environment": provenance["environment"],
        "inputs": {
            **dict(provenance.get("inputs", {})),
            "labeled_frame_content_sha256": frame_content_sha256(
                labeled, [c for c in input_columns if c not in ("date", "symbol")]
            ),
            "labeled_frame_rows_sha256": frame_rows_sha256(labeled),
        },
        "label": {
            "label_id": label_id(label_spec),
            "label_spec": asdict(label_spec),
            "target_column": TARGET_RETURN_COLUMN,
            "signal_timing_convention": DECISION_CONVENTION,
            "signal_timing_convention_sha256": hashlib.sha256(
                DECISION_CONVENTION.encode("utf-8")
            ).hexdigest(),
        },
        "lockbox": {
            "holdout_start": pd.Timestamp(lockbox_start).isoformat(),
            "mode": "development",
            "truncation": dict(provenance.get("truncation", {})),
        },
        "universe": {
            "symbols": sorted(universe),
            "symbol_codes": symbol_codes(universe),
            "excluded_symbols": dict(provenance.get("truncation", {}).get("excluded_symbols", {})),
            "note": provenance.get("universe_note"),
        },
        "features": {"diagnostic_features": list(diagnostic_features)},
        "folds": {
            "params": dict(provenance.get("fold_params", {})),
            "n_folds": len(folds),
            "fold_table_sha256": table_sha256(tables["folds"]),
        },
        "models": [spec.describe() for spec in models],
        "registered_variants": sorted(spec.describe()["variant_id"] for spec in models),
        "config": asdict(config),
        "seed_rule": SEED_RULE,
    }
    spec_sha = sha256_json(spec_record)
    identifier = run_id(spec_sha, started_at)
    tables_sha = {name: table_sha256(table) for name, table in tables.items()}
    record = {
        "run_id": identifier,
        "spec_sha256": spec_sha,
        "started_at": pd.Timestamp(started_at).isoformat(),
        "spec": spec_record,
        "lockbox_evidence": {
            **evidence,
            "holdout_rows_loaded": provenance.get("truncation", {}).get("holdout_rows_loaded"),
            "adjusted_price_disclosure": provenance.get("adjusted_price_disclosure"),
        },
        "estimator_audit": audit,
        "inference_calibration": {
            "n_dates": len(ctx.sessions),
            "reps": config.calibration_reps,
            "alpha": 0.05,
            "rates": calibration,
        },
        "outputs": {
            "predictions_sha256_label_free": prediction_hashes,
            "labels_sha256": table_sha256(expected.loc[:, ["date", "symbol", *LABEL_COLUMNS]]),
            "tables_sha256": tables_sha,
            "results_sha256": sha256_json(tables_sha),
        },
        "checks_summary": _checks_summary(tables["checks"]),
    }
    return DiagnosticsRun(identifier, record, tables)


def _checks(config, specs, scores, predictions, expected, evidence, calibration) -> list[dict]:
    rows = [
        _check(
            "lockbox_closed",
            None,
            "leakage",
            "held_out_rows_total",
            evidence["held_out_rows_total"],
            0,
            "==",
            f"max_date={evidence['max_date']}; max_target_end_date={evidence['max_target_end_date']}",
        ),
        _check(
            "hac_lag_meets_research_minimum",
            None,
            "instrument",
            "hac_lag",
            config.hac_lag,
            config.research_minimum_hac_lag,
            ">=",
        ),
    ]
    for name, frame_ in predictions.items():
        rows.append(
            _check(
                "predictions_complete", name, "leakage", "n_rows", len(frame_), len(expected), "=="
            )
        )
    for name, spec in specs.items():
        ic = scores[name].mean_rank_ic_by(DECISION_METHOD)
        if spec.role == "null":
            if ic.status == "ok":
                rows.append(
                    _check(
                        "null_shows_no_skill",
                        name,
                        "instrument",
                        "mean_rank_ic_abs_t_fold_block",
                        _abs_t(ic),
                        config.no_skill_abs_t,
                        "<",
                    )
                )
            else:
                rows.append(
                    _check(
                        "null_rank_ic_undefined",
                        name,
                        "instrument",
                        "n_dates_rank_ic_defined",
                        scores[name].n_dates_defined,
                        0.0,
                        "==",
                        ic.reason,
                    )
                )
        elif spec.role == "canary":
            rows.append(
                _check(
                    "canary_shows_no_skill_through_harness",
                    name,
                    "leakage",
                    "mean_rank_ic_abs_t_fold_block",
                    _abs_t(ic),
                    config.no_skill_abs_t,
                    "<",
                )
            )
        elif spec.role == "control":
            t = _abs_t(ic)
            signed = ic.estimate / ic.se if ic.status == "ok" and ic.se > 0 else math.nan
            rows.append(
                _check(
                    "control_apparent_skill",
                    name,
                    "research_warning",
                    "mean_rank_ic_t_fold_block",
                    signed,
                    2.0,
                    ">",
                    "passed=True means apparent skill: on a hindsight-selected universe "
                    "read it as a selection or benchmark effect, not as evidence"
                    if math.isfinite(t)
                    else ic.reason,
                )
            )
    unsafe = scores.get("canary_unsafe_reference")
    if unsafe is not None:
        ic = unsafe.mean_rank_ic_by(DECISION_METHOD)
        signed = ic.estimate / ic.se if ic.status == "ok" and ic.se > 0 else math.nan
        rows.append(
            _check(
                "canary_detects_unpurged_leakage",
                "canary_unsafe_reference",
                "instrument",
                "mean_rank_ic_t_fold_block",
                signed,
                config.canary_unsafe_min_t,
                ">",
                "positive control: the unpurged memorizer must show spurious skill",
            )
        )
    for signal, rates in calibration.items():
        for method in ("newey_west", "fold_block_t", "circular_block_bootstrap"):
            rows.append(
                _check(
                    "simulated_false_positive_rate",
                    None,
                    "instrument" if method == DECISION_METHOD else "info",
                    f"{method}:{signal}",
                    rates[method],
                    config.max_false_positive_rate,
                    "<=",
                    f"no-skill simulation at alpha=0.05; iid SE rate {rates['iid']:.3f} for contrast",
                )
            )
    return rows


def _checks_summary(checks: pd.DataFrame) -> dict[str, int]:
    failed = checks["passed"].eq(False).fillna(False)
    leakage = checks["severity"].eq("leakage")
    warnings = checks["severity"].eq("research_warning") & checks["passed"].eq(True).fillna(False)
    return {
        "leakage_checks_failed": int((failed & leakage).sum()),
        "instrument_checks_failed": int((failed & checks["severity"].eq("instrument")).sum()),
        "research_warnings": int(warnings.sum()),
    }
