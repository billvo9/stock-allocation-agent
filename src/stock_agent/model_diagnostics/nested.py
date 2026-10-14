"""
Nested purged temporal selection for T4's linear models (plumbing).

For each outer fold, hyperparameters are chosen only from that fold's
training rows:

    inner folds   make_expanding_folds(frame, last `n_windows` blocks of
                  `block_sessions` sessions before the outer test_start,
                  holdout_start = outer test_start, train_start = outer
                  train_start). The purge rule is the outer one: an inner
                  training row's label ends before its inner test start, and
                  an inner test row whose label ends at or after the outer
                  test start is held out. Forward-only folds need no embargo
                  (folds.py). Inner rows are asserted to lie inside the outer
                  training rows.
    grid points   each runs once through the validation harness on the inner
                  folds; the owner's linear.select_hyperparameters picks one.
    outer refit   ONE harness call over all outer folds, whose estimator
                  factory hands each fold its chosen scalars in fold order, so
                  the harness's cross-fold guards stay in force.

Inner folds depend only on dates and label timing, never on feature values,
so they are built once per outer fold and reused for every grid point,
permutation draw and stale-feature run. Grids, criteria and the inner design
live in each spec's params (part of its variant id); estimators receive only
scalars.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stock_agent.features.training import TARGET_END_COLUMN, LabelSpec
from stock_agent.model_diagnostics import contract, linear
from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.folds import (
    WalkForwardFold,
    make_expanding_folds,
    make_test_windows,
)
from stock_agent.model_validation.harness import run_walk_forward

# Params that describe a spec (and enter its variant id) but are not
# estimator arguments.
RECORD_ONLY_PARAMS = ("selection", "preprocessing", "label_id")
PREPROCESSING = "median impute, then standardize (population std); training rows only, per fit"
CRITERION = "mean over inner test dates of the per-date cross-sectional MSE (lower is better)"
TIE_RULE = "larger penalty, then larger l1_ratio"
SOLVER = {"tol": 1e-10, "max_iter": 100_000}

# Pre-registered, dimensionless grids (see linear.LinearForecast for the scales).
RIDGE_GRID = tuple({"penalty": c} for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
L1_RATIOS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
LASSO_GRID = tuple({"penalty": r} for r in L1_RATIOS)
ELASTIC_NET_GRID = tuple(
    {"penalty": r, "l1_ratio": mix} for mix in (0.2, 0.5, 0.8) for r in L1_RATIOS
)


@dataclass(frozen=True)
class InnerDesign:
    block_sessions: int = 63
    n_windows: int = 4
    min_train_sessions: int = 252


DEFAULT_DESIGN = InnerDesign()


def linear_models(
    label_spec: LabelSpec,
    feature_columns: Sequence[str],
    design: InnerDesign = DEFAULT_DESIGN,
    *,
    estimator: type = linear.LinearForecast,
    include_ols: bool = True,
) -> list:
    """Ridge, Lasso and Elastic Net candidates (nested selection) and the OLS diagnostic."""

    from stock_agent.model_diagnostics.runner import ModelSpec  # avoid an import cycle

    features = tuple(feature_columns)
    inner = asdict(design)
    common = {"preprocessing": PREPROCESSING, "label_id": label_spec.label_id, **SOLVER}

    def selection(grid):
        return {
            "grid": [dict(point) for point in grid],
            "criterion": CRITERION,
            "tie_rule": TIE_RULE,
            "inner": inner,
        }

    specs = [
        ModelSpec(
            "ridge",
            "candidate",
            estimator,
            features,
            params={"method": "ridge", **common, "selection": selection(RIDGE_GRID)},
            note="Ridge; penalty chosen by nested purged temporal validation",
        ),
        ModelSpec(
            "lasso",
            "candidate",
            estimator,
            features,
            params={"method": "lasso", **common, "selection": selection(LASSO_GRID)},
            note="Lasso; penalty chosen by nested purged temporal validation",
        ),
        ModelSpec(
            "elastic_net",
            "candidate",
            estimator,
            features,
            params={"method": "elastic_net", **common, "selection": selection(ELASTIC_NET_GRID)},
            note="Elastic Net; penalty and l1_ratio chosen by nested purged validation",
        ),
    ]
    if include_ols:
        specs.append(
            ModelSpec(
                "ols",
                contract.DIAGNOSTIC_ROLE,
                estimator,
                features,
                params={"method": "ols", **common},
                note="unpenalized OLS; nests the zero forecast (Clark-West comparator)",
            )
        )
    return specs


def comparison_settings(label_spec: LabelSpec, *, with_linear: bool) -> dict[str, tuple]:
    """
    Comparators and Clark-West pairs that fit the label. Under an excess label
    the pooled training mean is identically zero, so zero is the null; OLS
    nests zero (and, under a raw label, the intercept-only pooled mean).
    """

    excess = label_spec.benchmark is not None
    baselines = (
        ("zero", "per_symbol_mean") if excess else ("zero", "pooled_mean", "per_symbol_mean")
    )
    if not with_linear:
        return {"baselines": baselines, "nested_comparisons": ()}
    nested_pairs = (("ols", "zero"),) if excess else (("ols", "zero"), ("ols", "pooled_mean"))
    return {"baselines": (*baselines, "ols"), "nested_comparisons": nested_pairs}


def is_nested(spec) -> bool:
    return "selection" in spec.params


def estimator_kwargs(params: Mapping[str, object]) -> dict[str, object]:
    """Fresh scalar-only kwargs: never a shared grid, list or dict (harness rule)."""

    return {k: v for k, v in params.items() if k not in RECORD_ONLY_PARAMS}


def build_inner_folds(
    frame: pd.DataFrame, outer: WalkForwardFold, design: InnerDesign
) -> tuple[WalkForwardFold, ...]:
    """Inner purged forward folds inside one outer fold's training rows."""

    dates = pd.Series(frame["date"].unique()).sort_values()
    sessions = dates[dates < outer.test_start]
    if outer.train_start is not None:
        sessions = sessions[sessions >= outer.train_start]
    needed = design.min_train_sessions + design.n_windows * design.block_sessions
    if len(sessions) < needed:
        raise ValueError(
            f"Fold {outer.fold_id}: {len(sessions)} training sessions; nested selection needs "
            f"{needed} ({design.min_train_sessions} + {design.n_windows} x "
            f"{design.block_sessions})."
        )
    start = sessions.iloc[-design.n_windows * design.block_sessions]
    windows = make_test_windows(
        sessions, start=start, end=outer.test_start, block_sessions=design.block_sessions
    )
    inner = make_expanding_folds(
        frame, windows, holdout_start=outer.test_start, train_start=outer.train_start
    )
    verify_inner_folds(frame, outer, inner)
    return inner


def verify_inner_folds(
    frame: pd.DataFrame, outer: WalkForwardFold, inner: Sequence[WalkForwardFold]
) -> None:
    """Every inner row is an outer training row whose label ends before the outer test."""

    for fold in inner:
        rows = np.concatenate([fold.train_rows, fold.test_rows])
        if not np.isin(rows, outer.train_rows).all():
            raise LeakageError(f"Outer fold {outer.fold_id}: an inner row is not a training row.")
        ends = pd.to_datetime(frame[TARGET_END_COLUMN].iloc[rows], utc=True)
        if len(ends) and not (ends < outer.test_start).all():
            raise LeakageError(
                f"Outer fold {outer.fold_id}: an inner label ends at or after the outer test start."
            )


def inner_bounds(frame: pd.DataFrame, inner: Sequence[WalkForwardFold]) -> dict[str, object]:
    rows = np.concatenate([np.concatenate([f.train_rows, f.test_rows]) for f in inner])
    ends = pd.to_datetime(frame[TARGET_END_COLUMN].iloc[rows], utc=True)
    return {
        "inner_test_start": min(f.test_start for f in inner),
        "inner_test_end_exclusive": max(f.test_end for f in inner),
        "max_inner_label_end": ends.max(),
        "n_inner_folds": len(inner),
    }


@dataclass
class NestedResult:
    predictions: pd.DataFrame
    estimators: tuple
    inner_estimators: tuple
    selection_rows: list[dict]
    coefficient_rows: list[dict]


def _harness(frame, folds, spec, make_estimator, lockbox_start):
    with threadpool_limits(limits=1):  # reproducible floating-point sums
        return run_walk_forward(
            frame,
            folds,
            feature_columns=list(spec.feature_columns),
            make_estimator=make_estimator,
            lockbox_start=lockbox_start,
        )


def run_nested(
    frame: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    spec,
    inner_folds: Mapping[int, tuple[WalkForwardFold, ...]],
    *,
    selector: Callable[..., linear.Selection],
    lockbox_start: object,
    record: bool = True,
) -> NestedResult:
    """Select per outer fold on its inner folds, then refit all outer folds in one call."""

    base = estimator_kwargs(spec.params)
    grid = spec.params["selection"]["grid"]
    chosen_points, selection_rows, inner_estimators = [], [], []
    for outer in folds:
        inner = inner_folds[outer.fold_id]
        outputs = []
        for point in grid:

            def make(point=point):
                return spec.estimator(**{**base, **point})

            result = _harness(frame, inner, spec, make, lockbox_start)
            outputs.append(result.predictions)
            inner_estimators.extend(result.estimators)
        chosen = selector(grid, outputs)
        if not isinstance(chosen, linear.Selection) or len(chosen.scores) != len(grid):
            raise ValueError(f"{spec.name}: the selector must score every grid point.")
        chosen_points.append(dict(grid[chosen.chosen]))
        if record:
            bounds = inner_bounds(frame, inner)
            for index, point in enumerate(grid):
                selection_rows.append(
                    {
                        "model": spec.name,
                        "role": spec.role,
                        "scope": "training",
                        "fold_id": outer.fold_id,
                        "penalty": float(point["penalty"]),
                        "l1_ratio": float(point.get("l1_ratio", np.nan)),
                        "inner_score": float(chosen.scores[index]),
                        "n_inner_dates": int(chosen.n_dates[index]),
                        **bounds,
                        "chosen": index == chosen.chosen,
                        "status": "ok",
                        "reason": None,
                    }
                )

    calls = itertools.count()

    def make_outer():
        index = next(calls)
        return spec.estimator(**{**base, **chosen_points[index]})

    result = _harness(frame, folds, spec, make_outer, lockbox_start)
    if next(calls) != len(folds):
        raise RuntimeError(f"{spec.name}: the outer refit built one estimator per fold.")
    # Only linear fits expose coefficients; other nested models store selection rows only.
    coefficients = (
        coefficient_rows(spec, folds, result.estimators) if record and is_linear(spec) else []
    )
    return NestedResult(
        result.predictions,
        result.estimators,
        tuple(inner_estimators),
        selection_rows,
        coefficients,
    )


def coefficient_rows(spec, folds: Sequence[WalkForwardFold], estimators: Sequence) -> list[dict]:
    """Standardized coefficients and fitted transform parameters per fold and feature."""

    rows = []
    features = list(spec.feature_columns)
    for fold, estimator in zip(folds, estimators, strict=True):
        try:
            values = [
                np.asarray(getattr(estimator, name), dtype=np.float64)
                for name in ("coef_", "medians_", "means_", "scales_")
            ]
            intercept = float(estimator.intercept_)
            alpha = float(estimator.alpha_)
        except AttributeError as error:
            raise ValueError(f"{spec.name}: a fitted LinearForecast must expose {error}.") from None
        if any(value.shape != (len(features),) for value in values):
            raise ValueError(f"{spec.name}: coefficient arrays must have one entry per feature.")
        converged = linear.has_converged(estimator)
        coef, medians, means, scales = values
        common = {
            "model": spec.name,
            "role": spec.role,
            "scope": "training",
            "fold_id": fold.fold_id,
            "alpha": alpha,
            "converged": converged,
        }
        for index, feature in enumerate(features):
            rows.append(
                {
                    **common,
                    "feature": feature,
                    "coefficient": coef[index],
                    "median": medians[index],
                    "mean": means[index],
                    "scale": scales[index],
                }
            )
        rows.append(
            {
                **common,
                "feature": "__intercept__",
                "coefficient": intercept,
                "median": np.nan,
                "mean": np.nan,
                "scale": np.nan,
            }
        )
    return rows


def is_linear(spec) -> bool:
    """A spec whose fitted estimators expose coefficients (recorded per fold)."""

    return isinstance(spec.estimator, type) and issubclass(spec.estimator, linear.LinearForecast)


def empty_table(name: str) -> pd.DataFrame:
    return contract.empty(name)
