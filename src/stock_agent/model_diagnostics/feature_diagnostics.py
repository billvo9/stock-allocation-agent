"""
Feature and data diagnostics, tagged by the data they read.

TRAINING diagnostics (scope "training") read one fold's training rows only,
through fold.train_rows. They describe exactly the data a fold's fit sees
and are the only diagnostics allowed to inform choices inside that fold.

EVALUATION diagnostics (scope "evaluation") compare a fold's test-row
features with its training rows (drift). They read features only, never
labels, and are for monitoring: nothing in a run feeds them back into
fitting, feature lists, or preprocessing.

Diagnostics never modify, impute, or drop data. Every function copies its
inputs and returns tables; undefined statistics carry a status and reason
(single_feature, too_few_rows, constant_feature, singular, zero_mad, ...)
instead of a misleading number.

Rows are read only through fold row positions, so a frame that still held
lockbox rows would not reach these statistics through a development fold.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_agent.model_validation.audit import datetime_ns
from stock_agent.model_validation.folds import WalkForwardFold

TRAINING_SCOPE = "training"
EVALUATION_SCOPE = "evaluation"
DESIGN = "__design__"
STAT_COLUMNS = ["scope", "fold_id", "feature", "statistic", "basis", "value", "status", "reason"]
PAIR_COLUMNS = [
    "scope",
    "fold_id",
    "feature_a",
    "feature_b",
    "statistic",
    "value",
    "n",
    "status",
    "reason",
]
QUANTILES = {
    "q01": 0.01,
    "q05": 0.05,
    "q25": 0.25,
    "median": 0.5,
    "q75": 0.75,
    "q95": 0.95,
    "q99": 0.99,
}
ROBUST_SCALE = 1.4826  # MAD -> standard deviation under normality
UNIVARIATE_FINITE_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    *QUANTILES,
    "mad",
    "n_extreme_robust_z",
    "n_unique",
    "top_value_share",
    "near_constant",
    "skew",
    "excess_kurtosis",
    "date_variance_share",
)


def _stat(scope, fold_id, feature, statistic, value, *, basis, status="ok", reason=None) -> dict:
    value = float(value)
    if status == "ok" and not np.isfinite(value):
        status, reason = "unavailable", reason or "undefined"
    return {
        "scope": scope,
        "fold_id": int(fold_id),
        "feature": feature,
        "statistic": statistic,
        "basis": basis,
        "value": value if status == "ok" else np.nan,
        "status": status,
        "reason": reason,
    }


def _univariate(
    values: np.ndarray, dates: np.ndarray, *, extreme_z: float, near_constant_share: float
):
    """Statistic -> (value, status, reason) for one feature on one fold's training rows."""

    n = len(values)
    missing = np.isnan(values)
    nonfinite = np.isinf(values)
    finite = values[np.isfinite(values)]
    out: dict[str, tuple[float, str, str | None]] = {
        "n": (n, "ok", None),
        "n_missing": (int(missing.sum()), "ok", None),
        "n_nonfinite": (int(nonfinite.sum()), "ok", None),
        "missing_rate": (missing.mean() if n else np.nan, "ok", None),
    }
    if finite.size == 0:
        for name in UNIVARIATE_FINITE_STATISTICS:
            out[name] = (np.nan, "unavailable", "no_finite_values")
        return out
    mean = finite.mean()
    centered = finite - mean
    m2 = (centered**2).mean()
    out["mean"] = (mean, "ok", None)
    out["std"] = (finite.std(ddof=1) if finite.size > 1 else np.nan, "ok", None)
    out["min"] = (finite.min(), "ok", None)
    out["max"] = (finite.max(), "ok", None)
    quantiles = np.quantile(finite, list(QUANTILES.values()), method="linear")
    out.update({name: (q, "ok", None) for name, q in zip(QUANTILES, quantiles, strict=True)})
    median = quantiles[3]
    mad = np.median(np.abs(finite - median))
    out["mad"] = (mad, "ok", None)
    if mad > 0:
        z = np.abs(finite - median) / (ROBUST_SCALE * mad)
        out["n_extreme_robust_z"] = (int((z > extreme_z).sum()), "ok", None)
    else:
        out["n_extreme_robust_z"] = (np.nan, "unavailable", "zero_mad")
    uniques, counts = np.unique(finite, return_counts=True)
    top_share = counts.max() / finite.size
    out["n_unique"] = (len(uniques), "ok", None)
    out["top_value_share"] = (top_share, "ok", None)
    tiny = m2 <= (10.0 * np.finfo(np.float64).eps * abs(mean)) ** 2
    out["near_constant"] = (
        float(tiny or m2 == 0.0 or top_share >= near_constant_share),
        "ok",
        None,
    )
    if m2 > 0 and not tiny:
        out["skew"] = ((centered**3).mean() / m2**1.5, "ok", None)
        out["excess_kurtosis"] = ((centered**4).mean() / m2**2 - 3.0, "ok", None)
        finite_dates = dates[np.isfinite(values)]
        _, inverse = np.unique(finite_dates, return_inverse=True)
        date_means = np.bincount(inverse, weights=finite) / np.bincount(inverse)
        between = ((date_means[inverse] - mean) ** 2).sum()
        out["date_variance_share"] = (between / (centered**2).sum(), "ok", None)
    else:
        for name in ("skew", "excess_kurtosis", "date_variance_share"):
            out[name] = (np.nan, "unavailable", "constant_feature")
    return out


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    rounding = 10.0 * np.finfo(np.float64).eps * np.abs(mean)
    constant = (scale == 0.0) | (scale <= rounding)
    safe = np.where(constant, 1.0, scale)
    return (matrix - mean) / safe, constant


def _average_ranks(matrix: np.ndarray) -> np.ndarray:
    return pd.DataFrame(matrix).rank(method="average").to_numpy()


def _pairs(scope, fold_id, features, complete: np.ndarray) -> list[dict]:
    rows = []
    n = len(complete)
    for statistic, data in (("pearson", complete), ("spearman", _average_ranks(complete))):
        z, constant = _standardize(data) if n else (data, np.ones(len(features), bool))
        corr = z.T @ z / n if n else np.full((len(features),) * 2, np.nan)
        for i, a in enumerate(features):
            for j in range(i + 1, len(features)):
                reason = None
                if n < 3:
                    reason = "too_few_rows"
                elif constant[i] or constant[j]:
                    reason = "constant_feature"
                rows.append(
                    {
                        "scope": scope,
                        "fold_id": int(fold_id),
                        "feature_a": a,
                        "feature_b": features[j],
                        "statistic": statistic,
                        "value": float(corr[i, j]) if reason is None else np.nan,
                        "n": n,
                        "status": "ok" if reason is None else "unavailable",
                        "reason": reason,
                    }
                )
    return rows


def _collinearity(fold_id, features, complete: np.ndarray) -> list[dict]:
    """
    VIF per feature and condition number of the standardized design.

    The design is centered and scaled (no intercept column), so kappa is
    sqrt(max/min eigenvalue of the training correlation matrix). Belsley's
    rule of thumb (kappa > 30) is stated for an uncentered design with an
    intercept and reads higher; treat thresholds here as guide lines only.
    """

    basis = "complete_cases_standardized_with_fold_training_mean_std"
    n, p = complete.shape
    rows = [_stat(TRAINING_SCOPE, fold_id, DESIGN, "n_complete_cases", n, basis=basis)]

    def all_unavailable(reason: str) -> list[dict]:
        out = [
            _stat(
                TRAINING_SCOPE,
                fold_id,
                f,
                "vif",
                np.nan,
                basis=basis,
                status="unavailable",
                reason=reason,
            )
            for f in features
        ]
        out.append(
            _stat(
                TRAINING_SCOPE,
                fold_id,
                DESIGN,
                "condition_number",
                np.nan,
                basis=basis,
                status="unavailable",
                reason=reason,
            )
        )
        return out

    if p < 2:
        return rows + all_unavailable("single_feature")
    if n <= p + 1:
        return rows + all_unavailable("too_few_rows")
    z, constant = _standardize(complete)
    if constant.any():
        out = all_unavailable("constant_feature")
        for row, is_constant in zip(out, [*constant, False], strict=True):
            if not is_constant and row["feature"] != DESIGN:
                row["reason"] = "design_has_constant_feature"
        return rows + out
    singular_values = np.linalg.svd(z, compute_uv=False)
    tolerance = singular_values.max() * max(n, p) * np.finfo(np.float64).eps
    if singular_values.min() <= tolerance:
        return rows + all_unavailable("singular")
    correlation = z.T @ z / n
    vif = np.diag(np.linalg.inv(correlation))
    rows += [
        _stat(TRAINING_SCOPE, fold_id, f, "vif", v, basis=basis)
        for f, v in zip(features, vif, strict=True)
    ]
    rows.append(
        _stat(
            TRAINING_SCOPE,
            fold_id,
            DESIGN,
            "condition_number",
            singular_values.max() / singular_values.min(),
            basis=basis,
        )
    )
    eigenvalues = np.sort(np.linalg.eigvalsh(correlation))[::-1]
    rows += [
        _stat(
            TRAINING_SCOPE, fold_id, DESIGN, f"correlation_eigenvalue_{k + 1}", value, basis=basis
        )
        for k, value in enumerate(eigenvalues)
    ]
    return rows


def training_feature_diagnostics(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    feature_columns: Sequence[str],
    *,
    extreme_z: float = 5.0,
    near_constant_share: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(feature_stats, feature_pairs) from each fold's training rows only."""

    features = list(feature_columns)
    stats, pairs = [], []
    for fold in folds:
        train = labeled.iloc[fold.train_rows]
        matrix = train.loc[:, features].to_numpy(dtype=np.float64, copy=True)
        dates = datetime_ns(train["date"])
        for column, feature in enumerate(features):
            summary = _univariate(
                matrix[:, column],
                dates,
                extreme_z=extreme_z,
                near_constant_share=near_constant_share,
            )
            stats += [
                _stat(
                    TRAINING_SCOPE,
                    fold.fold_id,
                    feature,
                    name,
                    value,
                    basis="training_rows",
                    status=status,
                    reason=reason,
                )
                for name, (value, status, reason) in summary.items()
            ]
        complete = matrix[np.isfinite(matrix).all(axis=1)]
        stats += _collinearity(fold.fold_id, features, complete)
        pairs += _pairs(TRAINING_SCOPE, fold.fold_id, features, complete)
    return pd.DataFrame(stats, columns=STAT_COLUMNS), pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _psi(train: np.ndarray, test: np.ndarray, *, bins: int = 10, floor: float = 1e-6) -> float:
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)[1:-1], method="linear"))
    train_share = np.bincount(np.searchsorted(edges, train, side="right"), minlength=len(edges) + 1)
    test_share = np.bincount(np.searchsorted(edges, test, side="right"), minlength=len(edges) + 1)
    p = np.maximum(train_share / train.size, floor)
    q = np.maximum(test_share / test.size, floor)
    return float(((q - p) * np.log(q / p)).sum())


def _ks(train: np.ndarray, test: np.ndarray) -> float:
    grid = np.union1d(train, test)
    a = np.searchsorted(np.sort(train), grid, side="right") / train.size
    b = np.searchsorted(np.sort(test), grid, side="right") / test.size
    return float(np.abs(a - b).max())


def evaluation_feature_drift(
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Train-vs-test drift per fold and feature: PSI on training-decile bins,
    Kolmogorov-Smirnov distance, standardized mean difference (training
    std), and the test rows' missing and non-finite counts. Distances only;
    no p-values (the rows are dependent and the guide lines uncalibrated).
    """

    features = list(feature_columns)
    basis = "test_rows_vs_training_rows"
    rows = []
    for fold in folds:
        train_matrix = labeled.iloc[fold.train_rows][features].to_numpy(dtype=np.float64, copy=True)
        test_matrix = labeled.iloc[fold.test_rows][features].to_numpy(dtype=np.float64, copy=True)
        for column, feature in enumerate(features):
            train_values = train_matrix[:, column]
            test_values = test_matrix[:, column]
            a = train_values[np.isfinite(train_values)]
            b = test_values[np.isfinite(test_values)]
            rows.append(
                _stat(
                    EVALUATION_SCOPE,
                    fold.fold_id,
                    feature,
                    "test_missing_rate",
                    np.isnan(test_values).mean(),
                    basis=basis,
                )
            )
            rows.append(
                _stat(
                    EVALUATION_SCOPE,
                    fold.fold_id,
                    feature,
                    "test_n_nonfinite",
                    np.isinf(test_values).sum(),
                    basis=basis,
                )
            )
            if a.size < 2 or b.size < 1:
                for name in ("psi", "ks_distance", "standardized_mean_difference"):
                    rows.append(
                        _stat(
                            EVALUATION_SCOPE,
                            fold.fold_id,
                            feature,
                            name,
                            np.nan,
                            basis=basis,
                            status="unavailable",
                            reason="too_few_rows",
                        )
                    )
                continue
            sd = a.std(ddof=1)
            constant = not sd > 10.0 * np.finfo(np.float64).eps * abs(a.mean())
            reason = "constant_training_feature" if constant else None
            rows.append(
                _stat(
                    EVALUATION_SCOPE,
                    fold.fold_id,
                    feature,
                    "psi",
                    np.nan if constant else _psi(a, b),
                    basis=basis,
                    status="unavailable" if constant else "ok",
                    reason=reason,
                )
            )
            rows.append(
                _stat(
                    EVALUATION_SCOPE, fold.fold_id, feature, "ks_distance", _ks(a, b), basis=basis
                )
            )
            rows.append(
                _stat(
                    EVALUATION_SCOPE,
                    fold.fold_id,
                    feature,
                    "standardized_mean_difference",
                    np.nan if constant else (b.mean() - a.mean()) / sd,
                    basis=basis,
                    status="unavailable" if constant else "ok",
                    reason=reason,
                )
            )
    return pd.DataFrame(rows, columns=STAT_COLUMNS)
