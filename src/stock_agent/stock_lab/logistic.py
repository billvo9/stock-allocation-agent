"""
M1, the regularized multinomial logistic challenger (scikit-learn; never
loaded by the Lab's page, which reads stored results only).

One model per target (d+1..d+5 and the five-session total), fitted on a
pooled panel of watched stocks with volatility-normalized inputs, so one
coefficient means the same thing for a calm and a volatile stock:

    M1_INPUTS   pre-registered, interpretable, past-only (features.py), plus
                B3's log-odds (Decrease vs Stable, Increase vs Stable) for the
                same target, so M1 starts from the volatility baseline
    stock       one indicator per stock, penalized like every coefficient:
                partial pooling of stock intercepts toward the panel (an L2
                penalty is a Gaussian prior centred on zero)

Preprocessing is fitted on training rows only (median imputation, then
standardization); the penalty C comes from the pre-registered grid C_GRID,
chosen on the last INNER_SESSIONS sessions of the training window (purged:
the inner fit uses rows whose targets end before the inner window opens)
with the paired one-standard-error rule toward the smaller C (more
shrinkage; scores.one_se_choice).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stock_agent.stock_lab.scores import one_se_choice, rps

C_GRID = (0.01, 0.1, 1.0)
INNER_SESSIONS = 504
MAX_ITER = 1000
M1_INPUTS = (
    "log_band_over_vol",
    "signed_run_capped",
    "ret_1_z",
    "ret_5_z",
    "ret_20_z",
    "ma_gap_20_z",
    "drawdown_60",
    "bounce_20",
    "vol_ratio",
    "log_rel_dollar_volume",
    "stable_count_20",
)


def model_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """M1_INPUTS computed from the panel's feature columns (all past-only)."""

    sigma = panel["ewma_vol"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame(
            {
                "log_band_over_vol": np.log(panel["band_over_vol"]),
                "signed_run_capped": panel["signed_run"].clip(-5, 5),
                "ret_1_z": panel["ret_1"] / sigma,
                "ret_5_z": panel["ret_5"] / (sigma * np.sqrt(5)),
                "ret_20_z": panel["ret_20"] / (sigma * np.sqrt(20)),
                "ma_gap_20_z": panel["ma_gap_20"] / sigma,
                "drawdown_60": panel["drawdown_60"],
                "bounce_20": panel["bounce_20"],
                "vol_ratio": panel["vol_ratio"],
                "log_rel_dollar_volume": np.log(panel["rel_dollar_volume"]),
                "stable_count_20": panel["stable_count_20"],
            },
            index=panel.index,
        )
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True, eq=False)
class Prepared:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    symbols: tuple[str, ...]


def _design(
    inputs: pd.DataFrame, b3: np.ndarray, symbols: Sequence[str], prep: Prepared
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logits = np.column_stack([np.log(b3[:, 0] / b3[:, 1]), np.log(b3[:, 2] / b3[:, 1])])
    x = np.column_stack([inputs.to_numpy(dtype="float64"), logits])
    x = np.where(np.isfinite(x), x, prep.medians)
    x = (x - prep.means) / prep.scales
    stock = np.array(
        [[1.0 if s == known else 0.0 for known in prep.symbols] for s in symbols]
    ).reshape(len(symbols), len(prep.symbols))
    return np.column_stack([x, stock])


def prepare(inputs: pd.DataFrame, b3: np.ndarray, symbols: Sequence[str]) -> Prepared:
    with np.errstate(divide="ignore", invalid="ignore"):
        logits = np.column_stack([np.log(b3[:, 0] / b3[:, 1]), np.log(b3[:, 2] / b3[:, 1])])
    x = np.column_stack([inputs.to_numpy(dtype="float64"), logits])
    x = np.where(np.isfinite(x), x, np.nan)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(x), x, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    return Prepared(medians, means, scales, tuple(sorted(set(symbols))))


@dataclass(frozen=True, eq=False)
class FittedM1:
    prep: Prepared
    model: LogisticRegression
    c: float
    selection: tuple[dict, ...]

    def predict(self, inputs: pd.DataFrame, b3: np.ndarray, symbols: Sequence[str]) -> np.ndarray:
        x = _design(inputs, b3, symbols, self.prep)
        proba = self.model.predict_proba(x)
        out = np.zeros((len(x), 3))
        out[:, self.model.classes_.astype(int)] = proba
        return out


def _fit(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    model = LogisticRegression(C=c, max_iter=MAX_ITER)
    model.fit(x, y.astype(int))
    return model


def fit_m1(
    inputs: pd.DataFrame,
    b3: np.ndarray,
    symbols: Sequence[str],
    outcomes: np.ndarray,
    sessions: pd.Series,
    target_ends: pd.Series,
) -> FittedM1:
    """Fit on training rows; C chosen by the purged inner split (module docstring)."""

    symbols = list(symbols)
    order = pd.DatetimeIndex(sessions).unique().sort_values()
    selection = [{"c": C_GRID[0], "note": "too few sessions for an inner split: most shrinkage"}]
    chosen = C_GRID[0]
    if len(order) > INNER_SESSIONS + 252:
        inner_start = order[-INNER_SESSIONS]
        inner_fit = np.asarray(pd.DatetimeIndex(target_ends) < inner_start)
        inner_test = np.asarray(pd.DatetimeIndex(sessions) >= inner_start)
        prep = prepare(
            inputs[inner_fit], b3[inner_fit], [symbols[i] for i in np.flatnonzero(inner_fit)]
        )
        x_fit = _design(
            inputs[inner_fit], b3[inner_fit], [symbols[i] for i in np.flatnonzero(inner_fit)], prep
        )
        x_test = _design(
            inputs[inner_test],
            b3[inner_test],
            [symbols[i] for i in np.flatnonzero(inner_test)],
            prep,
        )
        dates = pd.DatetimeIndex(sessions)[inner_test]
        per_date = {}
        for c in C_GRID:
            model = _fit(x_fit, outcomes[inner_fit], c)
            proba = np.zeros((len(x_test), 3))
            proba[:, model.classes_.astype(int)] = model.predict_proba(x_test)
            per_date[c] = pd.Series(rps(proba, outcomes[inner_test])).groupby(dates).mean()
        chosen, rows = one_se_choice(per_date, sorted(C_GRID))  # smaller C: more shrinkage
        selection = [{"c": row.pop("candidate"), **row} for row in rows]
    prep = prepare(inputs, b3, symbols)
    model = _fit(_design(inputs, b3, symbols, prep), outcomes, chosen)
    return FittedM1(prep, model, chosen, tuple(selection))
