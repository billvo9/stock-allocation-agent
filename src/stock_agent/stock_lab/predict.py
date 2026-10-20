"""
Today's forecast from frozen parameters (research side; the page reads the
stored artifacts only).

`fit_final` freezes every parameter on the matured development rows (each
target ended on or before `outcome_end`), with the half-life the
evaluation's folds chose most often (ties: the longer). `forecast_stock`
then forecasts one stock as of its last session with a close, using only
that session and earlier ones as state, with the estimator the evaluation
selected (B1 unless a challenger beat it; evaluation.py). The artifact is
`forecast.build`, checked before it is returned.

For real data this needs owner decisions A2 (inference inputs from the
recent window) and A4 (recording real forecasts); until then only
synthetic demo forecasts are made, labelled so.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import logistic
from stock_agent.stock_lab.contracts import EvidenceStatus
from stock_agent.stock_lab.evaluation import EvaluationConfig, _frequency_state
from stock_agent.stock_lab.features import FEATURE_VERSION
from stock_agent.stock_lab.reversal import after_streak
from stock_agent.stock_lab.targets import HORIZONS, cumulative_band
from stock_agent.stock_lab.traps import TrapRule, evaluate_traps, trap_events


@dataclass(frozen=True, eq=False)
class FinalModel:
    config: EvaluationConfig
    half_life: float
    tilt: est.Tilt
    errors: Mapping[int, est.ErrorDistribution]
    cumulative_errors: est.ErrorDistribution
    m1: Mapping[str, logistic.FittedM1]
    version: str


def chosen_half_life(record: Mapping) -> float:
    counts = collections.Counter(fold["half_life"] for fold in record["folds"])
    top = max(counts.values())
    return max(h for h, n in counts.items() if n == top)


def fit_final(panel: pd.DataFrame, config: EvaluationConfig, record: Mapping) -> FinalModel:
    half_life = chosen_half_life(record)
    matured = panel["target_end_date"].notna().to_numpy()
    if config.outcome_end is not None:
        matured = matured & (panel["target_end_date"] <= config.outcome_end).to_numpy()
    params = est.FrequencyParams(config.alpha_prior, half_life, config.kappa)
    _, b1, _ = _frequency_state(panel, "category_code", params)
    states = est.states_of(panel)
    outcome = panel["target_1"].to_numpy(dtype="float64", na_value=np.nan)
    tilt = est.fit_tilt(b1[matured], states[matured], outcome[matured], ridge=config.tilt_ridge)
    symbols = panel["symbol"].to_numpy(dtype=object)
    errors = {
        h: est.errors_from_panel(
            panel, est.panel_standardized(panel, h), matured, h, config.error_kappa
        )
        for h in HORIZONS
    }
    cum_errors = est.errors_from_panel(
        panel, est.panel_standardized(panel, 5, sessions=5), matured, 5, config.error_kappa
    )
    m1 = {}
    if record["selection"]["selected"] == "M1":
        inputs = logistic.model_inputs(panel)
        sigma = panel["ewma_vol"].to_numpy(dtype="float64")
        for h in HORIZONS:
            target = f"target_{h}"
            b3 = est.b3(errors[h], symbols, sigma, config.band)
            y = panel[target].to_numpy(dtype="float64", na_value=np.nan)
            rows = np.flatnonzero(matured & np.isfinite(y) & np.isfinite(b3).all(axis=1))
            m1[target] = logistic.fit_m1(
                inputs.iloc[rows],
                b3[rows],
                list(symbols[rows]),
                y[rows],
                panel["date"].iloc[rows],
                panel["target_end_date"].iloc[rows],
            )
    version = artifact.forecast_id(
        {
            "config": config.as_dict(),
            "half_life": half_life,
            "theta": np.round(tilt.theta, 12).tolist(),
            "evaluation": record["run_id"],
            "selected": record["selection"]["selected"],
        }
    )[:16]
    return FinalModel(config, half_life, tilt, errors, cum_errors, m1, version)


def _frequencies(panel: pd.DataFrame, row: int, model: FinalModel):
    """(B0, B1, B1 of the five-session total) at `row`, from the panel's own history."""

    config = model.config
    params = est.FrequencyParams(config.alpha_prior, model.half_life, config.kappa)
    b0, b1, _ = _frequency_state(panel, "category_code", params)
    _, b1_cum, _ = _frequency_state(panel, "cum_code", params)
    return b0[row], b1[row], b1_cum[row]


def _probabilities(
    panel: pd.DataFrame, row: int, model: FinalModel, selected: str, frequencies=None
):
    """
    (daily (5, 3), cumulative (3,), B0 (3,)) of `selected` at `row`. `frequencies`
    is (B0, B1, B1 of the total) at that row; by default it is computed from the
    panel's own history (a frozen model passes its own: registry.probabilities).
    """

    config = model.config
    b0_row, b1_row, b1_cum_row = (
        _frequencies(panel, row, model) if frequencies is None else frequencies
    )
    symbols = panel["symbol"].to_numpy(dtype=object)
    sigma = panel["ewma_vol"].to_numpy(dtype="float64")
    one = slice(row, row + 1)
    if selected == "B0":
        daily = np.repeat(b0_row[None, :], 5, axis=0)
    elif selected == "B1":
        daily = np.repeat(b1_row[None, :], 5, axis=0)
    elif selected == "B2":
        daily = est.b2_marginals(b1_row[None, :], est.states_of(panel.iloc[one]), model.tilt)[0]
    elif selected in ("B3", "M1"):
        rows = []
        for h in HORIZONS:
            b3 = est.b3(model.errors[h], symbols[one], sigma[one], config.band)
            if selected == "M1":
                inputs = logistic.model_inputs(panel.iloc[one])
                b3 = model.m1[f"target_{h}"].predict(inputs, b3, list(symbols[one]))
            rows.append(b3[0])
        daily = np.array(rows)
    else:
        raise ValueError(f"unknown estimator {selected!r}")
    if selected in ("B3", "M1"):
        cumulative = est.b3(
            model.cumulative_errors,
            symbols[one],
            sigma[one] * math.sqrt(5),
            cumulative_band(config.band),
        )[0]
    else:
        cumulative = b1_cum_row
    return daily, cumulative, b0_row


def forecast_stock(
    panel: pd.DataFrame,
    symbol: str,
    model: FinalModel,
    record: Mapping,
    *,
    target_sessions: Sequence[str],
    created_at: pd.Timestamp,
    data: Mapping[str, object],
    data_class: str,
    supersedes: str | None = None,
) -> dict:
    """The forecast artifact for `symbol` as of its last session with a close."""

    rows = np.flatnonzero(
        (panel["symbol"] == symbol).to_numpy() & panel["close"].notna().to_numpy()
    )
    if not len(rows):
        raise ValueError(f"no close for {symbol}")
    row = int(rows[-1])
    selected = record["selection"]["selected"]
    stock = panel.iloc[rows].reset_index(drop=True)
    last = panel.iloc[row]
    limitations = [
        (
            "Experimental and unvalidated: the evaluation is a development test of a "
            "hindsight-selected list, not evidence of skill."
        ),
        (
            "Each session's probabilities are for that day's own move, not the move from the "
            "as-of close to that day; the five-session total is shown separately."
        ),
    ]
    if data_class == "synthetic":
        limitations.insert(0, "Synthetic demo stock: not a real company.")
    refusal = None
    daily = cumulative = usual = comparator = None
    if not np.isfinite(last["ewma_vol"]):
        refusal = {"reason": "insufficient_history", "detail": "Too little history for volatility."}
    else:
        frequencies = _frequencies(panel, row, model)
        daily, cumulative, usual = _probabilities(panel, row, model, selected, frequencies)
        daily = daily / daily.sum(axis=1, keepdims=True)
        cumulative = cumulative / cumulative.sum()
        # B1 on the same inputs, sessions and bands: the comparator recorded beside it.
        b1_daily, b1_cumulative, _ = _probabilities(panel, row, model, "B1", frequencies)
        b1_daily = b1_daily / b1_daily.sum(axis=1, keepdims=True)
        b1_cumulative = b1_cumulative / b1_cumulative.sum()
        values = (daily, cumulative, b1_daily, b1_cumulative)
        if not all(np.isfinite(v).all() for v in values):
            refusal = {"reason": "estimator_unavailable", "detail": "An input was missing."}
            daily = cumulative = None
        else:
            comparator = {
                "half_life": model.half_life,
                "daily": b1_daily.tolist(),
                "cumulative": b1_cumulative.tolist(),
            }
    direction = last["run_direction"]
    signals = {
        "current_run": {
            "direction": direction if isinstance(direction, str) else None,
            "length": int(last["run_length"]),
        },
        # The stock's long-run mix (B0) as of the same close: the "usual" beside a forecast.
        "usual_mix": None
        if usual is None
        else dict(zip(artifact.PROBABILITY_KEYS, (float(v) for v in usual), strict=True)),
    }
    if isinstance(direction, str) and direction != "stable" and int(last["run_length"]) >= 1:
        k = min(int(last["run_length"]), 7)  # the page's streak lengths are 1 to 7
        summary = after_streak(
            stock, k, direction, model.config.band, outcome_end=model.config.outcome_end
        )
        next_rows = summary.table[
            (summary.table["horizon"] == 1) & (summary.table["regime"] == "all")
        ]
        signals["after_this_streak"] = {
            "definition": summary.definition,
            "events": summary.events,
            "outcomes": {
                r["outcome"]: {
                    "estimate": _finite(r["estimate"]),
                    "n_eff": r["n_eff"],
                    "status": r["status"],
                }
                for r in next_rows.to_dict("records")
            },
        }
    rule = TrapRule()
    events = trap_events(stock, rule, outcome_end=model.config.outcome_end)
    evaluation = evaluate_traps(events, rule)
    latest = evaluation.latest_setup
    recent = latest is not None and (stock["session"].iloc[-1] - latest["session"]).days <= 21
    signals["bull_trap"] = {
        "setup_now": bool(recent and latest["status"] == "unresolved"),
        "setup_session": None if latest is None else str(pd.Timestamp(latest["session"]).date()),
        "historical_failure_rate": _finite(evaluation.setup_rate.estimate),
        "base_rate": _finite(evaluation.base_rate.estimate),
        "setups_resolved": evaluation.setup_rate.n,
    }
    calibration = (
        "development_tested_synthetic" if data_class == "synthetic" else "development_tested"
    )
    return artifact.build(
        symbol=symbol,
        data_class=data_class,
        as_of_session=str(pd.Timestamp(last["session"]).date()),
        created_at=created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        target_sessions=list(target_sessions),
        daily=None if daily is None else daily.tolist(),
        cumulative=None if cumulative is None else cumulative.tolist(),
        band=model.config.band.tau,
        cumulative_band=cumulative_band(model.config.band).tau,
        estimator=selected,
        model_version=model.version,
        feature_version=FEATURE_VERSION,
        evaluation_run_id=record["run_id"],
        data=data,
        evidence_status=(
            EvidenceStatus.SYNTHETIC_DEMO.value
            if data_class == "synthetic"
            else EvidenceStatus.EXPERIMENTAL_UNVALIDATED.value
        ),
        calibration_status=calibration,
        signals=signals,
        limitations=limitations,
        refusal=refusal,
        supersedes=supersedes,
        comparator=comparator,
    )


def _finite(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
