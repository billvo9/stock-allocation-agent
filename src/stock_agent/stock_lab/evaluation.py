"""
Walk-forward evaluation of the Stock Lab's estimators (research side: the
Lab's page never imports this module; it reads the stored result).

Protocol (pre-registered here; docs/research/stock_probability_lab.md, section 5)
    Panel      one row per (stock, session): past-only features (features.py)
               and targets (targets.py), with outcomes censored at
               `outcome_end` (2024-12-31 for real data).
    Folds      `model_validation.folds.make_expanding_folds`, development
               mode (the lockbox is enforced there): yearly test windows,
               expanding training, strict global purge (a row trains only if
               its five-session target ended before the test window opens).
    Freeze     at each test start: B1's half-life (from H_GRID, by the inner
               split, one-SE rule toward the longer half-life), B2's tilt,
               B3's error distributions and M1 (logistic.py). Estimator state
               (counts, volatility, the run) is online, from data through
               each decision session only.
    Scores     multiclass Brier, log loss (floored at estimators.FLOOR) and
               the ranked probability score (RPS, the primary score), plus
               the moving and direction decompositions (scores.py), per
               model and target.
    Inference  per decision date, the cross-sectional mean of each model's
               RPS minus B1's (the reference). The decision test (owner
               decision A7, 2026-10-19, DECISION_RULE) is the fold-block t: a
               t-test on the test years' means, t(folds - 1), one-sided; Holm
               across the pre-registered challengers at a family level of
               ALPHA (2.5%). Beside it, reported only: Newey-West (lag NW_LAG,
               at least h - 1) and a circular block bootstrap (block BLOCK)
               (size_simulation.one_sided_tests). Dates are the unit, never
               rows, so stocks that move together are not counted as
               independent evidence.
    Selection  the reported estimator is B1 unless a challenger has a lower
               mean RPS with a Holm-adjusted one-sided fold-block t p below
               ALPHA on the primary target set (all five daily targets pooled
               by date); among those, the simplest, replaced by a more complex
               one only when its own fold-block t against the current choice
               is below ALPHA too. Every candidate and every inner choice is
               recorded; nothing is selected on a test result alone.
    Reported   never used to select: the Newey-West and bootstrap p-values,
               B1's score at every half-life, B1 and B3 re-run at other stable
               bands on the same folds (SENSITIVITY_BANDS), and each test's
               simulated size on no-skill panels shaped like this one
               (size_simulation).

Nothing here is evidence of skill on its own: the watchlist is chosen in
hindsight, and synthetic runs only check the machinery.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from stock_agent.model_diagnostics.inference import draw_circular_blocks
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    make_expanding_folds,
)
from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import logistic, scores, size_simulation
from stock_agent.stock_lab.contracts import DEFAULT_BAND, StableBand
from stock_agent.stock_lab.features import FEATURE_VERSION, build_features
from stock_agent.stock_lab.frozen_model import A7_DECISION_RULE
from stock_agent.stock_lab.prices import BAR_COLUMNS, categorize, category_codes
from stock_agent.stock_lab.reversal import volatility_regime
from stock_agent.stock_lab.runs import RunRule
from stock_agent.stock_lab.targets import HORIZONS, build_targets, cumulative_band

SCHEMA = "stock_lab_evaluation"
SCHEMA_VERSION = "1.3"  # 1.1: fold-block t etc.; 1.2: history hashes; 1.3: A7 decides
MODELS = ("B0", "B1", "B2", "B3", "M1")
REFERENCE = "B1"
H_GRID = (21.0, 63.0, 126.0, 252.0)
NW_LAG = 10
BLOCK = 20
BOOTSTRAP_REPS = 499
BOOTSTRAP_SEED = 7
ALPHA = 0.025  # decision A7: the family level of Holm and of each simplest-first step
DECISION_TEST = "fold_t_p_one_sided"  # decision A7: the fold-block t decides
DECISION_RULE = A7_DECISION_RULE
MIN_TRAIN_SESSIONS = 756  # three years before the first test window
TARGETS = (*(f"target_{h}" for h in HORIZONS), "target_cum")
SENSITIVITY_BANDS = (0.0025, 0.01)  # reported only, never used to select
SIZE_REPS = 1000  # no-skill panels, shared by every placebo and half-life; MC SE ~0.7 pp at 5%
SIZE_SEED = 20261016
SIZE_CORRELATION = 0.4  # pairwise correlation of the no-skill panels' returns


@dataclass(frozen=True)
class EvaluationConfig:
    band: StableBand = DEFAULT_BAND
    half_life_grid: tuple[float, ...] = H_GRID
    alpha_prior: float = 50.0
    kappa: float = 40.0
    tilt_ridge: float = 10.0
    error_kappa: float = 250.0
    first_test_year: int | None = None  # default: three years after the panel starts
    last_test_year: int = 2024
    include_m1: bool = True
    outcome_end: pd.Timestamp | None = None  # real data: 2024-12-31
    sensitivity_bands: tuple[float, ...] = SENSITIVITY_BANDS
    size_reps: int = SIZE_REPS
    size_seed: int = SIZE_SEED
    size_correlation: float = SIZE_CORRELATION

    def as_dict(self) -> dict:
        return {
            "band": self.band.tau,
            "cumulative_band": cumulative_band(self.band).tau,
            "half_life_grid": list(self.half_life_grid),
            "alpha_prior": self.alpha_prior,
            "kappa": self.kappa,
            "tilt_ridge": self.tilt_ridge,
            "error_kappa": self.error_kappa,
            "first_test_year": self.first_test_year,
            "last_test_year": self.last_test_year,
            "include_m1": self.include_m1,
            "outcome_end": None if self.outcome_end is None else str(self.outcome_end.date()),
            "floor": est.FLOOR,
            "nw_lag": NW_LAG,
            "block": BLOCK,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
            "sensitivity_bands": list(self.sensitivity_bands),
            "size_reps": self.size_reps,
            "size_seed": self.size_seed,
            "size_correlation": self.size_correlation,
            "m1_c_grid": list(logistic.C_GRID),
            "m1_inputs": list(logistic.M1_INPUTS),
            "feature_version": FEATURE_VERSION,
        }


def build_panel(histories: Mapping[str, pd.DataFrame], config: EvaluationConfig) -> pd.DataFrame:
    """The (stock, session) panel: features, the session's own category and targets."""

    parts = []
    cum_band = cumulative_band(config.band)
    for symbol, bars in histories.items():
        frame = build_features(bars, band=config.band, rule=RunRule())
        targets = build_targets(frame, config.band, outcome_end=config.outcome_end)
        part = pd.concat([frame, targets], axis=1)
        part["symbol"] = symbol
        trailing = frame["level"] / frame["level"].shift(5) - 1.0
        part["cum_code"] = category_codes(categorize(trailing, cum_band)).to_numpy()
        # Calm, middle or volatile against the stock's own earlier volatility.
        part["regime"] = volatility_regime(frame).to_numpy()
        parts.append(part)
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.DatetimeIndex(panel["session"]).as_unit("ns")
    for column in ("target_start_date", "target_end_date"):
        panel[column] = pd.DatetimeIndex(panel[column]).as_unit("ns")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel


def _windows(
    panel: pd.DataFrame, config: EvaluationConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first_date = panel["date"].min()
    first_year = config.first_test_year or int(first_date.year) + 3
    end_cap = (
        MODEL_HOLDOUT_START
        if config.outcome_end is None
        else min(MODEL_HOLDOUT_START, config.outcome_end + pd.Timedelta(days=1))
    )
    windows = []
    for year in range(first_year, config.last_test_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = min(pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"), end_cap)
        if start < end and (panel["date"] >= start).any():
            windows.append((start, end))
    return windows


def _codes(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="float64", na_value=np.nan)


@dataclass
class _Predictions:
    """Test-row predictions per model and target: arrays (rows, 3)."""

    rows: list[np.ndarray] = field(default_factory=list)
    by_model: dict[str, dict[str, list[np.ndarray]]] = field(default_factory=dict)

    def add(self, model: str, target: str, values: np.ndarray) -> None:
        self.by_model.setdefault(model, {}).setdefault(target, []).append(values)


def _frequency_state(panel: pd.DataFrame, code_column: str, params: est.FrequencyParams):
    view = panel[["symbol", "session"]].copy()
    view["category_code"] = panel[code_column]
    prior = est.b0(view, params)
    b1, n_eff = est.b1(view, params, prior)
    return prior, b1, n_eff


def _inner_half_life(
    panel: pd.DataFrame,
    train: np.ndarray,
    b1_by_h: Mapping[float, np.ndarray],
    config: EvaluationConfig,
) -> tuple[float, list[dict]]:
    """
    B1's half-life from the last two years of the training rows by the paired
    one-standard-error rule (scores.one_se_choice), the longest half-life
    being the simplest (smoothest) choice. Inner rows are training rows,
    whose targets all ended before the outer test window.
    """

    dates = pd.DatetimeIndex(panel["date"])
    train_dates = dates[train]
    if train_dates.empty:
        return config.half_life_grid[1], []
    inner_start = train_dates.max() - pd.Timedelta(days=730)
    outcome = _codes(panel["target_1"])
    usable = train & np.asarray(dates >= inner_start) & np.isfinite(outcome)
    per_date = {
        h: pd.Series(scores.rps(b1_by_h[h][usable], outcome[usable])).groupby(dates[usable]).mean()
        for h in config.half_life_grid
    }
    chosen, rows = scores.one_se_choice(per_date, sorted(config.half_life_grid, reverse=True))
    return chosen, [{"half_life": row.pop("candidate"), **row} for row in rows]


def evaluate(
    histories: Mapping[str, pd.DataFrame],
    config: EvaluationConfig | None = None,
    *,
    data_class: str,
    run_id: str,
    created_at: pd.Timestamp,
) -> dict:
    """The full walk-forward evaluation as a JSON-ready record (module docstring)."""

    config = config or EvaluationConfig()
    if config.last_test_year >= MODEL_HOLDOUT_START.year:
        raise ValueError(
            f"test years end by {MODEL_HOLDOUT_START.year - 1}: the lockbox from "
            f"{MODEL_HOLDOUT_START.date()} is never evaluated here"
        )
    panel = build_panel(histories, config)
    windows = _windows(panel, config)
    if not windows:
        raise ValueError("no test window: the panel is too short")
    folds = make_expanding_folds(panel, windows, holdout_start=MODEL_HOLDOUT_START)
    symbols = panel["symbol"].to_numpy(dtype=object)
    sigma = panel["ewma_vol"].to_numpy(dtype="float64")
    states = est.states_of(panel)
    inputs = logistic.model_inputs(panel) if config.include_m1 else None
    cum_band = cumulative_band(config.band)

    # Online state for each pre-registered half-life, computed once (it uses data
    # through each decision session only, so it is the same in every fold).
    state = {}
    for h in config.half_life_grid:
        params = est.FrequencyParams(config.alpha_prior, h, config.kappa)
        state[h] = (
            _frequency_state(panel, "category_code", params),
            _frequency_state(panel, "cum_code", params),
        )
    b1_by_h = {h: state[h][0][1] for h in config.half_life_grid}
    outcome_1 = _codes(panel["target_1"])
    z_by_h = {h: est.panel_standardized(panel, h) for h in HORIZONS}
    z_total = est.panel_standardized(panel, 5, sessions=5)
    paths: list[dict] = []

    test_index: list[np.ndarray] = []
    preds = _Predictions()
    fold_rows = []
    for fold in folds:
        train, test = fold.train_rows, fold.test_rows
        train_mask = np.zeros(len(panel), dtype=bool)
        train_mask[train] = True
        half_life, inner = _inner_half_life(panel, train_mask, b1_by_h, config)
        (b0, b1, _), (b0_cum, b1_cum, _) = state[half_life]
        tilt = est.fit_tilt(b1[train], states[train], outcome_1[train], ridge=config.tilt_ridge)
        b2 = est.b2_marginals(b1[test], states[test], tilt)
        paths.append(
            {
                "rows": test,
                "probabilities": est.b2_path_probabilities(b1[test], states[test], tilt),
                "marginals": b2,
            }
        )
        fold_info = {
            "fold_id": fold.fold_id,
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            "train_rows": len(train),
            "test_rows": len(test),
            "purged_rows": len(fold.purged_rows),
            "half_life": half_life,
            "half_life_inner": inner,
            "tilt_rows_per_state": list(tilt.rows_per_state),
            "m1_c": {},
            "m1_inner": {},
        }
        for h in HORIZONS:
            target = f"target_{h}"
            preds.add("B0", target, b0[test])
            preds.add("B1", target, b1[test])
            preds.add("B2", target, b2[:, h - 1])
            errors = est.errors_from_panel(panel, z_by_h[h], train_mask, h, config.error_kappa)
            b3 = est.b3(errors, symbols[test], sigma[test], config.band)
            preds.add("B3", target, b3)
            if config.include_m1:
                b3_train = est.b3(errors, symbols[train], sigma[train], config.band)
                outcome = _codes(panel[target])
                usable = np.isfinite(outcome[train]) & np.isfinite(b3_train).all(axis=1)
                rows = train[usable]
                fitted = logistic.fit_m1(
                    inputs.iloc[rows],
                    b3_train[usable],
                    list(symbols[rows]),
                    outcome[rows],
                    panel["date"].iloc[rows],
                    panel["target_end_date"].iloc[rows],
                )
                fold_info["m1_c"][target] = fitted.c
                fold_info["m1_inner"][target] = list(fitted.selection)
                safe_b3 = np.where(np.isfinite(b3), b3, 1.0 / 3.0)
                preds.add(
                    "M1", target, fitted.predict(inputs.iloc[test], safe_b3, list(symbols[test]))
                )
        # The five-session total: frequency baselines of past totals, and B3 on 5-session moves.
        preds.add("B0", "target_cum", b0_cum[test])
        preds.add("B1", "target_cum", b1_cum[test])
        errors5 = est.errors_from_panel(panel, z_total, train_mask, 5, config.error_kappa)
        preds.add(
            "B3", "target_cum", est.b3(errors5, symbols[test], sigma[test] * math.sqrt(5), cum_band)
        )
        test_index.append(test)
        fold_rows.append(fold_info)

    test_rows = np.concatenate(test_index)
    fold_starts = pd.DatetimeIndex([fold.test_start for fold in folds])
    record = _score(panel, test_rows, preds, fold_starts)
    record["path_calibration"] = _path_calibration(panel, paths)
    record["sensitivity_half_life"] = _half_life_sensitivity(panel, test_rows, b1_by_h)
    record["sensitivity_band"] = _band_sensitivity(histories, config, panel, folds, z_by_h, b1_by_h)
    record["size_simulation"] = size_simulation.simulate_size(
        size_design(config, names=len(histories), years=len(folds))
    )
    record.update(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "decision_rule": DECISION_RULE,
            "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "data_class": data_class,
            "stocks": sorted(histories),
            "config": config.as_dict(),
            "folds": fold_rows,
            "candidate_family": list(MODELS if config.include_m1 else MODELS[:-1]),
            "panel_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(panel[["symbol", "date", "price_return"]], index=False)
                .to_numpy()
                .tobytes()
            ).hexdigest(),
            # Every stored field of every history (prices, volume, dividends, splits), so a model
            # frozen later can show it was fitted on exactly these bars (registry.freeze).
            "histories_sha256": {
                symbol: history_sha256(bars) for symbol, bars in sorted(histories.items())
            },
        }
    )
    return record


def history_sha256(bars: pd.DataFrame) -> str:
    """The content hash of one history's bars (prices.BAR_COLUMNS, in session order)."""

    frame = bars.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _per_date_mean(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values).groupby(dates).mean()


def _score(panel, test_rows, preds: _Predictions, fold_starts: pd.DatetimeIndex) -> dict:
    dates_all = pd.DatetimeIndex(panel["date"])
    metrics, comparisons, reliability, calibration, by_stock, by_regime = [], [], [], [], [], []
    pooled_diff: dict[str, list[pd.Series]] = {}
    for target in TARGETS:
        outcome_all = _codes(panel[target])
        y = outcome_all[test_rows]
        models = [m for m in MODELS if target in preds.by_model.get(m, {})]
        stacked = {m: np.concatenate(preds.by_model[m][target]) for m in models}
        valid = np.isfinite(y)
        for m in models:
            valid &= np.isfinite(stacked[m]).all(axis=1)
        y_v = y[valid]
        dates = dates_all[test_rows][valid]
        symbols = panel["symbol"].to_numpy()[test_rows][valid]
        tercile = panel["regime"].to_numpy(dtype=object)[test_rows][valid]
        per_model_rps = {}
        for m in models:
            p = stacked[m][valid]
            row = {"model": m, "target": target, "n": len(y_v), "dates": int(dates.nunique())}
            for name, function in scores.SCORES.items():
                row[name] = float(np.nanmean(function(p, y_v))) if len(y_v) else math.nan
            metrics.append(row)
            per_model_rps[m] = scores.rps(p, y_v)
            for bin_row in scores.reliability(p, y_v, groups=np.asarray(dates)):
                reliability.append({"model": m, "target": target, **bin_row})
            for c in range(3):
                intercept, slope = scores.calibration_line(p[:, c], (y_v == c).astype(float))
                calibration.append(
                    {
                        "model": m,
                        "target": target,
                        "category": c,
                        "intercept": intercept,
                        "slope": slope,
                    }
                )
            for symbol in sorted(set(symbols)):
                chosen = symbols == symbol
                by_stock.append(
                    {
                        "model": m,
                        "target": target,
                        "stock": symbol,
                        "n": int(chosen.sum()),
                        "rps": float(per_model_rps[m][chosen].mean()),
                    }
                )
            for regime in ("calm", "middle", "volatile"):
                chosen = tercile == regime
                by_regime.append(
                    {
                        "model": m,
                        "target": target,
                        "regime": regime,
                        "n": int(chosen.sum()),
                        "rps": float(per_model_rps[m][chosen].mean()) if chosen.any() else math.nan,
                    }
                )
        if REFERENCE not in per_model_rps:
            continue
        for m in models:
            if m == REFERENCE:
                continue
            diff = _per_date_mean(per_model_rps[m] - per_model_rps[REFERENCE], dates)
            comparisons.append(_comparison(m, target, diff, fold_starts))
            if target != "target_cum":
                pooled_diff.setdefault(m, []).append(diff)
    pooled_rows = []
    for m, series in pooled_diff.items():
        combined = pd.concat(series, axis=1).mean(axis=1)
        pooled_rows.append(_comparison(m, "daily_pooled", combined, fold_starts))
    beating = _beating(pooled_rows)
    selected, steps = _select(beating, pooled_diff, fold_starts)
    return {
        "metrics": metrics,
        "comparisons": comparisons + pooled_rows,
        "reliability": reliability,
        "calibration": calibration,
        "by_stock": by_stock,
        "by_regime": by_regime,
        "selection": {
            "rule": (
                f"keep {REFERENCE} unless a challenger's mean RPS over the five daily targets is "
                f"lower with a Holm-adjusted one-sided p below {ALPHA} (the fold-block t on the "
                "test years' means, t(folds - 1); decision A7); among those, start from the "
                "simplest and move to a more complex one only if it also beats the current choice "
                f"(one-sided fold-block t p below {ALPHA})"
            ),
            "decision_rule": DECISION_RULE,
            "decision_test": "fold_block_t",
            "selected": selected,
            "beating_reference": [row["model"] for row in beating],
            "steps": steps,
            "reported_only": (
                f"the Newey-West (lag {NW_LAG}) and circular block bootstrap p-values, the "
                "half-life and band sensitivities and the simulated sizes are reported beside the "
                "results and never used to select"
            ),
        },
    }


SIMPLEST_FIRST = ("B0", "B1", "B3", "B2", "M1")


def _beating(pooled_rows: list[dict]) -> list[dict]:
    """
    The challengers that beat B1 (decision A7): Holm across the pooled comparisons' one-sided
    decision-test p-values (DECISION_TEST, the fold-block t), each row given its adjusted
    `p_holm`; a challenger beats B1 when its mean RPS is lower and its p_holm is below ALPHA.
    """

    adjusted = scores.holm({row["model"]: row[DECISION_TEST] for row in pooled_rows})
    for row in pooled_rows:
        row["p_holm"] = adjusted.get(row["model"], math.nan)
    return [
        row
        for row in pooled_rows
        if row["mean_diff"] < 0 and not math.isnan(row["p_holm"]) and row["p_holm"] < ALPHA
    ]


def _select(
    beating: list[dict],
    pooled_diff: Mapping[str, list[pd.Series]],
    fold_starts: pd.DatetimeIndex,
):
    """
    The selection hierarchy: B1 unless a challenger beats it (Holm); then the
    simplest such challenger, replaced by a more complex one only when that one
    beats it too, on the same pooled per-date RPS differences.
    """

    winners = sorted((row["model"] for row in beating), key=SIMPLEST_FIRST.index)
    if not winners:
        return REFERENCE, []

    def against_reference(model: str) -> pd.Series:
        return pd.concat(pooled_diff[model], axis=1).mean(axis=1)

    current, steps = winners[0], []
    for challenger in winners[1:]:
        # (challenger - B1) - (current - B1) = challenger - current, per date.
        diff = (against_reference(challenger) - against_reference(current)).dropna()
        row = _comparison(challenger, "daily_pooled", diff, fold_starts)
        p = row[DECISION_TEST]
        moved = bool(row["mean_diff"] < 0 and not math.isnan(p) and p < ALPHA)
        steps.append(
            {
                "challenger": challenger,
                "against": current,
                "mean_diff": row["mean_diff"],
                "p_one_sided": row["p_one_sided"],
                "fold_t_p_one_sided": row["fold_t_p_one_sided"],
                "boot_p_one_sided": row["boot_p_one_sided"],
                "moved": moved,
            }
        )
        if moved:
            current = challenger
    return current, steps


def _path_calibration(panel: pd.DataFrame, paths: list[dict]) -> list[dict]:
    """
    Full-path calibration of B2's joint path probabilities on the test rows:
    all five sessions rising, all five falling, and the nine joint (d+1, d+2)
    cells; with the product of B2's own marginals beside it (what the
    marginals alone would say if sessions were independent).
    """

    rows = np.concatenate([p["rows"] for p in paths])
    outcome = np.column_stack([_codes(panel[f"target_{h}"])[rows] for h in HORIZONS])
    marginals = np.concatenate([p["marginals"] for p in paths])
    probabilities = {
        key: np.concatenate([p["probabilities"][key] for p in paths])
        for key in paths[0]["probabilities"]
    }
    dates = np.asarray(pd.DatetimeIndex(panel["date"]))[rows]
    out = []
    events = {"all_increase": (2, range(5)), "all_decrease": (0, range(5))}
    for name, (code, horizons) in events.items():
        known = np.isfinite(outcome).all(axis=1) & np.isfinite(probabilities[name])
        hit = (outcome[known] == code).all(axis=1)
        independent = np.prod(marginals[known][:, list(horizons), code], axis=1)
        out.append(_path_row(name, probabilities[name][known], independent, hit, dates[known]))
    for first in range(3):
        for second in range(3):
            name = f"joint_{first}{second}"
            known = np.isfinite(outcome[:, :2]).all(axis=1) & np.isfinite(probabilities[name])
            hit = (outcome[known, 0] == first) & (outcome[known, 1] == second)
            independent = marginals[known, 0, first] * marginals[known, 1, second]
            out.append(_path_row(name, probabilities[name][known], independent, hit, dates[known]))
    return out


def _path_row(name, predicted, independent, hit, dates) -> dict:
    n = len(hit)
    effective = len(set(dates.tolist())) if n else 0
    observed = float(hit.mean()) if n else math.nan
    low, high = scores.jeffreys(observed * effective, effective) if n else (math.nan, math.nan)
    return {
        "event": name,
        "n": n,
        "dates": effective,
        "predicted": float(predicted.mean()) if n else math.nan,
        "independent": float(independent.mean()) if n else math.nan,
        "observed": observed,
        "ci_low": low,
        "ci_high": high,
    }


def _half_life_sensitivity(panel, test_rows, b1_by_h) -> list[dict]:
    """B1's mean RPS over the five daily targets for every half-life (reported only)."""

    rows = []
    for h, b1 in b1_by_h.items():
        values = []
        for target in (f"target_{k}" for k in HORIZONS):
            y = _codes(panel[target])[test_rows]
            known = np.isfinite(y)
            values.append(float(scores.rps(b1[test_rows][known], y[known]).mean()))
        rows.append({"half_life": h, "rps": float(np.mean(values))})
    return rows


def _band_sensitivity(
    histories: Mapping[str, pd.DataFrame],
    config: EvaluationConfig,
    panel: pd.DataFrame,
    folds,
    z_by_h: Mapping[int, np.ndarray],
    b1_by_h: Mapping[float, np.ndarray],
) -> list[dict]:
    """
    B1 and B3 re-run at the band in use and every band in
    config.sensitivity_bands, on the same folds and rows (reported only, never
    used to select). Each band has its own categories and targets; B1's
    half-life is chosen inside each fold's training years by the same rule; B3
    uses the same standardized-return distributions, since z = r / sigma does
    not depend on the band. Mean RPS over the five daily targets, on the rows
    where both are defined. RPS levels are not comparable across bands (a wider
    band makes Stable more common); B3 minus B1 within a band is.
    """

    symbols = panel["symbol"].to_numpy(dtype=object)
    sigma = panel["ewma_vol"].to_numpy(dtype="float64")
    bands = sorted({config.band.tau, *config.sensitivity_bands})
    panels, b1s = {}, {}
    for tau in bands:
        if tau == config.band.tau:
            panels[tau], b1s[tau] = panel, b1_by_h
            continue
        other = build_panel(histories, replace(config, band=StableBand(tau)))
        if not (other["symbol"].equals(panel["symbol"]) and other["date"].equals(panel["date"])):
            raise ValueError("a band changed the panel's rows; the folds would differ")
        panels[tau] = other
        b1s[tau] = {
            h: _frequency_state(
                other, "category_code", est.FrequencyParams(config.alpha_prior, h, config.kappa)
            )[1]
            for h in config.half_life_grid
        }
    chosen = {tau: [] for tau in bands}
    b1_test = {tau: [] for tau in bands}
    b3_test = {tau: {h: [] for h in HORIZONS} for tau in bands}
    for fold in folds:
        train_mask = np.zeros(len(panel), dtype=bool)
        train_mask[fold.train_rows] = True
        test = fold.test_rows
        errors = {
            h: est.errors_from_panel(panel, z_by_h[h], train_mask, h, config.error_kappa)
            for h in HORIZONS
        }
        for tau in bands:
            half_life, _ = _inner_half_life(panels[tau], train_mask, b1s[tau], config)
            chosen[tau].append(half_life)
            b1_test[tau].append(b1s[tau][half_life][test])
            for h in HORIZONS:
                b3_test[tau][h].append(
                    est.b3(errors[h], symbols[test], sigma[test], StableBand(tau))
                )
    test_rows = np.concatenate([fold.test_rows for fold in folds])
    out = []
    for tau in bands:
        b1 = np.concatenate(b1_test[tau])
        b1_rps, b3_rps, stable, rows = [], [], [], 0
        for h in HORIZONS:
            y = _codes(panels[tau][f"target_{h}"])[test_rows]
            b3 = np.concatenate(b3_test[tau][h])
            valid = np.isfinite(y) & np.isfinite(b1).all(axis=1) & np.isfinite(b3).all(axis=1)
            b1_rps.append(float(scores.rps(b1[valid], y[valid]).mean()))
            b3_rps.append(float(scores.rps(b3[valid], y[valid]).mean()))
            stable.append(y[valid] == 1)
            rows += int(valid.sum())
        b1_mean, b3_mean = float(np.mean(b1_rps)), float(np.mean(b3_rps))
        out.append(
            {
                "band": tau,
                "primary": tau == config.band.tau,
                "rows": rows,
                "stable_share": float(np.concatenate(stable).mean()),
                "b1_rps": b1_mean,
                "b3_rps": b3_mean,
                "b3_minus_b1": b3_mean - b1_mean,
                "half_lives": chosen[tau],
            }
        )
    return out


def _comparison(model: str, target: str, diff: pd.Series, fold_starts: pd.DatetimeIndex) -> dict:
    """
    One per-date differential's mean and its three tests: the fold-block t with
    the test years as blocks (the decision test since A7, fold_t_p_one_sided),
    Newey-West (p_one_sided) and the circular block bootstrap (both reported
    only).
    """

    values = diff.to_numpy(dtype="float64")
    blocks = fold_starts.searchsorted(pd.DatetimeIndex(diff.index), side="right") - 1
    # Fewer than two blocks of dates leaves the bootstrap unavailable before any draw is used.
    draws = draw_circular_blocks(
        max(len(values), 1), block_length=BLOCK, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
    )
    tests = size_simulation.one_sided_tests(values, blocks, draws, lag=NW_LAG)
    nw, fold, boot = (tests[method] for method in size_simulation.METHODS)
    return {
        "model": model,
        "vs": REFERENCE,
        "target": target,
        "dates": len(values),
        "mean_diff": float(np.nanmean(values)) if len(values) else math.nan,
        "nw_se": nw.se,
        "nw_status": nw.status,
        "p_one_sided": size_simulation.one_sided_p(nw),
        "fold_t_mean": fold.estimate,
        "fold_t_se": fold.se,
        "fold_t_df": fold.params.get("df"),
        "fold_t_status": fold.status,
        "fold_t_p_one_sided": size_simulation.one_sided_p(fold),
        "boot_se": boot.se,
        "boot_ci_low": boot.ci_low,
        "boot_ci_high": boot.ci_high,
        "boot_status": boot.status,
        "boot_p_one_sided": size_simulation.one_sided_p(boot),
    }


def size_design(config: EvaluationConfig, *, names: int, years: int) -> size_simulation.SizeDesign:
    """The no-skill panels for this evaluation: its stocks, test years, band, B1 and tests."""

    return size_simulation.SizeDesign(
        names=names,
        years=years,
        half_lives=tuple(config.half_life_grid),
        reps=config.size_reps,
        seed=config.size_seed,
        nw_lag=NW_LAG,
        block=BLOCK,
        bootstrap_reps=BOOTSTRAP_REPS,
        bootstrap_seed=BOOTSTRAP_SEED,
        correlation=config.size_correlation,
        tau=config.band.tau,
        alpha_prior=config.alpha_prior,
        kappa=config.kappa,
    )


def dumps(record: dict) -> str:
    """Strict JSON (no NaN: missing numbers become null)."""

    def clean(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        return value

    return json.dumps(clean(record), sort_keys=True, indent=1, allow_nan=False)


def summary_tables(record: Mapping) -> Mapping[str, pd.DataFrame]:
    return {key: pd.DataFrame(record[key]) for key in ("metrics", "comparisons", "reliability")}


__all__: Sequence[str] = ()
