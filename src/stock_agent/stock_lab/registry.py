"""
Freezing and loading the Stock Lab's registered model (research side: the
Lab's page never imports it; the record's contract is `frozen_model`).

Freezing (`freeze`)
    From the development histories and their stored walk-forward evaluation,
    fit every parameter once (predict.fit_final: matured development rows
    only), and record the model with everything a forecast needs. Nothing is
    fitted, tuned, selected or calibrated afterwards.

    - Real data must stop at 2024-12-31 (DEVELOPMENT_END), and the evaluation
      must have censored its outcomes there: a protected-period session is
      refused (`protected_period`), never trimmed.
    - The histories must be the evaluation's own (the same stocks and the
      same panel hash) and the settings its own (`config_mismatch`).
    - Which estimator. The evaluation's selection, when the evaluation records
      a rule the owner confirmed for "beats B1" (decision A7: the fold-block t
      at 2.5%, frozen_model.CONFIRMED_DECISION_RULES); the frozen estimator is
      then exactly its selection (`not_the_selection` otherwise). The record
      must be whole (schema 1.3 or later, the rule and the fold-block t named
      in its selection) and its selection must follow from its own recorded
      comparisons and steps (`inconsistent_selection`). An evaluation without
      a confirmed rule freezes only B1, the pre-registered default, and never
      promotes a challenger (`a7_unresolved`). An evaluation stored before A7
      (decided by Newey-West at 5%) is refused even earlier, because its
      settings name the 5% level (`config_mismatch`). The record's own check
      enforces the A7 rule too, so the store refuses a hand-made record.
    - The histories must be the evaluation's bars exactly: every stored field
      of each one (its `histories_sha256`, evaluation schema 1.2).
    - The record is loaded back and must reproduce the in-memory model's
      probabilities for the selected estimator and B1 at every stock's last
      development session (`round_trip_failed` otherwise).

The state rule (owner decision A2, item 4: frozen_long_run, decided 2026-10-19)
    A frozen model's parameters never change. Its running state is a function
    of the frozen parameters and the inputs of one forecast:
        frozen_long_run  (the owner's choice) B0, the long-run mix, is frozen at the end
                         of development; B1's recency-weighted counts, the
                         volatility, the run and every feature come from the
                         inference window only.
        window_only      B0 too is recomputed from the window: a one-year mix
                         instead of a long-run one, pooled over the stocks
                         whose windows were read (so every model stock's
                         window is required). For synthetic checks only: a
                         real model is refused under it (frozen_model.check).
    Either way the window drops the part of B1's recency weight older than 252
    sessions, lambda^252 with lambda = 2^(-1/H): 1/16 at H = 63, but 1/4 at
    H = 126 and 1/2 at H = 252, the half-life being the folds' majority choice.
    The record keeps that share (`window_tail_weight`) and a development-only
    check of what the truncation does (`state_check`, reported only): the
    windowed B1 under each rule against the full-history B1 on the last
    development sessions. The registration records which rule is used.
    Continuing development state through every protected session would read
    more than the 252-session window, outside decision A2.

Loading (`load`) re-checks the id, refuses another feature version or
another environment (Python minor version, numpy, pandas, scikit-learn), and
rebuilds the model; `probabilities` gives a frozen model's forecast at a
stock's last session with a close, for its estimator or for B1.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata

import numpy as np
import pandas as pd

from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import frozen_model as contract
from stock_agent.stock_lab import logistic, monitoring, scores
from stock_agent.stock_lab.contracts import DEVELOPMENT_END, StableBand
from stock_agent.stock_lab.evaluation import (
    SIMPLEST_FIRST,
    EvaluationConfig,
    _frequency_state,
    build_panel,
    history_sha256,
)
from stock_agent.stock_lab.features import FEATURE_VERSION
from stock_agent.stock_lab.frozen_model import (
    BENCHMARK,
    FIELDS,
    FROZEN_LONG_RUN,
    SCHEMA,
    SCHEMA_VERSION,
    STATE_RULES,
    WINDOW_ONLY,
    WINDOW_SESSIONS,
    RegistryRefused,
    canonical,
    check,
    content_sha256,
    model_id,
    parameters_sha256,
    plain,
    version,
)
from stock_agent.stock_lab.predict import FinalModel, _probabilities, fit_final
from stock_agent.stock_lab.reversal import after_streak
from stock_agent.stock_lab.targets import HORIZONS

__all__ = [
    "FIELDS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STATE_RULES",
    "FrozenModel",
    "RegistryRefused",
    "canonical",
    "check",
    "environment",
    "evaluation_sha256",
    "freeze",
    "load",
    "model_id",
    "probabilities",
]

DEFAULT_STATE_RULE = FROZEN_LONG_RUN
REVERSAL_DIRECTIONS = ("increase", "decrease")
REVERSAL_LENGTHS = tuple(range(1, 8))  # the page's streak lengths
ROUND_TRIP_TOLERANCE = 1e-12
STATE_CHECK_SESSIONS = 252  # the last development sessions the state check covers
CODE_COLUMNS = {"daily": "category_code", "cumulative": "cum_code"}


def environment() -> dict[str, str]:
    """What a frozen model's forecasts depend on beyond the code: Python and three packages."""

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scikit-learn": metadata.version("scikit-learn"),
    }


def evaluation_sha256(evaluation: Mapping) -> str:
    """The evaluation's content hash, as the store records it."""

    return content_sha256(evaluation)


def _panel_sha256(panel: pd.DataFrame) -> str:
    """As evaluation.evaluate records it."""

    hashed = pd.util.hash_pandas_object(panel[["symbol", "date", "price_return"]], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _last_rows(panel: pd.DataFrame) -> dict[str, int]:
    """Each stock's last row with a close."""

    closes = panel["close"].notna().to_numpy()
    symbols = panel["symbol"].to_numpy(dtype=object)
    return {
        symbol: int(np.flatnonzero((symbols == symbol) & closes)[-1])
        for symbol in dict.fromkeys(symbols)
    }


# --- the frozen model in memory ---


@dataclass(frozen=True, eq=False)
class FrozenM1:
    """M1 rebuilt from its stored coefficients (the fitted model's own predict_proba)."""

    prep: logistic.Prepared
    classes: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    c: float

    def predict(self, inputs: pd.DataFrame, b3: np.ndarray, symbols) -> np.ndarray:
        x = logistic._design(inputs, b3, list(symbols), self.prep)
        z = x @ self.coef.T + self.intercept
        if len(self.classes) == 2:
            exponent = np.exp(-np.abs(z[:, 0]))  # the logistic function, without overflow
            high = np.where(z[:, 0] >= 0, 1.0 / (1.0 + exponent), exponent / (1.0 + exponent))
            proba = np.column_stack([1.0 - high, high])
        else:
            z = z - z.max(axis=1, keepdims=True)
            weights = np.exp(z)
            proba = weights / weights.sum(axis=1, keepdims=True)
        out = np.zeros((len(x), 3))
        out[:, self.classes.astype(int)] = proba
        return out


@dataclass(frozen=True, eq=False)
class FrozenModel:
    record: Mapping
    model_id: str
    version: str
    estimator: str
    state_rule: str
    data_class: str
    stocks: tuple[str, ...]
    final: FinalModel
    long_run: Mapping[str, Mapping[str, np.ndarray]]
    reversal: Mapping
    drift_reference: Mapping


@dataclass(frozen=True, eq=False)
class Probabilities:
    daily: np.ndarray  # (5, 3): Decrease, Stable, Increase per target session
    cumulative: np.ndarray  # (3,): the five-session total
    usual: np.ndarray  # (3,): B0, the long-run mix the forecast is read against
    row: int  # the panel row (the stock's last session with a close)


# --- freezing ---


def _version(text: object) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(text).split("."))
    except ValueError:
        return (0,)


def _consistent(selection: Mapping) -> bool:
    """
    Whether a recorded selection follows from its own recorded evidence: B1 when nothing beat
    it; otherwise the simplest challenger that beat B1, replaced by each step that moved.
    """

    beating = list(selection.get("beating_reference") or [])
    if not beating:
        return selection["selected"] == BENCHMARK and not selection.get("steps")
    order = list(SIMPLEST_FIRST)
    if any(model not in order for model in beating):
        return False
    current = min(beating, key=order.index)
    for step in selection.get("steps") or []:
        if step.get("against") != current or step.get("challenger") not in beating:
            return False
        if step.get("moved"):
            current = step["challenger"]
    return selection["selected"] == current


def _confirmed_rule(evaluation: Mapping) -> str | None:
    """The evaluation's decision rule when the owner confirmed it and the record is whole."""

    rule = evaluation.get("decision_rule")
    selection = evaluation["selection"]
    if rule is None or rule not in contract.CONFIRMED_DECISION_RULES:
        return None
    whole = (
        _version(evaluation.get("schema_version")) >= (1, 3)
        and selection.get("decision_rule") == rule
        and selection.get("decision_test") == "fold_block_t"
    )
    if not whole:
        raise RegistryRefused(
            "inconsistent_selection",
            "the evaluation names a confirmed rule its selection does not record",
        )
    if not _consistent(selection):
        raise RegistryRefused(
            "inconsistent_selection",
            "the recorded selection does not follow from its recorded comparisons and steps",
        )
    return rule


def _choose(evaluation: Mapping, requested: str | None) -> tuple[str, dict]:
    selected = evaluation["selection"]["selected"]
    rule = _confirmed_rule(evaluation)
    confirmed = rule is not None
    if confirmed:
        chosen = selected if requested is None else requested
        if chosen != selected:
            raise RegistryRefused(
                "not_the_selection",
                f"the evaluation selected {selected} under the confirmed rule; {chosen} is not it",
            )
        why = f"selected by the evaluation under the confirmed decision rule {rule!r}"
    else:
        chosen = BENCHMARK if requested is None else requested
        if chosen != BENCHMARK:
            raise RegistryRefused(
                "a7_unresolved",
                "the evaluation records no confirmed rule for whether a challenger beats B1 "
                "(decision A7), so only B1 may be frozen",
            )
        why = (
            "The evaluation records no confirmed rule for whether a challenger beats B1 (it "
            "predates decision A7), so B1, the pre-registered default, is frozen; the "
            "evaluation's own selection is reported only."
        )
    return chosen, {
        "evaluation_selected": selected,
        "evaluation_rule": evaluation["selection"]["rule"],
        "decision_rule": rule if confirmed else None,
        "frozen": chosen,
        "why": why,
    }


def _check_inputs(
    histories: Mapping[str, pd.DataFrame], evaluation: Mapping, config: EvaluationConfig
) -> pd.DataFrame:
    data_class = evaluation["data_class"]
    demo = {"_" in symbol for symbol in histories}
    if demo != {data_class == "synthetic"}:
        raise RegistryRefused(
            "data_class_mismatch",
            "real models cover real tickers only, synthetic ones demo identifiers only",
        )
    if data_class == "real":
        end = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
        late = [s for s, bars in histories.items() if (bars["session"] > end).any()]
        if late:
            raise RegistryRefused(
                "protected_period",
                f"{', '.join(sorted(late))}: sessions after {DEVELOPMENT_END} are protected and "
                "never used to fit a model",
            )
        if config.outcome_end is None or config.outcome_end.date() != DEVELOPMENT_END:
            raise RegistryRefused(
                "outcome_cutoff", f"a real model's outcomes must be censored at {DEVELOPMENT_END}"
            )
    if config.as_dict() != evaluation["config"]:
        raise RegistryRefused("config_mismatch", "the settings differ from the evaluation's")
    if sorted(histories) != list(evaluation["stocks"]):
        raise RegistryRefused("histories_mismatch", "the stocks differ from the evaluation's")
    recorded = evaluation.get("histories_sha256")
    if recorded is None:
        raise RegistryRefused(
            "evaluation_without_history_hashes",
            "the evaluation predates schema 1.2: re-run it, so the bars it used are recorded",
        )
    if {s: history_sha256(bars) for s, bars in histories.items()} != recorded:
        raise RegistryRefused("histories_mismatch", "the bars differ from the evaluation's")
    panel = build_panel(histories, config)
    if _panel_sha256(panel) != evaluation["panel_sha256"]:
        raise RegistryRefused("histories_mismatch", "the prices differ from the evaluation's")
    return panel


def _long_run(panel: pd.DataFrame, final: FinalModel, rows: Mapping[str, int]) -> dict:
    config = final.config
    params = est.FrequencyParams(config.alpha_prior, final.half_life, config.kappa)
    out = {}
    for name, column in CODE_COLUMNS.items():
        b0 = _frequency_state(panel, column, params)[0]
        out[name] = {symbol: b0[row].tolist() for symbol, row in rows.items()}
    return out


def _errors(final: FinalModel) -> dict:
    def one(distribution: est.ErrorDistribution) -> dict:
        return {
            "kappa": distribution.kappa,
            "by_stock": {s: values.tolist() for s, values in distribution.by_stock.items()},
        }

    return {
        **{str(h): one(final.errors[h]) for h in HORIZONS},
        "cum": one(final.cumulative_errors),
    }


def _m1(final: FinalModel) -> dict:
    out = {}
    for target, fitted in final.m1.items():
        out[target] = {
            "c": fitted.c,
            "classes": fitted.model.classes_.tolist(),
            "coef": fitted.model.coef_.tolist(),
            "intercept": fitted.model.intercept_.tolist(),
            "medians": fitted.prep.medians.tolist(),
            "means": fitted.prep.means.tolist(),
            "scales": fitted.prep.scales.tolist(),
            "symbols": list(fitted.prep.symbols),
        }
    return out


def _reversal(panel: pd.DataFrame, config: EvaluationConfig) -> dict:
    """What followed runs of 1 to 7 sessions, per stock, in development history."""

    out = {}
    for symbol in dict.fromkeys(panel["symbol"]):
        stock = panel[panel["symbol"] == symbol].reset_index(drop=True)
        out[symbol] = {}
        for direction in REVERSAL_DIRECTIONS:
            out[symbol][direction] = {}
            for k in REVERSAL_LENGTHS:
                summary = after_streak(
                    stock, k, direction, config.band, outcome_end=config.outcome_end
                )
                rows = summary.table[
                    (summary.table["horizon"] == 1) & (summary.table["regime"] == "all")
                ]
                out[symbol][direction][str(k)] = {
                    "definition": summary.definition,
                    "events": summary.events,
                    "outcomes": {
                        r["outcome"]: {
                            "estimate": r["estimate"],
                            "n_eff": r["n_eff"],
                            "status": r["status"],
                        }
                        for r in rows.to_dict("records")
                    },
                }
    return out


def _drift_reference(panel: pd.DataFrame) -> dict:
    return {
        feature: monitoring.drift_reference(panel[feature].to_numpy(dtype="float64"))
        for feature in monitoring.DRIFT_FEATURES
    }


def freeze(
    histories: Mapping[str, pd.DataFrame],
    evaluation: Mapping,
    config: EvaluationConfig,
    *,
    created_at: pd.Timestamp,
    estimator: str | None = None,
    state_rule: str = DEFAULT_STATE_RULE,
    code: Mapping | None = None,
    lineage: Mapping | None = None,
) -> dict:
    """
    The checked frozen-model record (module docstring); RegistryRefused when refused.
    `code` is the commit and whether the checkout was clean ({"sha", "clean"}), when known.
    """

    if state_rule not in STATE_RULES:
        raise RegistryRefused("unknown_state_rule", f"{state_rule!r} is not one of {STATE_RULES}")
    panel = _check_inputs(histories, evaluation, config)
    chosen, selection = _choose(evaluation, estimator)
    fit_record = {**evaluation, "selection": {**evaluation["selection"], "selected": chosen}}
    final = fit_final(panel, config, fit_record)
    rows = _last_rows(panel)
    data_class = evaluation["data_class"]
    notes = [
        (
            "Frozen once; never refitted, retuned, reselected or recalibrated. A change is a "
            "new model with a new id."
        ),
        f"State rule {state_rule!r}: see stock_agent.stock_lab.registry (decision A2, item 4).",
    ]
    if data_class == "real":
        notes.append(
            "Fitted on development history through 2024-12-31 of a watchlist chosen with "
            "hindsight: exploratory (decision A3), not evidence of skill."
        )
    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": pd.Timestamp(created_at).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "data_class": data_class,
        "estimator": chosen,
        "benchmark": BENCHMARK,
        "selection": selection,
        "state_rule": state_rule,
        "window_sessions": WINDOW_SESSIONS,
        "window_tail_weight": 2.0 ** (-WINDOW_SESSIONS / final.half_life),
        "state_check": state_check(panel, final),
        "evaluation_run_id": evaluation["run_id"],
        "evaluation_sha256": evaluation_sha256(evaluation),
        "config": config.as_dict(),
        "half_life": final.half_life,
        "feature_version": FEATURE_VERSION,
        "outcome_end": None if config.outcome_end is None else str(config.outcome_end.date()),
        "stocks": sorted(histories),
        "long_run": _long_run(panel, final, rows),
        "tilt": None
        if chosen != "B2"
        else {
            "theta": final.tilt.theta.tolist(),
            "ridge": final.tilt.ridge,
            "rows_per_state": list(final.tilt.rows_per_state),
        },
        "errors": _errors(final) if chosen in ("B3", "M1") else None,
        "m1": _m1(final) if chosen == "M1" else None,
        "reversal": _reversal(panel, config),
        "drift_reference": _drift_reference(panel),
        "data": {
            "panel_sha256": evaluation["panel_sha256"],
            "histories": {
                symbol: {
                    "first_session": str(bars["session"].iloc[0].date()),
                    "last_session": str(bars["session"].iloc[-1].date()),
                    "rows": len(bars),
                    "bars_sha256": history_sha256(bars),
                }
                for symbol, bars in sorted(histories.items())
            },
            "lineage": None if lineage is None else dict(lineage),
        },
        "environment": environment(),
        "code": None if code is None else dict(code),
        "notes": notes,
    }
    record = plain(record)
    record["parameters_sha256"] = parameters_sha256(record)
    record["model_id"] = model_id(record)
    check(record)
    _round_trip(record, panel, final, rows)
    return record


def state_check(panel: pd.DataFrame, final: FinalModel) -> dict:
    """
    What the 252-session window does to B1, on development data only (reported, never used to
    select or tune): on each stock's last STATE_CHECK_SESSIONS development sessions that have a
    full window behind them, B1 from the trailing window against B1 from the full history.

        frozen_long_run  window counts, with the prior pointing at the long-run mix as of that
                         session (B0 through it: the analogue of the mix frozen at the end of
                         development)
        window_only      window counts, with B0 recomputed from the window (pooled over every
                         stock's rows in the window's sessions)

    Scored on each session's next-session outcome where it matured (RPS), with the mean and
    largest probability differences from the full-history B1.
    """

    config = final.config
    params = est.FrequencyParams(config.alpha_prior, final.half_life, config.kappa)
    b0_full, b1_full, _ = _frequency_state(panel, "category_code", params)
    codes = panel["category_code"].to_numpy(dtype="float64", na_value=np.nan)
    outcome = panel["target_1"].to_numpy(dtype="float64", na_value=np.nan)
    symbols = panel["symbol"].to_numpy(dtype=object)
    days = pd.DatetimeIndex(panel["date"])
    onehot = np.stack([(codes == c).astype("float64") for c in range(est.K)], axis=1)
    unique_days, day_index = np.unique(np.asarray(days), return_inverse=True)
    pooled = np.cumsum(
        np.stack([np.bincount(day_index, weights=onehot[:, c], minlength=len(unique_days))
                  for c in range(est.K)], axis=1),
        axis=0,
    )  # fmt: skip
    rows, full, rules = [], [], {FROZEN_LONG_RUN: [], WINDOW_ONLY: []}
    for symbol in dict.fromkeys(symbols):
        mine = np.flatnonzero(symbols == symbol)
        for i in range(max(WINDOW_SESSIONS - 1, len(mine) - STATE_CHECK_SESSIONS), len(mine)):
            row, window = mine[i], mine[i - WINDOW_SESSIONS + 1 : i + 1]
            first, last = day_index[window[0]], day_index[row]
            pool = pooled[last] - (pooled[first - 1] if first > 0 else 0.0)
            share = pool / pool.sum() if pool.sum() > 0 else np.full(est.K, 1.0 / est.K)
            own = onehot[window].sum(axis=0)
            b0_window = (own + config.alpha_prior * share) / (own.sum() + config.alpha_prior)
            for rule, prior in ((FROZEN_LONG_RUN, b0_full[row]), (WINDOW_ONLY, b0_window)):
                rules[rule].append(
                    _recency_weighted(codes[window], final.half_life, config.kappa, prior)
                )
            rows.append(row)
            full.append(b1_full[row])
    if not rows:
        return {"rows": 0, "role": "reported only; development data; never used to select"}
    rows, full = np.array(rows), np.array(full)
    y = outcome[rows]
    scored = np.isfinite(y)
    out = {
        "role": "reported only; development data; never used to select",
        "sessions": STATE_CHECK_SESSIONS,
        "window_sessions": WINDOW_SESSIONS,
        "rows": len(rows),
        "scored_rows": int(scored.sum()),
        "dates": len(set(days[rows])),
        "first_session": str(days[rows].min().date()),
        "last_session": str(days[rows].max().date()),
        "rps": {"full_history": float(np.mean(scores.rps(full[scored], y[scored])))},
        "rps_difference": {},
        "mean_abs_difference": {},
        "max_abs_difference": {},
    }
    for rule, values in rules.items():
        values = np.array(values)
        gap = np.abs(values - full).max(axis=1)
        rps = scores.rps(values[scored], y[scored])
        out["rps"][rule] = float(np.mean(rps))
        out["rps_difference"][rule] = float(np.mean(rps - scores.rps(full[scored], y[scored])))
        out["mean_abs_difference"][rule] = float(gap.mean())
        out["max_abs_difference"][rule] = float(gap.max())
    return out


def _round_trip(record: Mapping, panel: pd.DataFrame, final: FinalModel, rows) -> None:
    model = load(record)
    for symbol, row in rows.items():
        for estimator in dict.fromkeys((model.estimator, BENCHMARK)):
            daily, cumulative, usual = _probabilities(panel, row, final, estimator)
            got = probabilities(panel, symbol, model, estimator)
            for name, expected, value in (
                ("daily", daily, got.daily),
                ("cumulative", cumulative, got.cumulative),
                ("long-run mix", usual, got.usual),
            ):
                if not np.allclose(value, expected, rtol=0, atol=ROUND_TRIP_TOLERANCE):
                    raise RegistryRefused(
                        "round_trip_failed",
                        f"{symbol} {estimator}: the loaded model's {name} differs from the fit",
                    )


# --- loading ---


def _config(stored: Mapping) -> EvaluationConfig:
    """The settings a forecast uses (the rest are the evaluation's and do not change one)."""

    end = stored["outcome_end"]
    return EvaluationConfig(
        band=StableBand(stored["band"]),
        alpha_prior=stored["alpha_prior"],
        kappa=stored["kappa"],
        tilt_ridge=stored["tilt_ridge"],
        error_kappa=stored["error_kappa"],
        outcome_end=None if end is None else pd.Timestamp(end, tz="UTC"),
    )


def _distribution(horizon: int, stored: Mapping) -> est.ErrorDistribution:
    by_stock = {s: np.asarray(values, dtype="float64") for s, values in stored["by_stock"].items()}
    pooled = np.sort(np.concatenate([*by_stock.values(), np.array([])]))
    return est.ErrorDistribution(horizon, pooled, by_stock, stored["kappa"])


def load(record: Mapping) -> FrozenModel:
    """The frozen model, after the checks in the module docstring."""

    check(record)
    if record["feature_version"] != FEATURE_VERSION:
        raise RegistryRefused(
            "feature_version_mismatch",
            f"the model's features are {record['feature_version']}, the code's {FEATURE_VERSION}",
        )
    current = environment()
    if record["environment"] != current:
        changed = sorted(
            key
            for key in set(current) | set(record["environment"])
            if current.get(key) != record["environment"].get(key)
        )
        raise RegistryRefused(
            "environment_mismatch", f"{', '.join(changed)} differ from the frozen environment"
        )
    config = _config(record["config"])
    tilt = est.Tilt.zero()
    if record["tilt"] is not None:
        stored = record["tilt"]
        tilt = est.Tilt(
            np.asarray(stored["theta"], dtype="float64"),
            ridge=stored["ridge"],
            rows_per_state=tuple(stored["rows_per_state"]),
        )
    errors, cumulative_errors = {}, est.ErrorDistribution(5, np.array([]), {})
    if record["errors"] is not None:
        errors = {h: _distribution(h, record["errors"][str(h)]) for h in HORIZONS}
        cumulative_errors = _distribution(5, record["errors"]["cum"])
    m1 = {}
    for target, stored in (record["m1"] or {}).items():
        prep = logistic.Prepared(
            np.asarray(stored["medians"], dtype="float64"),
            np.asarray(stored["means"], dtype="float64"),
            np.asarray(stored["scales"], dtype="float64"),
            tuple(stored["symbols"]),
        )
        m1[target] = FrozenM1(
            prep,
            np.asarray(stored["classes"]),
            np.asarray(stored["coef"], dtype="float64"),
            np.asarray(stored["intercept"], dtype="float64"),
            stored["c"],
        )
    final = FinalModel(
        config, float(record["half_life"]), tilt, errors, cumulative_errors, m1, version(record)
    )
    long_run = {
        name: {s: np.asarray(v, dtype="float64") for s, v in mixes.items()}
        for name, mixes in record["long_run"].items()
    }
    return FrozenModel(
        record=record,
        model_id=record["model_id"],
        version=version(record),
        estimator=record["estimator"],
        state_rule=record["state_rule"],
        data_class=record["data_class"],
        stocks=tuple(record["stocks"]),
        final=final,
        long_run=long_run,
        reversal=record["reversal"],
        drift_reference=record["drift_reference"],
    )


# --- a frozen model's forecast ---


def _recency_weighted(codes: np.ndarray, half_life: float, kappa: float, prior: np.ndarray):
    sums, weight, _ = est.ew_state(codes, half_life)
    return (sums[-1] + kappa * prior) / (weight[-1] + kappa)


def _frequencies(panel: pd.DataFrame, row: int, model: FrozenModel):
    """(B0, B1, B1 of the total) at `row` under the model's state rule (module docstring)."""

    config, half_life = model.final.config, model.final.half_life
    symbol = panel["symbol"].iloc[row]
    params = est.FrequencyParams(config.alpha_prior, half_life, config.kappa)
    mine = np.flatnonzero((panel["symbol"] == symbol).to_numpy())
    mine = mine[mine <= row]
    out = []
    for name, column in CODE_COLUMNS.items():
        if model.state_rule == WINDOW_ONLY:
            prior = _frequency_state(panel, column, params)[0][row]
        else:
            prior = model.long_run[name][symbol]
        codes = panel[column].to_numpy(dtype="float64", na_value=np.nan)[mine]
        out.append((prior, _recency_weighted(codes, half_life, config.kappa, prior)))
    (b0, b1), (_, b1_cum) = out
    return b0, b1, b1_cum


def probabilities(
    panel: pd.DataFrame, symbol: str, model: FrozenModel, estimator: str | None = None
) -> Probabilities:
    """The frozen model's forecast (its estimator, or B1) at `symbol`'s last session with a close."""

    if symbol not in model.stocks:
        raise RegistryRefused("stock_not_in_model", f"{symbol} is not in the frozen model")
    if model.state_rule == WINDOW_ONLY and set(panel["symbol"]) != set(model.stocks):
        raise RegistryRefused(
            "panel_stocks_mismatch",
            "under window_only, B0 pools every model stock's window: all of them are needed",
        )
    estimator = model.estimator if estimator is None else estimator
    if estimator not in (model.estimator, BENCHMARK):
        raise RegistryRefused("estimator_not_frozen", f"{estimator} is not frozen in this model")
    rows = np.flatnonzero(
        (panel["symbol"] == symbol).to_numpy() & panel["close"].notna().to_numpy()
    )
    if not len(rows):
        raise RegistryRefused("no_close", f"no close for {symbol}")
    row = int(rows[-1])
    frequencies = _frequencies(panel, row, model)
    daily, cumulative, usual = _probabilities(panel, row, model.final, estimator, frequencies)
    return Probabilities(daily, cumulative, usual, row)
