"""
The sealed holdout run (`protocol` has the rules): the frozen model's
forecasts at every decision session, scored against B0 on the outcomes that
followed. Pure: bars in, a sealed record out. No store, clock or network;
the store (`HoldoutStore.evaluate_once`) calls it once and keeps the record.

Forecasts, computed as the evening job computes them
    At decision d, each stock's window is its 252 fetched sessions ending at
    d; the windows of the stocks with a close at d form the panel
    (`evaluation.build_panel` with the model's own configuration), and
    `registry.probabilities` gives the model's forecast at d: the five daily
    targets and the five-session total, normalized to sum to one and refused
    without a finite volatility, as `frozen_forecast.forecast_record` does.
    B0 is the model's frozen long-run mix for the stock
    (`FrozenModel.long_run`), the same at every decision. A stock without a
    close at d gets no forecast (as_of_close_missing). The data path differs
    from the evening job's: its reader refuses a window reaching back before
    2025, while here the first windows hold 2024 development sessions from
    the holdout fetch.

Outcomes
    The categories of the sessions after d (`build_targets` through
    `evaluation.build_panel` on the whole fetched history, with outcomes
    allowed through HOLDOUT_END). A target whose sessions include a gap is
    unavailable and not scored, for the model and B0 alike.

The deciding comparison (protocol docstring): the pooled daily RPS
differential, per decision the mean over stocks and then over the five daily
targets, tested with the fold-block t over the protocol's blocks
(`model_diagnostics.inference.fold_block_t`), one-sided. The other scores and
the five-session total get the same test, reported only.

Leakage guard: a forecast reads only its window (sessions up to d); outcomes
come from sessions after d; nothing is fitted. Each forecast's panel row is
checked to be d.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from stock_agent.model_diagnostics.inference import fold_block_t
from stock_agent.stock_lab import registry, scores
from stock_agent.stock_lab.briefing.reliability import calibration_buckets, lags_for
from stock_agent.stock_lab.evaluation import build_panel
from stock_agent.stock_lab.frozen_model import plain
from stock_agent.stock_lab.holdout.protocol import (
    HoldoutRefused,
    Protocol,
    Refusal,
    Sessions,
    record_id,
)
from stock_agent.stock_lab.size_simulation import one_sided_p
from stock_agent.stock_lab.store import stamp
from stock_agent.stock_lab.targets import HORIZONS

SCHEMA = "stock_lab_holdout_evaluation"
SCHEMA_VERSION = "1.0"
COMPARATOR = "B0"
DAILY = tuple(f"target_{h}" for h in HORIZONS)
TOTAL = "target_cum"

CATEGORIES = ("decrease", "stable", "increase")
NO_CLOSE = "as_of_close_missing"
NO_VOLATILITY = "insufficient_history"  # frozen_forecast's refusal without a finite volatility
INCOMPLETE_BLOCKS = "incomplete_blocks"
LIMITATIONS = (
    "The four stocks were chosen in hindsight, after their run of strong years.",
    "One 21-month period: a single draw of market conditions, with low power.",
    (
        "Prices are raw closes and corporate actions as served on the fetch date, not "
        "point-in-time vintages."
    ),
    (
        "A result close to B0 is the expected outcome. A 'yes' would be evidence against B0 "
        "for this period and these stocks, not of a lasting edge."
    ),
    "The holdout is spent: from now on it serves fixed-rule descriptive backtests only.",
)


def _check(histories: Mapping[str, pd.DataFrame], model, protocol: Protocol, days) -> None:
    if model.model_id != protocol.model_id:
        raise HoldoutRefused(
            Refusal.MODEL, f"the protocol evaluates {protocol.model_id[:12]}, not this model"
        )
    if model.estimator == COMPARATOR or protocol.comparator != COMPARATOR:
        raise HoldoutRefused(Refusal.MODEL, f"the model is compared with {COMPARATOR}, not itself")
    if model.record.get("window_sessions") != protocol.window_sessions:
        raise HoldoutRefused(Refusal.MODEL, "the model's window is not the protocol's")
    if not set(protocol.tickers) <= set(model.stocks):
        raise HoldoutRefused(Refusal.MODEL, "a protocol stock is not in the frozen model")
    if set(histories) != set(protocol.tickers):
        raise HoldoutRefused(Refusal.INCOMPLETE, "the histories are not the protocol's stocks")
    for ticker, bars in histories.items():
        if not pd.DatetimeIndex(bars["session"]).as_unit("us").equals(days):
            raise HoldoutRefused(
                Refusal.INTEGRITY, f"{ticker}'s bars are not the protocol's sessions"
            )


def _forecasts(histories, model, protocol: Protocol, days, decisions):
    """Per (decision, stock) with a close: the model's and B0's probabilities."""

    config = model.final.config
    width = protocol.window_sessions
    position = {day: index for index, day in enumerate(days)}
    rows, skipped = [], Counter()
    for decision in decisions:
        end = position[decision] + 1
        windows = {
            ticker: histories[ticker].iloc[end - width : end].reset_index(drop=True)
            for ticker in protocol.tickers
        }
        ready = [t for t in protocol.tickers if math.isfinite(windows[t]["close"].iloc[-1])]
        skipped[NO_CLOSE] += len(protocol.tickers) - len(ready)
        if not ready:
            continue
        panel = build_panel({ticker: windows[ticker] for ticker in ready}, config)
        for ticker in ready:
            found = registry.probabilities(panel, ticker, model)
            last = panel.iloc[found.row]
            if pd.Timestamp(last["session"]) != decision:
                raise HoldoutRefused(Refusal.INTEGRITY, f"{ticker}'s forecast is not at {decision}")
            if not np.isfinite(last["ewma_vol"]):
                skipped[NO_VOLATILITY] += 1
                continue
            daily = np.asarray(found.daily, dtype="float64")
            total = np.asarray(found.cumulative, dtype="float64")
            if not (np.isfinite(daily).all() and np.isfinite(total).all()):
                # the evening job would refuse this forecast; the one run stops instead of
                # scoring a gap, and nothing is recorded
                raise HoldoutRefused(
                    Refusal.INTEGRITY, f"{ticker}'s forecast at {decision.date()} is not finite"
                )
            rows.append(
                {
                    "date": decision,
                    "ticker": ticker,
                    "daily": daily / daily.sum(axis=1, keepdims=True),
                    "total": total / total.sum(),
                    "b0_daily": np.asarray(found.usual, dtype="float64"),
                    "b0_total": np.asarray(model.long_run["cumulative"][ticker], dtype="float64"),
                }
            )
    return rows, skipped


def _outcomes(histories, model, protocol: Protocol) -> pd.DataFrame:
    """Every (stock, session)'s target categories, outcomes allowed through HOLDOUT_END."""

    config = replace(model.final.config, outcome_end=pd.Timestamp(protocol.holdout_end, tz="UTC"))
    panel = build_panel(dict(histories), config)
    panel["key_date"] = pd.DatetimeIndex(panel["session"]).as_unit("us")
    columns = [*DAILY, TOTAL, "category_code", "target_end_date"]
    return panel.set_index(["symbol", "key_date"])[columns]


def _codes(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")


def _target_arrays(forecasts, outcomes: pd.DataFrame, target: str):
    """(model probabilities, B0 probabilities, outcomes, dates, tickers) of one target."""

    keys = [(row["ticker"], row["date"]) for row in forecasts]
    y = _codes(outcomes.loc[keys, target].to_numpy()) if keys else np.empty(0)
    if target == TOTAL:
        model_p = np.array([row["total"] for row in forecasts]).reshape(-1, 3)
        b0_p = np.array([row["b0_total"] for row in forecasts]).reshape(-1, 3)
    else:
        h = DAILY.index(target)
        model_p = np.array([row["daily"][h] for row in forecasts]).reshape(-1, 3)
        b0_p = np.array([row["b0_daily"] for row in forecasts]).reshape(-1, 3)
    dates = pd.DatetimeIndex([row["date"] for row in forecasts])
    tickers = np.array([row["ticker"] for row in forecasts], dtype=object)
    valid = np.isfinite(y)
    return model_p[valid], b0_p[valid], y[valid].astype(int), dates[valid], tickers[valid]


def _test(series: pd.Series, protocol: Protocol) -> dict:
    labels = protocol.block_labels(pd.DatetimeIndex(series.index))
    found = fold_block_t(series.to_numpy(dtype="float64"), labels, ci_level=protocol.interval_level)
    block_means = series.groupby(labels).agg(["mean", "size"])
    return {
        "dates": len(series),
        "mean_diff": float(series.mean()) if len(series) else math.nan,
        "block_mean_diff": found.estimate,
        "se": found.se,
        "ci_low": found.ci_low,
        "ci_high": found.ci_high,
        "ci_level": protocol.interval_level,
        "df": found.params.get("df"),
        "blocks": found.params.get("blocks"),
        "status": found.status,
        "reason": found.reason,
        "p_one_sided": one_sided_p(found),
        "by_block": [
            {
                "block": name,
                "dates": int(block_means.loc[name, "size"]),
                "mean_diff": float(block_means.loc[name, "mean"]),
            }
            for name, _ in protocol.blocks
            if name in block_means.index
        ],
    }


def _per_date(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values).groupby(dates).mean()


def _scored(forecasts, outcomes, protocol: Protocol, estimator: str):
    """Metrics, comparisons, the per-stock and per-year breakdowns, and calibration."""

    metrics, by_stock, by_year, calibration = [], [], [], []
    differences: dict[tuple[str, str], list[pd.Series]] = {}
    scored_rows = {}
    pooled: dict[str, dict[str, list]] = {}
    first_year = pd.Timestamp(protocol.holdout_start).year
    for target in (*DAILY, TOTAL):
        model_p, b0_p, y, dates, tickers = _target_arrays(forecasts, outcomes, target)
        group = TOTAL if target == TOTAL else "daily_pooled"
        scored_rows[target] = len(y)
        years = np.maximum(dates.year.to_numpy(), first_year)
        for name in protocol.scores:
            function = scores.SCORES[name]
            mine, theirs = function(model_p, y), function(b0_p, y)
            for label, values in ((estimator, mine), (COMPARATOR, theirs)):
                metrics.append(
                    {
                        "estimator": label,
                        "target": target,
                        "score": name,
                        "n": len(y),
                        "dates": int(dates.nunique()),
                        "mean": float(values.mean()) if len(y) else math.nan,
                    }
                )
            differences.setdefault((group, name), []).append(_per_date(mine - theirs, dates))
            bucket = pooled.setdefault(group, {})
            bucket.setdefault(name, []).append((mine, theirs, tickers, years))
        for c, category in enumerate(CATEGORIES):
            buckets = calibration_buckets(
                model_p[:, c],
                (y == c).astype("float64"),
                dates.strftime("%Y-%m-%d").to_numpy(),
                lags=lags_for("cum" if target == TOTAL else target.removeprefix("target_")),
                level=protocol.calibration_level,
            )
            for found in buckets:
                calibration.append({"target": target, "category": category, **asdict(found)})
    comparisons = []
    for (group, name), series in differences.items():
        combined = pd.concat(series, axis=1).mean(axis=1) if len(series) > 1 else series[0]
        deciding = group == protocol.deciding_target and name == protocol.deciding_score
        comparisons.append(
            {
                "estimator": estimator,
                "vs": COMPARATOR,
                "target": group,
                "score": name,
                "deciding": deciding,
                **_test(combined.sort_index(), protocol),
            }
        )
    for group, by_score in pooled.items():
        for name, parts in by_score.items():
            mine = np.concatenate([p[0] for p in parts])
            theirs = np.concatenate([p[1] for p in parts])
            tickers = np.concatenate([p[2] for p in parts])
            years = np.concatenate([p[3] for p in parts])
            for key, labels, out in (("stock", tickers, by_stock), ("year", years, by_year)):
                for value in sorted(set(labels.tolist())):
                    chosen = labels == value
                    out.append(
                        {
                            key: value,
                            "target": group,
                            "score": name,
                            "n": int(chosen.sum()),
                            estimator: float(mine[chosen].mean()),
                            COMPARATOR: float(theirs[chosen].mean()),
                            "diff": float((mine[chosen] - theirs[chosen]).mean()),
                        }
                    )
    return metrics, comparisons, by_stock, by_year, calibration, scored_rows


def _base_rates(outcomes: pd.DataFrame, model, protocol: Protocol) -> list[dict]:
    """Each stock's holdout share of each category, by session, against the frozen B0."""

    rows = []
    start = pd.Timestamp(protocol.holdout_start, tz="UTC")
    end = pd.Timestamp(protocol.holdout_end, tz="UTC")
    for ticker in protocol.tickers:
        own = outcomes.loc[ticker]
        own = own[(own.index >= start) & (own.index <= end)]
        codes = _codes(own["category_code"].to_numpy())
        codes = codes[np.isfinite(codes)].astype(int)
        shares = np.bincount(codes, minlength=3) / max(len(codes), 1)
        frozen = np.asarray(model.long_run["daily"][ticker], dtype="float64")
        rows.append(
            {
                "stock": ticker,
                "sessions": len(codes),
                **{f"holdout_{c}": float(shares[i]) for i, c in enumerate(CATEGORIES)},
                **{f"b0_{c}": float(frozen[i]) for i, c in enumerate(CATEGORIES)},
            }
        )
    return rows


def _decision(deciding: Mapping, protocol: Protocol, estimator: str) -> dict:
    """
    The answer to the one question: the model beats B0 when every block holds scored
    decisions, the test is available, the mean of the block means is negative, and the
    one-sided p-value is below alpha. Otherwise the answer is no, or undetermined (a
    status other than ok).
    """

    p = deciding["p_one_sided"]
    status = deciding["status"]
    if status == "ok" and deciding["blocks"] != len(protocol.blocks):
        status = INCOMPLETE_BLOCKS
    beats = bool(
        status == "ok"
        and deciding["block_mean_diff"] < 0
        and math.isfinite(p)
        and p < protocol.alpha
    )
    return {
        "question": f"does {estimator} beat {COMPARATOR} on the pooled daily RPS?",
        "beats_comparator": beats,
        "block_mean_diff": deciding["block_mean_diff"],
        "p_one_sided": p,
        "alpha": protocol.alpha,
        "df": deciding["df"],
        "blocks": deciding["blocks"],
        "status": status,
        "reason": deciding["reason"],
    }


def run(
    histories: Mapping[str, pd.DataFrame],
    model: registry.FrozenModel,
    *,
    protocol: Protocol,
    sessions: Sessions,
    calendar_id: str,
    batches: Sequence[Mapping],
    created_at: pd.Timestamp,
    code: Mapping,
) -> dict:
    """The sealed record of the one holdout run (module docstring)."""

    days = protocol.fetch_sessions(sessions)
    _check(histories, model, protocol, days)
    decisions = protocol.decisions(sessions)
    forecasts, skipped = _forecasts(histories, model, protocol, days, decisions)
    outcomes = _outcomes(histories, model, protocol)
    metrics, comparisons, by_stock, by_year, calibration, scored = _scored(
        forecasts, outcomes, protocol, model.estimator
    )
    deciding = next(row for row in comparisons if row["deciding"])
    decision = _decision(deciding, protocol, model.estimator)
    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256(),
        "model": {
            "model_id": model.model_id,
            "version": model.version,
            "estimator": model.estimator,
            "state_rule": model.state_rule,
            "data_class": model.data_class,
            "comparator": COMPARATOR,
        },
        "data": {
            "calendar_id": calendar_id,
            "first_session": days[0].strftime("%Y-%m-%d"),
            "last_session": days[-1].strftime("%Y-%m-%d"),
            "sessions": len(days),
            "batches": [dict(batch) for batch in batches],
        },
        "code": dict(code),
        "environment": registry.environment(),
        "created_at": stamp(created_at),
        "counts": {
            "decisions": len(decisions),
            "first_decision": decisions[0].strftime("%Y-%m-%d"),
            "last_decision": decisions[-1].strftime("%Y-%m-%d"),
            "forecasts": len(forecasts),
            "skipped": dict(skipped),
            "scored": scored,
        },
        "decision": decision,
        "metrics": metrics,
        "comparisons": comparisons,
        "by_stock": by_stock,
        "by_year": by_year,
        "base_rates": _base_rates(outcomes, model, protocol),
        "calibration": calibration,
        "limitations": list(LIMITATIONS),
    }
    record = plain(record)
    record["record_id"] = record_id(record)
    return record
