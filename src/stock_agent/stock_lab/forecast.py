"""
The versioned forecast artifact, `StockProbabilityForecast` (schema 1.0),
and its checks (standard library only, so any reader can use it).

A forecast is made after the close of its as-of session, from data
available then, and is never edited: a recalculation is a new forecast that
names the one it supersedes. Each carries:

    forecast_id         sha256 of its canonical content (without the id)
    symbol, data_class  the stock, and "real" or "synthetic"
    as_of_session       the last session whose close it uses
    created_at          when it was made (UTC)
    target_sessions     the five sessions after the as-of session, in order
    daily               per session: P(Decrease), P(Stable), P(Increase),
                        summing to 1; the probability of the day's own move,
                        never of the move from today to that day
    cumulative          the five-session total's three probabilities (or None),
                        against its own band
    band, cumulative_band, estimator, model_version, feature_version,
    evaluation_run_id   how it was made
    data                source, batch ids, retrieval time, last close session
    evidence_status, calibration_status
    signals             the current run, the reversal statistics after it,
                        the latest trap setup (all descriptive)
    limitations         plain sentences; refusal: a reason and words when no
                        probability could be given
    supersedes          the forecast id it replaces, or None

Checks (`check`): probabilities in [0, 1] that sum to 1 within 1e-9; target
sessions strictly after the as-of session and increasing; the cumulative
forecast respects the bounds P_cum(Increase) >= sum_h p_h(Increase) - 4 and
the same for Decrease (all five sessions moving one way implies the total
moved that way; they never bind in practice, but a violation would mean an
incoherent artifact); the estimator is one of ESTIMATORS. A future
reader (for example a PPO observation) must use these records only through
an approved point-in-time reader, and must never treat a forecast rebuilt
after the fact as having been available at its as-of session.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

SCHEMA = "stock_probability_forecast"
SCHEMA_VERSION = "1.0"
TOLERANCE = 1e-9
FIELDS = (
    "schema",
    "schema_version",
    "forecast_id",
    "symbol",
    "data_class",
    "as_of_session",
    "created_at",
    "target_sessions",
    "daily",
    "cumulative",
    "band",
    "cumulative_band",
    "estimator",
    "model_version",
    "feature_version",
    "evaluation_run_id",
    "data",
    "evidence_status",
    "calibration_status",
    "signals",
    "limitations",
    "refusal",
    "supersedes",
)
PROBABILITY_KEYS = ("p_decrease", "p_stable", "p_increase")
ESTIMATORS = ("B0", "B1", "B2", "B3", "M1")


class ForecastRefused(ValueError):
    """An artifact that breaks the contract above."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def forecast_id(record: Mapping[str, object]) -> str:
    content = {key: value for key, value in record.items() if key != "forecast_id"}
    return hashlib.sha256(canonical(content).encode("utf-8")).hexdigest()


def _check_probabilities(row: Mapping[str, object], label: str) -> None:
    values = [row.get(key) for key in PROBABILITY_KEYS]
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        raise ForecastRefused(f"{label}: probabilities must be numbers")
    if not all(0.0 <= v <= 1.0 and math.isfinite(v) for v in values):
        raise ForecastRefused(f"{label}: probabilities must be in [0, 1]")
    if abs(sum(values) - 1.0) > TOLERANCE:
        raise ForecastRefused(f"{label}: probabilities must sum to 1")


def check(record: Mapping[str, object]) -> Mapping[str, object]:
    """The record, after every check in the module docstring (ForecastRefused if not)."""

    if tuple(sorted(record)) != tuple(sorted(FIELDS)):
        raise ForecastRefused("unexpected or missing fields")
    if record["schema"] != SCHEMA or record["schema_version"] != SCHEMA_VERSION:
        raise ForecastRefused("unknown schema")
    if record["forecast_id"] != forecast_id(record):
        raise ForecastRefused("forecast_id does not match the content")
    if record["data_class"] not in ("real", "synthetic"):
        raise ForecastRefused("data_class must be real or synthetic")
    if record["estimator"] not in ESTIMATORS:
        raise ForecastRefused(f"unknown estimator {record['estimator']!r}")
    sessions = record["target_sessions"]
    if not isinstance(sessions, list) or len(sessions) != 5:
        raise ForecastRefused("five target sessions are required")
    if not all(isinstance(s, str) and len(s) == 10 for s in sessions):
        raise ForecastRefused("target sessions are YYYY-MM-DD dates")
    if not record["as_of_session"] < sessions[0] or sessions != sorted(set(sessions)):
        raise ForecastRefused("target sessions must follow the as-of session, in order")
    refusal = record["refusal"]
    daily = record["daily"]
    if refusal is not None:
        if daily or record["cumulative"] is not None:
            raise ForecastRefused("a refusal carries no probability")
        return record
    if not isinstance(daily, list) or len(daily) != 5:
        raise ForecastRefused("five daily forecasts are required")
    for horizon, row in enumerate(daily, start=1):
        if row.get("horizon") != horizon or row.get("session") != sessions[horizon - 1]:
            raise ForecastRefused("daily forecasts must follow the target sessions")
        _check_probabilities(row, f"session {horizon}")
    cumulative = record["cumulative"]
    if cumulative is not None:
        _check_probabilities(cumulative, "five-session total")
        for key in ("p_increase", "p_decrease"):
            bound = sum(row[key] for row in daily) - 4.0
            if cumulative[key] < bound - TOLERANCE:
                raise ForecastRefused("the five-session total contradicts the daily forecasts")
    return record


def build(
    *,
    symbol: str,
    data_class: str,
    as_of_session: str,
    created_at: str,
    target_sessions: Sequence[str],
    daily: Sequence[Sequence[float]] | None,
    cumulative: Sequence[float] | None,
    band: float,
    cumulative_band: float,
    estimator: str,
    model_version: str,
    feature_version: str,
    evaluation_run_id: str | None,
    data: Mapping[str, object],
    evidence_status: str,
    calibration_status: str,
    signals: Mapping[str, object],
    limitations: Sequence[str],
    refusal: Mapping[str, str] | None = None,
    supersedes: str | None = None,
) -> dict:
    """A checked record; `daily` is five (Decrease, Stable, Increase) triples."""

    def triple(values: Sequence[float]) -> dict:
        return {key: float(value) for key, value in zip(PROBABILITY_KEYS, values, strict=True)}

    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "data_class": data_class,
        "as_of_session": as_of_session,
        "created_at": created_at,
        "target_sessions": list(target_sessions),
        "daily": []
        if daily is None
        else [
            {"horizon": h, "session": target_sessions[h - 1], **triple(values)}
            for h, values in enumerate(daily, start=1)
        ],
        "cumulative": None if cumulative is None else triple(cumulative),
        "band": float(band),
        "cumulative_band": float(cumulative_band),
        "estimator": estimator,
        "model_version": model_version,
        "feature_version": feature_version,
        "evaluation_run_id": evaluation_run_id,
        "data": dict(data),
        "evidence_status": evidence_status,
        "calibration_status": calibration_status,
        "signals": dict(signals),
        "limitations": list(limitations),
        "refusal": None if refusal is None else dict(refusal),
        "supersedes": supersedes,
    }
    record["forecast_id"] = forecast_id(record)
    return dict(check(record))
