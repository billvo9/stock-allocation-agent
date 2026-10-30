"""
The versioned forecast artifact, `StockProbabilityForecast` (schema 1.1; 1.0
records stay valid as stored), and its checks (standard library only, so any
reader can use it).

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
    comparator          (schema 1.1) B1's forecast for the same stock, as-of
                        session, target sessions, bands and inputs, recorded in
                        the same write-once record by the frozen forecast job:
                        {role "benchmark", estimator "B1", half_life, daily,
                        cumulative}. It is never a forecast of its own: it is
                        not a row of the ledger, is never live, never counts in
                        a rolling score or a calibration bucket, and never
                        changes model-selection evidence. It exists only so the
                        forecast can be compared with B1 on exactly matched
                        outcomes. None for a refusal; equal to the forecast
                        when the frozen estimator is B1 itself.

Checks (`check`): probabilities in [0, 1] that sum to 1 within 1e-9; target
sessions strictly after the as-of session and increasing; the cumulative
forecast respects the bounds P_cum(Increase) >= sum_h p_h(Increase) - 4 and
the same for Decrease (all five sessions moving one way implies the total
moved that way; they never bind in practice, but a violation would mean an
incoherent artifact); the estimator is one of ESTIMATORS. A future
reader (for example a PPO observation) must use these records only through
an approved point-in-time reader, and must never treat a forecast rebuilt
after the fact as having been available at its as-of session.

Timing (`opened_by`): a forecast must be made before its first target
session opens, at 09:30 America/New_York (the prospective ETF ledger's
recording deadline). Otherwise it could already contain that session's
opening move, or the outcome itself. The store refuses such a record, and
scoring leaves each target whose sessions had opened unscored.

Reconstructed (schema 1.2; owner decision of 2026-10-30, for the activated
model's missed evenings): `reconstructed` is True, and the forecast is made
after its evening closed (at or after its first target session's open), at
its true created_at, from prices dated up to its as-of close only. It is
never a live forecast: the store records it only through its own path,
scoring judges it by session dates, and every live aggregate leaves it out
(`is_reconstructed`). Live forecasts stay schema 1.1, without the field.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

SCHEMA = "stock_probability_forecast"
SCHEMA_VERSION = "1.1"  # 1.1 adds the B1 comparator
RECONSTRUCTED_VERSION = "1.2"  # 1.2: a reconstruction (1.1 and `reconstructed`, always True)
SCHEMA_VERSIONS = ("1.0", "1.1", RECONSTRUCTED_VERSION)
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
COMPARATOR_FIELDS = ("role", "estimator", "half_life", "daily", "cumulative")
COMPARATOR_ROLE = "benchmark"
COMPARATOR_ESTIMATOR = "B1"
PROBABILITY_KEYS = ("p_decrease", "p_stable", "p_increase")
ESTIMATORS = ("B0", "B1", "B2", "B3", "M1")
NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = (9, 30)  # America/New_York


class ForecastRefused(ValueError):
    """An artifact that breaks the contract above."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def forecast_id(record: Mapping[str, object]) -> str:
    content = {key: value for key, value in record.items() if key != "forecast_id"}
    return hashlib.sha256(canonical(content).encode("utf-8")).hexdigest()


def session_open(session: str) -> datetime:
    """The session's opening time (09:30 America/New_York), timezone-aware."""

    return datetime.combine(date.fromisoformat(session), time(*SESSION_OPEN), tzinfo=NEW_YORK)


def created(record: Mapping[str, object]) -> datetime:
    """The record's created_at as a timezone-aware time (ISO 8601 with Z or an offset)."""

    text = str(record["created_at"])
    when = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    if when.tzinfo is None:
        raise ForecastRefused("created_at must carry a timezone")
    return when


def opened_by(record: Mapping[str, object], sessions: Sequence[str]) -> bool:
    """True when any of `sessions` had opened at or before the record was created."""

    when = created(record)
    return any(session_open(session) <= when for session in sessions)


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

    version = record.get("schema_version")
    fields = FIELDS if version == "1.0" else (*FIELDS, "comparator")
    if version == RECONSTRUCTED_VERSION:
        fields = (*fields, "reconstructed")
    if tuple(sorted(record)) != tuple(sorted(fields)):
        raise ForecastRefused("unexpected or missing fields")
    if record["schema"] != SCHEMA or version not in SCHEMA_VERSIONS:
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
    if version == RECONSTRUCTED_VERSION:
        if record["reconstructed"] is not True or record["data_class"] != "real":
            raise ForecastRefused("a 1.2 record is a real forecast with reconstructed true")
        if not opened_by(record, sessions[:1]):
            raise ForecastRefused("a reconstruction is made only after its evening closed")
    refusal = record["refusal"]
    daily = record["daily"]
    if refusal is not None:
        if daily or record["cumulative"] is not None or record.get("comparator") is not None:
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
    if version != "1.0":
        _check_comparator(record)
    return record


def is_reconstructed(record: Mapping[str, object]) -> bool:
    """True for a reconstruction (schema 1.2), never a live forecast (module docstring)."""

    return record.get("reconstructed") is True


def _check_comparator(record: Mapping[str, object]) -> None:
    """B1's forecast beside it: same sessions, same targets, valid probabilities."""

    comparator = record["comparator"]
    if not isinstance(comparator, dict) or tuple(sorted(comparator)) != tuple(
        sorted(COMPARATOR_FIELDS)
    ):
        raise ForecastRefused("the comparator needs its role, estimator, half-life and forecast")
    if comparator["role"] != COMPARATOR_ROLE or comparator["estimator"] != COMPARATOR_ESTIMATOR:
        raise ForecastRefused("the comparator is B1, recorded as the benchmark")
    half_life = comparator["half_life"]
    if not isinstance(half_life, int | float) or isinstance(half_life, bool) or half_life <= 0:
        raise ForecastRefused("the comparator's half-life must be a positive number")
    sessions = record["target_sessions"]
    daily = comparator["daily"]
    if not isinstance(daily, list) or len(daily) != 5:
        raise ForecastRefused("the comparator needs five daily forecasts")
    for horizon, row in enumerate(daily, start=1):
        if row.get("horizon") != horizon or row.get("session") != sessions[horizon - 1]:
            raise ForecastRefused("the comparator's sessions must be the forecast's")
        _check_probabilities(row, f"comparator session {horizon}")
    if (comparator["cumulative"] is None) != (record["cumulative"] is None):
        raise ForecastRefused("the comparator covers the forecast's targets, no more and no fewer")
    if comparator["cumulative"] is not None:
        _check_probabilities(comparator["cumulative"], "comparator five-session total")


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
    comparator: Mapping[str, object] | None = None,
    reconstructed: bool = False,
) -> dict:
    """
    A checked record (schema 1.1; 1.2 when `reconstructed`); `daily` is five (Decrease,
    Stable, Increase) triples. `comparator` is B1's forecast on the same inputs: {"half_life",
    "daily" (five triples), "cumulative" (a triple or None)}; required unless the record is a
    refusal.
    """

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
        "comparator": None
        if comparator is None
        else {
            "role": COMPARATOR_ROLE,
            "estimator": COMPARATOR_ESTIMATOR,
            "half_life": float(comparator["half_life"]),
            "daily": [
                {"horizon": h, "session": target_sessions[h - 1], **triple(values)}
                for h, values in enumerate(comparator["daily"], start=1)
            ],
            "cumulative": None
            if comparator["cumulative"] is None
            else triple(comparator["cumulative"]),
        },
    }
    if reconstructed:
        record["schema_version"] = RECONSTRUCTED_VERSION
        record["reconstructed"] = True
    record["forecast_id"] = forecast_id(record)
    return dict(check(record))
