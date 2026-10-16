"""
Stock Lab health from the daily job's monitor files (artifact
`stock_lab_monitor`, format v1.0), read from `<inputs>/stock_lab_monitor/`
(the newest file by `generated_at`). The Stock Lab's daily job is the
producer (`stock_agent.stock_lab.monitoring`); this module only judges what
the file records and recomputes nothing.

Fields besides the envelope (`inputs`):

    job          {status: ok | partial | failed | skipped, started_at,
                  finished_at, expected_session, reason, failed: [ticker],
                  skipped: [ticker]}
    freshness    [{ticker, last_session, sessions_behind}]
    scores       {n, rps, baseline_rps, brier, log_loss, observed: {..},
                  predicted: {..}} (n 0 when nothing is scored yet)
    inputs_drift [{feature, psi}]
    traps        {resolved, failed, pending}
    ledger       {entries, verified: bool, reason}
    lineage      {estimator, model_version, feature_version, evaluation_run_id}
    schedule     {last_success_at, next_expected_at}

Checks (each says it is an experimental, unvalidated forecaster):

    data.stock_lab_freshness  every watched ticker has the expected session
                              (degraded when any is behind; insufficient
                              evidence without a watchlist)
    forecast.stock_lab_job    the last daily run (failed is failed, partial
                              or skipped degraded)
    forecast.stock_lab_scores rolling RPS against the baseline over scored
                              daily targets (insufficient evidence below
                              MIN_SCORED; degraded when worse than the
                              baseline); never "skill", only tracking
    forecast.stock_lab_drift  the largest input-distribution shift (PSI;
                              degraded above 0.25)
    forecast.stock_lab_ledger the forecast ledger's hash chain (failed when
                              it does not verify)

In the private scope a synthetic input is at most insufficient_evidence.
"""

from __future__ import annotations

from pathlib import Path

from stock_agent.health.inputs import (
    Artifact,
    Finding,
    FormatError,
    NotRead,
    choice,
    common_findings,
    load_latest,
    require,
    worst_finding,
)
from stock_agent.health.model import Area, Better, Check, HealthState

ARTIFACT = "stock_lab_monitor"
JOB_STATUSES = frozenset({"ok", "partial", "failed", "skipped"})
MIN_SCORED = 250
PSI_WARNING = 0.25
EXPERIMENTAL = "Experimental, unvalidated forecaster; tracking only, never evidence of skill."
TITLES = {
    "data.stock_lab_freshness": "Stock Lab data freshness",
    "forecast.stock_lab_job": "Stock Lab daily update",
    "forecast.stock_lab_scores": "Stock Lab rolling forecast scores",
    "forecast.stock_lab_drift": "Stock Lab input drift",
    "forecast.stock_lab_ledger": "Stock Lab forecast ledger",
}


def _parse(artifact: Artifact) -> dict:
    document = artifact.document
    job = require(document, "job", "object")
    choice(job, "status", JOB_STATUSES)
    for key in ("failed", "skipped"):
        require(job, key, "list")
    freshness = require(document, "freshness", "list")
    for row in freshness:
        if not isinstance(row, dict):
            raise FormatError("freshness rows must be objects")
        require(row, "ticker", "text")
        require(row, "sessions_behind", "nonnegative", optional=True)
    scores = require(document, "scores", "object")
    require(scores, "n", "nonnegative")
    drift = require(document, "inputs_drift", "list")
    for row in drift:
        require(row, "feature", "text")
        require(row, "psi", "nonnegative", optional=True)
    ledger = require(document, "ledger", "object")
    if not isinstance(ledger.get("verified"), bool):
        raise FormatError("ledger.verified must be true or false")
    require(document, "lineage", "object")
    require(document, "schedule", "object")
    return document


def _check(check_id, state, reason, *, measured, value, unit, threshold, summary, detail,
           evidence, better=None) -> Check:  # fmt: skip
    return Check(
        check_id=check_id,
        area=Area(check_id.split(".")[0]),
        title=TITLES[check_id],
        state=state,
        reason=reason,
        measured_at=measured if value is not None else None,
        value=value,
        unit=unit,
        threshold=threshold,
        better=better,
        tolerance=None,
        summary=summary,
        detail=detail,
        evidence=evidence,
    )


def stock_lab_checks(directory: Path | None, *, allow_synthetic: bool = False) -> list[Check]:
    loaded = load_latest(directory, ARTIFACT)
    evidence = None if directory is None else str(directory)
    if isinstance(loaded, NotRead):
        return [
            _check(
                check_id,
                loaded.state,
                loaded.reason,
                measured=None,
                value=None,
                unit=None,
                threshold=None,
                summary="No Stock Lab monitor file could be read."
                if loaded.state is HealthState.UNAVAILABLE
                else "The Stock Lab daily job has not written a monitor file yet.",
                detail=loaded.detail,
                evidence=evidence,
            )
            for check_id in TITLES
        ]
    try:
        document = _parse(loaded)
    except FormatError as error:
        return [
            _check(
                check_id,
                HealthState.UNAVAILABLE,
                "unrecognized_format",
                measured=None,
                value=None,
                unit=None,
                threshold=None,
                summary="The Stock Lab monitor file is not in the agreed format.",
                detail=str(error),
                evidence=evidence,
            )
            for check_id in TITLES
        ]
    common = common_findings(loaded, allow_synthetic=allow_synthetic)
    measured = loaded.generated_at
    lineage = document["lineage"]
    lineage_text = ", ".join(f"{k} {v}" for k, v in sorted(lineage.items()))
    described = f"{loaded.describe()}; lineage: {lineage_text}; {EXPERIMENTAL}"
    checks = []

    # Freshness
    rows = document["freshness"]
    behind = [
        r for r in rows if (r.get("sessions_behind") or 0) > 0 or r.get("sessions_behind") is None
    ]
    if not rows:
        finding = Finding(HealthState.INSUFFICIENT_EVIDENCE, "no_watchlist", "No stock is watched.")
    elif behind:
        names = ", ".join(r["ticker"] for r in behind[:10])
        finding = Finding(
            HealthState.DEGRADED,
            "stale_prices",
            f"{len(behind)} of {len(rows)} watched stocks lack the expected "
            f"session: {names}. Their last valid forecast is shown as stale.",
        )
    else:
        finding = Finding(HealthState.HEALTHY, None, f"All {len(rows)} watched stocks are current.")
    finding = worst_finding([finding, *common])
    checks.append(
        _check(
            "data.stock_lab_freshness",
            finding.state,
            finding.reason,
            measured=measured,
            value=len(behind),
            unit="stocks behind",
            threshold="every watched stock has the expected session",
            summary=finding.summary,
            detail=described,
            evidence=evidence,
            better=Better.LOWER,
        )
    )

    # The daily job
    job = document["job"]
    status = job["status"]
    states = {
        "ok": (HealthState.HEALTHY, None, "The last daily update completed."),
        "partial": (
            HealthState.DEGRADED,
            "job_partial",
            f"The last update finished with {len(job['failed'])} failed stock(s).",
        ),
        "skipped": (
            HealthState.DEGRADED,
            "job_skipped",
            f"The last update was skipped: {job.get('reason') or 'no reason recorded'}.",
        ),
        "failed": (
            HealthState.FAILED,
            "job_failed",
            (
                f"The last update failed: {job.get('reason') or 'no reason recorded'}. "
                "The last valid forecasts are kept and labelled stale."
            ),
        ),
    }
    state, reason, text = states[status]
    finding = worst_finding([Finding(state, reason, text), *common])
    schedule = document["schedule"]
    checks.append(
        _check(
            "forecast.stock_lab_job",
            finding.state,
            finding.reason,
            measured=measured,
            value=status,
            unit=None,
            threshold="the last scheduled update completed",
            summary=finding.summary,
            detail=f"last success {schedule.get('last_success_at')}; next expected "
            f"{schedule.get('next_expected_at')}; {described}",
            evidence=evidence,
        )
    )

    # Rolling scores
    scores = document["scores"]
    n = scores["n"]
    if n < MIN_SCORED:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "too_few_scored",
            f"{n} scored forecasts so far; at least {MIN_SCORED} are needed "
            "before a rolling score means anything.",
        )
    elif (
        scores.get("rps") is not None
        and scores.get("baseline_rps") is not None
        and (scores["rps"] > scores["baseline_rps"])
    ):
        finding = Finding(
            HealthState.DEGRADED,
            "worse_than_baseline",
            f"Rolling RPS {scores['rps']:.4f} is worse than the baseline's "
            f"{scores['baseline_rps']:.4f}.",
        )
    else:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"Rolling RPS {scores.get('rps')} over {n} scored targets, no worse "
            "than the baseline. Tracking only, not evidence of skill.",
        )
    finding = worst_finding([finding, *common])
    checks.append(
        _check(
            "forecast.stock_lab_scores",
            finding.state,
            finding.reason,
            measured=measured,
            value=scores.get("rps") if n else None,
            unit="ranked probability score",
            threshold=f">= {MIN_SCORED} scored and no worse than the baseline",
            summary=finding.summary,
            detail=described,
            evidence=evidence,
            better=Better.LOWER,
        )
    )

    # Input drift
    drift = [r for r in document["inputs_drift"] if r.get("psi") is not None]
    worst = max(drift, key=lambda r: r["psi"]) if drift else None
    if worst is None:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_drift_measured",
            "No input distribution was compared yet.",
        )
    elif worst["psi"] > PSI_WARNING:
        finding = Finding(
            HealthState.DEGRADED,
            "input_drift",
            f"{worst['feature']} shifted (PSI {worst['psi']:.2f}) against its "
            "development distribution.",
        )
    else:
        finding = Finding(HealthState.HEALTHY, None, "No input shifted beyond PSI 0.25.")
    finding = worst_finding([finding, *common])
    checks.append(
        _check(
            "forecast.stock_lab_drift",
            finding.state,
            finding.reason,
            measured=measured,
            value=None if worst is None else worst["psi"],
            unit="population stability index",
            threshold=f"<= {PSI_WARNING}",
            summary=finding.summary,
            detail=described,
            evidence=evidence,
            better=Better.LOWER,
        )
    )

    # Ledger
    ledger = document["ledger"]
    if ledger["verified"]:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"The forecast ledger's {ledger.get('entries', 0)} entries verify.",
        )
    else:
        finding = Finding(
            HealthState.FAILED,
            "ledger_broken",
            f"The forecast ledger does not verify: {ledger.get('reason')}.",
        )
    finding = worst_finding([finding, *common])
    checks.append(
        _check(
            "forecast.stock_lab_ledger",
            finding.state,
            finding.reason,
            measured=measured,
            value=ledger.get("entries", 0),
            unit="entries",
            threshold="every entry's hash chain verifies",
            summary=finding.summary,
            detail=described,
            evidence=evidence,
        )
    )
    return checks
