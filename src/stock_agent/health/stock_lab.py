"""
Stock Lab health from the daily job's monitor files (artifact
`stock_lab_monitor`, format v1.0), read from `<inputs>/stock_lab_monitor/`
(the newest file by `generated_at`). The Stock Lab's daily job is the
producer (`stock_agent.stock_lab.monitoring`); this module only judges what
the file records and recomputes nothing.

Fields besides the envelope (`inputs`):

    job          {status: ok | partial | failed | skipped, started_at,
                  finished_at, expected_session, reason, failed: [ticker],
                  skipped: [ticker], activation (optional): not_authorized
                  when the run was skipped because the owner's decisions are
                  not recorded yet}
    freshness    [{ticker, last_session, sessions_behind}]
    scores       {n, rps, baseline_rps, brier, log_loss, observed: {..},
                  predicted: {..}} (n 0 when nothing is scored yet)
    inputs_drift [{feature, psi}]
    traps        {resolved, failed, pending}
    ledger       {entries, verified: bool, reason, head, kept_head (optional):
                  {state, kept}}, where state is the job's kept head check
                  (stock_lab.job): match, started, not_configured, unusable,
                  not_in_ledger or ledger_deleted
    lineage      {estimator, model_version, feature_version, evaluation_run_id}
    schedule     {last_success_at, next_expected_at}
    symbols      optional (added 2026-10-17, format still 1.0): {tickers:
                  [{ticker, status, reason}]}, each watched ticker's latest
                  check against the provider's listing (status: supported,
                  unsupported, inactive, delisted, ambiguous, or unverified
                  when never checked; stock_lab.contracts.SymbolStatus)

Checks (each says it is an experimental, unvalidated forecaster):

    data.stock_lab_freshness  every watched ticker has the expected session
                              (degraded when any is behind; insufficient
                              evidence without a watchlist, or while the
                              price update is not authorized yet)
    data.stock_lab_tickers    every watched ticker is supported by the
                              provider's listing: degraded when any is
                              unsupported, inactive, delisted or ambiguous
                              (no forecast is made for it); insufficient
                              evidence when a ticker was never checked, without
                              a watchlist, or for a monitor file that does not
                              record the listing status
    forecast.stock_lab_job    the last daily run (failed is failed, partial
                              or skipped degraded; a run skipped because it
                              is not authorized yet is insufficient evidence,
                              "not activated"; an overdue run is failed)
    forecast.stock_lab_scores rolling RPS against the baseline over scored
                              daily targets (insufficient evidence below
                              MIN_SCORED; degraded when worse than the
                              baseline); never "skill", only tracking
    forecast.stock_lab_drift  the largest input-distribution shift (PSI;
                              degraded above 0.25)
    forecast.stock_lab_ledger the forecast ledger's hash chain and the kept
                              head: failed when the chain does not verify,
                              when the kept head is no longer in it, or when
                              the store is gone though the kept head shows
                              it held entries; unavailable when the kept file
                              is unusable; insufficient evidence when no head
                              is kept yet (a removed last entry would not
                              show); healthy only when both hold

In the private scope a synthetic input is at most insufficient_evidence.
"""

from __future__ import annotations

from datetime import timedelta
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
from stock_agent.health.model import Area, Better, Check, HealthState, parse_timestamp

ARTIFACT = "stock_lab_monitor"
JOB_STATUSES = frozenset({"ok", "partial", "failed", "skipped"})
NOT_AUTHORIZED = "not_authorized"
KEPT_STATES = frozenset(
    {"match", "started", "not_configured", "unusable", "not_in_ledger", "ledger_deleted"}
)
MIN_SCORED = 250
MIN_SESSIONS = 50  # distinct settled sessions: overlapping targets are not independent
OVERDUE = 6  # hours past the next expected run before a silent job counts as overdue
PSI_WARNING = 0.25
EXPERIMENTAL = "Experimental, unvalidated forecaster; tracking only, never evidence of skill."
# The listing statuses the job records (stock_lab.contracts.SymbolStatus, tested equal).
SYMBOL_STATUSES = frozenset(
    {"unverified", "supported", "unsupported", "inactive", "delisted", "ambiguous"}
)
NOT_SUPPORTED = frozenset({"unsupported", "inactive", "delisted", "ambiguous"})
SYMBOL_WORDS = {
    "unsupported": "unsupported",
    "inactive": "no recent prices",
    "delisted": "delisted or renamed",
    "ambiguous": "ambiguous listing",
}
TITLES = {
    "data.stock_lab_freshness": "Stock Lab data freshness",
    "data.stock_lab_tickers": "Stock Lab watchlist tickers",
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
    if job.get("activation") is not None:
        choice(job, "activation", frozenset({NOT_AUTHORIZED}))
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
    if ledger.get("kept_head") is not None:
        choice(require(ledger, "kept_head", "object"), "state", KEPT_STATES)
    require(document, "lineage", "object")
    require(document, "schedule", "object")
    if document.get("symbols") is not None:
        symbols = require(document, "symbols", "object")
        for row in require(symbols, "tickers", "list"):
            if not isinstance(row, dict):
                raise FormatError("symbols.tickers rows must be objects")
            require(row, "ticker", "text")
            choice(row, "status", SYMBOL_STATUSES)
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


def _tickers_finding(symbols: dict | None) -> tuple[Finding, int | None]:
    """The watchlist's listing status: the finding and the number not supported."""

    if symbols is None:
        return Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "not_recorded",
            "The monitor file does not record the tickers' listing status (it predates this "
            "check), so an unsupported ticker would not show.",
        ), None
    rows = symbols["tickers"]
    if not rows:
        return Finding(HealthState.INSUFFICIENT_EVIDENCE, "no_watchlist", "No stock is watched."), 0
    unusable = [row for row in rows if row["status"] in NOT_SUPPORTED]
    unchecked = [row["ticker"] for row in rows if row["status"] == "unverified"]
    waiting = (
        f" {len(unchecked)} more not checked yet: {', '.join(unchecked[:10])}." if unchecked else ""
    )
    if unusable:
        names = ", ".join(f"{row['ticker']} ({SYMBOL_WORDS[row['status']]})" for row in unusable)
        return Finding(
            HealthState.DEGRADED,
            "tickers_not_supported",
            f"{len(unusable)} of {len(rows)} watched tickers cannot be studied as listed: {names}. "
            f"No forecast is made for them.{waiting}",
        ), len(unusable)
    if unchecked:
        return Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "tickers_not_checked",
            f"{len(unchecked)} of {len(rows)} watched tickers have not been checked against the "
            f"provider's listing yet: {', '.join(unchecked[:10])}.",
        ), 0
    return Finding(
        HealthState.HEALTHY,
        None,
        f"All {len(rows)} watched tickers are supported by the provider's latest listing check.",
    ), 0


KEPT_FINDINGS = {
    "match": (HealthState.HEALTHY, None, "The kept copy of the ledger head is in the chain."),
    "started": (
        HealthState.INSUFFICIENT_EVIDENCE,
        "head_started",
        (
            "The kept copy of the ledger head was started this run, so a last entry removed "
            "before now would not show."
        ),
    ),
    "not_configured": (
        HealthState.INSUFFICIENT_EVIDENCE,
        "head_not_kept",
        "No copy of the ledger head is kept, so a removed last entry would not show.",
    ),
    "unusable": (
        HealthState.UNAVAILABLE,
        "head_unusable",
        "The kept ledger head file is not a 64-hex hash; it was left as it is.",
    ),
    "not_in_ledger": (
        HealthState.FAILED,
        "head_not_in_ledger",
        (
            "The ledger no longer holds the kept head hash: entries were removed or the store "
            "was replaced, which the chain alone cannot show."
        ),
    ),
    "ledger_deleted": (
        HealthState.FAILED,
        "ledger_deleted",
        "The Stock Lab store is gone, but the kept ledger head shows it held entries.",
    ),
}


def _kept_finding(kept: dict | None) -> Finding:
    """The kept head check's finding; a monitor file without one keeps no head."""

    state = "not_configured" if kept is None else kept["state"]
    return Finding(*KEPT_FINDINGS[state])


def stock_lab_checks(
    directory: Path | None, *, allow_synthetic: bool = False, now=None
) -> list[Check]:
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
    not_activated = document["job"].get("activation") == NOT_AUTHORIZED
    if not rows:
        finding = Finding(HealthState.INSUFFICIENT_EVIDENCE, "no_watchlist", "No stock is watched.")
    elif behind and not_activated:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            NOT_AUTHORIZED,
            f"{len(behind)} of {len(rows)} watched stocks lack the expected session, as expected: "
            "the daily price update is not activated until the owner's decisions are recorded.",
        )
    elif behind:
        names = ", ".join(r["ticker"] for r in behind[:10])
        never = [r["ticker"] for r in behind if r.get("last_session") is None]
        never_text = (
            f" {len(never)} have no stored session at all: {', '.join(never[:10])}."
            if never
            else ""
        )
        finding = Finding(
            HealthState.DEGRADED,
            "stale_prices",
            f"{len(behind)} of {len(rows)} watched stocks lack the expected "
            f"session: {names}. Their last valid forecast is shown as stale.{never_text}",
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

    # Listing status of the watched tickers
    finding, value = _tickers_finding(document.get("symbols"))
    finding = worst_finding([finding, *common])
    checks.append(
        _check(
            "data.stock_lab_tickers",
            finding.state,
            finding.reason,
            measured=measured,
            value=value,
            unit="tickers not supported",
            threshold="every watched ticker is supported by the provider's latest listing check",
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
    if status == "skipped" and job.get("activation") == NOT_AUTHORIZED:
        state, reason, text = (
            HealthState.INSUFFICIENT_EVIDENCE,
            NOT_AUTHORIZED,
            (
                "Not activated: the daily update ran its checks and skipped what is not "
                f"authorized yet ({job.get('reason') or 'no reason recorded'})."
            ),
        )
    findings = [Finding(state, reason, text), *common]
    next_expected = document["schedule"].get("next_expected_at")
    if now is not None and next_expected:
        due = parse_timestamp("next_expected_at", next_expected)
        if now > due + timedelta(hours=OVERDUE):
            findings.append(
                Finding(
                    HealthState.FAILED,
                    "job_overdue",
                    f"No daily update since {loaded.generated_at:%Y-%m-%d %H:%M} UTC; the next was "
                    f"due {due:%Y-%m-%d %H:%M} UTC. The scheduled job may have stopped.",
                )
            )
    finding = worst_finding(findings)
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
    settled = scores.get("sessions", n)
    if n < MIN_SCORED or settled < MIN_SESSIONS:
        finding = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "too_few_scored",
            f"{n} scored targets over {settled} sessions so far; at least {MIN_SCORED} targets "
            f"over {MIN_SESSIONS} sessions are needed before a rolling score means anything.",
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
            threshold=f">= {MIN_SCORED} targets over {MIN_SESSIONS} sessions, no worse than the "
            "baseline",
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
        chain = Finding(
            HealthState.HEALTHY,
            None,
            f"The forecast ledger's {ledger.get('entries', 0)} entries verify.",
        )
    else:
        chain = Finding(
            HealthState.FAILED,
            "ledger_broken",
            f"The forecast ledger does not verify: {ledger.get('reason')}.",
        )
    finding = worst_finding([chain, _kept_finding(ledger.get("kept_head")), *common])
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
