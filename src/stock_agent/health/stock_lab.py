"""
Stock Lab health from the daily job's monitor files (artifact
`stock_lab_monitor`, format v1.0), read from `<inputs>/stock_lab_monitor/`
(the newest file by `generated_at`). The Stock Lab's daily job is the
producer (`stock_agent.stock_lab.monitoring`); this module only judges what
the file records and recomputes nothing.

Where (`inputs_root`). The real job writes under
~/stock-agent-private/health_inputs/ (since 2026-10-21), outside every
checkout, so its evening can run from a runner clone outside ~/Desktop while
Model Health runs from another checkout. Until that folder holds a monitor
file (`stock_lab_monitor/*.json`), the checkout's own health inputs are read,
where the job wrote before: an empty folder never hides the last file there
(an overdue job stays overdue). The writer's test override
(STOCK_AGENT_STOCK_LAB_HEALTH_INPUTS) is not followed here; the local
script's --stock-lab-inputs reads another folder.

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
    outcomes     optional and additive (added 2026-10-19; format 1.0): the real
                  evening job's scoring counts {scored, unscorable, pending,
                  missing, refused: [{ticker, forecast_id, reason}]}; missing
                  matured outcomes or refused reads degrade the scores check
    evenings     optional and additive (added 2026-10-20; format 1.0): the real
                  evening job's prospective record {since, due, recorded,
                  missed: [session], recent, recent_missed}
                  (stock_lab.job.evening_record). A due evening is one whose
                  next 09:30 New York open has passed; a missed one recorded
                  no live forecast on time, and stays missed. The optional
                  `reconstructed` (added 2026-10-30) lists the due evenings
                  holding a forecast reconstructed after the fact, scored
                  apart from the live
                  record (missed or partly recorded: per stock), and is
                  noted. A miss among the
                  last `recent` due evenings degrades the job check
    symbols      optional and additive (added 2026-10-17; the format stays 1.0
                  because the envelope ignores extra fields, so earlier files
                  stay valid): {tickers: [{ticker, status, reason}]}, each
                  watched ticker's latest check against the provider's
                  listing (status: supported, unsupported, inactive,
                  delisted, ambiguous, or unverified when never checked;
                  stock_lab.contracts.SymbolStatus). Absent: the tickers
                  check is not_recorded; present: checked strictly

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
                              "not activated"; an overdue run is failed; a
                              recently missed evening is degraded)
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

from datetime import date, timedelta
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
# The real job's shared root under the home folder (stock_lab.monitoring.shared_inputs, tested
# equal; health imports nothing from the Stock Lab).
SHARED_INPUTS = ("stock-agent-private", "health_inputs")
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
    if document.get("outcomes") is not None:
        outcomes = require(document, "outcomes", "object")
        for key in ("scored", "unscorable", "pending", "missing"):
            require(outcomes, key, "nonnegative")
        for row in require(outcomes, "refused", "list"):
            if not isinstance(row, dict):
                raise FormatError("outcomes.refused rows must be objects")
            require(row, "ticker", "text")
            require(row, "reason", "text")
    if document.get("evenings") is not None:
        evenings = require(document, "evenings", "object")
        since = _session(require(evenings, "since", "text"))
        for key in ("due", "recorded", "recent", "recent_missed"):
            require(evenings, key, "int")
        missed = [_session(day) for day in require(evenings, "missed", "list")]
        if missed != sorted(set(missed)) or (missed and missed[0] < since):
            raise FormatError("evenings.missed must list distinct sessions since `since`, in order")
        if evenings["recorded"] + len(evenings["missed"]) != evenings["due"]:
            raise FormatError("evenings: recorded and missed must add up to due")
        if evenings["recent_missed"] > min(evenings["recent"], len(evenings["missed"])):
            raise FormatError("evenings.recent_missed exceeds the recent or missed evenings")
        if evenings.get("reconstructed") is not None:  # optional (owner decision, 2026-10-30)
            rebuilt = [_session(day) for day in require(evenings, "reconstructed", "list")]
            if rebuilt != sorted(set(rebuilt)) or (rebuilt and rebuilt[0] < since):
                raise FormatError(
                    "evenings.reconstructed must list distinct sessions since `since`, in order"
                )
    if document.get("symbols") is not None:
        symbols = require(document, "symbols", "object")
        for row in require(symbols, "tickers", "list"):
            if not isinstance(row, dict):
                raise FormatError("symbols.tickers rows must be objects")
            require(row, "ticker", "text")
            choice(row, "status", SYMBOL_STATUSES)
    return document


def _session(value: object) -> date:
    """A session written YYYY-MM-DD."""

    if isinstance(value, str) and len(value) == 10:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise FormatError(f"{value!r} is not a session (YYYY-MM-DD)")


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


def _outcome_findings(outcomes: dict | None) -> list[Finding]:
    """
    The real job's scoring counts (optional, added 2026-10-19): a matured target whose outcome
    is not stored yet, or an outcome that could not be read, degrades the scores check.
    """

    if not outcomes:
        return []
    found = []
    missing = outcomes.get("missing") or 0
    if missing:
        found.append(
            Finding(
                HealthState.DEGRADED,
                "missing_outcomes",
                f"{missing} target(s) whose session has closed have no stored outcome yet: the "
                "evening price update may have missed them. They are scored once it arrives.",
            )
        )
    refused = outcomes.get("refused") or []
    if refused:
        reasons = ", ".join(sorted({str(row.get("reason")) for row in refused}))
        found.append(
            Finding(
                HealthState.DEGRADED,
                "outcomes_refused",
                f"The outcomes of {len(refused)} forecast(s) could not be read ({reasons}); "
                "nothing was scored for them.",
            )
        )
    return found


def _evening_finding(evenings: dict | None) -> Finding | None:
    """
    The real job's prospective record (optional, added 2026-10-20): a missed evening among the
    last `recent` due evenings degrades the job check; an older one is reported in the detail.
    """

    if not evenings or not evenings["recent_missed"]:
        return None
    missed = evenings["missed"]
    count = evenings["recent_missed"]
    rebuilt = set(missed) & set(evenings.get("reconstructed") or ())
    noted = (
        f" {len(rebuilt)} of them reconstructed after the fact, scored apart from the live record."
        if rebuilt
        else ""
    )
    return Finding(
        HealthState.DEGRADED,
        "evening_missed",
        f"{count} of the last {min(evenings['recent'], evenings['due'])} closed evenings recorded "
        f"no forecast (latest {missed[-1]}): permanent gaps in the prospective record, since a "
        "forecast is never made after the next 09:30 open. Check the evening agent's log; "
        f"{len(missed)} missed since {evenings['since']}.{noted}",
    )


def _evening_text(evenings: dict | None) -> str:
    if not evenings:
        return ""
    missed = evenings["missed"]
    noted = evenings.get("reconstructed") or []
    rebuilt = set(noted)
    shown = (
        ", ".join(f"{day} (reconstructed)" if day in rebuilt else day for day in missed[-5:])
        if missed
        else "none"
    )
    later = f"reconstructed after the fact (latest 5): {', '.join(noted[-5:])}; " if noted else ""
    return (
        f"evenings since {evenings['since']}: {evenings['recorded']} of {evenings['due']} "
        f"recorded; missed (latest 5): {shown}; {later}"
    )


def inputs_root(home: Path, checkout_inputs: Path | None) -> Path | None:
    """
    The root the Stock Lab checks read (module docstring): the shared folder
    under `home` once its `stock_lab_monitor` folder holds a `*.json` file,
    else `checkout_inputs`. Only names are listed here; nothing is read.
    """

    shared = home.joinpath(*SHARED_INPUTS)
    return shared if any((shared / ARTIFACT).glob("*.json")) else checkout_inputs


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
    evening = _evening_finding(document.get("evenings"))
    if evening is not None:
        findings.append(evening)
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
            f"{schedule.get('next_expected_at')}; {_evening_text(document.get('evenings'))}"
            f"{described}",
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
    finding = worst_finding([finding, *_outcome_findings(document.get("outcomes")), *common])
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
