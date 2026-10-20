"""
Recording one daily-job run: its row in the store, its ledger entry, and its
Model Health input (`monitoring`). Standard library, numpy and pandas only.

The exchange calendar is passed in (`sessions(start, end)`), so this module
never imports a calendar of its own: the owner-run script supplies the
rules calendar, and the synthetic replay a weekday calendar.

What a run records, per watched stock:
    freshness       the newest stored close and how many sessions it is
                    behind the expected session (a stale stock keeps its last
                    valid forecast, shown as stale; nothing is fabricated)
    scores          the rolling summary of scored daily targets, beside the
                    recent-weighted baseline's development-test RPS
    inputs_drift    PSI of a few inputs, recent sessions against development
    traps           bull-trap setups since the first forecast: resolved
                    (failed or held) and still pending
    ledger          whether the hash chain verifies, its head, and the kept
                    head check (`check_kept_head`)
    lineage         the latest forecast's estimator and versions
    symbols         each watched ticker's latest check against the provider's
                    listing (`symbol_status`; unverified when never checked), so
                    Model Health shows an unsupported or delisted ticker

The kept head. A hash chain shows an edited or removed middle entry, but
not a removed last entry. So the job keeps the head hash in a private file
outside the store's database file (`kept_head`, mode 0600; the daily script
defaults to one beside it, and a copy kept elsewhere is stronger). Each run
looks for the kept hash in the chain, records the result, and replaces the
file only when the chain verifies and the kept hash was found:

    match           the kept hash is in the chain (nothing was removed)
    started         no kept file yet: one is started now
    not_configured  no kept file was given
    unusable        the file is not a 64-hex hash (left as it is)
    not_in_ledger   the chain no longer holds the kept hash: entries were
                    removed or the store was replaced (left as it is)
    ledger_deleted  the store is gone, but the kept hash shows it held
                    entries (left as it is)

A mismatch keeps the old file, so the failure stays until the owner looks.

Activation. A run skipped because the owner's decisions are not recorded
yet says so (`activation` "not_authorized"), so Model Health reads it as not
activated, not as a degraded job.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import monitoring, scoring
from stock_agent.stock_lab.contracts import (
    DEVELOPMENT_END,
    DataClass,
    SymbolReason,
    SymbolStatus,
)
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.store import GENESIS, StockLabStore, StoreRefused, stamp
from stock_agent.stock_lab.traps import CONFIRMED, UNRESOLVED, trap_events

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]
NOT_AUTHORIZED = "not_authorized"
_HASH = re.compile(r"[0-9a-f]{64}")


def _read_kept(path: Path) -> str | None:
    """The kept hash (the file's first line), or None when it is not a 64-hex hash."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    first = lines[0].strip() if lines else ""
    return first if _HASH.fullmatch(first) else None


def _write_kept(path: Path, head: str) -> None:
    """Replace the kept file atomically (mode 0600)."""

    partial = path.with_name(path.name + ".partial")
    descriptor = os.open(partial, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(head + "\n")
    os.replace(partial, path)


def check_kept_head(
    store: StockLabStore, kept_head: Path | None, *, existed: bool, verified: bool
) -> dict:
    """The kept head check (module docstring); replaces the file only on a verified match."""

    if kept_head is None:
        return {"state": "not_configured", "kept": None}
    head = store.ledger_head() or GENESIS
    if not kept_head.exists():
        if verified:
            _write_kept(kept_head, head)
        return {"state": "started", "kept": None}
    kept = _read_kept(kept_head)
    if kept is None:
        return {"state": "unusable", "kept": None}
    if not existed and kept != GENESIS:
        return {"state": "ledger_deleted", "kept": kept}
    if kept != GENESIS and kept not in store.ledger_hashes():
        return {"state": "not_in_ledger", "kept": kept}
    if verified:
        _write_kept(kept_head, head)
    return {"state": "match", "kept": kept}


def symbol_status(store: StockLabStore, tickers: Sequence[str]) -> dict:
    """Each watched ticker's latest listing check; unverified when it was never checked."""

    checks = store.latest_checks()
    rows = []
    for ticker in tickers:
        check = checks.get(ticker)
        status = SymbolStatus.UNVERIFIED if check is None else check.status
        reason = SymbolReason.NOT_CHECKED if check is None else check.reason
        rows.append({"ticker": ticker, "status": status.value, "reason": reason.value})
    return {"tickers": rows}


def freshness(
    store: StockLabStore, tickers: Sequence[str], expected: pd.Timestamp, sessions: Sessions
) -> list[dict]:
    rows = []
    for ticker in tickers:
        stored = store.bars(ticker)
        if stored is None or stored.bars["close"].notna().sum() == 0:
            rows.append({"ticker": ticker, "last_session": None, "sessions_behind": None})
            continue
        last = stored.bars.loc[stored.bars["close"].notna(), "session"].max()
        behind = len(sessions(last + pd.Timedelta(days=1), expected)) if last < expected else 0
        rows.append(
            {"ticker": ticker, "last_session": last.strftime("%Y-%m-%d"), "sessions_behind": behind}
        )
    return rows


def _frames(store: StockLabStore, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for ticker in tickers:
        stored = store.bars(ticker)
        if stored is not None:
            frames[ticker] = build_features(stored.bars.drop(columns=["batch_id"]))
    return frames


def trap_outcomes(frames: dict[str, pd.DataFrame], since: str | None) -> dict:
    resolved = failed = pending = 0
    for frame in frames.values():
        events = trap_events(frame)
        setups = events[events["is_setup"]]
        if since is not None:
            setups = setups[setups["session"] >= pd.Timestamp(since, tz="UTC")]
        pending += int((setups["status"] == UNRESOLVED).sum())
        resolved += int((setups["status"] != UNRESOLVED).sum())
        failed += int((setups["status"] == CONFIRMED).sum())
    return {"resolved": resolved, "failed": failed, "pending": pending}


def record_run(
    store: StockLabStore,
    *,
    run_id: str,
    started: pd.Timestamp,
    finished: pd.Timestamp,
    status: str,
    reason: str | None,
    expected: pd.Timestamp,
    tickers: Sequence[str],
    data_class: DataClass,
    health_inputs: Path,
    sessions: Sessions,
    calendar_id: str,
    code_sha: str | None = None,
    next_expected_at: str | None = None,
    failed: Sequence[str] = (),
    skipped: Sequence[str] = (),
    activation: str | None = None,
    kept_head: Path | None = None,
    measured_freshness: Sequence[Mapping] | None = None,
    measured_drift: Sequence[Mapping] | None = None,
    results: Sequence[Mapping] = (),
    outcomes: Mapping | None = None,
) -> Path:
    """
    Record the run, its ledger entry and its health input; returns the input's
    path. `activation` is NOT_AUTHORIZED for a run skipped because the owner's
    decisions are not recorded yet; `kept_head` is the private kept-head file.

    The real evening job (stock_lab.evening) measures what the research store
    cannot: each stock's newest close in the inference store
    (`measured_freshness`), input drift of its windows against the frozen
    reference (`measured_drift`), the per-stock `results` and the scoring
    `outcomes`. Without them, freshness and drift come from the research store.
    """

    existed = store.exists
    fresh = (
        [dict(row) for row in measured_freshness]
        if measured_freshness is not None
        else freshness(store, tickers, expected, sessions)
    )
    job = {
        "run_id": run_id,
        "started_at": stamp(started),
        "finished_at": stamp(finished),
        "status": status,
        "reason": reason,
        "expected_session": expected.strftime("%Y-%m-%d"),
        "data_class": DataClass(data_class).value,
        "failed": list(failed),
        "skipped": list(skipped),
        "calendar": calendar_id,
        "freshness": fresh,
        "activation": activation,
        "results": [dict(row) for row in results],
        "outcomes": None if outcomes is None else dict(outcomes),
    }
    store.record_job_run(job)
    entries, problem = store.verify_ledger()
    kept = check_kept_head(store, kept_head, existed=existed, verified=problem is None)

    all_scores = [score for ticker in tickers for score in store.scores(ticker)]
    summary = scoring.rolling_summary(all_scores, superseded=store.superseded_ids())
    try:
        forecasts = [store.forecasts(ticker) for ticker in tickers]
    except (ForecastRefused, StoreRefused):  # a record changed outside the store: the ledger
        forecasts = []  # check below reports it; the lineage is left empty, never guessed
    latest = max(
        (history[-1] for history in forecasts if history),
        key=lambda record: record["created_at"],
        default=None,
    )
    first = min((history[0]["as_of_session"] for history in forecasts if history), default=None)
    evaluation = None if latest is None else store.evaluation(latest["evaluation_run_id"])
    baseline = None
    development_end = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    if evaluation is not None:
        daily = [
            m["rps"]
            for m in evaluation["metrics"]
            if m["model"] == "B1" and m["target"] != "target_cum"
        ]
        baseline = sum(daily) / len(daily) if daily else None
        if evaluation["config"].get("outcome_end"):
            development_end = pd.Timestamp(evaluation["config"]["outcome_end"], tz="UTC")
    frames = _frames(store, tickers)
    runs = [
        run
        for run in store.job_runs()
        if run["status"] == "ok" and run.get("data_class") == DataClass(data_class).value
    ]
    document = monitoring.monitor_document(
        run_id=run_id,
        generated_at=finished,
        code_sha=code_sha,
        synthetic=DataClass(data_class) is DataClass.SYNTHETIC,
        job={
            key: job[key]
            for key in (
                "status",
                "started_at",
                "finished_at",
                "expected_session",
                "reason",
                "failed",
                "skipped",
                "activation",
                "results",
            )
        },
        freshness=fresh,
        scores={**summary, "baseline_rps": baseline},
        inputs_drift=[dict(row) for row in measured_drift]
        if measured_drift is not None
        else (monitoring.input_drift(frames, development_end) if frames else []),
        traps=trap_outcomes(frames, first),
        ledger={
            "entries": entries,
            "verified": problem is None,
            "reason": problem,
            "head": store.ledger_head(),
            "kept_head": kept,
        },
        lineage={}
        if latest is None
        else {
            "estimator": latest["estimator"],
            "model_version": latest["model_version"],
            "feature_version": latest["feature_version"],
            "evaluation_run_id": latest["evaluation_run_id"],
        },
        schedule={
            "last_success_at": runs[-1]["finished_at"] if runs else None,
            "next_expected_at": next_expected_at,
        },
        symbols=symbol_status(store, tickers),
        outcomes=outcomes,
    )
    return monitoring.write_monitor(health_inputs, document)
