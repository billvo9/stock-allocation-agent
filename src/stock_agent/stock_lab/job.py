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
    ledger          whether the hash chain verifies
    lineage         the latest forecast's estimator and versions
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import monitoring, scoring
from stock_agent.stock_lab.contracts import DEVELOPMENT_END, DataClass
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.store import StockLabStore, stamp
from stock_agent.stock_lab.traps import CONFIRMED, UNRESOLVED, trap_events

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]


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
) -> Path:
    """Record the run, its ledger entry and its health input; returns the input's path."""

    fresh = freshness(store, tickers, expected, sessions)
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
    }
    store.record_job_run(job)
    entries, problem = store.verify_ledger()

    all_scores = [score for ticker in tickers for score in store.scores(ticker)]
    summary = scoring.rolling_summary(all_scores)
    forecasts = [store.forecasts(ticker) for ticker in tickers]
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
            )
        },
        freshness=fresh,
        scores={**summary, "baseline_rps": baseline},
        inputs_drift=monitoring.input_drift(frames, development_end) if frames else [],
        traps=trap_outcomes(frames, first),
        ledger={
            "entries": entries,
            "verified": problem is None,
            "reason": problem,
            "head": store.ledger_head(),
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
    )
    return monitoring.write_monitor(health_inputs, document)
