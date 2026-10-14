"""
Data-quality monitors over an F1 fund store (`stock_agent.data.funds`):
freshness, coverage, missing sessions and cross-source discrepancies.

They read the store's frames and compute counts; they never modify, fill,
drop or adjust a row (AGENTS.md: anomaly detectors flag data). They compute
no return or other financial quantity.

Definitions (thresholds are operator defaults, documented in the runbook and
not owner-approved policy; change them there and here together):

known at the as-of time
    Only batches retrieved at or before the as-of time count (their rows were
    not known earlier); later batches are excluded and counted in the detail.
latest permitted session
    The last NYSE session strictly before the as-of date, before the holdout
    start (`MODEL_HOLDOUT_START`) and inside the frozen calendar. The store is
    capped at the lockbox by design, so freshness is measured against that
    cap, never against today.
data.freshness
    Per fund, the NYSE sessions after its newest stored close, up to the
    latest permitted session; the value is the largest. Healthy at <= 1.
data.coverage
    Requested fund-sessions (the union of every batch's request) that have a
    close of any adjustment basis, over all requested fund-sessions. Healthy
    at >= 99.5 %, failed below 95 %.
data.missing_sessions
    Requested fund-sessions without a close. Healthy at 0; failed when one
    fund misses 5 or more consecutive sessions (a broken feed, not a gap).
data.vendor_discrepancy
    For each fund, session and adjustment basis held by two or more sources
    (each source's latest retrieval), (max - min) / min of their closes.
    Healthy when none exceeds 0.1 %. With one source there is nothing to
    compare: insufficient_evidence.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.reader import FundStore
from stock_agent.health.model import Area, Better, Check, HealthState
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

FRESH_LAG_SESSIONS = 1
COVERAGE_HEALTHY = 0.995
COVERAGE_FAILED = 0.95
MISSING_RUN_FAILED = 5
DISCREPANCY_TOLERANCE = 0.001

_ONE_DAY = pd.Timedelta(days=1)
_SESSIONS = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
CHECK_IDS = ("data.freshness", "data.coverage", "data.missing_sessions", "data.vendor_discrepancy")
TITLES = {
    "data.freshness": "Data freshness",
    "data.coverage": "Session coverage",
    "data.missing_sessions": "Missing sessions",
    "data.vendor_discrepancy": "Source discrepancies",
}


def latest_permitted_session(as_of: pd.Timestamp) -> pd.Timestamp | None:
    cap = min(as_of.normalize() - _ONE_DAY, MODEL_HOLDOUT_START - _ONE_DAY, nyse.COVERAGE.last)
    eligible = _SESSIONS[_SESSIONS <= cap]
    return eligible[-1] if len(eligible) else None


def requested_pairs(batches: pd.DataFrame) -> pd.DataFrame:
    """Every (fund_id, session_date) any batch requested, once."""

    frames = []
    for row in batches.itertuples(index=False):
        sessions = _SESSIONS[
            (_SESSIONS >= row.request_start) & (_SESSIONS <= row.request_end_exclusive - _ONE_DAY)
        ]
        for fund_id in json.loads(row.request_fund_ids):
            frames.append(pd.DataFrame({"fund_id": fund_id, "session_date": sessions}))
    if not frames:
        return pd.DataFrame({"fund_id": pd.Series(dtype=object), "session_date": _SESSIONS[:0]})
    pairs = pd.concat(frames, ignore_index=True).drop_duplicates()
    return pairs.sort_values(["fund_id", "session_date"], kind="stable").reset_index(drop=True)


def _missing(requested: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    present = prices[["fund_id", "session_date"]].drop_duplicates()
    merged = requested.merge(present, how="left", on=["fund_id", "session_date"], indicator=True)
    merged["missing"] = merged["_merge"].eq("left_only")
    return merged.drop(columns="_merge")


def _longest_run(flags: pd.Series) -> int:
    longest = current = 0
    for flag in flags.tolist():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _day(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _check(check_id: str, **fields: object) -> Check:
    return Check(check_id=check_id, area=Area.DATA, title=TITLES[check_id], **fields)


def not_collected_checks(*, reason: str, state: HealthState, detail: str) -> list[Check]:
    """All four checks when the store was not configured or could not be read."""

    summary = {
        "input_not_configured": "No fund store was configured for this run, so data quality "
        "was not measured.",
    }.get(reason, "The fund store could not be read, so data quality was not measured.")
    return [
        _check(
            check_id,
            state=state,
            reason=reason,
            measured_at=None,
            value=None,
            unit=None,
            threshold=None,
            summary=summary,
            detail=detail,
            evidence=None,
        )
        for check_id in CHECK_IDS
    ]


def freshness_check(
    store: FundStore, requested: pd.DataFrame, *, as_of: pd.Timestamp, **common: object
) -> Check:
    threshold = f"<= {FRESH_LAG_SESSIONS} session behind the latest permitted session"
    permitted = latest_permitted_session(as_of)
    if permitted is None:
        return _check(
            "data.freshness",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="before_coverage",
            value=None,
            unit=None,
            threshold=threshold,
            summary="The as-of date is before the calendar's coverage; nothing is due yet.",
            detail=f"as of {as_of.isoformat()}",
            **common,
        )
    last = store.prices.groupby("fund_id")["session_date"].max()
    funds = sorted(set(requested["fund_id"]) | set(last.index))
    lags = {
        fund: int(((_SESSIONS > last[fund]) & (_SESSIONS <= permitted)).sum())
        for fund in funds
        if fund in last.index
    }
    without = [fund for fund in funds if fund not in last.index]
    newest = store.batches["retrieved_at"].max() if len(store.batches) else None
    retrieval = (
        f"newest retrieval {newest.isoformat()} ({(as_of - newest).days} days before as-of)"
        if newest is not None
        else "no batches"
    )
    listing = ", ".join(
        f"{fund} last close {_day(last[fund])} ({lags[fund]} behind)" for fund in lags
    )
    detail = (
        f"latest permitted session {_day(permitted)} (as of {as_of.isoformat()}); "
        f"{listing or 'no closes'}; {retrieval}"
        + (f"; funds without any close: {', '.join(without)}" if without else "")
    )
    unit = "sessions behind the latest permitted session (stalest fund)"
    if not lags:
        return _check(
            "data.freshness",
            state=HealthState.UNAVAILABLE,
            reason="no_prices",
            value=None,
            unit=unit,
            threshold=threshold,
            summary="The store holds no closes, so freshness cannot be measured.",
            detail=detail,
            **common,
        )
    worst_fund = max(lags, key=lags.__getitem__)
    value = lags[worst_fund]
    if without:
        state, reason = HealthState.DEGRADED, "fund_without_closes"
        summary = f"{len(without)} requested funds have no close at all."
    elif value > FRESH_LAG_SESSIONS:
        state, reason = HealthState.DEGRADED, "stale_data"
        summary = (
            f"{worst_fund}'s newest close is {value} sessions before the latest permitted "
            f"session ({_day(permitted)})."
        )
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"Every fund has closes up to the latest permitted session ({_day(permitted)})."
    return _check(
        "data.freshness",
        state=state,
        reason=reason,
        value=value,
        unit=unit,
        threshold=threshold,
        better=Better.LOWER,
        tolerance=0,
        summary=summary,
        detail=detail,
        **common,
    )


def coverage_check(flagged: pd.DataFrame, **common: object) -> Check:
    threshold = (
        f"healthy >= {COVERAGE_HEALTHY:.1%}; degraded below; failed below {COVERAGE_FAILED:.0%}"
    )
    unit = "fraction of requested fund-sessions with a close"
    expected = len(flagged)
    if expected == 0:
        return _check(
            "data.coverage",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_requests",
            value=None,
            unit=unit,
            threshold=threshold,
            summary="No batch requested any session, so coverage cannot be measured.",
            detail="the batches table is empty",
            **common,
        )
    present = expected - int(flagged["missing"].sum())
    value = round(present / expected, 6)
    per_fund = flagged.groupby("fund_id")["missing"].agg(["size", "sum"])
    detail = f"{present:,} of {expected:,} requested fund-sessions have a close; " + ", ".join(
        f"{fund} {int(row['size'] - row['sum'])}/{int(row['size'])}"
        for fund, row in per_fund.iterrows()
    )
    if value < COVERAGE_FAILED:
        state, reason = HealthState.FAILED, "low_coverage"
    elif value < COVERAGE_HEALTHY:
        state, reason = HealthState.DEGRADED, "low_coverage"
    else:
        state, reason = HealthState.HEALTHY, None
    return _check(
        "data.coverage",
        state=state,
        reason=reason,
        value=value,
        unit=unit,
        threshold=threshold,
        better=Better.HIGHER,
        tolerance=0.0005,
        summary=f"{value:.2%} of requested fund-sessions have a close.",
        detail=detail,
        **common,
    )


def missing_sessions_check(flagged: pd.DataFrame, **common: object) -> Check:
    threshold = f"0 missing; failed at {MISSING_RUN_FAILED}+ consecutive missing for one fund"
    unit = "requested fund-sessions without a close"
    gaps = flagged[flagged["missing"]]
    runs = {fund: _longest_run(group["missing"]) for fund, group in flagged.groupby("fund_id")}
    if len(gaps) == 0:
        return _check(
            "data.missing_sessions",
            state=HealthState.HEALTHY if len(flagged) else HealthState.INSUFFICIENT_EVIDENCE,
            reason=None if len(flagged) else "no_requests",
            value=0 if len(flagged) else None,
            unit=unit,
            threshold=threshold,
            better=Better.LOWER,
            tolerance=0,
            summary="No requested session is missing a close."
            if len(flagged)
            else "No batch requested any session.",
            detail=f"{len(flagged):,} requested fund-sessions checked",
            **common,
        )
    longest_fund = max(runs, key=runs.__getitem__)
    examples = ", ".join(
        f"{row.fund_id} {_day(row.session_date)}" for row in gaps.head(5).itertuples()
    )
    detail = (
        f"{len(gaps)} missing; longest run {runs[longest_fund]} sessions ({longest_fund}); "
        f"first: {examples}"
    )
    if runs[longest_fund] >= MISSING_RUN_FAILED:
        state, reason = HealthState.FAILED, "missing_run"
        summary = (
            f"{longest_fund} misses {runs[longest_fund]} consecutive sessions, which looks like "
            "a broken feed."
        )
    else:
        state, reason = HealthState.DEGRADED, "missing_sessions"
        summary = f"{len(gaps)} requested sessions have no close (none repaired or filled)."
    return _check(
        "data.missing_sessions",
        state=state,
        reason=reason,
        value=len(gaps),
        unit=unit,
        threshold=threshold,
        better=Better.LOWER,
        tolerance=0,
        summary=summary,
        detail=detail,
        **common,
    )


def discrepancy_check(store: FundStore, **common: object) -> Check:
    threshold = f"no fund-session where sources differ by more than {DISCREPANCY_TOLERANCE:.1%}"
    unit = f"fund-sessions where sources differ by more than {DISCREPANCY_TOLERANCE:.1%}"
    sources = sorted(store.prices["source_id"].unique())
    if len(sources) < 2:
        return _check(
            "data.vendor_discrepancy",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="single_source",
            value=None,
            unit=unit,
            threshold=threshold,
            summary="Only one data source is stored, so there is nothing to compare.",
            detail=f"sources: {', '.join(sources) or 'none'}",
            **common,
        )
    retrieved = store.batches.set_index("batch_id")["retrieved_at"]
    latest = (
        store.prices.assign(retrieved_at=store.prices["batch_id"].map(retrieved))
        .sort_values("retrieved_at", kind="stable")
        .drop_duplicates(["fund_id", "session_date", "adjustment_basis", "source_id"], keep="last")
    )
    grouped = latest.groupby(["fund_id", "session_date", "adjustment_basis"])["close"].agg(
        ["size", "min", "max"]
    )
    compared = grouped[grouped["size"] >= 2]
    if compared.empty:
        return _check(
            "data.vendor_discrepancy",
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_overlap",
            value=None,
            unit=unit,
            threshold=threshold,
            summary="The sources hold no fund-session in common, so nothing was compared.",
            detail=f"sources: {', '.join(sources)}",
            **common,
        )
    relative = (compared["max"] - compared["min"]) / compared["min"]
    over = relative[relative > DISCREPANCY_TOLERANCE].sort_values(ascending=False)
    largest = relative.idxmax()
    detail = (
        f"sources {', '.join(sources)}; {len(compared):,} fund-sessions compared; largest "
        f"difference {relative.max():.4%} ({largest[0]} {_day(largest[1])} {largest[2]})"
    )
    if len(over):
        examples = ", ".join(f"{f} {_day(d)} {b} {r:.3%}" for (f, d, b), r in over.head(5).items())
        state, reason = HealthState.DEGRADED, "source_discrepancy"
        summary = f"{len(over)} fund-sessions differ between sources beyond the tolerance."
        detail += f"; beyond tolerance: {examples}"
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"All {len(compared):,} compared fund-sessions agree within the tolerance."
    return _check(
        "data.vendor_discrepancy",
        state=state,
        reason=reason,
        value=len(over),
        unit=unit,
        threshold=threshold,
        better=Better.LOWER,
        tolerance=0,
        summary=summary,
        detail=detail,
        **common,
    )


def data_quality_checks(
    store: FundStore, *, as_of: pd.Timestamp, measured_at: datetime, evidence: str
) -> list[Check]:
    """The four data checks for one store, judged at `as_of`."""

    known = store.batches["retrieved_at"] <= as_of
    later = int((~known).sum())
    batches = store.batches[known]
    known_store = dataclasses.replace(
        store,
        batches=batches,
        prices=store.prices[store.prices["batch_id"].isin(batches["batch_id"])],
    )
    common = {"measured_at": measured_at, "evidence": evidence}
    requested = requested_pairs(known_store.batches)
    flagged = _missing(requested, known_store.prices)
    checks = [
        freshness_check(known_store, requested, as_of=as_of, **common),
        coverage_check(flagged, **common),
        missing_sessions_check(flagged, **common),
        discrepancy_check(known_store, **common),
    ]
    if not later:
        return checks
    note = f"; {later} batches retrieved after the as-of time were excluded"
    return [dataclasses.replace(check, detail=check.detail + note) for check in checks]
