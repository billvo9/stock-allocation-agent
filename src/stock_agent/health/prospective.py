"""
Data quality and vendor-basis warnings for the prospective track (local
monitor only): the owner's distribution-anchors file, the stored prospective
batches with their spot checks, and the window's coverage.

Research code may not import the prospective store, its rules calendar or
its collector (tests/test_prospective_boundaries.py), and this package is
research code. So nothing here imports them: the private script
(`scripts/health_check.py local`) passes in the store's only reader
(`load_inference_window`), the rules calendar and the anchors loader as
`ProspectiveSources`, and these checks read only what those return: counts,
dates, batch ids and hashes. No close, distribution amount or anchor amount
is ever copied into a summary.

data.anchors
    The anchors file in the prospective scope: its sha256, row counts by
    status, anchors per fund and rows outside the scope. Refused by the
    loader (the reason names a row and column, never a value) is failed; no
    file (an empty allowlist: nothing can pass the distribution-basis spot
    check) or no loader is insufficient_evidence; a fund with no anchor in
    scope is degraded; otherwise healthy.
data.store_batches
    The batches the store's reader selects for the window (WINDOW_START to
    the last complete session; for each fund and session the newest covering
    batch): each batch's id, retrieval time, request and the anchors file its
    spot check used (a batch is stored only after its distributions pass that
    check). A batch recording no anchors file has no spot-check evidence and
    is degraded (`spot_check_not_recorded`). Revisions between batches (a
    vendor serving a different value for the same fund and date) are
    degraded, a vendor-basis warning; no batch yet is insufficient_evidence
    (storing awaits the owner's approval).
data.window_coverage
    Coverage of the window: closes present over fund-sessions expected, per
    fund, with the reason for each missing close (`no_batch`,
    `missing_session`). Complete is healthy; any missing close is degraded
    (a forecast for that fund would be refused, never filled).

A refusal by the store's reader (an edited batch, a foreign entry) makes
both store checks unavailable (`store_refused`), as for the F1 store. With no
asset named, every check is insufficient_evidence (`no_assets`), never a
healthy "0 of 0". The
last complete session is the last rules-calendar session whose close is
available (18:00 America/New_York, the descriptive convention, not a vendor
publication time).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.data.funds.schema import FundDataRefused
from stock_agent.forecasting.prospective.window import local_instant
from stock_agent.health.inputs import Finding, worst_finding
from stock_agent.health.ledger import RulesCalendar
from stock_agent.health.model import Area, Better, Check, HealthState

TITLES = {
    "data.anchors": "Distribution anchors file",
    "data.store_batches": "Prospective batches and spot checks",
    "data.window_coverage": "Prospective window coverage",
}
CLOSE_LOCAL = "18:00"
TIMEZONE = "America/New_York"
SHOWN = 12


@dataclass(frozen=True)
class ProspectiveSources:
    """
    What the private script passes in from the prospective side.

    load_window(store_root, assets, forecast_close, lookback=..., retrieved_by=...)
        the store's reader, `load_inference_window`
    load_anchors()
        the anchors in the prospective scope (`load_anchors`), raising a
        ValueError (`AnchorsRefused`) when the file is refused
    lookback
        the registered config's daily returns per window, for the launch
        readiness report (None: not reported)
    """

    store_root: Path | None
    assets: tuple[str, ...]
    window_start: pd.Timestamp
    h_end: pd.Timestamp
    calendar: RulesCalendar
    load_window: Callable[..., object]
    load_anchors: Callable[[], object] | None = None
    lookback: int | None = None


def _check(check_id: str, **fields) -> Check:
    return Check(check_id=check_id, area=Area.DATA, title=TITLES[check_id], **fields)


def _not_judged(
    check_id: str,
    state: HealthState,
    reason: str,
    summary: str,
    detail: str,
    evidence: str | None,
    measured_at: datetime | None = None,
) -> Check:
    return _check(
        check_id,
        state=state,
        reason=reason,
        measured_at=measured_at,
        value=None,
        unit=None,
        threshold=None,
        summary=summary,
        detail=detail,
        evidence=evidence,
    )


# --- data.anchors -------------------------------------------------------------------


def anchors_check(sources: ProspectiveSources | None, *, now: datetime) -> tuple[Check, str | None]:
    """`data.anchors`, and the file's sha256 when it was read."""

    check_id = "data.anchors"
    threshold = "the file is valid and every prospective fund has an anchor in scope"
    if sources is None or sources.load_anchors is None:
        not_checked = _not_judged(
            check_id,
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            "The anchors file was not checked in this run.",
            "No anchors loader was passed to this run.",
            None,
        )
        return not_checked, None
    try:
        anchors = sources.load_anchors()
    except ValueError as error:
        reason = getattr(getattr(error, "reason", None), "value", None)
        refused = _check(
            check_id,
            state=HealthState.FAILED,
            reason="anchors_refused",
            measured_at=now,
            value=None,
            unit=None,
            threshold=threshold,
            summary="The anchors file was refused, so no distribution can pass its spot check.",
            detail=f"refused ({reason or 'invalid'}): {getattr(error, 'detail', error)}",
            evidence=None,
        )
        return refused, None
    path = getattr(anchors, "path", None)
    evidence = None if path is None else str(path)
    if not sources.assets:
        no_assets = _not_judged(
            check_id,
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_assets",
            "No prospective fund was named, so the anchors are not judged.",
            "The registered prospective config lists no asset.",
            evidence,
            measured_at=now,
        )
        return no_assets, anchors.sha256
    if anchors.sha256 is None:
        missing = _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="anchors_missing",
            measured_at=now,
            value=None,
            unit=None,
            threshold=threshold,
            summary="There is no anchors file yet: an empty allowlist, so no fund can pass the "
            "distribution-basis spot check and nothing can be stored.",
            detail=f"{path} does not exist; the owner fills it by hand.",
            evidence=evidence,
        )
        return missing, None
    per_fund = anchors.per_fund()
    counts = {fund: per_fund.get(fund, 0) for fund in sources.assets}
    without = [fund for fund, count in counts.items() if count == 0]
    statuses = ", ".join(f"{name} {count}" for name, count in anchors.status_counts.items())
    shown = ", ".join(f"{fund} {count}" for fund, count in counts.items())
    lines = [
        f"sha256 {anchors.sha256}",
        f"rows {anchors.rows} ({statuses or 'none'})",
        f"prospective-scope anchors per fund: {shown}",
        f"anchors outside the prospective scope (counted, not used): {anchors.outside_scope}",
        "amounts are never shown",
    ]
    total = sum(per_fund.values())
    if without:
        finding = Finding(
            HealthState.DEGRADED,
            "fund_without_anchors",
            f"{', '.join(without)} has no anchor in the prospective scope, so its distributions "
            "cannot pass the spot check.",
        )
    else:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"The anchors file is valid: {total} anchors in the prospective scope across "
            f"{len(counts)} funds.",
        )
    valid = _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=total,
        unit="anchors in the prospective scope",
        threshold=threshold,
        better=Better.HIGHER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        evidence=evidence,
    )
    return valid, anchors.sha256


# --- the window and its batches -------------------------------------------------------


def last_complete_session(calendar: RulesCalendar, start, now: datetime) -> pd.Timestamp | None:
    """The last calendar session from `start` whose 18:00 New York close is at or before `now`."""

    stamp = pd.Timestamp(now)
    sessions = calendar.sessions(pd.Timestamp(start), stamp.tz_convert("UTC").normalize())
    done = [s for s in sessions if local_instant(s, CLOSE_LOCAL, TIMEZONE) <= stamp]
    return done[-1] if done else None


@dataclass(frozen=True)
class _Window:
    window: object | None
    sessions: pd.DatetimeIndex | None = None
    state: HealthState | None = None
    reason: str | None = None
    summary: str = ""
    detail: str = ""


def _load(sources: ProspectiveSources | None, now: datetime) -> _Window:
    if sources is None or sources.store_root is None:
        return _Window(
            None,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="input_not_configured",
            summary="The prospective store was not checked in this run.",
            detail="No --prospective-store was given to this run.",
        )
    root = sources.store_root
    if not sources.assets:
        return _Window(
            None,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_assets",
            summary="No prospective fund was named, so the store is not judged.",
            detail="The registered prospective config lists no asset.",
        )
    if not root.exists() and not root.is_symlink():
        return _Window(
            None,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="store_missing",
            summary="Nothing has been stored in the prospective store yet; storing awaits the "
            "owner's approval.",
            detail=f"{root} does not exist.",
        )
    try:
        last = last_complete_session(sources.calendar, sources.window_start, now)
        if last is None or last < sources.h_end:
            return _Window(
                None,
                state=HealthState.INSUFFICIENT_EVIDENCE,
                reason="window_not_open",
                summary="The prospective window has no complete session yet.",
                detail=f"last complete session {last}; the window opens at {sources.h_end:%Y-%m-%d}.",
            )
        sessions = sources.calendar.sessions(sources.window_start, last)
        window = sources.load_window(
            root,
            sources.assets,
            last,
            lookback=len(sessions) - 1,
            retrieved_by=pd.Timestamp(now).tz_convert("UTC"),
        )
    except FundDataRefused as error:
        reason = getattr(error.reason, "value", str(error.reason))
        return _Window(
            None,
            state=HealthState.UNAVAILABLE,
            reason="store_refused",
            summary="The prospective store's reader refused the store, so it is not judged.",
            detail=f"load_inference_window refused {root}: {reason}: {error.detail}",
        )
    except (CalendarRefused, OSError) as error:
        return _Window(
            None,
            state=HealthState.UNAVAILABLE,
            reason="store_unreadable",
            summary="The prospective store could not be read.",
            detail=f"{root}: {type(error).__name__}: {error}",
        )
    return _Window(window, sessions=sessions)


def _dates(values: Sequence[str]) -> str:
    shown = ", ".join(values[:SHOWN])
    return shown + (f" and {len(values) - SHOWN} more" if len(values) > SHOWN else "")


def _stamp(value) -> str:
    return "unknown" if value is None or pd.isna(value) else f"{pd.Timestamp(value):%Y-%m-%d %H:%M}"


def _batches_check(loaded: _Window, sources, anchors_sha256, *, now, evidence) -> Check:
    check_id = "data.store_batches"
    if loaded.window is None:
        return _not_judged(
            check_id, loaded.state, loaded.reason, loaded.summary, loaded.detail, evidence
        )
    batches = loaded.window.batches
    revisions = loaded.window.revisions
    threshold = "every stored batch spot-checked; no value revised between batches"
    if len(batches) == 0:
        return _check(
            check_id,
            state=HealthState.INSUFFICIENT_EVIDENCE,
            reason="no_batches",
            measured_at=now,
            value=0,
            unit="batches used for the window",
            threshold=threshold,
            summary="No prospective batch covers the window yet; storing awaits the owner's "
            "approval.",
            detail=f"window {loaded.sessions[0]:%Y-%m-%d} to {loaded.sessions[-1]:%Y-%m-%d}",
            evidence=evidence,
        )
    lines, unchecked = [], []
    for row in batches.sort_values("retrieved_at").itertuples(index=False):
        sha = row.anchors_sha256 if isinstance(row.anchors_sha256, str) else None
        if sha is None:
            spot = "no anchors file recorded"
        elif anchors_sha256 is not None and sha == anchors_sha256:
            spot = f"spot-checked against the current anchors file {sha[:12]}"
        else:
            spot = f"spot-checked against an earlier anchors file {sha[:12]}"
        if sha is None:
            unchecked.append(str(row.batch_id))
        lines.append(
            f"{row.batch_id}: retrieved {_stamp(row.retrieved_at)} UTC, request "
            f"{_stamp(row.request_start)[:10]} to {_stamp(row.request_end)[:10]}, {spot}"
        )
    changes = Counter(revisions["change"]) if len(revisions) else Counter()
    lines.append(
        "revisions between batches: "
        + (", ".join(f"{change} {n}" for change, n in sorted(changes.items())) or "none")
        + (" (values are never shown)" if changes else "")
    )
    findings = []
    if unchecked:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "spot_check_not_recorded",
                f"{len(unchecked)} batches record no anchors file, so their distribution spot "
                f"check is not evidenced: {', '.join(unchecked[:3])}.",
            )
        )
    if len(revisions):
        by_fund = Counter(revisions["fund_id"])
        lines.append(
            "revised fund-dates per fund: "
            + ", ".join(f"{fund} {n}" for fund, n in sorted(by_fund.items()))
        )
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "vendor_revisions",
                f"{len(revisions)} stored values differ between batches: the vendor revised its "
                "data, so the newest batch is used and the difference is listed.",
            )
        )
    findings.append(
        Finding(
            HealthState.HEALTHY,
            None,
            f"{len(batches)} batches cover the window, each stored after its distribution "
            "spot check; no value differs between batches.",
        )
    )
    finding = worst_finding(findings)
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=len(batches),
        unit="batches used for the window",
        threshold=threshold,
        better=Better.HIGHER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        evidence=evidence,
    )


def _window_check(loaded: _Window, sources, *, now, evidence) -> Check:
    check_id = "data.window_coverage"
    if loaded.window is None:
        return _not_judged(
            check_id, loaded.state, loaded.reason, loaded.summary, loaded.detail, evidence
        )
    window = loaded.window
    if len(window.batches) == 0:
        return _not_judged(
            check_id,
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_batches",
            "No prospective batch covers the window yet, so its coverage is not judged.",
            f"window {loaded.sessions[0]:%Y-%m-%d} to {loaded.sessions[-1]:%Y-%m-%d}",
            evidence,
            measured_at=now,
        )
    sessions = len(window.sessions)
    expected = sessions * len(sources.assets)
    missing = window.missing
    span = f"{window.sessions[0]:%Y-%m-%d} to {window.sessions[-1]:%Y-%m-%d}"
    funds = len(sources.assets)
    lines = [f"window {span}: {sessions} sessions x {funds} funds = {expected} closes expected"]
    for fund in sources.assets:
        rows = missing[missing["fund_id"] == fund]
        reasons = Counter(rows["reason"])
        dates = [f"{pd.Timestamp(d):%Y-%m-%d}" for d in sorted(rows["session_date"])]
        text = f"{fund}: {sessions - len(rows)} of {sessions} closes"
        if len(rows):
            text += (
                " ("
                + ", ".join(f"{reason} {n}" for reason, n in sorted(reasons.items()))
                + f"): {_dates(dates)}"
            )
        lines.append(text)
    present = expected - len(missing)
    coverage = present / expected if expected else 0.0
    if len(missing):
        short = sorted(set(missing["fund_id"]))
        finding = Finding(
            HealthState.DEGRADED,
            "closes_missing",
            f"{len(missing)} of {expected} closes in the window are missing ({', '.join(short)}); "
            "a forecast for those funds would be refused, never filled.",
        )
    else:
        finding = Finding(
            HealthState.HEALTHY,
            None,
            f"Every close in the window is present: {expected} of {expected}.",
        )
    return _check(
        check_id,
        state=finding.state,
        reason=finding.reason,
        measured_at=now,
        value=round(coverage, 6),
        unit="fraction of expected closes present",
        threshold="every fund-session in the window has a close",
        better=Better.HIGHER,
        tolerance=0,
        summary=finding.summary,
        detail="\n".join(lines),
        evidence=evidence,
    )


def prospective_checks(sources: ProspectiveSources | None, *, now: datetime) -> list[Check]:
    """`data.anchors`, `data.store_batches` and `data.window_coverage`."""

    anchors, sha = anchors_check(sources, now=now)
    loaded = _load(sources, now)
    evidence = None if sources is None or sources.store_root is None else str(sources.store_root)
    return [
        anchors,
        _batches_check(loaded, sources, sha, now=now, evidence=evidence),
        _window_check(loaded, sources, now=now, evidence=evidence),
    ]


__all__ = ["TITLES", "ProspectiveSources", "last_complete_session", "prospective_checks"]
