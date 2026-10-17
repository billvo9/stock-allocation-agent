"""
The coverage and corporate-action audit of one ticker's served bars, before
anything is stored (standard library, numpy and pandas only). It reports
aggregates and events, never a price series, and changes nothing: the
owner reads it and then decides, separately, whether to store the batch.

For bars already on the NYSE calendar (sessions the vendor did not serve
are rows without prices), it reports:

    coverage    first and last session, sessions served, sessions missing
                (count and the first few), fields that are null
    sanity      sessions whose low/high do not bracket the open and close,
                sessions with zero volume
    splits      every split ex-date with its factor, the raw close ratio
                close_t / close_(t-1) and the split-consistent ratio
                close_t * factor / close_(t-1), and a verdict:
                consistent (the raw closes show the split), mismatch (they
                do not: the return is withheld and the level breaks there)
                or unverifiable (a factor too small to check)
    dividends   ex-dividend sessions, the largest dividend yield
                div_cash / close_(t-1), and how many ex-dates moved Stable
                at the default band (where the mechanical drop matters)
    moves       sessions that rose 90% or more, or fell 47% or more, without
                a split (flagged for review, never corrected)
    returns     sessions whose return is unavailable, by reason

`ready` is True when nothing calls for review: no split mismatch, no
bracket violation and no unavailable return other than first-close and
gap reasons. It is advice to the owner, not an approval.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import DEFAULT_BAND, ReturnReason
from stock_agent.stock_lab.prices import SPLIT_BAND, session_returns

SHOWN = 5  # dates listed per finding


def _days(values) -> list[str]:
    return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in list(values)[:SHOWN]]


def audit_bars(ticker: str, bars: pd.DataFrame) -> dict:
    """The audit of `bars` (prices.BAR_COLUMNS on the NYSE calendar); see the module docstring."""

    returns = session_returns(bars, DEFAULT_BAND)
    close = bars["close"].to_numpy(dtype="float64")
    served = np.isfinite(close)
    previous = np.concatenate([[np.nan], close[:-1]])
    report: dict = {"ticker": ticker}
    if not served.any():
        return {**report, "ready": False, "issues": ["no closes served"]}
    sessions = pd.DatetimeIndex(bars["session"])
    missing = sessions[~served]
    report["coverage"] = {
        "first_session": sessions[served][0].strftime("%Y-%m-%d"),
        "last_session": sessions[served][-1].strftime("%Y-%m-%d"),
        "sessions_served": int(served.sum()),
        "sessions_missing": len(missing),
        "missing_first": _days(missing),
        "null_fields": {
            column: int(bars.loc[served, column].isna().sum())
            for column in ("open", "high", "low", "volume")
            if bars.loc[served, column].isna().any()
        },
    }
    open_, high, low = (bars[c].to_numpy(dtype="float64") for c in ("open", "high", "low"))
    with np.errstate(invalid="ignore"):
        outside = (
            served
            & np.isfinite(high)
            & np.isfinite(low)
            & ((low > np.fmin(open_, close)) | (high < np.fmax(open_, close)))
        )
    volume = bars["volume"].to_numpy(dtype="float64")
    report["sanity"] = {
        "bracket_violations": int(outside.sum()),
        "bracket_first": _days(sessions[outside]),
        "zero_volume_sessions": int((served & (volume == 0)).sum()),
    }
    factor = bars["split_factor"].to_numpy(dtype="float64")
    splits = []
    for i in np.flatnonzero(factor != 1.0):
        raw = close[i] / previous[i] if math.isfinite(previous[i]) and served[i] else math.nan
        consistent_ratio = raw * factor[i]
        discriminable = abs(math.log(factor[i])) >= 3 * SPLIT_BAND
        if not math.isfinite(raw):
            verdict = "unverifiable"  # a close is missing on one side
        elif not discriminable:
            verdict = "unverifiable"
        elif abs(math.log(consistent_ratio)) < SPLIT_BAND:
            verdict = "consistent"
        else:
            verdict = "mismatch"
        splits.append(
            {
                "session": sessions[i].strftime("%Y-%m-%d"),
                "factor": float(factor[i]),
                "raw_close_ratio": None if not math.isfinite(raw) else round(raw, 4),
                "split_consistent_ratio": None
                if not math.isfinite(consistent_ratio)
                else round(consistent_ratio, 4),
                "verdict": verdict,
            }
        )
    report["splits"] = splits
    dividends = bars["div_cash"].to_numpy(dtype="float64")
    ex = dividends > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        yields = np.where(ex & np.isfinite(previous), dividends / previous, np.nan)
    stable_ex = ex & returns["category"].eq("stable").to_numpy()
    report["dividends"] = {
        "ex_dividend_sessions": int(ex.sum()),
        "largest_yield": None
        if not np.isfinite(yields).any()
        else round(float(np.nanmax(yields)), 5),
        "ex_dates_stable_at_default_band": int(stable_ex.sum()),
    }
    large = returns["large_move"].to_numpy() & np.isfinite(close)
    report["moves"] = {"large_moves": int(large.sum()), "large_first": _days(sessions[large])}
    reasons = returns["reason"].value_counts().to_dict()
    reasons.pop(ReturnReason.OK.value, None)
    report["returns"] = {"unavailable_by_reason": {k: int(v) for k, v in sorted(reasons.items())}}
    benign = {
        ReturnReason.FIRST_CLOSE.value,
        ReturnReason.MISSING_CLOSE.value,
        ReturnReason.PREVIOUS_CLOSE_MISSING.value,
    }
    issues = []
    if any(s["verdict"] == "mismatch" for s in splits):
        issues.append("a split the raw closes do not show")
    if report["sanity"]["bracket_violations"]:
        issues.append("sessions whose low/high do not bracket the open and close")
    if set(reasons) - benign:
        issues.append(f"unavailable returns: {sorted(set(reasons) - benign)}")
    if report["moves"]["large_moves"]:
        issues.append("large moves without a split (review; could be real or an unrecorded split)")
    report["issues"] = issues
    report["ready"] = not issues
    return report


def audit_lines(report: dict) -> list[str]:
    """The report as short lines for the terminal (no prices)."""

    ticker = report["ticker"]
    if "coverage" not in report:
        return [f"{ticker}: {'; '.join(report['issues'])}"]
    cov, san, div = report["coverage"], report["sanity"], report["dividends"]
    lines = [
        f"{ticker}: {cov['sessions_served']} sessions {cov['first_session']} to "
        f"{cov['last_session']}, {cov['sessions_missing']} missing NYSE sessions"
        + (f" (first {', '.join(cov['missing_first'])})" if cov["missing_first"] else ""),
    ]
    if cov["null_fields"]:
        lines.append(f"  null fields: {cov['null_fields']}")
    if san["bracket_violations"] or san["zero_volume_sessions"]:
        lines.append(
            f"  sanity: {san['bracket_violations']} low/high bracket violations, "
            f"{san['zero_volume_sessions']} zero-volume sessions"
        )
    for split in report["splits"]:
        lines.append(
            f"  split {split['session']} factor {split['factor']:g}: raw close ratio "
            f"{split['raw_close_ratio']}, split-consistent {split['split_consistent_ratio']} "
            f"-> {split['verdict']}"
        )
    lines.append(
        f"  dividends: {div['ex_dividend_sessions']} ex-dates, largest yield "
        f"{div['largest_yield']}, {div['ex_dates_stable_at_default_band']} ex-dates Stable at ±0.5%"
    )
    if report["moves"]["large_moves"]:
        lines.append(
            f"  large moves without a split: {report['moves']['large_moves']} "
            f"(first {', '.join(report['moves']['large_first'])})"
        )
    unavailable = report["returns"]["unavailable_by_reason"]
    if unavailable:
        lines.append(f"  unavailable returns: {unavailable}")
    lines.append(
        f"  audit: {'ready to store' if report['ready'] else 'review: ' + '; '.join(report['issues'])}"
    )
    return lines
