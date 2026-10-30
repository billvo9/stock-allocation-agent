"""
Scoring stored forecasts as their sessions close (standard library, numpy
and pandas only).

A forecast is scored against outcomes it could not have seen:

    - each target session's outcome is the category of that session's
      split-consistent price return, against the forecast's own band (the
      five-session total against its cumulative band);
    - every session a target needs (the target session, or all five for the
      total) must be ABSENT from the store when the forecast was created:
      it was first downloaded afterwards, by any vintage (so a forecast is
      never scored on data it could have seen, even after a later
      re-download, and a rebuilt "forecast" can never score itself). "When
      it was created" is the later of its created_at and the time the store
      itself recorded it (its ledger entry), so a backdated created_at
      cannot hide a batch the store already held;
    - every session a target needs must open after the forecast was made
      (forecast.opened_by), otherwise the target is unscorable;
    - a session without a usable return (a gap, an unverified split) is
      left unscored with its reason, never filled;
    - a score is written once per (forecast, target) and never rewritten:
      the forecast's probabilities are never touched.

A reconstruction (forecast.is_reconstructed; owner decision of 2026-10-30) is
made after its targets began, from prices dated up to its as-of close only,
so it is judged by session dates instead: every target after its as-of
session is scored once its outcome is stored, whenever that was. Its scores
are kept off the live track (store.scores), never mixed with a live one.

Scores per target: the realized category, the realized return, and the
Brier score, log loss and RPS of the stored probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import CATEGORY_CODES, StableBand
from stock_agent.stock_lab.forecast import PROBABILITY_KEYS, is_reconstructed
from stock_agent.stock_lab.forecast import opened_by as forecast_opened_by
from stock_agent.stock_lab.prices import on_calendar, session_returns
from stock_agent.stock_lab.scores import brier, log_loss, rps

TARGETS = ("1", "2", "3", "4", "5", "cum")
PENDING = "pending"
SCORED = "scored"
UNSCORABLE = "unscorable"


@dataclass(frozen=True)
class Score:
    forecast_id: str
    symbol: str
    target: str  # "1".."5" or "cum"
    session: str  # the session whose close settles the target
    status: str  # scored, pending or unscorable
    reason: str | None
    outcome: int | None  # category code (Decrease 0, Stable 1, Increase 2)
    realized_return: float | None
    brier: float | None
    log_loss: float | None
    rps: float | None
    batch_id: str | None  # the price batch that settled it

    def as_row(self) -> dict:
        return dict(self.__dict__)


def _probabilities(record: Mapping, target: str) -> np.ndarray:
    row = record["cumulative"] if target == "cum" else record["daily"][int(target) - 1]
    # Code order: Decrease 0, Stable 1, Increase 2.
    return np.array([[row["p_decrease"], row["p_stable"], row["p_increase"]]])


def _score(common: Mapping, session: str, status: str, reason: str | None, batch_id=None, *,
           outcome=None, value=None, p=None) -> Score:  # fmt: skip
    if status != SCORED:
        return Score(**common, session=session, status=status, reason=reason, outcome=None,
                     realized_return=None, brier=None, log_loss=None, rps=None,
                     batch_id=batch_id)  # fmt: skip
    y = np.array([outcome])
    return Score(
        **common,
        session=session,
        status=SCORED,
        reason=None,
        outcome=outcome,
        realized_return=float(value),
        brier=float(brier(p, y)[0]),
        log_loss=float(log_loss(p, y)[0]),
        rps=float(rps(p, y)[0]),
        batch_id=batch_id,
    )


def score_forecast(
    record: Mapping,
    bars: pd.DataFrame,
    batch_of: Mapping[str, str],
    *,
    seen_before: AbstractSet[str],
    sessions: Sequence[pd.Timestamp] | None = None,
) -> tuple[Score, ...]:
    """
    Every target of one forecast against `bars` (prices.BAR_COLUMNS).
    `batch_of` maps a session (YYYY-MM-DD) to the batch that supplied it
    (provenance only); `seen_before` holds every session the store already
    had when the forecast was created, from ANY vintage: a target needing such
    a session is never scored, even if a later download supplies it again.
    `sessions` is the exchange calendar over the bars' span, so a missing
    session is a gap, never a longer return read as one session.
    """

    if record["refusal"] is not None:
        return ()
    by_date = is_reconstructed(record)  # judged by session dates (module docstring)
    band = StableBand(record["band"])
    frame = session_returns(on_calendar(bars, sessions), band)
    by_day = {row.session.strftime("%Y-%m-%d"): row for row in frame.itertuples()}
    out = []
    targets = [str(h) for h in range(1, 6)] + (["cum"] if record["cumulative"] else [])
    for target in targets:
        session = record["target_sessions"][-1 if target == "cum" else int(target) - 1]
        needed = record["target_sessions"] if target == "cum" else [session]
        common = {
            "forecast_id": record["forecast_id"],
            "symbol": record["symbol"],
            "target": target,
        }
        if not by_date and forecast_opened_by(record, needed):
            out.append(_score(common, session, UNSCORABLE, "forecast_after_target_opened"))
            continue
        row = by_day.get(session)
        if row is None or session not in batch_of:
            out.append(_score(common, session, PENDING, "not_closed_yet"))
            continue
        batch_id = batch_of[session]
        if not by_date and any(day in seen_before for day in needed):
            out.append(_score(common, session, UNSCORABLE, "outcome_retrieved_before_forecast",
                              batch_id))  # fmt: skip
            continue
        if target == "cum":
            start = by_day.get(record["as_of_session"])
            window = [by_day.get(day) for day in needed]
            if start is None or not all(w is not None and w.reason == "ok" for w in window):
                out.append(_score(common, session, UNSCORABLE, "window_incomplete", batch_id))
                continue
            value = row.level / start.level - 1.0
            category = StableBand(record["cumulative_band"]).categorize(value)
        else:
            value = row.price_return if row.reason == "ok" else math.nan
            category = band.categorize(value)
        if category is None:
            out.append(_score(common, session, UNSCORABLE, row.reason, batch_id))
            continue
        out.append(_score(common, session, SCORED, None, batch_id,
                          outcome=CATEGORY_CODES[category], value=value,
                          p=_probabilities(record, target)))  # fmt: skip
    return tuple(out)


def rolling_summary(
    scores: Sequence[Mapping], window: int = 63, *, superseded: AbstractSet[str] = frozenset()
) -> dict:
    """
    The latest `window` scored daily targets (by session): mean Brier, log
    loss and RPS, observed frequency of each category, and the count. Scores
    of a `superseded` forecast (replaced by a recalculation) are left out,
    so one as-of session counts once.
    """

    daily = [
        s
        for s in scores
        if s["status"] == SCORED and s["target"] != "cum" and s["forecast_id"] not in superseded
    ]
    daily = sorted(daily, key=lambda s: (s["session"], s["target"]))[-window * 5 :]
    n = len(daily)
    if not n:
        return {"n": 0}
    outcomes = np.array([s["outcome"] for s in daily])
    return {
        "n": n,
        # Targets overlap (five horizons of different forecasts settle on one
        # session), so the number of distinct sessions is the honest sample size.
        "sessions": len({s["session"] for s in daily}),
        "brier": float(np.mean([s["brier"] for s in daily])),
        "log_loss": float(np.mean([s["log_loss"] for s in daily])),
        "rps": float(np.mean([s["rps"] for s in daily])),
        "observed": {
            key: float((outcomes == code).mean())
            for key, code in zip(PROBABILITY_KEYS, (0, 1, 2), strict=True)
        },
        "first_session": daily[0]["session"],
        "last_session": daily[-1]["session"],
    }


def score_store(
    store,
    symbols: Sequence[str],
    *,
    calendar: Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex] | None = None,
) -> int:
    """
    Score every stored forecast of `symbols` whose targets have closed and are
    not scored yet; appends the new scores (write-once). Returns how many.
    `calendar(first, last)` gives the exchange sessions between two sessions
    (the daily script passes the NYSE rules calendar; the replay weekdays).

    Real forecasts are never scored here: their outcomes come only from the
    inference store (the owner's choice of 2026-10-19; stock_lab.evening.score_real).
    """

    added = 0
    for symbol in symbols:
        stored = store.bars(symbol)
        if stored is None:
            continue
        batch_of = {
            row.session.strftime("%Y-%m-%d"): row.batch_id for row in stored.bars.itertuples()
        }
        bars = stored.bars.drop(columns=["batch_id"])
        sessions = None
        if calendar is not None and len(bars):
            sessions = calendar(bars["session"].iloc[0], bars["session"].iloc[-1])
        done = store.scored_keys(symbol)
        recorded = store.forecast_recorded_at(symbol)
        for record in store.forecasts(symbol):
            if record["data_class"] == "real":
                continue  # scored only from the inference store (docstring)
            if all((record["forecast_id"], t) in done for t in TARGETS):
                continue
            if record["forecast_id"] not in recorded:
                continue  # no ledger entry: the chain check reports it; never scored
            known = max(
                pd.Timestamp(record["created_at"]), pd.Timestamp(recorded[record["forecast_id"]])
            )
            visible = store.bars(symbol, as_of=known)
            seen = (
                set() if visible is None else set(visible.bars["session"].dt.strftime("%Y-%m-%d"))
            )
            fresh = [
                score
                for score in score_forecast(
                    record, bars, batch_of, seen_before=seen, sessions=sessions
                )
                if score.status != PENDING and (score.forecast_id, score.target) not in done
            ]
            if fresh:
                store.record_scores(fresh)
                added += len(fresh)
    return added
