"""
Scoring stored forecasts as their sessions close (standard library, numpy
and pandas only).

A forecast is scored against outcomes it could not have seen:

    - each target session's outcome is the category of that session's
      split-consistent price return, against the forecast's own band (the
      five-session total against its cumulative band);
    - the outcome must come from a price batch retrieved AFTER the forecast
      was created (so a forecast is never scored on data it was made with,
      and a rebuilt "forecast" can never score itself);
    - a session without a usable return (a gap, an unverified split) is
      left unscored with its reason, never filled;
    - a score is written once per (forecast, target) and never rewritten:
      the forecast's probabilities are never touched.

Scores per target: the realized category, the realized return, and the
Brier score, log loss and RPS of the stored probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import CATEGORY_CODES, StableBand
from stock_agent.stock_lab.forecast import PROBABILITY_KEYS
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


def score_forecast(
    record: Mapping,
    bars: pd.DataFrame,
    batch_of: Mapping[str, tuple[str, str]],
    *,
    sessions: Sequence[pd.Timestamp] | None = None,
) -> tuple[Score, ...]:
    """
    Every target of one forecast against `bars` (prices.BAR_COLUMNS), where
    `batch_of` maps a session (YYYY-MM-DD) to (batch_id, retrieved_at) of the
    vintage that supplied it.
    """

    if record["refusal"] is not None:
        return ()
    created = record["created_at"]
    band = StableBand(record["band"])
    frame = session_returns(on_calendar(bars, sessions), band)
    by_day = {row.session.strftime("%Y-%m-%d"): row for row in frame.itertuples()}
    out = []
    targets = [str(h) for h in range(1, 6)] + (["cum"] if record["cumulative"] else [])
    for target in targets:
        session = record["target_sessions"][-1 if target == "cum" else int(target) - 1]
        common = {
            "forecast_id": record["forecast_id"],
            "symbol": record["symbol"],
            "target": target,
        }
        row = by_day.get(session)
        settled = batch_of.get(session)
        if row is None or settled is None:
            out.append(
                Score(
                    **common,
                    session=session,
                    status=PENDING,
                    reason="not_closed_yet",
                    outcome=None,
                    realized_return=None,
                    brier=None,
                    log_loss=None,
                    rps=None,
                    batch_id=None,
                )
            )
            continue
        batch_id, retrieved_at = settled
        if retrieved_at <= created:
            out.append(
                Score(
                    **common,
                    session=session,
                    status=UNSCORABLE,
                    reason="outcome_retrieved_before_forecast",
                    outcome=None,
                    realized_return=None,
                    brier=None,
                    log_loss=None,
                    rps=None,
                    batch_id=batch_id,
                )
            )
            continue
        if target == "cum":
            start = by_day.get(record["as_of_session"])
            window = [by_day.get(day) for day in record["target_sessions"]]
            ok = start is not None and all(w is not None and w.reason == "ok" for w in window)
            value = (row.level / start.level - 1.0) if ok else math.nan
            category = StableBand(record["cumulative_band"]).categorize(value)
        else:
            value = row.price_return if row.reason == "ok" else math.nan
            category = band.categorize(value)
        if category is None:
            out.append(
                Score(
                    **common,
                    session=session,
                    status=UNSCORABLE,
                    reason=row.reason,
                    outcome=None,
                    realized_return=None,
                    brier=None,
                    log_loss=None,
                    rps=None,
                    batch_id=batch_id,
                )
            )
            continue
        code = CATEGORY_CODES[category]
        p = _probabilities(record, target)
        y = np.array([code])
        out.append(
            Score(
                **common,
                session=session,
                status=SCORED,
                reason=None,
                outcome=code,
                realized_return=float(value),
                brier=float(brier(p, y)[0]),
                log_loss=float(log_loss(p, y)[0]),
                rps=float(rps(p, y)[0]),
                batch_id=batch_id,
            )
        )
    return tuple(out)


def rolling_summary(scores: Sequence[Mapping], window: int = 63) -> dict:
    """
    The latest `window` scored daily targets (by session): mean Brier, log
    loss and RPS, observed frequency of each category, and the count.
    """

    daily = [s for s in scores if s["status"] == SCORED and s["target"] != "cum"]
    daily = sorted(daily, key=lambda s: (s["session"], s["target"]))[-window * 5 :]
    n = len(daily)
    if not n:
        return {"n": 0}
    outcomes = np.array([s["outcome"] for s in daily])
    return {
        "n": n,
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


def score_store(store, symbols: Sequence[str], *, sessions=None) -> int:
    """
    Score every stored forecast of `symbols` whose targets have closed and are
    not scored yet; appends the new scores (write-once). Returns how many.
    """

    added = 0
    for symbol in symbols:
        stored = store.bars(symbol)
        if stored is None:
            continue
        retrieved = {batch.batch_id: batch.retrieved_at for batch in stored.batches}
        batch_of = {
            row.session.strftime("%Y-%m-%d"): (row.batch_id, retrieved[row.batch_id])
            for row in stored.bars.itertuples()
        }
        bars = stored.bars.drop(columns=["batch_id"])
        done = store.scored_keys(symbol)
        for record in store.forecasts(symbol):
            fresh = [
                score
                for score in score_forecast(record, bars, batch_of, sessions=sessions)
                if score.status != PENDING and (score.forecast_id, score.target) not in done
            ]
            if fresh:
                store.record_scores(fresh)
                added += len(fresh)
    return added
