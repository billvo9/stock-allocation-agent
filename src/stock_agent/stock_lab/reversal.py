"""
The reversal analyzer: what happened after a streak or a large move, how
long reversals lasted, and how that differed between calm and volatile
spells. Descriptive statistics of one stock's past (numpy and pandas only).

Events (defined once, never redefined by the page)
    Streak event (k, direction)   a run in `direction` reaches exactly k
        sessions at session d (under the stated Stable rule). Each run is
        counted once per k, so overlapping "at least k" events never count
        one streak several times, and k values are never pooled.
    Move event (X, sign)          a session whose price return exceeds +X
        (or is below -X), declustered: an event within `decluster` sessions
        after the previous counted event of the same kind is skipped, so the
        m-session outcomes of counted events do not overlap.

Outcomes, over m = 1, 3, 5 sessions after d, from the cumulative return
R = L_(d+m) / L_d - 1 against the band tau_m = tau * sqrt(m):
    continuation  R beyond +tau_m in the event's direction
    stabilization |R| <= tau_m
    reversal      R beyond tau_m against the event's direction
For a streak event the one-session outcome is also reported as "the next
session rose / was Stable / fell". An outcome needs every session through
d+m to have a usable return and d+m on or before `outcome_end`; otherwise
the event is counted as censored (pending), never resolved from later data.

Uncertainty: Jeffreys intervals with the effective sample size n_eff (equal
to the event count when unweighted; Kish's formula under the optional
recency weighting, which should use a long half-life, at least 756
sessions). Results with n_eff below MIN_EFFECTIVE say "Insufficient
observations"; below LOW_SAMPLE they are labelled low sample.

Duration: the length of the opposite run that a reversal starts: a run that
immediately follows a completed run of at least k sessions in the other
direction (a Stable run in between means the streak ended without an
immediate reversal; it is not counted). Runs open at
the last session or cut by a gap are right-censored; the survival curve is
the discrete Kaplan-Meier estimate S(l) = prod_(j <= l) (1 - d_j / n_j),
whose hazard at length l equals 1 - P(continue | the run reached l).

Regimes: calm, middle or volatile by the stock's EWMA volatility at d
against the terciles of its own volatility through d (an expanding,
point-in-time rank).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import Category, StableBand
from stock_agent.stock_lab.runs import RunRule, run_table
from stock_agent.stock_lab.scores import jeffreys

MIN_EFFECTIVE = 20
LOW_SAMPLE = 50
OUTCOME_SESSIONS = (1, 3, 5)
OUTCOMES = ("continuation", "stabilization", "reversal")
REGIMES = ("calm", "middle", "volatile")
INSUFFICIENT = "insufficient"
LOW = "low_sample"
OK = "ok"


@dataclass(frozen=True)
class Proportion:
    count: float
    n: int
    n_eff: float
    estimate: float
    ci_low: float
    ci_high: float
    status: str


def proportion(
    hits: np.ndarray, weights: np.ndarray | None = None, level: float = 0.9
) -> Proportion:
    """A (weighted) proportion with its Jeffreys interval and evidence status."""

    hits = np.asarray(hits, dtype="float64")
    n = len(hits)
    if weights is None:
        weights = np.ones(n)
    total = float(weights.sum())
    if n == 0 or total <= 0:
        return Proportion(0.0, 0, 0.0, math.nan, math.nan, math.nan, INSUFFICIENT)
    n_eff = total**2 / float((weights**2).sum())
    estimate = float((weights * hits).sum() / total)
    low, high = jeffreys(estimate * n_eff, n_eff, level)
    status = INSUFFICIENT if n_eff < MIN_EFFECTIVE else LOW if n_eff < LOW_SAMPLE else OK
    if status == INSUFFICIENT:
        estimate, low, high = math.nan, math.nan, math.nan
    return Proportion(float(hits.sum()), n, n_eff, estimate, low, high, status)


def recency_weights(count: int, half_life: float | None) -> np.ndarray:
    """2^(-age / half_life), age in sessions before the last one; equal weights when None."""

    if half_life is None:
        return np.ones(count)
    age = np.arange(count)[::-1].astype("float64")
    return 2.0 ** (-age / half_life)


def volatility_regime(frame: pd.DataFrame) -> pd.Series:
    """
    "calm", "middle" or "volatile" per session: the session's EWMA volatility
    against the terciles of the stock's volatility through that session
    (expanding, point in time); None before 60 values exist.
    """

    sigma = frame["ewma_vol"]
    rank = sigma.expanding(min_periods=60).rank(pct=True)
    labels = np.where(rank <= 1 / 3, "calm", np.where(rank <= 2 / 3, "middle", "volatile"))
    return pd.Series(np.where(rank.notna(), labels, None), index=frame.index, dtype=object)


def _forward_total(frame: pd.DataFrame, m: int, outcome_end: pd.Timestamp | None):
    """(R over m sessions after each d, resolved?) aligned with `frame`."""

    level = frame["level"].to_numpy(dtype="float64")
    usable = frame["price_return"].notna().to_numpy()
    sessions = pd.DatetimeIndex(frame["session"])
    n = len(frame)
    total = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    for i in range(n - m):
        if not usable[i + 1 : i + m + 1].all():
            continue
        if outcome_end is not None and sessions[i + m] > outcome_end:
            continue
        total[i] = level[i + m] / level[i] - 1.0
        resolved[i] = True
    return total, resolved


def classify_outcome(total: np.ndarray, sign: int, band: StableBand, m: int) -> np.ndarray:
    """0 continuation, 1 stabilization, 2 reversal (NaN unresolved), for direction `sign`."""

    tau = band.tau * math.sqrt(m)
    signed = sign * total
    out = np.where(signed > tau, 0.0, np.where(signed < -tau, 2.0, 1.0))
    return np.where(np.isfinite(total), out, np.nan)


@dataclass(frozen=True, eq=False)
class EventSummary:
    """Outcome shares for one event definition, by horizon (and optionally regime)."""

    definition: str
    events: int
    censored: int
    table: (
        pd.DataFrame
    )  # horizon, regime, outcome, count, n, n_eff, estimate, ci_low, ci_high, status


def _summarize(
    definition: str,
    positions: np.ndarray,
    frame: pd.DataFrame,
    sign: int,
    band: StableBand,
    outcome_end,
    half_life: float | None,
    regimes: pd.Series,
) -> EventSummary:
    rows = []
    censored = 0
    for m in OUTCOME_SESSIONS:
        total, resolved = _forward_total(frame, m, outcome_end)
        outcome = classify_outcome(total, sign, band, m)
        chosen = positions[resolved[positions]]
        if m == max(OUTCOME_SESSIONS):
            censored = int(len(positions) - len(chosen))
        weights_all = recency_weights(len(frame), half_life)
        for regime in ("all", *REGIMES):
            subset = chosen if regime == "all" else chosen[regimes.to_numpy()[chosen] == regime]
            for code, name in enumerate(OUTCOMES):
                result = proportion(outcome[subset] == code, weights_all[subset])
                rows.append(
                    {
                        "horizon": m,
                        "regime": regime,
                        "outcome": name,
                        "count": result.count,
                        "n": result.n,
                        "n_eff": result.n_eff,
                        "estimate": result.estimate,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "status": result.status,
                    }
                )
    return EventSummary(definition, len(positions), censored, pd.DataFrame(rows))


def streak_events(frame: pd.DataFrame, k: int, direction: str) -> np.ndarray:
    """Positions where a `direction` run reaches exactly k (frame has run columns)."""

    if k < 1:
        raise ValueError("k must be at least 1")
    reached = frame["run_direction"].eq(direction) & frame["run_length"].eq(k)
    return np.flatnonzero(reached.to_numpy())


def after_streak(
    frame: pd.DataFrame,
    k: int,
    direction: str,
    band: StableBand,
    *,
    outcome_end: pd.Timestamp | None = None,
    half_life: float | None = None,
) -> EventSummary:
    """What followed a run of exactly k sessions in `direction` (module docstring)."""

    sign = 1 if direction == Category.INCREASE.value else -1
    positions = streak_events(frame, k, direction)
    words = {Category.INCREASE.value: "rising", Category.DECREASE.value: "falling"}[direction]
    return _summarize(
        f"{k} {words} sessions in a row",
        positions,
        frame,
        sign,
        band,
        outcome_end,
        half_life,
        volatility_regime(frame),
    )


def move_events(returns: pd.Series, threshold: float, sign: int, decluster: int = 5) -> np.ndarray:
    """Positions of declustered moves beyond `threshold` (in the direction `sign`)."""

    if threshold <= 0:
        raise ValueError("the move threshold must be positive")
    values = sign * returns.to_numpy(dtype="float64")
    candidates = np.flatnonzero(values > threshold)
    kept, last = [], -(10**9)
    for position in candidates:
        if position - last > decluster:
            kept.append(position)
            last = position
    return np.array(kept, dtype="int64")


def after_move(
    frame: pd.DataFrame,
    threshold: float,
    sign: int,
    band: StableBand,
    *,
    outcome_end: pd.Timestamp | None = None,
    half_life: float | None = None,
    decluster: int = 5,
) -> EventSummary:
    """What followed a session that moved more than `threshold` (module docstring)."""

    positions = move_events(frame["price_return"], threshold, sign, decluster)
    kind = "gain" if sign > 0 else "fall"
    return _summarize(
        f"a one-session {kind} of more than {threshold:.1%}",
        positions,
        frame,
        sign,
        band,
        outcome_end,
        half_life,
        volatility_regime(frame),
    )


def next_session(frame: pd.DataFrame, positions: np.ndarray, outcome_end=None) -> dict:
    """Shares of the next session's category after the events, with intervals."""

    sessions = pd.DatetimeIndex(frame["session"])
    codes = frame["category"].to_numpy(dtype=object)
    resolved = [
        p
        for p in positions
        if p + 1 < len(frame)
        and isinstance(codes[p + 1], str)
        and (outcome_end is None or sessions[p + 1] <= outcome_end)
    ]
    following = np.array([codes[p + 1] for p in resolved], dtype=object)
    return {
        category.value: proportion(following == category.value)
        for category in (Category.INCREASE, Category.STABLE, Category.DECREASE)
    }


# --- durations ---


@dataclass(frozen=True, eq=False)
class Survival:
    """Discrete Kaplan-Meier survival of run lengths."""

    lengths: np.ndarray  # 1..max
    at_risk: np.ndarray
    ended: np.ndarray
    hazard: np.ndarray
    survival: np.ndarray  # S(l) = P(length > l)
    median: float | None  # the first l with S(l) <= 0.5; None when not reached
    runs: int
    censored: int


def kaplan_meier(lengths: np.ndarray, censored: np.ndarray) -> Survival:
    lengths = np.asarray(lengths, dtype="int64")
    censored = np.asarray(censored, dtype=bool)
    longest = int(lengths.max()) if len(lengths) else 0
    grid = np.arange(1, longest + 1)
    # A run censored at length c is known to have lasted c sessions, not whether it
    # ended there: it is at risk only for lengths below c.
    at_risk = np.array(
        [((lengths >= l) & ~censored).sum() + ((lengths > l) & censored).sum() for l in grid]
    )
    ended = np.array([((lengths == l) & ~censored).sum() for l in grid])
    with np.errstate(invalid="ignore", divide="ignore"):
        hazard = np.where(at_risk > 0, ended / at_risk, np.nan)
    survival = np.cumprod(1.0 - np.nan_to_num(hazard))
    below = np.flatnonzero(survival <= 0.5)
    median = float(grid[below[0]]) if len(below) else None
    return Survival(
        grid, at_risk, ended, hazard, survival, median, len(lengths), int(censored.sum())
    )


def reversal_durations(
    frame: pd.DataFrame, k: int, direction: str, rule: RunRule, outcome_end=None
) -> Survival:
    """
    Kaplan-Meier survival of the opposite runs that follow a run of at least
    k sessions in `direction`. A run is censored when it is open at the last
    session, cut by a gap, or passes `outcome_end` (it is then cut there).
    """

    table = run_table(frame["session"], frame["category"], rule)
    opposite = (
        Category.INCREASE.value
        if direction == Category.DECREASE.value
        else (Category.DECREASE.value)
    )
    lengths, censored = [], []
    for previous, current in zip(
        table.iloc[:-1].itertuples(), table.iloc[1:].itertuples(), strict=True
    ):
        if previous.direction != direction or previous.length < k or previous.censored:
            continue
        if outcome_end is not None and previous.end > outcome_end:
            continue
        if current.direction != opposite:  # a Stable run in between: no immediate reversal
            continue
        length, cut = int(current.length), bool(current.censored)
        # A run ending on the cutoff is cut too: knowing it ended needs the next session.
        if outcome_end is not None and current.end >= outcome_end:
            if current.start > outcome_end:
                continue
            visible = frame["session"].between(current.start, outcome_end).sum()
            length, cut = int(visible), True
        lengths.append(length)
        censored.append(cut)
    return kaplan_meier(np.array(lengths), np.array(censored))
