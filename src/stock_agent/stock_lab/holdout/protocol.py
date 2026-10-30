"""
The holdout protocol: what the one sealed run evaluates, and how. It was
fixed before any protected outcome was read, and its hash is recorded with
the result, so a changed protocol cannot pass for the registered one.

What is evaluated
    The frozen model MODEL_ID exactly as registered (B1, half-life 252,
    state rule frozen_long_run): nothing is refit, retuned or reselected. At
    each decision session d, every stock's forecast comes from the 252
    sessions ending at d, through `build_panel` and `registry.probabilities`,
    as the evening job computes it: B1 counts the window's outcomes and
    shrinks them toward B0, the long-run mix frozen at 2024-12-31.

Sessions (NYSE rules calendar)
    decisions   every session from FIRST_DECISION (2024-12-31) through the
                session five before HOLDOUT_END (2026-10-07): 443 decisions,
                so that every target session falls in the holdout
    outcomes    2025-01-02 (HOLDOUT_START) through 2026-10-14 (HOLDOUT_END)
    fetched     FETCH_START (2024-01-02, the first session of the first
                decision's window) through HOLDOUT_END: the 2024 sessions are
                development data, inputs to the first windows only

Targets (as in development): each of the next five sessions as Decrease,
Stable or Increase against the stable band (0.5%), and the five-session total
against the band times sqrt(5).

Scores (lower is better): RPS, Brier and log loss.

The deciding test (one comparison, so no Holm)
    Does the model beat B0? Its per-decision RPS difference (model minus B0)
    is averaged over the stocks for each daily target, then over the five
    daily targets: the development's pooled daily differential. The test is
    decision A7's fold-block t, one-sided at ALPHA (2.5%), with calendar
    quarters as the blocks: 2025Q1 (with 2024-12-31) to 2026Q3 (with the
    October 2026 decisions), 7 blocks and 6 degrees of freedom. The model
    beats B0 when the mean difference is negative and its one-sided p-value
    is below 2.5%.

Reported, descriptive only: Brier and log loss, the five-session total,
calibration buckets with the date-clustered 90% Jeffreys intervals of the
briefing (lag 0 for a single session, 4 for the total; groups never pooled),
the per-stock and per-year breakdowns, and the holdout's own base rates
against the frozen B0.

Honest limits (recorded with the result): the stocks were chosen in
hindsight, this is one 21-month period with low power, and the prices are
raw closes and corporate actions as served on the fetch date. A result close
to B0 is the expected outcome; a "yes" would be evidence against B0 for this
period and these stocks, not of a lasting edge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from stock_agent.stock_lab.frozen_model import canonical

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]

SCHEMA = "stock_lab_holdout_protocol"


class Refusal(StrEnum):
    """Why the holdout path refused (machine-readable)."""

    INVALID_REQUEST = "invalid_request"
    PROTOCOL = "protocol"
    LOCATION = "location"
    NOT_HOLDOUT_STORE = "not_holdout_store"
    DATABASE_ERROR = "database_error"
    INTEGRITY = "integrity"
    INVALID_BARS = "invalid_bars"
    RETRIEVAL_TIME = "retrieval_time"
    DATA_CLASS = "data_class"
    ALREADY_FETCHED = "already_fetched"
    INCOMPLETE = "incomplete"
    SPENT = "spent"
    MODEL = "model"


class HoldoutRefused(ValueError):
    """A refusal with a machine-readable `reason` (a Refusal) and plain words."""

    def __init__(self, reason: Refusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def record_id(record: Mapping) -> str:
    """sha256 of an evaluation record's canonical JSON without its own id."""

    content = {key: value for key, value in record.items() if key != "record_id"}
    return hashlib.sha256(canonical(content).encode("utf-8")).hexdigest()


def _day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").as_unit("us")


@dataclass(frozen=True)
class Protocol:
    """One holdout protocol (`PROTOCOL` is the registered one); dates are YYYY-MM-DD."""

    model_id: str
    tickers: tuple[str, ...]
    fetch_start: str
    first_decision: str
    holdout_start: str
    holdout_end: str
    # (name, first date): a decision belongs to the last block starting on or before it;
    # decisions before the second block's start belong to the first.
    blocks: tuple[tuple[str, str], ...]
    window_sessions: int = 252
    horizons: int = 5
    comparator: str = "B0"
    deciding_score: str = "rps"
    deciding_target: str = "daily_pooled"
    deciding_test: str = "fold_block_t_one_sided"
    alpha: float = 0.025
    interval_level: float = 0.95  # the deciding mean's two-sided interval (reported)
    calibration_level: float = 0.9
    scores: tuple[str, ...] = ("rps", "brier", "log_loss")

    def as_dict(self) -> dict:
        found = asdict(self)
        found["tickers"] = list(self.tickers)
        found["blocks"] = [list(block) for block in self.blocks]
        found["scores"] = list(self.scores)
        return {"schema": SCHEMA, **found}

    def sha256(self) -> str:
        text = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def fetch_sessions(self, sessions: Sessions) -> pd.DatetimeIndex:
        """Every session fetched: FETCH_START through HOLDOUT_END, checked (`check_calendar`)."""

        self.check_calendar(sessions)
        return _sessions(sessions, _day(self.fetch_start), _day(self.holdout_end))

    def decisions(self, sessions: Sessions) -> pd.DatetimeIndex:
        """The decision sessions: FIRST_DECISION through `horizons` sessions before the end."""

        days = self.fetch_sessions(sessions)
        decisions = days[days >= _day(self.first_decision)]
        return decisions[: len(decisions) - self.horizons]

    def block_labels(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Each decision's block name (the docstring's rule)."""

        starts = pd.DatetimeIndex([_day(first) for _, first in self.blocks])
        index = starts.searchsorted(pd.DatetimeIndex(dates).as_unit("us"), side="right") - 1
        names = np.array([name for name, _ in self.blocks], dtype=object)
        return names[np.clip(index, 0, len(names) - 1)]

    def check_calendar(self, sessions: Sessions) -> None:
        """
        Refuses (protocol) a calendar on which the protocol's dates do not line up:
        FETCH_START begins the first decision's window of `window_sessions`
        sessions, HOLDOUT_START is the session after FIRST_DECISION, HOLDOUT_END
        is a session, and every block holds a decision.
        """

        first, end = _day(self.first_decision), _day(self.holdout_end)
        days = _sessions(sessions, _day(self.fetch_start), end)
        window = days[days <= first]
        later = days[days > first]
        problems = []
        if len(window) != self.window_sessions or window[-1] != first:
            problems.append(
                f"{self.fetch_start} to {self.first_decision} is not one window of "
                f"{self.window_sessions} sessions"
            )
        if not len(later) or later[0] != _day(self.holdout_start):
            problems.append(f"{self.holdout_start} is not the session after {self.first_decision}")
        if not len(days) or days[-1] != end:
            problems.append(f"{self.holdout_end} is not a session")
        decisions = later[: max(len(later) - self.horizons, 0)].insert(0, first)
        if not problems and len(set(self.block_labels(decisions))) != len(self.blocks):
            problems.append("a block holds no decision")
        if problems:
            raise HoldoutRefused(Refusal.PROTOCOL, "; ".join(problems))


def _sessions(sessions: Sessions, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        found = pd.DatetimeIndex(sessions(start, end))
    except (ValueError, TypeError, KeyError) as error:
        raise HoldoutRefused(
            Refusal.PROTOCOL,
            f"the calendar refused {start.date()} to {end.date()} ({type(error).__name__})",
        ) from None
    if found.tz is None or str(found.tz) != "UTC" or not (found == found.normalize()).all():
        raise HoldoutRefused(Refusal.PROTOCOL, "the calendar must give UTC session dates")
    if not found.is_monotonic_increasing or found.has_duplicates:
        raise HoldoutRefused(Refusal.PROTOCOL, "the calendar's sessions are not in order")
    return found.as_unit("us")


MODEL_ID = "2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7"
TICKERS = ("AMD", "MRVL", "MU", "NVDA")
FETCH_START = "2024-01-02"
FIRST_DECISION = "2024-12-31"
HOLDOUT_START = "2025-01-02"
HOLDOUT_END = "2026-10-14"
QUARTERS = (
    ("2025Q1", "2025-01-01"),
    ("2025Q2", "2025-04-01"),
    ("2025Q3", "2025-07-01"),
    ("2025Q4", "2025-10-01"),
    ("2026Q1", "2026-01-01"),
    ("2026Q2", "2026-04-01"),
    ("2026Q3", "2026-07-01"),
)

PROTOCOL = Protocol(
    model_id=MODEL_ID,
    tickers=TICKERS,
    fetch_start=FETCH_START,
    first_decision=FIRST_DECISION,
    holdout_start=HOLDOUT_START,
    holdout_end=HOLDOUT_END,
    blocks=QUARTERS,
)
