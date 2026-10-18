"""
Fixtures for the Stock Lab's inference store: a weekday exchange calendar
(passed in, as the owner-run scripts pass the NYSE rules calendar), bars on
it from a synthetic demo stock, New York times, and a store on a settable
clock. No network, no real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.inference.schema import DB_NAME, INFERENCE_SESSIONS
from stock_agent.stock_lab.inference.store import InferenceStore

CALENDAR_ID = "test_weekdays_v1"
AS_OF = "2026-10-15"  # a Thursday


def weekdays(start, end) -> pd.DatetimeIndex:
    """A weekday calendar (no holidays), sessions at 00:00 UTC, both ends inclusive."""

    first, last = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return pd.bdate_range(first, last, tz="UTC").as_unit("us")


def ny(day: str, hhmm: str) -> pd.Timestamp:
    """A New York wall-clock time, in UTC."""

    return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC")


def window_start(as_of: str = AS_OF, sessions: int = INFERENCE_SESSIONS) -> str:
    days = weekdays(pd.Timestamp(as_of) - pd.Timedelta(days=500), as_of)
    return days[-sessions].strftime("%Y-%m-%d")


def next_day(day: str, offset: int = 1) -> str:
    """The weekday `offset` weekdays after `day` (before it when negative)."""

    days = weekdays(
        pd.Timestamp(day) - pd.Timedelta(days=40), pd.Timestamp(day) + pd.Timedelta(days=40)
    )
    position = days.get_loc(pd.Timestamp(day, tz="UTC"))
    return days[position + offset].strftime("%Y-%m-%d")


def make_bars(first: str, last: str, *, stock: str = "DEMO_CALM", shift: float = 0.0):
    """Bars for every weekday in [first, last], values from a demo stock (plus `shift`)."""

    sessions = weekdays(first, last)
    source = synthetic.demo_bars(stock)
    frame = source.iloc[-len(sessions) :].copy().reset_index(drop=True)
    frame["session"] = sessions
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + shift
    return frame


class Clock:
    def __init__(self, now: pd.Timestamp) -> None:
        self.now = now

    def __call__(self) -> pd.Timestamp:
        return self.now


def open_store(directory: Path, now: pd.Timestamp | None = None) -> tuple[InferenceStore, Clock]:
    clock = Clock(now or ny(AS_OF, "21:00"))
    return InferenceStore(directory / "inference" / DB_NAME, clock=clock), clock


def record(
    store: InferenceStore,
    clock: Clock,
    first: str,
    last: str,
    *,
    retrieved_at: pd.Timestamp,
    ticker: str = "NVDA",
    bars: pd.DataFrame | None = None,
    data_class: str = "real",
    calendar_id: str = CALENDAR_ID,
):
    """Record one batch, with the store's clock a minute after the retrieval."""

    clock.now = retrieved_at + pd.Timedelta(minutes=1)
    return store.record_batch(
        ticker,
        make_bars(first, last) if bars is None else bars,
        source="fixture",
        retrieved_at=retrieved_at,
        request_start=first,
        request_end=last,
        sessions=weekdays,
        calendar_id=calendar_id,
        data_class=data_class,
        vintage_note="fixture",
    )
