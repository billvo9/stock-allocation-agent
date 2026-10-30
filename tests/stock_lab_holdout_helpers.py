"""
Fixtures for the sealed holdout path: a weekday calendar, a short test
protocol on it, four synthetic stocks under the protocol's tickers (each one
continuous series, development part through 2024-12-31 and holdout part
after it), a frozen model of their development part, and a holdout store on a
settable clock. No network, no real data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import registry, synthetic
from stock_agent.stock_lab.evaluation import EvaluationConfig, evaluate
from stock_agent.stock_lab.holdout.protocol import PROTOCOL, Protocol
from stock_agent.stock_lab.holdout.store import DB_NAME, HoldoutStore

CALENDAR_ID = "test_weekdays_v1"
DEVELOPMENT_END = "2024-12-31"
TEST_END = "2025-02-14"
CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp(DEVELOPMENT_END, tz="UTC")
)
CREATED = pd.Timestamp("2026-10-01T22:00:00Z")
# Each protocol ticker's source: a demo stock and how many sessions its series is shifted
# (the fourth reuses a demo stock at another offset, so no two stocks share returns).
SOURCES = {
    "AMD": ("DEMO_CALM", 0),
    "MRVL": ("DEMO_SWING", 0),
    "MU": ("DEMO_TREND", 0),
    "NVDA": ("DEMO_SWING", 300),
}


def weekdays(start, end) -> pd.DatetimeIndex:
    """A weekday calendar (no holidays), sessions at 00:00 UTC, both ends inclusive."""

    first, last = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return pd.bdate_range(first, last, tz="UTC").as_unit("us")


def ny(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC")


def series(ticker: str, end: str = TEST_END) -> pd.DataFrame:
    """The ticker's whole synthetic series, its sessions the weekdays ending at `end`."""

    stock, offset = SOURCES[ticker]
    source = synthetic.demo_bars(stock)
    source = source.iloc[offset:].reset_index(drop=True) if offset else source
    days = weekdays(pd.Timestamp(end) - pd.Timedelta(days=3 * len(source)), end)[-len(source) :]
    frame = source.copy()
    frame["session"] = days
    return frame


def development(ticker: str) -> pd.DataFrame:
    frame = series(ticker)
    return frame[frame["session"] <= pd.Timestamp(DEVELOPMENT_END, tz="UTC")].reset_index(drop=True)


def short_protocol(model_id: str) -> Protocol:
    """The registered protocol's rules on the weekday calendar, from 2025-01-01 to TEST_END."""

    window = weekdays("2023-06-01", DEVELOPMENT_END)[-PROTOCOL.window_sessions :]
    return replace(
        PROTOCOL,
        model_id=model_id,
        fetch_start=window[0].strftime("%Y-%m-%d"),
        first_decision=DEVELOPMENT_END,
        holdout_start="2025-01-01",
        holdout_end=TEST_END,
        blocks=(("A", "2025-01-01"), ("B", "2025-01-15"), ("C", "2025-02-01")),
    )


def holdout_bars(protocol: Protocol, ticker: str) -> pd.DataFrame:
    """The ticker's bars on the protocol's fetched sessions."""

    frame = series(ticker)
    chosen = frame["session"] >= pd.Timestamp(protocol.fetch_start, tz="UTC")
    return frame[chosen].reset_index(drop=True)


def frozen_model():
    """(evaluation record, frozen model record) of the development parts."""

    histories = {ticker: development(ticker) for ticker in SOURCES}
    record = evaluate(histories, CONFIG, data_class="real", run_id="R-holdout", created_at=CREATED)
    return record, registry.freeze(histories, record, CONFIG, created_at=CREATED)


class Clock:
    def __init__(self, now: pd.Timestamp) -> None:
        self.now = now

    def __call__(self) -> pd.Timestamp:
        return self.now


RETRIEVED = ny("2026-10-27", "21:00")


def open_store(directory: Path, protocol: Protocol) -> tuple[HoldoutStore, Clock]:
    clock = Clock(RETRIEVED + pd.Timedelta(minutes=1))
    return HoldoutStore(directory / "holdout" / DB_NAME, protocol=protocol, clock=clock), clock


def fill(store: HoldoutStore, protocol: Protocol, tickers=None) -> None:
    """One batch per ticker, as the fetch script stores them."""

    for ticker in tickers or protocol.tickers:
        store.record_batch(
            ticker,
            holdout_bars(protocol, ticker),
            source="fixture",
            retrieved_at=RETRIEVED,
            sessions=weekdays,
            calendar_id=CALENDAR_ID,
            vintage_note="fixture bars",
        )
