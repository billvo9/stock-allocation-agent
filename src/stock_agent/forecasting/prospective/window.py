"""
The inference window: what a frozen model may see at a forecast close, and
the timing conventions of a prospective record.

The prospective store is read only through the `real-etf-data` loader

    load_inference_window(store_root, assets, forecast_close, lookback=63)

whose result `coerce_window` turns into an `InferenceWindow` and checks:

    sessions        exactly `lookback` sessions (00:00 UTC), strictly
                    increasing, the last one equal to forecast_close
    prices          fund_id, session_date, close, adjustment_basis
                    (unadjusted rows are used, other bases ignored)
    distributions   fund_id, ex_date, amount, amount_basis
    splits          fund_id, effective_date, ratio_new_per_old
    fingerprint     SHA-256 of what was read (computed here if absent)

The sessions must equal an independent exchange calendar (a `calendar`
callable; the record names it). Any row dated after forecast_close is
refused (LeakageError): the forecast
can then depend only on sessions up to its close. Rows dated before the
first session are outside the window and dropped.

Timing conventions (assumed, not verified; recorded in every forecast):

    close available   18:00 America/New_York on the session (the OD9
                      descriptive convention; not proof of vendor
                      publication time)
    record deadline   09:30 America/New_York on the first weekday after the
                      as-of session. No session can open earlier (a holiday
                      only pushes the next session later), so a record made
                      before then cannot have seen any later session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from stock_agent.model_diagnostics.record import sha256_json
from stock_agent.model_validation.audit import LeakageError, table_sha256

PRICE_COLUMNS = ("fund_id", "session_date", "close", "adjustment_basis")
DISTRIBUTION_COLUMNS = ("fund_id", "ex_date", "amount", "amount_basis")
SPLIT_COLUMNS = ("fund_id", "effective_date", "ratio_new_per_old")
_DATE_COLUMNS = {
    "prices": ("session_date",),
    "distributions": ("ex_date",),
    "splits": ("effective_date",),
}
_COLUMNS = {"prices": PRICE_COLUMNS, "distributions": DISTRIBUTION_COLUMNS, "splits": SPLIT_COLUMNS}


Calendar = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]


class WindowRefused(ValueError):
    """The loader's result does not satisfy the window contract."""


@dataclass(frozen=True, eq=False)
class InferenceWindow:
    forecast_close: pd.Timestamp
    sessions: pd.DatetimeIndex
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame
    fingerprint: str


def session_stamp(value: object, name: str = "session") -> pd.Timestamp:
    """A session date as 00:00 UTC (naive dates are read as UTC dates)."""

    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise WindowRefused(f"{name} must be a session date, got {stamp}.")
    return stamp.as_unit("ns")


def _part(result: object, name: str) -> object:
    if isinstance(result, Mapping):
        if name not in result:
            raise WindowRefused(f"The inference window has no {name!r}.")
        return result[name]
    if not hasattr(result, name):
        raise WindowRefused(f"The inference window has no {name!r}.")
    return getattr(result, name)


def _table(result: object, name: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    frame = _part(result, name)
    if not isinstance(frame, pd.DataFrame):
        raise WindowRefused(f"window.{name} must be a DataFrame.")
    missing = sorted(set(_COLUMNS[name]) - set(frame.columns))
    if missing:
        raise WindowRefused(f"window.{name} is missing columns {missing}.")
    frame = frame.loc[:, list(_COLUMNS[name])].copy()
    keep = pd.Series(True, index=frame.index)
    for column in _DATE_COLUMNS[name]:
        dates = pd.to_datetime(frame[column], utc=True).dt.as_unit("ns")
        if (dates > last).any():
            raise LeakageError(
                f"window.{name}.{column} holds a date after the forecast close {last.date()}; "
                "a forecast may use only sessions up to its close."
            )
        frame[column] = dates
        keep &= dates >= first
    return frame.loc[keep].reset_index(drop=True)


def calendar_id(calendar: Calendar) -> str:
    """How a record names the exchange calendar it was checked against."""

    named = getattr(calendar, "calendar_id", None)
    return str(named) if named else f"{calendar.__module__}.{calendar.__qualname__}"


def coerce_window(
    result: object,
    *,
    assets: Sequence[str],
    forecast_close: object,
    lookback: int,
    calendar: Calendar,
) -> InferenceWindow:
    """
    Check a loader result against the contract above (refuses, never repairs).

    `calendar(start, end)` returns the exchange's sessions in [start, end],
    from a schedule independent of the price rows. The window's sessions
    must equal it exactly: a dropped session would turn a two-session return
    into a "one-session" one and shift every horizon by a session.
    """

    close = session_stamp(forecast_close, "forecast_close")
    sessions = pd.DatetimeIndex(pd.to_datetime(_part(result, "sessions"), utc=True)).as_unit("ns")
    if len(sessions) != lookback:
        raise WindowRefused(f"The window holds {len(sessions)} sessions, not {lookback}.")
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise WindowRefused("Window sessions must be strictly increasing.")
    if (sessions != sessions.normalize()).any():
        raise WindowRefused("Window sessions must be session dates at 00:00 UTC.")
    if sessions[-1] > close:
        raise LeakageError("The window's sessions run past the forecast close.")
    if sessions[-1] != close:
        raise WindowRefused(f"The window ends {sessions[-1].date()}, not at {close.date()}.")
    expected = pd.DatetimeIndex(pd.to_datetime(calendar(sessions[0], close), utc=True))
    if not sessions.equals(expected.as_unit("ns")):
        raise WindowRefused(
            f"The window's sessions differ from the exchange calendar {calendar_id(calendar)} "
            f"between {sessions[0].date()} and {close.date()}."
        )
    tables = {name: _table(result, name, sessions[0], close) for name in _COLUMNS}
    absent = sorted(set(assets) - set(tables["prices"]["fund_id"]))
    if absent:
        raise WindowRefused(f"The window has no prices for {absent}.")
    fingerprint = _part(result, "fingerprint") if _has(result, "fingerprint") else None
    if fingerprint is None:
        fingerprint = sha256_json(
            {
                "sessions": [str(s.date()) for s in sessions],
                **{name: table_sha256(frame) for name, frame in tables.items()},
            }
        )
    return InferenceWindow(close, sessions, **tables, fingerprint=str(fingerprint))


def _has(result: object, name: str) -> bool:
    return name in result if isinstance(result, Mapping) else hasattr(result, name)


def local_instant(session: pd.Timestamp, local_time: str, timezone: str) -> pd.Timestamp:
    """`local_time` on the session's calendar date in `timezone`, as UTC."""

    day = session_stamp(session).strftime("%Y-%m-%d")
    return pd.Timestamp(f"{day} {local_time}").tz_localize(timezone).tz_convert("UTC")


def next_weekday(session: pd.Timestamp) -> pd.Timestamp:
    """The first weekday after `session` (no session can come earlier)."""

    day = session_stamp(session) + pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    return day
