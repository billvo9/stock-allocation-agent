"""
The inference window: what a frozen model may see at a forecast close, and
the timing conventions of a prospective record.

The prospective store is read only through its loader (PR #73), which the
forecast job passes in:

    load_inference_window(store_root, assets, forecast_close, lookback=63,
                          *, retrieved_by)

`retrieved_by` is the record's decision time: only batches retrieved by then
are read. `coerce_window` checks the result and refuses, never repairs:

    sessions        lookback + 1 sessions (00:00 UTC) ending at
                    forecast_close, so `lookback` daily returns (64 closes
                    and 63 returns by default), equal to an independent
                    exchange calendar
    prices          fund_id, session_date, close, adjustment_basis
                    (unadjusted rows are used, other bases ignored)
    distributions   fund_id, ex_date, amount, amount_basis
    splits          fund_id, effective_date, ratio_new_per_old
    missing         fund_id, session_date (and reason): closes the selected
                    batch lacks; the loader never fills them
    batches         batch_id, retrieved_at (and request range, payload and
                    anchors hashes): the lineage of every row
    revisions       fund_id, session_date, table, change: disagreements with
                    older batches (values never shown)

- Any row dated after forecast_close is refused (LeakageError), and so is a
  batch retrieved after `retrieved_by`. The forecast can then depend only on
  sessions up to its close, as known at its decision time.
- Rows dated before the first session are outside the window and dropped.
- An asset with any close missing in the window is incomplete
  (`incomplete_assets`). Its forecast is refused, never computed from
  median-filled features.

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
MISSING_COLUMNS = ("fund_id", "session_date")
BATCH_COLUMNS = ("batch_id", "retrieved_at")
REVISION_COLUMNS = ("fund_id", "session_date", "table", "change")
_DATE_COLUMNS = {
    "prices": ("session_date",),
    "distributions": ("ex_date",),
    "splits": ("effective_date",),
    "missing": ("session_date",),
    "revisions": ("session_date",),
}
_DATA = {"prices": PRICE_COLUMNS, "distributions": DISTRIBUTION_COLUMNS, "splits": SPLIT_COLUMNS}
_LINEAGE = {"missing": MISSING_COLUMNS, "revisions": REVISION_COLUMNS}
INCOMPLETE = "incomplete_inference_window"

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
    missing: pd.DataFrame
    batches: pd.DataFrame
    revisions: pd.DataFrame
    fingerprint: str
    calendar: str

    @property
    def complete(self) -> bool:
        return self.missing.empty

    def incomplete_assets(self, assets: Sequence[str]) -> dict[str, list[str]]:
        """Asset -> the sessions in the window without an unadjusted close, if any."""

        result = {}
        unadjusted = self.prices.loc[self.prices["adjustment_basis"] == "unadjusted"]
        for asset in assets:
            have = set(unadjusted.loc[unadjusted["fund_id"] == asset, "session_date"])
            listed = set(self.missing.loc[self.missing["fund_id"] == asset, "session_date"])
            lacking = sorted({s for s in self.sessions if s not in have} | listed)
            if lacking:
                result[asset] = [str(s.date()) for s in lacking]
        return result

    def lineage(self) -> dict[str, object]:
        """What a record keeps of the window: batches, missing closes, revisions, completeness."""

        def rows(frame: pd.DataFrame) -> list[dict]:
            out = frame.copy()
            for column in out.columns:
                if isinstance(out[column].dtype, pd.DatetimeTZDtype):
                    out[column] = out[column].map(lambda v: None if pd.isna(v) else v.isoformat())
            return out.astype(object).where(out.notna(), None).to_dict("records")

        return {
            "batches": rows(self.batches),
            "missing": rows(self.missing),
            "revisions": rows(self.revisions),
            "complete": self.complete,
        }


def session_stamp(value: object, name: str = "session") -> pd.Timestamp:
    """A session date as 00:00 UTC (naive dates are read as UTC dates)."""

    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise WindowRefused(f"{name} must be a session date, got {stamp}.")
    return stamp.as_unit("ns")


def _has(result: object, name: str) -> bool:
    return name in result if isinstance(result, Mapping) else hasattr(result, name)


def _part(result: object, name: str) -> object:
    if not _has(result, name):
        raise WindowRefused(f"The inference window has no {name!r}.")
    return result[name] if isinstance(result, Mapping) else getattr(result, name)


def _frame(result: object, name: str, columns: Sequence[str]) -> pd.DataFrame:
    frame = _part(result, name)
    if not isinstance(frame, pd.DataFrame):
        raise WindowRefused(f"window.{name} must be a DataFrame.")
    lacking = sorted(set(columns) - set(frame.columns))
    if lacking:
        raise WindowRefused(f"window.{name} is missing columns {lacking}.")
    return frame.copy()


def _dated(frame: pd.DataFrame, name: str, first: pd.Timestamp, last: pd.Timestamp):
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
    retrieved_by: pd.Timestamp,
) -> InferenceWindow:
    """
    Check a loader result against the contract above (refuses, never repairs).

    `calendar(start, end)` returns the exchange's sessions in [start, end],
    from a schedule independent of the price rows. The window's sessions
    must equal it exactly: a dropped session would turn a two-session return
    into a "one-session" one and shift every horizon by a session.
    """

    close = session_stamp(forecast_close, "forecast_close")
    cutoff = pd.Timestamp(retrieved_by).tz_convert("UTC")
    sessions = pd.DatetimeIndex(pd.to_datetime(_part(result, "sessions"), utc=True)).as_unit("ns")
    if len(sessions) != lookback + 1:
        raise WindowRefused(
            f"The window holds {len(sessions)} sessions, not {lookback + 1} "
            f"({lookback} daily returns)."
        )
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
    tables = {
        name: _dated(_frame(result, name, columns), name, sessions[0], close)
        for name, columns in {**_DATA, **_LINEAGE}.items()
    }
    tables["prices"] = tables["prices"].loc[:, list(PRICE_COLUMNS)]
    tables["distributions"] = tables["distributions"].loc[:, list(DISTRIBUTION_COLUMNS)]
    tables["splits"] = tables["splits"].loc[:, list(SPLIT_COLUMNS)]
    batches = _frame(result, "batches", BATCH_COLUMNS)
    batches["retrieved_at"] = pd.to_datetime(batches["retrieved_at"], utc=True)
    if (batches["retrieved_at"] > cutoff).any():
        raise LeakageError(
            f"The window uses a batch retrieved after the decision time {cutoff.isoformat()}."
        )
    absent = sorted(
        set(assets) - set(tables["prices"]["fund_id"]) - set(tables["missing"]["fund_id"])
    )
    if absent:
        raise WindowRefused(f"The window has no prices for {absent}.")
    fingerprint = _part(result, "fingerprint") if _has(result, "fingerprint") else None
    if fingerprint is None:
        fingerprint = sha256_json(
            {
                "sessions": [str(s.date()) for s in sessions],
                "batches": table_sha256(batches.reset_index(drop=True)),
                **{name: table_sha256(frame) for name, frame in tables.items()},
            }
        )
    window_calendar = _part(result, "calendar_id") if _has(result, "calendar_id") else None
    return InferenceWindow(
        close,
        sessions,
        tables["prices"],
        tables["distributions"],
        tables["splits"],
        tables["missing"],
        batches.reset_index(drop=True),
        tables["revisions"],
        fingerprint=str(fingerprint),
        calendar=str(window_calendar or calendar_id(calendar)),
    )


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
