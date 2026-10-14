"""
Typed view inputs for the Explore page: one fund's history on its session
calendar and the metadata every number on the page must carry (source,
coverage, retrieval, vintage, evidence status, limitations, staleness).

A FundHistory is a contract, not a computation. Construction validates it
and raises ValueError on a malformed record set, so a page never has to
guess what a row means. The records are `stock_agent.portfolio.descriptive`
total returns (`RETURN_COLUMNS`), one row per calendar session in a
contiguous run:

- `date`: UTC session dates (midnight), strictly increasing, sessions of
  `calendar` with none skipped;
- `close_observed`: whether the source has a close on that session;
- `status` `ok`: a finite `total_return` over `sessions_spanned` sessions
  (1 for one session; more for a labelled multi-session return across
  sessions without a close, which carry no value), and no reason;
- `status` `unavailable`: a NaN return, `sessions_spanned` 0 and a
  machine-readable `reason` (for example `missing_input` for a session
  without a close, `no_prior_close`, or `distribution_in_gap`). Nothing is
  filled, interpolated or set to zero.

Coverage is derived from the records (the first and last session with a
close), so it cannot disagree with them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

ROLES = ("equity", "bonds", "cash proxy")
OK = "ok"
UNAVAILABLE = "unavailable"
STATUSES = (OK, UNAVAILABLE)
EVIDENCE_STATUS = "descriptive"
RECORD_COLUMNS = ("date", "close_observed", "total_return", "sessions_spanned", "status", "reason")


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or value == ""


def validate_records(records: pd.DataFrame) -> None:
    """Raise ValueError unless `records` follows the FundHistory record contract."""

    if tuple(records.columns) != RECORD_COLUMNS:
        raise ValueError(f"record columns must be {RECORD_COLUMNS}, got {tuple(records.columns)}")
    dates = records["date"]
    if not isinstance(dates.dtype, pd.DatetimeTZDtype) or str(dates.dt.tz) != "UTC":
        raise ValueError("record dates must be timezone-aware UTC timestamps")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("record dates must be session dates at 00:00 UTC")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("record dates must be strictly increasing")
    if records["close_observed"].dtype != bool:
        raise ValueError("close_observed must be boolean")
    status = records["status"]
    if not status.isin(STATUSES).all():
        raise ValueError(f"record status must be one of {STATUSES}")
    returns = pd.to_numeric(records["total_return"], errors="raise").astype("float64")
    spans = records["sessions_spanned"].to_numpy()
    observed = records["close_observed"].to_numpy()
    ok = status.eq(OK).to_numpy()
    if not np.isfinite(returns.to_numpy()[ok]).all():
        raise ValueError("an ok record needs a finite total_return")
    if not returns[~ok].isna().all() or (spans[~ok] != 0).any():
        raise ValueError("an unavailable record has a NaN total_return and spans 0 sessions")
    if (ok & ~observed).any():
        raise ValueError("an ok record needs an observed close")
    if (spans[ok] < 1).any():
        raise ValueError("an ok record spans at least one session")
    reasons = records["reason"]
    if not reasons[ok].map(_blank).all():
        raise ValueError("an ok record carries no reason")
    if reasons[~ok].map(_blank).any():
        raise ValueError("an unavailable record needs a machine-readable reason")
    for position in np.flatnonzero(ok):
        inside = observed[max(0, position - spans[position] + 1) : position]
        if inside.any():
            raise ValueError("a multi-session return may only span sessions without a close")
        origin = position - spans[position]
        if origin >= 0 and not observed[origin]:
            raise ValueError(
                "a return, one-session or multi-session, starts at a session with a close"
            )


@dataclass(frozen=True, eq=False)
class FundHistory:
    """
    One fund's daily total-return records plus the labels the page shows.

    `source_label`, `retrieved_label` and `vintage_note` are display text
    from the data's producer. `limitations` are codes (for example
    `synthetic_data`); the page translates known codes and shows unknown ones
    as they are. `stale` marks data not refreshed by its stated end, with a
    `stale_reason` in words. Every fund history is descriptive evidence:
    `evidence_status` is a class constant, not a field.
    """

    evidence_status: ClassVar[str] = EVIDENCE_STATUS

    fund_id: str
    name: str
    role: str
    records: pd.DataFrame
    calendar: pd.DatetimeIndex
    source_label: str
    retrieved_label: str
    vintage_note: str
    limitations: tuple[str, ...]
    stale: bool = False
    stale_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.fund_id or not self.name:
            raise ValueError("a fund needs an id and a name")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if not isinstance(self.limitations, tuple):
            raise TypeError("limitations must be a tuple of codes")
        if self.stale != bool(self.stale_reason):
            raise ValueError("a stale fund needs a stale_reason, and only a stale fund has one")
        validate_records(self.records)
        calendar = pd.DatetimeIndex(self.calendar)
        if not calendar.is_monotonic_increasing or calendar.has_duplicates:
            raise ValueError("the calendar must be strictly increasing sessions")
        dates = pd.DatetimeIndex(self.records["date"])
        if len(dates):
            first, last = calendar.get_indexer([dates[0], dates[-1]])
            if first < 0 or last < 0 or not calendar[first : last + 1].equals(dates):
                raise ValueError("records must cover a contiguous run of calendar sessions")
        object.__setattr__(self, "calendar", calendar)
        # A private copy: the page and the metrics provider never share a mutable frame.
        object.__setattr__(self, "records", self.records.reset_index(drop=True).copy())

    @property
    def observed(self) -> pd.DataFrame:
        """The records with an observed close."""

        return self.records[self.records["close_observed"]]

    @property
    def available_sessions(self) -> pd.DatetimeIndex:
        """The sorted sessions with an observed close: the possible start dates."""

        return pd.DatetimeIndex(self.observed["date"])

    @property
    def coverage_start(self) -> pd.Timestamp | None:
        """First session with a close; None when there is none."""

        observed = self.observed
        return None if observed.empty else observed["date"].iloc[0]

    @property
    def coverage_end(self) -> pd.Timestamp | None:
        """Last session with a close; None when there is none."""

        observed = self.observed
        return None if observed.empty else observed["date"].iloc[-1]

    @property
    def year_end_sessions(self) -> pd.DatetimeIndex:
        """
        The last session of each calendar year the calendar covers to its end.
        A calendar that stops mid-year has no year end for that year: its last
        session counts only when no weekday is left in the year after it.
        """

        calendar = self.calendar
        if calendar.empty:
            return pd.DatetimeIndex([], tz="UTC")
        ends = pd.DatetimeIndex(calendar.to_series().groupby(calendar.year).max().to_numpy())
        last = calendar[-1]
        year_end = pd.Timestamp(year=last.year, month=12, day=31, tz="UTC")
        remaining = pd.bdate_range(last + pd.Timedelta(days=1), year_end, tz="UTC")
        return ends if remaining.empty else ends[ends.year != last.year]

    @property
    def synthetic(self) -> bool:
        return "synthetic_data" in self.limitations

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> FundHistory:
        """The same fund restricted to records dated in [start, end]; nothing later is kept."""

        dates = self.records["date"]
        return dataclasses.replace(self, records=self.records[dates.ge(start) & dates.le(end)])
