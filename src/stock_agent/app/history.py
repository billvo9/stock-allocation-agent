"""
Typed view inputs for the Explore page: one fund's daily history and the
metadata every number on the page must carry (source, coverage, retrieval,
vintage, evidence status, limitations, staleness).

A FundHistory is a contract, not a computation. Construction validates it
and raises ValueError on a malformed record set, so a page never has to
guess what a row means:

- `date` holds UTC session dates (midnight), strictly increasing;
- `status` is `ok` or `unavailable`;
- an `ok` row has a finite `daily_total_return` and no reason;
- an `unavailable` row has a NaN return and a machine-readable `reason`
  (for example `missing_input`). It is never filled, interpolated or set
  to zero.

Coverage is derived from the records (the first and last available
session), so it cannot disagree with them.
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
RECORD_COLUMNS = ("date", "daily_total_return", "status", "reason")


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
    status = records["status"]
    if not status.isin(STATUSES).all():
        raise ValueError(f"record status must be one of {STATUSES}")
    returns = pd.to_numeric(records["daily_total_return"], errors="raise").astype("float64")
    ok = status.eq(OK).to_numpy()
    if not np.isfinite(returns.to_numpy()[ok]).all():
        raise ValueError("an ok record needs a finite daily_total_return")
    if not returns[~ok].isna().all():
        raise ValueError("an unavailable record must have a NaN daily_total_return")
    reasons = records["reason"]
    if not reasons[ok].map(_blank).all():
        raise ValueError("an ok record carries no reason")
    if reasons[~ok].map(_blank).any():
        raise ValueError("an unavailable record needs a machine-readable reason")


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
        # A private copy: the page and the metrics provider never share a mutable frame.
        object.__setattr__(self, "records", self.records.reset_index(drop=True).copy())

    @property
    def available(self) -> pd.DataFrame:
        """The `ok` records only."""

        return self.records[self.records["status"].eq(OK)]

    @property
    def available_sessions(self) -> pd.DatetimeIndex:
        """The sorted sessions with an available return."""

        return pd.DatetimeIndex(self.available["date"])

    @property
    def coverage_start(self) -> pd.Timestamp | None:
        """First session with an available return; None when there is none."""

        available = self.available
        return None if available.empty else available["date"].iloc[0]

    @property
    def coverage_end(self) -> pd.Timestamp | None:
        """Last session with an available return; None when there is none."""

        available = self.available
        return None if available.empty else available["date"].iloc[-1]

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> FundHistory:
        """The same fund restricted to records dated in [start, end]; nothing later is kept."""

        dates = self.records["date"]
        return dataclasses.replace(self, records=self.records[dates.ge(start) & dates.le(end)])
