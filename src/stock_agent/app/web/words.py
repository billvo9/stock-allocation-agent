"""
The Lab app's words: number formats, dates and times in plain English.

Times are New York time, the Stock Lab's clock (the evening job, the close and
the open). " ET" is added only when this Mac's own zone is elsewhere, so a
reader in New York never sees a label they do not need. Percentages are
rounded for display only; the value itself is never changed.
"""

from __future__ import annotations

import math
from datetime import date, datetime, tzinfo
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
DASH = "—"
MINUS = "−"
EVENING_FROM = 17  # from 5 PM, a time today is "tonight" and one yesterday "last night"
RECENT_DAYS = 6  # within this many days, a time is named by its weekday

# Company names are not in the stored data; these are the watchlist's.
NAMES = {
    "AMD": "Advanced Micro Devices",
    "MRVL": "Marvell Technology",
    "MU": "Micron Technology",
    "NVDA": "NVIDIA",
}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def percent(p: float | None) -> str:
    """A probability as a whole percentage, for display: 0.4712 -> '47%'."""

    return f"{p:.0%}" if _finite(p) else DASH


def signed_percent(r: float | None) -> str:
    """A price move with its sign: '+4.88%', '−0.32%', '0.00%'."""

    if not _finite(r):
        return DASH
    sign = "+" if r > 0 else MINUS if r < 0 else ""
    return f"{sign}{abs(r):.2%}"


def money(value: float | None) -> str:
    return f"${value:,.2f}" if _finite(value) else DASH


def _day(value: date | str) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


def short_day(value: date | str) -> str:
    """'Wed, Oct 21'."""

    day = _day(value)
    return f"{day:%a, %b} {day.day}"


def weekday(value: date | str) -> str:
    """'Wed'."""

    return f"{_day(value):%a}"


def long_weekday(value: date | str) -> str:
    """'Wednesday'."""

    return f"{_day(value):%A}"


def clock(moment: datetime) -> str:
    """'8:31 PM'."""

    return f"{moment:%I:%M %p}".lstrip("0")


def when(stamp: str, *, now: datetime, local_zone: tzinfo) -> str:
    """
    A stored UTC stamp as a reader would say it, in New York time: 'last night,
    8:31 PM', 'tonight, 8:31 PM', 'this morning, 6:02 AM', 'Friday, 8:31 PM' or
    'Oct 9, 8:31 PM'; with ' ET' when this Mac's zone is not New York's.
    """

    moment = pd.Timestamp(stamp).tz_convert(NEW_YORK).to_pydatetime()
    today = now.astimezone(NEW_YORK).date()
    days = (today - moment.date()).days
    evening = moment.hour >= EVENING_FROM
    if days == 0:
        lead = "tonight" if evening else "this morning" if moment.hour < 12 else "today"
    elif days == 1:
        lead = "last night" if evening else "yesterday"
    elif 1 < days <= RECENT_DAYS:
        lead = long_weekday(moment.date())
    else:
        lead = f"{moment:%b} {moment.day}"
    text = f"{lead}, {clock(moment)}"
    return text + (" ET" if _elsewhere(now, local_zone) else "")


def _elsewhere(now: datetime, local_zone: tzinfo) -> bool:
    """Whether this Mac's clock reads differently from New York's at `now`."""

    return now.astimezone(local_zone).utcoffset() != now.astimezone(NEW_YORK).utcoffset()
