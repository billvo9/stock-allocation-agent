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


# --- Home: the status line ---

RECORD_UNREADABLE = "The forecast record can't be read."
LEDGER_FAILED = "The forecast record didn't pass its check."
RECORD_FAILED = "A forecast record didn't pass its check."
NOT_STARTED = "Outlooks haven't started yet."
NO_RUN = "No update has run yet."
NO_SUCCESS = "No update has finished yet."
RUN_PROBLEM = {
    "failed": "The last update didn't finish ({when}).",
    "partial": "The last update was only partly done ({when}).",
    "skipped": "The last update recorded nothing ({when}).",
}
RUN_PROBLEM_OTHER = "The last update ended as {status} ({when})."


def updated(when_words: str) -> str:
    return f"Updated {when_words}"


def evening_name(prior: date, now: datetime) -> str:
    """Whose update recorded the outlooks for the session after `prior`: 'Last night's'."""

    days = (now.astimezone(NEW_YORK).date() - prior).days
    if days == 0:
        return "Tonight's"
    if days == 1:
        return "Last night's"
    return f"{long_weekday(prior)}'s"


def missed(evening: str, *, ran: bool, missing: int, total: int) -> str:
    if not ran:
        return f"{evening} update didn't run."
    return f"{evening} update didn't record {missing} of {total} stocks."


def withheld(count: int) -> str:
    return f"{count} outlook was withheld." if count == 1 else f"{count} outlooks were withheld."


# --- Home: the outlook chip and its sheet ---

PENDING_CHIP = "{day} outlook after tonight's update"
STALE_CHIP = "{day} outlook not recorded"
NONE_CHIP = "No outlook yet"
REFUSED_CHIP = "No outlook: the model gave none"
UNVERIFIED_CHIP = "Outlook can't be verified"
WITHHELD_CHIP = "Outlook withheld"
WHOLE_NUMBERS = "Shown as whole numbers, so they may not add up to exactly 100%."
EXPERIMENTAL = "Experimental: not yet proven, and not advice."

NO_PRICES = {
    "no_store": "No prices stored yet.",
    "no_data": "No prices stored for {ticker} yet.",
}
PRICES_UNREADABLE = "Prices can't be read right now."
NO_CLOSE = "No closing price stored yet."

FOOTER = (
    "Outlooks are experimental chances from a simple model, recorded each weekday evening. "
    "They are not advice, and nothing here can be bought or sold."
)
EXPLAIN_TITLE = "What do the numbers mean?"
DEMO = "Synthetic demo: made-up stocks and numbers, to try the app safely."


def prices_note(session: str) -> str:
    return f"Closing prices from {short_day(session)}."


def explain(band: str | None) -> list[str]:
    """The footer's explanation; `band` is the recorded band in words ('0.5%'), if any."""

    edge = band or "a small band"
    return [
        (
            f"↑ is the recorded chance that the stock closes the next trading day more than "
            f"{edge} higher. ↓ is the chance it closes more than {edge} lower. The rest is the "
            f"chance it stays within {edge} (Flat)."
        ),
        (
            "They come from a deliberately simple model, frozen so it can't change, that mostly "
            "reflects how often each stock went up or down in the past. It is a baseline for "
            "better ideas to beat."
        ),
        (
            "Each outlook is made after the close from closing prices, so it knows nothing about "
            "later news."
        ),
        "Green and red only ever mean a price went up or down. Outlooks are never green or red.",
    ]
