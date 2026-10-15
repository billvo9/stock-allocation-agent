"""
Rule-based NYSE full-day closures, and the session calendar built from them
for 2003-01-02 through 2027-12-31: the prospective path's calendar.

Why a second calendar
    `calendars/nyse.py` is the frozen research calendar: an explicit list for
    2003-2024 that refuses every date on or after the holdout start. This
    module generates closures from the exchange's holiday rules instead, so
    that coverage can extend past 2024 for the prospective store
    (`stock_agent.data.prospective`). A test requires the generated 2003-2024
    closures to equal `nyse.FULL_DAY_CLOSURES` exactly, and pins the frozen
    calendar's sessions byte for byte; `nyse.py` is unchanged and remains the
    research calendar. Research code must not import this module
    (`tests/test_prospective_boundaries.py`), because it serves dates inside
    the sealed holdout.

Holidays (NYSE Rule 7.2, as recalled; compare with NYSE's published holiday
list before relying on a year this code has not seen)
    New Year's Day (January 1); Martin Luther King Jr. Day (third Monday of
    January); Washington's Birthday (third Monday of February); Good Friday
    (two days before Western Easter, from the anonymous Gregorian computus);
    Memorial Day (last Monday of May); Juneteenth National Independence Day
    (June 19, from 2022); Independence Day (July 4); Labor Day (first Monday of
    September); Thanksgiving Day (fourth Thursday of November); Christmas Day
    (December 25).

Observance
    A holiday on a Sunday closes the following Monday. A holiday on a Saturday
    closes the preceding Friday, unless that Friday ends a monthly or yearly
    accounting period. So New Year's Day on a Saturday closes nothing: the NYSE
    traded on 2004-12-31, 2010-12-31 and 2021-12-31, and is expected to trade
    on 2027-12-31.

Unscheduled closures cannot be predicted
    National days of mourning, storms and emergencies are announced at short
    notice, so no rule produces them. `SPECIAL_CLOSURES` lists the ones known
    to this code: the five in the frozen list and 2025-01-09 (the national day
    of mourning for Jimmy Carter). A closure announced later is missing here
    until someone adds it with a source. Until then the prospective store
    reports that date as a missing session for every fund (never filled), and
    session counts that span it are one too high.

Out of scope
    Early closes (13:00 ET) are ordinary sessions: there are no open or close
    times here. The SIFMA bond calendar is not represented.

Fail closed
    Every function refuses a date outside `COVERAGE` with
    `nyse.CalendarRefused`. Unlike `nyse.py` it has no lockbox refusal: the
    prospective path enforces its own window (`data/prospective/window.py`).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.calendars.nyse import CalendarRefused, Coverage

CALENDAR_ID = "nyse_full_day_rules_2003_2027_v1"

COVERAGE = Coverage(
    first=pd.Timestamp("2003-01-02", tz="UTC").as_unit("us"),
    last=pd.Timestamp("2027-12-31", tz="UTC").as_unit("us"),
)
FIRST_YEAR, LAST_YEAR = 2003, 2027
JUNETEENTH_FIRST_YEAR = 2022

# Unscheduled full-day closures known to this code. The frozen list's five, then
# later ones, each with its occasion. Add an entry only with a primary source.
SPECIAL_CLOSURES: tuple[tuple[date, str], ...] = (
    *nyse.SPECIAL_CLOSURES,
    (date(2025, 1, 9), "national day of mourning for Jimmy Carter"),
)


def easter_sunday(year: int) -> date:
    """Western Easter Sunday: the anonymous Gregorian computus (Meeus/Jones/Butcher)."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + month // 12, month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed(day: date) -> date | None:
    """
    The weekday the NYSE closes for a holiday falling on `day`: Sunday ->
    Monday; Saturday -> Friday, unless that Friday is the month's last weekday
    (it ends a monthly, and possibly yearly, accounting period), in which case
    None (no closure).
    """

    if day.weekday() == 6:
        return day + timedelta(days=1)
    if day.weekday() == 5:
        friday = day - timedelta(days=1)
        ends_month = (friday + timedelta(days=3)).month != friday.month
        return None if ends_month else friday
    return day


def holiday_closures(year: int) -> dict[date, str]:
    """The rule-based full-day closures that fall in `year` (no special closures)."""

    fixed = [
        (date(year, 1, 1), "New Year's Day"),
        (date(year, 7, 4), "Independence Day"),
        (date(year, 12, 25), "Christmas Day"),
        (date(year + 1, 1, 1), "New Year's Day"),  # observed on December 31 only
    ]
    if year >= JUNETEENTH_FIRST_YEAR:
        fixed.append((date(year, 6, 19), "Juneteenth"))
    closures = {}
    for day, name in fixed:
        closed = observed(day)
        if closed is not None and closed.year == year:
            closures[closed] = name if closed == day else f"{name} (observed)"
    floating = {
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        easter_sunday(year) - timedelta(days=2): "Good Friday",
        _last_weekday(year, 5, 0): "Memorial Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving Day",
    }
    closures.update(floating)
    return dict(sorted(closures.items()))


def full_day_closures(first_year: int, last_year: int) -> tuple[date, ...]:
    """Every weekday closure in [first_year, last_year]: the rules plus `SPECIAL_CLOSURES`."""

    if not FIRST_YEAR <= first_year <= last_year <= LAST_YEAR:
        raise CalendarRefused(
            "outside_coverage", f"years {first_year}..{last_year} outside {FIRST_YEAR}..{LAST_YEAR}"
        )
    days = {day for year in range(first_year, last_year + 1) for day in holiday_closures(year)}
    days |= {day for day, _ in SPECIAL_CLOSURES if first_year <= day.year <= last_year}
    return tuple(sorted(days))


FULL_DAY_CLOSURES = full_day_closures(FIRST_YEAR, LAST_YEAR)


def session_date(value: object, *, label: str = "date") -> pd.Timestamp:
    """
    A date as 00:00 UTC (`Timestamp`, microsecond unit), refused outside
    `COVERAGE`. The same input rules as `nyse.session_date`: a `YYYY-MM-DD`
    string in ASCII digits, a `datetime.date`, or a naive or UTC timestamp at
    exactly midnight; a time of day or another timezone is refused.
    """

    if isinstance(value, str):
        if not nyse._DATE_PATTERN.fullmatch(value):
            raise CalendarRefused("invalid_date", f"{label} {value!r} is not YYYY-MM-DD")
        try:
            timestamp = pd.Timestamp(date.fromisoformat(value).isoformat())
        except ValueError as error:
            raise CalendarRefused("invalid_date", f"{label} {value!r} is not a date") from error
    elif isinstance(value, pd.Timestamp | datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, date):
        timestamp = pd.Timestamp(value.isoformat())
    else:
        raise CalendarRefused("invalid_date", f"{label} {value!r} is not a date")
    if timestamp is pd.NaT:
        raise CalendarRefused("invalid_date", f"{label} is missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    elif str(timestamp.tz) != "UTC":
        raise CalendarRefused("invalid_date", f"{label} {value!r} is not in UTC")
    if timestamp != timestamp.normalize():
        raise CalendarRefused("invalid_date", f"{label} {value!r} has a time of day")
    timestamp = timestamp.as_unit("us")
    if not COVERAGE.first <= timestamp <= COVERAGE.last:
        raise CalendarRefused(
            "outside_coverage",
            f"{label} {timestamp.date()} is outside {COVERAGE.first.date()}..{COVERAGE.last.date()}",
        )
    return timestamp


def _all_sessions() -> pd.DatetimeIndex:
    weekdays = pd.bdate_range(COVERAGE.first, COVERAGE.last, tz="UTC", name="session_date")
    closed = pd.DatetimeIndex([pd.Timestamp(day) for day in FULL_DAY_CLOSURES], tz="UTC")
    return weekdays[~weekdays.isin(closed)].as_unit("us")


_SESSIONS = _all_sessions()


def sessions(start: object, end: object) -> pd.DatetimeIndex:
    """Every session in [start, end], both ends inclusive and inside coverage."""

    first = session_date(start, label="start")
    last = session_date(end, label="end")
    if first > last:
        raise CalendarRefused("invalid_range", f"start {first.date()} is after end {last.date()}")
    return _SESSIONS[(_SESSIONS >= first) & (_SESSIONS <= last)]


def is_session(value: object) -> bool:
    """True when the date is a full NYSE session; refused outside coverage."""

    return session_date(value) in _SESSIONS


def session_offset(day: object, offset: int) -> pd.Timestamp:
    """
    The session `offset` sessions after the session `day` (before it when
    negative; `day` itself for 0). Refuses a `day` that is not a session and a
    result outside coverage.
    """

    if type(offset) is not int:
        raise CalendarRefused("invalid_offset", f"offset {offset!r} is not an int")
    session = session_date(day, label="day")
    position = _SESSIONS.searchsorted(session)
    if position == len(_SESSIONS) or _SESSIONS[position] != session:
        raise CalendarRefused("not_a_session", f"{session.date()} is not a session")
    target = position + offset
    if not 0 <= target < len(_SESSIONS):
        raise CalendarRefused(
            "outside_coverage", f"{offset} sessions from {session.date()} leaves the coverage"
        )
    return _SESSIONS[target]


__all__ = [
    "CALENDAR_ID",
    "COVERAGE",
    "FULL_DAY_CLOSURES",
    "SPECIAL_CLOSURES",
    "easter_sunday",
    "full_day_closures",
    "holiday_closures",
    "is_session",
    "observed",
    "session_date",
    "session_offset",
    "sessions",
]
