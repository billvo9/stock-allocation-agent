"""
A frozen, fail-closed NYSE full-day session calendar for 2003-01-02 through
2024-12-31.

What a session is
    A session date is a weekday on which the NYSE traded, stored as 00:00 UTC
    of that date (`datetime64[us, UTC]`), the convention used for session dates
    across the project. A session date says nothing about when its close
    became available: see `docs/research/f1_fund_data.md`.

Closures (the frozen list)
    `FULL_DAY_CLOSURES` is an explicit tuple of every weekday the NYSE was
    closed in coverage. It is data, not code: nothing here derives a holiday
    at run time. The tests rebuild it independently from the rules below (an
    Easter computus for Good Friday, nth-weekday rules, observance rules) for
    every year and require equality.

    Holidays: New Year's Day, Martin Luther King Jr. Day (third Monday of
    January), Washington's Birthday (third Monday of February), Good Friday,
    Memorial Day (last Monday of May), Juneteenth (June 19, from 2022 only),
    Independence Day, Labor Day (first Monday of September), Thanksgiving Day
    (fourth Thursday of November) and Christmas Day.

    Observance: a holiday on a Sunday closes the following Monday. A holiday
    on a Saturday closes the preceding Friday, unless that Friday ends a month
    or a year; so New Year's Day on a Saturday closes nothing, and the NYSE
    traded on 2004-12-31, 2010-12-31 and 2021-12-31.

    Special full-day closures (`SPECIAL_CLOSURES`): 2004-06-11 (Ronald Reagan),
    2007-01-02 (Gerald Ford), 2012-10-29 and 2012-10-30 (Hurricane Sandy) and
    2018-12-05 (George H. W. Bush).

Out of scope
    Early closes (13:00 ET, for example the day after Thanksgiving or
    Christmas Eve) are ordinary sessions here: this calendar has no open or
    close times, so it cannot serve an intraday cutoff. The SIFMA bond
    calendar (Columbus Day and Veterans Day closures; Good Friday only an
    early close) is a different calendar and is not represented.

Fail closed
    Every function refuses a date outside `COVERAGE`, and anything on or after
    `MODEL_HOLDOUT_START` (the lockbox, imported, never restated), with
    `CalendarRefused`. It never guesses beyond the frozen list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

CALENDAR_ID = "nyse_full_day_2003_2024_v1"

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")  # applied with fullmatch


class CalendarRefused(ValueError):
    """A calendar query was refused; `reason` is machine-readable."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Coverage:
    """The covered date range, both ends inclusive, as 00:00 UTC timestamps."""

    first: pd.Timestamp
    last: pd.Timestamp


# The frozen list below was checked for these dates only. The last date is the
# list's own extent; the lockbox is enforced separately with MODEL_HOLDOUT_START.
COVERAGE = Coverage(
    first=pd.Timestamp("2003-01-02", tz="UTC").as_unit("us"),
    last=pd.Timestamp("2024-12-31", tz="UTC").as_unit("us"),
)

SPECIAL_CLOSURES: tuple[tuple[date, str], ...] = (
    (date(2004, 6, 11), "national day of mourning for Ronald Reagan"),
    (date(2007, 1, 2), "national day of mourning for Gerald Ford"),
    (date(2012, 10, 29), "Hurricane Sandy"),
    (date(2012, 10, 30), "Hurricane Sandy"),
    (date(2018, 12, 5), "national day of mourning for George H. W. Bush"),
)

# Every weekday in coverage on which the NYSE was closed for the full day.
FULL_DAY_CLOSURES: tuple[date, ...] = (
    date(2003, 1, 1),  # Wed, New Year's Day
    date(2003, 1, 20),  # Mon, Martin Luther King Jr. Day
    date(2003, 2, 17),  # Mon, Washington's Birthday
    date(2003, 4, 18),  # Fri, Good Friday
    date(2003, 5, 26),  # Mon, Memorial Day
    date(2003, 7, 4),  # Fri, Independence Day
    date(2003, 9, 1),  # Mon, Labor Day
    date(2003, 11, 27),  # Thu, Thanksgiving Day
    date(2003, 12, 25),  # Thu, Christmas Day
    date(2004, 1, 1),  # Thu, New Year's Day
    date(2004, 1, 19),  # Mon, Martin Luther King Jr. Day
    date(2004, 2, 16),  # Mon, Washington's Birthday
    date(2004, 4, 9),  # Fri, Good Friday
    date(2004, 5, 31),  # Mon, Memorial Day
    date(2004, 6, 11),  # Fri, Ronald Reagan national day of mourning
    date(2004, 7, 5),  # Mon, Independence Day (observed)
    date(2004, 9, 6),  # Mon, Labor Day
    date(2004, 11, 25),  # Thu, Thanksgiving Day
    date(2004, 12, 24),  # Fri, Christmas Day (observed)
    date(2005, 1, 17),  # Mon, Martin Luther King Jr. Day
    date(2005, 2, 21),  # Mon, Washington's Birthday
    date(2005, 3, 25),  # Fri, Good Friday
    date(2005, 5, 30),  # Mon, Memorial Day
    date(2005, 7, 4),  # Mon, Independence Day
    date(2005, 9, 5),  # Mon, Labor Day
    date(2005, 11, 24),  # Thu, Thanksgiving Day
    date(2005, 12, 26),  # Mon, Christmas Day (observed)
    date(2006, 1, 2),  # Mon, New Year's Day (observed)
    date(2006, 1, 16),  # Mon, Martin Luther King Jr. Day
    date(2006, 2, 20),  # Mon, Washington's Birthday
    date(2006, 4, 14),  # Fri, Good Friday
    date(2006, 5, 29),  # Mon, Memorial Day
    date(2006, 7, 4),  # Tue, Independence Day
    date(2006, 9, 4),  # Mon, Labor Day
    date(2006, 11, 23),  # Thu, Thanksgiving Day
    date(2006, 12, 25),  # Mon, Christmas Day
    date(2007, 1, 1),  # Mon, New Year's Day
    date(2007, 1, 2),  # Tue, Gerald Ford national day of mourning
    date(2007, 1, 15),  # Mon, Martin Luther King Jr. Day
    date(2007, 2, 19),  # Mon, Washington's Birthday
    date(2007, 4, 6),  # Fri, Good Friday
    date(2007, 5, 28),  # Mon, Memorial Day
    date(2007, 7, 4),  # Wed, Independence Day
    date(2007, 9, 3),  # Mon, Labor Day
    date(2007, 11, 22),  # Thu, Thanksgiving Day
    date(2007, 12, 25),  # Tue, Christmas Day
    date(2008, 1, 1),  # Tue, New Year's Day
    date(2008, 1, 21),  # Mon, Martin Luther King Jr. Day
    date(2008, 2, 18),  # Mon, Washington's Birthday
    date(2008, 3, 21),  # Fri, Good Friday
    date(2008, 5, 26),  # Mon, Memorial Day
    date(2008, 7, 4),  # Fri, Independence Day
    date(2008, 9, 1),  # Mon, Labor Day
    date(2008, 11, 27),  # Thu, Thanksgiving Day
    date(2008, 12, 25),  # Thu, Christmas Day
    date(2009, 1, 1),  # Thu, New Year's Day
    date(2009, 1, 19),  # Mon, Martin Luther King Jr. Day
    date(2009, 2, 16),  # Mon, Washington's Birthday
    date(2009, 4, 10),  # Fri, Good Friday
    date(2009, 5, 25),  # Mon, Memorial Day
    date(2009, 7, 3),  # Fri, Independence Day (observed)
    date(2009, 9, 7),  # Mon, Labor Day
    date(2009, 11, 26),  # Thu, Thanksgiving Day
    date(2009, 12, 25),  # Fri, Christmas Day
    date(2010, 1, 1),  # Fri, New Year's Day
    date(2010, 1, 18),  # Mon, Martin Luther King Jr. Day
    date(2010, 2, 15),  # Mon, Washington's Birthday
    date(2010, 4, 2),  # Fri, Good Friday
    date(2010, 5, 31),  # Mon, Memorial Day
    date(2010, 7, 5),  # Mon, Independence Day (observed)
    date(2010, 9, 6),  # Mon, Labor Day
    date(2010, 11, 25),  # Thu, Thanksgiving Day
    date(2010, 12, 24),  # Fri, Christmas Day (observed)
    date(2011, 1, 17),  # Mon, Martin Luther King Jr. Day
    date(2011, 2, 21),  # Mon, Washington's Birthday
    date(2011, 4, 22),  # Fri, Good Friday
    date(2011, 5, 30),  # Mon, Memorial Day
    date(2011, 7, 4),  # Mon, Independence Day
    date(2011, 9, 5),  # Mon, Labor Day
    date(2011, 11, 24),  # Thu, Thanksgiving Day
    date(2011, 12, 26),  # Mon, Christmas Day (observed)
    date(2012, 1, 2),  # Mon, New Year's Day (observed)
    date(2012, 1, 16),  # Mon, Martin Luther King Jr. Day
    date(2012, 2, 20),  # Mon, Washington's Birthday
    date(2012, 4, 6),  # Fri, Good Friday
    date(2012, 5, 28),  # Mon, Memorial Day
    date(2012, 7, 4),  # Wed, Independence Day
    date(2012, 9, 3),  # Mon, Labor Day
    date(2012, 10, 29),  # Mon, Hurricane Sandy
    date(2012, 10, 30),  # Tue, Hurricane Sandy
    date(2012, 11, 22),  # Thu, Thanksgiving Day
    date(2012, 12, 25),  # Tue, Christmas Day
    date(2013, 1, 1),  # Tue, New Year's Day
    date(2013, 1, 21),  # Mon, Martin Luther King Jr. Day
    date(2013, 2, 18),  # Mon, Washington's Birthday
    date(2013, 3, 29),  # Fri, Good Friday
    date(2013, 5, 27),  # Mon, Memorial Day
    date(2013, 7, 4),  # Thu, Independence Day
    date(2013, 9, 2),  # Mon, Labor Day
    date(2013, 11, 28),  # Thu, Thanksgiving Day
    date(2013, 12, 25),  # Wed, Christmas Day
    date(2014, 1, 1),  # Wed, New Year's Day
    date(2014, 1, 20),  # Mon, Martin Luther King Jr. Day
    date(2014, 2, 17),  # Mon, Washington's Birthday
    date(2014, 4, 18),  # Fri, Good Friday
    date(2014, 5, 26),  # Mon, Memorial Day
    date(2014, 7, 4),  # Fri, Independence Day
    date(2014, 9, 1),  # Mon, Labor Day
    date(2014, 11, 27),  # Thu, Thanksgiving Day
    date(2014, 12, 25),  # Thu, Christmas Day
    date(2015, 1, 1),  # Thu, New Year's Day
    date(2015, 1, 19),  # Mon, Martin Luther King Jr. Day
    date(2015, 2, 16),  # Mon, Washington's Birthday
    date(2015, 4, 3),  # Fri, Good Friday
    date(2015, 5, 25),  # Mon, Memorial Day
    date(2015, 7, 3),  # Fri, Independence Day (observed)
    date(2015, 9, 7),  # Mon, Labor Day
    date(2015, 11, 26),  # Thu, Thanksgiving Day
    date(2015, 12, 25),  # Fri, Christmas Day
    date(2016, 1, 1),  # Fri, New Year's Day
    date(2016, 1, 18),  # Mon, Martin Luther King Jr. Day
    date(2016, 2, 15),  # Mon, Washington's Birthday
    date(2016, 3, 25),  # Fri, Good Friday
    date(2016, 5, 30),  # Mon, Memorial Day
    date(2016, 7, 4),  # Mon, Independence Day
    date(2016, 9, 5),  # Mon, Labor Day
    date(2016, 11, 24),  # Thu, Thanksgiving Day
    date(2016, 12, 26),  # Mon, Christmas Day (observed)
    date(2017, 1, 2),  # Mon, New Year's Day (observed)
    date(2017, 1, 16),  # Mon, Martin Luther King Jr. Day
    date(2017, 2, 20),  # Mon, Washington's Birthday
    date(2017, 4, 14),  # Fri, Good Friday
    date(2017, 5, 29),  # Mon, Memorial Day
    date(2017, 7, 4),  # Tue, Independence Day
    date(2017, 9, 4),  # Mon, Labor Day
    date(2017, 11, 23),  # Thu, Thanksgiving Day
    date(2017, 12, 25),  # Mon, Christmas Day
    date(2018, 1, 1),  # Mon, New Year's Day
    date(2018, 1, 15),  # Mon, Martin Luther King Jr. Day
    date(2018, 2, 19),  # Mon, Washington's Birthday
    date(2018, 3, 30),  # Fri, Good Friday
    date(2018, 5, 28),  # Mon, Memorial Day
    date(2018, 7, 4),  # Wed, Independence Day
    date(2018, 9, 3),  # Mon, Labor Day
    date(2018, 11, 22),  # Thu, Thanksgiving Day
    date(2018, 12, 5),  # Wed, George H. W. Bush national day of mourning
    date(2018, 12, 25),  # Tue, Christmas Day
    date(2019, 1, 1),  # Tue, New Year's Day
    date(2019, 1, 21),  # Mon, Martin Luther King Jr. Day
    date(2019, 2, 18),  # Mon, Washington's Birthday
    date(2019, 4, 19),  # Fri, Good Friday
    date(2019, 5, 27),  # Mon, Memorial Day
    date(2019, 7, 4),  # Thu, Independence Day
    date(2019, 9, 2),  # Mon, Labor Day
    date(2019, 11, 28),  # Thu, Thanksgiving Day
    date(2019, 12, 25),  # Wed, Christmas Day
    date(2020, 1, 1),  # Wed, New Year's Day
    date(2020, 1, 20),  # Mon, Martin Luther King Jr. Day
    date(2020, 2, 17),  # Mon, Washington's Birthday
    date(2020, 4, 10),  # Fri, Good Friday
    date(2020, 5, 25),  # Mon, Memorial Day
    date(2020, 7, 3),  # Fri, Independence Day (observed)
    date(2020, 9, 7),  # Mon, Labor Day
    date(2020, 11, 26),  # Thu, Thanksgiving Day
    date(2020, 12, 25),  # Fri, Christmas Day
    date(2021, 1, 1),  # Fri, New Year's Day
    date(2021, 1, 18),  # Mon, Martin Luther King Jr. Day
    date(2021, 2, 15),  # Mon, Washington's Birthday
    date(2021, 4, 2),  # Fri, Good Friday
    date(2021, 5, 31),  # Mon, Memorial Day
    date(2021, 7, 5),  # Mon, Independence Day (observed)
    date(2021, 9, 6),  # Mon, Labor Day
    date(2021, 11, 25),  # Thu, Thanksgiving Day
    date(2021, 12, 24),  # Fri, Christmas Day (observed)
    date(2022, 1, 17),  # Mon, Martin Luther King Jr. Day
    date(2022, 2, 21),  # Mon, Washington's Birthday
    date(2022, 4, 15),  # Fri, Good Friday
    date(2022, 5, 30),  # Mon, Memorial Day
    date(2022, 6, 20),  # Mon, Juneteenth (observed)
    date(2022, 7, 4),  # Mon, Independence Day
    date(2022, 9, 5),  # Mon, Labor Day
    date(2022, 11, 24),  # Thu, Thanksgiving Day
    date(2022, 12, 26),  # Mon, Christmas Day (observed)
    date(2023, 1, 2),  # Mon, New Year's Day (observed)
    date(2023, 1, 16),  # Mon, Martin Luther King Jr. Day
    date(2023, 2, 20),  # Mon, Washington's Birthday
    date(2023, 4, 7),  # Fri, Good Friday
    date(2023, 5, 29),  # Mon, Memorial Day
    date(2023, 6, 19),  # Mon, Juneteenth
    date(2023, 7, 4),  # Tue, Independence Day
    date(2023, 9, 4),  # Mon, Labor Day
    date(2023, 11, 23),  # Thu, Thanksgiving Day
    date(2023, 12, 25),  # Mon, Christmas Day
    date(2024, 1, 1),  # Mon, New Year's Day
    date(2024, 1, 15),  # Mon, Martin Luther King Jr. Day
    date(2024, 2, 19),  # Mon, Washington's Birthday
    date(2024, 3, 29),  # Fri, Good Friday
    date(2024, 5, 27),  # Mon, Memorial Day
    date(2024, 6, 19),  # Wed, Juneteenth
    date(2024, 7, 4),  # Thu, Independence Day
    date(2024, 9, 2),  # Mon, Labor Day
    date(2024, 11, 28),  # Thu, Thanksgiving Day
    date(2024, 12, 25),  # Wed, Christmas Day
)

_CLOSURES = frozenset(FULL_DAY_CLOSURES)


def _refuse_outside(label: str, value: pd.Timestamp) -> pd.Timestamp:
    if value >= MODEL_HOLDOUT_START:
        raise CalendarRefused(
            "lockbox_violation", f"{label} {value.date()} is on or after the holdout start"
        )
    if not COVERAGE.first <= value <= COVERAGE.last:
        raise CalendarRefused(
            "outside_coverage",
            f"{label} {value.date()} is outside {COVERAGE.first.date()}..{COVERAGE.last.date()}",
        )
    return value


def session_date(value: object, *, label: str = "date") -> pd.Timestamp:
    """
    A date as 00:00 UTC (`Timestamp`, microsecond unit), refused outside
    coverage.

    Accepts a `YYYY-MM-DD` string, a `datetime.date`, or a naive or UTC
    timestamp at exactly midnight. A time of day or another timezone is
    refused rather than converted, because conversion can move the date.
    """

    if isinstance(value, str):
        if not _DATE_PATTERN.fullmatch(value):
            raise CalendarRefused("invalid_date", f"{label} {value!r} is not YYYY-MM-DD")
        timestamp = pd.Timestamp(value)
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
    return _refuse_outside(label, timestamp.as_unit("us"))


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

    day = session_date(value)
    return day.dayofweek < 5 and day.date() not in _CLOSURES


__all__ = [
    "CALENDAR_ID",
    "COVERAGE",
    "FULL_DAY_CLOSURES",
    "SPECIAL_CLOSURES",
    "CalendarRefused",
    "Coverage",
    "is_session",
    "session_date",
    "sessions",
]
