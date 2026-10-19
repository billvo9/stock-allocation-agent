"""
Rule-based NYSE full-day closures, and the session calendars built from them:
the prospective path's calendars.

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

Calendar versions (each resolvable by its id with `calendar`)
    nyse_full_day_rules_2003_2027_v1   2003-01-02 to 2027-12-31 (`CALENDAR_ID_2027`)
    nyse_full_day_rules_2003_2030_v1   2003-01-02 to 2030-12-31 (`CALENDAR_ID_2030`)
    Both come from the same rules and the same special closures, so their
    sessions are identical through 2027-12-31 (a test). An id names one session
    list: a change to any session, or to the coverage, is a new id.

    The module-level names (`CALENDAR_ID`, `COVERAGE`, `FIRST_YEAR`,
    `LAST_YEAR`, `FULL_DAY_CLOSURES`, `session_date`, `sessions`, `is_session`,
    `session_offset`, `full_day_closures`) are the calendar every consumer
    uses and records: the 2030 version, since the owner approved the switch on
    2026-10-18 (`docs/research/prospective_data_path.md`, section 1). The
    prospective store and the Stock Lab inference store record and require
    this id, so data recorded under the 2027 id is refused; none existed when
    the switch was made. The 2027 version stays resolvable by its id.

Holidays (NYSE Rule 7.2)
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
    traded on 2004-12-31, 2010-12-31 and 2021-12-31, and NYSE's published
    schedule says no New Year's Day holiday is observed for Saturday
    2028-01-01, so 2027-12-31 trades.

Where each year's closures come from (`YEAR_SOURCES`, `year_source`)
    frozen_research_list  2003-2024: equal to the frozen list in `nyse.py`
                          (whose special closures came from the F1 brief and
                          were not re-checked against an external source)
    verified_published    2026-2028: equal to NYSE's published "Holidays &
                          Trading Hours" page (`NYSE_HOLIDAYS_URL`), retrieved
                          2026-10-18; the test embeds that list
    rule_generated        2025, 2029 and 2030: expected sessions from the rules
                          (2025 adds the special closure 2025-01-09), not
                          checked against a published schedule
    "Verified" describes the holiday schedule, not the sessions: a year's
    published schedule equalled the rules as of its `retrieved_on` date. A
    session after that date remains an expected session even in a verified
    year, because an unscheduled closure can still occur; a session in a
    rule-generated year is expected and its schedule unverified. Reports should
    say so. `check_published` compares a published list with the rules, so a
    year can be verified when NYSE publishes it (a new verified year is a code
    change: it edits `YEAR_SOURCES` and the test's embedded published lists).

Unscheduled closures cannot be predicted
    National days of mourning, storms and emergencies are announced at short
    notice, so no rule produces them. `SPECIAL_CLOSURES` lists the ones known
    to this code: the five in the frozen list and 2025-01-09 (the national day
    of mourning for Jimmy Carter). A closure announced later is missing here
    until someone adds it with a source. Until then the prospective store
    reports that date as a missing session for every fund (never filled), and
    session counts that span it are one too high. The proposed amendment rule,
    ETF-R2, is in `docs/research/prospective_data_path.md`, section 1. It is a
    design only, pending the owner: nothing here implements it.

Out of scope
    Early closes (13:00 ET) are ordinary sessions: there are no open or close
    times here. The SIFMA bond calendar is not represented.

Fail closed
    Every function refuses a date outside its calendar's coverage with
    `nyse.CalendarRefused`, and `calendar` refuses an unknown id
    (`unknown_calendar`). Unlike `nyse.py` there is no lockbox refusal: the
    prospective path enforces its own window (`data/prospective/window.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.calendars.nyse import CalendarRefused, Coverage

CALENDAR_ID_2027 = "nyse_full_day_rules_2003_2027_v1"
CALENDAR_ID_2030 = "nyse_full_day_rules_2003_2030_v1"
# Every known version: its id and the last year it covers. An id names one session list,
# so a version is constructed only from a pair listed here.
VERSION_LAST_YEARS: Mapping[str, int] = MappingProxyType(
    {CALENDAR_ID_2027: 2027, CALENDAR_ID_2030: 2030}
)
# The calendar every consumer uses and records (see the module docstring before changing it).
CALENDAR_ID = CALENDAR_ID_2030

FIRST_YEAR = 2003
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


def _closures(first_year: int, last_year: int) -> tuple[date, ...]:
    days = {day for year in range(first_year, last_year + 1) for day in holiday_closures(year)}
    days |= {day for day, _ in SPECIAL_CLOSURES if first_year <= day.year <= last_year}
    return tuple(sorted(days))


def _as_date(value: object, label: str) -> pd.Timestamp:
    """
    A date as 00:00 UTC (`Timestamp`, microsecond unit). The same input rules
    as `nyse.session_date`: a `YYYY-MM-DD` string in ASCII digits, a
    `datetime.date`, or a naive or UTC timestamp at exactly midnight; a time of
    day or another timezone is refused.
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
    return timestamp.as_unit("us")


@dataclass(frozen=True)
class RulesCalendarVersion:
    """
    One rules calendar: its id and the years it covers (2003-01-02 through
    December 31 of `last_year`), with its full-day closures and sessions
    derived once from the rules and `SPECIAL_CLOSURES`. Only a pair listed in
    `VERSION_LAST_YEARS` can be constructed (`unknown_calendar` otherwise), so
    no id can name another coverage; resolve a version with `calendar`.
    """

    calendar_id: str
    last_year: int
    coverage: Coverage = field(init=False)
    closures: tuple[date, ...] = field(init=False, repr=False)
    _sessions: pd.DatetimeIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        known = (
            VERSION_LAST_YEARS.get(self.calendar_id) if isinstance(self.calendar_id, str) else None
        )
        if known is None or type(self.last_year) is not int or self.last_year != known:
            raise CalendarRefused(
                "unknown_calendar",
                f"({self.calendar_id!r}, {self.last_year!r}) is not a known calendar version",
            )
        coverage = Coverage(
            first=pd.Timestamp(f"{FIRST_YEAR}-01-02", tz="UTC").as_unit("us"),
            last=pd.Timestamp(f"{self.last_year}-12-31", tz="UTC").as_unit("us"),
        )
        closures = _closures(FIRST_YEAR, self.last_year)
        weekdays = pd.bdate_range(coverage.first, coverage.last, tz="UTC", name="session_date")
        closed = pd.DatetimeIndex([pd.Timestamp(day) for day in closures], tz="UTC")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "closures", closures)
        object.__setattr__(self, "_sessions", weekdays[~weekdays.isin(closed)].as_unit("us"))

    def full_day_closures(self, first_year: int, last_year: int) -> tuple[date, ...]:
        """Every weekday closure in [first_year, last_year]: the rules plus `SPECIAL_CLOSURES`."""

        if not FIRST_YEAR <= first_year <= last_year <= self.last_year:
            raise CalendarRefused(
                "outside_coverage",
                f"years {first_year}..{last_year} outside {FIRST_YEAR}..{self.last_year} "
                f"({self.calendar_id})",
            )
        return tuple(day for day in self.closures if first_year <= day.year <= last_year)

    def session_date(self, value: object, *, label: str = "date") -> pd.Timestamp:
        """A date as 00:00 UTC (`Timestamp`, microsecond unit), refused outside coverage."""

        timestamp = _as_date(value, label)
        if not self.coverage.first <= timestamp <= self.coverage.last:
            raise CalendarRefused(
                "outside_coverage",
                f"{label} {timestamp.date()} is outside "
                f"{self.coverage.first.date()}..{self.coverage.last.date()}",
            )
        return timestamp

    def sessions(self, start: object, end: object) -> pd.DatetimeIndex:
        """Every session in [start, end], both ends inclusive and inside coverage."""

        first = self.session_date(start, label="start")
        last = self.session_date(end, label="end")
        if first > last:
            raise CalendarRefused(
                "invalid_range", f"start {first.date()} is after end {last.date()}"
            )
        return self._sessions[(self._sessions >= first) & (self._sessions <= last)]

    def is_session(self, value: object) -> bool:
        """True when the date is a full NYSE session; refused outside coverage."""

        return self.session_date(value) in self._sessions

    def session_offset(self, day: object, offset: int) -> pd.Timestamp:
        """
        The session `offset` sessions after the session `day` (before it when
        negative; `day` itself for 0). Refuses a `day` that is not a session
        and a result outside coverage.
        """

        if type(offset) is not int:
            raise CalendarRefused("invalid_offset", f"offset {offset!r} is not an int")
        session = self.session_date(day, label="day")
        position = self._sessions.searchsorted(session)
        if position == len(self._sessions) or self._sessions[position] != session:
            raise CalendarRefused("not_a_session", f"{session.date()} is not a session")
        target = position + offset
        if not 0 <= target < len(self._sessions):
            raise CalendarRefused(
                "outside_coverage", f"{offset} sessions from {session.date()} leaves the coverage"
            )
        return self._sessions[target]


CALENDARS: Mapping[str, RulesCalendarVersion] = MappingProxyType(
    {
        calendar_id: RulesCalendarVersion(calendar_id, last_year)
        for calendar_id, last_year in VERSION_LAST_YEARS.items()
    }
)


def calendar(calendar_id: str) -> RulesCalendarVersion:
    """The rules calendar a recorded id names; refuses an unknown id."""

    if not isinstance(calendar_id, str) or calendar_id not in CALENDARS:
        raise CalendarRefused(
            "unknown_calendar", f"{calendar_id!r} is not one of {sorted(CALENDARS)}"
        )
    return CALENDARS[calendar_id]


# --- where each year's closures come from ------------------------------------------------

NYSE_HOLIDAYS_URL = "https://www.nyse.com/markets/hours-calendars"
NYSE_HOLIDAYS_RETRIEVED_ON = date(2026, 10, 18)


class YearStatus(StrEnum):
    """How a year's closures were established."""

    FROZEN_RESEARCH_LIST = "frozen_research_list"
    VERIFIED_PUBLISHED = "verified_published"
    RULE_GENERATED = "rule_generated"


SCHEDULE_VERIFIED_NOTE = (
    "the holiday schedule equalled the published page as of retrieved_on; sessions after "
    "that date remain expected, because an unscheduled closure can still occur"
)
SCHEDULE_UNVERIFIED_NOTE = (
    "the holiday schedule was not checked against a published page; its sessions are expected"
)


@dataclass(frozen=True)
class YearSource:
    """
    Where one year's closures come from. `verified` means the year's holiday
    schedule was checked against a published exchange schedule, recorded with
    its URL and retrieval date, and equalled it as of `retrieved_on`. It never
    means a session is verified: a later unscheduled closure can still remove
    one. A verified source must carry both `source_url` and `retrieved_on`, and
    any other source neither (refused otherwise).
    """

    year: int
    status: YearStatus
    detail: str
    source_url: str | None = None
    retrieved_on: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, YearStatus):
            raise CalendarRefused("invalid_year_source", f"status {self.status!r}")
        cited = (
            isinstance(self.source_url, str)
            and self.source_url.startswith("https://")
            and type(self.retrieved_on) is date
        )
        uncited = self.source_url is None and self.retrieved_on is None
        if not (cited if self.verified else uncited):
            raise CalendarRefused(
                "invalid_year_source",
                f"{self.year}: a {self.status.value} source must "
                f"{'carry' if self.verified else 'carry neither'} an https source_url and a "
                "retrieved_on date",
            )

    @property
    def verified(self) -> bool:
        """True when the holiday schedule was verified as of `retrieved_on` (not the sessions)."""

        return self.status is YearStatus.VERIFIED_PUBLISHED

    def to_json_dict(self) -> dict[str, object]:
        return {
            "year": self.year,
            "status": self.status.value,
            "schedule_verified": self.verified,
            "detail": self.detail,
            "source_url": self.source_url,
            "retrieved_on": None if self.retrieved_on is None else self.retrieved_on.isoformat(),
            "note": SCHEDULE_VERIFIED_NOTE if self.verified else SCHEDULE_UNVERIFIED_NOTE,
        }


def _year_sources() -> dict[int, YearSource]:
    sources = {
        year: YearSource(
            year,
            YearStatus.FROZEN_RESEARCH_LIST,
            "equal to the frozen research list in calendars/nyse.py",
        )
        for year in range(FIRST_YEAR, 2025)
    }
    sources[2025] = YearSource(
        2025,
        YearStatus.RULE_GENERATED,
        "the holiday rules plus the special closure 2025-01-09; NYSE's published 2025 "
        "schedule was not retrieved",
    )
    for year in (2026, 2027, 2028):
        sources[year] = YearSource(
            year,
            YearStatus.VERIFIED_PUBLISHED,
            "equal to NYSE's published Holidays & Trading Hours page",
            source_url=NYSE_HOLIDAYS_URL,
            retrieved_on=NYSE_HOLIDAYS_RETRIEVED_ON,
        )
    for year in (2029, 2030):
        sources[year] = YearSource(
            year,
            YearStatus.RULE_GENERATED,
            "the holiday rules only; not yet published by NYSE when retrieved on 2026-10-18",
        )
    return sources


YEAR_SOURCES: Mapping[int, YearSource] = MappingProxyType(_year_sources())


def year_source(year: int) -> YearSource:
    """
    Where `year`'s closures come from. Refuses a year that is not an int
    (`invalid_year`; a bool is not a year) and a year outside every calendar's
    years (`outside_coverage`).
    """

    if type(year) is not int:
        raise CalendarRefused("invalid_year", f"year {year!r} is not an int")
    if year not in YEAR_SOURCES:
        raise CalendarRefused(
            "outside_coverage", f"year {year} outside {min(YEAR_SOURCES)}..{max(YEAR_SOURCES)}"
        )
    return YEAR_SOURCES[year]


@dataclass(frozen=True)
class ClosureCheck:
    """
    The rules' closures for one year compared with a published list.
    `unexpected`: closed by the rules, absent from the published list.
    `missing`: in the published list, not closed by the rules.
    """

    calendar_id: str
    year: int
    unexpected: tuple[date, ...]
    missing: tuple[date, ...]

    @property
    def matches(self) -> bool:
        return not self.unexpected and not self.missing


def check_published(
    year: int, published: Iterable[date], *, calendar_id: str = CALENDAR_ID_2030
) -> ClosureCheck:
    """
    Compare a published list of full-day closures for `year` with the rules
    calendar `calendar_id`. Refuses a published date outside `year`, on a
    weekend, or not a `datetime.date` (`invalid_date`), and a date listed twice
    (`duplicate_date`: a copying error, never silently merged); it reports
    differences, never repairs.
    """

    days = tuple(published)
    for day in days:
        if type(day) is not date:
            raise CalendarRefused("invalid_date", f"published closure {day!r} is not a date")
        if day.year != year or day.weekday() > 4:
            raise CalendarRefused(
                "invalid_date", f"published closure {day} is not a weekday in {year}"
            )
    repeated = sorted({day for day in days if days.count(day) > 1})
    if repeated:
        raise CalendarRefused(
            "duplicate_date",
            f"published closures list {', '.join(map(str, repeated))} more than once",
        )
    rules = set(calendar(calendar_id).full_day_closures(year, year))
    listed = set(days)
    return ClosureCheck(
        calendar_id=calendar_id,
        year=year,
        unexpected=tuple(sorted(rules - listed)),
        missing=tuple(sorted(listed - rules)),
    )


# --- the calendar consumers use (CALENDAR_ID) ---------------------------------------------

_CURRENT = calendar(CALENDAR_ID)
COVERAGE = _CURRENT.coverage
LAST_YEAR = _CURRENT.last_year
FULL_DAY_CLOSURES = _CURRENT.closures


def full_day_closures(first_year: int, last_year: int) -> tuple[date, ...]:
    """Every weekday closure in [first_year, last_year] of `CALENDAR_ID`'s calendar."""

    return _CURRENT.full_day_closures(first_year, last_year)


def session_date(value: object, *, label: str = "date") -> pd.Timestamp:
    """
    A date as 00:00 UTC (`Timestamp`, microsecond unit), refused outside
    `COVERAGE`. The same input rules as `nyse.session_date`: a `YYYY-MM-DD`
    string in ASCII digits, a `datetime.date`, or a naive or UTC timestamp at
    exactly midnight; a time of day or another timezone is refused.
    """

    return _CURRENT.session_date(value, label=label)


def sessions(start: object, end: object) -> pd.DatetimeIndex:
    """Every session in [start, end], both ends inclusive and inside coverage."""

    return _CURRENT.sessions(start, end)


def is_session(value: object) -> bool:
    """True when the date is a full NYSE session; refused outside coverage."""

    return _CURRENT.is_session(value)


def session_offset(day: object, offset: int) -> pd.Timestamp:
    """
    The session `offset` sessions after the session `day` (before it when
    negative; `day` itself for 0). Refuses a `day` that is not a session and a
    result outside coverage.
    """

    return _CURRENT.session_offset(day, offset)


__all__ = [
    "CALENDARS",
    "CALENDAR_ID",
    "CALENDAR_ID_2027",
    "CALENDAR_ID_2030",
    "COVERAGE",
    "FULL_DAY_CLOSURES",
    "NYSE_HOLIDAYS_RETRIEVED_ON",
    "NYSE_HOLIDAYS_URL",
    "SCHEDULE_UNVERIFIED_NOTE",
    "SCHEDULE_VERIFIED_NOTE",
    "SPECIAL_CLOSURES",
    "VERSION_LAST_YEARS",
    "YEAR_SOURCES",
    "ClosureCheck",
    "RulesCalendarVersion",
    "YearSource",
    "YearStatus",
    "calendar",
    "check_published",
    "easter_sunday",
    "full_day_closures",
    "holiday_closures",
    "is_session",
    "observed",
    "session_date",
    "session_offset",
    "sessions",
    "year_source",
]
