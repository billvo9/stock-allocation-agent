"""
The rule-based NYSE calendar (`calendars/nyse_rules.py`): it reproduces the
frozen research calendar exactly, and its default (the 2030 version since the
owner's 2026-10-18 switch) extends coverage through 2030 for the prospective
path. The frozen calendar's contract is pinned byte for byte. The 2027
version's own coverage is tested in test_nyse_rules_2030.py.
"""

import hashlib
from datetime import date

import pandas as pd
import pytest

from stock_agent.calendars import nyse, nyse_rules

# The frozen research calendar's contract, recorded 2026-10-14. A change to any
# of these is a change to nyse.py's contract for 2003-2024, which must not happen.
FROZEN_CALENDAR_ID = "nyse_full_day_2003_2024_v1"
FROZEN_SESSIONS_SHA256 = "f1ca058de450a6c916353697d847afeaac9285d9cb63bc57ffed8c52ddb2ffcf"
FROZEN_CLOSURES_SHA256 = "d26f606d661c1925c8b81697ba783728a9f86633de7b3002cb76bc159012dcfb"
FROZEN_SESSION_COUNT = 5537

# Full-day closures for 2025-2027. 2026 and 2027 equal NYSE's published "Holidays &
# Trading Hours" page, retrieved 2026-10-18 (test_nyse_rules_2030.py embeds it with its
# citation). 2025 was entered by hand as recalled; NYSE's 2025 page was not retrieved, so
# 2025 is rule-generated (expected, not verified: nyse_rules.year_source(2025)).
PUBLISHED = {
    2025: [
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 9),  # national day of mourning for Jimmy Carter
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Washington's Birthday
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving Day
        date(2025, 12, 25),  # Christmas Day
    ],
    2026: [
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),  # Good Friday (Easter 2026-04-05)
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),  # Independence Day on a Saturday, observed Friday
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    ],
    2027: [
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),  # Good Friday (Easter 2027-03-28)
        date(2027, 5, 31),
        date(2027, 6, 18),  # Juneteenth on a Saturday, observed Friday
        date(2027, 7, 5),  # Independence Day on a Sunday, observed Monday
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),  # Christmas Day on a Saturday, observed Friday
    ],
}


def test_the_rules_reproduce_the_frozen_closures_exactly():
    assert nyse_rules.full_day_closures(2003, 2024) == nyse.FULL_DAY_CLOSURES


@pytest.mark.parametrize("year", range(2003, 2025))
def test_each_frozen_year_is_reproduced(year):
    frozen = tuple(day for day in nyse.FULL_DAY_CLOSURES if day.year == year)
    assert nyse_rules.full_day_closures(year, year) == frozen


def test_the_extended_sessions_through_2024_equal_the_frozen_sessions():
    frozen = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
    extended = nyse_rules.sessions("2003-01-02", "2024-12-31")
    assert extended.equals(frozen)
    assert str(extended.dtype) == str(frozen.dtype) == "datetime64[us, UTC]"
    assert extended.asi8.tobytes() == frozen.asi8.tobytes()


def test_the_frozen_research_calendar_contract_is_byte_identical():
    frozen = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
    assert nyse.CALENDAR_ID == FROZEN_CALENDAR_ID
    assert (nyse.COVERAGE.first, nyse.COVERAGE.last) == (
        pd.Timestamp("2003-01-02", tz="UTC"),
        pd.Timestamp("2024-12-31", tz="UTC"),
    )
    assert len(frozen) == FROZEN_SESSION_COUNT
    assert hashlib.sha256(frozen.asi8.tobytes()).hexdigest() == FROZEN_SESSIONS_SHA256
    closures = repr(nyse.FULL_DAY_CLOSURES).encode()
    assert hashlib.sha256(closures).hexdigest() == FROZEN_CLOSURES_SHA256
    # The research calendar still refuses the holdout; the rules calendar is separate.
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse.is_session("2026-10-14")
    assert caught.value.reason == "lockbox_violation"
    assert nyse_rules.CALENDAR_ID != nyse.CALENDAR_ID


@pytest.mark.parametrize("year", sorted(PUBLISHED))
def test_later_years_match_the_published_schedule(year):
    assert list(nyse_rules.full_day_closures(year, year)) == PUBLISHED[year]


@pytest.mark.parametrize(("year", "count"), [(2025, 250), (2026, 251), (2027, 251)])
def test_later_session_counts(year, count):
    # 2025: 261 weekdays - 10 holidays - the 2025-01-09 closure. 2026 and 2027:
    # 261 weekdays - 10 weekday closures.
    assert len(nyse_rules.sessions(f"{year}-01-01", f"{year}-12-31")) == count


def test_saturday_and_sunday_observance_in_later_years():
    assert not nyse_rules.is_session("2026-07-03")  # Saturday July 4 -> Friday
    assert not nyse_rules.is_session("2027-06-18")  # Saturday Juneteenth -> Friday
    assert not nyse_rules.is_session("2027-07-05")  # Sunday July 4 -> Monday
    assert not nyse_rules.is_session("2027-12-24")  # Saturday Christmas -> Friday
    # Saturday 2028-01-01: the preceding Friday ends the year, so it trades.
    assert nyse_rules.is_session("2027-12-31")
    for day in ("2004-12-31", "2010-12-31", "2021-12-31"):
        assert nyse_rules.is_session(day)


def test_juneteenth_starts_in_2022():
    assert nyse_rules.is_session("2021-06-18")  # Saturday 2021-06-19 was not yet a holiday
    assert not nyse_rules.is_session("2022-06-20")
    assert nyse_rules.observed(date(2027, 6, 19)) == date(2027, 6, 18)
    assert nyse_rules.observed(date(2022, 1, 1)) is None  # Friday 2021-12-31 ends the year


def test_easter_matches_known_dates():
    known = [date(2008, 3, 23), date(2019, 4, 21), date(2025, 4, 20), date(2026, 4, 5)]
    known.append(date(2027, 3, 28))
    assert [nyse_rules.easter_sunday(day.year) for day in known] == known


def test_special_closures_extend_the_frozen_ones():
    frozen = tuple(nyse.SPECIAL_CLOSURES)
    assert nyse_rules.SPECIAL_CLOSURES[: len(frozen)] == frozen
    later = nyse_rules.SPECIAL_CLOSURES[len(frozen) :]
    assert later == ((date(2025, 1, 9), "national day of mourning for Jimmy Carter"),)
    assert not nyse_rules.is_session("2025-01-09")
    assert nyse_rules.is_session("2025-01-08") and nyse_rules.is_session("2025-01-10")


def test_rule_based_holidays_never_include_the_special_closures():
    for day, _ in nyse_rules.SPECIAL_CLOSURES:
        assert day not in nyse_rules.holiday_closures(day.year)


def test_the_full_coverage_is_typed_and_contiguous():
    every = nyse_rules.sessions(nyse_rules.COVERAGE.first, nyse_rules.COVERAGE.last)
    assert str(every.dtype) == "datetime64[us, UTC]"
    assert every[0] == pd.Timestamp("2003-01-02", tz="UTC")
    assert every[-1] == pd.Timestamp("2030-12-31", tz="UTC")
    assert every.is_monotonic_increasing and every.is_unique
    weekdays = pd.bdate_range("2003-01-02", "2030-12-31", tz="UTC")
    assert len(weekdays) - len(every) == len(nyse_rules.FULL_DAY_CLOSURES) - 1  # 2003-01-01


def test_session_offset_counts_sessions_not_days():
    assert nyse_rules.session_offset("2026-10-14", -63) == pd.Timestamp("2026-07-16", tz="UTC")
    assert nyse_rules.session_offset("2026-07-02", 1) == pd.Timestamp("2026-07-06", tz="UTC")
    assert nyse_rules.session_offset("2026-07-06", -1) == pd.Timestamp("2026-07-02", tz="UTC")
    assert nyse_rules.session_offset("2026-10-14", 0) == pd.Timestamp("2026-10-14", tz="UTC")
    window = nyse_rules.sessions("2026-07-16", "2026-10-14")
    assert len(window) == 64  # the 63 sessions before 2026-10-14, and 2026-10-14


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (lambda: nyse_rules.session_offset("2026-07-03", -1), "not_a_session"),
        (lambda: nyse_rules.session_offset("2030-12-31", 1), "outside_coverage"),
        (lambda: nyse_rules.session_offset("2003-01-02", -1), "outside_coverage"),
        (lambda: nyse_rules.session_offset("2026-10-14", 1.0), "invalid_offset"),
        (lambda: nyse_rules.session_offset("2026-10-14", True), "invalid_offset"),
    ],
)
def test_session_offset_refusals(call, reason):
    with pytest.raises(nyse.CalendarRefused) as caught:
        call()
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("2031-01-02", "outside_coverage"),
        ("2002-12-31", "outside_coverage"),
        ("2026-1-2", "invalid_date"),
        ("2026-02-29", "invalid_date"),
        ("２０２６-01-02", "invalid_date"),
        ("2026-01-02\n", "invalid_date"),
        (pd.Timestamp("2026-01-02 15:00"), "invalid_date"),
        (pd.Timestamp("2026-01-02", tz="America/New_York"), "invalid_date"),
        (pd.NaT, "invalid_date"),
        (20260102, "invalid_date"),
    ],
)
def test_dates_outside_coverage_or_malformed_are_refused(value, reason):
    for call in (nyse_rules.is_session, lambda v: nyse_rules.sessions(v, v)):
        with pytest.raises(nyse.CalendarRefused) as caught:
            call(value)
        assert caught.value.reason == reason


def test_full_day_closures_refuses_years_outside_coverage():
    for years in ((2002, 2003), (2030, 2031), (2026, 2025)):
        with pytest.raises(nyse.CalendarRefused):
            nyse_rules.full_day_closures(*years)


def test_every_closure_is_a_weekday_inside_coverage():
    closures = nyse_rules.FULL_DAY_CLOSURES
    assert all(day.weekday() < 5 for day in closures)
    assert list(closures) == sorted(set(closures))
    assert {day.year for day in closures} == set(range(2003, 2031))
