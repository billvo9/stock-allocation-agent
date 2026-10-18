"""
The rules calendar's 2030 version (`nyse_rules.CALENDAR_ID_2030`, the
calendar consumers use since the owner's 2026-10-18 switch), the 2027 version
still resolvable by its id, and where each year's closures come from: NYSE's
published schedule (2026-2028), the frozen research list (2003-2024), or the
rules alone (2025, 2029, 2030). Offline: the published schedule is embedded
below with its citation.
"""

import hashlib
from datetime import date

import pandas as pd
import pytest

from stock_agent.calendars import nyse, nyse_rules
from stock_agent.calendars.nyse_rules import RulesCalendarVersion, YearSource, YearStatus

V2027 = nyse_rules.calendar("nyse_full_day_rules_2003_2027_v1")
V2030 = nyse_rules.calendar("nyse_full_day_rules_2003_2030_v1")

# NYSE, "Holidays & Trading Hours", https://www.nyse.com/markets/hours-calendars,
# retrieved 2026-10-18 by the engineering lead and copied here by hand (this test
# makes no request). Full-day closures exactly as published. For 2028 the page
# says: "Because the holiday falls on Saturday, January 1, 2028, no New Year's
# Day holiday is observed." So 2027-12-31 trades.
PUBLISHED_SOURCE = ("https://www.nyse.com/markets/hours-calendars", date(2026, 10, 18))
PUBLISHED_FULL_DAY_CLOSURES = {
    2026: (
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Washington's Birthday
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day (observed)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day
    ),
    2027: (
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # Martin Luther King Jr. Day
        date(2027, 2, 15),  # Washington's Birthday
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed)
        date(2027, 7, 5),  # Independence Day (observed)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving Day
        date(2027, 12, 24),  # Christmas Day (observed)
    ),
    2028: (
        date(2028, 1, 17),  # Martin Luther King Jr. Day
        date(2028, 2, 21),  # Washington's Birthday
        date(2028, 4, 14),  # Good Friday
        date(2028, 5, 29),  # Memorial Day
        date(2028, 6, 19),  # Juneteenth
        date(2028, 7, 4),  # Independence Day
        date(2028, 9, 4),  # Labor Day
        date(2028, 11, 23),  # Thanksgiving Day
        date(2028, 12, 25),  # Christmas Day
    ),
}
# Early closes (1:00 p.m.) on the same page: ordinary sessions here, never closures.
PUBLISHED_EARLY_CLOSES = (
    date(2026, 11, 27),
    date(2026, 12, 24),
    date(2027, 11, 26),
    date(2028, 7, 3),
    date(2028, 11, 24),
)
# 2029 and 2030 are not published yet, so no published list can catch a moved holiday.
# These are the rules' expected closures, computed by hand from the holiday rules (the
# independent test review, 2026-10-18): expected, not verified.
EXPECTED_RULE_GENERATED_CLOSURES = {
    2029: (
        date(2029, 1, 1),  # New Year's Day (Monday)
        date(2029, 1, 15),  # Martin Luther King Jr. Day (third Monday)
        date(2029, 2, 19),  # Washington's Birthday (third Monday)
        date(2029, 3, 30),  # Good Friday (Easter 2029-04-01)
        date(2029, 5, 28),  # Memorial Day (last Monday)
        date(2029, 6, 19),  # Juneteenth (Tuesday)
        date(2029, 7, 4),  # Independence Day (Wednesday)
        date(2029, 9, 3),  # Labor Day (first Monday)
        date(2029, 11, 22),  # Thanksgiving Day (fourth Thursday)
        date(2029, 12, 25),  # Christmas Day (Tuesday)
    ),
    2030: (
        date(2030, 1, 1),  # New Year's Day (Tuesday)
        date(2030, 1, 21),  # Martin Luther King Jr. Day (third Monday)
        date(2030, 2, 18),  # Washington's Birthday (third Monday)
        date(2030, 4, 19),  # Good Friday (Easter 2030-04-21)
        date(2030, 5, 27),  # Memorial Day (last Monday)
        date(2030, 6, 19),  # Juneteenth (Wednesday)
        date(2030, 7, 4),  # Independence Day (Thursday)
        date(2030, 9, 2),  # Labor Day (first Monday)
        date(2030, 11, 28),  # Thanksgiving Day (fourth Thursday)
        date(2030, 12, 25),  # Christmas Day (Wednesday)
    ),
}
# Each version's full session list, pinned like the frozen calendar's in test_nyse_rules.py:
# the count and the sha256 of its datetime64[us, UTC] values (recorded 2026-10-18).
VERSION_PINS = {
    "nyse_full_day_rules_2003_2027_v1": (
        6289,
        "b7ec2cac42da120a65c728283c3cf897f40f6fd3ffa4c86e3bae3302f1d2a53a",
    ),
    "nyse_full_day_rules_2003_2030_v1": (
        7042,
        "68550059b40b9907fabf74a2945b1c79cb0f6d92f0572bb8eefe8e0c2075591b",
    ),
}


@pytest.mark.parametrize("year", sorted(PUBLISHED_FULL_DAY_CLOSURES))
def test_the_rules_equal_the_published_2026_2028_closures_exactly(year):
    published = PUBLISHED_FULL_DAY_CLOSURES[year]
    assert V2030.full_day_closures(year, year) == published
    check = nyse_rules.check_published(year, published)
    assert check.matches and check.unexpected == check.missing == ()
    assert check.calendar_id == nyse_rules.CALENDAR_ID_2030
    # The published sessions: every other weekday of the year.
    weekdays = pd.bdate_range(f"{year}-01-01", f"{year}-12-31", tz="UTC").as_unit("us")
    expected = weekdays[~weekdays.isin(pd.DatetimeIndex(published, tz="UTC"))]
    assert V2030.sessions(f"{year}-01-01", f"{year}-12-31").equals(expected)


@pytest.mark.parametrize("year", sorted(EXPECTED_RULE_GENERATED_CLOSURES))
def test_the_rule_generated_2029_2030_closures_are_pinned(year):
    expected = EXPECTED_RULE_GENERATED_CLOSURES[year]
    assert V2030.full_day_closures(year, year) == expected
    assert nyse_rules.full_day_closures(year, year) == expected
    assert nyse_rules.check_published(year, expected).matches
    assert not nyse_rules.year_source(year).verified  # expected, not published


@pytest.mark.parametrize(("calendar_id", "pin"), sorted(VERSION_PINS.items()))
def test_each_versions_sessions_are_pinned(calendar_id, pin):
    count, sha256 = pin
    version = nyse_rules.calendar(calendar_id)
    every = version.sessions(version.coverage.first, version.coverage.last)
    assert str(every.dtype) == "datetime64[us, UTC]"
    assert len(every) == count
    assert hashlib.sha256(every.asi8.tobytes()).hexdigest() == sha256


@pytest.mark.parametrize("year", [2026, 2027])
def test_the_2027_calendar_equals_the_published_closures_too(year):
    published = PUBLISHED_FULL_DAY_CLOSURES[year]
    assert V2027.full_day_closures(year, year) == published
    assert nyse_rules.full_day_closures(year, year) == published
    assert nyse_rules.check_published(year, published, calendar_id=V2027.calendar_id).matches


def test_2027_12_31_is_a_session():
    for version in (V2027, V2030):
        assert version.is_session("2027-12-31")
    assert nyse_rules.is_session("2027-12-31")
    assert nyse_rules.observed(date(2028, 1, 1)) is None  # Saturday; Friday ends the year
    assert date(2027, 12, 31) not in nyse_rules.holiday_closures(2027)
    assert V2030.session_offset("2027-12-31", 1) == pd.Timestamp("2028-01-03", tz="UTC")
    assert V2030.session_offset("2027-12-31", -1) == pd.Timestamp("2027-12-30", tz="UTC")


def test_published_early_closes_are_sessions():
    for day in PUBLISHED_EARLY_CLOSES:
        assert V2030.is_session(day), day
        assert day not in V2030.closures


def test_observance_edge_cases_through_2030():
    closed = {
        "2026-07-03": "Independence Day (observed)",  # Saturday July 4 -> Friday
        "2027-06-18": "Juneteenth (observed)",  # Saturday -> Friday
        "2027-07-05": "Independence Day (observed)",  # Sunday -> Monday
        "2027-12-24": "Christmas Day (observed)",  # Saturday -> Friday
        "2028-06-19": "Juneteenth",  # Monday
        "2028-07-04": "Independence Day",  # Tuesday
        "2028-12-25": "Christmas Day",  # Monday
        "2029-01-01": "New Year's Day",  # Monday
        "2029-06-19": "Juneteenth",  # Tuesday
        "2029-07-04": "Independence Day",  # Wednesday
        "2029-12-25": "Christmas Day",  # Tuesday
        "2030-01-01": "New Year's Day",  # Tuesday
        "2030-06-19": "Juneteenth",  # Wednesday
        "2030-07-04": "Independence Day",  # Thursday
        "2030-12-25": "Christmas Day",  # Wednesday
    }
    for text, name in closed.items():
        day = date.fromisoformat(text)
        assert not V2030.is_session(text), text
        assert nyse_rules.holiday_closures(day.year)[day] == name
    # Neighbours of the observed and fixed holidays trade.
    for text in ("2026-07-02", "2026-07-06", "2027-06-17", "2027-07-06", "2027-12-23"):
        assert V2030.is_session(text), text
    for text in ("2028-12-29", "2029-12-31", "2030-12-31", "2030-12-24", "2030-12-26"):
        assert V2030.is_session(text), text
    # No New Year's observance on December 31 when January 1 is a weekday.
    for year in (2028, 2029, 2030):
        assert date(year, 12, 31) not in nyse_rules.holiday_closures(year)


def test_good_friday_2029_and_2030():
    assert nyse_rules.easter_sunday(2028) == date(2028, 4, 16)
    assert nyse_rules.easter_sunday(2029) == date(2029, 4, 1)
    assert nyse_rules.easter_sunday(2030) == date(2030, 4, 21)
    for good_friday, thursday, monday in (
        ("2029-03-30", "2029-03-29", "2029-04-02"),
        ("2030-04-19", "2030-04-18", "2030-04-22"),
    ):
        day = date.fromisoformat(good_friday)
        assert nyse_rules.holiday_closures(day.year)[day] == "Good Friday"
        assert not V2030.is_session(good_friday)
        assert V2030.is_session(thursday) and V2030.is_session(monday)
        assert V2030.session_offset(thursday, 1) == pd.Timestamp(monday, tz="UTC")


@pytest.mark.parametrize(
    ("year", "weekdays", "closures", "count"),
    [
        (2025, 261, 11, 250),  # 10 holidays and the 2025-01-09 special closure
        (2026, 261, 10, 251),
        (2027, 261, 10, 251),
        (2028, 260, 9, 251),  # a leap year starting on Saturday; no New Year's closure
        (2029, 261, 10, 251),
        (2030, 261, 10, 251),
    ],
)
def test_session_counts_per_year(year, weekdays, closures, count):
    first, last = f"{year}-01-01", f"{year}-12-31"
    assert len(pd.bdate_range(first, last)) == weekdays
    assert len(V2030.full_day_closures(year, year)) == closures
    assert len(V2030.sessions(first, last)) == count == weekdays - closures


def test_the_2030_version_reproduces_the_frozen_research_calendar():
    assert V2030.full_day_closures(2003, 2024) == nyse.FULL_DAY_CLOSURES
    frozen = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
    extended = V2030.sessions("2003-01-02", "2024-12-31")
    assert extended.asi8.tobytes() == frozen.asi8.tobytes()
    assert str(extended.dtype) == "datetime64[us, UTC]"


def test_the_2027_version_is_an_exact_prefix_of_the_2030_version():
    old = V2027.sessions(V2027.coverage.first, V2027.coverage.last)
    new = V2030.sessions(V2027.coverage.first, V2027.coverage.last)
    assert old.asi8.tobytes() == new.asi8.tobytes() and str(old.dtype) == str(new.dtype)
    assert V2030.closures[: len(V2027.closures)] == V2027.closures
    assert V2030.full_day_closures(2003, 2027) == V2027.closures


def test_each_version_has_its_own_coverage():
    assert (V2027.coverage.first, V2027.coverage.last) == (
        pd.Timestamp("2003-01-02", tz="UTC"),
        pd.Timestamp("2027-12-31", tz="UTC"),
    )
    assert (V2030.coverage.first, V2030.coverage.last) == (
        pd.Timestamp("2003-01-02", tz="UTC"),
        pd.Timestamp("2030-12-31", tz="UTC"),
    )
    every = V2030.sessions(V2030.coverage.first, V2030.coverage.last)
    assert every[0] == pd.Timestamp("2003-01-02", tz="UTC")
    assert every[-1] == pd.Timestamp("2030-12-31", tz="UTC")
    assert every.is_monotonic_increasing and every.is_unique
    weekdays = pd.bdate_range("2003-01-02", "2030-12-31", tz="UTC")
    assert len(weekdays) - len(every) == len(V2030.closures) - 1  # 2003-01-01
    assert all(day.weekday() < 5 for day in V2030.closures)
    assert {day.year for day in V2030.closures} == set(range(2003, 2031))


@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (lambda: V2030.is_session("2031-01-02"), "outside_coverage"),
        (lambda: V2030.sessions("2030-12-31", "2031-01-02"), "outside_coverage"),
        (lambda: V2030.session_offset("2030-12-31", 1), "outside_coverage"),
        (lambda: V2030.session_offset("2029-03-30", 0), "not_a_session"),
        (lambda: V2030.full_day_closures(2030, 2031), "outside_coverage"),
        (lambda: V2027.is_session("2028-01-03"), "outside_coverage"),
        (lambda: V2027.full_day_closures(2027, 2028), "outside_coverage"),
        (lambda: V2030.is_session("2029-3-30"), "invalid_date"),
        (lambda: V2030.sessions("2030-01-02", "2030-01-01"), "invalid_range"),
    ],
)
def test_each_version_refuses_outside_its_coverage(call, reason):
    with pytest.raises(nyse.CalendarRefused) as caught:
        call()
    assert caught.value.reason == reason


def test_the_calendar_resolves_both_ids_and_refuses_others():
    assert set(nyse_rules.CALENDARS) == {
        "nyse_full_day_rules_2003_2027_v1",
        "nyse_full_day_rules_2003_2030_v1",
    }
    for calendar_id, version in nyse_rules.CALENDARS.items():
        assert nyse_rules.calendar(calendar_id) is version
        assert version.calendar_id == calendar_id
    for unknown in ("nyse_full_day_2003_2024_v1", "nyse_full_day_rules_2003_2030_v2", "", None):
        with pytest.raises(nyse.CalendarRefused) as caught:
            nyse_rules.calendar(unknown)
        assert caught.value.reason == "unknown_calendar"
    with pytest.raises(TypeError):
        nyse_rules.CALENDARS["x"] = V2030  # read-only


def test_the_calendar_consumers_record_is_the_2030_version():
    # The owner approved the switch on 2026-10-18 (docs/research/prospective_data_path.md,
    # section 1): the prospective store and the Stock Lab inference store record and
    # require this id. The 2027 version stays resolvable by its id.
    assert nyse_rules.CALENDAR_ID == nyse_rules.CALENDAR_ID_2030
    assert nyse_rules.CALENDAR_ID == "nyse_full_day_rules_2003_2030_v1"
    assert nyse_rules.calendar(nyse_rules.CALENDAR_ID) is V2030
    assert nyse_rules.COVERAGE == V2030.coverage and nyse_rules.LAST_YEAR == 2030
    assert nyse_rules.FULL_DAY_CLOSURES == V2030.closures
    every = nyse_rules.sessions(nyse_rules.COVERAGE.first, nyse_rules.COVERAGE.last)
    assert every.equals(V2030.sessions(V2030.coverage.first, V2030.coverage.last))
    assert every.asi8.tobytes() == V2030.sessions("2003-01-02", "2030-12-31").asi8.tobytes()
    assert nyse_rules.session_offset("2027-12-31", 1) == pd.Timestamp("2028-01-03", tz="UTC")
    assert nyse_rules.is_session("2030-12-31")
    assert nyse_rules.calendar(nyse_rules.CALENDAR_ID_2027) is V2027


@pytest.mark.parametrize(
    "call",
    [
        lambda: nyse_rules.session_date("2031-01-02"),
        lambda: nyse_rules.is_session("2031-01-02"),
        lambda: nyse_rules.sessions("2030-12-31", "2031-01-02"),
        lambda: nyse_rules.session_offset("2030-12-31", 1),
        lambda: nyse_rules.full_day_closures(2030, 2031),
    ],
)
def test_the_module_level_calendar_refuses_dates_after_2030(call):
    with pytest.raises(nyse.CalendarRefused) as caught:
        call()
    assert caught.value.reason == "outside_coverage"


@pytest.mark.parametrize(
    ("calendar_id", "last_year"),
    [
        ("nyse_full_day_rules_2003_2027_v1", 2030),  # a known id with another coverage
        ("nyse_full_day_rules_2003_2030_v1", 2027),
        ("nyse_full_day_rules_2003_2031_v1", 2031),  # an unknown id
        ("nyse_full_day_rules_2003_2030_v1", 2030.0),
        (None, 2030),
    ],
)
def test_only_known_versions_can_be_constructed(calendar_id, last_year):
    with pytest.raises(nyse.CalendarRefused) as caught:
        RulesCalendarVersion(calendar_id, last_year)
    assert caught.value.reason == "unknown_calendar"


def test_a_known_version_constructs_the_same_calendar():
    for calendar_id, last_year in nyse_rules.VERSION_LAST_YEARS.items():
        built = RulesCalendarVersion(calendar_id, last_year)
        assert built.closures == nyse_rules.calendar(calendar_id).closures
        assert built.coverage == nyse_rules.calendar(calendar_id).coverage


def test_every_year_has_one_source():
    assert sorted(nyse_rules.YEAR_SOURCES) == list(range(2003, 2031))
    for year, source in nyse_rules.YEAR_SOURCES.items():
        assert source.year == year and nyse_rules.year_source(year) is source
    statuses = {year: s.status for year, s in nyse_rules.YEAR_SOURCES.items()}
    assert {y for y, s in statuses.items() if s is YearStatus.FROZEN_RESEARCH_LIST} == set(
        range(2003, 2025)
    )
    assert {y for y, s in statuses.items() if s is YearStatus.VERIFIED_PUBLISHED} == {
        2026,
        2027,
        2028,
    }
    assert {y for y, s in statuses.items() if s is YearStatus.RULE_GENERATED} == {2025, 2029, 2030}


def test_exactly_the_years_with_an_embedded_published_list_are_verified():
    verified = {y for y, s in nyse_rules.YEAR_SOURCES.items() if s.verified}
    assert verified == set(PUBLISHED_FULL_DAY_CLOSURES)
    assert not verified & set(EXPECTED_RULE_GENERATED_CLOSURES)


@pytest.mark.parametrize("year", sorted(PUBLISHED_FULL_DAY_CLOSURES))
def test_verified_years_carry_their_source(year):
    source = nyse_rules.year_source(year)
    assert source.verified and source.status == "verified_published"
    assert (source.source_url, source.retrieved_on) == PUBLISHED_SOURCE
    record = source.to_json_dict()
    assert record == {
        "year": year,
        "status": "verified_published",
        "schedule_verified": True,
        "detail": "equal to NYSE's published Holidays & Trading Hours page",
        "source_url": "https://www.nyse.com/markets/hours-calendars",
        "retrieved_on": "2026-10-18",
        "note": nyse_rules.SCHEDULE_VERIFIED_NOTE,
    }
    # The schedule is verified as of retrieved_on; sessions are never called verified.
    assert "verified" not in record
    assert "as of retrieved_on" in record["note"] and "remain expected" in record["note"]


@pytest.mark.parametrize("year", [2025, 2029, 2030, 2003, 2024])
def test_other_years_are_labelled_unverified(year):
    source = nyse_rules.year_source(year)
    assert not source.verified
    assert source.source_url is None and source.retrieved_on is None
    expected = YearStatus.FROZEN_RESEARCH_LIST if year < 2025 else YearStatus.RULE_GENERATED
    assert source.status is expected
    record = source.to_json_dict()
    assert record["schedule_verified"] is False
    assert record["note"] == nyse_rules.SCHEDULE_UNVERIFIED_NOTE


URL, ON = PUBLISHED_SOURCE


@pytest.mark.parametrize(
    ("status", "url", "retrieved_on"),
    [
        (YearStatus.VERIFIED_PUBLISHED, None, None),  # verified without a citation
        (YearStatus.VERIFIED_PUBLISHED, URL, None),
        (YearStatus.VERIFIED_PUBLISHED, None, ON),
        (YearStatus.VERIFIED_PUBLISHED, "http://www.nyse.com/markets/hours-calendars", ON),
        (YearStatus.VERIFIED_PUBLISHED, URL, "2026-10-18"),  # not a date
        (YearStatus.VERIFIED_PUBLISHED, URL, pd.Timestamp("2026-10-18")),
        (YearStatus.RULE_GENERATED, URL, ON),  # a citation on an unverified year
        (YearStatus.RULE_GENERATED, URL, None),
        (YearStatus.FROZEN_RESEARCH_LIST, None, ON),
        ("verified_published", URL, ON),  # a plain string, not a YearStatus
    ],
)
def test_a_year_source_cites_a_source_exactly_when_verified(status, url, retrieved_on):
    with pytest.raises(nyse.CalendarRefused) as caught:
        YearSource(2029, status, "detail", source_url=url, retrieved_on=retrieved_on)
    assert caught.value.reason == "invalid_year_source"


def test_well_formed_year_sources_are_accepted():
    verified = YearSource(2029, YearStatus.VERIFIED_PUBLISHED, "d", URL, ON)
    assert verified.verified
    assert not YearSource(2029, YearStatus.RULE_GENERATED, "d").verified
    assert not YearSource(2010, YearStatus.FROZEN_RESEARCH_LIST, "d").verified


def test_2025_is_rule_generated_with_its_special_closure():
    source = nyse_rules.year_source(2025)
    assert source.status is YearStatus.RULE_GENERATED and "2025-01-09" in source.detail
    assert date(2025, 1, 9) in V2030.full_day_closures(2025, 2025)
    assert date(2025, 1, 9) not in nyse_rules.holiday_closures(2025)


@pytest.mark.parametrize(
    ("year", "reason"),
    [
        (2002, "outside_coverage"),
        (2031, "outside_coverage"),
        (True, "invalid_year"),  # a bool is not a year
        ("2026", "invalid_year"),
        (2026.0, "invalid_year"),
        (None, "invalid_year"),
    ],
)
def test_year_source_refuses_wrong_types_and_years_outside_the_table(year, reason):
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse_rules.year_source(year)
    assert caught.value.reason == reason


def test_a_fabricated_closure_fails_the_check():
    fabricated = date(2028, 3, 17)  # a Friday, not a holiday
    published = (*PUBLISHED_FULL_DAY_CLOSURES[2028], fabricated)
    check = nyse_rules.check_published(2028, published)
    assert not check.matches
    assert check.missing == (fabricated,) and check.unexpected == ()
    assert V2030.is_session(fabricated)  # the rules keep it a session


def test_a_missing_holiday_fails_the_check():
    published = tuple(d for d in PUBLISHED_FULL_DAY_CLOSURES[2028] if d != date(2028, 6, 19))
    check = nyse_rules.check_published(2028, published)
    assert not check.matches
    assert check.unexpected == (date(2028, 6, 19),) and check.missing == ()
    assert not V2030.is_session("2028-06-19")


def test_a_wrong_observance_fails_the_check():
    # Observing Saturday 2028-01-01 on Friday 2027-12-31 would be a closure the rules lack.
    published = (*PUBLISHED_FULL_DAY_CLOSURES[2027], date(2027, 12, 31))
    check = nyse_rules.check_published(2027, published)
    assert check.missing == (date(2027, 12, 31),) and not check.matches
    # And a Sunday holiday observed on the wrong day: Monday 2027-07-05 replaced by Friday.
    moved = tuple(
        date(2027, 7, 2) if d == date(2027, 7, 5) else d for d in PUBLISHED_FULL_DAY_CLOSURES[2027]
    )
    check = nyse_rules.check_published(2027, moved)
    assert check.missing == (date(2027, 7, 2),) and check.unexpected == (date(2027, 7, 5),)


@pytest.mark.parametrize(
    "bad",
    [
        date(2028, 1, 1),  # a Saturday: never a weekday closure
        date(2029, 1, 1),  # another year
        pd.Timestamp("2028-06-19"),  # not a datetime.date
        "2028-06-19",
    ],
)
def test_check_published_refuses_malformed_lists(bad):
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse_rules.check_published(2028, (*PUBLISHED_FULL_DAY_CLOSURES[2028], bad))
    assert caught.value.reason == "invalid_date"


def test_check_published_refuses_a_date_listed_twice():
    for repeated in (date(2028, 6, 19), date(2028, 1, 17)):
        published = (*PUBLISHED_FULL_DAY_CLOSURES[2028], repeated)
        with pytest.raises(nyse.CalendarRefused) as caught:
            nyse_rules.check_published(2028, published)
        assert caught.value.reason == "duplicate_date"
        assert str(repeated) in caught.value.detail
    # A repeated fabricated date is refused too, not reported once as missing.
    fabricated = date(2028, 3, 17)
    published = (*PUBLISHED_FULL_DAY_CLOSURES[2028], fabricated, fabricated)
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse_rules.check_published(2028, published)
    assert caught.value.reason == "duplicate_date"


def test_check_published_refuses_an_unknown_calendar_and_a_year_outside_it():
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse_rules.check_published(2026, (), calendar_id="nyse_full_day_2003_2024_v1")
    assert caught.value.reason == "unknown_calendar"
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse_rules.check_published(2028, (), calendar_id=nyse_rules.CALENDAR_ID_2027)
    assert caught.value.reason == "outside_coverage"
