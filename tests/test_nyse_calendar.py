from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from stock_agent.calendars import nyse
from stock_agent.model_validation import folds

YEARS = range(2003, 2025)
SPECIAL = {
    date(2004, 6, 11),
    date(2007, 1, 2),
    date(2012, 10, 29),
    date(2012, 10, 30),
    date(2018, 12, 5),
}


# --- an independent rule-based derivation (shares no code with nyse.py) ----


def _easter(year: int) -> date:
    """Anonymous Gregorian computus (Meeus/Jones/Butcher)."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """Sunday -> Monday; Saturday -> Friday unless that Friday ends a month or year."""

    if day.weekday() == 6:
        return day + timedelta(days=1)
    if day.weekday() == 5:
        friday = day - timedelta(days=1)
        ends_month = (friday + timedelta(days=3)).month != friday.month
        return None if ends_month else friday
    return day


def _derived_closures(year: int) -> set[date]:
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 12, 25), date(year + 1, 1, 1)]
    if year >= 2022:
        fixed.append(date(year, 6, 19))
    observed = {_observed(day) for day in fixed} - {None}
    floating = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving Day
    }
    special = {day for day in SPECIAL if day.year == year}
    return {day for day in observed | floating | special if day.year == year}


def test_the_computus_matches_known_easter_sundays():
    known = [date(2003, 4, 20), date(2008, 3, 23), date(2011, 4, 24), date(2019, 4, 21)]
    assert [_easter(day.year) for day in known] == known


@pytest.mark.parametrize("year", YEARS)
def test_the_frozen_list_equals_the_rule_based_derivation(year):
    frozen = {day for day in nyse.FULL_DAY_CLOSURES if day.year == year}
    assert frozen == _derived_closures(year)


def test_the_frozen_list_is_sorted_unique_weekdays_inside_coverage():
    closures = nyse.FULL_DAY_CLOSURES
    assert isinstance(closures, tuple)
    assert list(closures) == sorted(set(closures))
    assert all(day.weekday() < 5 for day in closures)
    assert {day.year for day in closures} == set(YEARS)
    assert len(closures) == 22 * 9 - 3 + 3 + 5  # three skipped New Years, Juneteenth x3


def test_special_closures_are_listed_and_are_not_sessions():
    assert {day for day, _ in nyse.SPECIAL_CLOSURES} == SPECIAL
    assert SPECIAL <= set(nyse.FULL_DAY_CLOSURES)
    for day in SPECIAL:
        assert not nyse.is_session(day)
        assert nyse.is_session(day - timedelta(days=7))  # same weekday a week earlier traded


# Counts by hand: weekdays in the year minus weekday closures.
#   2003: 261 - 9 = 252            2004: 262 - 9 - Reagan = 252
#   2008: 262 - 9 = 253            2012: 261 - 9 - 2 (Sandy) = 250
#   2018: 261 - 9 - Bush = 251     2021: 261 - 9 = 252 (2021-12-31 trades)
#   2022: 260 - 9 = 251 (no New Year closure; Juneteenth observed Monday)
#   2023: 260 - 10 = 250           2024: 262 - 10 = 252
@pytest.mark.parametrize(
    ("year", "count"),
    [
        (2003, 252),
        (2004, 252),
        (2008, 253),
        (2012, 250),
        (2018, 251),
        (2021, 252),
        (2022, 251),
        (2023, 250),
        (2024, 252),
    ],
)
def test_session_counts(year, count):
    start = "2003-01-02" if year == 2003 else f"{year}-01-01"
    assert len(nyse.sessions(start, f"{year}-12-31")) == count


def test_full_coverage_is_contiguous_and_typed():
    every = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
    assert str(every.dtype) == "datetime64[us, UTC]"
    assert every[0] == pd.Timestamp("2003-01-02", tz="UTC")
    assert every[-1] == pd.Timestamp("2024-12-31", tz="UTC")
    assert every.is_monotonic_increasing and every.is_unique
    assert (every == every.normalize()).all()
    weekdays = pd.bdate_range("2003-01-02", "2024-12-31", tz="UTC")
    assert len(weekdays) - len(every) == len(nyse.FULL_DAY_CLOSURES) - 1  # 2003-01-01


def test_sessions_are_inclusive_and_typed():
    window = nyse.sessions("2012-10-26", "2012-11-01")
    assert list(window.strftime("%Y-%m-%d")) == ["2012-10-26", "2012-10-31", "2012-11-01"]
    assert str(window.dtype) == "datetime64[us, UTC]"
    assert len(nyse.sessions("2012-10-29", "2012-10-30")) == 0


def test_juneteenth_is_a_session_before_2022():
    for year in (2003, 2015, 2020, 2021):
        day = date(year, 6, 19)
        if day.weekday() < 5:
            assert nyse.is_session(day), day
    assert not nyse.is_session("2022-06-20")  # Sunday holiday observed Monday
    assert not nyse.is_session("2023-06-19")
    assert not nyse.is_session("2024-06-19")


@pytest.mark.parametrize("day", ["2004-12-31", "2010-12-31", "2021-12-31"])
def test_saturday_new_year_does_not_close_the_preceding_friday(day):
    assert pd.Timestamp(day).day_name() == "Friday"
    assert nyse.is_session(day)


def test_saturday_holidays_close_the_preceding_friday_otherwise():
    for day in ("2004-12-24", "2009-07-03", "2010-12-24", "2015-07-03", "2020-07-03"):
        assert not nyse.is_session(day), day
    for day in ("2006-01-02", "2011-12-26", "2017-01-02", "2021-07-05", "2022-12-26"):
        assert not nyse.is_session(day), day  # Sunday holidays observed Monday


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("2025-01-01", "lockbox_violation"),
        ("2025-01-02", "lockbox_violation"),
        (pd.Timestamp("2026-03-02", tz="UTC"), "lockbox_violation"),
        ("2002-12-31", "outside_coverage"),
        ("2003-01-01", "outside_coverage"),
        ("1999-06-01", "outside_coverage"),
        ("2024-1-2", "invalid_date"),
        (pd.Timestamp("2024-01-02 15:00"), "invalid_date"),
        (pd.Timestamp("2024-01-02", tz="America/New_York"), "invalid_date"),
        (pd.NaT, "invalid_date"),
        (20240102, "invalid_date"),
    ],
)
def test_dates_outside_coverage_or_malformed_are_refused(value, reason):
    for call in (nyse.is_session, lambda v: nyse.sessions(v, v)):
        with pytest.raises(nyse.CalendarRefused) as caught:
            call(value)
        assert caught.value.reason == reason


def test_a_range_reaching_the_lockbox_is_refused():
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse.sessions("2024-12-02", "2025-01-03")
    assert caught.value.reason == "lockbox_violation"
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse.sessions("2024-12-31", "2024-12-02")
    assert caught.value.reason == "invalid_range"


def test_accepted_date_forms_agree():
    forms = ["2024-03-01", date(2024, 3, 1), datetime(2024, 3, 1, tzinfo=UTC)]
    forms += [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01", tz="UTC")]
    assert {nyse.session_date(form) for form in forms} == {pd.Timestamp("2024-03-01", tz="UTC")}


def test_the_lockbox_has_one_source_and_binds_the_calendar(monkeypatch):
    assert nyse.MODEL_HOLDOUT_START is folds.MODEL_HOLDOUT_START
    assert nyse.COVERAGE.last < folds.MODEL_HOLDOUT_START
    monkeypatch.setattr(nyse, "MODEL_HOLDOUT_START", pd.Timestamp("2020-01-01", tz="UTC"))
    with pytest.raises(nyse.CalendarRefused) as caught:
        nyse.is_session("2021-03-01")
    assert caught.value.reason == "lockbox_violation"
