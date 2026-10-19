"""
When a premarket briefing is for: 08:30 New York time on NYSE session
mornings, the prior session's close as its data, weekends and holidays named,
and daylight saving handled by the time zone, never by a fixed offset.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab.briefing import timing
from stock_agent.stock_lab.briefing.timing import (
    ADVANCE,
    AFTER_OPEN,
    CLOSED_DAY,
    PREMARKET,
    TimingRefused,
    briefing_timing,
    gap_note,
    scheduled_at,
)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def plan(text: str):
    return briefing_timing(utc(text), nyse_sessions)


# --- the target time and daylight saving ---


@pytest.mark.parametrize(
    ("market_date", "expected_utc"),
    [
        (date(2026, 3, 6), "2026-03-06T13:30:00"),  # Friday, standard time (UTC-5)
        (date(2026, 3, 9), "2026-03-09T12:30:00"),  # Monday after DST began on 8 March
        (date(2026, 10, 30), "2026-10-30T12:30:00"),  # Friday, daylight time (UTC-4)
        (date(2026, 11, 2), "2026-11-02T13:30:00"),  # Monday after DST ended on 1 November
    ],
)
def test_the_target_is_0830_new_york_time_across_daylight_saving(market_date, expected_utc):
    target = scheduled_at(market_date)
    assert (target.hour, target.minute) == (8, 30)
    assert target.astimezone(UTC) == utc(expected_utc)


def test_a_premarket_run_on_a_session_morning_is_for_that_session():
    run = plan("2026-10-19T12:30:00")  # Monday 08:30 New York (EDT)
    assert run.state == PREMARKET and run.deliverable
    assert run.market_date == date(2026, 10, 19)
    assert run.prior_session == date(2026, 10, 16)
    assert run.scheduled_at == scheduled_at(date(2026, 10, 19))


@pytest.mark.parametrize(
    ("moment", "state"),
    [
        ("2026-03-06T14:29:59", PREMARKET),  # 09:29:59 EST
        ("2026-03-06T14:30:00", AFTER_OPEN),  # 09:30 EST: the session has opened
        ("2026-03-09T13:29:59", PREMARKET),  # 09:29:59 EDT, the day after DST began
        ("2026-03-09T13:30:00", AFTER_OPEN),
    ],
)
def test_the_open_boundary_follows_daylight_saving(moment, state):
    run = plan(moment)
    assert run.state == state
    assert run.deliverable is (state == PREMARKET)


def test_after_the_close_is_available_the_next_session_is_prepared_in_advance():
    run = plan("2026-10-20T00:30:00")  # Monday 20:30 New York
    assert run.state == ADVANCE and not run.deliverable
    assert run.market_date == date(2026, 10, 20)
    assert run.prior_session == date(2026, 10, 19)
    assert run.closed_days == ()


def test_between_the_open_and_the_close_the_briefing_is_late_for_today():
    run = plan("2026-10-19T19:00:00")  # Monday 15:00 New York
    assert run.state == AFTER_OPEN
    assert run.market_date == date(2026, 10, 19)


# --- weekends and holidays ---


def test_a_monday_briefing_says_friday_data_has_no_weekend_information():
    run = plan("2026-10-19T12:30:00")
    assert run.closed_days == (date(2026, 10, 17), date(2026, 10, 18))
    note = gap_note(run)
    assert "Friday 16 October 2026" in note
    assert "weekend" in note and "no information" in note


def test_a_sunday_run_is_a_closed_day_skipped_unless_configured_and_says_so():
    run = plan("2026-10-18T12:30:00")  # Sunday 08:30 New York
    assert run.state == CLOSED_DAY and not run.deliverable
    assert run.market_date == date(2026, 10, 19)
    assert run.prior_session == date(2026, 10, 16)
    assert "no information" in gap_note(run)


@pytest.mark.parametrize(
    ("moment", "market_date", "prior", "holiday"),
    [
        # Thanksgiving (Thursday 26 November 2026): closed; Friday is a session.
        ("2026-11-26T13:30:00", date(2026, 11, 27), date(2026, 11, 25), date(2026, 11, 26)),
        # Good Friday 2026 (3 April): closed.
        ("2026-04-03T12:30:00", date(2026, 4, 6), date(2026, 4, 2), date(2026, 4, 3)),
        # Christmas 2026 is a Friday.
        ("2026-12-25T13:30:00", date(2026, 12, 28), date(2026, 12, 24), date(2026, 12, 25)),
    ],
)
def test_market_holidays_are_closed_days(moment, market_date, prior, holiday):
    run = plan(moment)
    assert run.state == CLOSED_DAY and not run.deliverable
    assert (run.market_date, run.prior_session) == (market_date, prior)
    assert holiday in run.closed_days
    assert "holiday" in gap_note(run)


def test_the_morning_after_a_monday_holiday_names_the_holiday_and_the_weekend():
    run = plan("2026-01-20T13:30:00")  # Tuesday after Martin Luther King Jr. Day
    assert run.state == PREMARKET
    assert run.prior_session == date(2026, 1, 16)
    assert run.closed_days == (date(2026, 1, 17), date(2026, 1, 18), date(2026, 1, 19))
    note = gap_note(run)
    assert "holiday" in note and "weekend" in note


def test_an_unscheduled_closure_known_to_the_calendar_is_a_closed_day():
    run = plan("2025-01-10T13:30:00")  # the morning after the 9 January 2025 closure
    assert run.prior_session == date(2025, 1, 8)
    assert run.closed_days == (date(2025, 1, 9),)


def test_an_ordinary_tuesday_has_no_gap_note():
    run = plan("2026-10-20T12:30:00")
    assert run.closed_days == () and gap_note(run) is None


# --- inputs ---


def test_a_naive_time_is_refused():
    with pytest.raises(TimingRefused):
        briefing_timing(datetime(2026, 10, 19, 8, 30), nyse_sessions)  # noqa: DTZ001


def test_a_calendar_without_a_coming_session_is_refused():
    def empty(start, end):
        return pd.DatetimeIndex([], tz="UTC")

    with pytest.raises(TimingRefused):
        briefing_timing(utc("2026-10-19T12:30:00"), empty)


def test_the_synthetic_weekday_calendar_works_the_same_way():
    def weekdays(start, end):
        return pd.bdate_range(start, end, tz="UTC")

    run = briefing_timing(utc("2025-01-01T13:30:00"), weekdays)
    assert run.state == PREMARKET
    assert (run.market_date, run.prior_session) == (date(2025, 1, 1), date(2024, 12, 31))


def test_the_module_uses_the_iana_zone_not_a_fixed_offset():
    assert timing.NEW_YORK.key == "America/New_York"


@pytest.mark.parametrize(
    ("moment", "state", "market_date"),
    [
        ("2026-10-19T23:59:59", AFTER_OPEN, date(2026, 10, 19)),  # 19:59:59 EDT
        ("2026-10-20T00:00:00", ADVANCE, date(2026, 10, 20)),  # 20:00 EDT: the close is in
    ],
)
def test_the_close_availability_boundary_moves_the_briefing_to_the_next_session(
    moment, state, market_date
):
    run = plan(moment)
    assert (run.state, run.market_date) == (state, market_date)
