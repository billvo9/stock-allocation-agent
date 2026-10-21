"""
Missed evenings in the prospective record (stock_lab.job.evening_record) and
how Model Health reads them (health.stock_lab, `forecast.stock_lab_job`).

A session's evening is due once the next session has opened (09:30 New
York): after that its forecast can never be recorded. A due session with no
real forecast is missed, a permanent gap. A miss among the last
RECENT_EVENINGS due evenings degrades the job check; an older one is only
reported. Weekday calendar, fake stores, no network.
"""

from __future__ import annotations

import pandas as pd
import pytest
from stock_lab_inference_helpers import next_day, ny, weekdays

from stock_agent.calendars import nyse_rules
from stock_agent.health import fixtures as health_fixtures
from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import ARTIFACT, stock_lab_checks
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.job import RECENT_EVENINGS, evening_record

FIRST = "2026-10-12"  # a Monday: the first real forecast's as-of session


class Store:
    """The two reads evening_record makes, over {symbol: [(as_of_session, data_class)]}."""

    def __init__(self, made: dict[str, list[tuple[str, str]]], refuse: bool = False):
        self.made, self.refuse = made, refuse

    def forecast_symbols(self, data_class):
        return tuple(s for s, rows in self.made.items() if any(c == "real" for _, c in rows))

    def forecasts(self, symbol):
        if self.refuse:
            raise ForecastRefused("a stored forecast no longer verifies")
        return tuple({"as_of_session": day, "data_class": c} for day, c in self.made[symbol])


def sessions(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    return weekdays(first.date(), last.date())


def record(made, now, expected, **kwargs):
    return evening_record(
        Store(made, **kwargs.pop("store", {})),
        now=now,
        expected=pd.Timestamp(expected, tz="UTC"),
        sessions=kwargs.pop("calendar", sessions),
        **kwargs,
    )


def days(first: str, count: int) -> list[str]:
    return [next_day(first, offset) for offset in range(count)]


def test_nothing_is_due_before_the_first_real_forecast():
    assert record({}, ny("2026-10-16", "20:35"), "2026-10-16") is None
    synthetic_only = {"DEMO": [(FIRST, "synthetic")]}
    assert record(synthetic_only, ny("2026-10-16", "20:35"), "2026-10-16") is None


def test_every_evening_recorded_misses_nothing_and_tonight_is_not_due_yet():
    week = days(FIRST, 5)  # Monday to Friday, each recorded by one stock or another
    made = {"AMD": [(day, "real") for day in week[::2]], "MU": [(d, "real") for d in week[1::2]]}
    assert record(made, ny(week[-1], "20:35"), week[-1]) == {
        "since": FIRST,
        "due": 4,  # Friday's evening stays open until Monday's 09:30 open
        "recorded": 4,
        "missed": [],
        "recent": RECENT_EVENINGS,
        "recent_missed": 0,
    }


def test_a_skipped_evening_is_missed_once_the_next_session_opens():
    monday, tuesday, wednesday = days(FIRST, 3)
    made = {"AMD": [(monday, "real"), (wednesday, "real")], "NVDA": [(monday, "real")]}
    found = record(made, ny(wednesday, "20:35"), wednesday)
    assert (found["due"], found["recorded"], found["missed"]) == (2, 1, [tuesday])
    assert found["recent_missed"] == 1
    # Tuesday evening at 21:00 (Tuesday not recorded yet): still open, so not missed.
    early = record({"AMD": [(monday, "real")]}, ny(tuesday, "21:00"), tuesday)
    assert (early["due"], early["missed"]) == (1, [])
    # Wednesday at 09:29 New York, the last minute it could still be recorded, then 09:30.
    late = record({"AMD": [(monday, "real")]}, ny(wednesday, "09:29"), tuesday)
    assert late["missed"] == []
    late = record({"AMD": [(monday, "real")]}, ny(wednesday, "09:30"), tuesday)
    assert late["missed"] == [tuesday]


def test_fridays_evening_stays_open_over_the_weekend():
    friday = next_day(FIRST, 4)
    made = {"AMD": [(day, "real") for day in days(FIRST, 4)]}  # Friday not recorded
    saturday = pd.Timestamp(friday) + pd.Timedelta(days=1)
    found = record(made, ny(saturday.strftime("%Y-%m-%d"), "10:00"), friday)
    assert found["missed"] == [] and found["due"] == 4
    monday = next_day(friday)
    found = record(made, ny(monday, "09:30"), friday)
    assert found["missed"] == [friday]


def test_an_old_miss_stays_listed_but_is_no_longer_recent():
    sessions_made = days(FIRST, RECENT_EVENINGS + 3)
    missed = sessions_made[1]
    made = {"AMD": [(day, "real") for day in sessions_made if day != missed]}
    last = sessions_made[-1]
    found = record(made, ny(last, "20:35"), last)
    assert found["missed"] == [missed] and found["due"] == RECENT_EVENINGS + 2
    assert found["recent_missed"] == 0  # older than the last RECENT_EVENINGS due evenings
    shorter = record(made, ny(last, "20:35"), last, recent=RECENT_EVENINGS + 1)
    assert shorter["recent_missed"] == 1


def test_a_calendar_that_ends_leaves_the_last_session_unjudged():
    week = days(FIRST, 5)
    made = {"AMD": [(week[0], "real")]}

    def ending(first, last):
        if last > pd.Timestamp(week[-1], tz="UTC"):
            raise ValueError("outside_coverage: beyond the calendar")
        return sessions(first, last)

    found = record(made, ny(next_day(week[-1], 3), "12:00"), week[-1], calendar=ending)
    assert found["missed"] == week[1:-1]  # Friday is not judged: its next open is unknown


def test_synthetic_forecasts_never_fill_a_real_evening():
    monday, tuesday, wednesday = days(FIRST, 3)
    made = {"AMD": [(monday, "real"), (tuesday, "synthetic"), (wednesday, "real")]}
    found = record(made, ny(wednesday, "20:35"), wednesday)
    assert found["missed"] == [tuesday] and found["recorded"] == 1


def nyse(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    return nyse_rules.sessions(first.date(), last.date())


def test_on_the_nyse_calendar_a_holiday_keeps_the_evening_open_until_the_next_session():
    # Thanksgiving, Thursday 2026-11-26, is no session: Wednesday's evening stays open until
    # Friday's 09:30 open, and Thursday itself is never due.
    assert not nyse_rules.is_session("2026-11-26")
    made = {"AMD": [("2026-11-24", "real")]}
    open_ = record(made, ny("2026-11-26", "10:00"), "2026-11-25", calendar=nyse)
    assert (open_["due"], open_["missed"]) == (1, [])
    closed = record(made, ny("2026-11-27", "09:30"), "2026-11-25", calendar=nyse)
    assert (closed["due"], closed["missed"]) == (2, ["2026-11-25"])


def test_the_end_of_the_nyse_calendars_coverage_leaves_its_last_session_unjudged():
    made = {"AMD": [("2030-12-27", "real")]}
    found = record(made, pd.Timestamp("2031-01-03T15:00:00Z"), "2030-12-31", calendar=nyse)
    assert found["missed"] == ["2030-12-30"]  # 12-31's next open is beyond 2030: not judged


@pytest.mark.parametrize(
    ("now", "missed"),
    [
        # Friday 2026-10-30's evening closes at Monday 2026-11-02 09:30 New York, which is
        # 14:30 UTC: the clocks went back on Sunday 1 November (13:30 UTC the week before).
        ("2026-11-02T13:30:00Z", []),
        ("2026-11-02T14:29:59Z", []),
        ("2026-11-02T14:30:00Z", ["2026-10-30"]),
    ],
)
def test_the_open_that_closes_an_evening_follows_new_yorks_clock_change(now, missed):
    made = {"AMD": [("2026-10-29", "real")]}
    found = record(made, pd.Timestamp(now), "2026-10-30", calendar=nyse)
    assert found["missed"] == missed


def test_a_store_whose_forecasts_no_longer_verify_records_no_evenings():
    made = {"AMD": [(FIRST, "real")]}
    assert record(made, ny(FIRST, "20:35"), FIRST, store={"refuse": True}) is None


# --- Model Health ----------------------------------------------------------------------


def _job(tmp_path, evenings, **job):
    document = health_fixtures.stock_lab_monitor()
    document["evenings"] = evenings
    document["job"].update(job)
    written = health_fixtures.write_input(tmp_path, document).parent
    checks = {c.check_id: c for c in stock_lab_checks(written, allow_synthetic=True)}
    return checks["forecast.stock_lab_job"], checks


EVENINGS = {"since": "2026-10-12", "due": 8, "recorded": 8, "missed": [], "recent": 10,
            "recent_missed": 0}  # fmt: skip


def test_a_recent_miss_degrades_the_job_check_and_names_the_evening(tmp_path):
    evenings = {**EVENINGS, "recorded": 6, "missed": ["2026-10-13", "2026-10-19"],
                "recent_missed": 2}  # fmt: skip
    job, _ = _job(tmp_path, evenings)
    assert (job.state, job.reason) == (HealthState.DEGRADED, "evening_missed")
    assert "2 of the last 8 closed evenings recorded no forecast (latest 2026-10-19)" in job.summary
    assert "permanent gaps" in job.summary and "2 missed since 2026-10-12" in job.summary
    assert "evenings since 2026-10-12: 6 of 8 recorded" in job.detail
    assert "2026-10-13, 2026-10-19" in job.detail


def test_an_old_miss_leaves_the_job_healthy_but_stays_in_the_detail(tmp_path):
    evenings = {**EVENINGS, "due": 30, "recorded": 29, "missed": ["2026-10-13"]}
    job, _ = _job(tmp_path, evenings)
    assert (job.state, job.reason) == (HealthState.HEALTHY, None)
    assert "29 of 30 recorded; missed (latest 5): 2026-10-13" in job.detail


def test_a_failed_job_outranks_a_missed_evening(tmp_path):
    evenings = {**EVENINGS, "recorded": 7, "missed": ["2026-10-19"], "recent_missed": 1}
    job, _ = _job(tmp_path, evenings, status="failed", reason="AMD: after_next_open")
    assert (job.state, job.reason) == (HealthState.FAILED, "job_failed")
    assert "2026-10-19" in job.detail


def test_a_file_without_evenings_keeps_every_verdict(tmp_path):
    plain = health_fixtures.stock_lab_monitor()
    plain.pop("evenings", None)
    clean = {**plain, "evenings": EVENINGS}
    found = []
    for name, document in (("plain", plain), ("clean", clean)):
        written = health_fixtures.write_input(tmp_path / name, document).parent
        found.append(
            {
                c.check_id: (c.state, c.reason)
                for c in stock_lab_checks(written, allow_synthetic=True)
            }
        )
    assert found[0] == found[1]


@pytest.mark.parametrize(
    "evenings",
    [
        {**EVENINGS, "recorded": 7},  # recorded and missed do not add up to due
        {**EVENINGS, "missed": ["2026-10-13"]},
        {**EVENINGS, "recent_missed": 1},  # more recent misses than misses
        {**EVENINGS, "missed": "2026-10-13"},
        {**EVENINGS, "missed": [None], "recorded": 7},
        {**EVENINGS, "missed": ["19 October"], "recorded": 7, "recent_missed": 1},
        {**EVENINGS, "missed": ["2026-10-19", "2026-10-13"], "recorded": 6, "recent_missed": 2},
        {**EVENINGS, "missed": ["2026-10-13", "2026-10-13"], "recorded": 6, "recent_missed": 2},
        {**EVENINGS, "missed": ["2026-10-09"], "recorded": 7, "recent_missed": 1},  # before since
        {**EVENINGS, "since": "last week"},
        {**EVENINGS, "due": -1},
        {**EVENINGS, "due": 8.5},
        {key: value for key, value in EVENINGS.items() if key != "since"},
        "every evening",
    ],
)
def test_malformed_evenings_make_the_file_unrecognized(tmp_path, evenings):
    _, checks = _job(tmp_path, evenings)
    assert {(c.state, c.reason) for c in checks.values()} == {
        (HealthState.UNAVAILABLE, "unrecognized_format")
    }


def test_the_smoke_fixture_records_evenings_with_none_missed():
    evenings = health_fixtures.stock_lab_monitor()["evenings"]
    assert evenings["missed"] == [] and evenings["recorded"] == evenings["due"] > 0
    assert ARTIFACT == "stock_lab_monitor"
