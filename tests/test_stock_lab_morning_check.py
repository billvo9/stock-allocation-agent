"""
The morning check (stock_lab.morning and scripts/stock_lab_morning_check.py):
on an NYSE session morning, before the 09:30 New York open, a prior session
whose evening has no real forecast yet raises one local notification, while it
can still be recorded. After the open the evening is a permanent gap, and a
late check says so. Weekends and market holidays are never checked, so they
raise no false alarm: the next session morning checks the evening before them
while it can still be recorded. A late check from 20:00 on a session day
reports a gap in the evening before that day, never tonight's (its run is still
to come). A check that cannot read the record or the calendar says that it
could not run. The real NYSE rules calendar, fake stores and a fake notifier:
nothing is shown, sent or written.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab import morning, notify
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.store import HOME_ENV, StoreRefused

ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")


class Store:
    """The two reads the check makes, over {symbol: [(as_of_session, data_class)]}."""

    def __init__(self, made=None, refuse: BaseException | None = None):
        self.made, self.refuse = made or {}, refuse

    def forecast_symbols(self, data_class):
        if self.refuse is not None:
            raise self.refuse
        return tuple(s for s, rows in self.made.items() if any(c == "real" for _, c in rows))

    def forecasts(self, symbol):
        return tuple({"as_of_session": day, "data_class": c} for day, c in self.made[symbol])


def real(*days: str) -> Store:
    return Store({"AMD": [(day, "real") for day in days]})


def ny(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=NEW_YORK)


def check(store: Store, at: str):
    return morning.morning_check(store, now=ny(at), calendar=nyse_sessions)


# --- the check ---


def test_a_recorded_evening_raises_nothing():
    # Thursday 22 October 2026, 08:00: Wednesday's evening has a real forecast.
    found = check(real("2026-10-20", "2026-10-21"), "2026-10-22T08:00")
    assert (found.status, found.notice, str(found.session)) == (
        morning.RECORDED,
        None,
        "2026-10-21",
    )


def test_an_evening_with_no_forecast_is_reported_while_it_can_still_be_recorded():
    found = check(real("2026-10-20"), "2026-10-22T08:00")
    assert (found.status, found.notice, str(found.session)) == (
        morning.NOT_RECORDED,
        "evening_not_recorded",
        "2026-10-21",
    )
    # One stock's forecast is enough: the evening is recorded (as evening_record counts it).
    other = Store({"AMD": [("2026-10-20", "real")], "MU": [("2026-10-21", "real")]})
    assert check(other, "2026-10-22T08:00").status == morning.RECORDED


def test_a_synthetic_forecast_does_not_record_a_real_evening():
    store = Store({"DEMO": [("2026-10-21", "synthetic")], "AMD": [("2026-10-20", "real")]})
    assert check(store, "2026-10-22T08:00").notice == "evening_not_recorded"


def test_an_empty_record_is_reported_rather_than_read_as_nothing_due():
    # A missing store, or one the agent cannot see (another HOME), is not a quiet morning.
    assert check(Store(), "2026-10-22T08:00").notice == "evening_not_recorded"


def test_the_last_minute_before_the_open_is_still_recordable_and_the_open_is_not():
    store = real("2026-10-20")
    assert check(store, "2026-10-22T09:29:59").notice == "evening_not_recorded"
    late = check(store, "2026-10-22T09:30")
    assert (late.status, late.notice, str(late.session)) == (
        morning.MISSED,
        "evening_missed",
        "2026-10-21",
    )
    # A Mac that woke in the afternoon still says so, once.
    assert check(store, "2026-10-22T15:00").notice == "evening_missed"


def test_a_session_days_evening_checks_the_evening_before_it_never_tonights():
    # Thursday 20:15 (a Mac that slept through 08:00): Wednesday's evening is a gap by now.
    late = check(real("2026-10-20"), "2026-10-22T20:15")
    assert (late.status, late.notice, str(late.session)) == (
        morning.MISSED,
        "evening_missed",
        "2026-10-21",
    )
    # Tonight's run is still to come: Thursday's own evening is never checked.
    found = check(real("2026-10-21"), "2026-10-22T23:59")
    assert (found.status, found.notice, str(found.session)) == (
        morning.RECORDED,
        None,
        "2026-10-21",
    )
    # The evening of a Monday after a weekend checks Friday's.
    assert str(check(Store(), "2026-10-26T21:00").session) == "2026-10-23"


@pytest.mark.parametrize(
    "at",
    [
        "2026-10-24T08:00",  # Saturday
        "2026-10-25T08:00",  # Sunday
        "2026-11-26T08:00",  # Thanksgiving: the market is closed
        "2027-01-18T08:00",  # Martin Luther King Jr. Day, a Monday holiday
        "2026-12-25T08:00",  # Christmas, a Friday
        "2027-03-26T08:00",  # Good Friday: a market holiday, not a federal one
        "2026-11-26T21:00",  # Thanksgiving evening: no session closed today
        "2026-10-24T21:00",  # Saturday evening
    ],
    ids=[
        "saturday",
        "sunday",
        "thanksgiving",
        "mlk-monday",
        "christmas",
        "good-friday",
        "thanksgiving-evening",
        "saturday-evening",
    ],
)
def test_weekends_and_market_holidays_raise_no_alarm_even_with_nothing_recorded(at):
    # Nothing is lost: the next session morning checks the same evening, still recordable.
    found = check(Store(), at)
    assert (found.status, found.notice) == (morning.NOT_DUE, None)


@pytest.mark.parametrize(
    "at, prior",
    [
        ("2026-10-26T08:00", "2026-10-23"),  # Monday: Friday's evening, over the weekend
        ("2026-11-27T08:00", "2026-11-25"),  # the Friday after Thanksgiving: Wednesday's
        ("2027-01-19T08:00", "2027-01-15"),  # Tuesday after a Monday holiday: Friday's
        ("2026-12-28T08:00", "2026-12-24"),  # Monday after Christmas: Thursday's
        ("2027-03-29T08:00", "2027-03-25"),  # Monday after Good Friday: Thursday's
    ],
    ids=["monday", "after-thanksgiving", "after-mlk", "after-christmas", "after-good-friday"],
)
def test_the_morning_after_a_closed_day_checks_the_last_session_before_it(at, prior):
    recorded = check(real(prior), at)
    assert (recorded.status, str(recorded.session)) == (morning.RECORDED, prior)
    missing = check(Store(), at)
    assert (missing.notice, str(missing.session)) == ("evening_not_recorded", prior)


@pytest.mark.parametrize(
    "problem",
    [
        ForecastRefused("a stored forecast no longer verifies"),
        StoreRefused("the store is readable by other users"),
    ],
    ids=["forecast-refused", "store-refused"],
)
def test_a_record_that_cannot_be_read_says_the_check_could_not_run(problem):
    found = check(Store(refuse=problem), "2026-10-22T08:00")
    assert (found.status, found.notice, found.reason) == (
        morning.FAILED,
        "check_failed",
        "record_unreadable",
    )


@pytest.mark.parametrize(
    "error", [sqlite3.DatabaseError("file is not a database"), PermissionError("denied")]
)
def test_a_database_or_file_error_says_the_check_could_not_run_and_names_only_its_type(error):
    found = check(Store(refuse=error), "2026-10-22T08:00")
    assert (found.status, found.notice, found.reason) == (
        morning.FAILED,
        "check_failed",
        "record_unreadable",
    )
    assert type(error).__name__ in found.detail and str(error) not in found.detail


@pytest.mark.parametrize("at", ["2031-01-02T08:00", "2030-12-23T08:00"], ids=["after", "near"])
def test_a_moment_beyond_the_calendar_says_the_check_could_not_run(at):
    # The rules calendar ends on 2030-12-31 and the timing looks ten days ahead, so from
    # 2030-12-22 the check says it cannot run (loudly, never a quiet morning) until the
    # calendar is extended.
    found = check(real("2030-12-20"), at)
    assert (found.status, found.notice, found.reason) == (
        morning.FAILED,
        "check_failed",
        "calendar_refused",
    )


def test_every_status_names_a_notice_the_notifier_knows_or_none():
    assert set(morning.NOTICE_FOR) == {
        morning.RECORDED,
        morning.NOT_RECORDED,
        morning.MISSED,
        morning.NOT_DUE,
        morning.FAILED,
    }
    assert set(morning.NOTICE_FOR.values()) - {None} <= set(notify.NOTICES)


# --- the script ---


def _script():
    path = ROOT / "scripts" / "stock_lab_morning_check.py"
    spec = importlib.util.spec_from_file_location("stock_lab_morning_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _script()


class Notifier:
    """Records each notice; reports it shown unless told otherwise. As notify.notify, it
    refuses a notice that does not exist."""

    def __init__(self, shown: bool = True):
        self.notices: list[str] = []
        self.shown = shown

    def __call__(self, notice: str) -> bool:
        if notice not in notify.NOTICES:
            raise ValueError(f"no such notice: {notice!r}")
        self.notices.append(notice)
        return self.shown


def run(*args: str, store=None, at="2026-10-22T08:00", notifier=None, factory=None):
    out = io.StringIO()
    notifier = notifier or Notifier()
    if factory is None:
        factory = (lambda: store) if store is not None else script.ReadOnlyRecords
    moment = at if isinstance(at, datetime) else ny(at)
    code = script.main(
        list(args), store_factory=factory, clock=lambda: moment, notifier=notifier, out=out
    )
    return code, out.getvalue(), notifier.notices


def test_the_script_notifies_once_and_exits_1_when_last_evening_is_not_recorded():
    code, printed, notices = run(store=real("2026-10-20"))
    assert (code, notices) == (1, ["evening_not_recorded"])
    (line,) = printed.splitlines()
    assert "2026-10-21" in line and "not recorded" in line and line.endswith("notified")


def test_the_script_is_silent_and_exits_0_when_last_evening_is_recorded():
    code, printed, notices = run(store=real("2026-10-21"))
    assert (code, notices) == (0, [])
    assert "2026-10-21" in printed and "recorded" in printed


@pytest.mark.parametrize("at", ["2026-10-24T08:00", "2026-11-26T08:00"], ids=["weekend", "holiday"])
def test_the_script_raises_no_alarm_on_a_closed_day(at):
    code, printed, notices = run(store=Store(), at=at)
    assert (code, notices) == (0, [])
    assert "closed" in printed or "nothing to check" in printed


def test_the_script_reports_a_late_check_as_a_permanent_gap():
    code, _, notices = run(store=real("2026-10-20"), at="2026-10-22T10:05")
    assert (code, notices) == (1, ["evening_missed"])


def test_a_check_that_cannot_run_notifies_and_exits_2():
    code, printed, notices = run(store=Store(refuse=StoreRefused("unreadable")))
    assert (code, notices) == (2, ["check_failed"])
    assert "could not run" in printed


def test_a_store_that_refuses_to_open_notifies_and_names_the_reason():
    def refuse():  # as StockLabStore refuses a store folder other users can read
        raise StoreRefused("the store folder is readable by other users")

    code, printed, notices = run(factory=refuse)
    assert (code, notices) == (2, ["check_failed"])
    assert "could not run" in printed and "readable by other users" in printed


@pytest.mark.parametrize(
    ("instant", "code", "notices"),
    [
        # Monday 9 March 2026, the day after DST began (UTC-4): the open is 13:30 UTC.
        (datetime(2026, 3, 9, 13, 29, 59, tzinfo=UTC), 1, ["evening_not_recorded"]),
        (datetime(2026, 3, 9, 13, 30, tzinfo=UTC), 1, ["evening_missed"]),
        # Monday 2 November 2026, the day after DST ended (UTC-5): the open is 14:30 UTC.
        (datetime(2026, 11, 2, 13, 30, tzinfo=UTC), 1, ["evening_not_recorded"]),
        (datetime(2026, 11, 2, 14, 29, 59, tzinfo=UTC), 1, ["evening_not_recorded"]),
        (datetime(2026, 11, 2, 14, 30, tzinfo=UTC), 1, ["evening_missed"]),
    ],
)
def test_the_scripts_utc_clock_meets_the_new_york_open_across_daylight_saving(
    instant, code, notices
):
    assert run(store=Store(), at=instant)[::2] == (code, notices)


def test_an_unexpected_error_still_notifies_and_prints_only_its_type():
    code, printed, notices = run(store=Store(refuse=RuntimeError("planted-secret-value")))
    assert (code, notices) == (2, ["check_failed"])
    assert "RuntimeError" in printed and "planted-secret-value" not in printed


def test_a_notification_that_cannot_be_shown_never_changes_the_outcome():
    code, printed, notices = run(store=real("2026-10-20"), notifier=Notifier(shown=False))
    assert (code, notices) == (1, ["evening_not_recorded"])
    assert "the notification could not be shown" in printed


def test_the_test_notification_shows_only_the_test_notice_and_reads_nothing():
    code, printed, notices = run("--test-notification", store=Store(refuse=RuntimeError("x")))
    assert (code, notices) == (0, ["test"])
    assert "shown" in printed
    code, _, _ = run("--test-notification", notifier=Notifier(shown=False))
    assert code == 1


def test_the_default_store_is_read_only_and_never_created(tmp_path, monkeypatch):
    home = tmp_path / "private"
    monkeypatch.setenv(HOME_ENV, str(home))
    code, _, notices = run()
    assert (code, notices) == (1, ["evening_not_recorded"])
    assert not home.exists()  # nothing was created or written


def test_the_printed_line_holds_no_forecast_value():
    store = Store({"AMD": [("2026-10-21", "real")]})
    store.forecasts = lambda symbol: (  # a full record, as the store returns it
        {"as_of_session": "2026-10-21", "data_class": "real", "p_increase": 0.4123456},
    )
    _, printed, _ = run(store=store)
    assert "0.41" not in printed and "AMD" not in printed
