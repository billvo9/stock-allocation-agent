"""
The Lab app's Home view model (docs/product/lab_app_design.md, "Home in
detail"): the status line, each stock's price as the recent-prices reader
gave it, and the next session's outlook exactly as recorded, with an honest
state for everything missing. Fixtures only: a Stock Lab store on a simulated
clock (real-class records use the made-up ticker SYNTH) and a stand-in for
the recent-prices reader. No network, no private store.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, ny
from test_app_destinations import scan_for_advice

from stock_agent.app.web import home
from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.store import StoreRefused

NEW_YORK = ZoneInfo("America/New_York")
T = REAL_TEST_TICKER
TUESDAY, WEDNESDAY, THURSDAY = "2026-10-20", "2026-10-21", "2026-10-22"
P = (0.42, 0.11, 0.47)  # Decrease, Stable, Increase (the helper's order)


def at(day: str, hhmm: str) -> datetime:
    return ny(day, hhmm).tz_convert(NEW_YORK).to_pydatetime()


def row(session, close, change, *, level=None, notes=()):
    return SimpleNamespace(
        session=session,
        close=close,
        level=close if level is None else level,
        change=change,
        notes=tuple(notes),
    )


class Prices:
    """A stand-in for the recent-prices reader: what it was asked, and fixed answers."""

    def __init__(self, rows=None, state="observed", reason=None):
        self.calls = []
        self.rows = (
            rows
            if rows is not None
            else [
                row(TUESDAY, 121.70, -0.0032),
                row(WEDNESDAY, 127.65, 0.0488),
            ]
        )
        self.state, self.reason = state, reason

    def __call__(self, symbol, *, count, retrieved_by):
        self.calls.append((symbol, count, retrieved_by))
        observed = self.state == "observed"
        sessions = tuple(self.rows) if observed else ()
        return SimpleNamespace(
            ticker=symbol,
            state=self.state,
            reason=self.reason,
            sessions=sessions,
            last_session=sessions[-1].session if sessions else None,
            retrieved_at="2026-10-22T00:31:00.000000Z" if observed else None,
        )


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def evening(lab, as_of, *, p=P, status="ok", ran=None, **forecast):
    """The evening job as of `as_of`: its forecast at 20:31 and its run at 20:32."""

    record = lab.forecast(T, as_of, created_at=ny(as_of, "20:31"), p=p, **forecast)
    lab.job(at=ran or ny(as_of, "20:32"), expected=as_of, status=status, activation=None)
    return record


def view(lab, now, *, prices=None, local_zone=NEW_YORK, records=None, data_class="real"):
    return home.build_home(
        records or ReadOnlyRecords(lab.path),
        now=now,
        calendar=nyse_sessions,
        calendar_id="test_nyse_rules",
        data_class=data_class,
        recent=prices or Prices(),
        local_zone=local_zone,
        tickers=[T],
    )


def only(found):
    (stock,) = found["stocks"]
    return stock


# --- the outlook chip: exactly as recorded ---


def test_a_fresh_outlook_shows_the_recorded_probabilities_exactly(lab):
    evening(lab, WEDNESDAY)
    outlook = only(view(lab, at(THURSDAY, "07:40")))["outlook"]
    stored = ReadOnlyRecords(lab.path).forecasts(T)[0]["daily"][0]
    assert outlook["state"] == "fresh" and outlook["level"] == "plain"
    assert outlook["day"] == "Thu" and outlook["for_session"] == THURSDAY
    assert outlook["chip"] == {"up": "47%", "down": "42%"}
    assert outlook["recorded"] == {
        "increase": stored["p_increase"],
        "stable": stored["p_stable"],
        "decrease": stored["p_decrease"],
    }
    sheet = outlook["sheet"]
    assert sheet["title"] == f"{T}: Thursday's outlook"
    assert sheet["rows"] == [
        ["Up more than 0.5%", "47%"],
        ["Flat, within 0.5%", "11%"],
        ["Down more than 0.5%", "42%"],
    ]
    text = " ".join(sheet["paragraphs"])
    assert "Thu, Oct 22" in text and "Wednesday's close" in text
    assert "last night, 8:31 PM" in text and "news" in text and "not advice" in text


def test_the_band_comes_from_the_record(lab):
    evening(lab, WEDNESDAY, band=0.0025)
    rows = only(view(lab, at(THURSDAY, "07:40")))["outlook"]["sheet"]["rows"]
    assert rows[0][0] == "Up more than 0.25%" and rows[1][0] == "Flat, within 0.25%"


def test_from_eight_in_the_evening_the_next_outlook_waits_for_tonight(lab):
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "20:15"))
    outlook = only(found)["outlook"]
    assert outlook["state"] == "pending" and outlook["level"] == "neutral"
    assert outlook["words"] == "Fri outlook after tonight's update"
    assert outlook["chip"] is None and outlook["recorded"] is None
    assert found["status"]["level"] == "ok"


def test_a_missed_evening_is_named_and_its_old_outlook_is_not_shown_as_today(lab):
    evening(lab, TUESDAY)
    found = view(lab, at(THURSDAY, "07:40"))
    outlook = only(found)["outlook"]
    assert outlook["state"] == "stale" and outlook["level"] == "warn"
    assert outlook["words"] == "Thu outlook not recorded"
    assert outlook["chip"] is None and outlook["recorded"] is None
    assert found["status"] == {
        "level": "warn",
        "words": "Last night's update didn't run.",
        "details": found["status"]["details"],
    }


def test_an_evening_that_ran_but_left_a_stock_out_says_so(lab):
    evening(lab, TUESDAY)
    lab.job(at=ny(WEDNESDAY, "20:32"), expected=WEDNESDAY, status="ok", activation=None)
    found = view(lab, at(THURSDAY, "07:40"))
    assert found["status"]["level"] == "warn"
    assert found["status"]["words"] == "Last night's update didn't record 1 of 1 stocks."


@pytest.mark.parametrize(
    ("status", "words"),
    [
        ("failed", "The last update didn't finish (last night, 8:40 PM)."),
        ("partial", "The last update was only partly done (last night, 8:40 PM)."),
    ],
)
def test_a_failed_or_partial_run_after_the_last_success_is_amber(lab, status, words):
    evening(lab, TUESDAY)
    lab.job(at=ny(WEDNESDAY, "20:40"), expected=WEDNESDAY, status=status, activation=None)
    found = view(lab, at(THURSDAY, "07:40"))
    assert (found["status"]["level"], found["status"]["words"]) == ("warn", words)


def test_a_failed_ledger_check_is_red_and_withholds_every_outlook(lab, monkeypatch):
    evening(lab, WEDNESDAY)
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    found = view(lab, at(THURSDAY, "07:40"))
    outlook = only(found)["outlook"]
    assert found["status"]["level"] == "bad"
    assert found["status"]["words"] == "The forecast record didn't pass its check."
    assert outlook["state"] == "unverified" and outlook["level"] == "bad"
    assert outlook["words"] == "Outlook can't be verified" and outlook["recorded"] is None


def test_a_record_that_fails_its_check_is_red(lab, monkeypatch):
    evening(lab, WEDNESDAY)

    def refuse(self, symbol):
        raise ForecastRefused("the forecast does not match its id")

    monkeypatch.setattr(ReadOnlyRecords, "forecasts", refuse)
    found = view(lab, at(THURSDAY, "07:40"))
    assert found["status"]["level"] == "bad"
    assert found["status"]["words"] == "A forecast record didn't pass its check."
    assert only(found)["outlook"]["state"] == "unverified"


def test_a_withheld_outlook_turns_the_status_amber(lab):
    targets = ["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"]
    evening(lab, WEDNESDAY, targets=targets)
    found = view(lab, at(THURSDAY, "07:40"))
    assert (found["status"]["level"], found["status"]["words"]) == (
        "warn",
        "1 outlook was withheld.",
    )


def test_a_skipped_run_after_the_last_success_is_amber(lab):
    evening(lab, TUESDAY)
    lab.job(at=ny(WEDNESDAY, "20:40"), expected=WEDNESDAY, status="skipped", activation=None)
    found = view(lab, at(THURSDAY, "07:40"))
    assert (found["status"]["level"], found["status"]["words"]) == (
        "warn",
        "The last update recorded nothing (last night, 8:40 PM).",
    )


def test_once_tonights_run_has_finished_a_missing_outlook_is_no_longer_pending(lab):
    evening(lab, WEDNESDAY)
    lab.job(at=ny(THURSDAY, "20:32"), expected=THURSDAY, status="ok", activation=None)
    found = view(lab, at(THURSDAY, "20:45"))
    outlook = only(found)["outlook"]
    assert outlook["state"] == "stale" and outlook["words"] == "Fri outlook not recorded"
    assert (found["status"]["level"], found["status"]["words"]) == (
        "warn",
        "Tonight's update didn't record 1 of 1 stocks.",
    )


def test_before_activation_the_status_is_neutral(lab):
    lab.job(at=ny(WEDNESDAY, "20:32"), expected=WEDNESDAY, status="skipped")
    found = view(lab, at(THURSDAY, "07:40"))
    assert (found["status"]["level"], found["status"]["words"]) == (
        "neutral",
        "Outlooks haven't started yet.",
    )
    assert only(found)["outlook"]["state"] == "none"


def test_before_any_run_the_status_is_neutral(lab):
    lab.forecast(T, WEDNESDAY, created_at=ny(WEDNESDAY, "20:31"), p=P)
    found = view(lab, at(THURSDAY, "07:40"))
    assert found["status"]["level"] == "neutral"
    assert found["status"]["words"] == "No update has run yet."


def test_a_weekend_names_the_day_of_the_last_update(lab):
    evening(lab, "2026-10-16")
    found = view(lab, at("2026-10-18", "10:00"))
    assert found["status"] == {
        "level": "ok",
        "words": "Updated Friday, 8:32 PM",
        "details": found["status"]["details"],
    }
    assert only(found)["outlook"]["day"] == "Mon"


def test_the_zone_is_named_when_this_mac_is_elsewhere(lab):
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "07:40"), local_zone=ZoneInfo("Europe/London"))
    assert found["status"]["words"] == "Updated last night, 8:32 PM ET"


@pytest.mark.parametrize(
    ("record", "state", "level", "words"),
    [
        ({}, "none", "neutral", "No outlook yet"),
        (
            {"refusal": {"reason": "too_few_sessions", "detail": "too few sessions"}},
            "refused",
            "neutral",
            "No outlook: the model gave none",
        ),
        (
            {"targets": ["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"]},
            "withheld",
            "warn",
            "Outlook withheld",
        ),
    ],
)
def test_every_missing_outlook_has_an_honest_state(lab, record, state, level, words):
    if record:
        evening(lab, WEDNESDAY, **record)
    else:
        lab.job(at=ny(WEDNESDAY, "20:32"), expected=WEDNESDAY, status="ok", activation=None)
    outlook = only(view(lab, at(THURSDAY, "07:40")))["outlook"]
    assert (outlook["state"], outlook["level"], outlook["words"]) == (state, level, words)
    assert outlook["chip"] is None and outlook["recorded"] is None
    assert outlook["sheet"]["paragraphs"]  # the reason, in words


def test_a_store_that_cannot_be_read_keeps_the_prices_and_withholds_the_outlooks(lab):
    class Refusing(ReadOnlyRecords):
        def job_runs(self):
            raise StoreRefused("not a Stock Lab store")

    found = view(lab, at(THURSDAY, "07:40"), records=Refusing(lab.path))
    stock = only(found)
    assert found["status"]["level"] == "bad"
    assert found["status"]["words"] == "The forecast record can't be read."
    assert stock["outlook"]["state"] == "unverified"
    assert stock["price"]["available"] and stock["price"]["close"] == 127.65


# --- the price: as the reader gave it (decision A6) ---


def test_the_price_is_the_readers_close_and_change_unchanged(lab):
    evening(lab, WEDNESDAY)
    prices = Prices()
    now = at(THURSDAY, "07:40")
    found = view(lab, now, prices=prices)
    price = only(found)["price"]
    assert prices.calls == [(T, home.SPARK_SESSIONS, pd.Timestamp(now))]
    assert price["available"] and price["words"] is None
    assert (price["close"], price["session"], price["change"]) == (127.65, WEDNESDAY, 0.0488)
    assert price["close_words"] == "$127.65" and price["change_words"] == "+4.88%"
    assert price["direction"] == "up" and price["session_words"] == "Wed, Oct 21"
    assert price["line"] == [121.70, 127.65]
    assert found["prices_note"] == "Closing prices from Wed, Oct 21."
    assert only(found)["name"] is None  # SYNTH has no company name


def test_nothing_is_derived_from_the_prices(lab):
    # Sentinels: these changes do not match the closes, so any value the view
    # computed from two prices would differ from every value the reader gave.
    rows = [row(TUESDAY, 100.0, 0.0101, level=50.0), row(WEDNESDAY, 200.0, -0.0202, level=60.0)]
    price = only(view(lab, at(THURSDAY, "07:40"), prices=Prices(rows)))["price"]
    given = {100.0, 0.0101, 50.0, 200.0, -0.0202, 60.0}
    numbers = {v for v in price.values() if isinstance(v, float)} | set(price["line"])
    assert numbers <= given
    assert price["direction"] == "down" and price["change_words"] == "−2.02%"


def test_a_newest_session_without_a_close_shows_the_last_served_close_and_a_gap(lab):
    rows = [
        row(TUESDAY, 121.70, -0.0032),
        row(WEDNESDAY, None, None, level=None, notes=("gap",)),
    ]
    found = view(lab, at(THURSDAY, "07:40"), prices=Prices(rows))
    price = only(found)["price"]
    assert (price["close"], price["session"], price["change"]) == (121.70, TUESDAY, -0.0032)
    assert price["session_words"] == "Tue, Oct 20" and price["line"] == [121.70, None]


@pytest.mark.parametrize(
    ("state", "words"),
    [
        ("no_store", "No prices stored yet."),
        ("no_data", "No prices stored for SYNTH yet."),
        ("unreadable", "Prices can't be read right now."),
    ],
)
def test_missing_prices_say_why_and_show_no_number(lab, state, words):
    found = view(lab, at(THURSDAY, "07:40"), prices=Prices(state=state, reason="x"))
    price = only(found)["price"]
    assert not price["available"] and price["words"] == words
    assert price["close"] is None and price["change"] is None and price["line"] == []
    assert price["direction"] == "none" and found["prices_note"] is None


def test_a_change_of_zero_or_unknown_has_no_colour(lab):
    rows = [row(TUESDAY, 121.70, None), row(WEDNESDAY, 121.70, 0.0)]
    assert only(view(lab, at(THURSDAY, "07:40"), prices=Prices(rows)))["price"]["direction"] == (
        "flat"
    )
    rows = [row(WEDNESDAY, 121.70, None)]
    price = only(view(lab, at(THURSDAY, "07:40"), prices=Prices(rows)))["price"]
    assert price["direction"] == "none" and price["change_words"] == "—"


# --- the whole view ---


def test_the_view_is_strict_json_with_no_advice(lab):
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "07:40"))
    text = json.dumps(found, allow_nan=False)
    assert found["schema"] == "lab_app_home" and found["version"] == 1
    assert found["data_class"] == "real" and found["demo"] is None
    scan_for_advice(text)
    assert "not advice" in found["footer"] and "bought or sold" in found["footer"]
    assert found["explain"]["paragraphs"]


def test_a_synthetic_view_says_it_is_a_demo(lab):
    lab.forecast("DEMO_CALM", WEDNESDAY, created_at=ny(WEDNESDAY, "20:31"), p=P)
    found = home.build_home(
        ReadOnlyRecords(lab.path),
        now=at(THURSDAY, "07:40"),
        calendar=nyse_sessions,
        calendar_id="test_nyse_rules",
        data_class="synthetic",
        recent=Prices(),
        local_zone=NEW_YORK,
        tickers=["DEMO_CALM"],
    )
    assert found["demo"] and "made-up" in found["demo"].lower()
    assert only(found)["outlook"]["state"] == "fresh"
