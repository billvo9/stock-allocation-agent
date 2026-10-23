"""The Lab app's words and number formats (docs/product/lab_app_design.md)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from stock_agent.app.web import words

NEW_YORK = ZoneInfo("America/New_York")
SOURCE = Path(__file__).resolve().parents[1] / "src"
LONDON = ZoneInfo("Europe/London")


def ny(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=NEW_YORK)


def utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_percent_rounds_for_display_only():
    assert words.percent(0.4712) == "47%"
    assert words.percent(0.0) == "0%" and words.percent(1.0) == "100%"
    assert words.percent(0.004) == "<1%" and words.percent(0.996) == ">99%"
    assert words.percent(0.005) == "<1%" and words.percent(0.9951) == ">99%"
    assert words.percent(None) == "—"
    assert words.percent(float("nan")) == "—"


def test_signed_percent_uses_a_true_minus_and_two_decimals():
    assert words.signed_percent(0.04876) == "+4.88%"
    assert words.signed_percent(-0.0032) == "−0.32%"
    assert words.signed_percent(0.0) == "0.00%"
    assert words.signed_percent(None) == "—"


def test_money():
    assert words.money(1234.5) == "$1,234.50"
    assert words.money(None) == "—"


def test_days():
    assert words.short_day(date(2026, 10, 21)) == "Wed, Oct 21"
    assert words.short_day("2026-10-21") == "Wed, Oct 21"
    assert words.weekday("2026-10-22") == "Thu"
    assert words.long_weekday("2026-10-22") == "Thursday"


@pytest.mark.parametrize(
    ("stamp", "now", "expected"),
    [
        ("2026-10-21T20:31", "2026-10-22T07:40", "last night, 8:31 PM"),
        ("2026-10-22T20:31", "2026-10-22T21:00", "tonight, 8:31 PM"),
        ("2026-10-22T06:02", "2026-10-22T07:40", "this morning, 6:02 AM"),
        ("2026-10-22T13:10", "2026-10-22T15:00", "today, 1:10 PM"),
        ("2026-10-21T14:10", "2026-10-22T07:40", "yesterday, 2:10 PM"),
        ("2026-10-16T20:31", "2026-10-19T07:40", "Friday, 8:31 PM"),
        ("2026-10-09T20:31", "2026-10-22T07:40", "Oct 9, 8:31 PM"),
    ],
)
def test_when_says_last_night_only_when_it_was(stamp, now, expected):
    assert words.when(utc_stamp(ny(stamp)), now=ny(now), local_zone=NEW_YORK) == expected


def test_when_names_the_zone_only_when_this_mac_is_elsewhere():
    stamp = utc_stamp(ny("2026-10-21T20:31"))
    assert words.when(stamp, now=ny("2026-10-22T07:40"), local_zone=LONDON).endswith(" ET")
    assert not words.when(stamp, now=ny("2026-10-22T07:40"), local_zone=NEW_YORK).endswith("ET")


def test_company_names_cover_the_watchlist():
    assert words.NAMES == {
        "AMD": "Advanced Micro Devices",
        "MRVL": "Marvell Technology",
        "MU": "Micron Technology",
        "NVDA": "NVIDIA",
    }


@pytest.mark.parametrize(
    ("moment", "stamp"),
    [
        ("2026, 12, 2, 7, 40", "2026-12-02T01:31:00.000000Z"),  # winter: 8:31 PM is 01:31 UTC
        ("2026, 7, 2, 7, 40", "2026-07-02T00:31:00.000000Z"),  # summer: 8:31 PM is 00:31 UTC
    ],
)
@pytest.mark.parametrize(("zone", "named"), [("America/New_York", False), ("Europe/London", True)])
def test_this_macs_own_zone_uses_the_offset_in_force_at_each_moment(zone, named, moment, stamp):
    # Whatever the season the test runs in, one of these moments is in the other season, so
    # a fixed offset captured "now" would name the zone for a Mac in New York. None must not.
    script = (
        "from datetime import datetime\n"
        "from zoneinfo import ZoneInfo\n"
        "from stock_agent.app.web import words\n"
        "ny = ZoneInfo('America/New_York')\n"
        f"now = datetime({moment}, tzinfo=ny)\n"
        f"print(words.when('{stamp}', now=now, local_zone=None))\n"
    )
    env = {**os.environ, "TZ": zone, "PYTHONPATH": str(SOURCE)}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "last night, 8:31 PM" + (" ET" if named else "")


def test_new_york_days_decide_tonight_even_when_this_mac_is_already_in_tomorrow():
    stamp = utc_stamp(ny("2026-10-22T20:31"))
    assert words.when(stamp, now=ny("2026-10-22T21:00"), local_zone=LONDON) == (
        "tonight, 8:31 PM ET"
    )


def test_checked_names_the_time_the_app_read_the_records():
    assert words.checked(ny("2026-10-22T07:40"), NEW_YORK) == "Checked 7:40 AM"
    assert words.checked(ny("2026-10-22T07:40"), LONDON) == "Checked 7:40 AM ET"


# --- Stock detail ---


@pytest.mark.parametrize(
    ("direction", "length", "text"),
    [
        ("increase", 3, "3 rising days in a row"),
        ("decrease", 1, "1 falling day"),
        ("stable", 2, "2 flat days in a row"),
        (None, 0, "No streak after a gap"),
        ("increase", 0, "No streak after a gap"),
    ],
)
def test_a_run_in_words(direction, length, text):
    assert words.run_chip(direction, length) == text


def test_whose_past_tests():
    assert words.stocks_words(["AMD", "MRVL", "MU", "NVDA"]) == "AMD, MRVL, MU and NVDA"
    assert words.stocks_words(list("ABCDEF")) == "6 stocks"
    assert words.development_scope(["AMD", "NVDA"], "2024-12-31", False) == (
        " on AMD and NVDA together, through 2024"
    )
    assert words.development_scope(["AMD"], None, False) == ""
    assert words.development_scope([], None, True) == " on the synthetic demo stocks"


def test_ranges_of_chances_and_of_days():
    assert words.about(0.4, 0.5) == "about 40–50%"
    assert words.happened(0.4712, 0.4499, 0.4901) == "47% (likely 45–49%)"
    assert words.range_note("2026-10-14", "2026-10-21", shown=6, asked=6) == (
        "Daily closes from Wed, Oct 14 to Wed, Oct 21."
    )
    assert words.range_note("2026-10-19", "2026-10-21", shown=3, asked=64) == (
        "Only 3 trading days are stored: Mon, Oct 19 to Wed, Oct 21."
    )


def test_a_day_of_the_chart_in_words():
    assert words.point_when("2026-10-21", [], latest=True) == "on Wed, Oct 21 (latest close)"
    assert words.point_label("2026-10-21", "$1.00", "—", ["split"]) == (
        "Wed, Oct 21: closed at $1.00; split."
    )
    assert words.day_and_date("2026-10-26") == "Mon 26"
    assert words.open_label(
        "NVDA", None, {"available": False, "words": "No prices stored yet."}
    ) == ("NVDA. No prices stored yet. Open details.")


def test_each_range_names_its_visible_label_first():
    # WCAG 2.5.3: the accessible name starts with the visible "1W", "1M", "3M" or "1Y".
    for _key, label, _moves, name in words.RANGES:
        assert name.startswith(f"{label}, ")
    assert [moves for *_, moves, _name in words.RANGES] == [5, 21, 63, 251]
