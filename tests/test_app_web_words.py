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
    assert words.percent(0.0) == "0%"
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
