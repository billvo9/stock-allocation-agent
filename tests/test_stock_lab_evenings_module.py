"""
The evening rule's one copy (stock_agent.stock_lab.evenings, moved unchanged from
stock_lab.job on 2026-10-23): the job re-exports the very same functions, and the
rule applied to a set of recorded evenings gives what the job records.
"""

from __future__ import annotations

import pandas as pd

from stock_agent.stock_lab import evenings, job


def weekdays(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, tz="UTC")


def ny(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC")


def test_the_job_re_exports_the_same_rule():
    assert job.evening_record is evenings.evening_record
    assert job.recorded_evenings is evenings.recorded_evenings
    assert job.RECENT_EVENINGS == evenings.RECENT_EVENINGS == 10
    assert job.Sessions is evenings.Sessions


def test_the_rule_on_a_set_of_recorded_evenings():
    found = evenings.evenings_from(
        {"2026-10-19", "2026-10-21"},
        now=ny("2026-10-22", "10:00"),  # Thursday after the open: Wednesday's evening is due
        expected=pd.Timestamp("2026-10-22", tz="UTC"),
        sessions=weekdays,
    )
    assert found == {
        "since": "2026-10-19",
        "due": 3,
        "recorded": 2,
        "missed": ["2026-10-20"],
        "recent": 10,
        "recent_missed": 1,
    }


def test_an_evening_is_not_due_before_the_next_open():
    found = evenings.evenings_from(
        {"2026-10-19"},
        now=ny("2026-10-20", "09:29"),
        expected=pd.Timestamp("2026-10-20", tz="UTC"),
        sessions=weekdays,
    )
    assert (found["due"], found["missed"]) == (0, [])


def test_nothing_recorded_means_no_record():
    assert evenings.evenings_from(set(), now=ny("2026-10-22", "10:00"),
                                  expected=pd.Timestamp("2026-10-22", tz="UTC"),
                                  sessions=weekdays) is None  # fmt: skip
