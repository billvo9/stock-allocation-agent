"""
The vintage-aware rates store (C2): availability by the h15_or_alfred_later rule,
append-only vintages, first release for booked wealth, as-of reads for decisions,
the 2024-12-31 vintage limit, and private storage. No network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_agent.data.rates.vintages import (
    H15_OR_ALFRED_LATER,
    PRE_ALFRED,
    VintageRefused,
    as_of,
    available_at,
    diagnostics,
    fed_holidays,
    first_release,
    legacy_rates,
    load_vintages,
    merge_vintages,
    next_fed_business_day,
    save_vintages,
    vintages_from_alfred,
)

RETRIEVED = pd.Timestamp("2026-10-27T03:00:00Z")
NY = "America/New_York"


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


def alfred(rows, retrieved=RETRIEVED) -> pd.DataFrame:
    """(date, value, realtime_start) rows, as FRED/ALFRED serves them."""

    observations = pd.DataFrame(rows, columns=["date", "value", "realtime_start"])
    return vintages_from_alfred(
        observations,
        rate_id="dgs3mo",
        provider_series_id="DGS3MO",
        currency="USD",
        tenor="3M",
        quote_convention="investment_basis",
        retrieved_at=retrieved,
    )


# --- availability ---


def test_the_columbus_day_example():
    # Thursday 2024-10-10 at 5.00%, posted Friday; Friday 2024-10-11 at 5.10%, posted
    # Tuesday 2024-10-15 (Monday is Columbus Day: an NYSE session, a Fed holiday).
    assert date(2024, 10, 14) in fed_holidays(2024)
    assert next_fed_business_day(date(2024, 10, 11)) == date(2024, 10, 15)
    frame = alfred([("2024-10-10", "5.00", "2024-10-11"), ("2024-10-11", "5.10", "2024-10-15")])
    monday, tuesday, wednesday = (
        ny("2024-10-14 18:00"),
        ny("2024-10-15 18:00"),
        ny("2024-10-16 18:00"),
    )
    assert as_of(frame, monday)["annual_yield"].tolist() == pytest.approx(
        [0.05]
    )  # Thursday's value only
    assert as_of(frame, tuesday)["annual_yield"].tolist() == pytest.approx(
        [0.05]
    )  # still not Friday's
    assert as_of(frame, wednesday)["annual_yield"].tolist() == pytest.approx([0.05, 0.051])
    assert frame["availability_rule"].tolist() == [H15_OR_ALFRED_LATER] * 2


def test_a_late_realtime_start_counts_only_from_the_end_of_its_day():
    # The H.15 posting would be Friday 16:15, but ALFRED first shows it on Monday.
    found = available_at(date(2024, 3, 7), date(2024, 3, 11))
    assert found == ny("2024-03-12 00:00").tz_convert("UTC")
    # Never earlier than the posting, even with an early (impossible) realtime_start.
    assert available_at(date(2024, 3, 7), date(2024, 3, 7)) == ny("2024-03-08 16:15").tz_convert(
        "UTC"
    )


def test_dates_before_alfred_count_from_alfreds_first_vintage():
    frame = alfred([("2003-01-02", "1.20", "2005-06-28"), ("2005-07-01", "3.05", "2005-07-05")])
    assert frame["availability_rule"].tolist() == [PRE_ALFRED, H15_OR_ALFRED_LATER]
    assert frame["available_at"].iloc[0] == ny("2005-06-29 00:00").tz_convert("UTC")
    assert as_of(frame, ny("2005-06-28 18:00")).empty  # not usable before it was archived


def test_the_fed_calendar_errs_late_never_early():
    holidays = fed_holidays(2022)
    # 2022-06-19 (Juneteenth) was a Sunday: the Monday is closed too.
    assert date(2022, 6, 20) in holidays
    # Christmas 2021 was a Saturday: both Friday 12-24 and Monday 12-27 count as closed.
    assert {date(2021, 12, 24), date(2021, 12, 27)} <= fed_holidays(2021)
    assert date(2020, 6, 19) not in fed_holidays(2020)  # Juneteenth only from 2021
    # 2021-06-19 was a Saturday: the Friday before counts as closed (conservative).
    assert date(2021, 6, 18) in fed_holidays(2021)
    assert date(2024, 11, 11) in fed_holidays(2024)  # Veterans Day (a Monday, NYSE open)


# --- revisions and decisions ---


def test_a_revision_is_a_new_row_and_never_reaches_an_earlier_decision():
    frame = alfred(
        [
            ("2024-01-02", "5.40", "2024-01-03"),
            ("2024-01-02", "5.42", "2024-02-01"),  # revised a month later
        ]
    )
    assert len(frame) == 2
    assert as_of(frame, ny("2024-01-10 18:00"))["annual_yield"].tolist() == pytest.approx([0.054])
    assert as_of(frame, ny("2024-02-02 18:00"))["annual_yield"].tolist() == [pytest.approx(0.0542)]
    assert first_release(frame)["annual_yield"].tolist() == pytest.approx([0.054])  # booked wealth
    stats = diagnostics(frame)
    assert stats["revised_observations"] == 1
    assert stats["largest_revision"] == pytest.approx(0.0002)


def test_a_holiday_first_release_is_missing_not_skipped():
    frame = alfred([("2024-10-14", ".", "2024-10-15"), ("2024-10-15", "4.60", "2024-10-16")])
    first = first_release(frame)
    assert np.isnan(first["annual_yield"].iloc[0])
    assert legacy_rates(frame)["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-10-15"]


def test_no_vintage_from_the_holdout_is_accepted():
    with pytest.raises(VintageRefused, match="holdout"):
        alfred([("2024-12-30", "4.30", "2025-01-02")])
    with pytest.raises(VintageRefused, match="predate"):
        alfred([("2024-12-30", "4.30", "2024-12-20")])


# --- append-only and prefix stability ---


def test_appending_later_vintages_never_changes_an_earlier_first_release():
    early = alfred([("2024-01-02", "5.40", "2024-01-03"), ("2024-01-03", "5.41", "2024-01-04")])
    later = alfred(
        [
            ("2024-01-02", "5.45", "2024-03-01"),  # a revision
            ("2024-01-04", "5.39", "2024-01-05"),  # a new date
        ],
        retrieved=RETRIEVED + pd.Timedelta(days=1),
    )
    merged = merge_vintages(early, later)
    assert len(merged) == 4
    before = first_release(early)
    after = first_release(merged).iloc[:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)  # bit-identical
    # A decision before the appended vintages were available sees exactly what it saw.
    moment = ny("2024-01-05 18:00")
    pd.testing.assert_frame_equal(as_of(merged, moment), as_of(early, moment), check_exact=True)
    # Booked returns from the first release are bit-identical too.
    from stock_agent.data.rates.transform import prepare_risk_free_returns

    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    old = prepare_risk_free_returns(legacy_rates(early), sessions, "dgs3mo")
    new = prepare_risk_free_returns(legacy_rates(merged), sessions, "dgs3mo")
    assert old.to_numpy().tobytes() == new.to_numpy().tobytes()


def test_a_vintage_earlier_than_a_stored_first_release_is_refused():
    stored = alfred([("2024-01-02", "5.50", "2024-02-01")])
    earlier = alfred([("2024-01-02", "5.49", "2024-01-03")])
    with pytest.raises(VintageRefused, match="booked history is never rewritten"):
        merge_vintages(stored, earlier)


def test_the_pre_alfred_label_is_decided_by_the_row_not_by_the_retrieval():
    revision_only = alfred([("2003-01-02", "1.21", "2010-03-01")])
    assert revision_only["availability_rule"].tolist() == [H15_OR_ALFRED_LATER]
    frame = alfred([("2003-01-02", "1.20", "2005-06-28")])
    frame.loc[0, "availability_rule"] = H15_OR_ALFRED_LATER
    with pytest.raises(VintageRefused, match="label"):
        merge_vintages(None, frame)


def test_a_merge_keeps_the_stored_row_for_a_known_key():
    stored = alfred([("2024-01-02", "5.40", "2024-01-03")])
    again = alfred(
        [("2024-01-02", "5.40", "2024-01-03")], retrieved=RETRIEVED + pd.Timedelta(days=3)
    )
    merged = merge_vintages(stored, again)
    assert merged["retrieved_at"].tolist() == stored["retrieved_at"].tolist()


def test_as_of_cutoffs():
    frame = alfred([("2024-10-10", "5.00", "2024-10-11")])
    edge = frame["available_at"].iloc[0]
    assert len(as_of(frame, edge)) == 1  # inclusive at available_at
    assert as_of(frame, edge - pd.Timedelta(microseconds=1)).empty
    assert as_of(frame, ny("2024-10-11 18:00")).empty  # Friday evening: not until its end
    with pytest.raises(ValueError, match="timezone-aware"):
        as_of(frame, pd.Timestamp("2024-10-12 12:00"))


def test_the_same_key_with_another_value_is_refused():
    early = alfred([("2024-01-02", "5.40", "2024-01-03")])
    clash = alfred([("2024-01-02", "5.41", "2024-01-03")])
    with pytest.raises(VintageRefused, match="never overwritten"):
        merge_vintages(early, clash)
    assert merge_vintages(early, early).equals(early)  # the same retrieval again: no change


def test_a_tampered_availability_is_refused():
    frame = alfred([("2024-01-02", "5.40", "2024-01-03")])
    frame.loc[0, "available_at"] = frame.loc[0, "available_at"] - pd.Timedelta(days=1)
    with pytest.raises(VintageRefused, match="h15_or_alfred_later"):
        merge_vintages(None, frame)


# --- storage ---


def test_storage_is_private_append_only_and_round_trips(tmp_path):
    home = tmp_path / "rates"
    first = alfred([("2024-01-02", "5.40", "2024-01-03")])
    path = save_vintages(first, "dgs3mo", home)
    assert oct(path.stat().st_mode & 0o777) == "0o444"  # read-only once written
    assert oct(home.stat().st_mode & 0o777) == "0o700"
    assert save_vintages(first, "dgs3mo", home) is None  # nothing new: no second file
    later = alfred(
        [("2024-01-02", "5.45", "2024-03-01")], retrieved=RETRIEVED + pd.Timedelta(days=1)
    )
    save_vintages(later, "dgs3mo", home)
    loaded = load_vintages("dgs3mo", home)
    assert len(list(home.glob("*.parquet"))) == 2
    expected = merge_vintages(first, later)
    pd.testing.assert_frame_equal(loaded, expected, check_exact=True)  # exact round trip
    assert str(loaded["available_at"].dtype).endswith("UTC]")
    clash = alfred(
        [("2024-01-02", "5.99", "2024-01-03")], retrieved=RETRIEVED + pd.Timedelta(days=2)
    )
    with pytest.raises(VintageRefused):
        save_vintages(clash, "dgs3mo", home)
    assert len(list(home.glob("*.parquet"))) == 2  # nothing written
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(VintageRefused, match="Git working tree"):
        save_vintages(first, "dgs3mo", tmp_path / "repo" / "rates")
    assert load_vintages("dgs3mo", tmp_path / "empty") is None


def test_the_home_folder_and_a_worktree_marker_file(tmp_path, monkeypatch):
    from stock_agent.data.rates.vintages import default_home

    monkeypatch.setenv("STOCK_AGENT_RATES_HOME", str(tmp_path / "custom"))
    assert default_home() == tmp_path / "custom"
    worktree = tmp_path / "linked"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: elsewhere\n")  # a linked worktree's .git is a file
    with pytest.raises(VintageRefused, match="Git working tree"):
        save_vintages(alfred([("2024-01-02", "5.40", "2024-01-03")]), "dgs3mo", worktree / "r")


def test_a_saturday_new_year_closes_the_friday_before_it_across_the_year_boundary():
    # 2011-01-01 and 2022-01-01 were Saturdays: Friday 2010-12-31 and 2021-12-31 count as
    # closed (filed under the next year's holidays), and so does the Monday after.
    assert next_fed_business_day(date(2010, 12, 30)) == date(2011, 1, 4)
    assert next_fed_business_day(date(2021, 12, 30)) == date(2022, 1, 4)


def test_the_posting_time_follows_daylight_saving():
    # Friday 2024-03-08: posted Monday 03-11 at 16:15 EDT (20:15 UTC), after the switch.
    assert available_at(date(2024, 3, 8), date(2024, 3, 8)) == pd.Timestamp("2024-03-11 20:15Z")
    # Friday 2024-11-01: posted Monday 11-04 at 16:15 EST (21:15 UTC), after the switch back.
    assert available_at(date(2024, 11, 1), date(2024, 11, 1)) == pd.Timestamp("2024-11-04 21:15Z")


def test_revisions_are_measured_against_the_first_release():
    frame = alfred(
        [
            ("2024-01-02", ".", "2024-01-03"),
            ("2024-01-02", "4.10", "2024-02-01"),  # a missing first release filled later
            ("2024-01-03", "5.00", "2024-01-04"),
            ("2024-01-03", "5.05", "2024-02-01"),
            ("2024-01-03", "4.98", "2024-03-01"),
        ]
    )
    stats = diagnostics(frame)
    assert stats["revised_observations"] == 2 and stats["revisions_from_missing"] == 1
    assert stats["largest_revision"] == pytest.approx(0.0005)  # |5.05 - 5.00|, not max - min
