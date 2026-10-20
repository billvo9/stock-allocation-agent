"""
The inference store's outcome reader (stock_agent.stock_lab.inference.outcomes):
what one recorded forecast is scored on, and each stock's newest stored close.
Fixtures on a weekday calendar; no network, no real data.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    make_bars,
    next_day,
    ny,
    open_store,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.stock_lab.inference.outcomes import latest_close, read_outcomes
from stock_agent.stock_lab.inference.schema import InferenceRefused, Refusal

TARGETS = [next_day(AS_OF, h) for h in range(1, 6)]
KNOWN = ny(AS_OF, "20:35")


def _read(store, *, scored_by, known_at=KNOWN, targets=TARGETS, calendar_id=CALENDAR_ID):
    return read_outcomes(
        store.path, "NVDA", as_of_session=AS_OF, target_sessions=targets, known_at=known_at,
        scored_by=scored_by, sessions=weekdays, calendar_id=calendar_id,
    )  # fmt: skip


@pytest.fixture
def stored(tmp_path):
    """The as-of window (retrieved at 20:30 on the as-of day), as the forecast saw it."""

    store, clock = open_store(tmp_path)
    record(store, clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    return store, clock


def test_before_any_later_batch_every_target_is_pending(stored):
    store, _ = stored
    found = _read(store, scored_by=ny(TARGETS[0], "21:00"))
    assert list(found.span.strftime("%Y-%m-%d")) == [AS_OF, *TARGETS]
    assert list(found.bars["session"].dt.strftime("%Y-%m-%d")) == [AS_OF]
    assert set(found.batch_of) == {AS_OF}
    assert found.seen_before == {AS_OF}  # the forecast's own as-of close


def test_a_later_batch_supplies_only_the_sessions_of_the_forecast(stored):
    store, clock = stored
    first, last = window_start(TARGETS[1]), TARGETS[1]
    later = record(store, clock, first, last, retrieved_at=ny(TARGETS[1], "20:30"))
    found = _read(store, scored_by=ny(TARGETS[1], "21:00"))
    days = list(found.bars["session"].dt.strftime("%Y-%m-%d"))
    assert days == [AS_OF, TARGETS[0], TARGETS[1]]  # nothing before the as-of, nothing later
    assert found.batch_of[TARGETS[0]] == later.batch_id == found.batch_of[AS_OF]
    assert found.seen_before == {AS_OF}


def test_a_batch_recorded_after_the_scoring_time_is_not_read(stored):
    store, clock = stored
    record(store, clock, window_start(TARGETS[0]), TARGETS[0], retrieved_at=ny(TARGETS[0], "20:30"))
    found = _read(store, scored_by=ny(TARGETS[0], "20:30"))  # recorded a minute later
    assert TARGETS[0] not in found.batch_of


def test_a_session_stored_before_the_forecast_is_known_counts_as_seen(tmp_path):
    store, clock = open_store(tmp_path)
    # A late forecast: the batch with the first target was stored before the forecast existed.
    record(store, clock, window_start(TARGETS[0]), TARGETS[0], retrieved_at=ny(TARGETS[0], "20:30"))
    found = _read(store, known_at=ny(TARGETS[0], "21:00"), scored_by=ny(TARGETS[0], "21:05"))
    assert TARGETS[0] in found.seen_before


def test_targets_that_do_not_follow_the_calendar_are_refused(stored):
    store, _ = stored
    for targets in (TARGETS[1:] + [next_day(AS_OF, 6)], [AS_OF, *TARGETS[1:]], []):
        with pytest.raises(InferenceRefused) as caught:
            _read(store, scored_by=ny(TARGETS[0], "21:00"), targets=targets)
        assert caught.value.reason is Refusal.INVALID_REQUEST


def test_scoring_before_the_forecast_existed_is_refused(stored):
    store, _ = stored
    with pytest.raises(InferenceRefused) as caught:
        _read(store, scored_by=ny(AS_OF, "20:00"))
    assert caught.value.reason is Refusal.INVALID_REQUEST


def test_no_store_another_calendar_and_a_changed_batch_are_refused(tmp_path, stored):
    store, _ = stored
    empty, _ = open_store(tmp_path / "other")
    with pytest.raises(InferenceRefused) as caught:
        _read(empty, scored_by=ny(TARGETS[0], "21:00"))
    assert caught.value.reason is Refusal.NO_STORE
    with pytest.raises(InferenceRefused) as caught:
        _read(store, scored_by=ny(TARGETS[0], "21:00"), calendar_id="other_calendar_v1")
    assert caught.value.reason is Refusal.CALENDAR_MISMATCH
    connection = sqlite3.connect(store.path)
    connection.execute("DROP TRIGGER bars_no_update")
    connection.execute("UPDATE bars SET close = close + 1 WHERE session = ?", (AS_OF,))
    connection.commit()
    connection.close()
    with pytest.raises(InferenceRefused) as caught:
        _read(store, scored_by=ny(TARGETS[0], "21:00"))
    assert caught.value.reason is Refusal.INTEGRITY


def test_the_latest_close_skips_a_trailing_gap_and_respects_the_decision_time(tmp_path):
    store, clock = open_store(tmp_path)
    assert latest_close(store.path, "NVDA", retrieved_by=KNOWN, calendar_id=CALENDAR_ID) is None
    bars = make_bars(window_start(), AS_OF)
    bars.loc[bars.index[-1], ["open", "high", "low", "close", "volume"]] = None  # not served
    record(store, clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=bars)
    found = latest_close(store.path, "NVDA", retrieved_by=KNOWN, calendar_id=CALENDAR_ID)
    assert found.session == next_day(AS_OF, -1)
    assert found.retrieved_at.startswith(
        pd.Timestamp(ny(AS_OF, "20:30")).strftime("%Y-%m-%dT%H:%M")
    )
    early = latest_close(
        store.path, "NVDA", retrieved_by=ny(AS_OF, "20:00"), calendar_id=CALENDAR_ID
    )
    assert early is None


def test_a_later_batch_beyond_the_targets_still_yields_only_the_span(stored):
    store, clock = stored
    late = next_day(TARGETS[-1], 3)
    record(store, clock, window_start(late), late, retrieved_at=ny(late, "20:30"),
           bars=steady_bars(window_start(late), late))  # fmt: skip
    found = _read(store, scored_by=ny(late, "21:00"))
    days = list(found.bars["session"].dt.strftime("%Y-%m-%d"))
    assert days == [AS_OF, *TARGETS]  # nothing after the last target, though the batch has it


def test_a_revised_close_is_listed_and_the_latest_vintage_settles_the_target(tmp_path):
    store, clock = open_store(tmp_path)
    first_batch = steady_bars(window_start(), AS_OF)
    record(store, clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=first_batch)
    later = steady_bars(window_start(TARGETS[0]), TARGETS[0])
    revised = later["session"] == pd.Timestamp(AS_OF, tz="UTC")
    later.loc[revised, "close"] = later.loc[revised, "close"] * 1.001  # the vendor revised it
    record(store, clock, window_start(TARGETS[0]), TARGETS[0],
           retrieved_at=ny(TARGETS[0], "20:30"), bars=later)  # fmt: skip
    found = _read(store, scored_by=ny(TARGETS[0], "21:00"))
    assert found.revised == (AS_OF,)
    base = found.bars.loc[found.bars["session"] == pd.Timestamp(AS_OF, tz="UTC"), "close"]
    assert float(base.iloc[0]) == float(later.loc[revised, "close"].iloc[0])
