"""
The reconstruction reader (stock_lab.inference.window.load_reconstruction_window;
owner decision of 2026-10-30): a missed evening's 252-session window, read
after the evening closed, from batches retrieved by the reconstruction's own
time, with sessions dated up to the as-of close only.

It must read exactly what the on-time reader read on the same data, never a
later session, never a batch retrieved after its own time, and never an
evening that can still be forecast live. Fixtures on a weekday calendar; no
network, nothing real.
"""

from __future__ import annotations

import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    next_day,
    ny,
    open_store,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.stock_lab.inference.schema import InferenceRefused, Refusal
from stock_agent.stock_lab.inference.window import (
    load_inference_window,
    load_reconstruction_window,
)

LATER_DAY = next_day(AS_OF, 2)  # two sessions after the missed one
LATER = ny(LATER_DAY, "21:15")  # when the reconstruction is made


@pytest.fixture
def stored(tmp_path):
    """The missed evening's own batch (20:30) and a later one through LATER_DAY (20:30)."""

    store, clock = open_store(tmp_path)
    record(store, clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
           bars=steady_bars(window_start(), AS_OF))  # fmt: skip
    later = steady_bars(window_start(LATER_DAY), LATER_DAY)
    after = later["session"] > pd.Timestamp(AS_OF, tz="UTC")
    later.loc[after, ["open", "high", "low", "close"]] *= 10.0  # wild moves after the as-of close
    record(store, clock, window_start(LATER_DAY), LATER_DAY,
           retrieved_at=ny(LATER_DAY, "20:30"), bars=later)  # fmt: skip
    return store


def _read(store, reader, retrieved_by):
    return reader(store.path, "NVDA", AS_OF, retrieved_by=retrieved_by, sessions=weekdays,
                  calendar_id=CALENDAR_ID)  # fmt: skip


def test_a_reconstruction_reads_what_the_evening_read_on_the_same_data(stored):
    on_time = _read(stored, load_inference_window, ny(AS_OF, "20:40"))
    rebuilt = _read(stored, load_reconstruction_window, LATER)
    assert (
        rebuilt.sessions.equals(on_time.sessions) and rebuilt.as_of_session == on_time.as_of_session
    )
    columns = [c for c in on_time.bars.columns if c != "batch_id"]
    pd.testing.assert_frame_equal(rebuilt.bars[columns], on_time.bars[columns])
    assert rebuilt.gaps == on_time.gaps and rebuilt.revised_sessions == ()
    assert rebuilt.retrieved_by == LATER  # its own time, recorded with it


def test_no_session_after_the_as_of_close_ever_enters(stored):
    rebuilt = _read(stored, load_reconstruction_window, LATER)
    assert rebuilt.bars["session"].max() == pd.Timestamp(AS_OF, tz="UTC")
    assert len(rebuilt.bars) == 252 and rebuilt.sessions[-1] == pd.Timestamp(AS_OF, tz="UTC")
    # The later batch supplied sessions too (the vintage rule), but none after the as-of close.
    assert LATER_DAY in {b.request_end for b in rebuilt.batches}


def test_only_batches_retrieved_by_its_own_time_are_read(stored):
    before_the_later_batch = ny(next_day(AS_OF), "21:00")
    rebuilt = _read(stored, load_reconstruction_window, before_the_later_batch)
    assert {b.request_end for b in rebuilt.batches} == {AS_OF}


def test_an_evening_that_can_still_be_forecast_live_is_refused(stored):
    for still_open in (ny(AS_OF, "20:40"), ny(next_day(AS_OF), "09:29:59")):
        with pytest.raises(InferenceRefused) as refused:
            _read(stored, load_reconstruction_window, still_open)
        assert refused.value.reason is Refusal.INVALID_REQUEST
        assert "forecast it live" in refused.value.detail
    assert _read(stored, load_reconstruction_window, ny(next_day(AS_OF), "09:30")).sessions[-1]


def test_the_live_reader_still_refuses_after_the_next_open(stored):
    with pytest.raises(InferenceRefused) as refused:
        _read(stored, load_inference_window, LATER)
    assert refused.value.reason is Refusal.AFTER_NEXT_OPEN


def test_a_short_window_is_refused_never_padded(tmp_path):
    store, clock = open_store(tmp_path)
    first = window_start(AS_OF, sessions=247)  # five sessions short of the window
    record(store, clock, first, AS_OF, retrieved_at=ny(AS_OF, "20:30"),
           bars=steady_bars(first, AS_OF))  # fmt: skip
    with pytest.raises(InferenceRefused) as refused:
        _read(store, load_reconstruction_window, LATER)
    assert refused.value.reason is Refusal.SHORT_WINDOW


def test_a_session_corrected_later_is_read_as_served_last_and_reported(tmp_path):
    # The vendor-revision limit: a close on or before the as-of session that the provider
    # served differently after the evening is read as served last, and listed as revised.
    store, clock = open_store(tmp_path)
    record(store, clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
           bars=steady_bars(window_start(), AS_OF))  # fmt: skip
    corrected = steady_bars(window_start(LATER_DAY), LATER_DAY)
    day = pd.Timestamp(AS_OF, tz="UTC") - pd.Timedelta(days=1)  # Wednesday, inside the window
    corrected.loc[corrected["session"] == day, "close"] += 1.0
    record(store, clock, window_start(LATER_DAY), LATER_DAY,
           retrieved_at=ny(LATER_DAY, "20:30"), bars=corrected)  # fmt: skip
    rebuilt = _read(store, load_reconstruction_window, LATER)
    on_time = _read(store, load_inference_window, ny(AS_OF, "20:40"))
    assert rebuilt.revised_sessions == (day.strftime("%Y-%m-%d"),)
    served = rebuilt.bars.set_index("session").loc[day, "close"]
    assert served == on_time.bars.set_index("session").loc[day, "close"] + 1.0
