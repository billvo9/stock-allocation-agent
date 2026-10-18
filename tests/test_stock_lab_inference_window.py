"""
The frozen-model reader (decision A2): exactly the 252 sessions ending at the
as-of close, as stored by the decision time, and nothing else. Negative tests
prove that a window reaching past the as-of, a batch retrieved after the
decision time, a short window and a retrospective (backtest) request are
refused or never seen. Fixtures only.
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
    weekdays,
    window_start,
)

from stock_agent.stock_lab.inference.schema import (
    INFERENCE_SESSIONS,
    InferenceRefused,
    Refusal,
    bars_sha256,
)
from stock_agent.stock_lab.inference.window import load_inference_window
from stock_agent.stock_lab.prices import BAR_COLUMNS
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

FIRST = window_start()
EVENING = ny(AS_OF, "21:00")
NEXT = next_day(AS_OF)  # 2026-10-16


def refused(reason: Refusal):
    return pytest.raises(InferenceRefused, match=f"^{reason.value}: ")


def load(store, as_of=AS_OF, lookback=INFERENCE_SESSIONS, *, retrieved_by=EVENING, **kwargs):
    options = {"sessions": weekdays, "calendar_id": CALENDAR_ID, **kwargs}
    return load_inference_window(
        store.path, "NVDA", as_of, lookback, retrieved_by=retrieved_by, **options
    )


@pytest.fixture
def stored(tmp_path):
    store, clock = open_store(tmp_path)
    batch = record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    return store, clock, batch


def test_the_window_is_exactly_the_252_sessions_ending_at_the_as_of_close(stored):
    store, _, batch = stored
    window = load(store)
    assert window.lookback == 252 and len(window.sessions) == 252
    assert window.sessions[-1] == pd.Timestamp(AS_OF, tz="UTC")
    assert window.sessions.equals(weekdays(FIRST, AS_OF))
    assert window.bars["session"].tolist() == list(window.sessions)
    assert list(window.bars.columns) == [*BAR_COLUMNS, "batch_id"]
    assert window.bars["close"].notna().all() and window.gaps == ()
    assert window.batches == (batch,) and window.revised_sessions == ()
    assert window.data_class == "real" and window.calendar_id == CALENDAR_ID
    assert window.retrieved_by == EVENING
    expected = make_bars(FIRST, AS_OF)
    assert bars_sha256(window.bars) == bars_sha256(expected)


@pytest.mark.parametrize("lookback", [63, 251, 253, 504])
def test_any_other_lookback_is_refused(stored, lookback):
    with refused(Refusal.WRONG_LOOKBACK):
        load(stored[0], lookback=lookback)


@pytest.mark.parametrize("lookback", [252.0, "252", True])
def test_a_lookback_that_is_not_an_int_is_refused(stored, lookback):
    with refused(Refusal.INVALID_REQUEST):
        load(stored[0], lookback=lookback)


def test_an_as_of_that_is_not_a_session_is_refused(stored):
    with refused(Refusal.INVALID_REQUEST):
        load(stored[0], as_of="2026-10-17", retrieved_by=ny("2026-10-17", "21:00"))
    with refused(Refusal.INVALID_REQUEST):
        load(stored[0], as_of=pd.Timestamp(f"{AS_OF} 16:00", tz="UTC"))


def test_a_naive_or_missing_decision_time_is_refused(stored):
    with refused(Refusal.INVALID_REQUEST):
        load(stored[0], retrieved_by=pd.Timestamp(f"{NEXT} 01:00"))
    with refused(Refusal.INVALID_REQUEST):
        load(stored[0], retrieved_by=None)
    with pytest.raises(TypeError):  # retrieved_by is required and keyword-only
        load_inference_window(stored[0].path, "NVDA", AS_OF, INFERENCE_SESSIONS, weekdays)


def test_a_window_reaching_into_development_data_is_refused(tmp_path):
    store, clock = open_store(tmp_path, now=ny("2025-06-02", "21:00"))
    record(store, clock, "2025-01-01", "2025-06-02", retrieved_at=ny("2025-06-02", "20:30"))
    with refused(Refusal.BEFORE_PROTECTED_PERIOD):
        load(store, as_of="2025-06-02", retrieved_by=ny("2025-06-02", "21:00"))


def test_a_decision_before_the_as_of_close_was_available_is_refused(stored):
    # The as-of close is not final until 20:00 New York: a decision at 19:00
    # could only have seen a partial session.
    with refused(Refusal.CLOSE_NOT_AVAILABLE):
        load(stored[0], retrieved_by=ny(AS_OF, "19:00"))
    with refused(Refusal.CLOSE_NOT_AVAILABLE):  # a window "reaching past" the decision
        load(stored[0], as_of=NEXT, retrieved_by=EVENING)


def test_a_decision_after_the_next_open_is_refused(stored):
    # Once the next session has opened, the as-of close is history: a "decision"
    # then would be a backtest on protected outcomes.
    with refused(Refusal.AFTER_NEXT_OPEN):
        load(stored[0], retrieved_by=ny(NEXT, "09:30"))
    with refused(Refusal.AFTER_NEXT_OPEN):
        load(stored[0], retrieved_by=pd.Timestamp("2027-01-04T00:00:00Z"))
    load(stored[0], retrieved_by=ny(NEXT, "09:29"))  # still before the open


def test_over_a_weekend_the_decision_may_wait_until_monday_open(tmp_path):
    friday, monday = "2026-10-16", "2026-10-19"
    store, clock = open_store(tmp_path, now=ny(friday, "21:00"))
    record(store, clock, window_start(friday), friday, retrieved_at=ny(friday, "20:30"))
    assert len(load(store, as_of=friday, retrieved_by=ny("2026-10-18", "12:00")).sessions) == 252
    with refused(Refusal.AFTER_NEXT_OPEN):
        load(store, as_of=friday, retrieved_by=ny(monday, "09:30"))


def test_a_batch_retrieved_after_the_decision_time_is_never_read(stored):
    store, clock, original = stored
    revised = make_bars(FIRST, AS_OF, shift=1.0)  # a later download serves other values
    later = record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "22:00"), bars=revised)
    before = load(store, retrieved_by=ny(AS_OF, "21:30"))
    assert before.batches == (original,) and set(before.bars["batch_id"]) == {original.batch_id}
    assert before.revised_sessions == ()  # the later vintage is invisible, not even as a revision
    after = load(store, retrieved_by=ny(AS_OF, "22:30"))
    assert after.batches == (later,) and len(after.revised_sessions) == 252
    assert after.bars["close"].tolist() == revised["close"].tolist()
    # A re-run with the original decision time reproduces the original inputs.
    assert bars_sha256(load(store, retrieved_by=ny(AS_OF, "21:30")).bars) == bars_sha256(
        before.bars
    )


def test_when_only_a_later_batch_holds_the_window_it_is_refused(tmp_path):
    store, clock = open_store(tmp_path)
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "22:00"))
    with refused(Refusal.SHORT_WINDOW):
        load(store, retrieved_by=ny(AS_OF, "21:00"))


def test_a_batch_is_eligible_only_once_it_was_recorded(tmp_path):
    # retrieved at 20:30 but written to the store at 21:20: a decision at 21:00
    # could not have read it.
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    clock.now = ny(AS_OF, "21:20")
    store.record_batch(
        "NVDA",
        bars,
        source="fixture",
        retrieved_at=ny(AS_OF, "20:30"),
        request_start=FIRST,
        request_end=AS_OF,
        sessions=weekdays,
        calendar_id=CALENDAR_ID,
        data_class="real",
        vintage_note="fixture",
    )
    with refused(Refusal.SHORT_WINDOW):
        load(store, retrieved_by=ny(AS_OF, "21:00"))
    assert len(load(store, retrieved_by=ny(AS_OF, "21:30")).sessions) == 252


def test_a_short_window_is_refused_never_padded(tmp_path):
    store, clock = open_store(tmp_path)
    days = weekdays(FIRST, AS_OF)
    record(store, clock, days[52].strftime("%Y-%m-%d"), AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    with refused(Refusal.SHORT_WINDOW) as caught:
        load(store)
    assert "52 of 252 sessions" in str(caught.value)


def test_a_hole_between_batches_is_a_short_window(tmp_path):
    store, clock = open_store(tmp_path, now=ny("2026-07-01", "21:00"))
    days = weekdays(FIRST, AS_OF)
    early_end = days[99].strftime("%Y-%m-%d")
    record(store, clock, FIRST, early_end, retrieved_at=ny(early_end, "20:30"))
    late_start = days[101].strftime("%Y-%m-%d")  # days[100] was never requested
    record(store, clock, late_start, AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    with refused(Refusal.SHORT_WINDOW) as caught:
        load(store)
    assert days[100].strftime("%Y-%m-%d") in str(caught.value)


def test_batches_overlapping_the_window_are_joined_newest_first(tmp_path):
    store, clock = open_store(tmp_path, now=ny("2026-07-01", "21:00"))
    days = weekdays(FIRST, AS_OF)
    middle = days[200].strftime("%Y-%m-%d")
    old = record(store, clock, FIRST, middle, retrieved_at=ny(middle, "20:30"))
    recent_start = days[150].strftime("%Y-%m-%d")
    recent_bars = make_bars(recent_start, AS_OF, shift=0.5)  # revises 51 shared sessions
    new = record(
        store, clock, recent_start, AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=recent_bars
    )
    window = load(store)
    assert window.batches == (old, new)
    assert (window.bars["batch_id"] == old.batch_id).sum() == 150
    assert (window.bars["batch_id"] == new.batch_id).sum() == 102
    assert len(window.revised_sessions) == 51


def test_a_longer_batch_is_cut_to_exactly_the_window(tmp_path):
    # A batch reaching further back than 252 sessions supplies only the window.
    store, clock = open_store(tmp_path)
    record(store, clock, window_start(sessions=300), AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    window = load(store)
    assert len(window.bars) == 252 and window.bars["session"].min() == pd.Timestamp(FIRST, tz="UTC")


def test_a_vendor_gap_stays_a_gap_and_is_listed(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    bars.loc[100, ["open", "high", "low", "close", "volume"]] = float("nan")
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=bars)
    window = load(store)
    assert len(window.bars) == 252 and window.bars["close"].isna().sum() == 1
    assert window.gaps == (weekdays(FIRST, AS_OF)[100].strftime("%Y-%m-%d"),)


def test_an_older_batch_never_fills_a_newer_batchs_gap(tmp_path):
    store, clock = open_store(tmp_path)
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    bars = make_bars(FIRST, AS_OF)
    bars.loc[100, ["open", "high", "low", "close", "volume"]] = float("nan")
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "21:00"), bars=bars)
    window = load(store, retrieved_by=ny(AS_OF, "21:30"))
    assert pd.isna(window.bars["close"].iloc[100]) and len(window.gaps) == 1
    assert weekdays(FIRST, AS_OF)[100].strftime("%Y-%m-%d") in window.revised_sessions
    unserved = make_bars(FIRST, AS_OF)
    unserved.loc[251, ["open", "high", "low", "close", "volume"]] = float("nan")
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "21:40"), bars=unserved)
    with refused(Refusal.AS_OF_CLOSE_MISSING):  # never the older batch's as-of close
        load(store, retrieved_by=ny(AS_OF, "21:50"))


def test_the_decision_interval_follows_new_york_time_in_winter(tmp_path):
    winter = "2026-12-15"  # a Tuesday in Eastern Standard Time (UTC-5)
    store, clock = open_store(tmp_path, now=pd.Timestamp("2026-12-16T01:30:00Z"))
    record(
        store, clock, window_start(winter), winter, retrieved_at=pd.Timestamp("2026-12-16T01:00Z")
    )
    with refused(Refusal.RETRIEVAL_TIME):  # 19:59 EST is before the close is available
        record(
            store,
            clock,
            window_start(winter),
            winter,
            retrieved_at=pd.Timestamp("2026-12-16T00:59Z"),
        )
    with refused(Refusal.CLOSE_NOT_AVAILABLE):
        load(store, as_of=winter, retrieved_by=pd.Timestamp("2026-12-16T00:59:59Z"))
    assert (
        len(load(store, as_of=winter, retrieved_by=pd.Timestamp("2026-12-16T01:01Z")).bars) == 252
    )
    load(store, as_of=winter, retrieved_by=pd.Timestamp("2026-12-16T14:29:59Z"))  # 09:29:59 EST
    with refused(Refusal.AFTER_NEXT_OPEN):
        load(store, as_of=winter, retrieved_by=pd.Timestamp("2026-12-16T14:30Z"))


def test_the_next_open_follows_the_exchange_calendar_across_a_holiday(tmp_path):
    from stock_agent.calendars import nyse_rules

    as_of = "2026-07-02"  # Friday 2026-07-03 is a holiday (July 4 falls on a Saturday)
    days = nyse_rules.sessions("2025-06-01", as_of)[-252:]
    first = days[0].strftime("%Y-%m-%d")
    store, clock = open_store(tmp_path)
    clock.now = ny(as_of, "20:31")
    bars = make_bars(first, as_of).iloc[-252:].reset_index(drop=True)
    bars["session"] = days
    store.record_batch(
        "NVDA",
        bars,
        source="fixture",
        retrieved_at=ny(as_of, "20:30"),
        request_start=first,
        request_end=as_of,
        sessions=nyse_rules.sessions,
        calendar_id=nyse_rules.CALENDAR_ID,
        data_class="real",
        vintage_note="fixture",
    )
    options = {"sessions": nyse_rules.sessions, "calendar_id": nyse_rules.CALENDAR_ID}
    window = load_inference_window(
        store.path, "NVDA", as_of, retrieved_by=ny("2026-07-03", "12:00"), **options
    )
    assert window.sessions.equals(days)
    with refused(Refusal.AFTER_NEXT_OPEN):
        load_inference_window(
            store.path, "NVDA", as_of, retrieved_by=ny("2026-07-06", "09:30"), **options
        )


def test_a_missing_as_of_close_is_refused(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    bars.loc[251, ["open", "high", "low", "close", "volume"]] = float("nan")
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=bars)
    with refused(Refusal.AS_OF_CLOSE_MISSING):
        load(store)


def test_no_row_after_the_as_of_is_ever_returned(tmp_path):
    # The store also holds later sessions, retrieved later: the earlier decision
    # sees only its own window.
    store, clock = open_store(tmp_path)
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    later_end = next_day(AS_OF, 3)
    record(store, clock, window_start(later_end), later_end, retrieved_at=ny(later_end, "20:30"))
    window = load(store)
    assert window.bars["session"].max() == pd.Timestamp(AS_OF, tz="UTC")
    late = load(store, as_of=later_end, retrieved_by=ny(later_end, "21:00"))
    assert late.bars["session"].max() == pd.Timestamp(later_end, tz="UTC")
    assert len(late.bars) == 252  # the first batch starts earlier: only the window is read
    assert late.bars["session"].min() == pd.Timestamp(window_start(later_end), tz="UTC")


def _forge(store, statement: str, *parameters) -> None:
    """Edit the file behind the writer's back: drop the write-once triggers, then change it."""

    with sqlite3.connect(store.path) as connection:
        triggers = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        ]
        for name in triggers:
            connection.execute(f"DROP TRIGGER {name}")
        connection.execute(statement, parameters)


def test_a_forged_batch_that_reaches_past_its_retrieval_is_refused(stored):
    # A batch claiming to have been retrieved before its own last close was
    # available (impossible through the writer) is refused, not trusted.
    store, _, batch = stored
    _forge(
        store,
        "UPDATE batches SET retrieved_at = ?, recorded_at = ? WHERE batch_id = ?",
        "2026-10-15T15:00:00.000000Z",
        "2026-10-15T15:01:00.000000Z",
        batch.batch_id,
    )
    with refused(Refusal.INTEGRITY):
        load(store)


def test_an_edited_value_breaks_the_content_hash(stored):
    store = stored[0]
    _forge(store, "UPDATE bars SET close = close * 2 WHERE session = ?", AS_OF)
    with refused(Refusal.INTEGRITY):
        load(store)


def test_a_deleted_row_is_refused(stored):
    store = stored[0]
    _forge(store, "DELETE FROM bars WHERE session = ?", FIRST)
    with refused(Refusal.INTEGRITY):
        load(store)


def test_a_batch_on_another_calendar_is_refused(stored):
    with refused(Refusal.CALENDAR_MISMATCH):
        load(stored[0], calendar_id="nyse_full_day_rules_2003_2030_v1")


def test_a_missing_store_and_a_research_store_are_refused(tmp_path):
    store, _ = open_store(tmp_path)
    with refused(Refusal.NO_STORE):
        load(store)
    assert not store.path.exists()  # reading creates nothing
    research = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME)
    research.add(parse_tickers("NVDA"))
    with refused(Refusal.LOCATION):  # not the inference store's file name
        load_inference_window(
            research.path,
            "NVDA",
            AS_OF,
            retrieved_by=EVENING,
            sessions=weekdays,
            calendar_id=CALENDAR_ID,
        )


def test_reading_changes_nothing(stored):
    store = stored[0]
    before = store.path.read_bytes()
    files = sorted(store.path.parent.iterdir())
    load(store)
    assert store.path.read_bytes() == before and sorted(store.path.parent.iterdir()) == files
