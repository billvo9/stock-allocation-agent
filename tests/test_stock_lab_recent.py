"""
The isolated descriptive reader for the Lab page (decision A6): recent
observed prices and their daily change from the inference store, labelled as
observed and never as research evidence. It computes no category, feature,
streak or score, reads only, and shows an honest state when nothing is
stored. Fixtures only.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from stock_lab_inference_helpers import AS_OF, make_bars, next_day, ny, open_store, record, weekdays

from stock_agent.stock_lab.inference.recent import (
    RECENT_SESSIONS,
    RecentPrices,
    RecentSession,
    read_recent,
)
from stock_agent.stock_lab.inference.schema import DB_NAME
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

FIRST = "2026-07-01"


@pytest.fixture
def stored(tmp_path):
    store, clock = open_store(tmp_path)
    batch = record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    return store, clock, batch


def test_without_a_store_it_says_so_and_creates_nothing(tmp_path):
    path = tmp_path / "inference" / DB_NAME
    recent = read_recent("NVDA", path=path)
    assert recent.state == "no_store" and recent.sessions == () and recent.last_close is None
    assert not path.parent.exists()


def test_a_stock_without_stored_prices_is_no_data(stored):
    recent = read_recent("AMD", path=stored[0].path)
    assert recent.state == "no_data" and recent.sessions == ()


def test_it_shows_the_latest_observed_sessions_with_their_lineage(stored):
    store, _, batch = stored
    recent = read_recent("NVDA", path=store.path)
    bars = make_bars(FIRST, AS_OF)
    assert recent.state == "observed" and recent.data_class == "real"
    assert len(recent.sessions) == RECENT_SESSIONS == 60
    assert recent.sessions[-1].session == AS_OF and recent.last_session == AS_OF
    assert recent.first_session == recent.sessions[0].session
    assert recent.last_close == pytest.approx(bars["close"].iloc[-1])
    closes = [row.close for row in recent.sessions]
    assert closes == pytest.approx(bars["close"].iloc[-60:].tolist())
    change = bars["close"].iloc[-1] * bars["split_factor"].iloc[-1] / bars["close"].iloc[-2] - 1
    assert recent.sessions[-1].change == pytest.approx(change)
    assert recent.retrieved_at == batch.retrieved_at and recent.batch_ids == (batch.batch_id,)
    assert recent.source == "fixture" and recent.revised == 0 and recent.gaps == 0


def test_it_never_computes_a_category_feature_or_score():
    names = {field.name for field in dataclasses.fields(RecentSession)}
    names |= {field.name for field in dataclasses.fields(RecentPrices)}
    for word in ("category", "direction", "streak", "run", "probability", "score", "feature"):
        assert not [name for name in names if word in name], word


def test_the_count_is_bounded(stored):
    assert len(read_recent("NVDA", path=stored[0].path, count=5).sessions) == 5
    for count in (0, 253, 5.0):
        with pytest.raises(ValueError):
            read_recent("NVDA", path=stored[0].path, count=count)


def test_gaps_revisions_and_splits_are_marked_never_filled(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    bars.loc[len(bars) - 3, ["open", "high", "low", "close", "volume"]] = float("nan")
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:30"), bars=bars)
    revised = bars.copy()
    revised.loc[len(bars) - 5, "close"] += 1.0
    revised.loc[len(bars) - 1, "div_cash"] = 0.25
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "21:30"), bars=revised)
    recent = read_recent("NVDA", path=store.path)
    rows = {row.session: row for row in recent.sessions}
    days = [day.strftime("%Y-%m-%d") for day in weekdays(FIRST, AS_OF)]
    assert rows[days[-3]].close is None and "gap" in rows[days[-3]].notes
    assert rows[days[-2]].change is None  # no return across a gap
    assert "revised" in rows[days[-5]].notes and recent.revised == 2
    assert "ex-dividend" in rows[days[-1]].notes and recent.gaps == 1


def test_only_the_latest_contiguous_stretch_is_shown(tmp_path):
    store, clock = open_store(tmp_path, now=ny("2026-08-03", "21:00"))
    record(store, clock, "2026-07-01", "2026-07-31", retrieved_at=ny("2026-07-31", "20:30"))
    # 2026-08-03 was never requested: whether anything lies between is unknown.
    record(store, clock, "2026-08-04", AS_OF, retrieved_at=ny(AS_OF, "20:30"))
    recent = read_recent("NVDA", path=store.path, count=252)
    assert recent.sessions[0].session == "2026-08-04" and recent.sessions[0].change is None


def test_only_the_latest_forecasts_five_targets_are_marked_as_its_outcomes(stored):
    as_of = next_day(AS_OF, -8)
    targets = [next_day(as_of, k) for k in range(1, 6)]
    recent = read_recent(
        "NVDA", path=stored[0].path, forecast_as_of=as_of, forecast_targets=targets
    )
    marked = [row.session for row in recent.sessions if row.forecast_target]
    assert marked == targets  # the three sessions after the last target are not its outcomes
    assert recent.forecast_as_of == as_of and recent.forecast_targets == tuple(targets)
    plain = read_recent("NVDA", path=stored[0].path)
    assert not any(row.forecast_target for row in plain.sessions)
    assert plain.forecast_as_of is None and plain.forecast_targets == ()


def test_forecast_targets_must_follow_the_as_of_in_order(stored):
    path = stored[0].path
    for as_of, targets in (
        (None, [AS_OF]),
        (AS_OF, []),
        (AS_OF, [next_day(AS_OF, -1)]),
        (next_day(AS_OF, -2), [AS_OF, next_day(AS_OF, -1)]),
        (AS_OF, ["not a date"]),
    ):
        with pytest.raises(ValueError):
            read_recent("NVDA", path=path, forecast_as_of=as_of, forecast_targets=targets)


def test_a_file_that_is_not_the_inference_store_is_unreadable_not_a_crash(tmp_path):
    research = StockLabStore(tmp_path / "x" / RESEARCH_DB_NAME)
    research.add(parse_tickers("NVDA"))
    directory = tmp_path / "y"
    directory.mkdir(mode=0o700)
    (directory / DB_NAME).write_bytes(research.path.read_bytes())
    recent = read_recent("NVDA", path=directory / DB_NAME)
    assert recent.state == "unreadable" and recent.reason == "not_inference_store"
    assert recent.sessions == ()


def test_reading_changes_nothing(stored):
    store = stored[0]
    before, files = store.path.read_bytes(), sorted(store.path.parent.iterdir())
    read_recent("NVDA", path=store.path)
    assert store.path.read_bytes() == before and sorted(store.path.parent.iterdir()) == files


def test_the_default_path_comes_from_the_environment(stored, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_INFERENCE_HOME", str(stored[0].path.parent))
    assert read_recent("NVDA").state == "observed"
    assert pd.Timestamp(read_recent("NVDA").last_session) == pd.Timestamp(AS_OF)


def _tamper(store, *statements) -> None:
    import sqlite3

    with sqlite3.connect(store.path) as connection:
        for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'"):
            connection.execute(f"DROP TRIGGER {name}")
        for statement in statements:
            connection.execute(statement)


@pytest.mark.parametrize(
    ("statements", "reason"),
    [
        (("UPDATE bars SET close = 'abc' WHERE rowid = 3",), "integrity"),
        (("DROP TABLE bars",), "database_error"),
    ],
)
def test_a_damaged_store_is_unreadable_never_a_crash(stored, statements, reason):
    _tamper(stored[0], *statements)
    recent = read_recent("NVDA", path=stored[0].path)
    assert recent.state == "unreadable" and recent.reason == reason
