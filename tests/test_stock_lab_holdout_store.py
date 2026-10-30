"""
The private holdout store (stock_lab.holdout.store): one write-once batch per
protocol stock, and one evaluation, ever. The bars are handed out only inside
the run that records the evaluation, and never once it exists. Fixture bars
on a weekday calendar; no network, no real data.
"""

from __future__ import annotations

import sqlite3
import stat
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from stock_lab_holdout_helpers import (
    CALENDAR_ID,
    RETRIEVED,
    fill,
    holdout_bars,
    ny,
    open_store,
    short_protocol,
    weekdays,
)

from stock_agent.stock_lab.holdout.protocol import HoldoutRefused, Refusal, record_id
from stock_agent.stock_lab.holdout.store import (
    DB_NAME,
    HoldoutStore,
    bars_sha256,
    check_location,
)
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore, StoreRefused
from stock_agent.stock_lab.symbols import parse_tickers

PROTOCOL = short_protocol("a" * 64)


@contextmanager
def refused(reason: Refusal):
    with pytest.raises(HoldoutRefused) as caught:
        yield caught
    assert caught.value.reason is reason, caught.value


def sealed(**content) -> dict:
    record = {
        "protocol_sha256": PROTOCOL.sha256(),
        "model": {"model_id": PROTOCOL.model_id},
        "result": 1,
        **content,
    }
    return {**record, "record_id": record_id(record)}


def _store(tmp_path):
    return open_store(tmp_path, PROTOCOL)


def _record(store, ticker="AMD", **change):
    arguments = {
        "source": "fixture",
        "retrieved_at": RETRIEVED,
        "sessions": weekdays,
        "calendar_id": CALENDAR_ID,
        "vintage_note": "fixture bars",
        **change,
    }
    bars = arguments.pop("bars", holdout_bars(PROTOCOL, ticker))
    return store.record_batch(ticker, bars, **arguments)


def test_a_batch_is_stored_once_per_stock_with_its_lineage(tmp_path):
    store, _ = _store(tmp_path)
    batch = _record(store)
    bars = holdout_bars(PROTOCOL, "AMD")
    assert (batch.ticker, batch.request_start, batch.request_end) == (
        "AMD",
        PROTOCOL.fetch_start,
        PROTOCOL.holdout_end,
    )
    assert batch.row_count == len(bars) and batch.content_sha256 == bars_sha256(bars)
    assert batch.protocol_sha256 == PROTOCOL.sha256() and batch.calendar_id == CALENDAR_ID
    assert store.batches() == (batch,)
    with refused(Refusal.ALREADY_FETCHED):
        _record(store)  # even identical bars: a stock is fetched once
    with refused(Refusal.ALREADY_FETCHED):
        _record(store, bars=holdout_bars(PROTOCOL, "MU"))
    assert store.batches() == (batch,)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


def test_only_the_protocols_stocks_and_sessions_are_stored(tmp_path):
    store, _ = _store(tmp_path)
    with refused(Refusal.INVALID_REQUEST):
        store.record_batch("AAPL", holdout_bars(PROTOCOL, "AMD"), source="fixture",
                           retrieved_at=RETRIEVED, sessions=weekdays, calendar_id=CALENDAR_ID,
                           vintage_note="fixture")  # fmt: skip
    bars = holdout_bars(PROTOCOL, "AMD")
    with refused(Refusal.INVALID_BARS):
        _record(store, bars=bars.iloc[1:])  # starts a session late
    with refused(Refusal.INVALID_BARS):
        _record(store, bars=bars.drop(index=40))  # a session missing, not a gap row
    with refused(Refusal.INVALID_BARS):
        _record(store, bars=bars.assign(close=-bars["close"]))
    assert not store.exists  # nothing was created by a refused write


def test_a_batch_is_recorded_when_it_is_retrieved_after_the_last_close(tmp_path):
    store, clock = _store(tmp_path)
    with refused(Refusal.RETRIEVAL_TIME):
        _record(store, retrieved_at=clock.now + pd.Timedelta(seconds=1))
    with refused(Refusal.RETRIEVAL_TIME):
        _record(store, retrieved_at=clock.now - pd.Timedelta(hours=2))
    clock.now = pd.Timestamp(f"{PROTOCOL.holdout_end} 23:30", tz="UTC")
    with refused(Refusal.RETRIEVAL_TIME):  # 18:30 in New York (EST): the close is not final
        _record(store, retrieved_at=pd.Timestamp(f"{PROTOCOL.holdout_end} 23:30", tz="UTC"))
    with refused(Refusal.INVALID_REQUEST):
        _record(store, retrieved_at=pd.Timestamp("2026-10-27 21:00"))  # naive
    assert not store.exists


def test_the_tables_are_write_once(tmp_path):
    store, _ = _store(tmp_path)
    _record(store)
    connection = sqlite3.connect(store.path)
    for statement in (
        "UPDATE bars SET close = 1.0",
        "DELETE FROM bars",
        "UPDATE batches SET ticker = 'MU'",
        "DELETE FROM batches",
        "UPDATE meta SET value = 'x'",
        "INSERT INTO bars SELECT * FROM bars LIMIT 1",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)
    connection.close()


def test_the_run_gets_every_stocks_stored_bars_and_its_record_is_kept(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL)
    seen = {}

    def run(histories, batches):
        seen["histories"], seen["batches"] = histories, batches
        return sealed()

    record = store.evaluate_once(run, calendar_id=CALENDAR_ID)
    assert record == sealed()
    assert sorted(seen["histories"]) == list(PROTOCOL.tickers)
    for ticker, bars in seen["histories"].items():
        expected = holdout_bars(PROTOCOL, ticker)
        pd.testing.assert_frame_equal(bars, expected, check_dtype=False)
    assert seen["batches"] == store.batches()
    assert store.evaluation() == record


def test_the_holdout_is_spent_after_its_one_evaluation(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL, tickers=("AMD", "MRVL", "MU"))
    with refused(Refusal.INCOMPLETE):
        store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    fill(store, PROTOCOL, tickers=("NVDA",))
    store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    calls = []
    with refused(Refusal.SPENT):
        store.evaluate_once(lambda h, b: calls.append(h) or sealed(result=2),
                            calendar_id=CALENDAR_ID)  # fmt: skip
    assert calls == []  # the bars were not handed out again
    with refused(Refusal.SPENT):
        _record(store, ticker="NVDA")
    connection = sqlite3.connect(store.path)
    for statement in (  # the triggers hold even without the store's checks
        (
            "INSERT INTO batches SELECT 'x', 'ZZZ', source, retrieved_at, recorded_at, "
            "request_start, request_end, calendar_id, row_count, content_sha256, "
            "protocol_sha256, vintage_note FROM batches LIMIT 1"
        ),
        (
            "INSERT INTO evaluation SELECT 2, 'm', protocol_sha256, 'r', recorded_at, record "
            "FROM evaluation"
        ),
        "DELETE FROM evaluation",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)
    connection.close()
    assert store.evaluation() == sealed()


def test_a_failed_or_unsealed_run_records_nothing(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL)

    def broken(histories, batches):
        raise RuntimeError("a bug in the run")

    with pytest.raises(RuntimeError):
        store.evaluate_once(broken, calendar_id=CALENDAR_ID)
    assert store.evaluation() is None
    unsealed = {**sealed(), "result": 3}  # its id no longer matches its content
    with refused(Refusal.INTEGRITY):
        store.evaluate_once(lambda h, b: unsealed, calendar_id=CALENDAR_ID)
    other = sealed(protocol_sha256="0" * 64)
    with refused(Refusal.INTEGRITY):
        store.evaluate_once(lambda h, b: other, calendar_id=CALENDAR_ID)
    assert store.evaluation() is None
    store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)  # nothing seen: may run
    assert store.evaluation() == sealed()


def test_batches_of_another_protocol_or_calendar_are_refused(tmp_path):
    store, clock = _store(tmp_path)
    fill(store, PROTOCOL)
    other = HoldoutStore(store.path, protocol=short_protocol("b" * 64), clock=clock)
    with refused(Refusal.PROTOCOL):
        other.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    with refused(Refusal.PROTOCOL):
        store.evaluate_once(lambda h, b: sealed(), calendar_id="another_calendar")
    assert store.evaluation() is None


def test_tampered_bars_or_records_are_refused(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL)
    connection = sqlite3.connect(store.path)
    connection.execute("DROP TRIGGER bars_no_update")
    connection.execute("UPDATE bars SET close = close * 1.01 WHERE session = '2025-01-10'")
    connection.commit()
    with refused(Refusal.INTEGRITY):
        store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    connection.execute("UPDATE bars SET close = close / 1.01 WHERE session = '2025-01-10'")
    connection.commit()
    store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    connection.execute("DROP TRIGGER evaluation_no_update")
    connection.execute(
        "UPDATE evaluation SET record = replace(record, '\"result\":1', '\"result\":9')"
    )
    connection.commit()
    connection.close()
    with refused(Refusal.INTEGRITY):
        store.evaluation()


def test_its_public_surface_returns_no_prices(tmp_path):
    names = {name for name in dir(HoldoutStore) if not name.startswith("_")}
    assert names == {"batches", "evaluation", "evaluate_once", "exists", "record_batch"}


def test_its_location_is_private_and_its_own(tmp_path):
    working_tree = tmp_path / "repository"
    (working_tree / ".git").mkdir(parents=True)
    with refused(Refusal.LOCATION):
        check_location(working_tree / "private" / DB_NAME)
    with refused(Refusal.LOCATION):
        check_location(tmp_path / "holdout" / "other.sqlite3")
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    StockLabStore(shared / RESEARCH_DB_NAME).add(parse_tickers("NVDA"))
    with refused(Refusal.LOCATION):
        HoldoutStore(shared / DB_NAME, protocol=PROTOCOL)
    assert check_location(tmp_path / "own" / DB_NAME) == (tmp_path / "own" / DB_NAME).resolve()


def test_the_research_store_refuses_the_holdout_file_and_this_store_refuses_others(tmp_path):
    store, clock = _store(tmp_path)
    _record(store)
    before = store.path.read_bytes()
    research = StockLabStore(store.path, clock=clock)
    for read in (research.watched, lambda: research.bars("AMD"), research.job_runs):
        with pytest.raises(StoreRefused, match="not a research store"):
            read()
    with pytest.raises(StoreRefused, match="not a research store"):
        research.add(parse_tickers("NVDA"))
    assert store.path.read_bytes() == before
    foreign = tmp_path / "foreign" / DB_NAME
    foreign.parent.mkdir(mode=0o700)
    connection = sqlite3.connect(foreign)
    connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    connection.close()
    foreign.chmod(0o600)
    other = HoldoutStore(foreign, protocol=PROTOCOL, clock=clock)
    with refused(Refusal.NOT_HOLDOUT_STORE):
        other.batches()
    with refused(Refusal.NOT_HOLDOUT_STORE):
        _record(other)


def test_reading_a_missing_store_creates_nothing(tmp_path):
    store, _ = _store(tmp_path)
    assert store.batches() == () and store.evaluation() is None
    assert not store.path.parent.exists()


def test_a_directory_or_file_open_to_other_users_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    store.path.parent.mkdir(mode=0o700)
    store.path.parent.chmod(0o755)
    with refused(Refusal.LOCATION):
        _record(store)
    store.path.parent.chmod(0o700)
    store.path.touch(mode=0o600)
    store.path.chmod(0o644)
    with refused(Refusal.LOCATION):
        _record(store)


def test_a_stored_record_that_is_not_its_canonical_text_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL)
    store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    connection = sqlite3.connect(store.path)
    connection.execute("DROP TRIGGER evaluation_no_update")
    for text in (
        '{"result": 1}',  # spacing: the same content, not the text stored
        "[1, 2]",
        "{not json",
    ):
        connection.execute("UPDATE evaluation SET record = ?", (text,))
        connection.commit()
        with refused(Refusal.INTEGRITY):
            store.evaluation()
    connection.close()


def test_the_default_location_follows_its_variable(monkeypatch, tmp_path):
    from stock_agent.stock_lab.holdout.store import HOME_ENV, default_path

    monkeypatch.setenv(HOME_ENV, str(tmp_path / "chosen"))
    assert default_path() == tmp_path / "chosen" / DB_NAME
    monkeypatch.delenv(HOME_ENV)
    assert default_path() == (Path.home() / "stock-agent-private" / "stock_lab_holdout" / DB_NAME)


def test_the_triggers_hold_bars_without_the_stores_checks(tmp_path):
    store, _ = _store(tmp_path)
    fill(store, PROTOCOL)
    connection = sqlite3.connect(store.path)
    new_bar = (
        "INSERT INTO bars SELECT batch_id, '2030-01-02', open, high, low, close, volume, "
        "div_cash, split_factor FROM bars LIMIT 1"
    )
    with pytest.raises(sqlite3.DatabaseError, match="the batch is complete"):
        connection.execute(new_bar)
    connection.execute("DROP TRIGGER bars_complete")
    connection.commit()
    connection.close()
    store.evaluate_once(lambda h, b: sealed(), calendar_id=CALENDAR_ID)
    connection = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="the holdout is spent"):
        connection.execute(new_bar)
    connection.close()


def test_a_batch_retrieved_at_the_close_is_accepted(tmp_path):
    store, clock = _store(tmp_path)
    close = ny(PROTOCOL.holdout_end, "20:00")
    clock.now = close + pd.Timedelta(minutes=59)
    with refused(Refusal.RETRIEVAL_TIME):
        _record(store, retrieved_at=close - pd.Timedelta(seconds=1))
    assert _record(store, retrieved_at=close).retrieved_at.startswith("2025-02-15T01:00:00")
