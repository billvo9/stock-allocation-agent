"""
The private SQLite store (stock_agent.stock_lab.store): a watchlist that
survives restarts, write-once vendor data with vintages, and privacy rules.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.store import (
    DB_NAME,
    HOME_ENV,
    StockLabStore,
    StoreRefused,
    default_path,
)
from stock_agent.stock_lab.symbols import Classification, ListingRow, parse_tickers

T0 = pd.Timestamp("2026-10-16T21:00:00Z")
REPOSITORY = Path(__file__).resolve().parents[1]
# Hard-coded (not read from the module), so dropping a table's protection fails a test.
RECORDED_TABLES = (
    "watchlist_events",
    "symbol_checks",
    "listing_snapshots",
    "listing_rows",
    "price_batches",
    "price_bars",
)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        self.now = self.now + pd.Timedelta(seconds=1)
        return self.now


@pytest.fixture
def store(tmp_path) -> StockLabStore:
    return StockLabStore(tmp_path / "private" / DB_NAME, clock=Clock())


def real_bars(n=30, start="2024-01-02"):
    bars = synthetic.demo_bars("DEMO_CALM").iloc[:n].copy()
    bars["session"] = pd.bdate_range(start, periods=n, tz="UTC").as_unit("us")
    return bars


def test_a_store_that_does_not_exist_reads_as_empty_and_creates_nothing(store):
    assert store.watchlist() == () and store.batches("AAPL") == () and store.bars("AAPL") is None
    assert store.latest_listing() is None and store.latest_checks() == {}
    assert store.remove(["AAPL"]) == ()
    assert not store.path.exists() and not store.path.parent.exists()


def test_the_watchlist_survives_a_restart_and_keeps_its_history(store):
    added = store.add(parse_tickers("msft aapl brk.b AAPL VOD.L"))
    assert [(r.ticker, r.added) for r in added] == [("MSFT", True), ("AAPL", True), ("BRK-B", True)]
    again = StockLabStore(store.path)  # a new process would open the same file
    assert again.watched() == ("AAPL", "BRK-B", "MSFT")
    assert [item.entered_as for item in again.watchlist()] == ["aapl", "brk.b", "msft"]
    assert again.add(parse_tickers("aapl"))[0].added is False
    assert again.remove(["MSFT", "ZZZ"]) == ("MSFT",)
    assert StockLabStore(store.path).watched() == ("AAPL", "BRK-B")
    store.add(parse_tickers("MSFT"))
    events = store.events()
    assert events["action"].tolist() == ["add", "add", "add", "remove", "add"]
    assert events["ticker"].tolist() == ["MSFT", "AAPL", "BRK-B", "MSFT", "MSFT"]


def test_the_store_is_private(store):
    store.add(parse_tickers("AAPL"))
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(store.path.parent).st_mode) == 0o700


def test_a_location_inside_the_repository_is_refused_unless_ignored_data_raw():
    with pytest.raises(StoreRefused):
        StockLabStore(REPOSITORY / "src" / DB_NAME)
    with pytest.raises(StoreRefused):
        StockLabStore(REPOSITORY / DB_NAME)
    StockLabStore(REPOSITORY / "data" / "raw" / "stock_lab" / DB_NAME)  # allowed, not created


def test_any_git_working_tree_is_refused_not_only_this_one(tmp_path):
    other = tmp_path / "another-checkout"
    (other / ".git").mkdir(parents=True)
    with pytest.raises(StoreRefused, match="Git working tree"):
        StockLabStore(other / "notes" / DB_NAME)
    StockLabStore(other / "data" / "raw" / DB_NAME)


def test_an_existing_directory_open_to_others_is_refused(tmp_path):
    loose = tmp_path / "shared"
    loose.mkdir()
    loose.chmod(0o755)
    store = StockLabStore(loose / DB_NAME)
    with pytest.raises(StoreRefused, match="chmod 700"):
        store.add(parse_tickers("AAPL"))
    assert not store.path.exists()


def test_the_default_location_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(HOME_ENV, str(tmp_path))
    assert default_path() == tmp_path / DB_NAME
    monkeypatch.delenv(HOME_ENV)
    assert default_path().parts[-3:] == ("stock-agent-private", "stock_lab", DB_NAME)


def _filled(store):
    store.add(parse_tickers("AAPL"))
    snapshot = store.record_listing(
        [ListingRow("AAPL", "NASDAQ", "Stock", "USD", date(1980, 12, 12), date(2026, 10, 15))],
        source="fixture",
        retrieved_at=T0,
        data_class=DataClass.SYNTHETIC,
    )
    store.record_checks(
        [Classification("AAPL", SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, "ok")],
        snapshot_id=snapshot,
    )
    return store.record_prices(
        "AAPL",
        real_bars(),
        source="fixture",
        retrieved_at=T0,
        data_class=DataClass.REAL,
        vintage_note="fixture",
    )


def _connect(store):
    # A plain connection, as any other writer would open: no pragmas set.
    return sqlite3.connect(store.path)


@pytest.mark.parametrize("table", RECORDED_TABLES)
def test_every_recorded_table_is_write_once(store, table):
    _filled(store)
    connection = _connect(store)
    try:
        for statement in (f"DELETE FROM {table}", f"UPDATE {table} SET rowid = rowid"):
            with pytest.raises(sqlite3.IntegrityError, match="write-once"):
                connection.execute(statement)
        row = connection.execute(f"SELECT rowid, * FROM {table} LIMIT 1").fetchone()
        columns = [d[1] for d in connection.execute(f"PRAGMA table_info({table})")]
        marks = ", ".join("?" for _ in row)
        with pytest.raises(sqlite3.IntegrityError, match="write-once"):
            connection.execute(
                f"INSERT OR REPLACE INTO {table} (rowid, {', '.join(columns)}) VALUES ({marks})",
                row,
            )
    finally:
        connection.close()


def test_a_recorded_batch_cannot_gain_bars(store):
    batch = _filled(store)
    connection = _connect(store)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="complete"):
            connection.execute(
                "INSERT INTO price_bars VALUES (?, '2030-01-02', 1, 1, 1, 1, 1, 0, 1)",
                (batch.batch_id,),
            )
    finally:
        connection.close()


def test_reads_refuse_a_batch_changed_outside_the_store(store):
    batch = _filled(store)
    connection = sqlite3.connect(store.path)
    connection.execute("DROP TRIGGER price_bars_no_update")  # tampering around the rules
    connection.execute(
        "UPDATE price_bars SET close = 999 WHERE batch_id = ? AND session = ?",
        (batch.batch_id, batch.first_session),
    )
    connection.commit()
    connection.close()
    with pytest.raises(StoreRefused, match="content hash"):
        StockLabStore(store.path).bars("AAPL")


def test_price_batches_are_vintages_never_overwritten(store):
    bars = real_bars(30)
    first = store.record_prices(
        "AAPL",
        bars.iloc[:20],
        source="f",
        retrieved_at=T0,
        data_class=DataClass.REAL,
        vintage_note="v",
    )
    revised = bars.iloc[15:].copy()
    revised.loc[revised.index[0], "close"] *= 1.01  # the vendor corrected a close
    later = T0 + pd.Timedelta(days=1)
    second = store.record_prices(
        "AAPL",
        revised,
        source="f",
        retrieved_at=later,
        data_class=DataClass.REAL,
        vintage_note="v",
    )
    stored = store.bars("AAPL")
    assert len(stored.bars) == 30 and [b.batch_id for b in stored.batches] == [
        first.batch_id,
        second.batch_id,
    ]
    corrected = bars["session"].iloc[15].strftime("%Y-%m-%d")
    assert stored.revised_sessions == (corrected,)
    assert stored.bars.loc[15, "close"] == pytest.approx(revised["close"].iloc[0])
    before = store.bars("AAPL", as_of=T0 + pd.Timedelta(hours=1))
    assert len(before.bars) == 20 and before.revised_sessions == ()
    assert before.bars.loc[15, "close"] == pytest.approx(bars["close"].iloc[15])
    assert store.bars("AAPL", as_of=T0 - pd.Timedelta(days=1)) is None
    assert len(store.bars("AAPL", as_of=T0).bars) == 20  # a batch retrieved at as_of counts
    assert store.bars("AAPL", as_of=T0 - pd.Timedelta(microseconds=1)) is None


def test_an_identical_retrieval_is_not_stored_twice(store):
    kwargs = {"source": "f", "data_class": DataClass.REAL, "vintage_note": "v"}
    one = store.record_prices("AAPL", real_bars(), retrieved_at=T0, **kwargs)
    two = store.record_prices("AAPL", real_bars(), retrieved_at=T0 + pd.Timedelta(days=1), **kwargs)
    assert one == two and len(store.batches("AAPL")) == 1


def test_synthetic_bars_go_only_under_demo_identifiers_and_real_ones_never_do(store):
    kwargs = {"source": "f", "retrieved_at": T0, "vintage_note": "v"}
    with pytest.raises(StoreRefused, match="demo identifier"):
        store.record_prices("AAPL", real_bars(), data_class=DataClass.SYNTHETIC, **kwargs)
    with pytest.raises(StoreRefused, match="demo identifier"):
        store.record_prices("DEMO_CALM", real_bars(), data_class=DataClass.REAL, **kwargs)
    store.record_prices("DEMO_CALM", real_bars(), data_class=DataClass.SYNTHETIC, **kwargs)
    assert store.bars("DEMO_CALM").data_class == "synthetic"


def test_naive_timestamps_and_empty_writes_are_refused(store):
    with pytest.raises(StoreRefused):
        store.record_prices(
            "AAPL",
            real_bars(),
            source="f",
            retrieved_at=pd.Timestamp("2026-10-16"),
            data_class=DataClass.REAL,
            vintage_note="v",
        )
    with pytest.raises(StoreRefused):
        store.record_prices(
            "AAPL",
            real_bars().iloc[0:0],
            source="f",
            retrieved_at=T0,
            data_class=DataClass.REAL,
            vintage_note="v",
        )
    with pytest.raises(StoreRefused):
        store.record_listing([], source="f", retrieved_at=T0, data_class=DataClass.REAL)


def test_listing_snapshots_and_the_latest_check_per_ticker(store):
    rows = [
        ListingRow("AAPL", "NASDAQ", "Stock", "USD", date(1980, 12, 12), date(2026, 10, 15)),
        ListingRow("VOO", "NYSE ARCA", "ETF", "USD", date(2010, 9, 9), None),
    ]
    snapshot = store.record_listing(rows, source="f", retrieved_at=T0, data_class=DataClass.REAL)
    meta, listing = store.latest_listing()
    assert meta["snapshot_id"] == snapshot and listing == tuple(rows)
    assert store.listing_rows_for("VOO") == (rows[1],)
    first = Classification("AAPL", SymbolStatus.INACTIVE, SymbolReason.NO_RECENT_PRICES, "a")
    second = Classification("AAPL", SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, "b")
    store.record_checks([first], snapshot_id=snapshot)
    store.record_checks([second], snapshot_id=snapshot)
    assert store.latest_checks() == {"AAPL": second}


def test_a_check_must_name_a_stored_snapshot(store):
    store.add(parse_tickers("AAPL"))
    check = Classification("AAPL", SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, "x")
    with pytest.raises(sqlite3.IntegrityError):
        store.record_checks([check], snapshot_id="missing")


def test_an_unknown_schema_version_is_refused(store):
    store.add(parse_tickers("AAPL"))
    connection = sqlite3.connect(store.path)
    connection.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(StoreRefused, match="schema version"):
        store.watchlist()


def test_the_page_shows_real_bars_only_through_2024_and_synthetic_ones_whole(store):
    from stock_agent.stock_lab.view import real_bars_shown

    bars = synthetic.demo_bars("DEMO_CALM").iloc[-300:].copy()
    bars["session"] = pd.bdate_range("2024-10-01", periods=300, tz="UTC").as_unit("us")
    kwargs = {"source": "f", "retrieved_at": T0, "vintage_note": "v"}
    store.record_prices("AAPL", bars, data_class=DataClass.REAL, **kwargs)
    store.record_prices("DEMO_CALM", bars, data_class=DataClass.SYNTHETIC, **kwargs)
    real = real_bars_shown(store.bars("AAPL"))
    assert real["session"].max() == pd.Timestamp("2024-12-31", tz="UTC")
    assert len(real_bars_shown(store.bars("DEMO_CALM"))) == 300
    assert real_bars_shown(None) is None
