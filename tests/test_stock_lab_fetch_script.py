"""The owner-run fetch script (scripts/stock_lab_fetch.py) on fakes: plan, gates, storage."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from test_stock_lab_tiingo import FakeTransport, listing_zip, price_row

from stock_agent.stock_lab.store import DB_NAME, StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers
from stock_agent.stock_lab.tiingo import StockAdapter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_fetch.py"
spec = importlib.util.spec_from_file_location("stock_lab_fetch", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

LISTING = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "AAPL,NASDAQ,Stock,USD,1980-12-12,2026-10-15\n"
    "VOO,NYSE ARCA,ETF,USD,2010-09-09,2026-10-15\n"
)


@pytest.fixture
def store(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    store.add(parse_tickers("AAPL VOO"))
    return store


def run(store, argv, responses=()):
    transport = FakeTransport(list(responses))

    def factory(**kwargs):
        return StockAdapter(
            token_provider=lambda: "k", transport=transport, sleep=lambda s: None, **kwargs
        )

    out = io.StringIO()
    code = fetch.main(argv, adapter_factory=factory, store_factory=lambda: store, out=out)
    return code, out.getvalue(), transport


def prices():
    # 2024-01-02..2024-01-05 are NYSE sessions; 2024-01-04 is left out (a gap).
    rows = [price_row(day) for day in ("2024-01-02", "2024-01-03", "2024-01-05")]
    return json.dumps(rows).encode()


def test_without_authorization_it_prints_the_plan_and_makes_no_request(store):
    code, text, transport = run(store, [])
    assert code == 0 and transport.calls == []
    assert "no request made" in text and "AAPL, VOO" in text
    assert "1. GET https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip" in text
    assert (
        "2. GET https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate=2003-01-02&endDate=2024-12-31"
        in text
    )
    assert "local side effects of --authorized without --store: none" in text
    assert "token=" not in text  # the key never travels in a URL (Tiingo also accepts ?token=)


def test_store_without_authorization_is_refused(store):
    code, _, transport = run(store, ["--store"])
    assert code == 2 and transport.calls == []


def test_more_than_thirty_tickers_are_refused_before_any_request(store):
    store.add(parse_tickers(" ".join(f"A{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(29))))
    assert len(store.watched()) == 31
    code, text, transport = run(store, ["--authorized"])
    assert code == 2 and transport.calls == [] and "at most 30" in text


def test_the_script_has_no_option_beyond_the_authorized_scope(store):
    with pytest.raises(SystemExit):
        run(store, ["--authorized", "--tickers", "NVDA"])
    with pytest.raises(SystemExit):
        run(store, ["--authorized", "--max-tickers", "500"])


def test_a_storing_run_stores_only_tickers_whose_audit_is_ready(store):
    rows = [price_row(day) for day in ("2024-01-02", "2024-01-03", "2024-01-05")]
    rows[2]["low"] = 200.0  # a low above the close: the audit asks for review
    code, text, _ = run(
        store,
        ["--authorized", "--store"],
        [(200, listing_zip(LISTING)), (200, json.dumps(rows).encode())],
    )
    assert code == 0, text
    assert "not stored: the audit needs review first" in text and store.batches("AAPL") == ()
    assert "audit summary: 0 ready" in text


def test_a_read_only_run_classifies_requests_pre_2025_prices_and_stores_nothing(store):
    database = store.path.read_bytes()
    events = store.events()
    code, text, transport = run(
        store, ["--authorized"], [(200, listing_zip(LISTING)), (200, prices())]
    )
    assert code == 0, text
    urls = [url for url, _ in transport.calls]
    assert len(urls) == 2 and "AAPL/prices?startDate=2003-01-02&endDate=2024-12-31" in urls[1]
    assert "VOO: unsupported (not_a_stock)" in text
    assert "1 missing NYSE sessions" in text and "audit: ready to store" in text
    assert "nothing was written" in text
    assert store.batches("AAPL") == () and store.latest_listing() is None
    # The database file and its records are unchanged (reading may add SQLite's
    # -shm/-wal companion files, which hold no record).
    assert store.path.read_bytes() == database and store.events().equals(events)


def test_a_storing_run_writes_the_listing_checks_and_a_batch_with_the_gap_kept(store):
    code, text, _ = run(
        store, ["--authorized", "--store"], [(200, listing_zip(LISTING)), (200, prices())]
    )
    assert code == 0, text
    stored = store.bars("AAPL")
    assert stored.data_class == "real" and len(stored.bars) == 4
    assert stored.bars["close"].isna().tolist() == [False, False, True, False]
    checks = store.latest_checks()
    assert (
        checks["AAPL"].status.value == "supported" and checks["VOO"].status.value == "unsupported"
    )
    assert stored.batches[0].retrieved_at.endswith("Z")


def test_a_listing_without_end_dates_stops_the_run(store):
    empty = LISTING.replace("2026-10-15", "")
    code, text, _ = run(store, ["--authorized"], [(200, listing_zip(empty))])
    assert code == 1 and "listing_dates" in text


def test_a_failed_request_stops_the_run_without_the_key(store):
    code, text, _ = run(store, ["--authorized", "--store"], [(401, b"")])
    assert code == 1 and "http_error" in text and store.latest_listing() is None
    code, text, _ = run(store, ["--authorized"], [(200, listing_zip(LISTING)), (401, b"")])
    assert code == 1 and "invalid_token" in text
    assert store.batches("AAPL") == ()


def test_weekdays_after_counts_weekdays():
    from datetime import date

    assert fetch.weekdays_after(date(2026, 10, 9), date(2026, 10, 16)) == 5
    assert fetch.weekdays_after(date(2026, 10, 16), date(2026, 10, 16)) == 0
    assert pd.Timestamp("2026-10-16").dayofweek == 4
