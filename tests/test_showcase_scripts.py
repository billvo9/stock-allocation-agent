"""
The owner-run showcase scripts on fakes: the fetch plan, its gates (store,
evening hour, key), storing only audit-ready tickers in the showcase store,
and the run script's refusals and write-once report folder.
"""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from showcase_fixtures import default_frames, sessions_2024, write_config
from test_stock_lab_tiingo import FakeTransport, listing_zip, price_row

from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused
from stock_agent.showcase.store import DB_NAME, open_store
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.tiingo import StockAdapter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load("showcase_fetch")
runner = _load("showcase_run")

LISTING = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "XAA,NYSE ARCA,ETF,USD,1998-12-22,2026-10-23\n"
    "XBB,NYSE ARCA,ETF,USD,1998-12-22,2026-10-23\n"
    "SAA,NYSE,Stock,USD,1980-01-02,2026-10-23\n"
    "SBB,NYSE ARCA,ETF,USD,2001-01-02,2026-10-23\n"
)
MONDAY_NOON = pd.Timestamp("2026-10-26T16:00:00Z")  # 12:00 New York
MONDAY_EVENING = pd.Timestamp("2026-10-27T00:15:00Z")  # 20:15 New York, Monday
SATURDAY_EVENING = pd.Timestamp("2026-10-25T00:15:00Z")  # 20:15 New York, Saturday


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_HOME", str(tmp_path / "lab"))
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_INFERENCE_HOME", str(tmp_path / "inference"))
    monkeypatch.setattr(fetch, "CONFIG", write_config(tmp_path / "tiny.yaml"))
    monkeypatch.setattr(runner, "CONFIG", fetch.CONFIG)


@pytest.fixture
def store(tmp_path):
    return open_store(tmp_path / "showcase" / DB_NAME)


def prices(*days, **extra):
    return (200, json.dumps([price_row(day, **extra) for day in days]).encode())


def run_fetch(store, argv, responses=(), *, clock=MONDAY_NOON, key_reader=None):
    transport = FakeTransport(list(responses))
    made = {}

    def factory(**kwargs):
        kwargs.setdefault("token_provider", lambda: "env-key")
        made.update(kwargs)
        return StockAdapter(transport=transport, sleep=lambda s: None, **kwargs)

    out = io.StringIO()
    code = fetch.main(
        argv,
        adapter_factory=factory,
        store_factory=lambda: store,
        key_reader=key_reader or (lambda: "keychain-key"),
        clock=lambda: clock,
        out=out,
    )
    return code, out.getvalue(), transport


def test_the_plan_makes_no_request_and_asks_for_no_date_after_2024(store):
    code, text, transport = run_fetch(store, [])
    assert code == 0 and transport.calls == []
    assert "no request made" in text and "5 requests in all" in text
    assert text.count("endDate=2024-12-31") == 4 and "2025" not in text
    assert "token=" not in text and "19:30 to 20:45" in text
    assert not store.path.exists()


def test_store_needs_authorization_and_no_option_widens_the_scope(store):
    code, text, transport = run_fetch(store, ["--store"])
    assert code == 2 and transport.calls == [] and "--store needs --authorized" in text
    with pytest.raises(SystemExit):
        run_fetch(store, ["--authorized", "--tickers", "NVDA"])


@pytest.mark.parametrize(
    "local, refused",
    [
        ("2026-10-26 19:29:59", False),  # Monday, EDT
        ("2026-10-26 19:30:00", True),
        ("2026-10-26 20:44:59", True),
        ("2026-10-26 20:45:00", False),
        ("2026-10-30 20:15:00", True),  # Friday evening (Saturday in UTC)
        ("2026-10-25 20:15:00", False),  # Sunday evening (Monday in UTC)
        ("2026-12-14 20:15:00", True),  # Monday, EST
        ("2026-12-14 19:15:00", False),
    ],
)
def test_the_evening_hour_guard_uses_new_york_time(local, refused):
    now = pd.Timestamp(local, tz="America/New_York").tz_convert("UTC")
    assert fetch.in_evening_quiet(now) is refused


@pytest.mark.parametrize("clock, refused", [(MONDAY_EVENING, True), (SATURDAY_EVENING, False)])
def test_an_authorized_run_is_refused_in_the_evening_jobs_hour_on_weekdays(store, clock, refused):
    responses = [(200, listing_zip(LISTING)), *[prices("2024-01-02")] * 3]
    code, text, transport = run_fetch(store, ["--authorized"], responses, clock=clock)
    if refused:
        assert code == 2 and transport.calls == [] and "evening job" in text
    else:
        assert code == 0, text


def test_a_read_only_run_classifies_by_registered_type_and_writes_nothing(store):
    responses = [(200, listing_zip(LISTING)), *[prices("2024-01-02", "2024-01-03")] * 3]
    code, text, transport = run_fetch(store, ["--authorized"], responses)
    assert code == 0, text
    assert "XAA: supported (listed_us_etf)" in text
    assert "SAA: supported (listed_us_stock)" in text
    assert "SBB: unsupported (not_a_stock)" in text  # registered as a stock, listed as an ETF
    urls = [url for url, _ in transport.calls]
    assert len(urls) == 4 and all("endDate=2024-12-31" in u for u in urls[1:])
    assert "SBB/prices" not in " ".join(urls)
    assert "nothing was written" in text and not store.path.exists()
    assert "env-key" not in text


def test_a_storing_run_stores_only_ready_tickers_in_the_showcase_store_only(store, tmp_path):
    bad = [price_row("2024-01-02"), price_row("2024-01-03", low=200.0)]  # low above close
    responses = [
        (200, listing_zip(LISTING)),
        prices("2024-01-02", "2024-01-03"),
        (200, json.dumps(bad).encode()),
        prices("2024-01-02", "2024-01-03", divCash=0.5),
    ]
    code, text, _ = run_fetch(store, ["--authorized", "--store"], responses)
    assert code == 0, text
    assert [b.ticker for b in store.batches("XAA")] == ["XAA"]
    assert store.batches("XBB") == () and "not stored: the audit needs review first" in text
    assert store.batches("SAA")[0].data_class == DataClass.REAL.value
    assert store.latest_listing() is not None and set(store.latest_checks()) == {
        "XAA", "XBB", "SAA", "SBB"
    }  # fmt: skip
    assert store.watched() == ()
    assert not (tmp_path / "lab").exists() and not (tmp_path / "inference").exists()
    assert "stored 2 of 4 tickers" in text and "requests made: 4" in text


def test_a_served_2025_session_refuses_the_response_and_stores_nothing_from_it(store):
    responses = [(200, listing_zip(LISTING)), prices("2024-12-31", "2025-01-02")]
    code, text, transport = run_fetch(store, ["--authorized", "--store"], responses)
    assert code == 1 and "outside_request" in text
    assert store.batches("XAA") == () and len(transport.calls) == 2
    # Records written before the stop stay (write-once); the next run stores the rest.
    assert store.latest_listing() is not None


def test_a_same_day_split_and_dividend_is_reported(store):
    responses = [
        (200, listing_zip(LISTING)),
        prices("2024-01-02"),
        prices("2024-01-02"),
        (200, json.dumps([price_row("2024-01-02"), price_row("2024-01-03", close=50.0,
                                                              divCash=0.2, splitFactor=2.0)]).encode()),
    ]  # fmt: skip
    code, text, _ = run_fetch(store, ["--authorized"], responses)
    assert code == 0, text
    assert "a split and a dividend on the same session (2024-01-03)" in text


def test_with_keychain_the_key_is_read_once_and_a_refusal_stops_before_any_request(store):
    calls = []

    def reader():
        calls.append(1)
        return "keychain-key"

    responses = [(200, listing_zip(LISTING)), *[prices("2024-01-02")] * 3]
    code, text, transport = run_fetch(
        store, ["--authorized", "--keychain"], responses, key_reader=reader
    )
    assert code == 0 and calls == [1] and "keychain-key" not in text
    assert transport.calls[1][1]["Authorization"] == "Token keychain-key"

    def refused():
        raise KeychainRefused(KeychainFailure.ITEM_NOT_FOUND, service="s", account="a")

    code, text, transport = run_fetch(store, ["--authorized", "--keychain"], key_reader=refused)
    assert code == 2 and transport.calls == [] and "nothing was requested" in text


# --- the run script ---


def _filled_store(store, tickers=("XAA", "XBB", "SAA", "SBB")):
    frames = default_frames(sessions_2024())
    for ticker in tickers:
        store.record_prices(
            ticker,
            frames[ticker],
            source="test",
            retrieved_at=pd.Timestamp("2026-10-26T18:00:00Z"),
            data_class=DataClass.REAL,
            vintage_note="test",
        )
    return store


def run_report(store, root):
    out = io.StringIO()
    code = runner.main(
        [],
        store_factory=lambda: store,
        root_factory=lambda: root,
        sha_reader=lambda: "b" * 40,
        now=lambda: datetime(2026, 10, 26, 22, 0, tzinfo=UTC),
        out=out,
    )
    return code, out.getvalue()


def test_the_run_refuses_a_universe_with_a_missing_holding(store, tmp_path):
    _filled_store(store, ("XAA", "XBB", "SAA"))
    code, text = run_report(store, tmp_path / "reports")
    assert code == 2 and "SBB: no stored bars" in text and "nothing was run" in text
    assert not (tmp_path / "reports").exists()


def test_the_run_writes_a_new_private_folder_once(store, tmp_path):
    _filled_store(store)
    code, text = run_report(store, tmp_path / "reports")
    assert code == 0, text
    folder = tmp_path / "reports" / "tiny_showcase-20261026T220000Z"
    assert sorted(p.name for p in folder.iterdir()) == [
        "fonts", "report.html", "report.md", "results.json"
    ]  # fmt: skip
    results = json.loads((folder / "results.json").read_text())
    assert results["variants_run"] == 10 and results["code_sha"] == "b" * 40
    page = (folder / "report.html").read_text()
    assert "Hindsight-selected" in page and "<svg" in page
    assert (folder / "fonts" / "OFL.txt").exists()
    before = (folder / "report.html").read_bytes()
    code, text = run_report(store, tmp_path / "reports")  # the same run id: never overwritten
    assert code == 1 and "already exists" in text
    assert (folder / "report.html").read_bytes() == before


def test_the_run_refuses_a_stored_2025_session(store, tmp_path):
    _filled_store(store, ("XAA", "XBB", "SAA"))
    days = pd.DatetimeIndex(["2024-12-31", "2025-01-02"], tz="UTC").as_unit("us")
    frame = default_frames(sessions_2024())["SBB"].iloc[-1:]
    late = pd.concat([frame, frame.assign(session=days[1])], ignore_index=True)
    late["session"] = days
    store.record_prices(
        "SBB",
        late,
        source="test",
        retrieved_at=pd.Timestamp("2026-10-26T18:00:00Z"),
        data_class=DataClass.REAL,
        vintage_note="test",
    )
    code, text = run_report(store, tmp_path / "reports")
    assert code == 2 and "leave the window" in text and not (tmp_path / "reports").exists()


def test_a_build_that_cannot_execute_refuses_the_run_and_writes_nothing(store, tmp_path):
    frames = default_frames(sessions_2024())
    frames["XAA"].loc[21, ["open", "high", "low", "close", "volume"]] = float("nan")
    for ticker, frame in frames.items():
        store.record_prices(
            ticker,
            frame,
            source="test",
            retrieved_at=pd.Timestamp("2026-10-26T18:00:00Z"),
            data_class=DataClass.REAL,
            vintage_note="test",
        )
    code, text = run_report(store, tmp_path / "reports")
    assert code == 1 and "build could not execute" in text
    assert not (tmp_path / "reports").exists()


def test_the_report_folder_may_not_be_inside_a_git_tree(store, tmp_path):
    _filled_store(store)
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    code, text = run_report(store, tmp_path / "repo" / "reports")
    assert code == 2 and "outside every Git working tree" in text
