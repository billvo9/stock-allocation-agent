"""
The owner-run recent-window fetch (scripts/stock_lab_fetch_recent.py) on
fakes: off by default (the plan only), read-only with --authorized, and
storage into the separate inference store only with --authorized --store.
The research store is only read. No network.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from stock_lab_inference_helpers import Clock, ny
from test_stock_lab_tiingo import FakeTransport, price_row

from stock_agent.calendars import nyse_rules
from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.inference.schema import DB_NAME
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import Classification, ListingRow, parse_tickers
from stock_agent.stock_lab.tiingo import StockAdapter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_fetch_recent.py"
spec = importlib.util.spec_from_file_location("stock_lab_fetch_recent", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

WATCHED = ("AMD", "MRVL", "MU", "NVDA")
KEY = "fixture-key-not-real-0123"
NOW = ny("2026-10-16", "21:00")  # a Friday evening: the 2026-10-16 close is complete
DAYS = nyse_rules.sessions("2025-10-01", "2026-10-16")[-252:]


@pytest.fixture
def research(tmp_path):
    store = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME, clock=lambda: NOW)
    store.add(parse_tickers("AMD MRVL MU NVDA SKHY SNDK"))
    store.remove(["SKHY", "SNDK"])  # removed by the owner: never requested
    return store


def supported(store: StockLabStore, tickers=WATCHED) -> None:
    rows = [ListingRow(t, "NASDAQ", "Stock", "USD", None, None) for t in tickers]
    snapshot = store.record_listing(
        rows, source="fixture", retrieved_at=NOW, data_class=DataClass.REAL
    )
    store.record_checks(
        [
            Classification(t, SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, "")
            for t in tickers
        ],
        snapshot_id=snapshot,
    )


def prices(start: int = 0, **overrides) -> tuple[int, bytes]:
    rows = [
        price_row(day.strftime("%Y-%m-%d"), close=100.0 + 0.1 * i) for i, day in enumerate(DAYS)
    ]
    for index, values in overrides.items():
        rows[int(index)].update(values)
    return 200, json.dumps(rows[start:]).encode()


def run(research, tmp_path, argv, responses=()):
    transport = FakeTransport(list(responses))
    clock = Clock(NOW)

    def factory(**kwargs):
        def stamped():
            clock.now += pd.Timedelta(seconds=5)
            return clock.now

        return StockAdapter(
            token_provider=lambda: KEY,
            transport=transport,
            sleep=lambda s: None,
            clock=stamped,
            **kwargs,
        )

    inference_path = tmp_path / "inference" / DB_NAME
    out = io.StringIO()
    code = fetch.main(
        argv,
        adapter_factory=factory,
        store_factory=lambda: research,
        inference_factory=lambda: InferenceStore(inference_path, clock=clock),
        clock=clock,
        out=out,
    )
    return code, out.getvalue(), transport, inference_path


def test_the_window_is_252_sessions_ending_at_the_last_complete_close():
    assert fetch.request_window(NOW).equals(DAYS)
    assert fetch.last_complete_session(ny("2026-10-16", "19:59")) == pd.Timestamp(
        "2026-10-15", tz="UTC"
    )
    assert fetch.last_complete_session(ny("2026-10-18", "12:00")) == DAYS[-1]  # a Sunday


def test_without_authorization_it_prints_the_plan_and_makes_no_request(research, tmp_path):
    supported(research)
    before = research.path.read_bytes()
    code, text, transport, inference = run(research, tmp_path, [])
    assert code == 0 and transport.calls == []
    assert "no request made" in text and "AMD, MRVL, MU, NVDA" in text
    assert "SKHY" not in text and "SNDK" not in text
    assert "the 252 NYSE sessions 2025-10-16 to 2026-10-16" in text
    url = "https://api.tiingo.com/tiingo/daily/NVDA/prices?startDate=2025-10-16&endDate=2026-10-16"
    assert url in text and "token=" not in text
    assert str(inference) in text and not inference.exists() and not inference.parent.exists()
    assert research.path.read_bytes() == before


def test_tickers_without_a_supported_check_are_not_requested(research, tmp_path):
    supported(research, ("AMD", "NVDA"))
    _, text, _, _ = run(research, tmp_path, [])
    assert "not requested (no supported listing check): MRVL, MU" in text
    assert "/MRVL/prices" not in text


def test_with_no_supported_ticker_nothing_is_requested(research, tmp_path):
    code, text, transport, inference = run(research, tmp_path, ["--authorized"])
    assert code == 0 and transport.calls == [] and "nothing to fetch" in text
    assert not inference.exists()


def test_store_without_authorization_is_refused(research, tmp_path):
    supported(research)
    code, _, transport, inference = run(research, tmp_path, ["--store"])
    assert code == 2 and transport.calls == [] and not inference.exists()


def test_the_script_has_no_option_beyond_the_authorized_scope(research, tmp_path):
    for argv in (["--authorized", "--tickers", "SKHY"], ["--authorized", "--start", "2025-01-02"]):
        with pytest.raises(SystemExit):
            run(research, tmp_path, argv)


def test_more_than_thirty_tickers_are_refused_before_any_request(research, tmp_path):
    research.add(
        parse_tickers(" ".join(f"A{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(27)))
    )
    code, text, transport, _ = run(research, tmp_path, ["--authorized"])
    assert code == 2 and transport.calls == [] and "at most 30" in text


def test_a_read_only_run_requests_the_window_and_writes_nothing(research, tmp_path):
    supported(research)
    before = research.path.read_bytes()
    code, text, transport, inference = run(
        research, tmp_path, ["--authorized"], [prices() for _ in WATCHED]
    )
    assert code == 0, text
    urls = [url for url, _ in transport.calls]
    assert [url.split("/")[5] for url in urls] == list(WATCHED)
    assert all("startDate=2025-10-16&endDate=2026-10-16" in url for url in urls)
    assert "audit summary: 4 ready" in text and "nothing was written" in text
    assert not inference.exists() and research.path.read_bytes() == before


def test_a_storing_run_writes_each_ready_batch_and_reads_its_window_back(research, tmp_path):
    supported(research)
    before = research.path.read_bytes()
    code, text, _, inference = run(
        research,
        tmp_path,
        ["--authorized", "--store"],
        [prices(), prices(start=1), prices(), prices(**{"100": {"low": 500.0}})],
    )
    assert code == 0, text
    store = InferenceStore(inference)
    assert [len(store.batches(t)) for t in WATCHED] == [1, 0, 1, 0]
    assert text.count("read back: the 2026-10-16 window has 252 sessions and 0 gaps") == 2
    # MRVL's first session was not served: a split there would be invisible, so it waits.
    assert "1 session(s) not served" in text and text.count("not stored: held for review") == 2
    assert "audit summary: 2 ready (AMD, MU); 2 to review (MRVL, NVDA)" in text
    assert research.path.read_bytes() == before  # the research store is never written
    assert research.batches("AMD") == ()


def test_abbreviated_flags_are_refused(research, tmp_path):
    supported(research)
    for argv in (["--auth"], ["--authorized", "--st"]):
        with pytest.raises(SystemExit):
            run(research, tmp_path, argv)


def test_printed_paths_hide_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert fetch.shown(tmp_path / "private" / "x.sqlite3") == "~/private/x.sqlite3"
    assert fetch.shown(Path("/elsewhere/x.sqlite3")) == "/elsewhere/x.sqlite3"


def test_a_failed_request_stops_the_run_without_the_key(research, tmp_path):
    supported(research)
    code, text, _, inference = run(research, tmp_path, ["--authorized", "--store"], [(401, b"")])
    assert code == 1 and "invalid_token" in text and KEY not in text
    assert not inference.exists()


# --- --keychain: the key read from the macOS Keychain at run time (the evening agent) ---

KEYCHAIN_KEY = "keychain-fixture-key-0456"


class Reader:
    """Stands in for keychain.read_key: counts reads, then answers or refuses."""

    def __init__(self, transport=None, refusal=None):
        self.transport, self.refusal, self.reads = transport, refusal, []

    def __call__(self):
        self.reads.append(len(self.transport.calls) if self.transport else 0)
        if self.refusal is not None:
            raise self.refusal
        return KEYCHAIN_KEY


def run_keychain(research, tmp_path, argv, responses=(), refusal=None):
    """`run` with the key reader injected and the adapter built with its token provider."""

    transport = FakeTransport(list(responses))
    clock = Clock(NOW)
    reader = Reader(transport, refusal)
    built = []

    def factory(**kwargs):
        built.append(set(kwargs))
        return StockAdapter(transport=transport, sleep=lambda s: None, clock=clock, **kwargs)

    inference_path = tmp_path / "inference" / DB_NAME
    out = io.StringIO()
    code = fetch.main(
        argv,
        adapter_factory=factory,
        store_factory=lambda: research,
        inference_factory=lambda: InferenceStore(inference_path, clock=clock),
        clock=clock,
        out=out,
        key_reader=reader,
    )
    return code, out.getvalue(), transport, inference_path, reader, built


def test_with_keychain_the_plan_names_the_item_and_reads_nothing(research, tmp_path):
    supported(research)
    code, text, transport, _, reader, built = run_keychain(research, tmp_path, ["--keychain"])
    assert code == 0 and transport.calls == [] and reader.reads == [] and built == []
    assert "service stock-agent-tiingo, account stock-lab" in text and "not read now" in text


def test_with_keychain_the_key_is_read_once_before_any_request_and_sent_only_in_the_header(
    research, tmp_path, monkeypatch
):
    supported(research)
    monkeypatch.setenv("TIINGO_API_KEY", "environment-key-never-used")
    code, text, transport, inference, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], [prices() for _ in WATCHED]
    )
    assert code == 0, text
    assert reader.reads == [0]  # once, and before the first request
    assert built == [{"authorized", "max_requests", "token_provider"}]
    assert [headers for _, headers in transport.calls] == [
        {"Authorization": f"Token {KEYCHAIN_KEY}"}
    ] * len(WATCHED)
    assert all(KEYCHAIN_KEY not in url for url, _ in transport.calls)
    assert KEYCHAIN_KEY not in text and "environment-key" not in text
    assert [len(InferenceStore(inference).batches(t)) for t in WATCHED] == [1, 1, 1, 1]


def test_without_keychain_the_reader_is_never_called(research, tmp_path, monkeypatch):
    supported(research)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    code, text, transport, _, reader, built = run_keychain(
        research, tmp_path, ["--authorized"], [prices() for _ in WATCHED]
    )
    # No --keychain: the adapter keeps its default provider (TIINGO_API_KEY), here unset.
    assert reader.reads == [] and built == [{"authorized", "max_requests"}]
    assert code == 1 and "missing_token" in text and transport.calls == []


def test_an_unreadable_keychain_item_refuses_before_any_request_and_writes_nothing(
    research, tmp_path
):
    from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused

    supported(research)
    refusal = KeychainRefused(
        KeychainFailure.ITEM_NOT_FOUND, service="stock-agent-tiingo", account="stock-lab", status=44
    )
    code, text, transport, inference, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], refusal=refusal
    )
    assert code == 2 and reader.reads == [0] and transport.calls == [] and built == []
    assert "refused: the Tiingo key could not be read from the macOS Keychain" in text
    assert "item_not_found" in text and "nothing was requested" in text
    assert "security add-generic-password -s stock-agent-tiingo -a stock-lab -U -w" in text
    assert not inference.exists()


def test_with_no_supported_ticker_the_keychain_is_never_read(research, tmp_path):
    code, text, transport, _, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"]
    )
    assert code == 0 and "nothing to fetch" in text
    assert reader.reads == [] and transport.calls == []


def test_a_refused_keychain_key_stops_the_run_without_showing_it(research, tmp_path):
    supported(research)
    code, text, transport, inference, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], [(401, b"")]
    )
    assert code == 1 and "invalid_token" in text and reader.reads == [0]
    assert len(transport.calls) == 1 and KEYCHAIN_KEY not in text
    assert not inference.exists()
