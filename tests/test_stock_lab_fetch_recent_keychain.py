"""
The recent-window fetch with --keychain (scripts/stock_lab_fetch_recent.py), as
the unattended evening agent runs it: the key is read from the macOS Keychain
once, before the first request, and reaches the adapter in memory only. An
item that cannot be read stops the run before any request or write. Fakes
only: no Keychain, no network.
"""

from __future__ import annotations

import io

import pytest
import test_stock_lab_fetch_recent_script as base
from stock_lab_inference_helpers import Clock
from test_stock_lab_tiingo import FakeTransport

from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused
from stock_agent.stock_lab.inference.schema import DB_NAME
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers
from stock_agent.stock_lab.tiingo import StockAdapter

fetch = base.fetch
WATCHED = base.WATCHED
KEYCHAIN_KEY = "keychain-fixture-key-0456"


@pytest.fixture
def research(tmp_path):
    store = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME, clock=lambda: base.NOW)
    store.add(parse_tickers(" ".join(WATCHED)))
    return store


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
    """The script with the key reader injected and the adapter built with its token provider."""

    transport = FakeTransport(list(responses))
    clock = Clock(base.NOW)
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
    base.supported(research)
    code, text, transport, _, reader, built = run_keychain(research, tmp_path, ["--keychain"])
    assert code == 0 and transport.calls == [] and reader.reads == [] and built == []
    assert "service stock-agent-tiingo, account stock-lab" in text and "not read now" in text


def test_with_keychain_the_key_is_read_once_before_any_request_and_sent_only_in_the_header(
    research, tmp_path, monkeypatch
):
    base.supported(research)
    monkeypatch.setenv("TIINGO_API_KEY", "environment-key-never-used")
    code, text, transport, inference, reader, built = run_keychain(
        research,
        tmp_path,
        ["--authorized", "--store", "--keychain"],
        [base.prices() for _ in WATCHED],
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
    base.supported(research)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    code, text, transport, _, reader, built = run_keychain(
        research, tmp_path, ["--authorized"], [base.prices() for _ in WATCHED]
    )
    # No --keychain: the adapter keeps its default provider (TIINGO_API_KEY), here unset.
    assert reader.reads == [] and built == [{"authorized", "max_requests"}]
    assert code == 1 and "missing_token" in text and transport.calls == []


@pytest.mark.parametrize("reason", list(KeychainFailure))
def test_an_unreadable_keychain_item_stops_the_run_before_any_request_or_write(
    research, tmp_path, reason
):
    base.supported(research)
    before = research.path.read_bytes()
    refusal = KeychainRefused(reason, service="stock-agent-tiingo", account="stock-lab")
    code, text, transport, inference, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], refusal=refusal
    )
    assert code == 2 and reader.reads == [0] and transport.calls == [] and built == []
    assert "refused: the Tiingo key could not be read from the macOS Keychain" in text
    assert reason.value in text and "nothing was requested" in text
    assert "security add-generic-password -s stock-agent-tiingo -a stock-lab -U -w" in text
    assert not inference.exists() and not inference.parent.exists()  # nothing written
    assert research.path.read_bytes() == before


def test_with_no_supported_ticker_the_keychain_is_never_read(research, tmp_path):
    code, text, transport, _, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"]
    )
    assert code == 0 and "nothing to fetch" in text
    assert reader.reads == [] and transport.calls == []


def test_a_refused_keychain_key_stops_the_run_without_showing_it(research, tmp_path):
    base.supported(research)
    code, text, transport, inference, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], [(401, b"")]
    )
    assert code == 1 and "invalid_token" in text and reader.reads == [0]
    assert len(transport.calls) == 1 and KEYCHAIN_KEY not in text
    assert not inference.exists()
