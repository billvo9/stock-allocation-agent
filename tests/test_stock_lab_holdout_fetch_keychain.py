"""
The holdout fetch with --keychain (scripts/stock_lab_holdout_fetch.py; owner
decision of 2026-10-29): the key is read from the macOS Keychain item the
evening job uses, once, before the first request, and reaches the adapter in
memory only. An item that cannot be read stops the run before any request or
write, and nothing reads it when nothing is requested. Fakes only: no
Keychain, no network.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
import test_stock_lab_holdout_scripts as base
from stock_lab_holdout_helpers import Clock
from test_stock_lab_tiingo import FakeTransport

from stock_agent.data.vendors.keychain import KeychainFailure, KeychainRefused
from stock_agent.stock_lab.holdout.protocol import PROTOCOL
from stock_agent.stock_lab.holdout.store import DB_NAME, HoldoutStore
from stock_agent.stock_lab.tiingo import StockAdapter

fetch = base.fetch
KEYCHAIN_KEY = "keychain-fixture-key-0789"


class Reader:
    """Stands in for keychain.read_key: records how many requests preceded each read."""

    def __init__(self, transport, refusal=None):
        self.transport, self.refusal, self.reads = transport, refusal, []

    def __call__(self):
        self.reads.append(len(self.transport.calls))
        if self.refusal is not None:
            raise self.refusal
        return KEYCHAIN_KEY


def run_keychain(research, tmp_path, argv, responses=(), refusal=None):
    transport = FakeTransport(list(responses))
    clock = Clock(base.NOW)
    reader = Reader(transport, refusal)
    built = []

    def factory(**kwargs):
        built.append(set(kwargs))

        def stamped():
            clock.now += pd.Timedelta(seconds=5)
            return clock.now

        return StockAdapter(transport=transport, sleep=lambda s: None, clock=stamped, **kwargs)

    path = tmp_path / "holdout" / DB_NAME
    out = io.StringIO()
    code = fetch.main(
        argv,
        adapter_factory=factory,
        store_factory=lambda: research,
        holdout_factory=lambda: HoldoutStore(path, clock=clock),
        out=out,
        key_reader=reader,
    )
    return code, out.getvalue(), transport, path, reader, built


@pytest.fixture
def research(tmp_path):
    return base._make_research(tmp_path)


def test_with_keychain_the_plan_names_the_item_and_reads_nothing(research, tmp_path):
    code, text, transport, path, reader, built = run_keychain(research, tmp_path, ["--keychain"])
    assert code == 0 and transport.calls == [] and reader.reads == [] and built == []
    assert "service stock-agent-tiingo, account stock-lab" in text and "not read now" in text
    assert not path.exists()


def test_with_keychain_the_key_is_read_once_before_any_request_and_sent_only_in_the_header(
    research, tmp_path, monkeypatch
):
    monkeypatch.setenv("TIINGO_API_KEY", "environment-key-never-used")
    responses = [base._response(t) for t in PROTOCOL.tickers]
    code, text, transport, path, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], responses
    )
    assert code == 0, text
    assert reader.reads == [0]  # once, and before the first request
    assert built == [{"authorized", "max_requests", "token_provider"}]
    assert [headers for _, headers in transport.calls] == [
        {"Authorization": f"Token {KEYCHAIN_KEY}"}
    ] * len(PROTOCOL.tickers)
    assert all(KEYCHAIN_KEY not in url for url, _ in transport.calls)
    assert KEYCHAIN_KEY not in text and "environment-key" not in text
    stored = HoldoutStore(path, clock=Clock(base.NOW + pd.Timedelta(minutes=5))).batches()
    assert [batch.ticker for batch in stored] == ["AMD", "MRVL", "NVDA"]  # MU held (2024 differs)


def test_without_keychain_the_reader_is_never_called(research, tmp_path, monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    responses = [base._response(t) for t in PROTOCOL.tickers]
    code, text, transport, _, reader, built = run_keychain(
        research, tmp_path, ["--authorized"], responses
    )
    # No --keychain: the adapter keeps its default provider (TIINGO_API_KEY), here unset.
    assert reader.reads == [] and built == [{"authorized", "max_requests"}]
    assert code == 1 and "missing_token" in text and transport.calls == []


@pytest.mark.parametrize("reason", list(KeychainFailure))
def test_an_unreadable_keychain_item_stops_the_run_before_any_request_or_write(
    research, tmp_path, reason
):
    before = research.path.read_bytes()
    refusal = KeychainRefused(reason, service="stock-agent-tiingo", account="stock-lab")
    code, text, transport, path, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], refusal=refusal
    )
    assert code == 2 and reader.reads == [0] and transport.calls == [] and built == []
    assert "refused: the Tiingo key could not be read from the macOS Keychain" in text
    assert reason.value in text and "nothing was requested" in text
    assert "security add-generic-password -s stock-agent-tiingo -a stock-lab -U -w" in text
    assert not path.exists()  # nothing written
    assert research.path.read_bytes() == before


def test_with_nothing_left_to_request_the_keychain_is_never_read(tmp_path):
    bars = {ticker: base._research_bars(ticker) for ticker in PROTOCOL.tickers}
    research = base._make_research(tmp_path, bars=bars)
    responses = [base._response(t) for t in PROTOCOL.tickers]
    code, _, _, _, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], responses
    )
    assert code == 0 and reader.reads == [0]  # every stock stored by this first run
    code, text, transport, _, reader, built = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"]
    )
    assert code == 0 and "nothing to fetch" in text
    assert reader.reads == [] and transport.calls == [] and built == []


def test_a_refused_keychain_key_stops_the_run_without_showing_it(research, tmp_path):
    code, text, transport, path, reader, _ = run_keychain(
        research, tmp_path, ["--authorized", "--store", "--keychain"], [(401, b"")]
    )
    assert code == 1 and "invalid_token" in text and reader.reads == [0]
    assert len(transport.calls) == 1 and KEYCHAIN_KEY not in text
    assert HoldoutStore(path, clock=Clock(base.NOW)).batches() == ()


def test_store_with_keychain_but_without_authorization_reads_nothing(research, tmp_path):
    code, text, transport, path, reader, built = run_keychain(
        research, tmp_path, ["--store", "--keychain"]
    )
    assert code == 2 and "needs --authorized" in text
    assert reader.reads == [] and transport.calls == [] and built == [] and not path.exists()
