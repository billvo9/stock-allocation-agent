"""
The two owner-run holdout scripts on fakes: the fetch (the plan by default,
read-only with --authorized, storage with --authorized --store, each stock
once, the 2024 sessions checked against the research store) and the one
evaluation (the plan by default, --run once, --show afterwards). No network,
no real data.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from stock_lab_holdout_helpers import (
    CALENDAR_ID,
    Clock,
    fill,
    frozen_model,
    ny,
    open_store,
    short_protocol,
    weekdays,
)
from test_stock_lab_tiingo import FakeTransport, price_row

from stock_agent.calendars import nyse_rules
from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.holdout.protocol import PROTOCOL, HoldoutRefused, Refusal, record_id
from stock_agent.stock_lab.holdout.store import DB_NAME, HoldoutStore
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import Classification, ListingRow, parse_tickers
from stock_agent.stock_lab.tiingo import StockAdapter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load("stock_lab_holdout_fetch")
evaluate_script = _load("stock_lab_holdout_evaluate")

KEY = "fixture-key-not-real-0123"
NOW = ny("2026-10-28", "21:00")
DAYS = PROTOCOL.fetch_sessions(nyse_rules.sessions)
DEVELOPMENT = DAYS[DAYS <= pd.Timestamp("2024-12-31", tz="UTC")]


def _close(ticker: str, i: int) -> float:
    return 50.0 + 10.0 * PROTOCOL.tickers.index(ticker) + 0.05 * i


def _rows(ticker: str) -> list[dict]:
    return [
        price_row(day.strftime("%Y-%m-%d"), close=_close(ticker, i)) for i, day in enumerate(DAYS)
    ]


def _response(ticker: str) -> tuple[int, bytes]:
    return 200, json.dumps(_rows(ticker)).encode()


def _research_bars(ticker: str, shift: float = 0.0) -> pd.DataFrame:
    close = [_close(ticker, i) + (shift if i == 100 else 0.0) for i in range(len(DEVELOPMENT))]
    return pd.DataFrame(
        {
            "session": DEVELOPMENT,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "div_cash": 0.0,
            "split_factor": 1.0,
        }
    )


def _make_research(tmp_path, *, supported=PROTOCOL.tickers, bars=None):
    """A research store: listing checks for `supported`, and 2024 bars per ticker (`bars`
    maps a ticker to its frame, or None for no history; by default MU differs on one close)."""

    store = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME, clock=lambda: NOW)
    store.add(parse_tickers(" ".join(PROTOCOL.tickers)))
    rows = [ListingRow(t, "NASDAQ", "Stock", "USD", None, None) for t in supported]
    snapshot = store.record_listing(
        rows, source="fixture", retrieved_at=NOW, data_class=DataClass.REAL
    )
    store.record_checks(
        [
            Classification(t, SymbolStatus.SUPPORTED, SymbolReason.LISTED_US_STOCK, "")
            for t in supported
        ],
        snapshot_id=snapshot,
    )
    if bars is None:
        bars = {t: _research_bars(t, shift=1.0 if t == "MU" else 0.0) for t in PROTOCOL.tickers}
    for ticker, frame in bars.items():
        if frame is None:
            continue
        store.record_prices(
            ticker,
            frame,
            source="fixture",
            retrieved_at=NOW - pd.Timedelta(days=30),
            data_class=DataClass.REAL,
            vintage_note="fixture",
        )
    return store


@pytest.fixture
def research(tmp_path):
    return _make_research(tmp_path)


def _fetch(research, tmp_path, argv, responses=()):
    transport = FakeTransport(list(responses))
    clock = Clock(NOW)
    built = []

    def factory(**kwargs):
        def stamped():
            clock.now += pd.Timedelta(seconds=5)
            return clock.now

        def token():
            built.append("key read")
            return KEY

        return StockAdapter(token_provider=token, transport=transport, sleep=lambda s: None,
                            clock=stamped, **kwargs)  # fmt: skip

    path = tmp_path / "holdout" / DB_NAME
    out = io.StringIO()
    code = fetch.main(
        argv,
        adapter_factory=factory,
        store_factory=lambda: research,
        holdout_factory=lambda: HoldoutStore(path, clock=clock),
        out=out,
    )
    return code, out.getvalue(), transport, path, built


def test_without_authorization_it_prints_the_plan_and_reads_no_key(research, tmp_path):
    before = research.path.read_bytes()
    code, text, transport, path, keys = _fetch(research, tmp_path, [])
    assert code == 0 and transport.calls == [] and keys == []
    assert "no request made" in text
    for ticker in PROTOCOL.tickers:
        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            "?startDate=2024-01-02&endDate=2026-10-14"
        )
        assert url in text
    assert text.count("GET https://") == 4 and "token=" not in text
    assert "699 NYSE sessions" in text
    assert not path.exists() and research.path.read_bytes() == before


def test_store_needs_authorization(research, tmp_path):
    code, text, transport, _, _ = _fetch(research, tmp_path, ["--store"])
    assert code == 2 and "needs --authorized" in text and transport.calls == []


def test_authorized_alone_audits_and_writes_nothing(research, tmp_path):
    responses = [_response(t) for t in PROTOCOL.tickers]
    code, text, transport, path, _ = _fetch(research, tmp_path, ["--authorized"], responses)
    assert code == 0 and len(transport.calls) == 4
    assert "2024 sessions: 0 differ" in text and "2024 sessions: 1 differ" in text
    assert "3 ready (AMD, MRVL, NVDA); 1 to review (MU)" in text
    assert "nothing was written" in text and not path.exists()
    assert KEY not in text and "50.0" not in text  # never the key or a price


def test_store_writes_each_ready_stock_once_and_holds_the_rest(research, tmp_path):
    responses = [_response(t) for t in PROTOCOL.tickers]
    code, text, _, path, _ = _fetch(research, tmp_path, ["--authorized", "--store"], responses)
    assert code == 0 and "not stored: held for review first" in text
    clock = Clock(NOW + pd.Timedelta(minutes=5))
    stored = HoldoutStore(path, clock=clock).batches()
    assert [batch.ticker for batch in stored] == ["AMD", "MRVL", "NVDA"]
    assert all(batch.row_count == 699 and batch.calendar_id == nyse_rules.CALENDAR_ID
               for batch in stored)  # fmt: skip
    # a second run requests only the stock not stored yet
    code, text, transport, _, _ = _fetch(research, tmp_path, ["--authorized"], [_response("MU")])
    assert code == 0 and len(transport.calls) == 1 and "/MU/" in transport.calls[0][0]
    code, text, transport, _, _ = _fetch(research, tmp_path, [])
    assert "already stored, not requested again: AMD, MRVL, NVDA" in text
    assert text.count("GET https://") == 1


def test_a_spent_holdout_is_never_fetched_again(research, tmp_path):
    protocol = short_protocol("a" * 64)
    store, _ = open_store(tmp_path, protocol)
    fill(store, protocol)
    content = {"protocol_sha256": protocol.sha256(), "model": {"model_id": protocol.model_id}}
    store.evaluate_once(lambda h, b: {**content, "record_id": record_id(content)},
                        calendar_id=CALENDAR_ID)  # fmt: skip
    out = io.StringIO()
    code = fetch.main(["--authorized", "--store"], adapter_factory=lambda **k: pytest.fail(),
                      store_factory=lambda: research, holdout_factory=lambda: store,
                      protocol=protocol, calendar=weekdays, calendar_id=CALENDAR_ID,
                      out=out)  # fmt: skip
    assert code == 2 and "spent" in out.getvalue()


def _stored(tmp_path) -> list[str]:
    clock = Clock(NOW + pd.Timedelta(minutes=5))
    return [
        batch.ticker
        for batch in HoldoutStore(tmp_path / "holdout" / DB_NAME, clock=clock).batches()
    ]


def test_a_stock_with_an_unserved_session_is_held_for_review(tmp_path):
    research = _make_research(tmp_path, bars={t: _research_bars(t) for t in PROTOCOL.tickers})
    rows = _rows("AMD")
    del rows[400]  # a 2025 session the provider did not serve
    responses = [(200, json.dumps(rows).encode())] + [_response(t) for t in PROTOCOL.tickers[1:]]
    code, text, _, _, _ = _fetch(research, tmp_path, ["--authorized", "--store"], responses)
    assert code == 0 and "1 session(s) not served: held for review" in text
    assert "3 ready (MRVL, MU, NVDA); 1 to review (AMD)" in text
    assert _stored(tmp_path) == ["MRVL", "MU", "NVDA"]


def test_a_stock_without_research_history_is_held_for_review(tmp_path):
    bars = {t: _research_bars(t) for t in PROTOCOL.tickers}
    bars["NVDA"] = None
    research = _make_research(tmp_path, bars=bars)
    responses = [_response(t) for t in PROTOCOL.tickers]
    code, text, _, _, _ = _fetch(research, tmp_path, ["--authorized", "--store"], responses)
    assert code == 0 and "no research history to compare: held for review" in text
    assert _stored(tmp_path) == ["AMD", "MRVL", "MU"]


@pytest.mark.parametrize("column", ["div_cash", "split_factor"])
def test_a_2024_dividend_or_split_unlike_the_research_stores_is_held_for_review(tmp_path, column):
    bars = {t: _research_bars(t) for t in PROTOCOL.tickers}
    bars["MRVL"].loc[50, column] = 0.06 if column == "div_cash" else 2.0
    research = _make_research(tmp_path, bars=bars)
    responses = [_response(t) for t in PROTOCOL.tickers]
    code, text, _, _, _ = _fetch(research, tmp_path, ["--authorized", "--store"], responses)
    assert code == 0 and "1 to review (MRVL)" in text
    assert _stored(tmp_path) == ["AMD", "MU", "NVDA"]


def test_a_stock_without_a_supported_listing_check_is_not_requested(tmp_path):
    research = _make_research(tmp_path, supported=("AMD", "MU", "NVDA"))
    code, text, transport, _, _ = _fetch(research, tmp_path, [])
    assert code == 0 and transport.calls == []
    assert "not requested (no supported listing check): MRVL" in text
    assert text.count("GET https://") == 3


def test_a_bar_on_a_day_that_is_not_a_session_stops_the_fetch(research, tmp_path):
    rows = _rows("AMD") + [price_row("2026-10-10", close=60.0)]  # a Saturday
    responses = [(200, json.dumps(rows).encode())]
    code, text, _, _, _ = _fetch(research, tmp_path, ["--authorized", "--store"], responses)
    assert code == 1 and "stopped:" in text and _stored(tmp_path) == []


# --- the evaluation script ---


@pytest.fixture(scope="module")
def frozen():
    return frozen_model()


def _lab(tmp_path, frozen, *, register=True):
    evaluation, model_record = frozen
    research = StockLabStore(tmp_path / "research" / RESEARCH_DB_NAME, clock=lambda: NOW)
    if register:
        research.record_evaluation(evaluation)
        research.record_frozen_model(model_record)
    protocol = short_protocol(model_record["model_id"])
    store, _ = open_store(tmp_path, protocol)
    fill(store, protocol)
    return research, store, protocol


def _evaluate(lab, argv, code=lambda: {"sha": "1" * 40, "clean": True}):
    research_store, store, protocol = lab
    out = io.StringIO()
    status = evaluate_script.main(
        argv,
        store_factory=lambda: research_store,
        holdout_factory=lambda: store,
        protocol=protocol,
        calendar=weekdays,
        calendar_id=CALENDAR_ID,
        clock=lambda: NOW,
        code=code,
        out=out,
    )
    return status, out.getvalue()


def test_the_evaluation_runs_once_and_its_result_can_be_shown(tmp_path, frozen):
    lab = _lab(tmp_path, frozen)
    _, store, protocol = lab
    status, text = _evaluate(lab, [])
    assert status == 0 and "nothing read" in text and store.evaluation() is None
    assert all(f"{t}: fetched" in text for t in protocol.tickers)
    status, text = _evaluate(lab, ["--run"])
    assert status == 0, text
    record = store.evaluation()
    assert record is not None and record["record_id"] in text
    assert "The question: does B1 beat B0 on the pooled daily RPS?" in text
    assert "answer: no" in text and "Limits:" in text
    assert record["code"] == {"sha": "1" * 40, "clean": True}
    assert [b["ticker"] for b in record["data"]["batches"]] == list(protocol.tickers)
    status, again = _evaluate(lab, ["--run"])
    assert status == 2 and "spent" in again
    status, shown = _evaluate(lab, ["--show"])
    assert status == 0 and shown.strip() == text.split("\nstored as")[0].strip()
    assert store.evaluation() == record


def test_the_run_is_refused_without_the_model_or_a_clean_checkout(tmp_path, frozen):
    lab = _lab(tmp_path / "a", frozen, register=False)
    status, text = _evaluate(lab, ["--run"])
    assert status == 2 and "no registered real frozen model" in text
    assert lab[1].evaluation() is None
    lab = _lab(tmp_path / "b", frozen)
    status, text = _evaluate(lab, ["--run"], code=lambda: {"sha": "1" * 40, "clean": False})
    assert status == 2 and "clean checkout" in text
    status, text = _evaluate(lab, ["--run"], code=lambda: None)
    assert status == 2 and lab[1].evaluation() is None


def test_show_before_the_run_says_so(tmp_path, frozen):
    status, text = _evaluate(_lab(tmp_path, frozen), ["--show"])
    assert status == 0 and "not been evaluated" in text


def test_the_recorded_commit_is_clean_only_for_this_checkouts_own_code(monkeypatch):
    answers = {
        ("rev-parse", "HEAD"): "a" * 40 + "\n",
        ("status", "--porcelain", "--untracked-files=no"): "",
        ("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts"): "",
    }
    monkeypatch.setattr(evaluate_script, "_git", lambda *arguments: answers[arguments])
    own = evaluate_script.PROJECT / "src" / "stock_agent" / "__init__.py"
    monkeypatch.setattr(evaluate_script.stock_agent, "__file__", str(own))
    assert evaluate_script.code_state() == {"sha": "a" * 40, "clean": True}
    other = Path("/elsewhere/checkout/src/stock_agent/__init__.py")
    monkeypatch.setattr(evaluate_script.stock_agent, "__file__", str(other))
    assert evaluate_script.code_state()["clean"] is False  # another checkout's code
    monkeypatch.setattr(evaluate_script.stock_agent, "__file__", str(own))
    answers[("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts")] = (
        "?? src/stock_agent/stock_lab/extra.py\n"
    )
    assert evaluate_script.code_state()["clean"] is False  # an untracked module
    answers[("rev-parse", "HEAD")] = None
    assert evaluate_script.code_state() is None


def test_a_tracked_change_makes_the_checkout_dirty(monkeypatch):
    answers = {
        ("rev-parse", "HEAD"): "a" * 40 + "\n",
        (
            "status",
            "--porcelain",
            "--untracked-files=no",
        ): " M src/stock_agent/stock_lab/registry.py\n",
        ("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts"): "",
    }
    monkeypatch.setattr(evaluate_script, "_git", lambda *arguments: answers[arguments])
    own = evaluate_script.PROJECT / "src" / "stock_agent" / "__init__.py"
    monkeypatch.setattr(evaluate_script.stock_agent, "__file__", str(own))
    assert evaluate_script.code_state() == {"sha": "a" * 40, "clean": False}


def test_a_synthetic_or_unloadable_model_is_refused_and_nothing_is_recorded(
    tmp_path, frozen, monkeypatch
):
    lab = _lab(tmp_path, frozen)
    research_store, store, _ = lab
    original = research_store.frozen_model
    monkeypatch.setattr(
        research_store, "frozen_model", lambda i: {**original(i), "data_class": "synthetic"}
    )
    status, text = _evaluate(lab, ["--run"])
    assert status == 2 and "no registered real frozen model" in text
    monkeypatch.setattr(research_store, "frozen_model", original)

    def unloadable(record):
        raise evaluate_script.RegistryRefused("environment_mismatch", "another numpy")

    monkeypatch.setattr(evaluate_script.registry, "load", unloadable)
    status, text = _evaluate(lab, ["--run"])
    assert status == 2 and "does not load here" in text
    assert store.evaluation() is None


def test_a_run_that_refuses_exits_1_and_records_nothing(tmp_path, frozen, monkeypatch):
    lab = _lab(tmp_path, frozen)

    def refusing(*args, **kwargs):
        raise HoldoutRefused(Refusal.INTEGRITY, "a forecast is not finite")

    monkeypatch.setattr(evaluate_script.evaluate, "run", refusing)
    status, text = _evaluate(lab, ["--run"])
    assert status == 1 and "nothing was recorded" in text
    assert lab[1].evaluation() is None
