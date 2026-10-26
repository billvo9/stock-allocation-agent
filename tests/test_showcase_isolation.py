"""
The showcase's data path stays apart from the Stock Lab: its own store, never
a Stock Lab store or watchlist, never inside a Git tree, and never a 2025
session; ETFs are classified only where the showcase registers them.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from showcase_fixtures import bars

from stock_agent.showcase import store as showcase_store
from stock_agent.showcase.store import (
    DB_NAME,
    ShowcaseStoreRefused,
    check_location,
    open_store,
    window_bars,
)
from stock_agent.stock_lab.contracts import DataClass, SymbolReason, SymbolStatus
from stock_agent.stock_lab.symbols import ETF_ASSET_TYPES, ListingRow, classify

SRC = Path(__file__).resolve().parents[1] / "src" / "stock_agent"


@pytest.fixture(autouse=True)
def private_homes(tmp_path, monkeypatch):
    """Point every Stock Lab store at a temporary home, so no test can reach a real one."""

    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_HOME", str(tmp_path / "lab"))
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_INFERENCE_HOME", str(tmp_path / "inference"))
    monkeypatch.delenv(showcase_store.HOME_ENV, raising=False)


def test_the_default_store_is_its_own_private_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert showcase_store.default_path() == (
        tmp_path / "stock-agent-private" / "showcase_store" / DB_NAME
    )


def test_a_stock_lab_store_location_is_refused(tmp_path):
    with pytest.raises(ShowcaseStoreRefused, match="Stock Lab store"):
        check_location(tmp_path / "lab" / DB_NAME)
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "stock_lab.sqlite3").write_bytes(b"")
    with pytest.raises(ShowcaseStoreRefused, match="holds another store"):
        check_location(beside / DB_NAME)
    with pytest.raises(ShowcaseStoreRefused, match="holds another store"):
        check_location(beside / "nested" / DB_NAME)  # under another store's folder


def test_the_inference_stores_folder_is_refused_without_naming_it(tmp_path):
    from stock_agent.stock_lab.inference.store import InferenceStore

    inference = InferenceStore()  # its home is tmp_path / "inference" (autouse fixture)
    inference.path.parent.mkdir(parents=True)
    inference.path.write_bytes(b"")
    for path in (inference.path.parent / DB_NAME, inference.path.parent / "x" / DB_NAME):
        with pytest.raises(ShowcaseStoreRefused, match="holds another store"):
            check_location(path)
    with pytest.raises(ShowcaseStoreRefused, match="named"):
        check_location(tmp_path / "elsewhere" / "stock_lab.sqlite3")


def test_a_location_inside_a_git_tree_is_refused_without_exception(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    for path in (tmp_path / "repo" / "x" / DB_NAME, tmp_path / "repo" / "data" / "raw" / DB_NAME):
        with pytest.raises(ShowcaseStoreRefused, match="Git working tree"):
            check_location(path)


def _store_with(tmp_path, ticker, frame):
    store = open_store(tmp_path / "showcase" / DB_NAME)
    store.record_prices(
        ticker,
        frame,
        source="test",
        retrieved_at=pd.Timestamp("2026-10-26T18:00:00Z"),
        data_class=DataClass.REAL,
        vintage_note="test",
    )
    return store


def test_a_stored_2025_session_refuses_the_read_rather_than_being_dropped(tmp_path):
    days = pd.DatetimeIndex(["2024-12-30", "2024-12-31", "2025-01-02"], tz="UTC").as_unit("us")
    store = _store_with(tmp_path, "XAA", bars(days, [10.0, 10.1, 10.2]))
    with pytest.raises(ShowcaseStoreRefused, match="leave the window"):
        window_bars(store, "XAA", date(2003, 1, 2), date(2024, 12, 31))


def test_window_bars_reads_back_and_refuses_a_missing_ticker(tmp_path):
    days = pd.DatetimeIndex(["2024-12-30", "2024-12-31"], tz="UTC").as_unit("us")
    store = _store_with(tmp_path, "XAA", bars(days, [10.0, 10.1]))
    read = window_bars(store, "XAA", date(2003, 1, 2), date(2024, 12, 31))
    assert list(read.bars["close"]) == [10.0, 10.1]
    with pytest.raises(ShowcaseStoreRefused, match="no stored bars"):
        window_bars(store, "XBB", date(2003, 1, 2), date(2024, 12, 31))


def test_the_showcase_store_writes_nothing_into_the_stock_lab_homes(tmp_path):
    days = pd.DatetimeIndex(["2024-12-30", "2024-12-31"], tz="UTC").as_unit("us")
    store = _store_with(tmp_path, "XAA", bars(days, [10.0, 10.1]))
    assert store.watched() == ()  # the showcase never adds to a watchlist
    assert not (tmp_path / "lab").exists() and not (tmp_path / "inference").exists()


def _row(ticker, asset_type):
    return ListingRow(
        ticker, "NYSE ARCA", asset_type, "USD", date(1998, 12, 22), date(2026, 10, 23)
    )


def _classify(ticker, rows, **kwargs):
    return classify(
        ticker, rows, recent_cutoff=date(2026, 10, 20), sessions_since=lambda d: 0, **kwargs
    )


def test_an_etf_is_supported_only_where_the_caller_registers_etfs():
    rows = [_row("XLK", "ETF")]
    default = _classify("XLK", rows)
    assert (default.status, default.reason) == (SymbolStatus.UNSUPPORTED, SymbolReason.NOT_A_STOCK)
    registered = _classify("XLK", rows, asset_types=ETF_ASSET_TYPES)
    assert registered.status is SymbolStatus.SUPPORTED
    assert registered.reason is SymbolReason.LISTED_US_ETF
    assert registered.detail == "An ETF listed on NYSE ARCA, priced in USD."


def test_registering_etfs_does_not_relax_the_other_checks():
    stock_as_etf = _classify("MSFT", [_row("MSFT", "Stock")], asset_types=ETF_ASSET_TYPES)
    assert stock_as_etf.reason is SymbolReason.NOT_REGISTERED_TYPE
    otc = ListingRow("XLK", "OTC", "ETF", "USD", date(1998, 12, 22), date(2026, 10, 23))
    assert _classify("XLK", [otc], asset_types=ETF_ASSET_TYPES).reason is (
        SymbolReason.OVER_THE_COUNTER
    )
    two = [_row("XLK", "ETF"), ListingRow("XLK", "NASDAQ", "ETF", "USD", None, date(2026, 10, 23))]
    assert _classify("XLK", two, asset_types=ETF_ASSET_TYPES).status is SymbolStatus.AMBIGUOUS
    with pytest.raises(ValueError, match="upper-case"):
        _classify("XLK", [_row("XLK", "ETF")], asset_types=frozenset({"etf"}))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_lab_job_or_app_module_imports_the_showcase():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    paths = [p for p in SRC.rglob("*.py") if "showcase" not in p.parts]
    paths += [p for p in scripts.glob("*.py") if not p.name.startswith("showcase_")]
    assert any(p.name == "stock_lab_daily.py" for p in paths)  # the evening job is checked
    for path in paths:
        assert not any(name.startswith("stock_agent.showcase") for name in _imports(path)), path


def test_a_location_under_a_stock_lab_store_directory_is_refused(tmp_path):
    with pytest.raises(ShowcaseStoreRefused, match="in or under a Stock Lab store"):
        check_location(tmp_path / "lab" / "nested" / DB_NAME)


def test_synthetic_bars_are_never_read_as_showcase_data(tmp_path):
    days = pd.DatetimeIndex(["2024-12-30", "2024-12-31"], tz="UTC").as_unit("us")
    store = open_store(tmp_path / "showcase" / DB_NAME)
    store.record_prices(
        "XAA_DEMO",
        bars(days, [10.0, 10.1]),
        source="test",
        retrieved_at=pd.Timestamp("2026-10-26T18:00:00Z"),
        data_class=DataClass.SYNTHETIC,
        vintage_note="test",
    )
    with pytest.raises(ShowcaseStoreRefused, match="real bars only"):
        window_bars(store, "XAA_DEMO", date(2003, 1, 2), date(2024, 12, 31))


def test_the_default_stock_classification_is_unchanged():
    row = ListingRow("MSFT", "NASDAQ", "Stock", "USD", date(1986, 3, 13), date(2026, 10, 23))
    result = _classify("MSFT", [row])
    assert result.reason is SymbolReason.LISTED_US_STOCK
    assert result.detail == "A stock listed on NASDAQ, priced in USD."
    with pytest.raises(ValueError, match="non-empty"):
        _classify("MSFT", [row], asset_types=frozenset())
