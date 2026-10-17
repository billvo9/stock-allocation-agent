"""
The Lab's Stock Probability page under AppTest: the watchlist persists in the
private store across app runs, refusals are explained, demo stocks are
labelled synthetic, real prices after 2024 are never shown, and every state
passes the advice scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from test_app_destinations import scan_for_advice, text_of

from stock_agent.app import navigation
from stock_agent.app import stock_wording as words
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.store import DB_NAME, HOME_ENV, StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

RUN = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "app" / "run.py"
TIMEOUT = 120


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv(HOME_ENV, str(tmp_path))
    return tmp_path


def page() -> AppTest:
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    app.switch_page(navigation.PAGES["stock_lab"]).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def add(app: AppTest, text: str) -> AppTest:
    app.text_area[0].input(text)
    submit = next(b for b in app.button if b.label == words.ADD_BUTTON)
    submit.click().run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_the_page_opens_on_an_empty_watchlist_without_creating_a_store(home):
    app = page()
    text = text_of(app)
    assert words.TITLE in text and words.EMPTY_TITLE in text and words.NOT_YET in text
    assert words.EXPERIMENTAL_BADGE in text and words.SYNTHETIC_BADGE in text
    assert not (home / DB_NAME).exists()


def test_added_tickers_are_saved_and_survive_a_restart(home):
    app = add(page(), "aapl, brk.b VOD.L ^GSPC msft")
    text = text_of(app)
    assert "Added: AAPL, BRK-B, MSFT." in text
    assert "brk.b is saved as BRK-B" in text
    refused = next(t.value for t in app.table if "Why" in t.value.columns)
    assert refused.index.tolist() == ["VOD.L", "^GSPC"]
    assert StockLabStore(home / DB_NAME).watched() == ("AAPL", "BRK-B", "MSFT")
    why = next(t.value for t in app.table if "Why" in t.value.columns)
    assert "outside the US" in why.loc["VOD.L", "Why"]
    restarted = page()  # a fresh app process reads the same private store
    table = next(f.value for f in restarted.dataframe if "Current streak" in f.value.columns)
    assert table["Stock"].tolist() == ["AAPL", "BRK-B", "MSFT"]
    assert set(table["Status"]) == {"Not yet checked"} and set(table["Data"]) == {"Missing data"}
    assert words.NOT_CHECKED_NOTE in text_of(restarted)


def test_adding_a_ticker_twice_says_so(home):
    app = add(page(), "AAPL")
    app = add(app, "aapl")
    assert "Already in the watchlist: AAPL." in text_of(app)


def test_search_filters_and_remove_keeps_the_history(home):
    app = add(page(), "AAPL MSFT NVDA")
    search = next(t for t in app.text_input if t.label == words.SEARCH_LABEL)
    search.input("ms").run()
    table = next(f.value for f in app.dataframe if "Current streak" in f.value.columns)
    assert table["Stock"].tolist() == ["MSFT"]
    search.input("zz").run()
    assert "No saved ticker matches “zz”." in [t.value for t in app.text]  # plain text
    app.multiselect[0].select("NVDA").run()
    next(b for b in app.button if b.label == words.REMOVE_BUTTON).click().run()
    assert not app.exception
    assert "Removed: NVDA." in text_of(app)
    store = StockLabStore(home / DB_NAME)
    assert store.watched() == ("AAPL", "MSFT")
    assert store.events()["action"].tolist() == ["add", "add", "add", "remove"]


def test_the_detail_view_of_a_demo_stock_is_labelled_synthetic_and_follows_the_band(home):
    app = page()
    text = text_of(app)
    assert "Calm demo stock (synthetic)" in text and words.SYNTHETIC_BADGE in text
    bands = next(t.value for t in app.table if t.value.index.name == "Stable band")
    assert bands.index.tolist() == ["±0.25%", "±0.5%", "±1%"]
    runs = next(t.value for t in app.table if t.value.index.name == "Direction")
    assert runs.index.tolist() == ["Rising", "Falling"]
    assert "1 session" in runs.columns and "6 or more" in runs.columns
    assert int(runs.to_numpy().sum()) > 100  # whole runs, counted
    assert words.SYNTHETIC_CHART in str(app.get("plotly_chart")[0].proto.spec)
    cards = {m.label: m.value for m in app.metric}
    assert cards["Last close"].replace(",", "").replace(".", "").isdigit()  # a short value
    app.select_slider[0].set_value(0.01).run()
    assert "Stable sessions at ±1%" in [m.label for m in app.metric]
    app.radio[0].set_value("pauses").run()
    assert "Up to 1 Stable session(s) in a row pause a streak" in text_of(app)
    app.selectbox[0].set_value("DEMO_SWING").run()
    assert not app.exception
    assert "split(s) applied" in text_of(app)


def test_a_viewer_in_dark_mode_gets_dark_charts_with_the_same_content(home, monkeypatch):
    from stock_agent.ui import components, theme

    light = page()
    monkeypatch.setattr(components, "colour_scheme", lambda: "dark")
    dark = page()
    assert text_of(dark) == text_of(light)  # the scheme changes colours, never the words
    dark_colours = theme.LAB_DARK_COLOURS
    assert len(dark.get("plotly_chart")) == 2  # the price and the recent sessions
    for chart, before in zip(dark.get("plotly_chart"), light.get("plotly_chart"), strict=True):
        spec, light_spec = json.loads(chart.proto.spec), json.loads(before.proto.spec)
        layout = spec["layout"]
        assert layout["paper_bgcolor"] == layout["plot_bgcolor"] == dark_colours["surface"]
        assert layout["font"]["color"] == dark_colours["text"]
        assert [t.get("y") for t in spec["data"]] == [t.get("y") for t in light_spec["data"]]
        markers = {t["name"]: t["marker"]["color"] for t in spec["data"] if "marker" in t}
        assert set(markers.values()) <= set(theme.DIRECTION_DARK_COLOURS.values())
        for shape in layout.get("shapes", []):  # the stable band's edges
            assert shape["line"]["color"] == dark_colours["reference_line"]


def test_real_prices_after_2024_are_never_shown(home):
    store = StockLabStore(home / DB_NAME)
    store.add(parse_tickers("AAPL"))
    bars = synthetic.demo_bars("DEMO_CALM").iloc[-400:].copy()
    bars["session"] = pd.bdate_range("2024-06-03", periods=400, tz="UTC").as_unit("us")
    store.record_prices(
        "AAPL",
        bars,
        source="fixture",
        retrieved_at=pd.Timestamp("2026-10-16T22:00:00Z"),
        data_class=DataClass.REAL,
        vintage_note="fixture",
    )
    app = page()
    table = next(f.value for f in app.dataframe if "Current streak" in f.value.columns)
    assert table.loc[0, "Last session"] == "2024-12-31"
    assert table.loc[0, "Data"] == "Freshness not checked"
    text = text_of(app)
    assert "Shown through 2024-12-31." in text and words.REAL_CUTOFF in text
    assert "2025-" not in " ".join(m.value for m in app.metric)
    assert words.SYNTHETIC_CHART not in str(app.get("plotly_chart")[0].proto.spec)
    assert app.selectbox[0].value == "AAPL"  # saved stocks with prices come first


def test_every_state_passes_the_advice_scan(home):
    app = add(page(), "AAPL zz.L")
    scan_for_advice(text_of(app))
    scan_for_advice(" ".join(str(v) for v in vars(words).values() if isinstance(v, str)))


def test_saved_page_state_does_not_grow_across_reruns(home):
    app = page()
    for _ in range(4):
        app.run()
    saved = app.session_state["stock_lab_saved"]
    assert not any(isinstance(value, dict) for value in saved.values())
    assert set(saved) <= {"stock-lab-search", "stock-lab-stock", "stock-lab-band", "stock-lab-rule"}
