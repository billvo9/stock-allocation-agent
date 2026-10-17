"""
The Lab's Stock Probability page under AppTest: the watchlist persists in the
private store across app runs, refusals are explained, demo stocks are
labelled synthetic, real prices after 2024 are never shown, and every state
passes the advice scan.
"""

from __future__ import annotations

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
    assert set(saved) <= {
        "stock-lab-search",
        "stock-lab-stock",
        "stock-lab-band",
        "stock-lab-rule",
        "stock-lab-rev-direction",
        "stock-lab-rev-k",
        "stock-lab-move-x",
        "stock-lab-move-sign",
    }


def _demo_store(home: Path) -> None:
    import importlib.util
    import io

    script = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_evaluate.py"
    spec = importlib.util.spec_from_file_location("stock_lab_evaluate_app", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = StockLabStore(home / DB_NAME)
    now = pd.Timestamp("2026-10-16T22:00:00Z")
    code = module.main(
        ["--demo"], store_factory=lambda: store, clock=lambda: now, out=io.StringIO()
    )
    assert code == 0


def test_without_stored_records_the_modelled_tabs_say_how_to_get_them(home):
    text = text_of(page())
    assert words.FORECAST_NONE_DEMO_TITLE in text and words.DEMO_COMMAND in text
    assert words.DIAGNOSTICS_NONE in text
    assert "%" not in "".join(m.value for m in page().metric if m.label in ("Increase", "Decrease"))


def test_demo_forecasts_show_five_sessions_beside_the_usual_mix_and_their_tests(home):
    _demo_store(home)
    app = page()
    text = text_of(app)
    assert words.FORECAST_BADGE in text and words.SYNTHETIC_BADGE in text
    assert "in 10 for a rise beyond ±0.5%" in text and "usual mix" in text
    cards = [m for m in app.metric if m.label in ("Increase", "Stable", "Decrease")]
    assert len(cards) == 6  # the next session, and the five-session total
    assert words.DAILY_NOTE in text and words.TRACK_NOTE in text
    five = next(t.value for t in app.table if t.value.index.name == "Session")
    assert len(five) == 5 and five["Sessions ahead"].tolist() == [1, 2, 3, 4, 5]
    assert "Synthetic demo stock: not a real company." in text
    charts = [str(c.proto.spec) for c in app.get("plotly_chart")]
    assert sum(words.SYNTHETIC_CHART in c for c in charts) == len(charts)  # every chart says so
    assert "beat the recent-weighted" in text or "B1 is kept" in text
    assert words.DIAGNOSTICS_SYNTHETIC in text
    # Reported beside the results, never used to select: the other tests, their
    # simulated sizes and the band sensitivity.
    assert "One-sided p (fold-block t)" in text and words.SIZE_TITLE in text
    assert "Placebo half-life (sessions)" in text and "Newey-West (decides)" in text
    assert words.BAND_TITLE in text and "±0.5% (in use)" in text and "±0.25%" in text
    scan_for_advice(text)


def test_reversal_and_trap_tabs_describe_the_selected_stock(home):
    app = page()
    app.selectbox[0].set_value("DEMO_TREND").run()
    text = text_of(app)
    assert words.REVERSALS_LEDE in text and words.TRAPS_LEDE in text
    assert "Step 1" in text and "Failure" in text
    assert "Setup failure rate" in text and "for every breakout" in text
    streaks = next(t.value for t in app.table if t.value.index.name == "After")
    assert len(streaks) == 7 and "Cases" in streaks.columns
    app.radio[1].set_value("increase").run()
    assert not app.exception
    scan_for_advice(text_of(app))


def test_a_real_stock_has_no_probabilities_until_the_approvals(home):
    store = StockLabStore(home / DB_NAME)
    store.add(parse_tickers("AAPL"))
    bars = synthetic.demo_bars("DEMO_CALM").copy()
    bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit("us")
    store.record_prices(
        "AAPL",
        bars,
        source="fixture",
        retrieved_at=pd.Timestamp("2026-10-16T22:00:00Z"),
        data_class=DataClass.REAL,
        vintage_note="fixture",
    )
    _demo_store(home)
    app = page()
    assert app.selectbox[0].value == "AAPL"
    text = text_of(app)
    assert words.FORECAST_NONE_REAL_TITLE in text
    assert words.DIAGNOSTICS_NONE in text  # no real evaluation: the synthetic one is not shown
    assert words.FORECAST_BADGE not in text


def _replay(home: Path, sessions: int = 4) -> StockLabStore:
    import importlib.util
    import io

    script = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_daily.py"
    spec = importlib.util.spec_from_file_location("stock_lab_daily_app", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = StockLabStore(home / DB_NAME)
    argv = ["--demo-replay", str(sessions), "--health-inputs", str(home / "inputs")]
    assert module.main(argv, store_factory=lambda: store, out=io.StringIO()) == 0
    return store


def test_the_forecast_history_shows_what_happened_beside_each_stored_forecast(home):
    _replay(home)
    app = page()
    history = next(t.value for t in app.table if t.value.index.name == "As of")
    assert history["What happened"].iloc[0] == "pending"  # the newest session has not closed
    assert set(history["What happened"].iloc[1:]) <= {"Increase", "Stable", "Decrease"}
    text = text_of(app)
    assert "session targets scored so far" in text and "Last update: ok" in text
    assert words.REPLAY_NOTE not in text  # scores exist, so no replay prompt
    scan_for_advice(text)


def test_a_forecast_older_than_the_expected_session_is_labelled_stale_never_replaced(home):
    store = _replay(home)
    latest = store.latest_forecast("DEMO_CALM")
    store.record_job_run(
        {
            "run_id": "replay-later",
            "started_at": "2025-01-03T21:10:00.000000Z",
            "finished_at": "2025-01-03T21:11:00.000000Z",
            "status": "failed",
            "reason": "planted",
            "expected_session": "2025-01-03",
            "data_class": "synthetic",
        }
    )
    app = page()
    warning = " ".join(w.value for w in app.warning)
    assert f"Stale: made as of {latest['as_of_session']}" in warning and "(failed)" in warning
    assert store.latest_forecast("DEMO_CALM") == latest  # nothing was invented
