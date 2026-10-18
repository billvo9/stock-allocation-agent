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
from stock_lab_inference_helpers import AS_OF, CALENDAR_ID, make_bars, ny, weekdays, window_start
from streamlit.testing.v1 import AppTest
from test_app_destinations import scan_for_advice, text_of

from stock_agent.app import navigation
from stock_agent.app import stock_wording as words
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.inference.schema import DB_NAME as INFERENCE_DB_NAME
from stock_agent.stock_lab.inference.schema import HOME_ENV as INFERENCE_HOME_ENV
from stock_agent.stock_lab.inference.store import InferenceStore
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
    keys = [chart.key.removeprefix("stock-lab-") for chart in dark.get("plotly_chart")]
    assert {"price-chart", "recent-chart", "streak-chart"} <= set(keys)
    for chart, before in zip(dark.get("plotly_chart"), light.get("plotly_chart"), strict=True):
        spec, light_spec = json.loads(chart.proto.spec), json.loads(before.proto.spec)
        layout = spec["layout"]
        assert layout["paper_bgcolor"] == layout["plot_bgcolor"] == dark_colours["surface"]
        assert layout["font"]["color"] == dark_colours["text"]
        assert [t.get("y") for t in spec["data"]] == [t.get("y") for t in light_spec["data"]]
        markers = {t.get("marker", {}).get("color") for t in spec["data"]} - {None}
        assert markers <= set(theme.DIRECTION_DARK_COLOURS.values()), (chart.key, markers)
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
    # Every calibration status carries its qualifier inline: synthetic, never "calibrated".
    assert text.count(words.CALIBRATION_SYNTHETIC) == 3 and words.CALIBRATION_REAL not in text
    assert "hindsight-selected and non-confirmatory" in text
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


def test_a_real_evaluation_marks_every_calibration_status_experimental(home):
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
    evaluation = store.latest_evaluation(DataClass.SYNTHETIC)
    store.record_evaluation({**evaluation, "run_id": "R-fixture", "data_class": "real"})
    text = text_of(page())
    assert text.count(words.CALIBRATION_REAL) == 3 and words.CALIBRATION_SYNTHETIC not in text
    assert "hindsight-selected and non-confirmatory" in text
    scan_for_advice(text)


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


# --- recent observed prices (decision A6) ---


@pytest.fixture
def inference_home(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "inference-home"
    monkeypatch.setenv(INFERENCE_HOME_ENV, str(directory))
    return directory


def _real_history(home: Path) -> StockLabStore:
    store = StockLabStore(home / DB_NAME)
    store.add(parse_tickers("NVDA"))
    bars = synthetic.demo_bars("DEMO_CALM").copy()
    bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit("us")
    store.record_prices(
        "NVDA",
        bars,
        source="fixture",
        retrieved_at=pd.Timestamp("2026-10-16T22:00:00Z"),
        data_class=DataClass.REAL,
        vintage_note="fixture",
    )
    return store


def _recent_batch(inference_home: Path) -> None:
    first = window_start(AS_OF)
    retrieved = ny(AS_OF, "20:30")
    store = InferenceStore(
        inference_home / INFERENCE_DB_NAME, clock=lambda: retrieved + pd.Timedelta(minutes=1)
    )
    store.record_batch(
        "NVDA",
        make_bars(first, AS_OF),
        source="fixture",
        retrieved_at=retrieved,
        request_start=first,
        request_end=AS_OF,
        sessions=weekdays,
        calendar_id=CALENDAR_ID,
        data_class="real",
        vintage_note="fixture",
    )


def _observed_keys(app: AppTest) -> list[str]:
    return [chart.key for chart in app.get("plotly_chart") if "observed" in chart.key]


def test_a_demo_stock_has_no_recent_observed_prices(home):
    text = text_of(page())
    assert words.OBSERVED_TITLE in text and words.OBSERVED_DEMO_TITLE in text


def test_recent_prices_show_an_honest_empty_state_until_stored(home, inference_home):
    _real_history(home)
    app = page()
    assert app.selectbox[0].value == "NVDA"
    text = text_of(app)
    assert words.OBSERVED_NONE_TITLE in text and words.OBSERVED_NONE["no_store"] in text
    assert words.OBSERVED_BADGE not in text and _observed_keys(app) == []
    assert not inference_home.exists()  # reading created nothing
    scan_for_advice(text)


def test_stored_recent_prices_are_shown_apart_from_the_research_history(home, inference_home):
    _real_history(home)
    _recent_batch(inference_home)
    app = page()
    text = text_of(app)
    assert words.OBSERVED_BADGE in text and words.OBSERVED_CHANGE_NOTE in text
    assert "Retrieved 2026-10-16 00:30 UTC from fixture" in text
    cards = {m.label: m.value for m in app.metric}
    assert (
        cards[words.OBSERVED_LAST]
        == f"{make_bars(window_start(AS_OF), AS_OF)['close'].iloc[-1]:,.2f}"
    )
    assert cards[words.OBSERVED_SHOWN] == "60"
    assert cards[words.OBSERVED_RETRIEVED] == "2026-10-16 00:30"
    table = next(
        t.value
        for t in app.table
        if t.value.index.name == "Session date" and "Notes" in t.value.columns
    )
    assert table.index[0] == AS_OF and len(table) == 60  # newest first
    assert not {"Increase", "Stable", "Decrease"} & set(" ".join(table.to_numpy().ravel()).split())
    assert _observed_keys(app) == ["stock-lab-observed-chart"]
    chart = next(c for c in app.get("plotly_chart") if c.key == "stock-lab-observed-chart")
    assert words.OBSERVED_CHART_SUBTITLE.format(retrieved="2026-10-16 00:30") in chart.proto.spec
    # The research history is unchanged: still through 2024-12-31 only.
    assert "Shown through 2024-12-31." in text and words.REAL_CUTOFF in text
    research = next(c for c in app.get("plotly_chart") if c.key == "stock-lab-price-chart")
    assert "2025-" not in research.proto.spec and "2026-" not in research.proto.spec
    scan_for_advice(text)


def test_recent_prices_follow_the_dark_scheme(home, inference_home, monkeypatch):
    from stock_agent.ui import components, theme

    _real_history(home)
    _recent_batch(inference_home)
    light = page()
    monkeypatch.setattr(components, "colour_scheme", lambda: "dark")
    dark = page()
    assert text_of(dark) == text_of(light)
    spec = json.loads(next(c for c in dark.get("plotly_chart") if "observed" in c.key).proto.spec)
    colours = theme.LAB_DARK_COLOURS
    assert spec["layout"]["paper_bgcolor"] == spec["layout"]["plot_bgcolor"] == colours["surface"]
    assert spec["data"][0]["line"]["color"] == colours["accent"]
    for scheme in theme.SCHEMES:  # the observed line keeps 3:1 against the chart surface
        palette = theme.LAB_PALETTES[scheme]
        assert theme.contrast_ratio(palette["accent"], palette["surface"]) >= 3.0


def test_an_unreadable_recent_store_is_said_never_shown(home, inference_home):
    _real_history(home)
    inference_home.mkdir(mode=0o700)
    (inference_home / INFERENCE_DB_NAME).write_bytes(b"not a database at all" * 20)
    app = page()
    text = text_of(app)
    assert words.OBSERVED_UNREADABLE_TITLE in text and "not_inference_store" in text
    assert _observed_keys(app) == []


def _recent_fixture(forecast_as_of=None, targets=()):
    from types import SimpleNamespace

    days = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
    return SimpleNamespace(
        sessions=tuple(
            SimpleNamespace(
                session=day,
                close=None if day == "2026-10-13" else 10.0,
                level=None if day == "2026-10-13" else 10.0,
                change=None,
                notes=("gap",) if day == "2026-10-13" else (),
                forecast_target=day in targets,
            )
            for day in days
        ),
        forecast_as_of=forecast_as_of,
        retrieved_at="2026-10-16T00:30:00.000000Z",
    )


def test_the_forecast_line_is_drawn_only_inside_the_shown_sessions():
    from stock_agent.app import stock_figures

    inside = stock_figures.observed_figure(_recent_fixture("2026-10-13"))
    assert [shape.x0 for shape in inside.layout.shapes] == [pd.Timestamp("2026-10-13")]
    assert words.OBSERVED_FORECAST_LINE in [a.text for a in inside.layout.annotations]
    old = stock_figures.observed_figure(_recent_fixture("2026-02-02"))  # a stale forecast
    assert old.layout.shapes == () and not old.layout.annotations
    assert "retrieved 2026-10-16 00:30 UTC" in old.layout.title.subtitle.text


def test_the_observed_table_names_gaps_and_the_forecasts_targets_newest_first():
    from stock_agent.app.stock_lab import _observed_table

    table = _observed_table(_recent_fixture("2026-10-13", ("2026-10-14", "2026-10-15")))
    assert table.index.tolist() == ["2026-10-15", "2026-10-14", "2026-10-13", "2026-10-12"]
    notes = table["Notes"].tolist()
    assert notes[:2] == [words.OBSERVED_FORECAST_TARGET] * 2
    assert notes[2] == "gap" and notes[3] == "—"
    assert table["Close as served"].tolist()[2] == "—"
