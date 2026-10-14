"""
The Explore page under Streamlit's AppTest, one test per first-class state:
the running app with every metric pending approval, a fake provider with
hand-written results, loading, no fund selected, stale data, an unavailable
fund, data gaps, periods (presets, custom years, clamping), no common start
and a metric unavailable for one fund. Plus the launcher and the app's
import boundary.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stock_agent.app import explore, labels, synthetic
from stock_agent.app.provider import (
    GROWTH_START_VALUE,
    PENDING_DETAIL,
    DrawdownSummary,
    MetricUnavailable,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"
PACKAGE = SOURCE / "stock_agent" / "app"
RUN = PACKAGE / "run.py"
TIMEOUT = 120
EQUITY, BOND, CASH, STALE, MISSING = (h.fund_id for h in synthetic.demo_histories())


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


DRAWDOWN = DrawdownSummary(
    max_drawdown=0.3456,
    peak_date=utc("2007-10-09"),
    trough_date=utc("2009-03-09"),
    recovery_date=utc("2012-03-30"),
    duration_sessions=1166,
    duration_days=1634,
    dollar_decline=4321.0,
    recovered=True,
)
OPEN_DRAWDOWN = DrawdownSummary(
    max_drawdown=0.123,
    peak_date=utc("2021-12-31"),
    trough_date=utc("2022-10-12"),
    recovery_date=None,
    duration_sessions=783,
    duration_days=1096,
    dollar_decline=1500.0,
    recovered=False,
)
CALENDAR = pd.DataFrame(
    {
        "year": [2022, 2023, 2024],
        "total_return": [-0.0004, 0.08, -0.05],
        "partial": [True, False, True],
        "losing": [True, False, True],
    }
)


class FakeMetrics:
    """Fixed, hand-written results. It computes no formula: the gap NaNs are copied through."""

    def __init__(self, start="2007-01-03", unavailable=(), drawdown=DRAWDOWN):
        self.start = None if start is None else utc(start)
        self.unavailable = set(unavailable)
        self.drawdown = drawdown
        self.calls = []

    def common_start(self, histories, window_start):
        self.calls.append((tuple(h.fund_id for h in histories), window_start))
        return self.start

    def growth_index(self, daily_total_returns, start_value=GROWTH_START_VALUE):
        if daily_total_returns.name in self.unavailable:
            raise MetricUnavailable(
                "growth_index", "too_few_sessions", "fewer sessions than needed"
            )
        values = np.full(len(daily_total_returns), 12_345.0)
        values[daily_total_returns.isna().to_numpy()] = np.nan
        values[0] = start_value
        return pd.Series(values, index=daily_total_returns.index)

    def drawdown_summary(self, growth):
        return self.drawdown

    def calendar_year_returns(self, growth):
        return CALENDAR.copy()


def _page(histories, metrics):
    from stock_agent.app import explore
    from stock_agent.ui.components import apply_theme

    apply_theme()
    explore.render(lambda: histories, metrics)


def _app(metrics=None, histories=None) -> AppTest:
    app = AppTest.from_function(
        _page,
        default_timeout=TIMEOUT,
        kwargs={
            "histories": histories or synthetic.demo_histories(),
            "metrics": metrics or FakeMetrics(),
        },
    )
    return app.run()


def _text(app) -> str:
    """Every rendered text as read: markdown's dollar escapes and bold markers removed."""

    parts = [
        e.value
        for e in (
            *app.title,
            *app.subheader,
            *app.markdown,
            *app.caption,
            *app.info,
            *app.warning,
            *app.error,
        )
    ]
    parts += [html.proto.body for html in app.get("html") if "<style>" not in html.proto.body]
    parts += [frame.value.to_csv(index=False) for frame in app.dataframe]
    parts += [f"{metric.label} {metric.value}" for metric in app.metric]
    return " ".join(parts).replace("\\$", "$").replace("**", "")


def _charts(app) -> dict[str, object]:
    """Charts by their key without the page prefix: growth, calendar."""

    charts = app.get("plotly_chart")
    return {chart.proto.id.rsplit("-", 1)[-1].removeprefix("explore_"): chart for chart in charts}


def _choose(app, *fund_ids):
    return app.multiselect(key=explore.FUNDS_KEY).set_value(list(fund_ids)).run()


def _assert_honest_labels(text: str) -> None:
    for label in (
        labels.GROWTH_TITLE,
        *labels.HONEST_LABELS,
        labels.EVIDENCE_BADGE,
        labels.CANNOT_TELL,
    ):
        assert label in text, label


# --- the running app: synthetic demo mode, every metric pending ---


def test_the_running_app_shows_every_metric_pending_approval():
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    assert not app.exception, [e.value for e in app.exception]
    text = _text(app)
    assert not app.get("plotly_chart")
    assert text.count(f"Unavailable: {PENDING_DETAIL}.") == 5  # growth, 3 decline cards, calendar
    assert labels.PENDING_EXPLAINER in text
    assert [m.value for m in app.metric] == ["—", "—", "—"]
    assert text.count(labels.PENDING_BADGE) == 2
    _assert_honest_labels(text)
    assert labels.SYNTHETIC_NOTICE in text and labels.SYNTHETIC_BADGE in text
    assert "Synthetic demo data, not real fund performance" in text
    # Sources, coverage and vintage are shown even while metrics are pending.
    sources = next(f.value for f in app.dataframe if "Vintage" in f.value.columns)
    assert sources["Coverage"].tolist() == ["2007-01-01 to 2024-12-31"] * 3
    assert sources["Evidence"].eq("descriptive").all()
    assert sources["Vintage"].str.contains(str(synthetic.SEED)).all()


def test_the_theme_is_applied_once():
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    styles = [h for h in app.get("html") if "<style>" in h.proto.body]
    assert len(styles) == 1 and ".sa-pills" in styles[0].proto.body


# --- a provider with results ---


def test_a_provider_fills_every_section():
    app = _app()
    assert not app.exception, [e.value for e in app.exception]
    text = _text(app)
    assert set(_charts(app)) == {"growth", "calendar"}
    # Worst decline: a card per fund, percent with one decimal, dollars, dates, duration.
    assert [m.value for m in app.metric] == ["−34.6%"] * 3
    assert "$4,321 below its peak, from a hypothetical $10,000 start." in text
    assert "Peak 2007-10-09 · low 2009-03-09." in text
    assert "Recovered on 2012-03-30: 1,166 sessions (1,634 calendar days)" in text
    assert labels.DAILY_CLOSES in text
    # Growth: one sentence, the chart and its table fallback.
    assert "A hypothetical $10,000 placed in each fund on 2007-01-03." in text
    assert "Synthetic equity fund $12,345 on 2024-12-31" in text
    year_ends = next(f.value for f in app.dataframe if "Session date" in f.value.columns)
    assert year_ends.iloc[0].tolist() == ["2007-01-03", "$10,000", "$10,000", "$10,000"]
    assert year_ends["Session date"].iloc[-1] == "2024-12-31" and len(year_ends) == 19
    # Calendar-year returns: losing years in words, partial years labelled.
    assert "lost money in 2022, 2024: 2 of 3 calendar years shown (2 partial)" in text
    years = next(f.value for f in app.dataframe if "Year" in f.value.columns)
    assert years["Synthetic bond fund"].tolist() == [
        "−0.0% · lost money · partial year",
        "+8.0%",
        "−5.0% · lost money · partial year",
    ]
    assert labels.PENDING_EXPLAINER not in text
    _assert_honest_labels(text)


def test_the_growth_chart_is_labelled_and_draws_each_fund_distinctly():
    app = _app()
    spec = json.loads(_charts(app)["growth"].proto.spec)
    traces = spec["data"]
    assert [t["name"] for t in traces] == [
        "Synthetic equity fund",
        "Synthetic bond fund",
        "Synthetic cash fund",
    ]
    assert (
        len({(t["line"]["color"], t["line"]["dash"], t["marker"]["symbol"]) for t in traces}) == 3
    )
    assert spec["layout"]["yaxis"]["type"] == "log"
    assert "log scale" in spec["layout"]["yaxis"]["title"]["text"]
    assert labels.LOG_SCALE in _text(app)


def test_an_unrecovered_decline_says_so():
    app = _app(FakeMetrics(drawdown=OPEN_DRAWDOWN))
    text = _text(app)
    assert "Not recovered by 2024-12-31: 783 sessions (1,096 calendar days) since the peak." in text
    assert [m.value for m in app.metric] == ["−12.3%"] * 3


# --- loading ---


def _loading():
    from stock_agent.app import explore

    explore.loading_state()


def test_the_loading_state_is_announced_and_then_replaced():
    loading = AppTest.from_function(_loading, default_timeout=TIMEOUT).run()
    assert [c.value for c in loading.caption] == [labels.LOADING]
    assert any('role="status"' in h.proto.body for h in loading.get("html"))
    assert labels.LOADING not in _text(_app())


# --- no fund selected ---


def test_no_fund_selected_shows_the_empty_state():
    app = _choose(_app())
    text = _text(app)
    assert any(labels.EMPTY in i.value for i in app.info)
    assert not app.get("plotly_chart") and not app.metric
    assert not app.radio  # no period without a fund
    _assert_honest_labels(text)


# --- stale data ---


def test_a_stale_fund_is_flagged_and_limits_the_shared_period():
    app = _choose(_app(), EQUITY, STALE)
    text = _text(app)
    assert labels.STALE_BADGE in text
    assert any(
        "Synthetic bond fund, stale copy is stale. Not refreshed: its records end 2023-06-30"
        in w.value
        for w in app.warning
    )
    assert "Shown: 2007-01-01 to 2023-06-30." in text
    sources = next(f.value for f in app.dataframe if "Freshness" in f.value.columns)
    assert sources["Freshness"].tolist()[1].startswith("Stale: Not refreshed")
    app.radio(key=explore.PERIOD_KEY).set_value("last_10").run()
    limited = "Shown: 2014-01-01 to 2023-06-30, limited to the dates every selected fund covers."
    assert limited in _text(app)


# --- an unavailable fund ---


def test_an_unavailable_fund_is_excluded_not_set_to_zero():
    metrics = FakeMetrics()
    app = _choose(_app(metrics), EQUITY, MISSING)
    message = (
        "Synthetic international equity fund has no available data, so it is excluded, "
        "not set to zero (reason: missing_input)."
    )
    assert any(message in w.value for w in app.warning)
    assert metrics.calls[-1][0] == (EQUITY,)  # only the fund with data reaches the provider
    spec = json.loads(_charts(app)["growth"].proto.spec)
    assert [t["name"] for t in spec["data"]] == ["Synthetic equity fund"]


def test_only_unavailable_funds_show_a_message_not_charts():
    app = _choose(_app(), MISSING)
    assert any(labels.NO_AVAILABLE_FUND in i.value for i in app.info)
    assert not app.get("plotly_chart") and not app.metric
    assert "Choose a fund with data to set a period." in _text(app)


# --- data gaps ---


def test_a_data_gap_is_listed_and_drawn_as_a_gap():
    app = _app()
    gap = (
        "Synthetic equity fund has 5 sessions without data in this period: 2015-08-17 to "
        "2015-08-21, 5 sessions (missing_input). Missing returns are never filled or set to zero."
    )
    assert any(gap in i.value for i in app.info)
    spec = json.loads(_charts(app)["growth"].proto.spec)
    equity = spec["data"][0]
    assert equity["connectgaps"] is False
    app.radio(key=explore.PERIOD_KEY).set_value("year_2022").run()
    assert not any("without data" in i.value for i in app.info)


# --- periods ---


@pytest.mark.parametrize(
    ("preset", "shown"),
    [
        ("full", "Shown: 2007-01-01 to 2024-12-31."),
        ("last_10", "Shown: 2015-01-01 to 2024-12-31."),
        ("crisis_2008", "Shown: 2007-01-01 to 2009-12-31."),
        ("year_2022", "Shown: 2022-01-01 to 2022-12-31."),
    ],
)
def test_period_presets_select_calendar_years(preset, shown):
    # The fake's fixed start lies in 2007; later windows get "no common start" instead.
    metrics = FakeMetrics(start="2007-01-03" if preset in {"full", "crisis_2008"} else None)
    app = _app(metrics)
    app.radio(key=explore.PERIOD_KEY).set_value(preset).run()
    assert not app.exception
    assert shown in _text(app)
    window_start = pd.Timestamp(shown.split()[1], tz="UTC")
    assert metrics.calls[-1][1] == window_start


def test_custom_years_are_chosen_within_the_coverage():
    metrics = FakeMetrics(start="2010-01-04")
    app = _app(metrics)
    app.radio(key=explore.PERIOD_KEY).set_value("custom").run()
    slider = app.select_slider[0]
    assert slider.options[0] == "2007" and slider.options[-1] == "2024"
    slider.set_range(2010, 2012).run()
    assert "Shown: 2010-01-01 to 2012-12-31." in _text(app)
    assert metrics.calls[-1][1] == utc("2010-01-01")


def _windowed(*windows):
    """Demo funds cut to the given (start, end) windows, in order."""

    funds = synthetic.demo_histories()
    return tuple(f.between(utc(s), utc(e)) for f, (s, e) in zip(funds, windows, strict=False))


def test_funds_without_a_common_period_show_a_message_not_charts():
    app = _app(histories=_windowed(("2015-01-01", "2024-12-31"), ("2007-01-01", "2009-12-31")))
    assert app.multiselect(key=explore.FUNDS_KEY).value == [EQUITY, BOND]
    assert any(labels.NO_SHARED_PERIOD in i.value for i in app.info)
    assert not app.get("plotly_chart") and not app.radio


def test_a_preset_outside_the_coverage_says_so():
    histories = _windowed(("2010-01-01", "2024-12-31"), ("2010-01-01", "2024-12-31"))
    app = _app(FakeMetrics(start=None), histories)
    app.radio(key=explore.PERIOD_KEY).set_value("crisis_2008").run()
    message = (
        "2008 crisis (2007-2009) is outside the selected funds' shared coverage "
        "(2010-01-01 to 2024-12-31). Choose another period."
    )
    assert any(message in i.value for i in app.info)
    assert not app.get("plotly_chart") and not app.metric


# --- the shared start and unavailable metrics ---


def test_no_common_start_is_a_state_not_a_chart():
    app = _app(FakeMetrics(start=None))
    text = _text(app)
    assert not app.get("plotly_chart")
    assert f"Unavailable: {labels.NO_COMMON_START}" in text
    assert [m.value for m in app.metric] == ["—", "—", "—"]
    assert labels.UNAVAILABLE_BADGE in text and labels.PENDING_BADGE not in text


def test_a_metric_unavailable_for_one_fund_names_it_and_keeps_the_others():
    app = _app(FakeMetrics(unavailable={CASH}))
    text = _text(app)
    assert "Hypothetical growth of $10,000 is not shown for Synthetic cash fund." in text
    assert "Unavailable: fewer sessions than needed." in text
    spec = json.loads(_charts(app)["growth"].proto.spec)
    assert [t["name"] for t in spec["data"]] == ["Synthetic equity fund", "Synthetic bond fund"]
    assert [m.value for m in app.metric] == ["−34.6%", "−34.6%", "—"]


# --- wording ---


def test_no_state_uses_trading_or_advice_wording():
    apps = [
        AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run(),
        _app(),
        _choose(_app()),
        _choose(_app(), EQUITY, STALE, MISSING),
    ]
    for app in apps:
        text = _text(app).lower()
        assert not re.search(r"\b(buy|sell|buying|selling|recommend\w*)\b", text)


# --- structure, launcher and boundary ---


def test_the_page_builds_from_the_shared_components():
    source = (PACKAGE / "explore.py").read_text(encoding="utf-8")
    for banned in (
        "st.title(",
        "unsafe_allow_html",
        "st.error(",
        "st.warning(",
        "st.info(",
        "st.success(",
        "st.html(",
        "rgba(",
        "#",
    ):
        lines = [line for line in source.splitlines() if banned in line]
        if banned == "#":  # comments are fine; colour literals are not
            lines = [line for line in lines if re.search(r"#[0-9A-Fa-f]{6}\b", line)]
        assert not lines, (banned, lines)


def test_the_launcher_binds_to_localhost_without_usage_statistics():
    path = REPO / "scripts" / "run_portfolio_lab.py"
    spec = importlib.util.spec_from_file_location("stock_agent_run_portfolio_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.command(8502)
    pairs = dict(zip(command[5::2], command[6::2], strict=True))
    assert command[1:4] == ["-m", "streamlit", "run"] and command[4] == str(RUN)
    assert pairs["--server.address"] == "127.0.0.1"
    assert pairs["--browser.gatherUsageStats"] == "false"
    assert pairs["--server.headless"] == "true"


ALLOWED_THIRD_PARTY = {"numpy", "pandas", "plotly", "streamlit"}
FORBIDDEN = (
    "stock_agent.dashboard",
    "stock_agent.model_diagnostics",
    "stock_agent.model_validation",
    "stock_agent.data",
    "stock_agent.features",
    "stock_agent.publication.writer",
    "sklearn",
    "duckdb",
    "yfinance",
    "edgar",
)


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.stem)
def test_app_modules_import_only_ui_their_own_modules_and_allowed_libraries(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for module in imported:
        top = module.split(".")[0]
        allowed = (
            top in sys.stdlib_module_names
            or top in ALLOWED_THIRD_PARTY
            or module.startswith(("stock_agent.app", "stock_agent.ui"))
        )
        assert allowed, (path.name, module)


def test_rendering_the_app_loads_no_forbidden_module():
    script = f"""
import json, sys
from streamlit.testing.v1 import AppTest
app = AppTest.from_file({str(RUN)!r}, default_timeout={TIMEOUT})
app.run()
assert not app.exception, [e.value for e in app.exception]
print(json.dumps(sorted(sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert not sorted(m for m in loaded if m.startswith(FORBIDDEN))
    project = sorted(m for m in loaded if m.startswith("stock_agent"))
    outside = [
        m
        for m in project
        if m != "stock_agent" and not m.startswith(("stock_agent.app", "stock_agent.ui"))
    ]
    assert not outside, outside
    assert "stock_agent.app.explore" in loaded
