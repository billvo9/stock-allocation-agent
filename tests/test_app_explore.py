"""
The Explore page under Streamlit's AppTest, one test per first-class state:
the running app with the approved calculations on synthetic data, a fake
provider with hand-written results, loading, no fund selected, stale data,
an unavailable fund, sessions without a close, a labelled multi-session
return, a growth chain that cannot continue, an incomplete history, periods
(presets, custom years, clamping), no common start and a metric unavailable
for one fund. Plus the launcher and the import boundaries of the app and of
`stock_agent.portfolio`.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
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
from stock_agent.portfolio.descriptive import (
    GROWTH_START_VALUE,
    DescriptiveCalculator,
    DrawdownSummary,
    MetricUnavailable,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"
PACKAGE = SOURCE / "stock_agent" / "app"
PORTFOLIO = SOURCE / "stock_agent" / "portfolio"
RUN = PACKAGE / "run.py"
EXPLORE_PAGE = "destinations/explore.py"
TIMEOUT = 120
EQUITY, BOND, CASH, STALE, DISTRIBUTION_GAP, MISSING = (
    h.fund_id for h in synthetic.demo_histories()
)


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
    history_complete=True,
    missing_sessions=0,
)
OPEN_DRAWDOWN = DrawdownSummary(
    max_drawdown=0.123,
    peak_date=utc("2021-12-31"),
    trough_date=utc("2022-10-12"),
    recovery_date=None,
    duration_sessions=783,
    duration_days=1096,
    dollar_decline=1500.0,
    history_complete=True,
    missing_sessions=0,
)
CALENDAR = pd.DataFrame(
    {
        "year": [2022, 2023, 2024],
        "total_return": [-0.0004, 0.08, -0.05],
        "partial": [True, False, True],
        "losing": [True, False, True],
        "status": ["ok", "ok", "ok"],
        "reason": [None, None, None],
    }
)


class FakeMetrics:
    """
    Fixed, hand-written results that honour the provider contract. It
    computes no formula: growth is 12,345 after t0 and NaN wherever the
    return is NaN, as the contract requires.
    """

    def __init__(self, start="2007-01-03", unavailable=(), drawdown=DRAWDOWN, calendar=CALENDAR):
        self.start = None if start is None else utc(start)
        self.unavailable = set(unavailable)
        self.drawdown = drawdown
        self.calendar = calendar
        self.calls = []

    def common_start(self, available_sessions, window_start):
        self.calls.append((tuple(available_sessions), window_start))
        return self.start

    def growth_index(self, returns, sessions_spanned, start_value=GROWTH_START_VALUE):
        if returns.name in self.unavailable:
            raise MetricUnavailable(
                "growth_index", "too_few_sessions", "fewer sessions than needed"
            )
        values = np.full(len(returns), 12_345.0)
        values[returns.isna().to_numpy()] = np.nan
        values[0] = start_value
        return pd.Series(values, index=returns.index)

    def drawdown_summary(self, growth):
        return self.drawdown

    def calendar_year_returns(self, growth, year_end_sessions):
        return self.calendar.copy()


class GapFillingMetrics(FakeMetrics):
    """Breaks the contract: it carries the last value through a gap."""

    def growth_index(self, returns, sessions_spanned, start_value=GROWTH_START_VALUE):
        return super().growth_index(returns, sessions_spanned, start_value).ffill()


ADVICE_WORDS = (" buy", " sell", "recommend", "should invest", "outperform")


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
    text = " ".join(parts).replace("\\$", "$").replace("**", "")
    # Every state test reads the page through here, so every rendered state is
    # checked for trading or advice wording, including composed strings.
    lowered = text.lower()
    for word in ADVICE_WORDS:
        assert word not in lowered, word
    return text


def _charts(app) -> dict[str, object]:
    """Charts by their key without the page prefix: growth, calendar."""

    charts = app.get("plotly_chart")
    return {chart.proto.id.rsplit("-", 1)[-1].removeprefix("explore_"): chart for chart in charts}


def _running_explore() -> AppTest:
    """The running app (run.py) on its Explore destination, the approved calculations included."""

    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    return app.switch_page(EXPLORE_PAGE).run()


def _choose(app, *fund_ids):
    return app.multiselect(key=explore.FUNDS_KEY).set_value(list(fund_ids)).run()


def _assert_honest_labels(text: str) -> None:
    for label in (
        labels.GROWTH_TITLE,
        *labels.HONEST_LABELS,
        labels.FRACTIONAL_SHARES,
        labels.EVIDENCE_BADGE,
        labels.CANNOT_TELL,
    ):
        assert label in text, label


def _gapped_equity(*dates: str):
    """
    The demo funds with the equity fund's closes on consecutive `dates`
    removed. The next close's return then spans the gap; with no
    distribution or split inside, it is the product of the one-session
    returns it replaces.
    """

    equity, *others = synthetic.demo_histories()
    records = equity.records.copy()
    hit = records["date"].isin([utc(d) for d in dates])
    after = int(np.flatnonzero(hit.to_numpy())[-1]) + 1
    covered = records.index[hit.to_numpy()].tolist() + [after]
    spanned = float(np.prod(1 + records.loc[covered, "total_return"]) - 1)
    records.loc[hit, ["close_observed", "total_return", "sessions_spanned", "status", "reason"]] = [
        False,
        np.nan,
        0,
        "unavailable",
        "missing_input",
    ]
    records.loc[after, ["total_return", "sessions_spanned"]] = [spanned, len(covered)]
    return (dataclasses.replace(equity, records=records), *others)


# --- the running app: the approved calculations on synthetic data ---


def _independent_growth(history, start):
    """Growth recomputed here by plain compounding, apart from the calculator."""

    records = history.records[history.records["date"].ge(start)]
    ok = records["status"].eq("ok").to_numpy().copy()
    ok[0] = False  # t0's own return is never applied
    return GROWTH_START_VALUE * float(np.prod(1 + records["total_return"].to_numpy()[ok]))


def test_the_running_app_shows_calculated_synthetic_results():
    app = _running_explore()
    assert not app.exception, [e.value for e in app.exception]
    text = _text(app)
    assert set(_charts(app)) == {"growth", "calendar"}
    _assert_honest_labels(text)
    assert labels.SYNTHETIC_NOTICE in text and labels.SYNTHETIC_BADGE in text
    assert labels.HISTORICAL_BADGE not in text  # synthetic records are not market data
    # Values: each fund's last growth value matches plain compounding of its returns.
    equity, bond, cash = synthetic.demo_histories()[:3]
    start = utc("2007-01-01")
    assert "A hypothetical $10,000 placed in each fund on 2007-01-01." in text
    for history in (equity, bond, cash):
        expected = view_money(_independent_growth(history, start))
        assert f"{history.name} {expected} on 2024-12-31" in text
    # Maximum drawdown: complete histories say so; the equity fund's gap makes it observed only.
    assert [m.label for m in app.metric] == [
        labels.OBSERVED_DECLINE,
        labels.MAX_DRAWDOWN,
        labels.MAX_DRAWDOWN,
    ]
    assert "Incomplete history: 5 sessions in this period have no value" in text
    # The equity gap is one labelled six-session return.
    assert "Synthetic equity fund, multi-session return. 2015-08-24:" in text
    assert "over 6 sessions (5 sessions without a close)" in text
    sources = next(f.value for f in app.dataframe if "Vintage" in f.value.columns)
    assert sources["Coverage"].tolist() == ["2007-01-01 to 2024-12-31"] * 3
    assert sources["Evidence"].eq("descriptive").all()
    assert sources["Vintage"].str.contains(str(synthetic.SEED)).all()


def view_money(value: float) -> str:
    from stock_agent.app import view

    return view.money(value)


def test_a_break_stops_the_growth_and_makes_later_years_unavailable():
    app = _choose(_running_explore(), DISTRIBUTION_GAP)
    assert not app.exception, [e.value for e in app.exception]
    text = _text(app)
    assert "From 2019-03-08, the hypothetical growth cannot be continued" in text
    assert labels.reason_text("distribution_in_gap") in text
    assert "on 2019-02-28" in text  # the last value before the break
    years = next(f.value for f in app.dataframe if "Year" in f.value.columns)
    by_year = dict(zip(years["Year"], years.iloc[:, 1], strict=True))
    assert by_year[2018] != "unavailable"
    assert by_year[2019] == by_year[2024] == "unavailable"
    assert [m.label for m in app.metric] == [labels.OBSERVED_DECLINE]


def test_real_records_are_badged_as_historical_market_data():
    real = [
        dataclasses.replace(
            history,
            limitations=(),
            source_label="Test vendor, private evaluation",
            retrieved_label="Retrieved in a test",
            vintage_note="Test vintage",
        )
        for history in synthetic.demo_histories()[:3]
    ]
    app = _app(DescriptiveCalculator(), tuple(real))
    text = _text(app)
    assert labels.HISTORICAL_BADGE in text
    assert labels.SYNTHETIC_BADGE not in text and labels.SYNTHETIC_NOTICE not in text


def test_the_theme_is_applied_once():
    app = _running_explore()
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
    assert [m.label for m in app.metric] == [labels.MAX_DRAWDOWN] * 3
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
    _assert_honest_labels(text)


def test_the_honest_labels_sit_directly_under_the_growth_heading():
    app = _app()
    elements = [e for e in app.main if not hasattr(e, "children")]  # leaves, in page order
    headings = [
        i
        for i, e in enumerate(elements)
        if e.type == "subheader" and e.value.replace("\\$", "$") == labels.GROWTH_TITLE
    ]
    assert len(headings) == 1
    below = elements[headings[0] + 1]
    assert below.type == "caption"
    assert below.value == " ".join(labels.HONEST_LABELS)


def test_a_viewer_in_dark_mode_gets_dark_charts_with_the_same_content(monkeypatch):
    from stock_agent.ui import components, theme

    light = _app()
    monkeypatch.setattr(components, "colour_scheme", lambda: "dark")
    dark = _app()
    assert _text(dark) == _text(light)  # the scheme changes colours, never the words
    for key in ("growth", "calendar"):
        spec = json.loads(_charts(dark)[key].proto.spec)
        layout = spec["layout"]
        assert (
            layout["paper_bgcolor"] == layout["plot_bgcolor"] == theme.LAB_DARK_COLOURS["surface"]
        )
        assert layout["font"]["color"] == theme.LAB_DARK_COLOURS["text"]
        light_spec = json.loads(_charts(light)[key].proto.spec)
        assert [t["y"] for t in spec["data"]] == [t["y"] for t in light_spec["data"]]


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


def test_an_unrecovered_decline_ends_at_the_last_session_with_data():
    # The equity fund's last two sessions of 2009 are a gap: the crisis period ends inside it.
    open_2008 = dataclasses.replace(DRAWDOWN, recovery_date=None)
    calendar = CALENDAR.assign(year=[2007, 2008, 2009])
    app = _app(
        FakeMetrics(drawdown=open_2008, calendar=calendar),
        _gapped_equity("2009-12-30", "2009-12-31"),
    )
    app.radio(key=explore.PERIOD_KEY).set_value("crisis_2008").run()
    assert not app.exception, [e.value for e in app.exception]
    captions = [c.value for c in app.caption if c.value.startswith("Not recovered")]
    assert [c.split(":")[0] for c in captions] == [
        "Not recovered by 2009-12-29",  # equity: its last session with data
        "Not recovered by 2009-12-31",
        "Not recovered by 2009-12-31",
    ]


# --- loading ---


def _loading():
    from stock_agent.app import explore

    explore.loading_state()


def _failing_load():
    from stock_agent.app import explore
    from stock_agent.portfolio.descriptive import DescriptiveCalculator

    def load():
        raise RuntimeError("the history source is unreachable")

    explore.render(load, DescriptiveCalculator())


def test_the_loading_state_is_announced_and_then_replaced():
    loading = AppTest.from_function(_loading, default_timeout=TIMEOUT).run()
    assert [c.value for c in loading.caption] == [labels.LOADING]
    assert any('role="status"' in h.proto.body for h in loading.get("html"))
    assert labels.LOADING not in _text(_app())


def test_the_page_shows_loading_while_histories_load():
    # A loader that never finishes leaves the page in its loading state; the failure
    # itself is a defect and propagates.
    app = AppTest.from_function(_failing_load, default_timeout=TIMEOUT).run()
    assert [e.value for e in app.exception] and "unreachable" in app.exception[0].value
    assert labels.LOADING in [c.value for c in app.caption]
    assert any('role="status"' in h.proto.body for h in app.get("html"))
    assert not app.multiselect and not app.metric


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
    # The fake's fixed results describe 2007-2024; this period ends in 2023.
    app = _choose(_app(FakeMetrics(start=None)), EQUITY, STALE)
    assert not app.exception, [e.value for e in app.exception]
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
        "not set to zero (no close from the source)."
    )
    assert any(message in w.value for w in app.warning)
    assert "missing_input" not in _text(app)  # reason codes are translated
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
        "Synthetic equity fund has 5 sessions without a close in this period: 2015-08-17 to "
        "2015-08-21, 5 sessions. Missing values are never filled or set to zero."
    )
    assert any(gap in i.value for i in app.info)
    spec = json.loads(_charts(app)["growth"].proto.spec)
    equity = spec["data"][0]
    assert equity["connectgaps"] is False
    # Plotly ships y as a typed array; the five gap sessions are drawn as NaN, not filled.
    y = np.frombuffer(base64.b64decode(equity["y"]["bdata"]), dtype=equity["y"]["dtype"])
    x = pd.to_datetime(pd.Series(equity["x"]), utc=True)
    in_gap = x.between(utc("2015-08-17"), utc("2015-08-21")).to_numpy()
    assert in_gap.sum() == 5 and np.isnan(y[in_gap]).all() and not np.isnan(y[~in_gap]).any()
    app.radio(key=explore.PERIOD_KEY).set_value("year_2022").run()
    assert not any("without a close" in i.value for i in app.info)


def test_a_provider_that_fills_a_gap_is_refused():
    app = _app(GapFillingMetrics())
    assert app.exception and "filled a gap" in app.exception[0].value
    assert not app.get("plotly_chart")


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
    # The fake's fixed results describe the full history; shorter windows get no start.
    metrics = FakeMetrics(start="2007-01-03" if preset == "full" else None)
    app = _app(metrics)
    app.radio(key=explore.PERIOD_KEY).set_value(preset).run()
    assert not app.exception, [e.value for e in app.exception]
    assert shown in _text(app)
    window_start = pd.Timestamp(shown.split()[1], tz="UTC")
    assert metrics.calls[-1][1] == window_start


def test_custom_years_are_chosen_within_the_coverage():
    metrics = FakeMetrics(start=None)
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
    assert labels.UNAVAILABLE_BADGE in text


def test_a_metric_unavailable_for_one_fund_names_it_and_keeps_the_others():
    app = _app(FakeMetrics(unavailable={CASH}))
    text = _text(app)
    assert f"{labels.GROWTH_TITLE} is not shown for Synthetic cash fund." in text
    assert "Unavailable: fewer sessions than needed." in text
    spec = json.loads(_charts(app)["growth"].proto.spec)
    assert [t["name"] for t in spec["data"]] == ["Synthetic equity fund", "Synthetic bond fund"]
    assert [m.value for m in app.metric] == ["−34.6%", "−34.6%", "—"]


# --- markdown escaping ---


def test_dollar_signs_are_escaped_exactly_once():
    for app in (_app(), _running_explore()):
        rendered = [
            e.value for e in (*app.subheader, *app.markdown, *app.caption, *app.info, *app.warning)
        ]
        with_dollars = [text for text in rendered if "$" in text]
        assert with_dollars
        for text in with_dollars:
            assert "\\\\$" not in text, text  # never escaped twice
            assert re.search(r"(?<!\\)\$", text) is None, text  # never left bare


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
    # One helper escapes markdown dollar signs.
    assert source.count('.replace("$", "\\\\$")') == 1


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
DESTINATIONS = [f"destinations/{path.name}" for path in (PACKAGE / "destinations").glob("*.py")]
# The Model Health page reads summaries with the health center's strict reader
# and view model (standard library only); no other health module, and never a
# collector (which reads private stores).
HEALTH_MODULES = ("stock_agent.health.model", "stock_agent.health.view")
# The Stock Probability page reads the Stock Lab's private store and computes its
# descriptive views (standard library, numpy and pandas only); never its Tiingo
# adapter, which only the owner-run fetch script uses, nor the fitted models and
# their evaluation (logistic, evaluation, predict), whose results it reads as
# stored records. From the inference store it reads only recent observed prices
# through the descriptive reader (decision A6), never the writer or the
# frozen-model reader.
STOCK_LAB_MODULES = tuple(
    f"stock_agent.stock_lab.{name}"
    for name in (
        "contracts",
        "symbols",
        "prices",
        "runs",
        "features",
        "store",
        "synthetic",
        "view",
        "forecast",
        "forecast_view",
        "frozen_model",
        "reversal",
        "traps",
        "scores",
        "estimators",
        "inference.recent",
        "inference.schema",
    )
)
APP_PROJECT_PREFIXES = (
    "stock_agent.app",
    "stock_agent.ui",
    "stock_agent.portfolio",
    *HEALTH_MODULES,
    *STOCK_LAB_MODULES,
)
# Loaded with them: the health and Stock Lab packages themselves (docstrings only).
APP_PROJECT_PACKAGES = {
    "stock_agent",
    "stock_agent.health",
    "stock_agent.stock_lab",
    "stock_agent.stock_lab.inference",
}
FORBIDDEN = (
    "stock_agent.stock_lab.inference.store",
    "stock_agent.stock_lab.inference.window",
    "stock_agent.stock_lab.tiingo",
    "stock_agent.stock_lab.logistic",
    "stock_agent.stock_lab.evaluation",
    "stock_agent.stock_lab.predict",
    "stock_agent.stock_lab.registry",
    "stock_agent.stock_lab.size_simulation",
    "scipy",
    "stock_agent.health.collect",
    "stock_agent.health.github",
    "stock_agent.health.ledger",
    "stock_agent.health.prospective",
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


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def _loaded_after(script: str) -> set[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import json, sys\n{script}\nprint(json.dumps(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_app_modules_import_only_ui_portfolio_their_own_modules_and_allowed_libraries(path):
    for module in _imports(path):
        top = module.split(".")[0]
        allowed = (
            top in sys.stdlib_module_names
            or top in ALLOWED_THIRD_PARTY
            or module.startswith(APP_PROJECT_PREFIXES)
        )
        assert allowed, (path.name, module)


def test_rendering_the_app_loads_no_forbidden_module():
    loaded = _loaded_after(
        "from streamlit.testing.v1 import AppTest\n"
        f"app = AppTest.from_file({str(RUN)!r}, default_timeout={TIMEOUT})\n"
        "app.run()\n"
        "assert not app.exception, [e.value for e in app.exception]\n"
        f"for page in {sorted(DESTINATIONS)!r}:\n"
        "    app.switch_page(page).run()\n"
        "    assert not app.exception, (page, [e.value for e in app.exception])"
    )
    assert not sorted(m for m in loaded if m.startswith(FORBIDDEN))
    project = sorted(m for m in loaded if m.startswith("stock_agent"))
    outside = [
        m
        for m in project
        if m not in APP_PROJECT_PACKAGES and not m.startswith(APP_PROJECT_PREFIXES)
    ]
    assert not outside, outside
    for module in (
        "explore",
        "overview",
        "builder",
        "stock_lab",
        "experiments",
        "model_health",
        "evidence",
    ):
        assert f"stock_agent.app.{module}" in loaded, module
    assert {*HEALTH_MODULES, "stock_agent.app.health_summary"} <= loaded
    assert {"stock_agent.stock_lab.store", "stock_agent.stock_lab.view"} <= loaded


@pytest.mark.parametrize("package", ["health", "stock_lab", "stock_lab/inference"])
def test_the_health_and_stock_lab_package_inits_import_nothing(package):
    # The app may load these packages only because their __init__ runs no import.
    source = (SOURCE / "stock_agent" / package / "__init__.py").read_text(encoding="utf-8")
    assert not [n for n in ast.parse(source).body if isinstance(n, ast.Import | ast.ImportFrom)]


PORTFOLIO_FORBIDDEN = ("stock_agent.app", "stock_agent.ui", "stock_agent.dashboard")


@pytest.mark.parametrize("path", sorted(PORTFOLIO.glob("*.py")), ids=lambda p: p.stem)
def test_portfolio_never_imports_the_app_the_ui_or_the_dashboard(path):
    imported = _imports(path)
    assert not [m for m in imported if m.startswith(PORTFOLIO_FORBIDDEN)]
    assert not [m for m in imported if m.split(".")[0] in {"streamlit", "plotly"}]


def test_importing_portfolio_loads_no_presentation_module():
    loaded = _loaded_after("import stock_agent.portfolio.descriptive")
    assert "stock_agent.portfolio.descriptive" in loaded
    assert not sorted(m for m in loaded if m.startswith(PORTFOLIO_FORBIDDEN))
    assert not [m for m in loaded if m.split(".")[0] in {"streamlit", "plotly"}]
