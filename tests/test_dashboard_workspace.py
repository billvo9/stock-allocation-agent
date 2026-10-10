"""
T3.1 research workspace: the Goyal-Welch page, maturity, descriptive
filters, regimes, the future-state portfolio view, and the design system.
AppTest runs against the compact fixture run and writer-built variants.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from stock_agent.dashboard import components, evaluation, figures, loader, status, style, theme
from stock_agent.model_diagnostics import artifacts, contract

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "stock_agent" / "dashboard" / "app.py"
PAGE = "page_scripts/forecast_error.py"
VARIANT = "20991231T000000Z-dddddddddddd"


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(root, monkeypatch, page: str | None = None) -> AppTest:
    monkeypatch.setenv(loader.ROOT_ENVIRONMENT_VARIABLE, str(root))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    if page:
        app.switch_page(page).run()
    return app


def _variant_root(run, tmp_path, *, record=None, tables=None):
    record = copy.deepcopy(run.record if record is None else record)
    record["run_id"] = VARIANT
    variant = dataclasses.replace(
        run, run_id=VARIANT, record=record, tables={**run.tables, **(tables or {})}
    )
    artifacts.write_run(variant, tmp_path)
    return tmp_path


def _charts(app) -> dict[str, object]:
    return {chart.proto.id.rsplit("-", 1)[-1]: chart for chart in app.get("plotly_chart")}


def _text(app) -> str:
    parts = [e.value for e in (*app.markdown, *app.caption, *app.info, *app.warning, *app.error)]
    return " ".join(parts)


# --- the Goyal-Welch page ---


def test_the_page_draws_the_primary_daily_and_secondary_views(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    assert not app.exception, [e.value for e in app.exception]
    charts = _charts(app)
    assert {"gw_primary", "gw_daily", "gw_observation_weighted"} <= set(charts)
    assert "cumulative date-normalized" in charts["gw_primary"].proto.spec
    assert "observation-weighted" in charts["gw_observation_weighted"].proto.spec
    assert any(s.value == "Stored inference (T2, full sample)" for s in app.subheader)


def test_only_forecast_models_and_recorded_comparators_are_offered(dashboard_root, monkeypatch):
    root, run = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    models = app.selectbox(key="gw_model").options
    forecasts = [m["name"] for m in run.record["spec"]["models"] if m["output_kind"] == "forecast"]
    assert [m.split(" (")[0] for m in models] == forecasts
    assert not any("canary_unsafe_reference" in m for m in models)
    picker = app.multiselect(key="gw_comparators")
    assert "per_symbol_mean: selection control (not a null baseline)" in picker.options
    # Default: every recorded null baseline, never the selection control.
    view = loader.load_run(loader.list_runs(root)[0].path)
    found = evaluation.comparators(run.record, view.models)
    nulls = [c.name for c in found if c.kind == evaluation.NULL_BASELINE]
    assert picker.value == nulls and "per_symbol_mean" not in picker.value


def test_comparing_with_the_selection_control_keeps_its_label(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    app.multiselect(key="gw_comparators").set_value(["per_symbol_mean"]).run()
    assert not app.exception
    spec = _charts(app)["gw_primary"].proto.spec
    assert "selection control (not a null baseline)" in spec and '"dash":"dash"' in spec


def test_a_filter_shows_descriptive_numbers_and_never_inference(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    first, last = app.slider(key="gw_dates").value
    app.slider(key="gw_dates").set_value((first + (last - first) / 2, last)).run()
    assert not app.exception
    assert any("Filtered inference is not precomputed" in i.value for i in app.info)
    assert not any(s.value.startswith("Stored inference") for s in app.subheader)
    assert not app.metric  # no T2 estimate or interval card for a filtered sample
    table = next(f.value for f in app.dataframe if "mean d_t" in f.value.columns)
    assert list(table.columns) == [  # counts and plain averages only
        "comparator",
        "dates",
        "observations",
        "mean d_t",
        "share of dates model better",
    ]
    assert "filtered sample (descriptive)" in _charts(app)["gw_primary"].proto.spec
    assert any("effective sample size" in c.value for c in app.caption)


def test_retrospective_regimes_are_flagged_and_unavailable_ones_explained(
    dashboard_root, monkeypatch
):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    app.selectbox(key="gw_regime").set_value("drawdown_episode_ex_post").run()
    assert any("Retrospective (ex post) labels" in w.value for w in app.warning)
    app.selectbox(key="gw_regime").set_value("universe_volatility_pit").run()
    assert any("stored inputs have no volatility_20d" in i.value for i in app.info)
    assert not app.exception


def test_point_in_time_regimes_filter_when_their_inputs_are_stored(
    dashboard_root, tmp_path, monkeypatch
):
    _, run = dashboard_root
    inputs = run.tables["inputs"].copy()
    rng = np.random.RandomState(0)
    inputs["volatility_20d"] = 0.2 + 0.05 * rng.rand(len(inputs))
    inputs["momentum_20d"] = rng.randn(len(inputs)) * 0.05
    root = _variant_root(run, tmp_path, tables={"inputs": inputs})
    app = _app(root, monkeypatch, PAGE)
    app.selectbox(key="gw_regime").set_value("universe_trend_pit").run()
    assert any(c.value.startswith("Point-in-time.") for c in app.caption)
    assert set(app.selectbox(key="gw_regime_value").options) <= {"up", "down", "flat"}
    app.selectbox(key="gw_regime_value").set_value("up").run()
    assert not app.exception
    assert any("Filtered inference is not precomputed" in i.value for i in app.info)


def test_a_comparator_that_does_not_reproduce_t2_is_not_drawn(
    dashboard_root, tmp_path, monkeypatch
):
    _, run = dashboard_root
    metrics = run.tables["metrics"].copy()
    target = (
        (metrics["model"] == "memorizer")
        & (metrics["metric"] == "mse_improvement_vs_zero")
        & (metrics["inference_method"] == "newey_west")
        & metrics["fold_id"].isna()
    )
    assert target.sum() == 1
    metrics.loc[target, "estimate"] += 0.01
    root = _variant_root(run, tmp_path, tables={"metrics": contract.conform("metrics", metrics)})
    app = _app(root, monkeypatch, PAGE)
    assert app.selectbox(key="gw_model").value == "memorizer"
    assert any("vs zero: null baseline: not drawn" in e.value for e in app.error)
    spec = _charts(app)["gw_primary"].proto.spec
    assert "vs pooled_mean: null baseline" in spec and "vs zero: null baseline" not in spec


def test_the_portfolio_view_is_a_disabled_future_state(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    button = app.button(key="future_portfolio-equity")
    assert button.disabled
    assert "T5" in _text(app)
    assert set(_charts(app)) == {"gw_primary", "gw_daily", "gw_observation_weighted"}
    # Shown even when the page stops early (here: no comparator selected).
    app.multiselect(key="gw_comparators").set_value([]).run()
    assert app.button(key="future_portfolio-equity").disabled and not _charts(app)


def test_the_selection_control_is_never_the_default_comparator(
    dashboard_root, tmp_path, monkeypatch
):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    record["spec"]["config"]["baselines"] = ["per_symbol_mean"]
    root = _variant_root(run, tmp_path, record=record)
    app = _app(root, monkeypatch, PAGE)
    assert app.multiselect(key="gw_comparators").options == [
        "per_symbol_mean: selection control (not a null baseline)"
    ]
    assert app.multiselect(key="gw_comparators").value == []
    assert any("Select at least one comparator" in i.value for i in app.info)
    assert not _charts(app)


def test_the_page_says_when_no_candidate_is_registered(dashboard_root, monkeypatch):
    root, run = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    assert any("registers no candidate model" in c.value for c in app.caption)
    assert any(run.record["spec"]["label"]["label_id"] in c.value for c in app.caption)
    assert "label ends 6 sessions later" in _charts(app)["gw_primary"].proto.spec


# --- maturity ---


def test_pending_predictions_block_the_run_and_stay_listed(dashboard_root, tmp_path, monkeypatch):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    ends = run.tables["predictions"]["target_end_date"].sort_values().unique()
    asof = pd.Timestamp(ends[-6])
    record["lockbox_evidence"]["max_date"] = asof.isoformat()
    root = _variant_root(run, tmp_path, record=record)
    view = loader.load_run(loader.list_runs(root)[0].path)
    predictions = view.tables["predictions"]
    late = predictions["target_end_date"] > asof
    assert view.evaluation_asof == asof and len(view.pending) == int(late.sum()) > 0
    result = status.run_status(
        view.tables["checks"], view.record["checks_summary"], record_issues=view.record_issues
    )
    assert result.level == status.BLOCKED
    assert any("not mature at the evaluation as-of" in reason for reason in result.reasons)
    scored = view.pending[view.pending["role"] != contract.UNSAFE_REFERENCE_ROLE]
    app = _app(root, monkeypatch)
    pending_metric = next(m for m in app.metric if m.label == "Pending predictions")
    assert int(pending_metric.value) == len(scored)
    listed = next(f.value for f in app.dataframe if len(f.value) == len(scored))
    assert set(listed["model"]) == set(scored["model"])  # listed, without the unsafe reference
    assert contract.UNSAFE_REFERENCE_ROLE not in set(listed["role"])
    app.switch_page("page_scripts/validation.py").run()
    assert any("T2's stored results on this page include" in e.value for e in app.error)
    app.switch_page(PAGE).run()
    # T2's stored estimates include the pending rows, so no derived line reproduces them.
    assert any("not drawn" in e.value for e in app.error)
    assert "gw_primary" not in _charts(app)


def test_a_run_without_an_evaluation_asof_cannot_be_verified(dashboard_root, tmp_path):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    record["lockbox_evidence"].pop("max_date")
    root = _variant_root(run, tmp_path, record=record)
    view = loader.load_run(loader.list_runs(root)[0].path)
    assert view.evaluation_asof is None
    assert len(view.pending) == len(view.tables["predictions"])
    assert any("evaluation as-of not recorded" in issue for issue in view.record_issues)
    result = status.run_status(
        view.tables["checks"], view.record["checks_summary"], record_issues=view.record_issues
    )
    assert result.level == status.BLOCKED


def test_the_overview_reports_the_evaluation_sample(dashboard_root, monkeypatch):
    root, run = dashboard_root
    app = _app(root, monkeypatch)
    labels = {m.label: m.value for m in app.metric}
    asof = pd.Timestamp(run.record["lockbox_evidence"]["max_date"]).date()
    assert labels["Evaluation as-of"] == str(asof)
    assert labels["Pending predictions"] == "0"
    predictions = run.tables["predictions"]
    registered = predictions["role"] != contract.UNSAFE_REFERENCE_ROLE
    assert int(labels["Mature predictions"]) == int(registered.sum())


# --- figures ---


@pytest.fixture(scope="module")
def view(dashboard_root):
    root, _ = dashboard_root
    return loader.load_run(loader.list_runs(root)[0].path)


def test_the_primary_figure_plots_the_running_sum_of_d_with_identity(view):
    asof = view.evaluation_asof
    found = {c.name: c for c in evaluation.comparators(view.record, view.models)}
    series = [
        (
            found[name],
            evaluation.squared_error_advantage(view.tables["predictions"], "memorizer", name, asof),
        )
        for name in ("zero", "per_symbol_mean")
    ]
    figure = figures.goyal_welch_date_normalized(series, "memorizer", view.folds, pending=3)
    for (comparator, frame), trace in zip(series, figure.data, strict=True):
        assert trace.name == f"vs {comparator.label}"
        np.testing.assert_allclose(trace.y, frame["d"].cumsum())
    assert figure.data[0].line.dash == "solid" and figure.data[1].line.dash == "dash"
    assert any("3 pending observations" in a.text for a in figure.layout.annotations)
    assert "full sample" in figure.layout.title.subtitle.text
    # A filtered sample sums over included dates only and leaves gaps elsewhere.
    comparator, frame = series[0]
    included = frame["fold_id"] == frame["fold_id"].max()
    marked = frame.assign(included=included)
    filtered = figures.goyal_welch_date_normalized([(comparator, marked)], "memorizer", view.folds)
    y = pd.Series(filtered.data[0].y, dtype="float64")
    assert y[~included.to_numpy()].isna().all()
    np.testing.assert_allclose(y[included.to_numpy()], frame.loc[included, "d"].cumsum())
    assert filtered.data[0].connectgaps is False
    daily = figures.squared_error_advantage_by_date(series[0][1], series[0][0], "memorizer")
    assert list(daily.data[0].y) == series[0][1]["d"].tolist()
    assert list(daily.data[1].y) == series[0][1]["n_obs"].tolist()


# --- design system ---


def test_the_streamlit_theme_is_generated_from_the_tokens():
    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text())
    expected = theme.streamlit_theme()
    assert config["theme"] == {**expected["theme"], "sidebar": expected["theme.sidebar"]}
    assert config["server"]["address"] == "127.0.0.1"
    assert config["browser"]["gatherUsageStats"] is False


def test_the_contrast_ratio_matches_wcag_reference_values():
    assert theme.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert theme.contrast_ratio("#767676", "#FFFFFF") == pytest.approx(4.54, abs=0.01)
    assert theme.contrast_ratio("#2D5D6C", "#2D5D6C") == pytest.approx(1.0)


SURFACES = ["canvas", "surface", "surface_muted"]


@pytest.mark.parametrize("surface", SURFACES)
def test_text_tokens_meet_wcag_aa_on_every_surface(surface):
    for text in ("text", "text_muted", "text_subtle", "accent"):
        assert theme.contrast_ratio(theme.COLOURS[text], theme.COLOURS[surface]) >= 4.5, text


def test_captions_render_opaque_in_an_aa_colour():
    # Streamlit draws captions at 60% opacity; the CSS layer makes them opaque
    # text_muted, which must pass on every surface and alert background.
    css = theme.css()
    rule = css.split('[data-testid="stCaptionContainer"]')[1].split("}")[0]
    assert "opacity: 1" in rule and "var(--sa-text-muted)" in rule
    backgrounds = [theme.COLOURS[s] for s in SURFACES] + [t["bg"] for t in theme.STATUS.values()]
    for background in backgrounds:
        assert theme.contrast_ratio(theme.COLOURS["text_muted"], background) >= 4.5


def test_keyboard_focus_is_visible():
    css = theme.css()
    assert ":focus-visible" in css and "outline: 2px solid var(--sa-accent)" in css
    assert theme.contrast_ratio(theme.COLOURS["accent"], theme.COLOURS["canvas"]) >= 3.0


def test_status_treatments_are_legible_and_spelled_out():
    for level, tones in theme.STATUS.items():
        assert theme.contrast_ratio(tones["fg"], tones["bg"]) >= 4.5, level
        assert level.split("/")[-1] in components.pill_html(level).upper()
        assert f".sa-pill--{theme.status_slug(level)}" in theme.css()
    assert style.STATUS_COLOURS["VALID"] == theme.STATUS["VALID"]["fg"]
    # Only the run status is a live region; scope tags are static.
    assert 'role="status"' in components.pill_html("VALID", live=True)
    assert "role=" not in components.pill_html("NOTICE", "Training scope")


def test_notices_spell_out_their_severity(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, "page_scripts/research_warnings.py")
    alerts = [*app.error, *app.warning, *app.info]
    assert alerts
    words = {"error": "**Blocking.**", "warning": "**Warning.**", "info": "**Note.**"}
    for kind, word in words.items():
        for alert in getattr(app, kind):
            assert alert.value.startswith(word), alert.value
            assert alert.icon


def test_motion_never_tweens_values_and_respects_reduced_motion():
    css = theme.css()
    reduced = css.split("@media (prefers-reduced-motion: reduce)")[1]
    assert "animation: none !important" in reduced and "transition: none !important" in reduced
    assert "infinite" not in css
    assert "*," not in reduced  # scoped to the dashboard's own motion
    assert max(v for k, v in theme.MOTION.items() if k.endswith("_ms")) <= 300
    layout = go.Figure().update_layout(**style.base_layout("t")).layout
    assert layout.transition.duration is None  # Plotly never animates between values
    keyframes = css.split("@keyframes sa-appear")[1].split("}}")[0]
    assert "opacity" in keyframes and "transform" not in keyframes


def test_the_theme_is_applied_once_per_page(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    styles = [
        element for element in app.get("html") if "<style>" in getattr(element.proto, "body", "")
    ]
    assert len(styles) == 1 and ":root" in styles[0].proto.body


def test_pages_build_from_the_shared_components():
    views = (REPO / "src" / "stock_agent" / "dashboard" / "views.py").read_text()
    # Titles, callouts, cards, status markup and colours come from components
    # and tokens only.
    for banned in (
        "st.title(",
        "unsafe_allow_html",
        "st.error(",
        "st.warning(",
        "st.info(",
        "st.success(",
        ".error(",
        "border=True",
        "st.html(",
        "rgba(",
    ):
        assert banned not in views, banned
    colour_lines = [
        line for line in views.splitlines() if "#" in line and ("color" in line or "colour" in line)
    ]
    assert not colour_lines


def test_every_forecast_model_renders_against_every_comparator(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch, PAGE)
    for model in app.selectbox(key="gw_model").options:
        app.selectbox(key="gw_model").set_value(model).run()
        picker = app.multiselect(key="gw_comparators")
        picker.set_value(picker.options).run()
        assert not app.exception, (model, [e.value for e in app.exception])
        assert not app.error, (model, [e.value for e in app.error])
        assert "gw_primary" in _charts(app)


# --- a universe that grows during the test period ---


def test_date_normalization_differs_from_observation_weighting_and_reproduces_t2(
    varying_universe_root,
):
    root, _ = varying_universe_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    predictions = view.tables["predictions"]
    pairs = 0
    for model in ("memorizer", "per_symbol_mean", "pooled_mean"):
        for comparator in evaluation.comparators(view.record, view.models):
            if comparator.name == model:
                continue
            series = evaluation.squared_error_advantage(
                predictions, model, comparator.name, view.evaluation_asof
            )
            assert series["n_obs"].nunique() > 1  # N_t changes
            result = evaluation.reproduce_t2(
                series, view.tables["metrics"], view.tables["curves"], model, comparator.name
            )
            assert result.ok, (model, comparator.name, result.detail)
            # The two weightings give different curves once N_t varies.
            rescaled = series["cumulative_sse"] / series["n_obs"].mean()
            assert not np.allclose(series["cumulative_d"], rescaled)
            pairs += 1
    assert pairs == 7  # 3 + 2 + 2: a model is never compared with itself


def test_a_retrospective_slice_is_labelled_on_the_chart(varying_universe_root, monkeypatch):
    root, _ = varying_universe_root
    app = _app(root, monkeypatch, PAGE)
    app.selectbox(key="gw_regime").set_value("drawdown_episode_ex_post").run()
    covid = next(v for v in app.selectbox(key="gw_regime_value").options if "COVID" in v)
    app.selectbox(key="gw_regime_value").set_value(covid).run()
    assert not app.exception
    spec = _charts(app)["gw_primary"].proto.spec
    assert "filtered sample (descriptive), retrospective regime" in spec
    table = next(f.value for f in app.dataframe if "mean d_t" in f.value.columns)
    assert (table["dates"] > 0).all()


def test_an_empty_filtered_sample_shows_zero_dates_not_an_error(varying_universe_root, monkeypatch):
    root, _ = varying_universe_root
    app = _app(root, monkeypatch, PAGE)
    _, last = app.slider(key="gw_dates").value
    app.slider(key="gw_dates").set_value((dt.date(2020, 6, 1), last)).run()
    app.selectbox(key="gw_regime").set_value("drawdown_episode_ex_post").run()
    covid = next(v for v in app.selectbox(key="gw_regime_value").options if "COVID" in v)
    app.selectbox(key="gw_regime_value").set_value(covid).run()
    assert not app.exception
    table = next(f.value for f in app.dataframe if "mean d_t" in f.value.columns)
    assert (table["dates"] == 0).all() and table["mean d_t"].isna().all()
    assert any("no defined dates" in c.value for c in app.caption)
