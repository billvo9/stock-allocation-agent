"""
Figure builders plot the stored T2 values exactly: same numbers, same
session dates, intervals only from stored ci columns (both sides), stored
reference values, unavailable rows as labelled gaps, models grouped by
role (never ranked), and four-name rank-position views with positions
1..4 only.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.dashboard import evaluation, figures, loader

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "dashboard"


@pytest.fixture(scope="module")
def view(dashboard_root):
    root, _ = dashboard_root
    return loader.load_run(loader.list_runs(root)[0].path)


def _days(values):
    return pd.to_datetime(values, utc=True).dt.strftime("%Y-%m-%d").tolist()


def _pooled(view, model, metric, method):
    return figures.pooled_row(view.tables["metrics"], model, metric, method)


def _assert_interval(error, row, column="estimate"):
    assert error.array[0] == pytest.approx(row["ci_high"] - row[column], abs=0)
    assert error.arrayminus[0] == pytest.approx(row[column] - row["ci_low"], abs=0)


def _method_of(trace_name):
    return next(m for m, name in figures.METHOD_NAMES.items() if trace_name.startswith(name))


def test_daily_rank_ic_keeps_stored_values_dates_and_gaps(view):
    curves = view.tables["curves"]
    stored = curves[(curves["model"] == "random_noise") & (curves["curve"] == "rank_ic")]
    stored = stored.sort_values("date")
    figure = figures.rank_ic_series(curves, "random_noise", view.folds, window=5)
    daily, trailing = figure.data[0], figure.data[1]
    assert list(daily.x) == _days(stored["date"])
    expected = [
        v if s == "ok" else None for v, s in zip(stored["value"], stored["status"], strict=True)
    ]
    assert list(daily.y) == expected
    defined = stored.loc[stored["status"] == "ok", "value"]
    fifth = list(stored.index).index(defined.index[4])
    assert trailing.y[fifth] == pytest.approx(defined.iloc[:5].mean())
    assert all(value is None for value in trailing.y[:fifth])  # no mean before 5 defined dates


def test_constant_forecasts_get_a_labelled_gap_not_zeros(view):
    figure = figures.rank_ic_series(view.tables["curves"], "zero", view.folds)
    assert all(value is None for value in figure.data[0].y)
    assert any("constant_prediction" in a.text for a in figure.layout.annotations)


def test_fold_timeline_draws_the_stored_fold_dates(view):
    figure = figures.fold_timeline(view.folds, view.holdout_start)
    training, _, test_window, _ = figure.data[:4]
    assert list(training.x[0::3]) == _days(view.folds["train_first_date"])
    assert list(training.x[1::3]) == _days(view.folds["train_last_date"])
    assert list(test_window.x[0::3]) == _days(view.folds["test_start"])
    assert list(test_window.x[1::3]) == _days(view.folds["test_end_exclusive"])
    lockbox = view.holdout_start.strftime("%Y-%m-%d")
    assert any(shape.x0 == lockbox for shape in figure.layout.shapes)


def test_forest_uses_stored_estimates_and_intervals_and_never_ranks(view):
    metrics = view.tables["metrics"]
    models = view.models[view.models["role"] != "canary_unsafe_reference"]
    figure = figures.rank_ic_forest(
        metrics, models, methods=list(figures.METHOD_NAMES), decision_method="fold_block_t"
    )
    assert figure.layout.scattermode == "group"
    for trace in figure.data:
        method = _method_of(trace.name)
        rows = zip(trace.y, trace.x, trace.error_x.array, trace.error_x.arrayminus, strict=True)
        for label, x, plus, minus in rows:
            row = _pooled(view, label.split(" (")[0], "mean_rank_ic", method)
            assert x == row["estimate"]
            assert plus == pytest.approx(row["ci_high"] - row["estimate"], abs=0)
            assert minus == pytest.approx(row["estimate"] - row["ci_low"], abs=0)
    assert figure.data[0].name == "fold-block t (decision method)"
    assert all("corroborating" in trace.name for trace in figure.data[1:])
    order = list(figure.layout.yaxis.categoryarray)[::-1]
    roles = [label.split("(")[-1].rstrip(")") for label in order]
    assert roles == sorted(roles, key=["null", "control", "canary"].index)
    notes = " ".join(a.text for a in figure.layout.annotations)
    assert "unavailable: constant_prediction" in notes
    assert "canary_unsafe_reference" not in " ".join(order)


def test_method_labels_carry_stored_simulated_sizes(view):
    rates = view.record["inference_calibration"]["rates"]
    label = figures.method_label("newey_west", "fold_block_t", rates)
    assert label.startswith("Newey-West (corroborating; simulated size")
    assert f"{rates['momentum']['newey_west']:.1%}" in label
    assert figures.method_label("fold_block_t", "fold_block_t", rates) == (
        "fold-block t (decision method)"
    )
    bootstrap = figures.method_label(
        "circular_block_bootstrap", "fold_block_t", rates, "percentile_uncalibrated"
    )
    assert "uncalibrated" in bootstrap
    assert figures.method_label("newey_west", "fold_block_t", rates, pd.NA).startswith("Newey")


def test_four_name_rank_positions_stay_positions_one_to_four(view):
    curves = view.tables["curves"]
    figure = figures.rank_position_bars(curves, "per_symbol_mean", "rank_position_realized")
    stored = curves[
        (curves["model"] == "per_symbol_mean") & (curves["curve"] == "rank_position_realized")
    ].sort_values("x")
    assert list(figure.layout.xaxis.categoryarray) == ["#1", "#2", "#3", "#4"]
    bars = figure.data[0]
    assert list(bars.x) == ["#1", "#2", "#3", "#4"]
    assert list(bars.y) == stored["value"].tolist()
    np.testing.assert_array_equal(
        bars.error_y.array, (stored["ci_high"] - stored["value"]).to_numpy()
    )
    np.testing.assert_array_equal(
        bars.error_y.arrayminus, (stored["value"] - stored["ci_low"]).to_numpy()
    )


def test_occupancy_keeps_symbols_positions_and_shares(view):
    curves = view.tables["curves"]
    stored = curves[
        (curves["model"] == "per_symbol_mean") & (curves["curve"] == "rank_position_occupancy")
    ]
    figure = figures.rank_position_occupancy(curves, "per_symbol_mean")
    assert list(figure.layout.xaxis.categoryarray) == [f"#{p} (n_names=4)" for p in range(1, 5)]
    assert sorted(trace.name for trace in figure.data) == sorted(stored["x_label"].unique())
    for trace in figure.data:
        rows = stored[stored["x_label"] == trace.name].sort_values("x")
        assert list(trace.x) == [f"#{int(x)} (n_names=4)" for x in rows["x"]]
        assert list(trace.y) == rows["value"].tolist()


def test_hit_rate_panel_plots_stored_rate_interval_and_benchmarks(view):
    metrics = view.tables["metrics"]
    models = view.models[view.models["name"].isin(["random_noise", "pooled_mean", "zero"])]
    figure = figures.hit_rate_panel(metrics, models, "fold_block_t")
    noise = next(t for t in figure.data if t.name == "random_noise (null)")
    row = _pooled(view, "random_noise", "hit_rate", "fold_block_t")
    assert noise.x[0] == row["estimate"]
    _assert_interval(noise.error_x, row)
    benchmark = _pooled(view, "random_noise", "hit_independence_expected", "point")
    assert any(
        t.hovertext == "independence benchmark" and t.x[0] == benchmark["estimate"]
        for t in figure.data
    )
    pooled = next(t for t in figure.data if t.name == "pooled_mean (null)")
    assert pooled.text[0] == "warning: constant_sign_prediction"
    assert "no_signed_predictions" in " ".join(a.text for a in figure.layout.annotations)
    assert list(figure.layout.yaxis.categoryarray)[::-1][0] == "zero (null)"


def test_oos_r2_panel_never_substitutes_another_method(view):
    metrics = view.tables["metrics"]
    models = view.models[view.models["name"].isin(["pooled_mean", "per_symbol_mean"])]
    figure = figures.oos_r2_panel(metrics, models, "zero", "fold_block_t")
    r2 = _pooled(view, "per_symbol_mean", "oos_r2_vs_zero", "circular_block_bootstrap")
    test = _pooled(view, "per_symbol_mean", "mse_improvement_vs_zero", "fold_block_t")
    traces = {(trace.xaxis, trace.y[0]): trace for trace in figure.data}
    left = traces[("x", "per_symbol_mean (control)")]
    assert left.x[0] == r2["estimate"]
    _assert_interval(left.error_x, r2)
    assert traces[("x2", "per_symbol_mean (control)")].x[0] == test["estimate"]
    missing = figures.oos_r2_panel(
        metrics.assign(inference_method="other"), models, "zero", "fold_block_t"
    )
    assert not missing.data
    assert "not stored" in " ".join(a.text for a in missing.layout.annotations)


def test_stale_canary_fold_and_curve_builders_use_stored_values(view):
    metrics, curves = view.tables["metrics"], view.tables["curves"]
    stale = figures.stale_comparison(metrics, "feature_a", "fold_block_t", "fold_block_t")
    row = _pooled(view, "feature_a", "rank_ic_minus_stale_15", "fold_block_t")
    assert stale.data[0].y[0] == row["estimate"]
    assert stale.data[0].error_y.arrayminus[0] == pytest.approx(
        row["estimate"] - row["ci_low"], abs=0
    )

    canary = figures.canary_panel(
        metrics, "memorizer", "canary_unsafe_reference", "fold_block_t", "fold_block_t"
    )
    for trace, name in zip(canary.data, ("memorizer", "canary_unsafe_reference"), strict=True):
        stored = _pooled(view, name, "mean_rank_ic", "fold_block_t")
        assert trace.y[0] == stored["estimate"]
        _assert_interval(trace.error_y, stored)
    assert canary.data[1].marker.pattern.shape == "/"

    by_fold = figures.ic_by_fold(metrics, "random_noise", "fold_block_t")
    rows = metrics[
        (metrics["model"] == "random_noise")
        & (metrics["metric"] == "mean_rank_ic")
        & metrics["fold_id"].notna()
    ]
    assert list(by_fold.data[0].y) == rows.sort_values("fold_id")["estimate"].tolist()

    comparators = evaluation.comparators(view.record, view.models)
    welch = figures.goyal_welch_observation_weighted(
        curves, "per_symbol_mean", view.folds, comparators
    )
    stored = curves[
        (curves["model"] == "per_symbol_mean")
        & (curves["curve"] == "cumulative_sse_improvement")
        & (curves["group"] == "zero")
    ].sort_values("date")
    trace = next(t for t in welch.data if t.name == "vs zero: null baseline")
    assert list(trace.y) == stored["value"].tolist()
    assert list(trace.x) == _days(stored["date"])

    acf = figures.residual_acf(curves, "per_symbol_mean", "date_mean", 5)
    stored = curves[
        (curves["model"] == "per_symbol_mean")
        & (curves["curve"] == "residual_acf")
        & (curves["x_label"] == "date_mean")
    ].sort_values("x")
    assert list(acf.data[0].y) == [None if pd.isna(v) else v for v in stored["value"]]


def test_null_views_plot_stored_draws_and_label_the_observed_estimate(view):
    draws = view.tables["null_draws"]
    histogram = figures.null_draws_histogram(draws, "per_symbol_mean", 0.1, "pooled mean IC")
    stored = draws[
        (draws["model"] == "per_symbol_mean") & (draws["null_kind"] == "block_permutation")
    ]
    assert sorted(histogram.data[0].x) == sorted(stored["value"])
    assert any("pooled mean IC: 0.100" in a.text for a in histogram.layout.annotations)
    strip = figures.static_ordering_strip(
        draws, {"per_symbol_mean": (0.1, "control")}, "pooled mean IC"
    )
    orderings = draws[draws["null_kind"] == "static_ordering"].sort_values("draw")
    assert list(strip.data[0].x) == orderings["value"].tolist()
    assert strip.data[1].marker.symbol == "square"  # control marker, not colour alone


def test_feature_views_plot_stored_values_with_distinct_markers(view):
    stats = view.tables["feature_stats"]
    figure = figures.feature_statistic_by_fold(
        stats, "missing_rate", "training", title="t", y_title="y"
    )
    stored = stats[(stats["scope"] == "training") & (stats["statistic"] == "missing_rate")]
    for trace in figure.data:
        rows = stored[stored["feature"] == trace.name].sort_values("fold_id")
        assert list(trace.y) == rows["value"].tolist()
    assert len({trace.marker.symbol for trace in figure.data}) == len(figure.data)
    heat = figures.drift_heatmap(stats, "standardized_mean_difference")
    wide = stats[
        (stats["scope"] == "evaluation") & (stats["statistic"] == "standardized_mean_difference")
    ]
    wide = wide.pivot(index="feature", columns="fold_id", values="value").sort_index()
    assert [list(row) for row in heat.data[0].z] == [list(row) for row in wide.to_numpy()]
    assert heat.data[0].zmid == 0  # signed: diverging around zero
    assert heat.layout.title.text.startswith("EVALUATION-ONLY")
    training = figures.feature_quantiles(stats, "feature_a")
    assert training.layout.title.text.startswith("TRAINING-SCOPE")


def test_pooled_deciles_are_labelled_pooled_and_only_forecasts_get_a_45_degree_line(view):
    curves = view.tables["curves"]
    forecast = figures.pooled_deciles(curves, "per_symbol_mean", "forecast")
    score = figures.pooled_deciles(curves, "feature_a", "score")
    constant = figures.pooled_deciles(curves, "pooled_mean", "forecast")
    assert "POOLED" in forecast.layout.title.text
    assert any("45°" in (t.name or "") for t in forecast.data)
    assert not any("45°" in (t.name or "") for t in score.data)
    assert not any("45°" in (t.name or "") for t in constant.data)
    assert "constant_within_date" in constant.layout.title.text
    stored = curves[
        (curves["model"] == "per_symbol_mean") & (curves["curve"] == "pooled_decile_realized")
    ]
    assert list(forecast.data[0].y) == stored.sort_values("x")["value"].tolist()


def test_calibration_line_comes_from_stored_coefficients(view):
    predictions = view.tables["predictions"]
    figure = figures.prediction_vs_realized(predictions, "per_symbol_mean", 0.01, 0.5)
    line = next(t for t in figure.data if (t.name or "").startswith("stored MZ line"))
    rows = predictions[predictions["model"] == "per_symbol_mean"]
    assert list(line.x) == [rows["prediction"].min(), rows["prediction"].max()]
    assert list(line.y) == [
        0.01 + 0.5 * rows["prediction"].min(),
        0.01 + 0.5 * rows["prediction"].max(),
    ]
    assert list(figure.data[0].x) == rows["prediction"].tolist()


def test_uncertainty_forest_uses_stored_rows_and_the_stored_null_value(view):
    metrics = view.tables["metrics"]
    figure = figures.uncertainty_forest(metrics, "random_noise", "hit_rate", "fold_block_t")
    assert list(figure.layout.yaxis.categoryarray)[-1] == "fold-block t (decision method)"
    for trace in figure.data:
        method = _method_of(trace.name)
        row = _pooled(view, "random_noise", "hit_rate", method)
        assert trace.x[0] == row["estimate"]
        assert trace.y[0].startswith(figures.METHOD_NAMES[method])  # short axis label
    null_value = _pooled(view, "random_noise", "hit_rate", "fold_block_t")["null_value"]
    assert any(shape.x0 == null_value for shape in figure.layout.shapes)
    r2 = figures.uncertainty_forest(metrics, "per_symbol_mean", "oos_r2_vs_zero", "fold_block_t")
    assert r2.data[0].name == "circular block bootstrap (percentile interval, uncalibrated)"


def test_false_positive_rates_plot_the_stored_simulation_without_verdict_colours(view):
    rates = view.record["inference_calibration"]["rates"]
    figure = figures.false_positive_rates(rates, 0.065, "fold_block_t")
    names = [t.name.split(" (")[0] for t in figure.data]
    assert names == list(figures.METHOD_NAMES.values())
    primary = figure.data[0]
    assert list(primary.x) == [rates[s]["fold_block_t"] for s in primary.y]
    colours = {t.marker.color.upper() for t in figure.data}
    assert not colours & {"#FF0000", "#00FF00", "#D55E00", "#009E73"}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_pure_modules_do_not_import_streamlit_and_nothing_imports_the_dashboard():
    for name in ("figures", "loader", "status", "style", "glossary"):
        assert not any(m.startswith("streamlit") for m in _imports(PACKAGE / f"{name}.py")), name
    for path in PACKAGE.parent.rglob("*.py"):
        if PACKAGE in path.parents:
            continue
        assert not any(m.startswith("stock_agent.dashboard") for m in _imports(path)), path


def _interval_rows(methods, levels, lags):
    return pd.DataFrame(
        {
            "ci_method": pd.array(methods, dtype="str"),
            "ci_level": pd.array(levels, dtype="float64"),
            "hac_lag": pd.array(lags, dtype="Int64"),
        }
    )


NW = "newey_west_normal_over_dates"


@pytest.mark.parametrize(
    ("methods", "levels", "lags", "label"),
    [
        (
            [NW, NW],
            [0.9, 0.9],
            [7, 7],
            "bars: nominal 90% Newey-West (normal critical values) over dates, lag 7",
        ),
        ([None, None], [np.nan, np.nan], [None, None], "bars: interval method not recorded"),
        ([NW, NW], [0.95, 0.9], [3, 3], "bars: mixed interval methods (see the stored rows)"),
        (
            [NW, None],
            [0.95, np.nan],
            [3, None],
            "bars: mixed interval methods (see the stored rows)",
        ),
        (["other_method"], [np.nan], [None], "bars: other_method, level not recorded"),
    ],
    ids=["recorded", "legacy", "mixed_levels", "partly_recorded", "unknown_method"],
)
def test_interval_labels_come_from_the_stored_rows(methods, levels, lags, label):
    assert figures.interval_label(_interval_rows(methods, levels, lags)) == label


def test_curve_figures_label_intervals_from_the_stored_rows(view):
    curves = view.tables["curves"].copy()
    bars = figures.rank_position_bars(curves, "feature_a", "rank_position_realized")
    assert all("nominal 95%" in trace.name and "lag 3" in trace.name for trace in bars.data)
    curves["ci_level"] = 0.8  # the label follows the data, not a presentation default
    deciles = figures.pooled_deciles(curves, "feature_a", "score")
    assert "nominal 80%" in deciles.data[0].name
