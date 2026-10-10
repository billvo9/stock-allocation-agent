"""
Descriptive evaluation series derived from stored T2 rows: the
date-normalized Goyal-Welch advantage, maturity at the evaluation as-of,
identical observations, comparator identity, reproduction of T2's stored
estimates, and descriptive-only filtering.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.dashboard import evaluation as ev
from stock_agent.dashboard import loader

DAY = pd.Timedelta(days=1)


def _rows(model, cells, *, fold=0, end_offset=5):
    """cells: (date, symbol, prediction, label) for one model."""

    return [
        {
            "model": model,
            "role": "null",
            "fold_id": fold,
            "date": pd.Timestamp(date, tz="UTC"),
            "symbol": symbol,
            "prediction": prediction,
            "target_start_date": pd.Timestamp(date, tz="UTC") + DAY,
            "target_end_date": pd.Timestamp(date, tz="UTC") + end_offset * DAY,
            "target_return": label,
        }
        for date, symbol, prediction, label in cells
    ]


def _pair(model_cells, baseline_cells):
    return pd.DataFrame(_rows("f", model_cells) + _rows("b", baseline_cells))


ASOF = pd.Timestamp("2030-01-01", tz="UTC")


def test_the_advantage_is_the_mean_squared_error_difference_per_date():
    # Date 1 (2 names): b errors 0.04, 0.01; f errors 0.01, 0.00 -> d = (0.03 + 0.01) / 2.
    # Date 2 (4 names): every name b error 0.01, f error 0.04 -> d = -0.03.
    model = [
        ("2024-01-02", "A", 0.1, 0.2),
        ("2024-01-02", "B", 0.1, 0.1),
        *[("2024-01-03", s, 0.2, 0.0) for s in "ABCD"],
    ]
    base = [
        ("2024-01-02", "A", 0.0, 0.2),
        ("2024-01-02", "B", 0.0, 0.1),
        *[("2024-01-03", s, 0.1, 0.0) for s in "ABCD"],
    ]
    series = ev.squared_error_advantage(_pair(model, base), "f", "b", ASOF)
    assert series["n_obs"].tolist() == [2, 4]
    np.testing.assert_allclose(series["d"], [0.02, -0.03])
    np.testing.assert_allclose(series["sse_difference"], [0.04, -0.12])
    np.testing.assert_allclose(series["cumulative_d"], [0.02, -0.01])
    np.testing.assert_allclose(series["cumulative_sse"], [0.04, -0.08])
    assert set(series["model"]) == {"f"} and set(series["baseline"]) == {"b"}


def test_a_growing_universe_does_not_steepen_the_primary_curve():
    # Each name gains the same advantage every date while the universe grows
    # from 2 to 8 names: the date-normalized curve rises by the same step each
    # date; the observation-weighted one rises faster only because N grows.
    model, base = [], []
    for day, size in enumerate((2, 4, 8), start=2):
        for symbol in "ABCDEFGH"[:size]:
            model.append((f"2024-01-0{day}", symbol, 0.0, 0.1))
            base.append((f"2024-01-0{day}", symbol, 0.2, 0.1))
    series = ev.squared_error_advantage(_pair(model, base), "f", "b", ASOF)
    np.testing.assert_allclose(series["d"], [0.0, 0.0, 0.0])  # equal errors
    model = [(d, s, 0.1, 0.1) for d, s, _, _ in model]  # model now exact
    series = ev.squared_error_advantage(_pair(model, base), "f", "b", ASOF)
    np.testing.assert_allclose(np.diff(series["cumulative_d"], prepend=0), [0.01, 0.01, 0.01])
    np.testing.assert_allclose(np.diff(series["cumulative_sse"], prepend=0), [0.02, 0.04, 0.08])


def test_comparators_follow_the_record_not_a_fixed_list(view):
    record = copy.deepcopy(view.record)
    record["spec"]["config"]["baselines"] = ["pooled_mean", "unknown", "random_noise", "zero"]
    found = ev.comparators(record, view.models)
    assert [c.name for c in found] == ["pooled_mean", "zero"]
    omitted = dict(ev.omitted_comparators(record, view.models))
    assert omitted == {
        "unknown": "not a registered model of this run",
        "random_noise": "output kind score: no forecast scale",
    }
    record["spec"]["config"]["baselines"] = ["per_symbol_mean"]
    assert [c.kind for c in ev.comparators(record, view.models)] == [ev.SELECTION_CONTROL]


def test_only_identical_observations_are_compared():
    model = [("2024-01-02", "A", 0.1, 0.2), ("2024-01-02", "B", 0.1, 0.1)]
    with pytest.raises(ValueError, match="identical observations"):
        ev.squared_error_advantage(_pair(model, model[:1]), "f", "b", ASOF)
    relabelled = [("2024-01-02", "A", 0.1, 0.2), ("2024-01-02", "B", 0.1, 0.5)]
    with pytest.raises(ValueError, match="different labels"):
        ev.squared_error_advantage(_pair(model, relabelled), "f", "b", ASOF)
    with pytest.raises(ValueError, match="itself"):
        ev.squared_error_advantage(_pair(model, model), "f", "f", ASOF)
    frame = _pair(model, model)
    frame.loc[frame["model"] == "b", "target_end_date"] += DAY  # same label, other window
    with pytest.raises(ValueError, match="different labels"):
        ev.squared_error_advantage(frame, "f", "b", ASOF)
    doubled = pd.concat([_pair(model, model), _pair(model, model).iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        ev.squared_error_advantage(doubled, "f", "b", ASOF)
    split = _pair(model, model)
    split.loc[split["symbol"] == "B", "fold_id"] = 1  # one date in two folds
    with pytest.raises(ValueError, match="more than one fold"):
        ev.squared_error_advantage(split, "f", "b", ASOF)


def test_missing_label_windows_are_pending_not_a_mismatch():
    frame = _pair([("2024-01-02", "A", 0.1, 0.2)], [("2024-01-02", "A", 0.0, 0.2)])
    frame["target_end_date"] = pd.NaT
    mature, pending = ev.aligned_observations(frame, "f", "b", ASOF)
    assert mature.empty and len(pending) == 1


def test_pending_forecasts_never_contribute_and_stay_visible():
    cells = [("2024-01-02", "A", 0.1, 0.2), ("2024-01-03", "A", 0.1, 0.2)]
    frame = _pair(cells, [(d, s, 0.0, y) for d, s, _, y in cells])
    asof = pd.Timestamp("2024-01-07", tz="UTC")  # the 2024-01-03 label ends 2024-01-08
    assert ev.maturity(frame, asof).tolist() == ["mature", "pending"] * 2
    series = ev.squared_error_advantage(frame, "f", "b", asof)
    assert series["date"].dt.day.tolist() == [2]
    mature, pending = ev.aligned_observations(frame, "f", "b", asof)
    assert len(mature) == 1 and pending["date"].dt.day.tolist() == [3]
    assert len(ev.pending_rows(frame, asof)) == 2
    # No as-of means nothing is mature; a missing label is pending whatever its end date.
    assert set(ev.maturity(frame, None)) == {"pending"}
    assert ev.squared_error_advantage(frame, "f", "b", None).empty
    frame.loc[0, "target_return"] = np.nan
    assert ev.maturity(frame, ASOF).iloc[0] == "pending"


def test_the_evaluation_asof_is_the_recorded_last_session():
    record = {"lockbox_evidence": {"max_date": "2024-12-31T00:00:00+00:00"}}
    assert ev.evaluation_asof(record) == pd.Timestamp("2024-12-31", tz="UTC")
    assert ev.evaluation_asof({"lockbox_evidence": {}}) is None
    assert ev.evaluation_asof({}) is None


# --- against the stored fixture run ---


@pytest.fixture(scope="module")
def view(dashboard_root):
    root, _ = dashboard_root
    return loader.load_run(loader.list_runs(root)[0].path)


def test_comparators_keep_their_recorded_identity(view):
    found = ev.comparators(view.record, view.models)
    recorded = view.record["spec"]["config"]["baselines"]
    assert [c.name for c in found] == [n for n in recorded if n in {c.name for c in found}]
    kinds = {c.name: c.kind for c in found}
    assert kinds["zero"] == ev.NULL_BASELINE and kinds["pooled_mean"] == ev.NULL_BASELINE
    assert kinds["per_symbol_mean"] == ev.SELECTION_CONTROL
    notes = {m["name"]: m["note"] for m in view.record["spec"]["models"]}
    assert all(c.note == notes[c.name] for c in found)  # recorded text, nothing added
    assert all(
        view.models.set_index("name").loc[c.name, "output_kind"] == "forecast" for c in found
    )


def test_the_derived_series_reproduces_t2_for_every_forecast_pair(view):
    asof = ev.evaluation_asof(view.record)
    forecasts = view.models[
        (view.models["output_kind"] == "forecast")
        & (view.models["role"] != "canary_unsafe_reference")
    ]["name"]
    pairs = 0
    for model in forecasts:
        for comparator in ev.comparators(view.record, view.models):
            if comparator.name == model:
                continue
            series = ev.squared_error_advantage(
                view.tables["predictions"], model, comparator.name, asof
            )
            result = ev.reproduce_t2(
                series, view.tables["metrics"], view.tables["curves"], model, comparator.name
            )
            assert result.ok, (model, comparator.name, result.detail)
            pairs += 1
    assert pairs >= 6


def test_a_derived_series_that_differs_from_t2_is_reported(view):
    asof = ev.evaluation_asof(view.record)
    series = ev.squared_error_advantage(view.tables["predictions"], "pooled_mean", "zero", asof)
    metrics = view.tables["metrics"].copy()
    target = (
        (metrics["model"] == "pooled_mean")
        & (metrics["metric"] == "mse_improvement_vs_zero")
        & (metrics["inference_method"] == "fold_block_t")
        & metrics["fold_id"].isna()
    )
    metrics.loc[target, "estimate"] += 1e-6
    result = ev.reproduce_t2(series, metrics, view.tables["curves"], "pooled_mean", "zero")
    assert result.status == "differs" and "fold_block_t" in result.detail
    curves = view.tables["curves"].copy()
    curve = (curves["model"] == "pooled_mean") & (curves["group"] == "zero")
    curves.loc[curves[curve].index[-1], "value"] += 1e-6
    result = ev.reproduce_t2(series, view.tables["metrics"], curves, "pooled_mean", "zero")
    assert result.status == "differs" and "cumulative" in result.detail
    result = ev.reproduce_t2(
        series, metrics[~metrics["metric"].str.endswith("_vs_zero")], curves, "pooled_mean", "zero"
    )
    assert result.status == "not_stored"


def test_estimates_t2_could_not_form_are_skipped_not_matched(view):
    asof = ev.evaluation_asof(view.record)
    series = ev.squared_error_advantage(view.tables["predictions"], "pooled_mean", "zero", asof)
    metrics = view.tables["metrics"].copy()
    rows = (
        (metrics["model"] == "pooled_mean")
        & (metrics["metric"] == "mse_improvement_vs_zero")
        & metrics["fold_id"].isna()
    )
    curves = view.tables["curves"]
    nw = rows & (metrics["inference_method"] == "newey_west")
    metrics.loc[nw, "estimate"] = np.nan
    assert ev.reproduce_t2(series, metrics, curves, "pooled_mean", "zero").ok
    metrics.loc[rows & (metrics["inference_method"] == "fold_block_t"), "estimate"] = np.nan
    assert ev.reproduce_t2(series, metrics, curves, "pooled_mean", "zero").status == "not_stored"


def test_the_curve_tolerance_scales_with_the_losses_summed():
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    derived = pd.DataFrame(
        {
            "date": dates,
            "fold_id": [0, 0, 1],
            "d": [0.1, 0.2, 0.3],
            "cumulative_sse": [1.0, 2.0, 3.0],
            "loss_total": [1e5, 1e5, 1e5],  # large losses: rounding grows with them
        }
    )
    metrics = pd.DataFrame(
        {
            "model": "f",
            "metric": "mse_improvement_vs_b",
            "fold_id": pd.array([None, None], dtype="Int64"),
            "inference_method": ["newey_west", "fold_block_t"],
            "estimate": [0.2, (0.15 + 0.3) / 2],
        }
    )

    def curves(offset):
        return pd.DataFrame(
            {
                "model": "f",
                "curve": "cumulative_sse_improvement",
                "group": "b",
                "date": dates,
                "value": derived["cumulative_sse"] + offset,
            }
        )

    assert ev.reproduce_t2(derived, metrics, curves(1e-8), "f", "b").ok  # rounding scale
    assert not ev.reproduce_t2(derived, metrics, curves(1e-3), "f", "b").ok


def test_filters_select_dates_and_never_carry_inference(view):
    asof = ev.evaluation_asof(view.record)
    series = ev.squared_error_advantage(view.tables["predictions"], "pooled_mean", "zero", asof)
    assert ev.SampleFilter().is_full_sample
    folds = tuple(sorted(series["fold_id"].unique())[:2])
    chosen = ev.SampleFilter(folds=folds)
    assert not chosen.is_full_sample
    assert set(series[chosen.mask(series)]["fold_id"]) == set(folds)
    middle = series["date"].iloc[len(series) // 2]
    later = ev.SampleFilter(start=middle)
    assert series[later.mask(series)]["date"].min() == middle
    regimes = pd.DataFrame(
        {"date": series["date"], "r": ["x", "y"] * (len(series) // 2) + ["x"] * (len(series) % 2)}
    )
    by_regime = ev.SampleFilter(regime="r", regime_value="y")
    assert (
        series[by_regime.mask(series, regimes)]["date"].isin(regimes[regimes["r"] == "y"]["date"])
    ).all()
    assert not ev.SampleFilter(regime="missing", regime_value="y").mask(series, regimes).any()
    summary = ev.descriptive_summary(series[chosen.mask(series)])
    # Counts and plain averages only: no standard error, interval or p-value.
    assert list(summary) == ["dates", "observations", "mean_d", "share_dates_model_better", "sum_d"]
    assert math.isnan(ev.descriptive_summary(series.iloc[:0])["mean_d"])


def test_stored_rank_ic_dates_carry_their_maturity(view):
    asof = ev.evaluation_asof(view.record)
    dates = ev.rank_ic_dates(
        view.tables["curves"], view.tables["predictions"], "random_noise", asof
    )
    assert len(dates) and set(dates["maturity"]) == {"mature"}
    early = dates["date"].iloc[len(dates) // 2]
    later = ev.rank_ic_dates(
        view.tables["curves"], view.tables["predictions"], "random_noise", early
    )
    assert (later.loc[later["maturity"] == "mature", "date"] < early).all()
    assert (later["maturity"] == "pending").any()
