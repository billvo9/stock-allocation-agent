"""
Targets, the walk-forward evaluation, the forecast artifact and today's
forecast (stock_agent.stock_lab.targets / evaluation / forecast / predict).
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import StableBand
from stock_agent.stock_lab.evaluation import EvaluationConfig, build_panel, dumps, evaluate
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.predict import fit_final, forecast_stock
from stock_agent.stock_lab.targets import build_targets, cumulative_band

CREATED = pd.Timestamp("2026-10-16T22:00:00Z")


def iid_bars(seed: int, n: int = 1500, sigma: float = 0.015) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    r = rng.standard_t(5, size=n) / np.sqrt(5 / 3) * sigma
    close = 50.0 * np.cumprod(1 + r)
    return pd.DataFrame(
        {
            "session": pd.bdate_range(end="2024-12-31", periods=n, tz="UTC").as_unit("us"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1e6),
            "div_cash": np.zeros(n),
            "split_factor": np.ones(n),
        }
    )


@pytest.fixture(scope="module")
def demo_record():
    return evaluate(
        synthetic.demo_histories(), data_class="synthetic", run_id="demo", created_at=CREATED
    )


# --- targets ---


def test_targets_are_the_next_sessions_categories_and_never_cross_a_gap():
    bars = synthetic.demo_bars("DEMO_CALM").iloc[:60]
    frame = build_features(bars.drop(index=[30]), sessions=bars["session"])
    targets = build_targets(frame, StableBand())
    codes = frame["category_code"]
    assert targets.loc[10, "target_1"] == codes.loc[11]
    assert targets.loc[10, "target_3"] == codes.loc[13]
    # Session 30 has no close; 31's return spans the gap, so it is unavailable.
    assert pd.isna(targets.loc[29, "target_1"]) and pd.isna(targets.loc[29, "target_2"])
    assert pd.isna(targets.loc[27, "target_cum"])  # its five sessions include the gap
    assert targets.loc[31, "target_1"] == codes.loc[32]
    total = frame.loc[15, "level"] / frame.loc[10, "level"] - 1
    assert targets.loc[10, "target_return"] == pytest.approx(total)
    assert pd.isna(targets["target_1"].iloc[-1]) and pd.isna(targets["target_cum"].iloc[-5])


def test_outcomes_after_the_cutoff_are_censored():
    bars = synthetic.demo_bars("DEMO_CALM").iloc[-40:]
    frame = build_features(bars)
    cutoff = frame["session"].iloc[-10]
    targets = build_targets(frame, StableBand(), outcome_end=cutoff)
    assert targets["target_1"].iloc[-10:].isna().all()
    assert targets["target_5"].iloc[-14:].isna().all() and targets["target_5"].notna().iloc[-15]


def test_the_five_session_band_is_scaled_by_the_square_root_of_five():
    assert cumulative_band(StableBand(0.005)).tau == pytest.approx(0.005 * math.sqrt(5))
    assert cumulative_band(StableBand(0.03)).tau == 0.05  # capped at the band's maximum


# --- the evaluation ---


def test_the_evaluation_records_every_fold_choice_and_candidate(demo_record):
    assert demo_record["schema"] == "stock_lab_evaluation"
    assert demo_record["candidate_family"] == ["B0", "B1", "B2", "B3", "M1"]
    assert [fold["test_start"] for fold in demo_record["folds"]] == [
        "2022-01-01",
        "2023-01-01",
        "2024-01-01",
    ]
    for fold in demo_record["folds"]:
        assert fold["half_life"] in (21.0, 63.0, 126.0, 252.0) and len(fold["half_life_inner"]) == 4
        assert set(fold["m1_c"]) == {f"target_{h}" for h in range(1, 6)}
    selection = demo_record["selection"]
    assert selection["selected"] in demo_record["candidate_family"]
    assert "Holm" in selection["rule"]
    json.loads(dumps(demo_record))  # strict JSON, no NaN


def test_every_model_is_scored_on_the_same_rows(demo_record):
    metrics = pd.DataFrame(demo_record["metrics"])
    for _, group in metrics.groupby("target"):
        assert group["n"].nunique() == 1


def test_no_test_row_trains_its_own_fold(demo_record):
    panel = build_panel(synthetic.demo_histories(), EvaluationConfig())
    from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, make_expanding_folds

    windows = [
        (pd.Timestamp(f["test_start"], tz="UTC"), pd.Timestamp(f["test_end"], tz="UTC"))
        for f in demo_record["folds"]
    ]
    for fold in make_expanding_folds(panel, windows, holdout_start=MODEL_HOLDOUT_START):
        train_ends = panel.loc[fold.train_rows, "target_end_date"]
        assert (train_ends < fold.test_start).all()


def test_with_no_predictability_the_baseline_is_kept():
    histories = {f"IID_{i}": iid_bars(seed) for i, seed in enumerate((101, 202, 303))}
    record = evaluate(
        histories,
        EvaluationConfig(include_m1=True),
        data_class="synthetic",
        run_id="null",
        created_at=CREATED,
    )
    assert record["selection"]["selected"] in ("B1", "B3")  # B3 may win only through volatility
    pooled = [r for r in record["comparisons"] if r["target"] == "daily_pooled"]
    direction = [
        r for r in record["comparisons"] if r["model"] == "B2" and r["target"] == "daily_pooled"
    ]
    assert pooled and direction[0]["p_holm"] >= 0.05  # no false streak skill


def test_the_lockbox_cannot_be_evaluated():
    config = EvaluationConfig(last_test_year=2025)
    bars = iid_bars(1, n=2000)
    bars["session"] = pd.bdate_range(end="2025-12-31", periods=2000, tz="UTC").as_unit("us")
    with pytest.raises(ValueError):
        evaluate({"IID_A": bars}, config, data_class="synthetic", run_id="x", created_at=CREATED)


# --- the forecast artifact ---


@pytest.fixture(scope="module")
def demo_forecast(demo_record):
    histories = synthetic.demo_histories()
    config = EvaluationConfig()
    panel = build_panel(histories, config)
    model = fit_final(panel, config, demo_record)
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-01", periods=5)]
    return forecast_stock(
        panel,
        "DEMO_TREND",
        model,
        demo_record,
        target_sessions=sessions,
        created_at=CREATED,
        data={"source": "synthetic", "batch_ids": [], "last_close_session": "2024-12-31"},
        data_class="synthetic",
    )


def test_a_forecast_is_five_coherent_daily_distributions_and_a_separate_total(demo_forecast):
    artifact.check(demo_forecast)
    assert demo_forecast["as_of_session"] == "2024-12-31"
    assert [row["horizon"] for row in demo_forecast["daily"]] == [1, 2, 3, 4, 5]
    for row in demo_forecast["daily"]:
        total = row["p_decrease"] + row["p_stable"] + row["p_increase"]
        assert total == pytest.approx(1.0, abs=1e-12)
    assert demo_forecast["cumulative"] is not None
    assert demo_forecast["evidence_status"] == "synthetic_demo"
    assert "Synthetic demo stock" in demo_forecast["limitations"][0]
    assert demo_forecast["signals"]["current_run"]["length"] >= 1


@pytest.mark.parametrize(
    "tamper",
    [
        lambda r: r["daily"][0].update(p_increase=r["daily"][0]["p_increase"] + 0.1),
        lambda r: r["daily"][0].update(p_stable=-0.01),
        lambda r: r.update(target_sessions=["2024-12-30", *r["target_sessions"][1:]]),
        lambda r: r.update(as_of_session="2025-02-01"),
        lambda r: r.update(estimator="B9"),
        lambda r: r.pop("signals"),
    ],
)
def test_a_tampered_or_incoherent_artifact_is_refused(demo_forecast, tamper):
    record = copy.deepcopy(demo_forecast)
    tamper(record)
    with pytest.raises(artifact.ForecastRefused):
        artifact.check(record)


def test_the_total_may_not_contradict_the_daily_forecasts(demo_forecast):
    kwargs = {
        k: demo_forecast[k]
        for k in artifact.FIELDS
        if k not in {"schema", "schema_version", "forecast_id", "daily", "cumulative"}
    }
    with pytest.raises(artifact.ForecastRefused, match="contradicts"):
        artifact.build(**{**kwargs, "daily": [[0.0, 0.0, 1.0]] * 5, "cumulative": [0.5, 0.5, 0.0]})


def test_a_refusal_carries_no_probability(demo_forecast):
    kwargs = {
        k: demo_forecast[k]
        for k in artifact.FIELDS
        if k not in {"schema", "schema_version", "forecast_id", "daily", "cumulative", "refusal"}
    }
    refused = artifact.build(
        **kwargs, daily=None, cumulative=None, refusal={"reason": "stale_close", "detail": "x"}
    )
    assert refused["daily"] == [] and refused["cumulative"] is None


# --- review additions ---


def test_changing_later_prices_never_changes_an_earlier_fold():
    base = {f"IID_{i}": iid_bars(seed, n=2000) for i, seed in enumerate((101, 202, 303))}
    later = {}
    for key, bars in base.items():
        bars = bars.copy()
        future = (bars["session"] >= pd.Timestamp("2023-02-01", tz="UTC")).to_numpy()
        shock = np.cumprod(1 + np.random.RandomState(9).normal(0, 0.04, future.sum()))
        for column in ("open", "high", "low", "close"):
            bars.loc[future, column] = bars.loc[future, column].to_numpy() * shock
        later[key] = bars
    config = EvaluationConfig(last_test_year=2022)
    first = evaluate(base, config, data_class="synthetic", run_id="a", created_at=CREATED)
    second = evaluate(later, config, data_class="synthetic", run_id="a", created_at=CREATED)
    for key in ("folds", "metrics", "comparisons", "reliability", "selection"):
        assert first[key] == second[key], key


def test_the_five_session_total_is_censored_too():
    bars = synthetic.demo_bars("DEMO_CALM").iloc[-40:]
    frame = build_features(bars)
    targets = build_targets(frame, StableBand(), outcome_end=frame["session"].iloc[-10])
    assert targets["target_cum"].iloc[-14:].isna().all()
    assert targets["target_return"].iloc[-14:].isna().all()
    assert targets["target_return"].notna().iloc[-15]


def test_b3_depends_on_the_horizon_in_the_evaluation(demo_record):
    metrics = pd.DataFrame(demo_record["metrics"])
    b3 = metrics[metrics["model"] == "B3"].set_index("target")["rps"]
    assert b3["target_1"] != b3["target_5"]


def test_the_selection_rule_needs_holm_and_climbs_only_when_beaten():
    from stock_agent.stock_lab.evaluation import _select

    dates = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.RandomState(3)

    def series(mean):
        return pd.Series(mean + rng.normal(0, 0.002, 600), index=dates)

    # B3 beats B1; M1 beats B1 by the same amount, so it does not beat B3: keep B3.
    pooled = {"B3": [series(-0.001)], "M1": [series(-0.001)]}
    beating = [{"model": "M1"}, {"model": "B3"}]
    chosen, steps = _select(beating, pooled)
    assert chosen == "B3" and steps[0]["challenger"] == "M1" and not steps[0]["moved"]
    # M1 clearly better than B3 as well: climb to M1.
    pooled = {"B3": [series(-0.001)], "M1": [series(-0.004)]}
    assert _select(beating, pooled)[0] == "M1"
    assert _select([], pooled) == ("B1", [])


def test_selection_ignores_a_challenger_that_passes_only_before_holm(demo_record):
    pooled = [r for r in demo_record["comparisons"] if r["target"] == "daily_pooled"]
    beaten = set(demo_record["selection"]["beating_reference"])
    for row in pooled:
        assert (row["model"] in beaten) == (row["mean_diff"] < 0 and row["p_holm"] < 0.05)


def test_the_record_keeps_every_inner_choice_path_calibration_and_sensitivity(demo_record):
    for fold in demo_record["folds"]:
        assert set(fold["m1_inner"]) == set(fold["m1_c"])
        for target, rows in fold["m1_inner"].items():
            tried = {row["c"] for row in rows}
            assert tried == {0.01, 0.1, 1.0} or (tried == {0.01} and "note" in rows[0]), target
    events = {row["event"] for row in demo_record["path_calibration"]}
    assert {"all_increase", "all_decrease", "joint_22", "joint_00"} <= events
    joint = [r for r in demo_record["path_calibration"] if r["event"].startswith("joint_")]
    assert sum(r["observed"] for r in joint) == pytest.approx(1.0)
    assert [r["half_life"] for r in demo_record["sensitivity_half_life"]] == [
        21.0,
        63.0,
        126.0,
        252.0,
    ]


def test_the_inner_logistic_split_is_purged_and_preprocessing_uses_training_rows():
    from stock_agent.stock_lab import logistic

    n = 1200
    sessions = pd.Series(pd.bdate_range("2015-01-01", periods=n, tz="UTC"))
    ends = sessions.shift(-5).fillna(sessions.iloc[-1])
    rng = np.random.RandomState(4)
    inputs = pd.DataFrame(rng.normal(size=(n, len(logistic.M1_INPUTS))), columns=logistic.M1_INPUTS)
    b3 = rng.dirichlet([3, 3, 3], size=n)
    y = rng.choice(3, size=n).astype(float)
    fitted = logistic.fit_m1(inputs, b3, ["A"] * n, y, sessions, ends)
    assert {row["c"] for row in fitted.selection} == set(logistic.C_GRID)
    # Preprocessing statistics are those of the rows given (the training rows).
    raw = np.column_stack(
        [inputs.to_numpy(), np.log(b3[:, 0] / b3[:, 1]), np.log(b3[:, 2] / b3[:, 1])]
    )
    np.testing.assert_allclose(fitted.prep.means, raw.mean(axis=0))


def test_tampering_is_refused_by_the_content_checks_not_only_the_id(demo_forecast):
    def rebuilt(change):
        record = copy.deepcopy(demo_forecast)
        change(record)
        record["forecast_id"] = artifact.forecast_id(record)
        return record

    for change, message in (
        (lambda r: r["daily"][0].update(p_increase=r["daily"][0]["p_increase"] + 0.1), "sum"),
        (lambda r: r["daily"][1].update(p_stable=-0.01), r"\[0, 1\]"),
        (lambda r: r.update(as_of_session="2025-02-01"), "follow"),
        (lambda r: r.update(estimator="B9"), "estimator"),
    ):
        with pytest.raises(artifact.ForecastRefused, match=message):
            artifact.check(rebuilt(change))
    only_id = copy.deepcopy(demo_forecast)
    only_id["forecast_id"] = "0" * 64
    with pytest.raises(artifact.ForecastRefused, match="forecast_id"):
        artifact.check(only_id)


def test_the_decrease_side_bound_is_checked_too(demo_forecast):
    kwargs = {
        k: demo_forecast[k]
        for k in artifact.FIELDS
        if k not in {"schema", "schema_version", "forecast_id", "daily", "cumulative"}
    }
    with pytest.raises(artifact.ForecastRefused, match="contradicts"):
        artifact.build(**{**kwargs, "daily": [[1.0, 0.0, 0.0]] * 5, "cumulative": [0.0, 0.5, 0.5]})
