"""
The forecast artifact's B1 comparator (StockProbabilityForecast schema 1.1):
B1's forecast recorded inside the same write-once record, on the same inputs,
sessions and bands. Schema 1.0 records stay valid as stored. Synthetic data.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.evaluation import EvaluationConfig, build_panel, evaluate
from stock_agent.stock_lab.predict import _probabilities, fit_final, forecast_stock

CREATED = pd.Timestamp("2026-10-16T22:00:00Z")
CONFIG = EvaluationConfig(size_reps=8, sensitivity_bands=())


@pytest.fixture(scope="module")
def made():
    histories = dict(synthetic.demo_histories())
    record = evaluate(histories, CONFIG, data_class="synthetic", run_id="S-c", created_at=CREATED)
    panel = build_panel(histories, CONFIG)
    model = fit_final(panel, CONFIG, record)
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-01", periods=5)]
    forecast = forecast_stock(
        panel, "DEMO_SWING", model, record, target_sessions=sessions, created_at=CREATED,
        data={"source": "synthetic_demo"}, data_class="synthetic",
    )  # fmt: skip
    return panel, model, record, forecast


def _resealed(record: dict) -> dict:
    record["forecast_id"] = artifact.forecast_id(record)
    return record


def test_a_new_forecast_carries_b1_beside_it_on_the_same_sessions(made):
    panel, model, _, forecast = made
    assert forecast["schema_version"] == "1.1" == artifact.SCHEMA_VERSION
    comparator = forecast["comparator"]
    assert comparator["role"] == "benchmark" and comparator["estimator"] == "B1"
    assert comparator["half_life"] == model.half_life
    assert [r["session"] for r in comparator["daily"]] == forecast["target_sessions"]
    row = int(np.flatnonzero((panel["symbol"] == "DEMO_SWING").to_numpy())[-1])
    daily, cumulative, _ = _probabilities(panel, row, model, "B1")
    keys = artifact.PROBABILITY_KEYS
    stored = np.array([[r[k] for k in keys] for r in comparator["daily"]])
    assert np.allclose(stored, daily / daily.sum(axis=1, keepdims=True), rtol=0, atol=1e-15)
    assert np.allclose([comparator["cumulative"][k] for k in keys], cumulative / cumulative.sum())
    if forecast["estimator"] == "B1":
        assert comparator["daily"] == forecast["daily"]


def test_a_challengers_comparator_is_b1_not_the_challenger(made):
    panel, _, record, _ = made
    challenger = copy.deepcopy(record)
    challenger["selection"]["selected"] = "B3"  # fit and forecast B3, whatever was selected
    model = fit_final(panel, CONFIG, challenger)
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-01", periods=5)]
    forecast = forecast_stock(
        panel, "DEMO_SWING", model, challenger, target_sessions=sessions, created_at=CREATED,
        data={"source": "synthetic_demo"}, data_class="synthetic",
    )  # fmt: skip
    assert forecast["estimator"] == "B3" and forecast["comparator"]["estimator"] == "B1"
    row = int(np.flatnonzero((panel["symbol"] == "DEMO_SWING").to_numpy())[-1])
    b1, b1_total, _ = _probabilities(panel, row, model, "B1")
    b3, _, _ = _probabilities(panel, row, model, "B3")
    keys = artifact.PROBABILITY_KEYS
    stored = np.array([[r[k] for k in keys] for r in forecast["comparator"]["daily"]])
    assert np.allclose(stored, b1 / b1.sum(axis=1, keepdims=True), rtol=0, atol=1e-15)
    assert not np.allclose(stored, b3 / b3.sum(axis=1, keepdims=True))
    own = np.array([[r[k] for k in keys] for r in forecast["daily"]])
    assert np.allclose(own, b3 / b3.sum(axis=1, keepdims=True), rtol=0, atol=1e-15)
    total = [forecast["comparator"]["cumulative"][k] for k in keys]
    assert np.allclose(total, b1_total / b1_total.sum(), rtol=0, atol=1e-15)


def test_the_comparator_is_covered_by_the_forecast_id(made):
    forecast = copy.deepcopy(made[3])
    forecast["comparator"]["half_life"] = 21.0
    with pytest.raises(artifact.ForecastRefused, match="forecast_id"):
        artifact.check(forecast)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda c: c.update(role="forecast"),
        lambda c: c.update(estimator="B3"),
        lambda c: c.update(half_life=0),
        lambda c: c.update(half_life=True),
        lambda c: c.pop("half_life"),
        lambda c: c.update(extra=1),
        lambda c: c["daily"].pop(),
        lambda c: c["daily"][0].update(session="2025-01-09"),
        lambda c: c["daily"][2].update(horizon=4),
        lambda c: c["daily"][1].update(p_stable=c["daily"][1]["p_stable"] + 0.1),
        lambda c: c.update(cumulative=None),
        lambda c: c["cumulative"].update(p_increase=-0.1),
    ],
)
def test_a_comparator_that_breaks_the_contract_is_refused(made, tamper):
    forecast = copy.deepcopy(made[3])
    tamper(forecast["comparator"])
    with pytest.raises(artifact.ForecastRefused):
        artifact.check(_resealed(forecast))


def test_a_schema_1_1_forecast_needs_its_comparator_and_a_refusal_has_none(made):
    forecast = copy.deepcopy(made[3])
    missing = _resealed({**forecast, "comparator": None})
    with pytest.raises(artifact.ForecastRefused, match="comparator"):
        artifact.check(missing)
    absent = _resealed({k: v for k, v in forecast.items() if k != "comparator"})
    with pytest.raises(artifact.ForecastRefused, match="fields"):
        artifact.check(absent)  # 1.1 without the field
    refusal = _resealed(
        {**forecast, "daily": [], "cumulative": None, "refusal": {"reason": "x", "detail": "y"}}
    )
    with pytest.raises(artifact.ForecastRefused, match="no probability"):
        artifact.check(refusal)  # a refusal never carries B1's probabilities either
    artifact.check(_resealed({**refusal, "comparator": None}))


def test_a_schema_1_0_record_stays_valid_as_stored(made):
    forecast = copy.deepcopy(made[3])
    old = {k: v for k, v in forecast.items() if k != "comparator"}
    old["schema_version"] = "1.0"
    artifact.check(_resealed(old))
    with pytest.raises(artifact.ForecastRefused, match="fields"):
        artifact.check(_resealed({**old, "comparator": forecast["comparator"]}))
    with pytest.raises(artifact.ForecastRefused, match="schema"):
        artifact.check(_resealed({**forecast, "schema_version": "1.3"}))
    # 1.2 is a reconstruction (owner decision of 2026-10-30): never a 1.1 record relabelled.
    with pytest.raises(artifact.ForecastRefused, match="fields"):
        artifact.check(_resealed({**forecast, "schema_version": "1.2"}))


def test_a_cumulative_mismatch_between_forecast_and_comparator_is_refused(made):
    forecast = copy.deepcopy(made[3])
    no_total = {**forecast, "cumulative": None}
    with pytest.raises(artifact.ForecastRefused, match="same targets|no more and no fewer"):
        artifact.check(_resealed(no_total))
