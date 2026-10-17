"""
Stored evaluations and forecasts (stock_agent.stock_lab.store), the evaluate
script (scripts/stock_lab_evaluate.py) and the modelled views
(stock_agent.stock_lab.forecast_view), on synthetic data only.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import forecast_view as fv
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import DataClass, StableBand
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, StoreRefused
from stock_agent.stock_lab.symbols import parse_tickers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_evaluate.py"
spec = importlib.util.spec_from_file_location("stock_lab_evaluate", SCRIPT)
evaluate_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_script)
NOW = pd.Timestamp("2026-10-16T22:00:00Z")


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory):
    store = StockLabStore(tmp_path_factory.mktemp("demo") / DB_NAME)
    out = io.StringIO()
    code = evaluate_script.main(["--demo"], store_factory=lambda: store, clock=lambda: NOW, out=out)
    assert code == 0, out.getvalue()
    return store


def test_the_demo_run_stores_one_evaluation_and_a_forecast_per_demo_stock(demo_store):
    evaluation = demo_store.latest_evaluation(DataClass.SYNTHETIC)
    assert evaluation["data_class"] == "synthetic" and evaluation["run_id"].startswith("S-")
    assert demo_store.latest_evaluation(DataClass.REAL) is None
    for stock_id in synthetic.DEMO_IDS:
        record = demo_store.latest_forecast(stock_id)
        artifact.check(record)
        assert record["data_class"] == "synthetic" and record["as_of_session"] == "2024-12-31"
        assert record["evaluation_run_id"] == evaluation["run_id"]
        assert set(record["signals"]["usual_mix"]) == set(artifact.PROBABILITY_KEYS)


def test_a_second_run_supersedes_and_keeps_every_earlier_forecast(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    for minute in (0, 1):
        evaluate_script.main(
            ["--demo"],
            store_factory=lambda: store,
            clock=lambda m=minute: NOW + pd.Timedelta(minutes=m),
            out=io.StringIO(),
        )
    history = store.forecasts("DEMO_CALM")
    assert len(history) == 2 and history[1]["supersedes"] == history[0]["forecast_id"]


def test_real_stocks_are_refused_without_the_owners_evaluation_decision(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    store.add(parse_tickers("AAPL"))
    out = io.StringIO()
    code = evaluate_script.main(["--real"], store_factory=lambda: store, clock=lambda: NOW, out=out)
    assert code == 2 and "A3" in out.getvalue()
    assert store.latest_evaluation(DataClass.REAL) is None


def test_stored_records_are_write_once_and_checked(demo_store, tmp_path):
    record = copy.deepcopy(demo_store.latest_forecast("DEMO_CALM"))
    with pytest.raises(StoreRefused, match="backdating"):
        demo_store.record_forecast(record)  # the same record again, later: backdated
    again = StockLabStore(demo_store.path, clock=lambda: pd.Timestamp(record["created_at"]))
    with pytest.raises(StoreRefused, match="live forecast"):
        again.record_forecast(record)  # the same id twice: its key already has a live forecast
    real_name = {**record, "symbol": "AAPL"}
    real_name["forecast_id"] = artifact.forecast_id(real_name)
    with pytest.raises(StoreRefused, match="demo identifiers"):
        StockLabStore(tmp_path / DB_NAME).record_forecast(real_name)
    broken = copy.deepcopy(record)
    broken["daily"][0]["p_stable"] += 0.2
    with pytest.raises(artifact.ForecastRefused):
        StockLabStore(tmp_path / DB_NAME).record_forecast(broken)
    connection = sqlite3.connect(demo_store.path)
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        connection.execute("INSERT INTO forecasts SELECT * FROM forecasts LIMIT 1")
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        connection.execute("UPDATE forecasts SET record = '{}'")
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        connection.execute("DELETE FROM evaluations")
    connection.close()


def test_an_evaluation_changed_outside_the_store_is_refused(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    store.record_evaluation(
        {"run_id": "S-1", "created_at": "2026-10-16T22:00:00.000000Z", "data_class": "synthetic"}
    )
    connection = sqlite3.connect(store.path)
    connection.execute("DROP TRIGGER evaluations_no_update")
    connection.execute('UPDATE evaluations SET record = \'{"run_id": "S-1"}\'')
    connection.commit()
    connection.close()
    with pytest.raises(StoreRefused, match="hash"):
        store.latest_evaluation(DataClass.SYNTHETIC)


def test_missing_numbers_are_stored_as_null_never_as_nan(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    store.record_evaluation(
        {
            "run_id": "S-2",
            "created_at": "2026-10-16T22:00:00.000000Z",
            "data_class": "synthetic",
            "x": float("nan"),
        }
    )
    assert store.latest_evaluation(DataClass.SYNTHETIC)["x"] is None


# --- the modelled views ---


def test_cards_show_the_probability_its_track_record_and_the_usual_rate(demo_store):
    record = demo_store.latest_forecast("DEMO_SWING")
    evaluation = demo_store.latest_evaluation(DataClass.SYNTHETIC)
    cards = fv.next_session_cards(record, evaluation)
    assert [card.label for card in cards] == ["Increase", "Stable", "Decrease"]
    for card, key in zip(cards, ("p_increase", "p_stable", "p_decrease"), strict=True):
        assert card.value == fv.pct(record["daily"][0][key])
        assert "usual" in card.caption and "tests:" in card.caption
    track = fv.track_record(
        evaluation, record["estimator"], "target_1", "stable", record["daily"][0]["p_stable"]
    )
    assert track.bin_low <= record["daily"][0]["p_stable"] < track.bin_high and track.n > 0
    assert "in 10" in fv.plain_sentence(record)


def test_the_five_session_frame_has_every_session_and_direction(demo_store):
    record = demo_store.latest_forecast("DEMO_TREND")
    frame = fv.five_session_frame(record, demo_store.latest_evaluation(DataClass.SYNTHETIC))
    assert len(frame) == 15 and set(frame["category"]) == {"increase", "stable", "decrease"}
    totals = frame.groupby("horizon")["probability"].sum()
    assert totals.round(9).eq(1.0).all()
    assert fv.five_session_table(record).index.tolist() == record["target_sessions"]
    assert len(fv.cumulative_cards(record)) == 3


def test_streak_profiles_count_and_refuse_thin_evidence():
    frame = build_features(synthetic.demo_bars("DEMO_TREND"))
    profile = fv.streak_profile(frame, "decrease", StableBand())
    assert profile["length"].tolist() == list(range(1, 8))
    assert profile["events"].is_monotonic_decreasing
    thin = profile[profile["status"] == "insufficient"]
    assert thin["rose"].isna().all()
    sentence = fv.current_streak_sentence(profile, "decrease", 1)
    assert sentence.startswith("After 1 falling session in a row") and "rose" in sentence
    table = fv.streak_table(profile, "decrease")
    assert "Insufficient observations" in set(table["Evidence"]) or thin.empty


def test_trap_views_compare_the_setup_with_every_breakout():
    frame = build_features(synthetic.demo_bars("DEMO_TREND"))
    view = fv.trap_view(frame)
    assert view.setups_resolved >= 3 and view.base_rate != fv.DASH
    assert set(view.by_regime.index) == {"Calm", "Middle", "Volatile"}
    assert view.latest is not None and view.latest["status"]


def test_diagnostics_views_read_the_stored_evaluation(demo_store):
    evaluation = demo_store.latest_evaluation(DataClass.SYNTHETIC)
    assert "B1" in fv.selection_sentence(evaluation)
    scores = fv.scores_table(evaluation)
    assert len(scores) == 5 and "RPS (primary)" in scores.columns
    comparisons = fv.comparison_table(evaluation)
    assert "Holm-adjusted p" in comparisons.columns and len(comparisons) == 4
    frame = fv.reliability_frame(evaluation, [evaluation["selection"]["selected"], "B1"])
    assert fv.calibration_status(frame, "B1", 1) in {
        "Close to calibrated",
        "Over-confident",
        "Under-confident",
        "Too few forecasts",
    }
    assert len(fv.fold_table(evaluation)) == len(evaluation["folds"])
    json.dumps(evaluation)


def test_the_store_refuses_real_forecasts_until_the_owners_decisions(demo_store, tmp_path):
    record = copy.deepcopy(demo_store.latest_forecast("DEMO_CALM"))
    real = {
        **record,
        "symbol": "AAPL",
        "data_class": "real",
        "evidence_status": "experimental_unvalidated",
    }
    real["forecast_id"] = artifact.forecast_id(real)
    store = StockLabStore(tmp_path / DB_NAME)
    with pytest.raises(StoreRefused, match="A2 and A4"):
        store.record_forecast(real)
    with pytest.raises(StoreRefused, match="A2 and A4"):
        store.record_forecast(real, real_authorized="yes")
    assert store.latest_forecast("AAPL") is None


def test_calibration_labels_are_judged_against_the_base_rate():
    rows = []
    # Base rate 0.45; forecasts too extreme (0.25 and 0.65) while outcomes stay near 0.45.
    for low, high, predicted, observed in ((0.2, 0.3, 0.25, 0.42), (0.6, 0.7, 0.65, 0.48)):
        rows.append({"model": "M", "category": 2, "n": 500, "bin_low": low, "bin_high": high,
                     "mean_predicted": predicted, "observed": observed,
                     "ci_low": observed - 0.04, "ci_high": observed + 0.04})  # fmt: skip
    frame = pd.DataFrame(rows)
    assert fv.calibration_status(frame, "M", 2) == "Over-confident"
    timid = frame.assign(mean_predicted=[0.40, 0.50], observed=[0.25, 0.65])
    timid = timid.assign(ci_low=timid["observed"] - 0.04, ci_high=timid["observed"] + 0.04)
    assert fv.calibration_status(timid, "M", 2) == "Under-confident"
    assert fv.calibration_status(frame.assign(n=10), "M", 2) == "Too few forecasts"


def test_the_real_evaluation_with_the_owners_flag_evaluates_through_2024_only(tmp_path):
    store = StockLabStore(tmp_path / DB_NAME)
    store.add(parse_tickers("AAPL MSFT"))
    for ticker, stock_id in (("AAPL", "DEMO_SWING"), ("MSFT", "DEMO_TREND")):
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2025-06-30", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        store.record_prices(ticker, bars, source="fixture", retrieved_at=NOW,
                            data_class=DataClass.REAL, vintage_note="fixture")  # fmt: skip
    out = io.StringIO()
    code = evaluate_script.main(
        ["--real", "--authorized-evaluation"],
        store_factory=lambda: store,
        clock=lambda: NOW,
        out=out,
    )
    assert code == 0, out.getvalue()
    record = store.latest_evaluation(DataClass.REAL)
    assert record["config"]["outcome_end"] == "2024-12-31"
    assert all(fold["test_end"] <= "2025-01-01" for fold in record["folds"])
    assert store.latest_forecast("AAPL") is None and "no forecast recorded" in out.getvalue()
