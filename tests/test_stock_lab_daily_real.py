"""
The daily job's real path (scripts/stock_lab_daily.py --dry-run / --record-real
--model ID): a registered frozen model, each watched stock's inference window,
the B1 comparator, scoring from the inference store, the job run and its Model
Health input. Real-classified fixtures on a weekday calendar with an injected
clock; nothing real, no network, and activation only through a test-only
monkeypatch of the owner's (empty) activation set.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    Clock,
    next_day,
    ny,
    open_store,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.stock_lab import evening, registry, synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.evaluation import EvaluationConfig, evaluate
from stock_agent.stock_lab.store import DB_NAME, StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_daily.py"
spec = importlib.util.spec_from_file_location("stock_lab_daily_real", SCRIPT)
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)

SOURCES = {"AMD": "DEMO_CALM", "MU": "DEMO_SWING", "NVDA": "DEMO_TREND"}
CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp("2024-12-31", tz="UTC")
)
FROZEN_AT = pd.Timestamp("2026-10-01T22:00:00Z")


@pytest.fixture(scope="module")
def frozen():
    histories = {}
    for ticker, stock_id in SOURCES.items():
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        histories[ticker] = bars
    run = evaluate(histories, CONFIG, data_class="real", run_id="R-daily", created_at=FROZEN_AT)
    return run, registry.freeze(histories, run, CONFIG, created_at=FROZEN_AT)


@pytest.fixture
def lab(tmp_path, frozen):
    evaluation, model = frozen
    clock = Clock(ny(AS_OF, "20:35"))
    store = StockLabStore(tmp_path / "research" / DB_NAME, clock=clock)
    store.add(parse_tickers(" ".join(SOURCES)))
    store.record_evaluation(evaluation)
    store.record_frozen_model(model)
    inference, inference_clock = open_store(tmp_path)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
               ticker=ticker, bars=steady_bars(window_start(), AS_OF, stock=stock_id))  # fmt: skip
    return store, inference, model, clock, tmp_path


def _main(lab, *flags):
    store, inference, model, clock, tmp_path = lab
    out = io.StringIO()
    code = daily.main(
        [*flags, "--model", model["model_id"][:16], "--health-inputs", str(tmp_path / "health"),
         "--ledger-head-backup", str(tmp_path / "kept_head")],
        store_factory=lambda: store, clock=lambda: clock.now, out=out,
        inference_path=inference.path, calendar=weekdays, calendar_id=CALENDAR_ID,
    )  # fmt: skip
    return code, out.getvalue()


def test_a_dry_run_prints_every_forecast_with_b1_and_writes_nothing(lab):
    store = lab[0]
    before = store.path.read_bytes()
    code, text = _main(lab, "--dry-run")
    assert code == 0, text
    assert "dry run: nothing recorded" in text and text.count(": dry_run; next session") == 3
    assert "B1 Incr." in text and f"expected session {AS_OF}" in text
    assert store.path.read_bytes() == before
    assert store.latest_job_run() is None and not (lab[4] / "health").exists()


def test_recording_is_refused_until_the_owner_activates_the_model(lab):
    code, text = _main(lab, "--record-real")
    assert code == 2 and "is not activated" in text and "decision A4" in text
    assert all(lab[0].forecasts(t) == () for t in SOURCES)


def test_an_activated_model_records_forecasts_scores_and_the_health_input(lab, monkeypatch):
    store, _, model, clock, tmp_path = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    for ticker in SOURCES:
        (forecast,) = store.forecasts(ticker)
        assert forecast["data_class"] == "real" and forecast["as_of_session"] == AS_OF
        assert forecast["comparator"]["estimator"] == "B1"
    run = store.latest_job_run()
    assert run["status"] == "ok" and run["expected_session"] == AS_OF and run["activation"] is None
    assert [r["status"] for r in run["results"]] == ["recorded"] * 3
    assert {r["ticker"]: r["sessions_behind"] for r in run["freshness"]} == dict.fromkeys(
        SOURCES, 0
    )
    (written,) = (tmp_path / "health" / "stock_lab_monitor").glob("*.json")
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["synthetic"] is False and document["ledger"]["verified"] is True
    assert document["outcomes"]["pending"] == 0 and document["job"]["results"] == run["results"]
    assert all(row["psi"] is not None for row in document["inputs_drift"])
    assert document["lineage"]["model_version"] == model["model_id"][:16]
    # The next evening: the new session's batch is stored first, then the job scores and forecasts.
    first = next_day(AS_OF)
    inference_clock = Clock(ny(first, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=inference_clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(first), first,
               retrieved_at=ny(first, "20:30"), ticker=ticker,
               bars=steady_bars(window_start(first), first, stock=stock_id))  # fmt: skip
    clock.now = ny(first, "20:35")
    code, text = _main(lab, "--record-real")
    assert code == 0, text
    assert "scores: 3 scored" in text
    assert [len(store.forecasts(t)) for t in SOURCES] == [2, 2, 2]
    assert {s["target"] for t in SOURCES for s in store.scores(t)} == {"1"}


def test_a_stale_close_is_reported_and_never_invented(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    clock.now = ny(next_day(AS_OF), "20:35")  # no batch for that session was stored
    code, text = _main(lab, "--record-real")
    run = store.latest_job_run()
    assert run["status"] == "partial" and sorted(run["skipped"]) == sorted(SOURCES)
    assert all(store.forecasts(t) == () for t in SOURCES) and "stale_prices" in text
    assert code == 0


def test_an_unknown_or_ambiguous_model_id_is_refused(lab):
    store, inference, _, clock, _ = lab
    out = io.StringIO()
    code = daily.main(
        ["--dry-run", "--model", "f" * 16], store_factory=lambda: store, clock=lambda: clock.now,
        out=out, inference_path=inference.path, calendar=weekdays, calendar_id=CALENDAR_ID,
    )  # fmt: skip
    assert code == 2 and "--model must name one registered real frozen model" in out.getvalue()
    assert store.frozen_models(DataClass.REAL)  # the registered one is still there


def test_a_run_after_the_next_open_records_a_failed_run_and_exits_1(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    clock.now = ny(next_day(AS_OF), "09:31")  # the evening job ran too late: no forecast
    code, text = _main(lab, "--record-real")
    run = store.latest_job_run()
    assert code == 1 and run["status"] == "failed"
    assert sorted(run["failed"]) == sorted(SOURCES) and "after_next_open" in text
    assert all(store.forecasts(t) == () for t in SOURCES)


def test_a_dry_run_after_a_recorded_evening_scores_nothing(lab, monkeypatch):
    store, _, model, clock, _ = lab
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    assert _main(lab, "--record-real")[0] == 0
    first = next_day(AS_OF)
    inference_clock = Clock(ny(first, "20:31"))
    inference = type(lab[1])(lab[1].path, clock=inference_clock)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(first), first,
               retrieved_at=ny(first, "20:30"), ticker=ticker,
               bars=steady_bars(window_start(first), first, stock=stock_id))  # fmt: skip
    clock.now = ny(first, "20:35")
    code, text = _main(lab, "--dry-run")
    assert code == 0 and "scores: 3 scored" in text  # what recording would write ...
    assert all(store.scores(t) == () for t in SOURCES)  # ... and nothing was written
    assert [len(store.forecasts(t)) for t in SOURCES] == [1, 1, 1]


def test_a_model_id_shorter_than_16_characters_is_refused(lab):
    store, inference, model, clock, _ = lab
    out = io.StringIO()
    code = daily.main(
        ["--dry-run", "--model", model["model_id"][:15]], store_factory=lambda: store,
        clock=lambda: clock.now, out=out, inference_path=inference.path, calendar=weekdays,
        calendar_id=CALENDAR_ID,
    )  # fmt: skip
    assert code == 2 and "at least 16 characters" in out.getvalue()
