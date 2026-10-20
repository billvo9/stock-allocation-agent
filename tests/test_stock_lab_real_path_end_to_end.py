"""
ENGINEERING DEMONSTRATION, NOT EVIDENCE OF SKILL.

One offline walk through the Stock Lab's real path, from an owner-authorized
price update to a premarket briefing and a scored forecast. The records are
real-classified FIXTURES: the tickers AMD, MRVL, MU and NVDA carry generated
bars labelled "real". The clock is injected and the calendar is the weekday
fixture calendar. Nothing touches the network, a vendor or a private store,
and the owner's activation set (empty: decision A4) is replaced for this test
only. The frozen model is a B3 challenger, selected on a hand-made evaluation
record that follows the confirmed A7 rule, so that B1 recorded beside it
differs from it; nothing here says B3 is better.

What it demonstrates, in order:
    1. a frozen, registered model fitted on development data through
       2024-12-31;
    2. an eligible price update stored in the inference store after the close;
    3. the evening job: the model loaded and verified, one forecast per stock
       with B1 beside it, a stock without prices skipped, the ledger verified;
    4. the 08:30 briefing shows exactly the recorded probabilities, the latest
       observed close with its retrieval time, and the recorded state;
    5. nothing is scored before the next session is stored; once it is, only
       the first target is scored, and B1 is matched on that same outcome;
    6. refusals: a stale close, absent data, a tampered record, a decision
       after the next open, an inactive model;
    7. an evening that did not run is never backfilled.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    CALENDAR_ID,
    Clock,
    ny,
    open_store,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.stock_lab import evening, frozen_model, registry, synthetic
from stock_agent.stock_lab.briefing import reliability as rel
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords, build_briefing
from stock_agent.stock_lab.briefing.timing import briefing_timing
from stock_agent.stock_lab.evaluation import EvaluationConfig, evaluate
from stock_agent.stock_lab.forecast import PROBABILITY_KEYS
from stock_agent.stock_lab.inference.recent import read_recent
from stock_agent.stock_lab.scores import rps
from stock_agent.stock_lab.store import DB_NAME, StockLabStore
from stock_agent.stock_lab.symbols import parse_tickers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_daily.py"
spec = importlib.util.spec_from_file_location("stock_lab_daily_e2e", SCRIPT)
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)

THURSDAY, FRIDAY, MONDAY, TUESDAY = "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"
PRICED = {"AMD": "DEMO_CALM", "MU": "DEMO_SWING", "NVDA": "DEMO_TREND"}  # MRVL: none stored
DEVELOPMENT = {**PRICED, "MRVL": "DEMO_SWING"}
CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp("2024-12-31", tz="UTC")
)
FROZEN_AT = pd.Timestamp("2026-10-01T22:00:00Z")


@pytest.fixture(scope="module")
def frozen():
    histories = {}
    for ticker, stock_id in DEVELOPMENT.items():
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        histories[ticker] = bars
    run = evaluate(histories, CONFIG, data_class="real", run_id="R-e2e", created_at=FROZEN_AT)
    # A hand-made selection that follows the confirmed A7 rule's own form (a fixture, not a
    # result): B3 beat B1 and no step moved.
    rule = frozen_model.A7_DECISION_RULE
    run["selection"].update(selected="B3", beating_reference=["B3"], steps=[])
    assert run["decision_rule"] == run["selection"]["decision_rule"] == rule
    model = registry.freeze(histories, run, CONFIG, created_at=FROZEN_AT)
    assert model["estimator"] == "B3" and model["state_rule"] == "frozen_long_run"
    return run, model


@pytest.fixture
def lab(tmp_path, frozen, monkeypatch):
    run, model = frozen
    clock = Clock(ny(THURSDAY, "20:35"))
    research = StockLabStore(tmp_path / "research" / DB_NAME, clock=clock)
    research.add(parse_tickers("AMD MRVL MU NVDA"))
    research.record_evaluation(run)
    research.record_frozen_model(model)
    inference, _ = open_store(tmp_path)
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model["model_id"]}))
    return {"research": research, "inference": inference, "model": model, "clock": clock,
            "tmp": tmp_path}  # fmt: skip


def _update(lab, day, retrieved="20:30"):
    """The owner-run recent-window fetch: one batch per priced stock, after the close."""

    clock = Clock(ny(day, retrieved))
    store = type(lab["inference"])(lab["inference"].path, clock=clock)
    for ticker, stock_id in PRICED.items():
        record(store, clock, window_start(day), day, retrieved_at=ny(day, retrieved),
               ticker=ticker, bars=steady_bars(window_start(day), day, stock=stock_id))  # fmt: skip


def _evening(lab, at, *flags, research=None):
    lab["clock"].now = at
    store = research or lab["research"]
    out = io.StringIO()
    mode = flags or ("--record-real",)
    code = daily.main(
        [*mode, "--model", lab["model"]["model_id"][:16],
         "--health-inputs", str(lab["tmp"] / "health"),
         "--ledger-head-backup", str(lab["tmp"] / "kept_head")],
        store_factory=lambda: store, clock=lambda: lab["clock"].now, out=out,
        inference_path=lab["inference"].path, calendar=weekdays, calendar_id=CALENDAR_ID,
    )  # fmt: skip
    return code, out.getvalue()


def _briefing(lab, morning, research_path=None):
    timing = briefing_timing(ny(morning, "08:30").to_pydatetime(), weekdays)
    path = lab["inference"].path

    def recent(symbol, *, retrieved_by):
        return read_recent(symbol, path=path, retrieved_by=retrieved_by)

    return build_briefing(
        ReadOnlyRecords(research_path or lab["research"].path), timing=timing, calendar=weekdays,
        calendar_id=CALENDAR_ID, data_class="real", tickers=["AMD", "MRVL", "MU", "NVDA"],
        recent=recent,
    )  # fmt: skip


def _triple(row) -> tuple[float, float, float]:
    return (row["p_increase"], row["p_stable"], row["p_decrease"])


def test_engineering_demonstration_from_a_price_update_to_a_scored_forecast(lab):
    research = lab["research"]
    # 2. The price update after Thursday's close, then 3. the evening job.
    _update(lab, THURSDAY)
    code, text = _evening(lab, ny(THURSDAY, "20:35"))
    assert code == 0, text
    run = research.latest_job_run()
    assert run["status"] == "partial" and run["skipped"] == ["MRVL"]  # absent data
    assert {r["ticker"]: r["reason"] for r in run["results"]}["MRVL"] == "stale_prices"
    assert research.forecasts("MRVL") == ()
    forecasts = {t: research.forecasts(t)[0] for t in PRICED}
    for forecast in forecasts.values():
        assert forecast["data_class"] == "real" and forecast["estimator"] == "B3"
        assert forecast["model_version"] == lab["model"]["model_id"][:16]
        assert forecast["as_of_session"] == THURSDAY and forecast["target_sessions"][0] == FRIDAY
        assert forecast["comparator"]["estimator"] == "B1"
        assert forecast["comparator"]["daily"] != forecast["daily"]  # B1 beside, not the same
        assert forecast["signals"]["market_state"]["regime"] is not None
    assert research.verify_ledger()[1] is None

    # 4. Friday 08:30: exactly the recorded probabilities, the observed close, the state.
    briefing = _briefing(lab, FRIDAY)
    stocks = {s.symbol: s for s in briefing.stocks}
    assert stocks["MRVL"].status == "unavailable"
    for ticker, forecast in forecasts.items():
        stock = stocks[ticker]
        assert stock.status == "fresh"
        shown = stock.next_session.probabilities
        assert (shown.increase, shown.stable, shown.decrease) == _triple(forecast["daily"][0])
        assert [(s.probabilities.increase, s.probabilities.stable, s.probabilities.decrease)
                for s in stock.five_sessions] == [_triple(r) for r in forecast["daily"]]  # fmt: skip
        total = stock.cumulative
        assert (total.increase, total.stable, total.decrease) == _triple(forecast["cumulative"])
        bars = steady_bars(window_start(), THURSDAY, stock=PRICED[ticker])
        assert stock.close.source == "observed" and stock.close.session == THURSDAY
        assert stock.close.value == pytest.approx(bars["close"].iloc[-1])
        assert stock.close.retrieved_at.startswith("2026-10-16T00:30")  # 20:30 New York
        state = forecast["signals"]["market_state"]
        assert (stock.context.ret_5, stock.context.regime) == (state["ret_5"], state["regime"])
    assert "Insufficient evidence" in briefing.reliability.prospective.comparison_words or (
        "No scored" in briefing.reliability.prospective.comparison_words
    )

    # 5. Friday evening, before the new close is stored: nothing to score, AMD and the others
    #    are stale, so no Friday forecast is invented.
    code, text = _evening(lab, ny(FRIDAY, "20:31"))
    assert code == 0 and "scores: 0 scored" in text
    reasons = {r["ticker"]: r["reason"] for r in research.latest_job_run()["results"]}
    assert reasons == dict.fromkeys(("AMD", "MRVL", "MU", "NVDA"), "stale_prices")
    assert {r["ticker"]: r["last_session"] for r in research.latest_job_run()["freshness"]} == {
        "AMD": THURSDAY, "MRVL": None, "MU": THURSDAY, "NVDA": THURSDAY}  # fmt: skip
    assert all(research.scores(t) == () for t in PRICED)
    assert all(len(research.forecasts(t)) == 1 for t in PRICED)
    #    The update arrives; the same evening's next run scores exactly the first targets.
    _update(lab, FRIDAY, retrieved="20:32")
    code, text = _evening(lab, ny(FRIDAY, "20:40"))
    assert code == 0 and "scores: 3 scored" in text, text
    for ticker, forecast in forecasts.items():
        (score,) = research.scores(ticker)
        assert (score["forecast_id"], score["target"]) == (forecast["forecast_id"], "1")
        assert score["session"] == FRIDAY
        bars = steady_bars(window_start(FRIDAY), FRIDAY, stock=PRICED[ticker])
        realized = bars["close"].iloc[-1] * bars["split_factor"].iloc[-1] / bars["close"].iloc[-2]
        assert score["realized_return"] == pytest.approx(realized - 1.0)
        assert len(research.forecasts(ticker)) == 2  # Friday's forecast, made after its close
    #    B1 matched on the same outcome, from the probabilities recorded before it existed.
    stored = [f for t in PRICED for f in research.forecasts(t)]
    scored = [s for t in PRICED for s in research.scores(t)]
    (b1,) = [o for o in rel.comparator_observations(stored, scored) if o.symbol == "AMD"]
    (own,) = [o for o in rel.observations(stored, scored) if o.symbol == "AMD"]
    assert b1.outcome == own.outcome and b1.as_of_session == own.as_of_session == THURSDAY
    p = np.array([[forecasts["AMD"]["comparator"]["daily"][0][k] for k in
                   ("p_decrease", "p_stable", "p_increase")]])  # fmt: skip
    assert b1.rps == pytest.approx(rps(p, np.array([own.outcome]))[0])
    #    Through the briefing: Monday morning compares the three matched forecasts (too few to
    #    report a difference); Friday morning, rebuilt now, still shows only what existed then.
    monday = _briefing(lab, MONDAY).reliability.prospective
    assert (monday.comparison.n, monday.comparison.dates) == (3, 1)
    assert (
        monday.comparison.status == "insufficient" and monday.comparison.candidate_estimator == "B3"
    )
    again = _briefing(lab, FRIDAY)
    assert again.reliability.prospective.comparison is None
    assert {s.symbol: s.close.session for s in again.stocks if s.close.available} == dict.fromkeys(
        PRICED, THURSDAY)  # fmt: skip

    # 7. Monday's evening job never runs; Tuesday's forecasts Tuesday only.
    _update(lab, MONDAY)
    _update(lab, TUESDAY)
    code, text = _evening(lab, ny(TUESDAY, "20:40"))
    assert code == 0, text
    for ticker in PRICED:
        sessions = [f["as_of_session"] for f in research.forecasts(ticker)]
        assert sessions == [THURSDAY, FRIDAY, TUESDAY] and MONDAY not in sessions
    assert research.verify_ledger()[1] is None


def test_refusals_on_the_real_path(lab, monkeypatch):
    research = lab["research"]
    _update(lab, THURSDAY)
    # An inactive model: nothing is recorded, nothing is scored.
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset())
    code, text = _evening(lab, ny(THURSDAY, "20:35"))
    assert code == 2 and "is not activated" in text and research.latest_job_run() is None
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({lab["model"]["model_id"]}))
    # A decision after the next open: refused, recorded as a failed run, exit 1.
    code, text = _evening(lab, ny(FRIDAY, "09:31"))
    assert code == 1 and "after_next_open" in text
    assert research.latest_job_run()["status"] == "failed"
    assert all(research.forecasts(t) == () for t in PRICED)
    # A recorded forecast changed outside the store: the briefing withholds every
    # probability, and the evening job writes nothing more.
    assert _evening(lab, ny(THURSDAY, "20:36"))[0] == 0
    untouched = {t: research.forecasts(t)[0]["daily"] for t in PRICED}
    copy = lab["tmp"] / "tampered" / DB_NAME
    copy.parent.mkdir(mode=0o700)
    shutil.copy2(research.path, copy)
    connection = sqlite3.connect(copy)
    connection.execute("DROP TRIGGER forecasts_no_update")
    connection.execute(
        "UPDATE forecasts SET record = replace(record, '\"p_stable\":0.', '\"p_stable\":0.9') "
        "WHERE symbol = 'AMD'"
    )
    connection.commit()
    connection.close()
    briefing = _briefing(lab, FRIDAY, research_path=copy)
    assert all(s.next_session is None for s in briefing.stocks)
    assert {s.reason for s in briefing.stocks if s.symbol == "AMD"} <= {
        "record_check_failed", "ledger_unverified"}  # fmt: skip
    _update(lab, FRIDAY)
    tampered = StockLabStore(copy, clock=lab["clock"])
    runs_before = len(tampered.job_runs())
    code, text = _evening(lab, ny(FRIDAY, "20:40"), research=tampered)
    assert code == 1 and "the ledger does not verify" in text
    assert all(tampered.scores(t) == () for t in ("MU", "NVDA"))  # nothing written at all
    assert len(tampered.job_runs()) == runs_before + 1
    failed = tampered.latest_job_run()
    assert failed["status"] == "failed" and failed["reason"].startswith("ledger:")
    # The owner's own store is untouched: it still verifies, with the same probabilities.
    assert research.verify_ledger()[1] is None
    assert {t: research.forecasts(t)[0]["daily"] for t in PRICED} == untouched
    assert set(PROBABILITY_KEYS) <= set(untouched["AMD"][0])
