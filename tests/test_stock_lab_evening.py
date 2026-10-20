"""
The real evening job's frozen-model side (stock_agent.stock_lab.evening):
forecasts from the inference window with the B1 comparator, scoring from the
inference store, and the Model Health fields. Real-classified fixtures on a
weekday calendar and an injected clock; nothing real, no network.
"""

from __future__ import annotations

import numpy as np
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
from stock_agent.stock_lab.evaluation import EvaluationConfig, build_panel, evaluate
from stock_agent.stock_lab.inference.window import load_inference_window
from stock_agent.stock_lab.store import DB_NAME, StockLabStore

TICKERS = ("AMD", "MU", "NVDA")
SOURCES = dict(zip(TICKERS, synthetic.DEMO_IDS, strict=True))
CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp("2024-12-31", tz="UTC")
)
CREATED = pd.Timestamp("2026-10-01T22:00:00Z")  # frozen before the evenings below
EXPECTED = pd.Timestamp(AS_OF, tz="UTC")


@pytest.fixture(scope="module")
def frozen():
    histories = {}
    for ticker, stock_id in SOURCES.items():
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        histories[ticker] = bars
    record_ = evaluate(histories, CONFIG, data_class="real", run_id="R-evening", created_at=CREATED)
    return record_, registry.freeze(histories, record_, CONFIG, created_at=CREATED)


@pytest.fixture
def lab(tmp_path, frozen, monkeypatch):
    """The stores, the frozen model and the clock; the model is activated for these tests only
    (a test-only stand-in for the owner's decision A4, whose set is empty)."""

    evaluation, model_record = frozen
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset({model_record["model_id"]}))
    clock = Clock(ny(AS_OF, "20:35"))
    research = StockLabStore(tmp_path / "research" / DB_NAME, clock=clock)
    research.record_evaluation(evaluation)
    research.record_frozen_model(model_record)
    inference, inference_clock = open_store(tmp_path)
    for ticker, stock_id in SOURCES.items():
        record(inference, inference_clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
               ticker=ticker, bars=steady_bars(window_start(), AS_OF, stock=stock_id))  # fmt: skip
    return research, inference, registry.load(model_record), clock


def _run(lab, *, record_it, now=None, tickers=TICKERS):
    research, inference, model, clock = lab
    if now is not None:
        clock.now = now
    return evening.forecast_watchlist(
        research, model, tickers=tickers, now=clock.now, expected=EXPECTED, sessions=weekdays,
        calendar_id=CALENDAR_ID, path=inference.path, record=record_it,
    )  # fmt: skip


def test_only_the_owner_activated_model_is_activated():
    # Decision A4 (2026-10-20): exactly this frozen model, nothing else.
    activated = "2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7"
    assert evening.ACTIVATED_MODELS == frozenset({activated})
    evening.check_activated(activated)
    for other in ("0" * 64, activated[:-1] + "0", activated[:16]):
        with pytest.raises(evening.NotActivated):
            evening.check_activated(other)


def test_recording_or_writing_scores_is_refused_for_a_model_not_activated(lab, monkeypatch):
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset())
    with pytest.raises(evening.NotActivated):
        _run(lab, record_it=True)
    with pytest.raises(evening.NotActivated):
        _score(lab, now=ny(next_day(AS_OF), "20:40"), expected=next_day(AS_OF))
    assert all(lab[0].forecasts(t) == () for t in TICKERS)
    dry = _run(lab, record_it=False, now=ny(AS_OF, "20:36"))
    assert dry.results[0].status == evening.DRY_RUN  # a dry run needs no activation


def test_a_refused_newest_close_read_is_reported_not_raised(lab):
    found = evening.forecast_watchlist(
        lab[0], lab[2], tickers=TICKERS, now=lab[3].now, expected=EXPECTED, sessions=weekdays,
        calendar_id="other_calendar_v1", path=lab[1].path, record=False,
    )  # fmt: skip
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "calendar_mismatch")}
    assert all(row["last_session"] is None for row in found.freshness)


def test_a_dry_run_computes_every_forecast_and_records_nothing(lab):
    research = lab[0]
    before = research.path.read_bytes()
    found = _run(lab, record_it=False)
    assert [r.status for r in found.results] == [evening.DRY_RUN] * 3
    assert research.path.read_bytes() == before
    assert all(research.forecasts(t) == () for t in TICKERS)


def test_a_recorded_forecast_is_the_frozen_model_on_the_window_with_b1_beside_it(lab):
    research, inference, model, clock = lab
    found = _run(lab, record_it=True)
    assert [r.status for r in found.results] == [evening.RECORDED] * 3
    for ticker in TICKERS:
        (stored,) = research.forecasts(ticker)
        assert stored["data_class"] == "real" and stored["as_of_session"] == AS_OF
        assert stored["target_sessions"] == [next_day(AS_OF, h) for h in range(1, 6)]
        assert stored["model_version"] == model.version and stored["estimator"] == "B1"
        assert stored["created_at"] == clock.now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        window = load_inference_window(inference.path, ticker, AS_OF, retrieved_by=clock.now,
                                       sessions=weekdays, calendar_id=CALENDAR_ID)  # fmt: skip
        panel = build_panel({t: w for t, w in [(ticker, window.bars.drop(columns=["batch_id"]))]},
                            model.final.config)  # fmt: skip
        expected = registry.probabilities(panel, ticker, model, "B1")
        daily = np.array(
            [[r["p_decrease"], r["p_stable"], r["p_increase"]] for r in stored["daily"]]
        )
        assert np.allclose(daily, expected.daily / expected.daily.sum(axis=1, keepdims=True))
        comparator = stored["comparator"]
        assert comparator["role"] == "benchmark" and comparator["estimator"] == "B1"
        assert comparator["daily"] == stored["daily"]  # the frozen estimator is B1 itself
        state = stored["signals"]["market_state"]
        last = panel.iloc[expected.row]
        assert state["ret_5"] == pytest.approx(last["ret_5"]) and state["regime"] is not None
        assert stored["data"]["window"]["sessions"] == 252
        assert stored["data"]["decision_time"] == stored["created_at"]
    assert research.verify_ledger()[1] is None


def test_a_second_run_never_recalculates_and_a_stale_stock_gets_no_forecast(lab, tmp_path):
    _run(lab, record_it=True)
    again = _run(lab, record_it=True, now=ny(AS_OF, "21:00"))
    assert {r.reason for r in again.results} == {"already_recorded"}
    stale = evening.forecast_watchlist(
        lab[0], lab[2], tickers=TICKERS, now=ny(next_day(AS_OF), "20:35"),
        expected=pd.Timestamp(next_day(AS_OF), tz="UTC"), sessions=weekdays,
        calendar_id=CALENDAR_ID, path=lab[1].path, record=True,
    )  # fmt: skip
    assert {(r.status, r.reason) for r in stale.results} == {(evening.STALE, "stale_prices")}
    assert all(row["sessions_behind"] == 1 for row in stale.freshness)


def test_a_decision_after_the_next_open_is_refused(lab):
    found = _run(lab, record_it=True, now=ny(next_day(AS_OF), "09:31"))
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "after_next_open")}
    assert all(lab[0].forecasts(t) == () for t in TICKERS)


def test_a_stock_outside_the_frozen_model_is_skipped(lab):
    found = _run(lab, record_it=False, tickers=("MRVL", "NVDA"))
    assert [(r.ticker, r.status, r.reason) for r in found.results] == [
        ("MRVL", evening.SKIPPED, "not_in_model"),
        ("NVDA", evening.DRY_RUN, None),
    ]


# --- scoring from the inference store ---


def _score(lab, *, now, expected, write=True, calendar_id=CALENDAR_ID):
    research, inference, _, clock = lab
    clock.now = now
    return evening.score_real(
        research, symbols=TICKERS, now=now, expected=pd.Timestamp(expected, tz="UTC"),
        sessions=weekdays, calendar_id=calendar_id, path=inference.path, write=write,
        model_id=lab[2].model_id,
    )  # fmt: skip


def _next_batch(lab, day, retrieved="20:30"):
    _, inference, _, _ = lab
    clock = Clock(ny(day, retrieved))
    store = type(inference)(inference.path, clock=clock)
    for ticker, stock_id in SOURCES.items():
        record(store, clock, window_start(day), day, retrieved_at=ny(day, retrieved),
               ticker=ticker, bars=steady_bars(window_start(day), day, stock=stock_id))  # fmt: skip


def test_a_target_is_scored_only_once_its_session_is_stored(lab):
    research = lab[0]
    _run(lab, record_it=True)
    first = next_day(AS_OF)
    before = _score(lab, now=ny(first, "20:20"), expected=first)
    assert (before.scored, before.unscorable, before.pending) == (0, 0, 18)  # 3 x (5 + total)
    assert before.missing == 3  # the first targets' session has closed, its outcome is not stored
    assert all(research.scores(t) == () for t in TICKERS)
    _next_batch(lab, first)
    after = _score(lab, now=ny(first, "20:40"), expected=first)
    assert (after.scored, after.missing, after.pending) == (3, 0, 15)
    for ticker, stock_id in SOURCES.items():
        (score,) = research.scores(ticker)
        assert score["target"] == "1" and score["session"] == first
        bars = steady_bars(window_start(first), first, stock=stock_id)
        realized = bars["close"].iloc[-1] * bars["split_factor"].iloc[-1] / bars["close"].iloc[-2]
        assert score["realized_return"] == pytest.approx(realized - 1.0)
        assert score["batch_id"].startswith(f"{ticker}-")
    assert research.verify_ledger()[1] is None
    again = _score(lab, now=ny(first, "20:50"), expected=first)
    assert again.scored == 0 and len(research.scores("NVDA")) == 1  # written once


def test_a_dry_run_scores_nothing_and_a_refused_read_is_reported(lab):
    research = lab[0]
    _run(lab, record_it=True)
    first = next_day(AS_OF)
    _next_batch(lab, first)
    dry = _score(lab, now=ny(first, "20:40"), expected=first, write=False)
    assert dry.scored == 3 and all(research.scores(t) == () for t in TICKERS)
    refused = _score(lab, now=ny(first, "20:41"), expected=first, calendar_id="other_v1")
    assert {r["reason"] for r in refused.refused} == {"calendar_mismatch"}
    assert len(refused.refused) == 3 and refused.scored == 0


def test_the_drift_check_compares_the_windows_with_the_frozen_reference(lab):
    found = _run(lab, record_it=False)
    rows = evening.input_drift(found.panel, lab[2])
    assert [r["feature"] for r in rows] == [
        "ewma_vol",
        "ret_20",
        "rel_dollar_volume",
        "band_over_vol",
    ]
    assert all(r["psi"] is not None and r["psi"] >= 0 for r in rows)
    assert evening.input_drift(None, lab[2])[0]["psi"] is None


# --- a frozen challenger: its own probabilities, with B1 beside them ---


def test_a_challengers_forecast_records_b1_as_its_comparator(tmp_path, frozen, monkeypatch):
    from stock_agent.stock_lab import frozen_model

    evaluation, _ = frozen
    rule = "fixture_confirmed_rule"
    monkeypatch.setattr(frozen_model, "CONFIRMED_DECISION_RULES", frozenset({rule}))
    selected = {**evaluation, "decision_rule": rule, "schema_version": "1.3"}
    selected["selection"] = {**evaluation["selection"], "selected": "B3", "decision_rule": rule,
                           "decision_test": "fold_block_t", "beating_reference": ["B3"],
                           "steps": []}  # fmt: skip
    histories = {}
    for ticker, stock_id in SOURCES.items():
        bars = synthetic.demo_bars(stock_id).copy()
        bars["session"] = pd.bdate_range(end="2024-12-31", periods=len(bars), tz="UTC").as_unit(
            "us"
        )
        histories[ticker] = bars
    model = registry.load(registry.freeze(histories, selected, CONFIG, created_at=CREATED))
    clock = Clock(ny(AS_OF, "20:35"))
    research = StockLabStore(tmp_path / "research" / DB_NAME, clock=clock)
    inference, inference_clock = open_store(tmp_path)
    record_bars = steady_bars(window_start(), AS_OF, stock="DEMO_SWING")
    record(inference, inference_clock, window_start(), AS_OF, retrieved_at=ny(AS_OF, "20:30"),
           ticker="MU", bars=record_bars)  # fmt: skip
    found = evening.forecast_watchlist(
        research, model, tickers=("MU",), now=clock.now, expected=EXPECTED, sessions=weekdays,
        calendar_id=CALENDAR_ID, path=inference.path, record=False,
    )  # fmt: skip
    (result,) = found.results
    forecast = result.forecast
    assert forecast["estimator"] == "B3" and forecast["comparator"]["estimator"] == "B1"
    b1 = registry.probabilities(found.panel, "MU", model, "B1")
    stored = np.array([[r[k] for k in ("p_decrease", "p_stable", "p_increase")]
                       for r in forecast["comparator"]["daily"]])  # fmt: skip
    assert np.allclose(stored, b1.daily / b1.daily.sum(axis=1, keepdims=True), rtol=0, atol=1e-15)
    assert forecast["comparator"]["daily"] != forecast["daily"]


def test_one_damaged_stock_never_stops_the_others(lab):
    import sqlite3

    connection = sqlite3.connect(lab[1].path)
    connection.execute("DROP TRIGGER bars_no_update")
    connection.execute(
        "UPDATE bars SET close = close + 1 WHERE batch_id IN "
        "(SELECT batch_id FROM batches WHERE ticker = 'MU')"
    )
    connection.commit()
    connection.close()
    found = _run(lab, record_it=True)
    statuses = {r.ticker: (r.status, r.reason) for r in found.results}
    assert statuses["MU"] == (evening.REFUSED, "integrity")
    assert statuses["AMD"][0] == statuses["NVDA"][0] == evening.RECORDED


def test_missing_outcomes_count_watched_stocks_once_per_session(lab):
    _run(lab, record_it=True)
    second = next_day(AS_OF, 2)
    research, inference, _, clock = lab
    clock.now = ny(second, "20:40")
    found = evening.score_real(
        research, symbols=TICKERS, now=clock.now, expected=pd.Timestamp(second, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, write=False,
        watched=("AMD",),
    )  # fmt: skip
    # Two closed sessions without outcomes for AMD only (MU and NVDA are no longer watched).
    assert found.missing == 2 and found.pending == 18


def test_a_rerun_still_measures_drift_on_the_windows_it_skips(lab):
    _run(lab, record_it=True)
    again = _run(lab, record_it=True, now=ny(AS_OF, "21:00"))
    assert {r.reason for r in again.results} == {"already_recorded"}
    rows = evening.input_drift(again.panel, lab[2])
    assert all(r["psi"] is not None and r["stocks"] == 3 and r["values"] > 0 for r in rows)


def test_a_tampered_record_is_refused_for_its_stock_and_the_others_continue(lab, tmp_path):
    import shutil
    import sqlite3

    _run(lab, record_it=True)
    copy = tmp_path / "tampered" / DB_NAME
    copy.parent.mkdir(mode=0o700)
    shutil.copy2(lab[0].path, copy)
    connection = sqlite3.connect(copy)
    connection.execute("DROP TRIGGER forecasts_no_update")
    connection.execute(
        "UPDATE forecasts SET record = replace(record, '\"p_stable\":0.', '\"p_stable\":0.9') "
        "WHERE symbol = 'MU'"
    )
    connection.commit()
    connection.close()
    tampered = StockLabStore(copy, clock=lab[3])
    second = next_day(AS_OF)
    lab[3].now = ny(second, "20:40")
    _next_batch(lab, second)
    found = evening.forecast_watchlist(
        tampered, lab[2], tickers=TICKERS, now=lab[3].now, expected=pd.Timestamp(second, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=lab[1].path, record=False,
    )  # fmt: skip
    statuses = {r.ticker: (r.status, r.reason) for r in found.results}
    assert statuses["MU"] == (evening.REFUSED, "record_check_failed")
    assert statuses["AMD"][0] == statuses["NVDA"][0] == evening.DRY_RUN
    scored = evening.score_real(
        tampered, symbols=TICKERS, now=lab[3].now, expected=pd.Timestamp(second, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=lab[1].path, write=False,
    )  # fmt: skip
    assert [r["ticker"] for r in scored.refused] == ["MU"] and scored.scored == 2
