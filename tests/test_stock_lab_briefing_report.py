"""
The premarket briefing assembled from stored records: stored probabilities
shown exactly, never manufactured; stale and missing forecasts named; only
records that existed when it was prepared; the store opened read-only and
left byte-for-byte unchanged. Synthetic fixtures (see the helpers).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, build, next_sessions, ny

from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.briefing import report
from stock_agent.stock_lab.briefing.render import render_text
from stock_agent.stock_lab.briefing.report import (
    FRESH,
    STALE,
    UNAVAILABLE,
    Probabilities,
    ReadOnlyRecords,
    briefing_document,
)
from stock_agent.stock_lab.briefing.timing import briefing_timing
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.scores import jeffreys
from stock_agent.stock_lab.store import StoreRefused

MONDAY = ny("2026-10-19", "08:30")
THURSDAY, FRIDAY = "2026-10-15", "2026-10-16"
SIGNALS = {
    "current_run": {"direction": "increase", "length": 3},
    "after_this_streak": {
        "definition": "3 rising sessions in a row",
        "events": 120,
        "outcomes": {
            "continuation": {"estimate": 0.4, "n_eff": 118.0, "status": "ok"},
            "stabilization": {"estimate": 0.25, "n_eff": 118.0, "status": "ok"},
            "reversal": {"estimate": None, "n_eff": 12.0, "status": "insufficient"},
        },
    },
    "bull_trap": {
        "setup_now": True,
        "setup_session": "2026-10-14",
        "historical_failure_rate": 0.43,
        "base_rate": 0.42,
        "setups_resolved": 23,
    },
}


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def test_a_fresh_forecast_shows_its_stored_probabilities_exactly(lab):
    record = lab.forecast("DEMO_CALM", FRIDAY, p=(0.25, 0.45, 0.30), signals=SIGNALS)
    found = build(lab, MONDAY, tickers=["DEMO_CALM"])
    (stock,) = found.stocks
    assert stock.status == FRESH and found.data_as_of == FRIDAY
    assert stock.next_session.session == "2026-10-19"
    assert stock.next_session.probabilities == Probabilities(0.30, 0.45, 0.25)
    assert [s.session for s in stock.five_sessions] == record["target_sessions"]
    assert stock.cumulative == Probabilities(0.45, 0.2, 0.35)
    assert stock.cumulative_end == record["target_sessions"][-1]
    assert stock.lineage.forecast_id == record["forecast_id"]
    assert stock.streak.words == "3 rising sessions in a row."


def test_without_a_record_nothing_is_shown_and_the_reason_is_given(lab):
    lab.watch("AMD", "MRVL", "MU", "NVDA", "SKHY", "SNDK")
    lab.job(at=ny(FRIDAY, "20:31"), expected=FRIDAY)
    found = build(lab, MONDAY, data_class="real", excluded={"SKHY", "SNDK"})
    assert [s.symbol for s in found.stocks] == ["AMD", "MRVL", "MU", "NVDA"]
    assert found.excluded == ("SKHY", "SNDK")
    for stock in found.stocks:
        assert stock.status == UNAVAILABLE and stock.reason == "no_recorded_forecast"
        assert "decision A4" in stock.words
        assert stock.next_session is None and stock.five_sessions == ()
        assert stock.cumulative is None and stock.track == ()
    assert found.data_as_of is None and found.model_versions == ()
    assert any("decision A4" in warning for warning in found.warnings)
    assert "not activated" in found.latest_update.words
    assert found.reliability.prospective.status == "none"


def test_a_stale_forecast_is_named_but_its_probabilities_are_not_today(lab):
    lab.forecast("DEMO_CALM", THURSDAY)
    found = build(lab, MONDAY, tickers=["DEMO_CALM"])
    (stock,) = found.stocks
    assert stock.status == STALE and stock.sessions_behind == 1
    assert stock.lineage.as_of_session == THURSDAY
    assert stock.next_session is None and stock.five_sessions == () and stock.cumulative is None
    assert "Stale" in stock.words and "none is invented" in stock.words
    assert any("stale" in warning for warning in found.warnings)


def test_a_forecast_recorded_after_the_briefing_was_prepared_is_not_read(lab):
    lab.forecast("DEMO_CALM", THURSDAY)
    lab.forecast("DEMO_CALM", FRIDAY)
    before = build(lab, ny(FRIDAY, "20:30"), tickers=["DEMO_CALM"])  # Friday's not made yet
    assert before.stocks[0].status == STALE
    assert build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks[0].status == FRESH


def test_a_recalculation_replaces_the_forecast_it_supersedes_once_it_exists(lab):
    first = lab.forecast("DEMO_CALM", FRIDAY, p=(0.3, 0.3, 0.4))
    lab.forecast(
        "DEMO_CALM",
        FRIDAY,
        p=(0.2, 0.3, 0.5),
        created_at=ny(FRIDAY, "21:20"),
        supersedes=first["forecast_id"],
    )
    early = build(lab, ny(FRIDAY, "21:10"), tickers=["DEMO_CALM"])
    late = build(lab, MONDAY, tickers=["DEMO_CALM"])
    assert early.stocks[0].lineage.forecast_id == first["forecast_id"]
    assert late.stocks[0].next_session.probabilities.increase == 0.5


def test_a_refused_forecast_is_unavailable_with_its_reason(lab):
    refusal = {"reason": "insufficient_history", "detail": "Too little history for volatility."}
    lab.forecast("DEMO_CALM", FRIDAY, refusal=refusal)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == UNAVAILABLE and stock.reason == "refused"
    assert "Too little history" in stock.words and stock.next_session is None


def test_a_forecast_whose_first_target_is_not_the_market_date_is_withheld(lab):
    # A weekday calendar puts Thanksgiving (26 November 2026) among the targets.
    targets = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-11-26", periods=5)]
    lab.forecast("DEMO_CALM", "2026-11-25", targets=targets)
    (stock,) = build(lab, ny("2026-11-27", "08:30"), tickers=["DEMO_CALM"]).stocks
    assert stock.status == UNAVAILABLE and stock.reason == "target_mismatch"
    assert stock.next_session is None


def test_change_since_the_prior_forecast_compares_identical_definitions_only(lab):
    lab.forecast("DEMO_CALM", THURSDAY, p=(0.3, 0.3, 0.4))
    lab.forecast("DEMO_SWING", THURSDAY, band=0.01, created_at=ny(THURSDAY, "21:06"))
    lab.forecast("DEMO_CALM", FRIDAY, p=(0.2, 0.3, 0.5))
    lab.forecast("DEMO_SWING", FRIDAY, band=0.005, created_at=ny(FRIDAY, "21:06"))
    calm, swing = build(lab, MONDAY, tickers=["DEMO_CALM", "DEMO_SWING"]).stocks
    assert calm.change.comparable and calm.change.prior_as_of == THURSDAY
    assert calm.change.next_session.increase == pytest.approx(0.1)
    assert calm.change.next_session.decrease == pytest.approx(-0.1)
    assert calm.change.cumulative == Probabilities(0.0, 0.0, 0.0)
    assert not swing.change.comparable and swing.change.reason == "band_changed"
    assert swing.change.next_session is None


def test_missing_prices_leave_the_close_unavailable_but_keep_the_recorded_forecast(lab):
    lab.forecast("DEMO_CALM", FRIDAY)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == FRESH and stock.next_session is not None
    assert not stock.close.available and stock.close.reason == "no_stored_prices"
    assert not stock.context.available and not stock.bear_trap.available


def test_synthetic_context_is_computed_from_stored_prices_through_the_as_of_close(lab):
    bars = synthetic.demo_bars("DEMO_TREND")
    lab.prices("DEMO_TREND", bars, at=ny("2024-12-31", "21:00"))
    lab.forecast("DEMO_TREND", "2024-12-31", signals=SIGNALS)
    found = build(lab, ny("2025-01-02", "08:30"), tickers=["DEMO_TREND"])
    (stock,) = found.stocks
    expected = build_features(bars).iloc[-1]
    assert found.closed_days == ("2025-01-01",) and "holiday" in found.gap_note
    assert stock.status == FRESH
    assert stock.close.available and stock.close.value == pytest.approx(bars["close"].iloc[-1])
    assert stock.context.available
    assert stock.context.ret_5 == pytest.approx(expected["ret_5"])
    assert stock.context.vol_20 == pytest.approx(expected["vol_20"])
    assert stock.context.regime in ("calm", "middle", "volatile")
    assert stock.bear_trap.available and stock.bear_trap.source == "stored prices"
    assert stock.bull_trap.available and stock.bull_trap.setup_now
    assert "unresolved" in stock.bull_trap.words


def test_real_data_never_computes_context_from_recent_prices_and_no_trap_flag_before_f(lab):
    # Research-store prices of the protected period are never read for a real stock: no
    # close or context is computed from them (decision A6); see the observed-close tests.
    sessions = pd.bdate_range(end=FRIDAY, periods=30, tz="UTC")
    bars = synthetic.demo_bars("DEMO_CALM").tail(30).assign(session=sessions.as_unit("us"))
    lab.prices(REAL_TEST_TICKER, bars.reset_index(drop=True), at=ny(FRIDAY, "21:00"))
    lab.forecast(REAL_TEST_TICKER, FRIDAY, signals=SIGNALS)
    (stock,) = build(lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER]).stocks
    assert stock.status == FRESH
    assert not stock.close.available and stock.close.reason == "recent_not_read"
    assert not stock.context.available and stock.context.reason == "not_recorded"
    assert "frozen forecast job records" in stock.context.words
    for trap in (stock.bull_trap, stock.bear_trap):
        assert not trap.available and trap.reason == "trap_margin_unconfirmed"
        assert trap.setup_now is None and trap.historical_failure_rate is None
    confirmed = build(
        lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER], trap_margin_confirmed=True
    )
    assert confirmed.stocks[0].bull_trap.available


def test_reversal_statistics_keep_their_counts_and_intervals(lab):
    lab.forecast("DEMO_CALM", FRIDAY, signals=SIGNALS)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    reversal = stock.reversal
    assert reversal.available and reversal.events == 120
    outcomes = {o.outcome: o for o in reversal.outcomes}
    low, high = jeffreys(0.4 * 118, 118, 0.9)
    assert (outcomes["continuation"].low, outcomes["continuation"].high) == (low, high)
    assert outcomes["reversal"].status == "insufficient"
    assert outcomes["reversal"].estimate is None and outcomes["reversal"].low is None


def test_the_header_names_the_weekend_the_schedule_and_the_activation_state(lab):
    lab.job(at=ny(FRIDAY, "20:31"), expected=FRIDAY)
    found = build(lab, MONDAY, data_class="real", tickers=[])
    assert found.state == "premarket"
    assert found.scheduled_at == "2026-10-19T08:30:00-04:00"
    assert found.closed_days == ("2026-10-17", "2026-10-18")
    assert "no information" in found.gap_note and found.gap_note in found.warnings
    assert found.latest_update.activation == "not_authorized"
    assert found.latest_update.last_success_at is None
    assert any("watchlist is empty" in warning for warning in found.warnings)


def test_a_failed_ledger_check_withholds_every_probability(lab, monkeypatch):
    lab.forecast("DEMO_CALM", FRIDAY)
    monkeypatch.setattr(
        ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 does not match its hash")
    )
    found = build(lab, MONDAY, tickers=["DEMO_CALM"])
    (stock,) = found.stocks
    assert stock.status == UNAVAILABLE and stock.reason == "ledger_unverified"
    assert stock.next_session is None
    assert not found.ledger.verified
    assert any("ledger check failed" in warning for warning in found.warnings)


def test_synthetic_records_never_appear_in_a_real_briefing(lab):
    record = lab.forecast("DEMO_CALM", FRIDAY)
    lab.score(record, "1", 2, at=ny("2026-10-19", "20:10"))
    found = build(lab, ny("2026-10-20", "08:30"), data_class="real", tickers=["DEMO_CALM"])
    assert found.stocks[0].status == UNAVAILABLE
    assert found.reliability.prospective.scored_targets == 0


def _scored_history(lab, symbols, days, *, estimator="B1", run_id=None, **forecast):
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=FRIDAY, periods=days)]
    previous = {}
    for index, session in enumerate(sessions):
        for symbol in symbols:
            if symbol in previous:
                outcome = (index + len(symbol)) % 3
                lab.score(previous[symbol], "1", outcome, at=ny(session, "20:05"))
        for symbol in symbols:
            previous[symbol] = lab.forecast(
                symbol,
                session,
                targets=[d.strftime("%Y-%m-%d") for d in pd.bdate_range(session, periods=6)[1:]],
                p=(0.2, 0.15, 0.65),
                estimator=estimator,
                evaluation_run_id=run_id,
                **forecast,
            )


def _evaluation(run_id, selected):
    return {
        "run_id": run_id,
        "created_at": "2026-01-02T21:00:00.000000Z",
        "data_class": "synthetic",
        "stocks": ["DEMO_CALM", "DEMO_SWING", "DEMO_TREND"],
        "config": {"outcome_end": "2026-01-02"},
        "metrics": [
            {
                "model": m,
                "target": "target_1",
                "n": 900,
                "dates": 300,
                "rps": r,
                "brier": 0.6,
                "log_loss": 1.0,
            }
            for m, r in (("B1", 0.207), ("B3", 0.206))
        ],
        "reliability": [
            {
                "model": selected,
                "target": "target_1",
                "category": 2,
                "bin_low": 0.6,
                "bin_high": 0.7,
                "n": 300,
                "dates": 200,
                "mean_predicted": 0.64,
                "observed": 0.6,
                "ci_low": 0.54,
                "ci_high": 0.66,
            },
        ],
        "comparisons": [],
        "selection": {"selected": selected, "rule": "keep B1 unless ..."},
    }


def test_reliability_keeps_development_and_prospective_evidence_apart(lab):
    lab.store.record_evaluation(_evaluation("S-dev", "B1"))
    symbols = ["DEMO_CALM", "DEMO_SWING", "DEMO_TREND"]
    _scored_history(lab, symbols, 12, run_id="S-dev")
    found = build(lab, MONDAY, tickers=symbols)
    section = found.reliability
    assert section.development.run_id == "S-dev" and section.development.data_class == "synthetic"
    assert "synthetic" in section.development_words
    prospective = section.prospective
    assert prospective.status == "insufficient" and prospective.scored_targets == 33
    assert {(g.target, g.category) for g in prospective.groups} == {
        ("1", "increase"),
        ("1", "stable"),
        ("1", "decrease"),
    }
    increase = next(g for g in prospective.groups if g.category == "increase")
    (bucket,) = increase.buckets
    assert (bucket.n, bucket.dates) == (33, 11) and bucket.status == "insufficient"
    assert bucket.observed is None
    (rolling,) = prospective.rolling
    assert rolling.rolling.status == "insufficient" and rolling.rolling.means is None
    assert prospective.comparison is None and "B1's own" in prospective.comparison_words
    assert "Synthetic replay" in prospective.words
    assert any(
        "since decision A7 (2026-10-19), by the fold-block t with Holm" in note
        and "reported, never a decision" in note
        for note in section.notes
    )
    (stock_track,) = [t for t in found.stocks[0].track if t.category == "increase"]
    assert (stock_track.bin_low, stock_track.observed, stock_track.dates) == (0.6, 0.6, 200)


def test_a_challenger_without_b1_records_beside_it_cannot_be_compared(lab):
    lab.store.record_evaluation(_evaluation("S-dev3", "B3"))
    _scored_history(lab, ["DEMO_CALM"], 4, estimator="B3", run_id="S-dev3", legacy=True)
    prospective = build(lab, MONDAY, tickers=["DEMO_CALM"]).reliability.prospective
    assert prospective.comparison is None
    assert "not recorded beside" in prospective.comparison_words


def _bytes(path):
    """The database file's hash, and its write-ahead log's content (SQLite's companion)."""

    wal = path.with_name(path.name + "-wal")
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        wal.read_bytes() if wal.exists() else b"",
    )


def test_the_briefing_opens_the_store_read_only_and_leaves_it_unchanged(lab, monkeypatch):
    lab.watch("DEMO_CALM")
    record = lab.forecast("DEMO_CALM", THURSDAY, signals=SIGNALS)
    lab.score(record, "1", 1, at=ny(FRIDAY, "20:05"))
    lab.job(at=ny(FRIDAY, "20:31"), expected=FRIDAY, status="ok", activation=None,
            data_class="synthetic")  # fmt: skip
    lab.forecast("DEMO_CALM", FRIDAY, signals=SIGNALS)
    before = _bytes(lab.path)
    ledger_before = lab.store.verify_ledger()
    opened = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(report.StockLabStore, "_write", None)  # any write would fail
    monkeypatch.setattr(sqlite3, "connect", spy)
    found = build(lab, MONDAY)
    json.dumps(briefing_document(found), allow_nan=False)
    assert opened and all(target.endswith("?mode=ro") for target in opened)
    # A read-only open may create SQLite's empty -wal/-shm companions; the database
    # file and any logged content stay byte-for-byte the same.
    assert _bytes(lab.path) == before
    monkeypatch.undo()
    assert lab.store.verify_ledger() == ledger_before


def test_the_read_only_view_offers_no_way_to_write():
    writers = [
        name for name in dir(ReadOnlyRecords) if name.startswith(("record", "add", "remove"))
    ]
    assert writers == []
    assert not hasattr(ReadOnlyRecords, "_write")


def test_the_document_is_strict_json(lab):
    lab.forecast("DEMO_CALM", FRIDAY, signals=SIGNALS)
    document = briefing_document(build(lab, MONDAY, tickers=["DEMO_CALM"]))
    text = json.dumps(document, allow_nan=False, sort_keys=True)
    assert json.loads(text)["schema"] == report.SCHEMA
    assert document["stocks"][0]["next_session"]["probabilities"]["increase"] == 0.4


def test_an_old_refusal_counts_as_stale(lab):
    refusal = {"reason": "insufficient_history", "detail": "Too little history."}
    lab.forecast("DEMO_CALM", THURSDAY, refusal=refusal)
    found = build(lab, MONDAY, tickers=["DEMO_CALM"])
    assert found.stocks[0].status == STALE and found.freshness["stale"] == 1


def test_a_stale_stock_labels_its_notes_with_their_own_close(lab):
    lab.forecast("DEMO_CALM", THURSDAY, signals=SIGNALS)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == STALE and stock.described_as_of == THURSDAY
    text = render_text(build(lab, MONDAY, tickers=["DEMO_CALM"]))
    assert f"describe the close of {THURSDAY}" in " ".join(text.split())


# --- only what existed when the briefing was prepared (prefix stability) ---


def _bars(last: str, periods: int = 300) -> pd.DataFrame:
    sessions = pd.bdate_range(end=last, periods=periods, tz="UTC").as_unit("us")
    bars = synthetic.demo_bars("DEMO_CALM").tail(periods).reset_index(drop=True)
    return bars.assign(session=sessions)


def test_later_records_never_change_a_briefing_prepared_earlier(lab):
    lab.store.record_evaluation(_evaluation("S-early", "B1"))
    lab.prices("DEMO_CALM", _bars(THURSDAY), at=ny(THURSDAY, "21:00"))
    first = lab.forecast("DEMO_CALM", THURSDAY, signals=SIGNALS, evaluation_run_id="S-early")
    lab.job(at=ny(THURSDAY, "21:30"), expected=THURSDAY, status="ok", activation=None,
            data_class="synthetic")  # fmt: skip
    prepared = ny(FRIDAY, "08:30")

    def document():
        found = briefing_document(build(lab, prepared, tickers=["DEMO_CALM"]))
        found.pop("ledger")  # a check of the file as it is now, by design not time-filtered
        return found

    before = document()
    assert before["stocks"][0]["status"] == FRESH
    # After the briefing was prepared: a recalculation, a score, a job run, a newer
    # evaluation and a later price vintage that revises Thursday's close.
    lab.forecast("DEMO_CALM", THURSDAY, p=(0.1, 0.1, 0.8), created_at=ny(FRIDAY, "09:00"),
                 supersedes=first["forecast_id"], signals=SIGNALS,
                 evaluation_run_id="S-early")  # fmt: skip
    lab.score(first, "1", 2, at=ny(FRIDAY, "20:05"))
    lab.job(at=ny(FRIDAY, "20:31"), expected=FRIDAY, status="ok", activation=None,
            data_class="synthetic")  # fmt: skip
    lab.store.record_evaluation(
        {**_evaluation("S-late", "B1"), "created_at": "2026-10-17T02:00:00.000000Z"}
    )
    revised = _bars(FRIDAY, 301)
    revised.loc[revised.index[-2], "close"] *= 1.01
    lab.prices("DEMO_CALM", revised, at=ny(FRIDAY, "21:00"))
    assert document() == before


def test_a_forecast_counts_only_once_it_was_both_created_and_recorded(lab):
    # Created 4 minutes after the store recorded it, then recorded 4 minutes after it was
    # created (both inside the store's 5-minute clock tolerance).
    lab.forecast(
        "DEMO_CALM", FRIDAY, created_at=ny(FRIDAY, "21:09"), recorded_at=ny(FRIDAY, "21:05")
    )
    lab.forecast(
        "DEMO_SWING", FRIDAY, created_at=ny(FRIDAY, "21:10"), recorded_at=ny(FRIDAY, "21:14")
    )
    assert build(lab, ny(FRIDAY, "21:07"), tickers=["DEMO_CALM"]).stocks[0].status == UNAVAILABLE
    assert build(lab, ny(FRIDAY, "21:12"), tickers=["DEMO_SWING"]).stocks[0].status == UNAVAILABLE
    later = build(lab, ny(FRIDAY, "21:15"), tickers=["DEMO_CALM", "DEMO_SWING"])
    assert [s.status for s in later.stocks] == [FRESH, FRESH]


def test_each_session_shows_its_own_stored_horizon(lab):
    rows = [(0.1, 0.2, 0.7), (0.2, 0.2, 0.6), (0.3, 0.2, 0.5), (0.4, 0.2, 0.4), (0.5, 0.2, 0.3)]
    record = lab.forecast("DEMO_CALM", FRIDAY, p=rows)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.next_session.probabilities == Probabilities(0.7, 0.2, 0.1)
    for index, session in enumerate(stock.five_sessions):
        decrease, stable, increase = rows[index]
        assert (session.horizon, session.session) == (index + 1, record["target_sessions"][index])
        assert session.probabilities == Probabilities(increase, stable, decrease)


# --- prices ---


def test_a_stale_stock_describes_its_own_close_even_when_later_prices_exist(lab):
    lab.forecast("DEMO_CALM", THURSDAY)
    bars = _bars(FRIDAY)
    lab.prices("DEMO_CALM", bars, at=ny(FRIDAY, "21:00"))
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == STALE and stock.close.session == THURSDAY
    assert stock.close.value == pytest.approx(bars["close"].iloc[-2])
    expected = build_features(bars.iloc[:-1]).iloc[-1]
    assert stock.context.ret_5 == pytest.approx(expected["ret_5"])


def test_a_missing_close_for_the_session_is_named_not_filled(lab):
    lab.prices("DEMO_CALM", _bars(THURSDAY), at=ny(THURSDAY, "21:00"))
    lab.forecast("DEMO_CALM", FRIDAY)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == FRESH and stock.close.reason == "close_not_stored"
    assert THURSDAY in stock.close.words and stock.close.value is None
    assert not stock.context.available


def test_prices_that_fail_their_check_are_reported(lab, monkeypatch):
    lab.forecast("DEMO_CALM", FRIDAY)

    def refuse(self, symbol, *, as_of=None):
        raise StoreRefused("batch b does not match its content hash")

    monkeypatch.setattr(ReadOnlyRecords, "bars", refuse)
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == FRESH and stock.close.reason == "prices_refused"


# --- integrity ---


def test_a_tampered_forecast_is_withheld(lab):
    record = lab.forecast("DEMO_CALM", FRIDAY)
    tampered = json.loads(json.dumps(record))
    tampered["daily"][0]["p_increase"], tampered["daily"][0]["p_decrease"] = 0.3, 0.4
    connection = sqlite3.connect(lab.path)
    connection.execute("DROP TRIGGER forecasts_no_update")
    connection.execute(
        "UPDATE forecasts SET record = ? WHERE forecast_id = ?",
        (json.dumps(tampered, sort_keys=True), record["forecast_id"]),
    )
    connection.commit()
    connection.close()
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == UNAVAILABLE and stock.reason == "record_check_failed"
    assert stock.next_session is None


def test_a_forecast_without_a_ledger_entry_is_never_read(lab):
    stored = lab.forecast("DEMO_SWING", FRIDAY)
    orphan = {**stored, "symbol": "DEMO_CALM", "supersedes": None}
    orphan["forecast_id"] = artifact.forecast_id(orphan)
    connection = sqlite3.connect(lab.path)
    connection.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (orphan["forecast_id"], "DEMO_CALM", FRIDAY, orphan["created_at"], "synthetic", None,
         artifact.canonical(orphan)),
    )  # fmt: skip
    connection.commit()
    connection.close()
    found = build(lab, MONDAY, tickers=["DEMO_CALM", "DEMO_SWING"])
    calm, swing = found.stocks
    assert calm.status == UNAVAILABLE and calm.next_session is None and calm.lineage is None
    assert swing.reason == "ledger_unverified" and not found.ledger.verified


# --- evaluation and calendar ---


def test_the_newest_evaluation_known_at_preparation_is_used(lab):
    lab.store.record_evaluation(_evaluation("S-old", "B1"))
    lab.store.record_evaluation(
        {**_evaluation("S-new", "B1"), "created_at": "2026-10-17T02:00:00.000000Z"}
    )
    lab.forecast("DEMO_CALM", FRIDAY)  # names no evaluation: the newest known one is read
    early = build(lab, ny(FRIDAY, "21:30"), tickers=["DEMO_CALM"])
    late = build(lab, MONDAY, tickers=["DEMO_CALM"])
    assert early.reliability.development.run_id == "S-old"
    assert late.reliability.development.run_id == "S-new"


def test_a_forecast_as_of_a_day_that_is_not_the_prior_session_is_withheld(lab):
    saturday = "2026-10-17"
    lab.forecast("DEMO_CALM", saturday, targets=next_sessions(FRIDAY))
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert stock.status == UNAVAILABLE and stock.reason == "as_of_not_prior_session"
    assert stock.next_session is None


def test_a_missing_store_reads_as_empty_and_is_not_created(tmp_path):
    path = tmp_path / "absent" / "stock_lab.sqlite3"
    records = ReadOnlyRecords(path)
    plan = briefing_timing(MONDAY.to_pydatetime(), nyse_sessions)
    found = report.build_briefing(
        records, timing=plan, calendar=nyse_sessions, calendar_id="x", data_class="real"
    )
    assert found.stocks == () and not path.exists() and not path.parent.exists()


def test_an_evaluation_recorded_after_the_briefing_was_prepared_is_not_read(lab):
    lab.forecast("DEMO_CALM", FRIDAY, evaluation_run_id="S-future")
    lab.store.record_evaluation(
        {**_evaluation("S-future", "B1"), "created_at": "2026-10-17T02:00:00.000000Z"}
    )
    early = build(lab, ny(FRIDAY, "21:30"), tickers=["DEMO_CALM"])
    assert early.reliability.development is None and early.stocks[0].track == ()
    assert build(lab, MONDAY, tickers=["DEMO_CALM"]).reliability.development.run_id == "S-future"


# --- the real briefing: observed close, recorded state, B1 beside each forecast ---

MARKET_STATE = {
    "ret_5": 0.031,
    "ret_20": -0.044,
    "vol_20": 0.027,
    "ewma_vol": 0.025,
    "regime": "volatile",
    "basis": "the inference window the forecast read, through its as-of close",
}


class Observed:
    """A stand-in for the descriptive reader: what it was asked, and one fixed answer."""

    def __init__(self, state="observed", session=FRIDAY, close=131.25):
        self.calls = []
        self.state, self.session, self.close = state, session, close

    def __call__(self, symbol, *, retrieved_by):
        self.calls.append((symbol, retrieved_by))
        observed = self.state == "observed"
        rows = (
            [type("Row", (), {"session": self.session, "close": self.close})()] if observed else []
        )
        return type("Recent", (), {
            "state": self.state, "reason": None if observed else "x",
            "last_session": self.session if observed else None,
            "last_close": self.close if observed else None, "sessions": tuple(rows),
            "retrieved_at": "2026-10-17T00:30:00.000000Z",
        })()  # fmt: skip


def test_a_real_forecast_shows_the_observed_close_and_the_recorded_state(lab):
    lab.forecast(REAL_TEST_TICKER, FRIDAY, signals={**SIGNALS, "market_state": MARKET_STATE})
    reader = Observed()
    briefing = build(lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER], recent=reader)
    (stock,) = briefing.stocks
    assert reader.calls == [(REAL_TEST_TICKER, pd.Timestamp(briefing.generated_at))]
    assert stock.close.available and stock.close.source == "observed"
    assert (stock.close.session, stock.close.value) == (FRIDAY, 131.25)
    assert stock.close.retrieved_at == "2026-10-17T00:30:00.000000Z"
    assert "not research evidence" in stock.close.words
    context = stock.context
    assert context.available and context.regime_basis == "window" and context.regime == "volatile"
    assert (context.ret_5, context.ret_20, context.vol_20) == (0.031, -0.044, 0.027)
    text = render_text(briefing)
    flat = " ".join(text.split())  # the renderer wraps lines
    assert "131.25 at the close of" in flat and "own inference window" in flat
    assert "Recorded with the forecast" in flat and "Not comparable with the development" in flat
    assert any("overnight news or premarket move" in line for line in briefing.limitations)


@pytest.mark.parametrize(("state", "words"), [("no_store", "has not stored"),
                                              ("no_data", "No recent prices")])  # fmt: skip
def test_without_observed_prices_the_close_says_why(lab, state, words):
    lab.forecast(REAL_TEST_TICKER, FRIDAY, signals=SIGNALS)
    (stock,) = build(
        lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER], recent=Observed(state)
    ).stocks
    assert not stock.close.available and stock.close.reason == state and words in stock.close.words


def test_a_newer_observed_close_than_the_forecast_is_named_as_such(lab):
    lab.forecast(REAL_TEST_TICKER, THURSDAY, signals=SIGNALS)
    (stock,) = build(lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER],
                     recent=Observed(session=FRIDAY)).stocks  # fmt: skip
    assert stock.status == "stale" and stock.close.session == FRIDAY
    assert f"The newest forecast is as of the close of {THURSDAY}" in stock.close.words


def test_frozen_reversal_statistics_name_development_history(lab):
    frozen = {**SIGNALS["after_this_streak"], "source": "development history through 2024-12-31",
              "run_length": 9, "table_length": 7}  # fmt: skip
    lab.forecast("DEMO_CALM", FRIDAY, signals={**SIGNALS, "after_this_streak": frozen})
    (stock,) = build(lab, MONDAY, tickers=["DEMO_CALM"]).stocks
    assert "past cases in development history through 2024-12-31" in stock.reversal.words
    assert "the current run is 9 sessions, so the 7-session figures" in stock.reversal.words


def test_b1_beside_a_challenger_gives_a_matched_comparison(lab):
    from stock_agent.stock_lab.scores import rps

    lab.store.record_evaluation(_evaluation("S-dev3", "B3"))
    b1 = ((0.3, 0.4, 0.3), (0.3, 0.4, 0.3))
    _scored_history(lab, ["DEMO_CALM", "DEMO_SWING"], 106, estimator="B3", run_id="S-dev3", b1=b1)
    prospective = build(lab, MONDAY, tickers=["DEMO_CALM", "DEMO_SWING"]).reliability.prospective
    found = prospective.comparison
    assert found is not None and found.status == "ok" and found.dates >= 100
    assert (found.candidate_estimator, found.baseline_estimator) == ("B3", "B1")
    observations = [
        (o.outcome, o.as_of_session)
        for o in report.rel.observations(
            [f for s in ("DEMO_CALM", "DEMO_SWING") for f in lab.store.forecasts(s)],
            [x for s in ("DEMO_CALM", "DEMO_SWING") for x in lab.store.scores(s)],
        )
        if o.target == "1"
    ]
    by_date = {}
    for outcome, day in observations:
        y = np.array([outcome])
        gap = rps(np.array([[0.2, 0.15, 0.65]]), y)[0] - rps(np.array([[0.3, 0.4, 0.3]]), y)[0]
        by_date.setdefault(day, []).append(gap)
    expected = float(np.mean([np.mean(v) for v in by_date.values()]))
    (rps_row,) = [d for d in found.differences if d.score == "rps"]
    assert rps_row.difference == pytest.approx(expected, abs=1e-12)
    assert "matched forecasts" in prospective.comparison_words
    assert "it never promotes a model" in prospective.comparison_words
    assert "negative means B3 scored better" in prospective.comparison_words


def test_too_few_matched_dates_say_insufficient(lab):
    lab.store.record_evaluation(_evaluation("S-dev3", "B3"))
    b1 = ((0.3, 0.4, 0.3), (0.3, 0.4, 0.3))
    _scored_history(lab, ["DEMO_CALM"], 12, estimator="B3", run_id="S-dev3", b1=b1)
    prospective = build(lab, MONDAY, tickers=["DEMO_CALM"]).reliability.prospective
    assert prospective.comparison.status == "insufficient"
    assert "at least 100 matched dates" in prospective.comparison_words


def test_a_newest_session_served_without_a_close_shows_the_last_served_close(lab):
    lab.forecast(REAL_TEST_TICKER, FRIDAY, signals=SIGNALS)
    reader = Observed(session="2026-10-19", close=None)
    served = type("Row", (), {"session": FRIDAY, "close": 130.5})()
    gap = type("Row", (), {"session": "2026-10-19", "close": None})()

    def recent(symbol, *, retrieved_by):
        found = reader(symbol, retrieved_by=retrieved_by)
        found.sessions, found.last_close = (served, gap), None
        return found

    briefing = build(lab, MONDAY, data_class="real", tickers=[REAL_TEST_TICKER], recent=recent)
    (stock,) = briefing.stocks
    assert stock.close.available and (stock.close.session, stock.close.value) == (FRIDAY, 130.5)
    assert "2026-10-19, was served without a close" in stock.close.words
    assert "130.50 at the close of" in " ".join(render_text(briefing).split())


def test_b1_observations_come_from_the_comparator_on_the_stored_outcome(lab):
    from stock_agent.stock_lab.scores import brier, log_loss, rps

    b1 = ((0.6, 0.3, 0.1), (0.5, 0.3, 0.2))  # asymmetric: a reversed order would show
    kept = lab.forecast("DEMO_CALM", THURSDAY, estimator="B3", b1=b1, model_version="model-a")
    other = lab.forecast("DEMO_SWING", THURSDAY, estimator="B3", b1=b1, model_version="model-b")
    refused = lab.forecast("DEMO_TREND", THURSDAY, refusal={"reason": "x", "detail": "y"})
    lab.score(kept, "1", 2, at=ny(FRIDAY, "20:05"))
    lab.score(kept, "cum", 0, at=ny(FRIDAY, "20:06"))
    lab.score(other, "1", 1, at=ny(FRIDAY, "20:07"))
    own = lab.forecast("DEMO_TREND", FRIDAY, estimator="B1")
    lab.score(own, "1", 1, at=ny("2026-10-19", "20:05"))
    forecasts = [*lab.store.forecasts("DEMO_CALM"), *lab.store.forecasts("DEMO_SWING"),
                 *lab.store.forecasts("DEMO_TREND")]  # fmt: skip
    scores = [*lab.store.scores("DEMO_CALM"), *lab.store.scores("DEMO_SWING"),
              *lab.store.scores("DEMO_TREND")]  # fmt: skip
    found = report.rel.comparator_observations(forecasts, scores)
    assert [(o.symbol, o.target, o.model_version) for o in found] == [
        ("DEMO_CALM", "1", "model-a"), ("DEMO_SWING", "1", "model-b"), ("DEMO_CALM", "cum", "model-a")
    ]  # B1's own forecasts and refusals add nothing  # fmt: skip
    assert refused["refusal"] is not None
    first = found[0]
    p, y = np.array([[0.6, 0.3, 0.1]]), np.array([2])
    assert (first.p_decrease, first.p_stable, first.p_increase) == (0.6, 0.3, 0.1)
    assert first.outcome == 2 and first.estimator == "B1"
    assert first.rps == rps(p, y)[0] and first.brier == brier(p, y)[0]
    assert first.log_loss == log_loss(p, y)[0]
    total = found[2]
    assert (total.p_decrease, total.p_increase, total.outcome) == (0.5, 0.2, 0)
    assert (
        report.rel.comparator_observations(forecasts, scores, superseded={kept["forecast_id"]})[
            0
        ].symbol
        == "DEMO_SWING"
    )
