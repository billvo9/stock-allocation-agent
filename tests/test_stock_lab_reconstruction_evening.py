"""
Reconstructing missed evenings (stock_agent.stock_lab.evening.reconstruct_missed;
owner decision of 2026-10-30, replacing decision A4's "missed sessions are
never backfilled" for the activated model).

The next evening run reconstructs every missed (stock, session) since the
first live forecast, automatically: the frozen model on the 252-session
window as of that session, sessions dated up to its close only, recorded
write-once as reconstructed with its true created_at. A reconstruction equals
what the on-time run would have produced on the same data. It is scored by
session dates, kept off the live track, and the evening stays missed on time.

Scenario on a weekday calendar: Thursday AS_OF recorded live; Friday missed
(its prices stored, its forecasts never written); Monday recorded live, then
reconstructed. Injected clocks; no network, nothing real.
"""

from __future__ import annotations

import pandas as pd
import pytest
import test_stock_lab_evening as base
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    Clock,
    next_day,
    ny,
    record,
    steady_bars,
    weekdays,
    window_start,
)

from stock_agent.stock_lab import evening, evenings, forecast, registry
from stock_agent.stock_lab.store import stamp
from stock_agent.stock_lab.symbols import parse_tickers

frozen, lab = base.frozen, base.lab  # the evening tests' frozen model and stores
TICKERS, SOURCES = base.TICKERS, base.SOURCES
FRIDAY, MONDAY, TUESDAY = next_day(AS_OF), next_day(AS_OF, 2), next_day(AS_OF, 3)
MADE = ny(MONDAY, "20:40")  # the Monday run, after its live forecasts
COMPARED = ("symbol", "as_of_session", "target_sessions", "daily", "cumulative", "comparator",
            "signals", "band", "cumulative_band", "estimator", "model_version", "feature_version",
            "evaluation_run_id", "evidence_status", "calibration_status", "refusal", "supersedes")  # fmt: skip


def _live(lab, day: str, at: str = "20:35", tickers=TICKERS):
    research, inference, model, clock = lab
    clock.now = ny(day, at)
    return evening.forecast_watchlist(
        research, model, tickers=tickers, now=clock.now, expected=pd.Timestamp(day, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, record=True,
    )  # fmt: skip


def _reconstruct(lab, *, at=MADE, expected=MONDAY, model=None):
    research, inference, frozen_model, clock = lab
    clock.now = at
    return evening.reconstruct_missed(
        research, model or frozen_model, clock=clock, expected=pd.Timestamp(expected, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path,
    )  # fmt: skip


def _watch(lab, tickers: str, at: pd.Timestamp, *, remove: bool = False) -> None:
    """An add (or remove) in the store's append-only watchlist, recorded at `at`."""

    research, _, _, clock = lab
    before, clock.now = clock.now, at
    if remove:
        research.remove(tickers.split())
    else:
        research.add(parse_tickers(tickers))
    clock.now = before


@pytest.fixture
def watching(lab):
    """The three stocks watched since Thursday noon, before any evening below began."""

    _watch(lab, " ".join(TICKERS), ny(AS_OF, "12:00"))
    return lab


def _batch(lab, day: str, *, wild_after: str | None = None, tickers=TICKERS) -> None:
    """Each ticker's window ending `day`, retrieved at 20:30 that day; prices after `wild_after`
    multiplied tenfold (moves no forecast as of `wild_after` may ever see)."""

    inference = lab[1]
    clock = Clock(ny(day, "20:30"))
    store = type(inference)(inference.path, clock=clock)
    for ticker, stock_id in ((t, SOURCES[t]) for t in tickers):
        bars = steady_bars(window_start(day), day, stock=stock_id)
        if wild_after is not None:
            later = bars["session"] > pd.Timestamp(wild_after, tz="UTC")
            bars.loc[later, ["open", "high", "low", "close"]] *= 10.0
        record(store, clock, window_start(day), day, retrieved_at=ny(day, "20:30"), ticker=ticker,
               bars=bars)  # fmt: skip


@pytest.fixture
def missed_friday(watching):
    """Thursday live; Friday's prices stored but its evening missed; Monday's prices stored."""

    lab = watching
    _live(lab, AS_OF)
    _batch(lab, FRIDAY)
    _batch(lab, MONDAY, wild_after=FRIDAY)
    return lab


def _on_time(lab, ticker: str) -> dict:
    """What Friday's run would have recorded, computed now as a dry run on the same data."""

    research, inference, model, _ = lab
    found = evening.forecast_watchlist(
        research, model, tickers=(ticker,), now=ny(FRIDAY, "20:35"),
        expected=pd.Timestamp(FRIDAY, tz="UTC"), sessions=weekdays, calendar_id=CALENDAR_ID,
        path=inference.path, record=False,
    )  # fmt: skip
    return found.results[0].forecast


def test_the_next_run_reconstructs_every_missed_stock_and_session(missed_friday):
    research = missed_friday[0]
    _live(missed_friday, MONDAY)
    found = _reconstruct(missed_friday)
    assert [(r.ticker, r.session, r.status) for r in found] == [
        (ticker, FRIDAY, evening.RECORDED) for ticker in TICKERS
    ]
    for ticker in TICKERS:
        (rebuilt,) = research.forecasts(ticker, reconstructed=True)
        assert rebuilt["as_of_session"] == FRIDAY and forecast.is_reconstructed(rebuilt)
        assert rebuilt["created_at"] == rebuilt["data"]["decision_time"] == stamp(MADE)
        assert pd.Timestamp(rebuilt["data"]["retrieved_at"]) <= MADE
        assert rebuilt["target_sessions"][0] == MONDAY  # its evening closed at Monday's open
        assert [f["as_of_session"] for f in research.forecasts(ticker)] == [AS_OF, MONDAY]
    assert research.verify_ledger()[1] is None


def test_a_reconstruction_equals_the_on_time_forecast_on_the_same_data(missed_friday):
    research = missed_friday[0]
    _reconstruct(missed_friday)
    for ticker in TICKERS:
        (rebuilt,) = research.forecasts(ticker, reconstructed=True)
        on_time = _on_time(missed_friday, ticker)
        assert {k: rebuilt[k] for k in COMPARED} == {k: on_time[k] for k in COMPARED}
        for key in ("source", "last_close_session", "window"):
            assert rebuilt["data"][key] == on_time["data"][key], key
        assert rebuilt["limitations"][1:] == on_time["limitations"]


def test_no_session_after_the_missed_close_ever_enters(missed_friday):
    # Monday's batch moved every price after Friday tenfold; the reconstruction never sees it.
    research, inference = missed_friday[0], missed_friday[1]
    _reconstruct(missed_friday)
    for ticker in TICKERS:
        (rebuilt,) = research.forecasts(ticker, reconstructed=True)
        assert rebuilt["data"]["window"]["sessions"] == 252
        state = rebuilt["signals"]["market_state"]
        assert state == _on_time(missed_friday, ticker)["signals"]["market_state"]
        monday = inference.batches(ticker)[-1]
        assert monday.request_end == MONDAY and monday.batch_id in rebuilt["data"]["batch_ids"]


def test_a_second_run_reconstructs_nothing_more(missed_friday):
    research = missed_friday[0]
    _reconstruct(missed_friday)
    before = {t: research.forecasts(t, reconstructed=True) for t in TICKERS}
    assert _reconstruct(missed_friday, at=MADE + pd.Timedelta(hours=1)) == ()
    assert {t: research.forecasts(t, reconstructed=True) for t in TICKERS} == before


def test_a_stock_missing_from_a_partly_recorded_evening_is_reconstructed_too(watching):
    # The owner's answer of 2026-10-30: per stock. MU was recorded live on Friday, AMD and NVDA
    # were not: those two are reconstructed, MU never, and the evening stays recorded.
    lab = watching
    research = lab[0]
    _live(lab, AS_OF)
    _batch(lab, FRIDAY)
    _live(lab, FRIDAY, tickers=("MU",))
    _batch(lab, MONDAY)
    found = _reconstruct(lab)
    assert [(r.ticker, r.session, r.status) for r in found] == [
        ("AMD", FRIDAY, evening.RECORDED),
        ("NVDA", FRIDAY, evening.RECORDED),
    ]
    assert research.forecasts("MU", reconstructed=True) == ()
    assert [f["as_of_session"] for f in research.forecasts("MU")] == [AS_OF, FRIDAY]
    noted = evenings.evening_record(research, now=ny(MONDAY, "20:45"),
                                    expected=pd.Timestamp(MONDAY, tz="UTC"), sessions=weekdays)  # fmt: skip
    assert noted["missed"] == [] and noted["reconstructed"] == [FRIDAY]


def test_a_missed_evening_partly_reconstructed_is_completed_at_the_next_run(watching):
    # NVDA's prices for Friday arrive only with Tuesday's batch: its window is short on Monday.
    lab = watching
    research = lab[0]
    _live(lab, AS_OF)
    _batch(lab, FRIDAY, tickers=("AMD", "MU"))
    _batch(lab, MONDAY, tickers=("AMD", "MU"))
    first = _reconstruct(lab)
    assert [(r.ticker, r.status, r.reason) for r in first] == [
        ("AMD", evening.RECORDED, None),
        ("MU", evening.RECORDED, None),
        ("NVDA", evening.REFUSED, "short_window"),
    ]
    _batch(lab, TUESDAY, tickers=("NVDA",))
    later = _reconstruct(lab, at=ny(TUESDAY, "20:40"), expected=TUESDAY)
    assert [(r.ticker, r.session, r.status) for r in later] == [
        ("NVDA", FRIDAY, evening.RECORDED),  # completed once its window can be read
        ("AMD", MONDAY, evening.RECORDED),  # Monday was missed too
        ("MU", MONDAY, evening.RECORDED),
        ("NVDA", MONDAY, evening.RECORDED),
    ]
    assert all(
        {f["as_of_session"] for f in research.forecasts(t, reconstructed=True)} == {FRIDAY, MONDAY}
        for t in TICKERS
    )


def test_only_the_stocks_watched_when_the_evening_began_are_reconstructed(lab):
    # AMD and MU watched since Thursday; NVDA added on Monday, after Friday's evening began;
    # MU removed on Monday, after it. The present watchlist chooses nothing.
    _watch(lab, "AMD MU", ny(AS_OF, "12:00"))
    _live(lab, AS_OF)
    _batch(lab, FRIDAY)
    _watch(lab, "NVDA", ny(MONDAY, "12:00"))
    _watch(lab, "MU", ny(MONDAY, "12:01"), remove=True)
    _batch(lab, MONDAY)
    found = _reconstruct(lab)
    assert [(r.ticker, r.session, r.status) for r in found] == [
        ("AMD", FRIDAY, evening.RECORDED),
        ("MU", FRIDAY, evening.RECORDED),
    ]
    assert lab[0].forecasts("NVDA", reconstructed=True) == ()


def test_a_window_only_model_is_never_reconstructed(missed_friday):
    import dataclasses

    pooled = dataclasses.replace(missed_friday[2], state_rule=registry.WINDOW_ONLY)
    found = _reconstruct(missed_friday, model=pooled)
    assert {(r.session, r.status, r.reason) for r in found} == {
        (FRIDAY, evening.REFUSED, "window_only")
    }
    assert len(found) == len(TICKERS)
    assert all(missed_friday[0].forecasts(t, reconstructed=True) == () for t in TICKERS)


def test_an_evening_that_can_still_be_recorded_live_is_never_reconstructed(missed_friday):
    # Monday's evening is open until Tuesday's open: only Friday is reconstructed.
    found = _reconstruct(missed_friday, at=ny(TUESDAY, "09:29"), expected=MONDAY)
    assert {r.session for r in found} == {FRIDAY}
    assert all(missed_friday[0].forecasts(t, reconstructed=True)[0]["as_of_session"] == FRIDAY
               for t in TICKERS)  # fmt: skip


def test_nothing_is_reconstructed_before_the_first_live_forecast(watching):
    lab = watching
    _batch(lab, FRIDAY)
    assert _reconstruct(lab) == ()  # no live forecast yet: no prospective record to fill
    assert all(lab[0].forecasts(t, reconstructed=True) == () for t in TICKERS)


def test_a_model_not_activated_reconstructs_nothing(missed_friday, monkeypatch):
    monkeypatch.setattr(evening, "ACTIVATED_MODELS", frozenset())
    with pytest.raises(evening.NotActivated):
        _reconstruct(missed_friday)
    assert all(missed_friday[0].forecasts(t, reconstructed=True) == () for t in TICKERS)


def test_a_session_without_its_window_is_reported_and_tried_again_later(watching):
    lab = watching
    _live(lab, AS_OF)
    # Friday was never stored, and Monday's batch begins after it: no window ends on Friday.
    first = next_day(FRIDAY, 1)
    clock = Clock(ny(MONDAY, "20:30"))
    store = type(lab[1])(lab[1].path, clock=clock)
    for ticker, stock_id in SOURCES.items():
        record(store, clock, first, MONDAY, retrieved_at=ny(MONDAY, "20:30"), ticker=ticker,
               bars=steady_bars(first, MONDAY, stock=stock_id))  # fmt: skip
    found = _reconstruct(lab)
    assert {(r.session, r.status, r.reason) for r in found} == {
        (FRIDAY, evening.REFUSED, "short_window")
    }
    assert all(lab[0].forecasts(t, reconstructed=True) == () for t in TICKERS)


# --- the evening record and scoring --------------------------------------------------------


def test_a_reconstructed_evening_stays_missed_on_time_and_is_noted(missed_friday):
    research, _, _, clock = missed_friday
    _live(missed_friday, MONDAY)
    _reconstruct(missed_friday)
    clock.now = ny(TUESDAY, "20:35")
    record_ = evenings.evening_record(
        research, now=clock.now, expected=pd.Timestamp(TUESDAY, tz="UTC"), sessions=weekdays
    )
    assert record_["since"] == AS_OF and record_["missed"] == [FRIDAY]
    assert record_["recorded"] + len(record_["missed"]) == record_["due"]
    assert record_["reconstructed"] == [FRIDAY]
    assert FRIDAY not in evenings.recorded_evenings(research)


def test_a_reconstruction_is_scored_by_session_dates_and_kept_off_the_live_track(missed_friday):
    research, inference, model, clock = missed_friday
    _live(missed_friday, MONDAY)
    _reconstruct(missed_friday)
    _batch(missed_friday, TUESDAY, wild_after=FRIDAY)
    clock.now = ny(TUESDAY, "20:35")
    summary = evening.score_real(
        research, symbols=TICKERS, now=clock.now, expected=pd.Timestamp(TUESDAY, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, write=True,
        model_id=model.model_id,
    )  # fmt: skip
    for ticker in TICKERS:
        (rebuilt,) = research.forecasts(ticker, reconstructed=True)
        scored = research.scores(ticker, reconstructed=True)
        # Monday's outcome was stored before it was made; it is scored all the same (by date).
        assert {s["target"] for s in scored if s["status"] == "scored"} >= {"1", "2"}
        assert {s["forecast_id"] for s in scored} == {rebuilt["forecast_id"]}
        assert rebuilt["forecast_id"] not in {s["forecast_id"] for s in research.scores(ticker)}
    # The live counts are the live forecasts' alone; the reconstructed ones are counted apart.
    live_scored = sum(len([s for s in research.scores(t) if s["status"] == "scored"])
                      for t in TICKERS)  # fmt: skip
    assert summary.scored == live_scored
    assert summary.reconstructed["scored"] == sum(
        len([s for s in research.scores(t, reconstructed=True) if s["status"] == "scored"])
        for t in TICKERS
    )
    assert summary.as_dict()["reconstructed"] == summary.reconstructed


def test_live_scoring_is_unchanged_by_the_reconstruction_rule(missed_friday):
    # A live forecast still never scores a target whose outcome it could have seen.
    research, inference, model, clock = missed_friday
    _batch(missed_friday, TUESDAY)
    clock.now = ny(TUESDAY, "20:35")
    evening.score_real(
        research, symbols=TICKERS, now=clock.now, expected=pd.Timestamp(TUESDAY, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, write=True,
        model_id=model.model_id,
    )  # fmt: skip
    for ticker in TICKERS:
        statuses = {s["target"]: s["status"] for s in research.scores(ticker)}
        assert statuses["1"] == "scored"  # Thursday's forecast: Friday stored after it
