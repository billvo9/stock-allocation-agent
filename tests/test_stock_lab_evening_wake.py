"""
The real evening's record step when the Mac sleeps mid-run
(stock_agent.stock_lab.evening.forecast_watchlist with a clock).

On 2026-10-27 the Mac fell asleep between the record step's decision and its
writes, so the store refused all four forecasts as backdated: each was made
at the decision time, more than CLOCK_SKEW before the write. Now, when the
store refuses a forecast as backdated, the job makes that forecast again,
once, at its clock: a new decision, its window read as of then. A forecast
is always made at its decision time (created_at == data.decision_time), from
batches retrieved by then, and never at or after the next open. Any other
refusal, and a second one, is final.

Injected clocks and a fake store write (the real store behind a wrapper that
lets time jump); no network, nothing real.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
import test_stock_lab_evening as base
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    Clock,
    make_bars,
    next_day,
    ny,
    record,
    weekdays,
    window_start,
)

from stock_agent.stock_lab import evening, registry
from stock_agent.stock_lab.frozen_model import RegistryRefused
from stock_agent.stock_lab.inference.window import load_inference_window
from stock_agent.stock_lab.store import CLOCK_SKEW, BackdatingRefused, StoreRefused, stamp

frozen, lab = base.frozen, base.lab  # the evening tests' frozen model and stores
TICKERS, EXPECTED = base.TICKERS, base.EXPECTED
DECISION = ny(AS_OF, "21:07")  # the record step's decision time (the window is read as of it)
OPEN = ny(next_day(AS_OF), "09:30")  # the first target session opens: no forecast from then


def _evening(lab, *, now=DECISION, record_it=True, clocked=True):
    research, inference, model, clock = lab
    return evening.forecast_watchlist(
        research, model, tickers=TICKERS, now=now, expected=EXPECTED, sessions=weekdays,
        calendar_id=CALENDAR_ID, path=inference.path, record=record_it,
        clock=clock if clocked else None,
    )  # fmt: skip


class Writes:
    """The store's record_forecast, counted (with each artifact), `before` run ahead of each."""

    def __init__(self, store, before=None):
        self.write, self.before, self.calls, self.artifacts = store.record_forecast, before, [], []

    def __call__(self, artifact, **kwargs):
        self.calls.append(artifact["symbol"])
        self.artifacts.append(dict(artifact))
        if self.before is not None:
            self.before(len(self.calls))
        return self.write(artifact, **kwargs)


def _writes(lab, monkeypatch, before=None) -> Writes:
    writes = Writes(lab[0], before)
    monkeypatch.setattr(lab[0], "record_forecast", writes)
    return writes


def _honest(lab, stored: dict, *, decided: pd.Timestamp, written: pd.Timestamp) -> None:
    """A forecast made at `decided`, from the window as of then, and written at `written`."""

    assert stored["created_at"] == stored["data"]["decision_time"] == stamp(decided)
    window = load_inference_window(lab[1].path, stored["symbol"], AS_OF, retrieved_by=decided,
                                   sessions=weekdays, calendar_id=CALENDAR_ID)  # fmt: skip
    assert stored["data"]["batch_ids"] == [batch.batch_id for batch in window.batches]
    assert pd.Timestamp(stored["data"]["retrieved_at"]) <= decided  # nothing retrieved later
    assert abs(decided - written) <= CLOCK_SKEW  # recorded when made: no backdating
    assert decided < OPEN  # before its first target session opens


def test_a_sleep_between_the_decision_and_the_writes_makes_each_forecast_again(lab, monkeypatch):
    # 2026-10-27: the decision at 21:07, the writes after the Mac slept until 21:40.
    research, _, _, clock = lab
    clock.now = ny(AS_OF, "21:40")
    writes = _writes(lab, monkeypatch)
    found = _evening(lab)
    assert [r.status for r in found.results] == [evening.RECORDED] * len(TICKERS)
    assert writes.calls == [ticker for ticker in TICKERS for _ in range(2)]  # refused, then made
    for ticker in TICKERS:
        (stored,) = research.forecasts(ticker)
        _honest(lab, stored, decided=clock.now, written=clock.now)
    # The same evening's forecast: equal to the refused one but for its decision time and id.
    changed = {"forecast_id", "created_at", "data"}
    for refused, made in zip(writes.artifacts[::2], writes.artifacts[1::2], strict=True):
        assert refused["created_at"] == stamp(DECISION) and made["created_at"] == stamp(clock.now)
        assert {k: v for k, v in refused.items() if k not in changed} == {
            k: v for k, v in made.items() if k not in changed
        }
        assert {**refused["data"], "decision_time": None} == {**made["data"], "decision_time": None}
    assert research.verify_ledger()[1] is None


def test_a_forecast_made_again_says_so_in_its_result(lab):
    clock = lab[3]
    clock.now = ny(AS_OF, "21:40")
    found = _evening(lab)
    made = stamp(clock.now)
    assert all(r.detail.endswith(f"made again at {made} after a backdating refusal")
               for r in found.results)  # fmt: skip


def test_a_window_only_model_is_never_made_again_on_a_panel_of_its_own(lab):
    research, inference, model, clock = lab
    pooled = dataclasses.replace(model, state_rule=registry.WINDOW_ONLY)
    clock.now = ny(AS_OF, "21:40")
    found = evening.forecast_watchlist(
        research, pooled, tickers=TICKERS, now=DECISION, expected=EXPECTED, sessions=weekdays,
        calendar_id=CALENDAR_ID, path=inference.path, record=True, clock=clock,
    )  # fmt: skip
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "store_refused")}
    assert all("backdating is refused" in r.detail for r in found.results)


def test_a_refused_remake_is_reported_and_never_stops_the_evening(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = ny(AS_OF, "21:40")
    made = evening.forecast_record

    def refuses(*args, **kwargs):
        if pd.Timestamp(kwargs["created_at"]) == clock.now:  # only the forecast made again
            raise RegistryRefused("environment_mismatch", "a package changed meanwhile")
        return made(*args, **kwargs)

    monkeypatch.setattr(evening, "forecast_record", refuses)
    found = _evening(lab)
    assert {(r.status, r.reason) for r in found.results} == {
        (evening.REFUSED, "environment_mismatch")
    }
    assert all(research.forecasts(t) == () for t in TICKERS)


def test_without_the_clock_a_backdating_refusal_stays_final(lab):
    research, _, _, clock = lab
    clock.now = ny(AS_OF, "21:40")
    found = _evening(lab, clocked=False)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "store_refused")}
    assert all("backdating is refused" in r.detail for r in found.results)
    assert all(research.forecasts(t) == () for t in TICKERS)


def test_a_forecast_is_made_again_only_once(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = DECISION

    def sleeps(call):  # the Mac sleeps before every write
        clock.now += pd.Timedelta(minutes=10)

    writes = _writes(lab, monkeypatch, sleeps)
    found = _evening(lab)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "store_refused")}
    assert all("backdating is refused" in r.detail for r in found.results)
    assert writes.calls == [ticker for ticker in TICKERS for _ in range(2)]  # once more each
    assert all(research.forecasts(t) == () for t in TICKERS)


def test_a_forecast_refused_after_a_sleep_past_the_open_is_never_made_again(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = DECISION

    def sleeps(call):  # the first write reaches the store after the Mac slept past the open
        if call == 1:
            clock.now = ny(next_day(AS_OF), "09:31")

    writes = _writes(lab, monkeypatch, sleeps)
    found = _evening(lab)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "after_next_open")}
    # The first was refused as backdated, and its window as of 09:31 refused; no other write
    # started once the job's clock had passed the open.
    assert writes.calls == [TICKERS[0]]
    assert all(research.forecasts(t) == () for t in TICKERS)
    assert research.verify_ledger()[1] is None


def test_no_write_starts_at_or_after_the_open_even_for_a_forecast_made_before_it(lab, monkeypatch):
    # Decided at 09:29, within the store's skew of the open; the Mac sleeps while computing.
    research, _, _, clock = lab
    clock.now = ny(next_day(AS_OF), "09:29")
    made = evening.forecast_record

    def sleeps(*args, **kwargs):
        clock.now = OPEN
        return made(*args, **kwargs)

    monkeypatch.setattr(evening, "forecast_record", sleeps)
    writes = _writes(lab, monkeypatch)
    found = _evening(lab, now=clock.now)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "after_next_open")}
    assert writes.calls == [] and all(research.forecasts(t) == () for t in TICKERS)


def test_any_other_refusal_fails_at_once_and_is_never_made_again(lab, monkeypatch):
    research = lab[0]
    calls = []

    def refuses(artifact, **kwargs):
        calls.append(artifact["symbol"])
        raise StoreRefused("a forecast cannot predate data already stored for its stock")

    monkeypatch.setattr(research, "record_forecast", refuses)
    found = _evening(lab, now=DECISION)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "store_refused")}
    assert calls == list(TICKERS)  # one attempt each


def _later_batch(lab, retrieved: str, shift: float) -> None:
    """A batch for every ticker retrieved at `retrieved` New York, with other prices."""

    later = Clock(ny(AS_OF, retrieved))
    other = type(lab[1])(lab[1].path, clock=later)
    for ticker in TICKERS:
        record(other, later, window_start(), AS_OF, retrieved_at=ny(AS_OF, retrieved),
               ticker=ticker, bars=make_bars(window_start(), AS_OF, shift=shift))  # fmt: skip


def test_a_batch_retrieved_after_the_decision_never_reaches_its_forecast(lab):
    # The later batch is already stored when the job reads, and its clock is past it.
    research, _, _, clock = lab
    _later_batch(lab, "21:20", shift=50.0)
    clock.now = DECISION + pd.Timedelta(minutes=2)
    found = _evening(lab)
    assert [r.status for r in found.results] == [evening.RECORDED] * len(TICKERS)
    for ticker in TICKERS:
        (stored,) = research.forecasts(ticker)
        _honest(lab, stored, decided=DECISION, written=clock.now)
        assert len(stored["data"]["batch_ids"]) == 1  # the 20:30 batch only, never the 21:20
    assert all("made again" not in r.detail for r in found.results)  # written as made


def test_a_forecast_made_again_reads_only_what_was_retrieved_by_its_new_decision(lab, monkeypatch):
    research, _, _, clock = lab
    _later_batch(lab, "21:20", shift=50.0)  # retrieved before the new decision: may be used
    _later_batch(lab, "21:50", shift=80.0)  # retrieved after it: never
    clock.now = ny(AS_OF, "21:40")
    writes = _writes(lab, monkeypatch)
    found = _evening(lab)
    assert [r.status for r in found.results] == [evening.RECORDED] * len(TICKERS)
    newest = {t: lab[1].batches(t)[-1].batch_id for t in TICKERS}  # the 21:50 batches
    for ticker in TICKERS:
        (stored,) = research.forecasts(ticker)
        _honest(lab, stored, decided=clock.now, written=clock.now)
        assert newest[ticker] not in stored["data"]["batch_ids"]
    assert len(writes.artifacts[0]["data"]["batch_ids"]) == 1  # the refused one: 20:30 only


def test_a_second_evening_run_skips_and_writes_nothing(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = DECISION
    _evening(lab)
    before = {t: research.forecasts(t) for t in TICKERS}
    clock.now = ny(AS_OF, "23:52")
    writes = _writes(lab, monkeypatch)
    again = _evening(lab, now=ny(AS_OF, "23:52"))
    assert {(r.status, r.reason) for r in again.results} == {(evening.SKIPPED, "already_recorded")}
    assert writes.calls == [] and {t: research.forecasts(t) for t in TICKERS} == before


def test_a_dry_run_records_nothing(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = ny(AS_OF, "21:40")
    writes = _writes(lab, monkeypatch)
    found = _evening(lab, record_it=False)
    assert [r.status for r in found.results] == [evening.DRY_RUN] * len(TICKERS)
    assert writes.calls == [] and all(research.forecasts(t) == () for t in TICKERS)


def test_only_the_backdating_check_raises_the_retryable_refusal(lab):
    # The guard itself is unchanged: CLOCK_SKEW either way, the same message, a StoreRefused.
    assert issubclass(BackdatingRefused, StoreRefused) and CLOCK_SKEW == pd.Timedelta(minutes=5)
    research, _, _, clock = lab
    clock.now = DECISION
    (result,) = _evening(lab, record_it=False).results[:1]
    artifact = dict(result.forecast)
    clock.now = DECISION + CLOCK_SKEW + pd.Timedelta(microseconds=1)
    with pytest.raises(BackdatingRefused, match="backdating is refused"):
        research.record_forecast(artifact, real_authorized=True)
    clock.now = DECISION - CLOCK_SKEW - pd.Timedelta(microseconds=1)
    with pytest.raises(StoreRefused, match="in the future") as raised:
        research.record_forecast(artifact, real_authorized=True)
    assert not isinstance(raised.value, BackdatingRefused)


def test_a_clock_stepped_back_beyond_the_skew_is_refused_and_never_made_again(lab, monkeypatch):
    research, _, _, clock = lab
    clock.now = DECISION - CLOCK_SKEW - pd.Timedelta(minutes=1)
    writes = _writes(lab, monkeypatch)
    found = _evening(lab)
    assert {(r.status, r.reason) for r in found.results} == {(evening.REFUSED, "store_refused")}
    assert all("cannot be created in the future" in r.detail for r in found.results)
    assert writes.calls == list(TICKERS) and all(research.forecasts(t) == () for t in TICKERS)
