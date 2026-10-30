"""
The Stock Lab's forecast history as known at a time (`stock_lab.briefing.history`):
each evening's next-session outlook per stock exactly as recorded, its stored result,
and the missed evenings by the evening job's own rule, never filled in. Point in time:
a forecast, a recalculation, a ledger entry or a score that did not exist yet is not
read. Fixtures only: a Stock Lab store on a simulated clock; real-class records use the
made-up tickers SYNTH and SYNTQ. No network, no private store.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, ny

from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab import evenings
from stock_agent.stock_lab.briefing import history
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.store import StoreRefused

T, Q = REAL_TEST_TICKER, "SYNTQ"
MONDAY, TUESDAY, WEDNESDAY, THURSDAY = "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"
# Decrease, Stable, Increase, with more digits than any display.
P = (0.42137, 0.10931, 0.46932)
P2 = (0.40011, 0.12013, 0.47976)


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def made(lab, symbol, as_of, p=P, **options):
    return lab.forecast(symbol, as_of, created_at=ny(as_of, "20:31"), p=p, **options)


def score(lab, record, *, at, status="scored", outcome=2, move=0.012347, reason=None, target="1"):
    lab.at(at)
    lab.store.record_scores(
        [
            {
                "forecast_id": record["forecast_id"],
                "target": target,
                "symbol": record["symbol"],
                "session": record["target_sessions"][-1 if target == "cum" else int(target) - 1],
                "status": status,
                "reason": reason,
                "outcome": outcome if status == "scored" else None,
                "realized_return": move if status == "scored" else None,
                "brier": 0.1 if status == "scored" else None,
                "log_loss": 0.2 if status == "scored" else None,
                "rps": 0.3 if status == "scored" else None,
                "batch_id": None,
            }
        ]
    )


def read(lab, now, *, tickers=(T, Q), records=None, **options):
    return history.forecast_history(
        records or ReadOnlyRecords(lab.path),
        now=now,
        sessions=nyse_sessions,
        data_class="real",
        tickers=None if tickers is None else list(tickers),
        **options,
    )


def entry(found, as_of, symbol):
    (evening,) = [e for e in found.evenings if e.as_of == as_of]
    (item,) = [x for x in evening.entries if x.symbol == symbol]
    return item


def test_each_evening_shows_the_recorded_chances_and_the_stored_result_newest_first(lab):
    monday = made(lab, T, MONDAY)
    made(lab, Q, MONDAY, p=P2)
    made(lab, T, TUESDAY)
    made(lab, Q, TUESDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"), outcome=2, move=0.012347)
    found = read(lab, ny(WEDNESDAY, "10:00"))
    assert found.problem is None and found.data_class == "real"
    assert [e.as_of for e in found.evenings] == [TUESDAY, MONDAY]
    assert [e.session for e in found.evenings] == [WEDNESDAY, TUESDAY]
    assert (found.since, found.due, found.recorded, found.missed) == (MONDAY, 2, 2, ())
    first = entry(found, MONDAY, T)
    assert (first.decrease, first.stable, first.increase) == P  # exactly as stored
    assert first.session == TUESDAY and first.recorded and first.refusal is None
    assert first.band == 0.005
    assert first.outcome == history.Outcome("scored", "increase", 0.012347, None)
    assert (entry(found, MONDAY, Q).decrease, entry(found, MONDAY, Q).increase) == (P2[0], P2[2])
    assert entry(found, MONDAY, Q).outcome.status == "pending"  # no score written for it
    assert entry(found, TUESDAY, T).outcome == history.Outcome("pending", None, None, None)


def test_a_missed_evening_is_named_and_never_filled_in(lab):
    made(lab, T, MONDAY)
    made(lab, T, WEDNESDAY)
    found = read(lab, ny(THURSDAY, "10:00"), tickers=[T])
    assert [(e.as_of, e.missed) for e in found.evenings] == [
        (WEDNESDAY, False),
        (TUESDAY, True),
        (MONDAY, False),
    ]
    tuesday = found.evenings[1]
    assert tuesday.session == WEDNESDAY and tuesday.entries == ()
    assert (found.due, found.recorded, found.missed) == (3, 2, (TUESDAY,))


def test_the_missed_evenings_are_the_evening_jobs_own(lab):
    made(lab, T, MONDAY)
    made(lab, T, WEDNESDAY)
    now = ny(THURSDAY, "10:00")
    found = read(lab, now, tickers=[T])
    job = evenings.evening_record(
        lab.store, now=now, expected=pd.Timestamp(THURSDAY, tz="UTC"), sessions=nyse_sessions
    )
    assert (found.since, found.due, found.recorded, list(found.missed)) == (
        job["since"],
        job["due"],
        job["recorded"],
        job["missed"],
    )


def test_an_evening_is_not_missed_before_the_next_open(lab):
    made(lab, T, MONDAY)
    found = read(lab, ny(WEDNESDAY, "09:29"), tickers=[T])  # Tuesday's is due at 09:30
    assert [e.as_of for e in found.evenings] == [MONDAY] and found.missed == ()
    later = read(lab, ny(WEDNESDAY, "09:30"), tickers=[T])
    assert later.missed == (TUESDAY,)


def test_nothing_made_after_the_reading_time_is_read(lab):
    monday = made(lab, T, MONDAY)
    # After 07:40 on Tuesday: a recalculation (before Tuesday's open), Tuesday's evening,
    # and the recalculation's score.
    again = lab.forecast(
        T, MONDAY, created_at=ny(TUESDAY, "07:45"), p=P2, supersedes=monday["forecast_id"]
    )
    made(lab, T, TUESDAY)
    score(lab, again, at=ny(TUESDAY, "20:40"))
    found = read(lab, ny(TUESDAY, "07:40"), tickers=[T])
    first = entry(found, MONDAY, T)
    assert (first.decrease, first.stable, first.increase) == P
    assert first.created_at == monday["created_at"] and first.outcome.status == "pending"
    assert [e.as_of for e in found.evenings] == [MONDAY]
    # Each shows once it existed: the live recalculation, its score, Tuesday's evening.
    later = read(lab, ny(WEDNESDAY, "10:00"), tickers=[T])
    shown = entry(later, MONDAY, T)
    assert (shown.decrease, shown.stable, shown.increase) == P2
    assert shown.created_at == again["created_at"] and shown.outcome.status == "scored"
    assert [e.as_of for e in later.evenings] == [TUESDAY, MONDAY]


def test_a_score_counts_from_when_it_was_written(lab):
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"))
    assert entry(read(lab, ny(TUESDAY, "20:39"), tickers=[T]), MONDAY, T).outcome.status == (
        "pending"
    )
    assert entry(read(lab, ny(TUESDAY, "20:40"), tickers=[T]), MONDAY, T).outcome.status == (
        "scored"
    )


def test_the_result_is_the_next_sessions_score_only(lab):
    # Later targets (the second session, the five-day total) are scored later; never shown here.
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"), outcome=2, move=0.012347)
    score(lab, monday, at=ny(WEDNESDAY, "20:40"), target="2", outcome=0, move=-0.020401)
    score(lab, monday, at=ny("2026-10-26", "20:40"), target="cum", outcome=1, move=0.0009)
    found = entry(read(lab, ny("2026-10-27", "10:00"), tickers=[T]), MONDAY, T)
    assert found.outcome == history.Outcome("scored", "increase", 0.012347, None)


def test_a_replaced_forecasts_score_is_never_borrowed(lab):
    monday = made(lab, T, MONDAY)
    lab.forecast(T, MONDAY, created_at=ny(TUESDAY, "07:45"), p=P2, supersedes=monday["forecast_id"])
    score(lab, monday, at=ny(TUESDAY, "20:40"))  # the original's, as scoring writes it
    shown = entry(read(lab, ny(TUESDAY, "21:00"), tickers=[T]), MONDAY, T)
    assert (shown.increase, shown.outcome.status) == (P2[2], "pending")


def test_a_reading_at_a_time_never_changes_afterwards(lab):
    monday = made(lab, T, MONDAY)
    now = ny(TUESDAY, "07:40")
    before = read(lab, now)
    lab.forecast(T, MONDAY, created_at=ny(TUESDAY, "07:45"), p=P2, supersedes=monday["forecast_id"])
    made(lab, Q, TUESDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"))
    lab.watch("SYNTR", at=ny(TUESDAY, "21:00"))
    assert read(lab, now) == before


def test_a_forecast_counts_only_once_its_ledger_entry_exists(lab):
    lab.forecast(T, MONDAY, created_at=ny(MONDAY, "20:31"), recorded_at=ny(MONDAY, "20:35"))
    assert read(lab, ny(MONDAY, "20:33"), tickers=[T]).evenings == ()
    assert [e.as_of for e in read(lab, ny(MONDAY, "20:35"), tickers=[T]).evenings] == [MONDAY]


def test_an_unscorable_target_keeps_its_reason_and_shows_no_result(lab):
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"), status="unscorable", reason="missing_close")
    found = entry(read(lab, ny(WEDNESDAY, "10:00"), tickers=[T]), MONDAY, T)
    assert found.outcome == history.Outcome("unscorable", None, None, "missing_close")


def test_a_refusal_and_a_stock_without_a_forecast_are_named(lab):
    lab.forecast(
        T,
        MONDAY,
        created_at=ny(MONDAY, "20:31"),
        refusal={"reason": "stale_close", "detail": "The latest close is too old."},
    )
    found = read(lab, ny(TUESDAY, "10:00"))
    refused = entry(found, MONDAY, T)
    assert refused.recorded and refused.refusal == "The latest close is too old."
    assert refused.increase is None and refused.outcome is None
    absent = entry(found, MONDAY, Q)
    assert not absent.recorded and absent.increase is None and absent.created_at is None


def test_other_data_classes_and_excluded_stocks_stay_out(lab):
    lab.forecast("DEMO_CALM", MONDAY, created_at=ny(MONDAY, "20:31"))  # synthetic
    made(lab, "SNDK", TUESDAY)
    found = read(lab, ny(WEDNESDAY, "10:00"), excluded=("SNDK",))
    assert [e.as_of for e in found.evenings] == [TUESDAY]  # SNDK's evening was recorded
    assert {x.symbol for x in found.evenings[0].entries} == {T, Q}
    assert not any(x.recorded for x in found.evenings[0].entries)


def test_the_watchlist_is_read_when_no_tickers_are_given(lab):
    lab.watch(T, at=ny(MONDAY, "12:00"))
    made(lab, T, MONDAY)
    found = read(lab, ny(TUESDAY, "10:00"), tickers=None)
    assert [x.symbol for x in found.evenings[0].entries] == [T]


def test_an_excluded_stock_on_the_watchlist_is_left_off(lab):
    lab.watch(T, "SNDK", at=ny(MONDAY, "12:00"))
    made(lab, T, MONDAY)
    made(lab, "SNDK", MONDAY)
    found = read(lab, ny(TUESDAY, "10:00"), tickers=None, excluded=("SNDK",))
    assert found.listed == (T,)
    assert [x.symbol for x in found.evenings[0].entries] == [T]


def test_no_forecast_yet_is_an_empty_history_not_a_problem(lab):
    found = read(lab, ny(TUESDAY, "10:00"))
    assert found.problem is None and found.evenings == () and found.since is None
    assert found.listed == (T, Q)


def test_a_failed_ledger_check_withholds_the_whole_history(lab, monkeypatch):
    made(lab, T, MONDAY)
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    found = read(lab, ny(TUESDAY, "10:00"))
    assert found.problem == "ledger_unverified" and found.evenings == () and found.due == 0


def test_a_record_that_fails_its_check_withholds_the_whole_history(lab):
    made(lab, T, MONDAY)
    made(lab, Q, MONDAY)

    class Failing(ReadOnlyRecords):
        def forecasts(self, symbol, **kwargs):
            if symbol == Q:
                raise ForecastRefused("the record does not match its id")
            return super().forecasts(symbol, **kwargs)

    found = read(lab, ny(TUESDAY, "10:00"), records=Failing(lab.path))
    assert found.problem == "record_check_failed" and found.evenings == ()


def test_a_store_that_cannot_be_read_withholds_the_whole_history(lab):
    made(lab, T, MONDAY)

    class Refusing(ReadOnlyRecords):
        def watched(self):
            raise StoreRefused("not a Stock Lab store")

    found = read(lab, ny(TUESDAY, "10:00"), tickers=None, records=Refusing(lab.path))
    assert found.problem == "record_unreadable" and found.evenings == ()


def test_every_evening_is_kept_newest_first_with_the_stocks_listed(lab):
    for day in (MONDAY, TUESDAY, WEDNESDAY):
        made(lab, T, day)
    found = read(lab, ny(THURSDAY, "10:00"), tickers=[Q, T])
    assert [e.as_of for e in found.evenings] == [WEDNESDAY, TUESDAY, MONDAY]
    assert found.listed == (Q, T)
    assert [x.symbol for x in found.evenings[0].entries] == [Q, T]
    assert read(lab, ny(MONDAY, "10:00"), tickers=[T]).listed == (T,)  # before any evening


def test_reading_the_history_changes_nothing_in_the_store(lab):
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"))
    before = hashlib.sha256(lab.path.read_bytes()).hexdigest()
    read(lab, ny(WEDNESDAY, "10:00"))
    assert hashlib.sha256(lab.path.read_bytes()).hexdigest() == before
