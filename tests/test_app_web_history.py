"""
The Lab app's History view model (docs/product/lab_app_design.md, "History in
detail"): each evening's next-session outlook exactly as recorded and its stored
result, newest first; missed evenings shown as missed and never filled in; an
honest state for everything missing. Fixtures only: a Stock Lab store on a
simulated clock (real-class records use the made-up tickers SYNTH and SYNTQ).
No network, no private store.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, ny
from test_app_destinations import scan_for_advice

from stock_agent.app.web import history, words
from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab.briefing.history import History
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords

NEW_YORK = ZoneInfo("America/New_York")
T, Q = REAL_TEST_TICKER, "SYNTQ"
MONDAY, TUESDAY, WEDNESDAY, THURSDAY = "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"
# Decrease, Stable, Increase, with more digits than any display.
P = (0.42137, 0.10931, 0.46932)


def at(day: str, hhmm: str) -> datetime:
    return ny(day, hhmm).tz_convert(NEW_YORK).to_pydatetime()


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def made(lab, symbol, as_of, p=P, **options):
    return lab.forecast(symbol, as_of, created_at=ny(as_of, "20:31"), p=p, **options)


def score(lab, record, *, at, status="scored", outcome=2, move=0.012347, reason=None):
    lab.at(at)
    lab.store.record_scores(
        [
            {
                "forecast_id": record["forecast_id"],
                "target": "1",
                "symbol": record["symbol"],
                "session": record["target_sessions"][0],
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


def checked(found: dict) -> dict:
    """Every view: strict JSON with no advice words, and outlooks never coloured."""

    scan_for_advice(json.dumps(found, allow_nan=False))
    for evening in found["evenings"]:
        for item in evening["items"]:
            assert set(item) == {
                "ticker", "state", "said", "said_words", "result", "result_words", "recorded",
                "label", "technical",
            }  # fmt: skip
            if item["result"] is not None:
                assert item["result"]["direction"] in ("up", "flat", "down")
    return found


def view(lab, now, *, tickers=(T, Q), records=None, local_zone=NEW_YORK, **options):
    return checked(
        history.build_history(
            records or ReadOnlyRecords(lab.path),
            now=now,
            calendar=nyse_sessions,
            data_class="real",
            local_zone=local_zone,
            tickers=None if tickers is None else list(tickers),
            **options,
        )
    )


def item(found, as_of, ticker):
    (evening,) = [e for e in found["evenings"] if e["as_of"] == as_of]
    (one,) = [x for x in evening["items"] if x["ticker"] == ticker]
    return one


def test_an_evening_shows_what_it_said_and_what_happened(lab):
    monday = made(lab, T, MONDAY)
    made(lab, Q, MONDAY, p=(0.6, 0.3, 0.1))
    score(lab, monday, at=ny(TUESDAY, "20:40"), outcome=2, move=0.012347)
    found = view(lab, at(TUESDAY, "21:00"))  # Tuesday's evening is due at Wednesday's open
    assert (found["schema"], found["version"], found["data_class"]) == (
        "lab_app_history",
        1,
        "real",
    )
    (evening,) = found["evenings"]
    assert evening["title"] == "Tue, Oct 20" and evening["session"] == TUESDAY
    assert evening["sub"] == "Outlook recorded last night, 8:31 PM · waiting for the result"
    scored = item(found, MONDAY, T)
    assert scored["state"] == "scored"
    assert scored["said"] == {"lead": "Said", "up": "47%", "down": "42%"}
    assert scored["recorded"] == {
        "increase": P[2],
        "stable": P[1],
        "decrease": P[0],
        "category": "increase",
        "move": 0.012347,
    }  # exactly as stored
    assert scored["result"] == {"words": "Up +1.23%", "direction": "up", "gave": "It gave Up 47%"}
    assert scored["label"] == (
        f"{T}: Said 47% chance up, 42% chance down. Tuesday: Up, +1.23%. It gave Up 47%."
    )
    waiting = item(found, MONDAY, Q)
    assert (waiting["state"], waiting["result"]) == ("pending", None)
    assert waiting["result_words"] == "No result yet"  # Tuesday's close has passed
    assert found["counts"] == "Since Oct 19: 1 evening recorded, 1 with a result, 0 missed."
    assert found["filters"] == ["All", T, Q]


@pytest.mark.parametrize(
    ("outcome", "move", "words_", "direction", "gave"),
    [
        (0, -0.020401, "Down −2.04%", "down", "It gave Down 42%"),
        (1, 0.000812, "Flat +0.08%", "flat", "It gave Flat 11%"),
    ],
)
def test_the_result_is_the_stored_category_and_move(lab, outcome, move, words_, direction, gave):
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"), outcome=outcome, move=move)
    result = item(view(lab, at(WEDNESDAY, "10:00"), tickers=[T]), MONDAY, T)["result"]
    assert result == {"words": words_, "direction": direction, "gave": gave}


def test_a_result_still_to_come_names_the_close_it_waits_for(lab):
    made(lab, T, WEDNESDAY)
    found = view(lab, at(THURSDAY, "10:00"), tickers=[T])
    (evening,) = found["evenings"]
    assert evening["state"] == "pending" and evening["title"] == "Thu, Oct 22"
    assert item(found, WEDNESDAY, T)["result_words"] == "After Thu's close"
    later = view(lab, at(THURSDAY, "20:00"), tickers=[T])
    assert item(later, WEDNESDAY, T)["result_words"] == "No result yet"


def test_a_missed_evening_is_shown_as_missed_and_never_filled_in(lab):
    made(lab, T, MONDAY)
    made(lab, T, WEDNESDAY)
    found = view(lab, at(THURSDAY, "10:00"), tickers=[T])
    assert [e["state"] for e in found["evenings"]] == ["pending", "missed", "pending"]
    missed = found["evenings"][1]
    assert missed["title"] == "Wed, Oct 21" and missed["items"] == []
    assert missed["sub"] == "No outlook: nothing was recorded on the evening of Tue, Oct 20."
    assert missed["missed_words"] == words.MISSED_STAYS
    assert found["counts"] == "Since Oct 19: 2 evenings recorded, 0 with a result, 1 missed."


def test_a_stock_with_no_outlook_that_evening_and_a_refusal_say_so(lab):
    lab.forecast(
        T,
        MONDAY,
        created_at=ny(MONDAY, "20:31"),
        refusal={"reason": "stale_close", "detail": "The latest close is too old."},
    )
    found = view(lab, at(TUESDAY, "10:00"))
    refused = item(found, MONDAY, T)
    assert (refused["state"], refused["said"], refused["result"]) == ("refused", None, None)
    assert refused["said_words"] == words.REFUSED_CHIP
    assert refused["technical"] == "Technical detail: The latest close is too old."
    absent = item(found, MONDAY, Q)
    assert (absent["state"], absent["said_words"]) == ("absent", words.NO_OUTLOOK_RECORDED)
    assert absent["label"] == f"{Q}: {words.NO_OUTLOOK_RECORDED}."


def test_an_unscorable_day_says_so_with_its_reason(lab):
    monday = made(lab, T, MONDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"), status="unscorable", reason="missing_close")
    found = item(view(lab, at(WEDNESDAY, "10:00"), tickers=[T]), MONDAY, T)
    assert (found["state"], found["result"], found["result_words"]) == (
        "unscorable",
        None,
        words.NO_RESULT,
    )
    assert found["technical"] == "Technical detail: missing_close"
    assert found["said"] == {"lead": "Said", "up": "47%", "down": "42%"}


def test_an_outlook_for_another_day_than_the_calendars_is_withheld(lab):
    made(lab, T, MONDAY, targets=["2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26",
                                  "2026-10-27"])  # fmt: skip
    found = item(view(lab, at(TUESDAY, "10:00"), tickers=[T]), MONDAY, T)
    assert (found["state"], found["said"]) == ("withheld", None)
    assert found["said_words"] == words.HISTORY_WITHHELD
    assert found["technical"] == "Technical detail: recorded for 2026-10-21"


def test_records_made_after_the_request_time_are_not_shown(lab):
    monday = made(lab, T, MONDAY)
    made(lab, T, TUESDAY)
    score(lab, monday, at=ny(TUESDAY, "20:40"))
    before = view(lab, at(TUESDAY, "20:00"), tickers=[T])
    assert [e["as_of"] for e in before["evenings"]] == [MONDAY]
    assert item(before, MONDAY, T)["state"] == "pending"
    after = view(lab, at(TUESDAY, "20:45"), tickers=[T])
    assert [e["as_of"] for e in after["evenings"]] == [TUESDAY, MONDAY]
    assert item(after, MONDAY, T)["state"] == "scored"


@pytest.mark.parametrize(
    "reason", ["ledger_unverified", "record_check_failed", "record_unreadable"]
)
def test_a_problem_shows_nothing_rather_than_a_partial_list(lab, reason):
    found = checked(
        history.history_view(
            History("2026-10-20T14:00:00.000000Z", "real", reason, None, 0, 0, (), (), (T,)),
            now=at(TUESDAY, "10:00"),
            local_zone=NEW_YORK,
        )
    )
    assert found["problem"] == {
        "title": words.HISTORY_PROBLEM_TITLE,
        "words": words.HISTORY_PROBLEMS[reason],
        "technical": f"Technical detail: {reason}",
    }
    assert found["evenings"] == [] and found["counts"] is None and found["filters"] == []
    assert found["empty"] is None


def test_a_failed_ledger_check_withholds_the_history(lab, monkeypatch):
    made(lab, T, MONDAY)
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    found = view(lab, at(TUESDAY, "10:00"))
    assert found["problem"]["words"] == words.HISTORY_PROBLEMS["ledger_unverified"]
    assert found["evenings"] == []


def test_no_outlook_yet_is_an_honest_empty_state(lab):
    found = view(lab, at(TUESDAY, "10:00"))
    assert found["problem"] is None and found["evenings"] == []
    assert found["empty"] == words.HISTORY_EMPTY and found["counts"] is None


def test_the_page_shows_the_latest_evenings_and_counts_them_all(lab, monkeypatch):
    monkeypatch.setattr(history, "SHOWN", 2)
    for day in (MONDAY, TUESDAY, WEDNESDAY):
        made(lab, T, day)
    found = view(lab, at(THURSDAY, "10:00"), tickers=[T])
    assert [e["as_of"] for e in found["evenings"]] == [WEDNESDAY, TUESDAY]
    assert found["more"] == words.history_more(2)
    assert found["counts"] == "Since Oct 19: 3 evenings recorded, 0 with a result, 0 missed."


def test_the_zone_is_named_when_this_mac_is_elsewhere(lab):
    made(lab, T, MONDAY)
    found = view(lab, at(TUESDAY, "10:00"), tickers=[T], local_zone=ZoneInfo("Europe/London"))
    assert found["evenings"][0]["sub"].startswith("Outlook recorded last night, 8:31 PM ET")


def test_the_judge_sheet_and_the_demo_ribbon(lab):
    found = view(lab, at(TUESDAY, "10:00"))
    assert found["note"]["button"] == words.JUDGE
    assert found["note"]["sheet"]["paragraphs"][-1].startswith("Technical detail: each result")
    assert found["demo"] is None
    lab.forecast("DEMO_CALM", MONDAY, created_at=ny(MONDAY, "20:31"))
    demo = checked(
        history.build_history(
            ReadOnlyRecords(lab.path),
            now=at(TUESDAY, "10:00"),
            calendar=nyse_sessions,
            data_class="synthetic",
            local_zone=NEW_YORK,
            tickers=["DEMO_CALM"],
        )
    )
    assert demo["demo"] == words.DEMO and item(demo, MONDAY, "DEMO_CALM")["state"] == "pending"
