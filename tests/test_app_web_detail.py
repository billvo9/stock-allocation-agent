"""
The Lab app's Stock detail view model (docs/product/lab_app_design.md, "Stock
detail in detail"): the chart exactly as the recent-prices reader gave it
(decision A6), the next session's outlook and the five-session total exactly
as recorded, how reliable the model has been (the development record as
stored, the live record's progress), context chips only from the forecast
record, and an honest state for everything missing. Fixtures only: a Stock Lab
store on a simulated clock (real-class records use the made-up ticker SYNTH)
and a stand-in for the recent-prices reader. No network, no private store.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, build, ny
from test_app_destinations import scan_for_advice

from stock_agent.app.web import detail, words
from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.store import StoreRefused

NEW_YORK = ZoneInfo("America/New_York")
T = REAL_TEST_TICKER
MONDAY, TUESDAY, WEDNESDAY, THURSDAY = "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"
FIVE = ["2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28"]
# Decrease, Stable, Increase (the helper's order), with more digits than any display, so a
# rounded or renormalized value could not pass for the recorded one.
P = (0.42137, 0.10931, 0.46932)
DAYS = [
    P,
    (0.40011, 0.12013, 0.47976),
    (0.41203, 0.09871, 0.48926),
    (0.43307, 0.11119, 0.45574),
    (0.39983, 0.13037, 0.4698),
]
TOTAL = (0.35113, 0.20117, 0.4477)
SIGNALS = {
    "current_run": {"direction": "increase", "length": 3},
    "market_state": {
        "ret_5": 0.0123,
        "ret_20": 0.04213,
        "vol_20": 0.0211,
        "ewma_vol": 0.0207,
        "regime": "calm",
    },
}
RUN = "R-dev"
DEV_STOCKS = [T, "SYNTQ", "SYNTR", "SYNTS"]


def at(day: str, hhmm: str) -> datetime:
    return ny(day, hhmm).tz_convert(NEW_YORK).to_pydatetime()


def row(session, close, change, *, level=None, notes=()):
    return SimpleNamespace(
        session=session,
        close=close,
        level=close if level is None else level,
        change=change,
        notes=tuple(notes),
    )


ROWS = [
    row(MONDAY, 120.0, 0.0101),
    row(TUESDAY, 121.70, -0.0032, notes=("ex-dividend",)),
    row(WEDNESDAY, 127.65, 0.0488),
]


class Prices:
    """A stand-in for the recent-prices reader: what it was asked, and fixed answers."""

    def __init__(self, rows=None, state="observed", reason=None):
        self.calls = []
        self.rows = ROWS if rows is None else rows
        self.state, self.reason = state, reason

    def __call__(self, symbol, *, count, retrieved_by):
        self.calls.append((symbol, count, retrieved_by))
        observed = self.state == "observed"
        sessions = tuple(self.rows) if observed else ()
        return SimpleNamespace(
            ticker=symbol,
            state=self.state,
            reason=self.reason,
            sessions=sessions,
            last_session=sessions[-1].session if sessions else None,
        )


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


def evening(lab, as_of, *, p=DAYS, ran=None, **forecast):
    """The evening job as of `as_of`: its forecast at 20:31 and its run at 20:32."""

    forecast.setdefault("cumulative", TOTAL)
    forecast.setdefault("signals", SIGNALS)
    record = lab.forecast(T, as_of, created_at=ny(as_of, "20:31"), p=p, **forecast)
    lab.job(at=ran or ny(as_of, "20:32"), expected=as_of, status="ok", activation=None)
    return record


def evaluation(
    *,
    increase=(0.4, 0.5, 0.47, 0.45, 0.49, 4697, 1310),
    data_class="real",
    flat_dates=1311,
    more=(),
):
    """A stored development evaluation: one reliability bucket per category, and `more`."""

    low, high, observed, ci_low, ci_high, n, dates = increase
    buckets = [
        (2, low, high, observed, ci_low, ci_high, n, dates),
        (1, 0.1, 0.2, 0.11, 0.1, 0.12, 5120, flat_dates),
        (0, 0.4, 0.5, 0.42, 0.4, 0.44, 4410, 1308),
        *more,
    ]
    return {
        "run_id": RUN,
        "created_at": "2026-01-02T21:00:00.000000Z",
        "data_class": data_class,
        "stocks": DEV_STOCKS,
        "config": {"outcome_end": "2024-12-31"},
        "metrics": [
            {
                "model": "B1",
                "target": "target_1",
                "n": 20000,
                "dates": 1400,
                "rps": 0.2,
                "brier": 0.6,
                "log_loss": 1.0,
            }
        ],
        "reliability": [
            {
                "model": "B1",
                "target": "target_1",
                "category": code,
                "bin_low": lo,
                "bin_high": hi,
                "n": n,
                "dates": d,
                "mean_predicted": (lo + hi) / 2,
                "observed": obs,
                "ci_low": cl,
                "ci_high": ch,
            }
            for code, lo, hi, obs, cl, ch, n, d in buckets
        ],
        "comparisons": [],
        "selection": {"selected": "B1", "rule": "keep B1 unless ..."},
    }


def numbers(value) -> list[float]:
    """Every number in a view, at any depth."""

    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [value]
    if isinstance(value, dict):
        return [n for item in value.values() for n in numbers(item)]
    if isinstance(value, list):
        return [n for item in value for n in numbers(item)]
    return []


def checked(found: dict | None) -> dict | None:
    """Every view, in every test: strict JSON, no advice words, outlooks with no colour."""

    if found is None:
        return None
    text = json.dumps(found, allow_nan=False, ensure_ascii=False)
    scan_for_advice(text)
    for part in ("outlook", "five"):
        assert "direction" not in json.dumps(found[part] or {}), part  # never green or red
    return found


def view(lab, now, *, prices=None, records=None, tickers=(T,), symbol=T, **options):
    return checked(
        detail.build_detail(
            records or ReadOnlyRecords(lab.path),
            symbol=symbol,
            now=now,
            calendar=nyse_sessions,
            calendar_id="test_nyse_rules",
            data_class=options.pop("data_class", "real"),
            recent=prices or Prices(),
            local_zone=NEW_YORK,
            tickers=None if tickers is None else list(tickers),
            **options,
        )
    )


# --- the outlook and the five-session total: exactly as recorded ---


def test_a_fresh_outlook_shows_the_recorded_probabilities_exactly(lab):
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "07:40"))
    stored = ReadOnlyRecords(lab.path).forecasts(T)[0]["daily"][0]
    outlook = found["outlook"]
    assert (outlook["state"], outlook["level"]) == ("fresh", "plain")
    assert outlook["bar"]["recorded"] == {
        "increase": stored["p_increase"],
        "stable": stored["p_stable"],
        "decrease": stored["p_decrease"],
    }
    assert outlook["bar"]["recorded"]["increase"] == 0.46932
    assert outlook["bar"]["legend"] == [["↑ Up", "47%"], ["→ Flat", "11%"], ["↓ Down", "42%"]]
    assert outlook["bar"]["label"] == "Up 47%, Flat 11%, Down 42%"
    assert outlook["title"] == "Thursday's outlook"
    assert outlook["sub"] == "For Thu, Oct 22. Recorded last night, 8:31 PM, after the close."
    assert outlook["definition"] == (
        f"Up means {T} closes more than 0.5% above Wednesday's close, and Down more than 0.5% "
        "below it. Flat means it ends within 0.5% either way."
    )
    assert outlook["pill"]["words"] == "Experimental"


def test_the_five_session_total_and_each_day_are_exactly_as_recorded(lab):
    record = evening(lab, WEDNESDAY)
    five = view(lab, at(THURSDAY, "07:40"))["five"]
    total = record["cumulative"]
    assert five["bar"]["recorded"] == {
        "increase": total["p_increase"],
        "stable": total["p_stable"],
        "decrease": total["p_decrease"],
    }
    assert five["bar"]["recorded"]["increase"] == 0.4477
    assert (
        five["sub"] == f"Where {T} closes on Wed, Oct 28, compared with the close on Wed, Oct 21."
    )
    # The record's own five-session band (0.5% x sqrt 5), in words.
    assert five["definition"].startswith("Over five days the line between Up and Flat is 1.12%,")
    assert [day["day"] for day in five["days"]] == [
        "Thu 22",
        "Fri 23",
        "Mon 26",
        "Tue 27",
        "Wed 28",
    ]
    for day, stored in zip(five["days"], record["daily"], strict=True):
        assert day["bar"]["recorded"] == {
            "increase": stored["p_increase"],
            "stable": stored["p_stable"],
            "decrease": stored["p_decrease"],
        }
        assert day["up"] == words.percent(stored["p_increase"])
        assert day["down"] == words.percent(stored["p_decrease"])
    assert five["days"][2]["label"] == "Monday, Oct 26: Up 49%, Flat 10%, Down 41%."
    assert five["toggle"] == ["Each day on its own", "Hide each day"]
    assert "not a path" in five["days_note"] and "same 0.5% line" in five["days_note"]


def test_the_bands_come_from_the_record(lab):
    evening(lab, WEDNESDAY, band=0.0025)
    found = view(lab, at(THURSDAY, "07:40"))
    assert "more than 0.25% above" in found["outlook"]["definition"]
    assert "Up and Flat is 0.56%," in found["five"]["definition"]  # 0.25% x sqrt 5
    assert "same 0.25% line" in found["five"]["days_note"]


def test_the_five_session_band_is_read_from_the_record_not_recomputed(lab):
    # The fixture stores band x sqrt 5; a stored value off that rule must show as stored.
    evening(lab, WEDNESDAY)
    briefing = build(lab, ny(THURSDAY, "07:40"), data_class="real", tickers=[T])
    (stock,) = briefing.stocks
    lineage = dataclasses.replace(stock.lineage, cumulative_band=0.0099)
    edited = dataclasses.replace(briefing, stocks=(dataclasses.replace(stock, lineage=lineage),))
    found = checked(
        detail.detail_view(
            edited,
            Prices()(T, count=252, retrieved_by=None),
            symbol=T,
            now=at(THURSDAY, "07:40"),
            local_zone=NEW_YORK,
            data_class="real",
        )
    )
    assert "Up and Flat is 0.99%," in found["five"]["definition"]


def test_an_outlook_without_a_recorded_total_says_so(lab):
    # The fixture store records a total with every outlook, so the briefing is edited.
    evening(lab, WEDNESDAY)
    briefing = build(lab, ny(THURSDAY, "07:40"), data_class="real", tickers=[T])
    (stock,) = briefing.stocks
    briefing = dataclasses.replace(briefing, stocks=(dataclasses.replace(stock, cumulative=None),))
    found = detail.detail_view(
        briefing,
        Prices()(T, count=252, retrieved_by=None),
        symbol=T,
        now=at(THURSDAY, "07:40"),
        local_zone=NEW_YORK,
        data_class="real",
    )
    five = checked(found)["five"]
    assert five["bar"] is None and five["sub"] is None and five["definition"] is None
    assert five["words"] == "No five-day total was recorded with this outlook."
    assert len(five["days"]) == 5


CHIPS = ["run", "momentum", "regime"]


@pytest.mark.parametrize(
    ("record", "now", "state", "level", "title", "lately"),
    [
        ({}, (THURSDAY, "07:40"), "none", "neutral", "No outlook yet", words.NOTHING_LATELY),
        (
            {"refusal": {"reason": "too_few_sessions", "detail": "too few sessions"}},
            (THURSDAY, "07:40"),
            "refused",
            "neutral",
            "The model gave no outlook",
            CHIPS,  # a refusal still carries what it recorded about the past
        ),
        (
            {"targets": ["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"]},
            (THURSDAY, "07:40"),
            "withheld",
            "warn",
            "Outlook withheld",
            words.NOTHING_SHOWN,  # a withheld record's context is withheld too
        ),
        (
            {"as_of": TUESDAY},
            (THURSDAY, "07:40"),
            "stale",
            "warn",
            "No outlook for Thursday",
            CHIPS,
        ),
        ({}, (THURSDAY, "20:10"), "pending", "neutral", "Friday's outlook comes tonight", CHIPS),
    ],
)
def test_every_missing_outlook_has_an_honest_state(lab, record, now, state, level, title, lately):
    record = dict(record)
    as_of = record.pop("as_of", WEDNESDAY)
    if state == "none":
        lab.job(at=ny(WEDNESDAY, "20:32"), expected=WEDNESDAY, status="ok", activation=None)
    else:
        evening(lab, as_of, **record)
    found = view(lab, at(*now))
    outlook = found["outlook"]
    assert (outlook["state"], outlook["level"], outlook["title"]) == (state, level, title)
    assert "bar" not in outlook and found["five"] is None
    assert outlook["words"] == outlook["sheet"]["paragraphs"][0]
    assert outlook["sheet"]["paragraphs"][-1].startswith("Technical detail: ")
    assert outlook["details"] == "Details"
    assert found["reliable"]["development"] == {
        "state": "later",
        "words": "The past-test result appears next to an outlook.",
    }
    # No probability of any kind reaches a screen without a fresh outlook.
    assert "%" not in json.dumps([found["outlook"], found["reliable"]["development"]])
    if isinstance(lately, str):
        assert found["lately"]["chips"] == [] and found["lately"]["words"] == lately
    else:
        assert [chip["key"] for chip in found["lately"]["chips"]] == lately
    live = found["reliable"]["live"]
    assert (live["state"], live["flag"]) == ("none", "Not enough data yet")


def test_a_failed_ledger_check_withholds_the_outlook_and_its_context(lab, monkeypatch):
    evening(lab, WEDNESDAY)
    monkeypatch.setattr(ReadOnlyRecords, "verify_ledger", lambda self: (3, "entry 2 differs"))
    found = view(lab, at(THURSDAY, "07:40"))
    assert (found["outlook"]["state"], found["outlook"]["level"]) == ("unverified", "bad")
    assert found["five"] is None and found["lately"]["chips"] == []


def test_a_store_that_cannot_be_read_keeps_the_prices_and_withholds_everything_else(lab):
    class Refusing(ReadOnlyRecords):
        def job_runs(self):
            raise StoreRefused("not a Stock Lab store")

    found = view(lab, at(THURSDAY, "07:40"), records=Refusing(lab.path))
    assert (found["outlook"]["state"], found["outlook"]["level"]) == ("unverified", "bad")
    assert found["outlook"]["sheet"]["paragraphs"][-1] == (
        "Technical detail: the record could not be opened (StoreRefused)."
    )
    assert found["reliable"]["development"]["state"] == "unreadable"
    assert found["reliable"]["live"] is None and found["lately"]["chips"] == []
    assert found["chart"]["available"] and found["price"]["close"] == 127.65


def test_a_store_whose_watchlist_cannot_be_read_refuses_rather_than_saying_not_listed(lab):
    # As the launcher calls it (no tickers): the server then answers with its plain error.
    class Refusing(ReadOnlyRecords):
        def watched(self):
            raise StoreRefused("not a Stock Lab store")

    prices = Prices()
    with pytest.raises(StoreRefused):
        view(lab, at(THURSDAY, "07:40"), records=Refusing(lab.path), tickers=None, prices=prices)
    assert prices.calls == []


def test_a_stock_off_the_watchlist_is_not_found_and_nothing_else_is_read(lab, monkeypatch):
    lab.watch(T, "SNDK", at=ny(MONDAY, "12:00"))

    def never(*args, **kwargs):
        raise AssertionError("the briefing was prepared for a stock that is not listed")

    monkeypatch.setattr(detail, "build_briefing", never)
    prices = Prices()
    now = at(THURSDAY, "07:40")
    assert view(lab, now, symbol="NVDA", prices=prices) is None
    assert view(lab, now, symbol="NVDA", prices=prices, tickers=None) is None
    assert view(lab, now, symbol="SNDK", prices=prices, tickers=None, excluded=["SNDK"]) is None
    assert prices.calls == []
    monkeypatch.undo()
    found = view(lab, at(THURSDAY, "07:40"), tickers=None, excluded=["SNDK"])
    assert found["ticker"] == T and found["title"] == T  # SYNTH has no company name


# --- the chart: the reader's sessions as it gave them (decision A6) ---


def test_the_chart_is_the_readers_sessions_unchanged(lab):
    prices = Prices()
    now = at(THURSDAY, "07:40")
    chart = view(lab, now, prices=prices)["chart"]
    assert prices.calls == [(T, 252, pd.Timestamp(now))]
    assert chart["available"] and chart["latest"] == 2
    assert [p["session"] for p in chart["points"]] == [MONDAY, TUESDAY, WEDNESDAY]
    assert [p["level"] for p in chart["points"]] == [120.0, 121.70, 127.65]
    assert chart["points"][1] == {
        "session": TUESDAY,
        "level": 121.70,
        "close_words": "$121.70",
        "change_words": "−0.32%",
        "direction": "down",
        "day": "Tue, Oct 20",
        "when": "on Tue, Oct 20 · ex-dividend day",
        "label": "Tue, Oct 20: closed at $121.70, −0.32% that day; ex-dividend day.",
    }
    assert chart["points"][2]["when"] == "on Wed, Oct 21 (latest close)"
    assert chart["label"].startswith(f"Price chart of {T}'s daily closes.")
    assert chart["why"]["words"] == "Why no total for the period?"
    assert "adds nothing up" in " ".join(chart["why"]["sheet"]["paragraphs"])


POINT_KEYS = {
    "session",
    "level",
    "close_words",
    "change_words",
    "direction",
    "day",
    "when",
    "label",
}


def test_nothing_is_derived_from_the_prices(lab):
    # Sentinels that no sum, difference, ratio, high, low or average of the others can
    # equal, with the latest close neither the highest nor the lowest: a value the view
    # computed from the prices would show as a number, a dollar amount or a percentage.
    rows = [
        row(MONDAY, 103.37, 0.01013, level=57.29, notes=("split",)),
        row(TUESDAY, 211.53, -0.02027, level=61.71),
        row(WEDNESDAY, 158.91, 0.03037, level=53.13),
    ]
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "07:40"), prices=Prices(rows))
    # The shape is pinned, so no extra field (a total, a low) can slip in.
    assert set(found) == {
        "schema",
        "version",
        "data_class",
        "generated_at",
        "ticker",
        "name",
        "title",
        "price",
        "chart",
        "outlook",
        "five",
        "reliable",
        "lately",
        "learning",
        "demo",
    }
    assert set(found["chart"]) == {
        "available",
        "words",
        "label",
        "points",
        "latest",
        "ranges",
        "default",
        "drag",
        "why",
    }
    assert all(set(point) == POINT_KEYS for point in found["chart"]["points"])
    assert all(
        set(r) == {"key", "label", "name", "first", "note"} for r in found["chart"]["ranges"]
    )
    # The counts, by their paths.
    assert found["version"] == 1 and found["chart"]["latest"] == 2
    assert [r["first"] for r in found["chart"]["ranges"]] == [0, 0, 0, 0]
    assert found["reliable"]["live"]["meter"] == {
        "value": 0,
        "max": 50,
        "label": "0 of 50 trading days",
    }
    # Every other number is a served value or a recorded probability.
    rest = json.loads(json.dumps(found))
    del rest["version"], rest["chart"]["latest"], rest["chart"]["ranges"]
    del rest["reliable"]["live"]["meter"]
    served = {57.29, 61.71, 53.13, 158.91, 0.03037}  # the levels; the latest close and change
    recorded = {value for triple in [*DAYS, TOTAL] for value in triple}
    assert set(numbers(rest)) <= served | recorded
    assert [p["level"] for p in found["chart"]["points"]] == [57.29, 61.71, 53.13]
    # Every amount and percentage in the words is a served or recorded one.
    text = json.dumps(found, ensure_ascii=False)
    assert set(re.findall(r"\$[\d,]+\.\d\d", text)) == {"$103.37", "$211.53", "$158.91"}
    allowed = (
        {words.signed_percent(change) for change in (0.01013, -0.02027, 0.03037)}
        | {words.percent(value) for value in recorded}
        | {"0.5%", "1.12%", "+4.21%"}  # the recorded bands and 4-week change
        | {"47%", "100%"}  # the judging sheet's example and "add up to exactly 100%"
    )
    assert set(re.findall(r"[+\u2212]?\d+(?:\.\d+)?%", text)) <= allowed
    assert [p["change_words"] for p in found["chart"]["points"]] == [
        "+1.01%",
        "\u22122.03%",
        "+3.04%",
    ]


def test_the_ranges_show_the_latest_sessions(lab):
    days = pd.bdate_range(end=WEDNESDAY, periods=252)
    rows = [row(d.strftime("%Y-%m-%d"), 100.0 + i, 0.001) for i, d in enumerate(days)]
    chart = view(lab, at(THURSDAY, "07:40"), prices=Prices(rows))["chart"]
    assert [(r["key"], r["first"]) for r in chart["ranges"]] == [
        ("1W", 246),
        ("1M", 230),
        ("3M", 188),
        ("1Y", 0),
    ]
    assert chart["ranges"][0]["note"] == "Daily closes from Wed, Oct 14 to Wed, Oct 21."
    assert chart["default"] == "3M" and len(chart["points"]) == 252
    short = view(lab, at(THURSDAY, "07:40"))["chart"]
    assert {r["first"] for r in short["ranges"]} == {0}
    assert (
        short["ranges"][3]["note"] == "Only 3 trading days are stored: Mon, Oct 19 to Wed, Oct 21."
    )


def test_a_gap_stays_a_gap(lab):
    rows = [*ROWS, row("2026-10-22", None, None, level=None, notes=("gap",))]
    chart = view(lab, at("2026-10-23", "07:40"), prices=Prices(rows))["chart"]
    gap = chart["points"][-1]
    assert gap["level"] is None and gap["close_words"] == "—" and gap["change_words"] == "—"
    assert gap["direction"] == "none" and gap["when"] == "on Thu, Oct 22 · no price served"
    assert gap["label"] == "Thu, Oct 22: no close; no price served."
    assert chart["latest"] == 2  # the latest served close, not the gap


@pytest.mark.parametrize(
    ("state", "text"),
    [
        ("no_store", "No prices stored yet."),
        ("no_data", f"No prices stored for {T} yet."),
        ("unreadable", "Prices can't be read right now."),
    ],
)
def test_missing_prices_say_why_and_show_no_number(lab, state, text):
    found = view(lab, at(THURSDAY, "07:40"), prices=Prices(state=state, reason="x"))
    assert found["chart"] == {
        "available": False,
        "label": None,
        "points": [],
        "latest": None,
        "ranges": [],
        "words": text,
    }
    assert found["price"]["available"] is False and found["price"]["close"] is None


# --- how reliable: the development record as stored, the live record's progress ---


def test_the_development_record_is_read_as_stored(lab):
    lab.store.record_evaluation(evaluation())
    evening(lab, WEDNESDAY, evaluation_run_id=RUN)
    development = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]
    assert development["state"] == "ok"
    assert development["headline"] == (
        "When the model said about 40–50% for Up, the stock went up 47% of the time."
    )
    assert development["meta"] == (
        f"Past tests on {T}, SYNTQ, SYNTR and SYNTS together, through 2024. 4,697 forecasts on "
        "1,310 dates. Likely range 45–49%."
    )
    sheet = development["sheet"]
    assert sheet["columns"] == ["When it said", "It happened"]
    assert sheet["rows"] == [
        ["↑ Up about 40–50%", "47% (likely 45–49%)"],
        ["→ Flat about 10–20%", "11% (likely 10–12%)"],
        ["↓ Down about 40–50%", "42% (likely 40–44%)"],
    ]
    assert "picked knowing they had done well" in " ".join(sheet["paragraphs"])
    assert sheet["paragraphs"][-1].startswith("Technical detail: Development test on pre-2025")


def test_a_bucket_with_too_few_dates_shows_no_frequency(lab):
    lab.store.record_evaluation(evaluation(increase=(0.4, 0.5, 0.47, 0.3, 0.6, 40, 12)))
    evening(lab, WEDNESDAY, evaluation_run_id=RUN)
    development = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]
    assert development["state"] == "insufficient" and development["flag"] == "Not enough data yet"
    assert development["words"].startswith("Too few past tests gave about 40–50% for Up")
    assert "47%" not in json.dumps(development, ensure_ascii=False)


def test_an_evaluation_stored_after_the_request_time_is_not_read(lab):
    later = {**evaluation(), "created_at": "2026-10-22T12:00:00.000000Z"}
    lab.store.record_evaluation(later)
    evening(lab, WEDNESDAY, evaluation_run_id=RUN)
    development = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]
    assert development == {
        "state": "none",
        "words": "No past-test result is stored for this outlook.",
    }
    # Once it exists, it is read (08:00 New York is when it was stored).
    assert view(lab, at(THURSDAY, "08:05"))["reliable"]["development"]["state"] == "ok"


def test_the_live_record_counts_toward_model_healths_minimum(lab):
    found = view(lab, at(THURSDAY, "07:40"))["reliable"]["live"]
    assert found["state"] == "none" and found["flag"] == "Not enough data yet"
    assert found["meter"] == {"value": 0, "max": 50, "label": "0 of 50 trading days"}
    assert found["words"].startswith("Live record: no scored days yet.")
    tuesday = evening(lab, TUESDAY)
    lab.score(tuesday, "1", 2, at=ny(WEDNESDAY, "20:10"))
    evening(lab, WEDNESDAY)
    found = view(lab, at(THURSDAY, "07:40"))["reliable"]["live"]
    assert found["state"] == "insufficient" and found["meter"]["value"] == 1
    assert found["words"] == (
        "Live record: 1 of the 50 trading days Model Health needs before it judges the outlooks."
    )
    assert found["sheet"]["title"] == "How to judge an outlook"


def test_the_live_record_words_past_the_minimum():
    def said(status, sessions, targets):
        return words.live_words(status, sessions, targets, sessions_needed=50, targets_needed=250)

    assert said("ok", 63, 1260) == (
        "Live record: 63 trading days scored, past Model Health's minimum of 50. The research "
        "page shows how it has done."
    )
    assert said("insufficient", 52, 200) == (
        "Live record: 52 trading days scored, but only 200 of the 250 scored outlooks Model "
        "Health needs."
    )


# --- lately: only what the evening job recorded with the forecast ---


def test_the_context_chips_come_from_the_forecast_record(lab):
    evening(lab, WEDNESDAY)
    lately = view(lab, at(THURSDAY, "07:40"))["lately"]
    assert lately["sub"] == (
        "As of the close on Wed, Oct 21, recorded with that outlook. Facts about the past, not "
        "predictions."
    )
    assert [(c["key"], c["words"]) for c in lately["chips"]] == [
        ("run", "3 rising days in a row"),
        ("momentum", "Past 4 weeks +4.21%"),
        ("regime", "Calm swings"),
    ]
    assert "more than 0.5% higher" in lately["chips"][0]["sheet"]["paragraphs"][0]


def test_a_stale_outlooks_context_names_its_own_close(lab):
    evening(lab, TUESDAY, signals={"current_run": {"direction": "decrease", "length": 1}})
    lately = view(lab, at(THURSDAY, "07:40"))["lately"]
    assert lately["sub"].startswith("As of the close on Tue, Oct 20,")
    assert [c["words"] for c in lately["chips"]] == ["1 falling day"]  # nothing else recorded


def test_a_context_the_briefing_computed_from_prices_is_never_shown(lab):
    # A synthetic stock's context is computed by the briefing from stored prices; the
    # detail shows only what the forecast record holds (here, the run).
    from stock_lab_inference_helpers import make_bars

    symbol = "DEMO_CALM"
    lab.watch(symbol, at=ny(MONDAY, "12:00"))
    lab.prices(symbol, make_bars("2025-10-01", WEDNESDAY), at=ny(WEDNESDAY, "20:00"))
    lab.forecast(symbol, WEDNESDAY, created_at=ny(WEDNESDAY, "20:31"), signals=SIGNALS)
    found = view(
        lab, at(THURSDAY, "07:40"), tickers=(symbol,), symbol=symbol, data_class="synthetic"
    )
    briefing = build(lab, ny(THURSDAY, "07:40"), tickers=[symbol])
    assert briefing.stocks[0].context.available  # computed from prices: not the record's
    assert [c["key"] for c in found["lately"]["chips"]] == ["run"]
    assert found["demo"] == words.DEMO


def test_without_a_record_there_is_no_context(lab):
    lately = view(lab, at(THURSDAY, "07:40"))["lately"]
    assert lately == {
        "title": "Lately",
        "sub": None,
        "chips": [],
        "words": "Nothing is recorded for this stock yet.",
    }


# --- point in time: only what existed at the request time ---


def test_records_made_after_the_request_time_are_ignored(lab):
    first = evening(lab, TUESDAY)
    wednesday = evening(lab, WEDNESDAY)
    now = at(THURSDAY, "07:40")
    prices = Prices()
    before = view(lab, now, prices=prices)
    # After `now`: a recalculation of Wednesday's outlook, a score, tonight's outlook and run.
    lab.forecast(
        T,
        WEDNESDAY,
        created_at=ny(THURSDAY, "07:45"),
        p=(0.1, 0.1, 0.8),
        signals={"current_run": {"direction": "stable", "length": 9}},
        supersedes=wednesday["forecast_id"],
    )
    lab.score(first, "1", 2, at=ny(THURSDAY, "07:50"))
    # Tonight's outlook is made at 20:31 but recorded only at 20:35; its run ends at 20:36.
    lab.forecast(
        T,
        THURSDAY,
        created_at=ny(THURSDAY, "20:31"),
        recorded_at=ny(THURSDAY, "20:35"),
        p=DAYS,
        cumulative=TOTAL,
        signals=SIGNALS,
    )
    lab.job(at=ny(THURSDAY, "20:36"), expected=THURSDAY, status="ok", activation=None)
    assert view(lab, now, prices=prices) == before
    assert before["outlook"]["bar"]["recorded"]["increase"] == 0.46932
    assert before["lately"]["chips"][0]["words"] == "3 rising days in a row"
    assert before["reliable"]["live"]["state"] == "none"
    assert {when for _, _, when in prices.calls} == {pd.Timestamp(now)}  # the reader's cutoff
    # Each of them shows once it existed, so the test above could have failed.
    after = view(lab, at(THURSDAY, "07:55"))
    assert after["outlook"]["bar"]["recorded"]["increase"] == 0.8
    assert after["lately"]["chips"][0]["words"] == "9 flat days in a row"
    assert after["reliable"]["live"]["state"] == "insufficient"
    made_not_recorded = view(lab, at(THURSDAY, "20:33"))["outlook"]
    assert made_not_recorded["state"] == "pending" and "bar" not in made_not_recorded
    assert view(lab, at(THURSDAY, "21:00"))["outlook"]["title"] == "Friday's outlook"


def test_the_real_recent_prices_reader_fits_the_chart(lab, tmp_path):
    # The stand-in above must keep the real reader's shape: one run through the real one.
    from functools import partial

    from stock_lab_inference_helpers import AS_OF, next_day, open_store, record
    from stock_lab_inference_helpers import ny as ny_utc

    from stock_agent.stock_lab.inference.recent import read_recent

    store, clock = open_store(tmp_path / "inference")
    record(store, clock, "2025-07-01", AS_OF, retrieved_at=ny_utc(AS_OF, "20:30"))
    now = ny_utc(next_day(AS_OF), "07:40").to_pydatetime()
    found = checked(
        detail.build_detail(
            ReadOnlyRecords(lab.path),
            symbol="NVDA",
            now=now,
            calendar=nyse_sessions,
            calendar_id="test_nyse_rules",
            data_class="real",
            recent=partial(read_recent, path=store.path),
            local_zone=NEW_YORK,
            tickers=["NVDA"],
        )
    )
    real = read_recent("NVDA", path=store.path, count=detail.CHART_SESSIONS)
    chart = found["chart"]
    assert len(chart["points"]) == len(real.sessions) == 252
    assert [p["level"] for p in chart["points"]] == [s.level for s in real.sessions]
    assert chart["points"][-1]["close_words"] == words.money(real.sessions[-1].close)
    assert found["title"] == "NVDA · NVIDIA"


# --- the research page ---


def test_the_research_link_opens_the_stock_probability_page(lab):
    learning = view(lab, at(THURSDAY, "07:40"), research_url="http://127.0.0.1:8502")["learning"]
    assert learning["url"] == "http://127.0.0.1:8502/stock-probability"
    assert learning["words"] == f"Open {T} in the research view"
    assert learning["sub"].endswith(f"Choose {T} there.")
    alone = view(lab, at(THURSDAY, "07:40"))["learning"]
    assert alone["url"] is None and alone["sub"] == words.RESEARCH_ELSEWHERE


def test_the_research_page_path_is_the_research_apps_own():
    from stock_agent.app import wording

    assert wording.NAV["stock_lab"][1] == words.RESEARCH_PAGE


# --- review additions: the buckets, the live record, the reader's cutoff, odd values ---


def test_the_bucket_holding_the_recorded_chance_is_the_one_shown_as_stored(lab):
    neighbours = (
        (2, 0.3, 0.4, 0.33, 0.30, 0.36, 900, 800),
        (2, 0.5, 0.6, 0.55, 0.52, 0.58, 700, 650),
    )
    # A stored range no fresh calculation would give: it is shown as stored.
    stored = evaluation(increase=(0.4, 0.5, 0.47, 0.31, 0.62, 4697, 1310), more=neighbours)
    lab.store.record_evaluation(stored)
    evening(lab, WEDNESDAY, evaluation_run_id=RUN)
    development = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]
    assert development["headline"].startswith("When the model said about 40–50% for Up")
    assert development["meta"].endswith("Likely range 31–62%.")


def test_a_chance_on_a_bucket_edge_falls_in_the_upper_bucket(lab):
    neighbours = ((2, 0.5, 0.6, 0.55, 0.52, 0.58, 700, 650),)
    lab.store.record_evaluation(evaluation(more=neighbours))
    evening(lab, WEDNESDAY, evaluation_run_id=RUN, p=(0.4, 0.1, 0.5))
    development = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]
    assert development["headline"] == (
        "When the model said about 50–60% for Up, the stock went up 55% of the time."
    )


def test_a_flat_bucket_with_too_few_dates_is_named_without_a_frequency(lab):
    lab.store.record_evaluation(evaluation(flat_dates=12))
    evening(lab, WEDNESDAY, evaluation_run_id=RUN)
    rows = view(lab, at(THURSDAY, "07:40"))["reliable"]["development"]["sheet"]["rows"]
    assert rows[1] == ["→ Flat about 10–20%", "Not enough data yet"]


OTHER = "SYNTQ"  # a second made-up real-class ticker


def test_the_live_record_counts_every_stocks_scored_outlooks(lab):
    other = lab.forecast(OTHER, MONDAY, created_at=ny(MONDAY, "20:31"), p=DAYS, cumulative=TOTAL)
    lab.score(other, "1", 0, at=ny(TUESDAY, "20:05"))
    tuesday = evening(lab, TUESDAY)
    lab.score(tuesday, "1", 2, at=ny(WEDNESDAY, "20:10"))
    evening(lab, WEDNESDAY)
    live = view(lab, at(THURSDAY, "07:40"))["reliable"]["live"]  # this stock's screen only
    assert live["meter"]["value"] == 2  # Tuesday (the other stock) and Wednesday (this one)


@pytest.mark.parametrize(
    ("status", "sessions", "targets", "flag", "lead"),
    [
        ("ok", 63, 1260, None, "Live record: 63 trading days scored, past Model Health's"),
        (
            "insufficient",
            52,
            200,
            "Not enough data yet",
            "Live record: 52 trading days scored, but",
        ),
    ],
)
def test_the_live_record_past_fifty_sessions(lab, status, sessions, targets, flag, lead):
    evening(lab, WEDNESDAY)
    briefing = build(lab, ny(THURSDAY, "07:40"), data_class="real", tickers=[T])
    prospective = dataclasses.replace(
        briefing.reliability.prospective,
        status=status,
        settled_sessions=sessions,
        scored_targets=targets,
    )
    reliability = dataclasses.replace(briefing.reliability, prospective=prospective)
    found = checked(
        detail.detail_view(
            dataclasses.replace(briefing, reliability=reliability),
            Prices()(T, count=252, retrieved_by=None),
            symbol=T,
            now=at(THURSDAY, "07:40"),
            local_zone=NEW_YORK,
            data_class="real",
        )
    )
    live = found["reliable"]["live"]
    assert (live["state"], live["flag"]) == (status, flag)
    assert live["meter"] == {"value": 50, "max": 50, "label": "50 of 50 trading days"}
    assert live["words"].startswith(lead)


def test_a_price_batch_retrieved_after_the_request_time_is_not_charted(lab, tmp_path):
    from functools import partial

    from stock_lab_inference_helpers import AS_OF, next_day, open_store, record, steady_bars
    from stock_lab_inference_helpers import ny as ny_utc

    from stock_agent.stock_lab.inference.recent import read_recent

    store, clock = open_store(tmp_path / "inference")
    friday = next_day(AS_OF)
    for last in (AS_OF, friday):  # the window fetched again the next evening
        bars = steady_bars("2025-07-01", last)
        record(store, clock, "2025-07-01", last, retrieved_at=ny_utc(last, "20:30"), bars=bars)

    def newest(when):
        found = detail.build_detail(
            ReadOnlyRecords(lab.path),
            symbol="NVDA",
            now=when.to_pydatetime(),
            calendar=nyse_sessions,
            calendar_id="test_nyse_rules",
            data_class="real",
            recent=partial(read_recent, path=store.path),
            local_zone=NEW_YORK,
            tickers=["NVDA"],
        )
        return found["chart"]["points"][-1]["session"]

    assert newest(ny_utc(friday, "07:40")) == AS_OF  # Friday's batch did not exist yet
    assert newest(ny_utc(next_day(friday), "07:40")) == friday


def test_a_range_always_reaches_back_from_the_latest_close(lab):
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end="2026-10-30", periods=12)]
    rows = [row(day, 100.0 + i, 0.001) for i, day in enumerate(days[:5])]
    rows += [row(day, None, None, level=None, notes=("gap",)) for day in days[5:]]
    chart = view(lab, at("2026-11-02", "07:40"), prices=Prices(rows))["chart"]
    assert chart["latest"] == 4
    assert [r["first"] for r in chart["ranges"]] == [4, 0, 0, 0]
    assert chart["points"][4]["when"] == "on Wed, Oct 21 (latest close)"


def test_the_why_sheet_names_the_4_week_change_only_when_it_is_shown(lab):
    evening(lab, WEDNESDAY)
    shown = view(lab, at(THURSDAY, "07:40"))["chart"]["why"]["sheet"]["paragraphs"]
    assert any('"Lately"' in text for text in shown)
    lab.forecast(
        T,
        THURSDAY,
        created_at=ny(THURSDAY, "20:31"),
        p=DAYS,
        cumulative=TOTAL,
        signals={"current_run": {"direction": "increase", "length": 1}},
    )
    lab.job(at=ny(THURSDAY, "20:32"), expected=THURSDAY, status="ok", activation=None)
    found = view(lab, at("2026-10-23", "07:40"))
    assert [chip["key"] for chip in found["lately"]["chips"]] == ["run"]
    assert not any('"Lately"' in text for text in found["chart"]["why"]["sheet"]["paragraphs"])


def test_only_recorded_context_values_become_chips(lab):
    signals = {
        "current_run": {"direction": None, "length": 0},
        "market_state": {"ret_20": None, "regime": "stormy"},
    }
    evening(lab, WEDNESDAY, signals=signals)
    chips = view(lab, at(THURSDAY, "07:40"))["lately"]["chips"]
    assert [(chip["key"], chip["words"]) for chip in chips] == [("run", "No streak after a gap")]


def test_odd_reader_values_are_missing_and_unknown_notes_are_shown_as_given(lab):
    nan, inf = float("nan"), float("inf")
    rows = [row(MONDAY, 120.0, nan, level=inf, notes=("halted",)), row(TUESDAY, 121.7, -0.0032)]
    chart = view(lab, at(WEDNESDAY, "07:40"), prices=Prices(rows))["chart"]
    first = chart["points"][0]
    assert first["level"] is None and first["change_words"] == "—"
    assert first["direction"] == "none" and first["when"] == "on Mon, Oct 19 · halted"
    gaps = [row(MONDAY, None, None, level=None, notes=("gap",))]
    empty = view(lab, at(WEDNESDAY, "07:40"), prices=Prices(gaps))["chart"]
    assert (empty["available"], empty["words"]) == (False, "No closing price stored yet.")
