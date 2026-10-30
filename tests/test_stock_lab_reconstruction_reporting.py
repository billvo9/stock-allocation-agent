"""
Reconstructed forecasts in the Stock Lab's reports (owner decision of
2026-10-30): scored, but kept apart from the live record.

- The live track is unchanged by them: the briefing's stocks, reliability and
  comparison with B1 (and so Home, Detail and Health, which read it) read as
  if no reconstruction existed.
- History shows them under their missed evening, labelled "reconstructed after
  the fact (made <time>)", with their own count; the evening stays missed.
- Model Health still reports the evening missed on time, with the
  reconstruction noted.

The scenario of tests/test_stock_lab_reconstruction_evening.py; injected
clocks, no network, nothing real.
"""

from __future__ import annotations

import pandas as pd
import pytest
import test_stock_lab_evenings as evenings_tests
import test_stock_lab_reconstruction_evening as rec
from stock_lab_inference_helpers import AS_OF, CALENDAR_ID, ny, weekdays

from stock_agent.app.web import history as web_history
from stock_agent.app.web import words
from stock_agent.health.model import HealthState
from stock_agent.health.stock_lab import _evening_finding, _evening_text
from stock_agent.stock_lab import evening
from stock_agent.stock_lab.briefing.history import forecast_history
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords, _known, build_briefing
from stock_agent.stock_lab.briefing.timing import briefing_timing
from stock_agent.stock_lab.store import stamp

frozen, lab, watching, missed_friday = rec.frozen, rec.lab, rec.watching, rec.missed_friday
TICKERS, FRIDAY, MONDAY, TUESDAY = rec.TICKERS, rec.FRIDAY, rec.MONDAY, rec.TUESDAY
WEDNESDAY = rec.next_day(AS_OF, 4)
EVENINGS = evenings_tests.EVENINGS


def _score(lab, at):
    research, inference, model, clock = lab
    clock.now = at
    return evening.score_real(
        research, symbols=TICKERS, now=at, expected=pd.Timestamp(TUESDAY, tz="UTC"),
        sessions=weekdays, calendar_id=CALENDAR_ID, path=inference.path, write=True,
        model_id=model.model_id,
    )  # fmt: skip


def _scenario(lab):
    """Thursday and Monday live; Friday missed; Tuesday's prices in and the live track scored."""

    rec._live(lab, MONDAY)
    rec._batch(lab, TUESDAY)
    _score(lab, ny(TUESDAY, "20:35"))


def _briefing(lab, at):
    plan = briefing_timing(at.to_pydatetime(), weekdays)
    return build_briefing(ReadOnlyRecords(lab[0].path), timing=plan, calendar=weekdays,
                          calendar_id=CALENDAR_ID, data_class="real")  # fmt: skip


def test_the_live_track_reads_as_if_no_reconstruction_existed(missed_friday):
    _scenario(missed_friday)
    before = _briefing(missed_friday, ny(WEDNESDAY, "08:30"))
    rec._reconstruct(missed_friday, at=ny(WEDNESDAY, "08:32"), expected=TUESDAY)
    _score(missed_friday, ny(WEDNESDAY, "08:33"))  # the reconstructions scored, nothing live
    after = _briefing(missed_friday, ny(WEDNESDAY, "08:40"))
    assert any(missed_friday[0].scores(t, reconstructed=True) for t in TICKERS)
    for part in ("stocks", "reliability", "freshness", "data_as_of", "model_versions", "state",
                 "warnings", "limitations"):  # fmt: skip
        assert getattr(after, part) == getattr(before, part), part
    records = ReadOnlyRecords(missed_friday[0].path)
    cutoff = stamp(ny(WEDNESDAY, "08:40"))
    for ticker in TICKERS:
        live = _known(records, ticker, cutoff, "real")
        assert all(not f.get("reconstructed") for f in live.forecasts)
        rebuilt = _known(records, ticker, cutoff, "real", reconstructed=True)
        assert [f["as_of_session"] for f in rebuilt.forecasts] == [FRIDAY]
        assert {s["forecast_id"] for s in rebuilt.scores} <= {rebuilt.forecasts[0]["forecast_id"]}
        assert not {s["forecast_id"] for s in live.scores} & {rebuilt.forecasts[0]["forecast_id"]}


def test_history_shows_a_reconstruction_under_its_missed_evening_labelled(missed_friday):
    _scenario(missed_friday)
    made = ny(WEDNESDAY, "08:32")
    rec._reconstruct(missed_friday, at=made, expected=TUESDAY)
    _score(missed_friday, ny(WEDNESDAY, "08:33"))
    records = ReadOnlyRecords(missed_friday[0].path)
    found = forecast_history(records, now=ny(WEDNESDAY, "09:00").to_pydatetime(),
                             sessions=weekdays, data_class="real", tickers=TICKERS)  # fmt: skip
    assert found.missed == (FRIDAY,) and found.reconstructed == (FRIDAY,)
    assert found.recorded + len(found.missed) == found.due  # still missed on time
    (friday,) = [e for e in found.evenings if e.as_of == FRIDAY]
    assert friday.missed and [x.symbol for x in friday.entries] == list(TICKERS)
    assert all(x.reconstructed and x.created_at == stamp(made) for x in friday.entries)
    assert all(not x.reconstructed for e in found.evenings if not e.missed for x in e.entries)
    view = web_history.history_view(found, now=ny(WEDNESDAY, "09:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    (shown,) = [e for e in view["evenings"] if e["as_of"] == FRIDAY]
    assert shown["state"] == "missed" and len(shown["items"]) == len(TICKERS)
    assert shown["missed_words"].startswith("Missed on time. Reconstructed after the fact (made ")
    assert all(
        item["label"].startswith("Reconstructed after the fact: ") for item in shown["items"]
    )
    assert view["counts"].endswith("1 missed; 3 outlooks reconstructed after the fact.")


def test_history_before_the_reconstruction_existed_shows_the_evening_missed_only(missed_friday):
    _scenario(missed_friday)
    rec._reconstruct(missed_friday, at=ny(WEDNESDAY, "08:32"), expected=TUESDAY)
    records = ReadOnlyRecords(missed_friday[0].path)
    found = forecast_history(records, now=ny(WEDNESDAY, "08:00").to_pydatetime(),
                             sessions=weekdays, data_class="real", tickers=TICKERS)  # fmt: skip
    assert found.reconstructed == ()
    (friday,) = [e for e in found.evenings if e.as_of == FRIDAY]
    assert friday.missed and friday.entries == ()
    view = web_history.history_view(found, now=ny(WEDNESDAY, "08:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    (shown,) = [e for e in view["evenings"] if e["as_of"] == FRIDAY]
    assert shown["missed_words"] == words.MISSED_STAYS and shown["items"] == []


def test_model_health_still_reports_the_evening_missed_with_its_reconstruction_noted():
    evenings = {"since": "2026-10-20", "due": 8, "recorded": 6, "missed": ["2026-10-27",
                "2026-10-29"], "recent": 10, "recent_missed": 2,
                "reconstructed": ["2026-10-27"]}  # fmt: skip
    finding = _evening_finding(evenings)
    assert finding.reason == "evening_missed" and finding.state.value == "degraded"
    assert "1 of them reconstructed after the fact, scored apart from the live record" in (
        finding.summary
    )
    assert "missed (latest 5): 2026-10-27 (reconstructed), 2026-10-29" in _evening_text(evenings)
    plain = {key: value for key, value in evenings.items() if key != "reconstructed"}
    assert "reconstructed" not in _evening_finding(plain).summary
    assert "(reconstructed)" not in _evening_text(plain)


def test_only_a_reconstructed_stock_is_labelled_on_its_missed_evening(watching):
    # NVDA's Friday window can't be read yet: AMD and MU are reconstructed, NVDA stays absent.
    lab = watching
    rec._live(lab, AS_OF)
    rec._batch(lab, FRIDAY, tickers=("AMD", "MU"))
    rec._batch(lab, MONDAY, tickers=("AMD", "MU"))
    rec._reconstruct(lab)
    records = ReadOnlyRecords(lab[0].path)
    found = forecast_history(records, now=ny(TUESDAY, "09:00").to_pydatetime(),
                             sessions=weekdays, data_class="real", tickers=TICKERS)  # fmt: skip
    view = web_history.history_view(found, now=ny(TUESDAY, "09:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    (shown,) = [e for e in view["evenings"] if e["as_of"] == FRIDAY]
    labels = {item["ticker"]: item["label"] for item in shown["items"]}
    assert labels["AMD"].startswith(words.RECONSTRUCTED_LABEL)
    assert labels["MU"].startswith(words.RECONSTRUCTED_LABEL)
    assert labels["NVDA"] == f"NVDA: {words.NO_OUTLOOK_RECORDED}."


@pytest.mark.parametrize(
    "rebuilt",
    [
        ["2026-10-09"],
        ["2026-10-19", "2026-10-19"],
        ["2026-10-19", "2026-10-13"],
        "2026-10-19",
        ["19 October"],
    ],
    ids=["before-since", "repeated", "out-of-order", "not-a-list", "not-a-session"],
)
def test_model_health_refuses_malformed_reconstructed_sessions(tmp_path, rebuilt):
    missed = {**EVENINGS, "recorded": 7, "missed": ["2026-10-19"], "recent_missed": 1}
    _, checks = evenings_tests._job(tmp_path, {**missed, "reconstructed": rebuilt})
    assert {(c.state, c.reason) for c in checks.values()} == {
        (HealthState.UNAVAILABLE, "unrecognized_format")
    }
    # A missed evening, and a partly recorded one (not missed), may each hold reconstructions.
    fine = {**missed, "reconstructed": ["2026-10-13", "2026-10-19"]}
    _, checks = evenings_tests._job(tmp_path / "ok", fine)
    assert checks["forecast.stock_lab_job"].reason == "evening_missed"


def test_model_health_notes_a_partly_recorded_evening_reconstructed():
    evenings = {**EVENINGS, "reconstructed": ["2026-10-13"]}  # nothing missed
    assert _evening_finding(evenings) is None  # recorded evenings raise no miss
    assert "reconstructed after the fact (latest 5): 2026-10-13" in _evening_text(evenings)


def test_history_shows_a_partly_recorded_evenings_reconstructed_stocks_labelled(watching):
    lab = watching
    rec._live(lab, AS_OF)
    rec._batch(lab, FRIDAY)
    rec._live(lab, FRIDAY, tickers=("MU",))
    rec._batch(lab, MONDAY)
    rec._reconstruct(lab)
    records = ReadOnlyRecords(lab[0].path)
    found = forecast_history(records, now=ny(TUESDAY, "09:00").to_pydatetime(),
                             sessions=weekdays, data_class="real", tickers=TICKERS)  # fmt: skip
    assert found.missed == () and found.reconstructed == (FRIDAY,)
    (friday,) = [e for e in found.evenings if e.as_of == FRIDAY]
    assert not friday.missed
    assert {x.symbol: x.reconstructed for x in friday.entries} == {
        "AMD": True,
        "MU": False,
        "NVDA": True,
    }
    view = web_history.history_view(found, now=ny(TUESDAY, "09:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    (shown,) = [e for e in view["evenings"] if e["as_of"] == FRIDAY]
    labels = {item["ticker"]: item["label"] for item in shown["items"]}
    assert labels["AMD"].startswith(words.RECONSTRUCTED_LABEL)
    assert labels["NVDA"].startswith(words.RECONSTRUCTED_LABEL)
    assert not labels["MU"].startswith(words.RECONSTRUCTED_LABEL)
    assert view["counts"].endswith("0 missed; 2 outlooks reconstructed after the fact.")


def test_a_reconstructed_outlook_never_changes_a_recorded_evenings_live_words():
    from stock_agent.stock_lab.briefing.history import Entry, Evening, History, Outcome

    day, session = "2026-10-16", "2026-10-19"
    live = Entry("MU", True, "2026-10-17T00:35:00.000000Z", session, 0.4, 0.3, 0.3, 0.01, None,
                 Outcome("pending", None, None, None))  # fmt: skip
    rebuilt = Entry("AMD", True, "2026-10-20T00:40:00.000000Z", session, 0.4, 0.3, 0.3, 0.01,
                    None, Outcome("scored", "increase", 0.02, None), reconstructed=True)  # fmt: skip
    found = History("2026-10-21T13:00:00.000000Z", "real", None, day, 1, 1, (),
                    (Evening(day, session, False, (live, rebuilt)),), ("AMD", "MU"), (day,))  # fmt: skip
    view = web_history.history_view(found, now=ny("2026-10-21", "09:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    # Only the reconstruction has a result: the live evening has none yet.
    assert (
        "1 evening recorded, 0 with a result, 0 missed; 1 outlook reconstructed" in view["counts"]
    )
    # A live result, with the reconstruction still waiting: the evening is not pending.
    live = Entry("MU", True, "2026-10-17T00:35:00.000000Z", session, 0.4, 0.3, 0.3, 0.01, None,
                 Outcome("scored", "stable", 0.001, None))  # fmt: skip
    rebuilt = Entry("AMD", True, "2026-10-20T00:40:00.000000Z", session, 0.4, 0.3, 0.3, 0.01,
                    None, Outcome("pending", None, None, None), reconstructed=True)  # fmt: skip
    found = History("2026-10-21T13:00:00.000000Z", "real", None, day, 1, 1, (),
                    (Evening(day, session, False, (live, rebuilt)),), ("AMD", "MU"), (day,))  # fmt: skip
    view = web_history.history_view(found, now=ny("2026-10-21", "09:00").to_pydatetime(),
                                    local_zone=None)  # fmt: skip
    assert view["evenings"][0]["state"] == "recorded"
