"""
The Experiments page's "Prospective forecasts (unvalidated)" panel: the
empty state, a recorded forecast, a scored one and a broken ledger, each
rendered through the running app and scanned for advice wording; and the
app's own ledger check agreeing with the forecasting job's.
"""

from __future__ import annotations

import pytest
from forecast_prospective_helpers import AS_OF, record, score
from test_app_destinations import running, text_of

from stock_agent.app import forecast_records, wording
from stock_agent.forecasting.prospective import ledger


@pytest.fixture
def ledger_path(tmp_path, monkeypatch):
    path = tmp_path / "private" / "ledger.jsonl"
    monkeypatch.setenv(forecast_records.LEDGER_ENV, str(path))
    return path


def _tables(app):
    return [frame.value for frame in app.dataframe]


def test_without_a_ledger_the_panel_says_no_forecast_is_recorded():
    app = running("experiments")
    text = text_of(app)
    assert wording.PROSPECTIVE_TITLE in text
    assert wording.PROSPECTIVE_EMPTY_TITLE in text and wording.PROSPECTIVE_EMPTY in text
    assert wording.EXPERIMENTS_EMPTY_TITLE in text
    assert not app.dataframe and not app.metric


def test_a_recorded_forecast_shows_its_point_band_null_as_of_and_labels(
    prospective_frozen, ledger_path
):
    entry = record(prospective_frozen, ledger_path)
    app = running("experiments")
    text = text_of(app)  # scanned for advice wording
    assert wording.PROSPECTIVE_TITLE in text and wording.PROSPECTIVE_EMPTY_TITLE not in text
    assert wording.EXPERIMENTS_EMPTY_TITLE not in text
    assert f"As of the {AS_OF} close" in text
    assert "unvalidated, no demonstrated skill" in text
    assert wording.PROSPECTIVE_BADGE in text and wording.PROSPECTIVE_SYNTHETIC in text
    assert wording.PROSPECTIVE_BAND in text and wording.PROSPECTIVE_NO_SCORES in text
    (table,) = _tables(app)
    columns = wording.PROSPECTIVE_COLUMNS
    assert list(table.columns) == list(columns.values())
    assert len(table) == len(entry.payload["rows"])
    first = entry.payload["rows"][0]
    shown = table.iloc[0]
    assert shown[columns["fund"]] == first["asset"]
    assert shown[columns["forecast"]] == f"{first['display_percent']['point']:+.2f}%"
    assert shown[columns["low"]] == f"{first['display_percent']['band_low']:+.2f}%"
    assert shown[columns["high"]] == f"{first['display_percent']['band_high']:+.2f}%"
    assert shown[columns["null"]] == f"{first['display_percent']['null']:+.2f}%"
    assert shown[columns["stage"]] == "Engineering only"
    assert set(table[columns["outcome"]]) == {wording.PROSPECTIVE_PENDING}  # nothing matured


def test_scores_appear_only_once_matured(prospective_frozen, ledger_path):
    record(prospective_frozen, ledger_path)
    score(ledger_path, "2026-10-23", "2026-10-23T23:00:00Z")  # only the 5-session rows matured
    app = running("experiments")
    text = text_of(app)
    forecasts, cumulative = _tables(app)
    columns = wording.PROSPECTIVE_COLUMNS
    matured = forecasts[columns["horizon"]] == "5 sessions"
    assert (forecasts.loc[matured, columns["outcome"]] != wording.PROSPECTIVE_PENDING).all()
    assert set(forecasts.loc[matured, columns["inside"]]) <= {"Yes", "No"}
    assert (forecasts.loc[~matured, columns["outcome"]] == wording.PROSPECTIVE_PENDING).all()
    assert wording.PROSPECTIVE_SCORES_TITLE in text
    assert len(cumulative) == 6 and set(cumulative["Scored"]) == {"1"}
    assert "far too few to judge calibration or skill" in text


def test_a_ledger_that_fails_its_check_shows_nothing_but_that(prospective_frozen, ledger_path):
    record(prospective_frozen, ledger_path)
    text = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(text.replace('"VTI"', '"VXX"', 1), encoding="utf-8")
    app = running("experiments")
    shown = text_of(app)
    assert "failed its integrity check" in shown
    assert not app.dataframe


def test_the_apps_ledger_check_agrees_with_the_jobs(prospective_frozen, ledger_path):
    record(prospective_frozen, ledger_path)
    text = ledger_path.read_text(encoding="utf-8")
    entries = ledger.read_ledger(ledger_path)
    mine = forecast_records.verified_entries(text)
    assert [e["entry_sha256"] for e in mine] == [e.entry_sha256 for e in entries]
    for edited in (text.replace("2026-10-14", "2026-10-13", 1), text.rstrip("\n"), text * 2):
        ledger_path.write_text(edited, encoding="utf-8")
        with pytest.raises(ledger.LedgerError):
            ledger.read_ledger(ledger_path)
        with pytest.raises(forecast_records.LedgerUnreadable):
            forecast_records.read_records(ledger_path)


def test_the_default_ledger_is_private_and_ignored_by_git():
    from forecast_prospective_helpers import REPO

    relative = forecast_records.DEFAULT_LEDGER.relative_to(REPO)
    assert relative == ledger.DEFAULT_LEDGER
    assert relative.parts[:2] == ("data", "raw")
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/*" in ignored.splitlines()
