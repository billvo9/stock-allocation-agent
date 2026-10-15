"""scripts/forecast_prospective.py: the report, and a clean refusal without the store loader."""

from __future__ import annotations

import importlib.util

import pytest
from forecast_prospective_helpers import AS_OF, CREATED, REPO

SCRIPT = REPO / "scripts" / "forecast_prospective.py"


def _script():
    spec = importlib.util.spec_from_file_location("forecast_prospective", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_report_on_an_empty_ledger_says_so(tmp_path, capsys):
    assert _script().main(["--ledger", str(tmp_path / "ledger.jsonl"), "report"]) == 0
    out = capsys.readouterr().out
    assert "0 forecasts" in out and "no scores yet" in out


def test_the_report_refuses_a_ledger_without_the_kept_head(tmp_path):
    from stock_agent.forecasting.prospective import ledger

    path = tmp_path / "ledger.jsonl"
    with pytest.raises(ledger.LedgerError, match="not in the ledger"):
        _script().main(["--ledger", str(path), "report", "--expect-head", "a" * 64])


def test_the_script_forecasts_through_the_real_loader_and_calendar(
    prospective_frozen, prospective_store_root, tmp_path, monkeypatch
):
    from stock_agent.calendars import nyse_rules
    from stock_agent.forecasting.prospective import ledger

    script = _script()
    monkeypatch.setattr(script, "_now", lambda: CREATED)
    path = tmp_path / "ledger.jsonl"
    argv = [
        "--ledger",
        str(path),
        "forecast",
        "--bundle",
        str(prospective_frozen.result.bundle.path),
        "--store-root",
        str(prospective_store_root),
        "--forecast-close",
        AS_OF,
    ]
    assert script.main(argv) == 0
    (entry,) = ledger.read_ledger(path)
    assert entry.kind == "forecast"
    assert entry.payload["data"]["calendar"] == nyse_rules.CALENDAR_ID
    assert entry.payload["data"]["retrieved_by"] == CREATED.isoformat()
    assert script.exchange_calendar.calendar_id == nyse_rules.CALENDAR_ID


def test_the_defaults_are_private_paths():
    script = _script()
    assert script.DEFAULT_LEDGER.relative_to(REPO).parts[:2] == ("data", "raw")
    assert script.DEFAULT_OUTPUT.relative_to(REPO).parts[0] == "reports"
    assert script.DEFAULT_CONFIG.name == "etf_prospective_v1.yaml"
