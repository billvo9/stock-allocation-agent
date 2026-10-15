"""scripts/forecast_prospective.py: the report, and a clean refusal without the store loader."""

from __future__ import annotations

import importlib.util

import pytest
from prospective_helpers import REPO

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


def test_without_the_store_loader_nothing_is_read_or_written(
    prospective_frozen, tmp_path, monkeypatch
):
    script = _script()
    monkeypatch.setattr(script, "LOADER", ("stock_agent.no_such_loader", "load_inference_window"))
    ledger_path = tmp_path / "ledger.jsonl"
    with pytest.raises(SystemExit, match="is not available on this branch"):
        script.main(
            [
                "--ledger",
                str(ledger_path),
                "forecast",
                "--bundle",
                str(prospective_frozen.result.bundle.path),
                "--store-root",
                str(tmp_path / "store"),
                "--forecast-close",
                "2026-10-14",
            ]
        )
    assert not ledger_path.exists()


def test_the_defaults_are_private_paths():
    script = _script()
    assert script.DEFAULT_LEDGER.relative_to(REPO).parts[:2] == ("data", "raw")
    assert script.DEFAULT_OUTPUT.relative_to(REPO).parts[0] == "reports"
    assert script.DEFAULT_CONFIG.name == "etf_prospective_v1.yaml"
