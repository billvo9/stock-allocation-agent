"""
The Lab app's launcher (scripts/run_lab_app.py): the synthetic demo end to end,
the real wiring (the NYSE rules calendar, the recent-prices reader, the real
data class, the owner's exclusions) with fakes, and its refusals. The server is
never left running: `serve` is replaced by a function that asks the app once.
"""

from __future__ import annotations

import importlib.util
import io
import json
import socket
import subprocess
from pathlib import Path

import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, ny

from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.contracts import DataClass

REPO = Path(__file__).resolve().parents[1]


def _launcher():
    path = REPO / "scripts" / "run_lab_app.py"
    spec = importlib.util.spec_from_file_location("stock_agent_run_lab_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _launcher()


class Ask:
    """A stand-in for serving: one GET of the Home API, then the server is closed."""

    def __init__(self):
        self.home = None

    def __call__(self, server):
        app = server.app
        try:
            response = app.handle("GET", "/api/home", f"127.0.0.1:{app.port}")
            assert response.status == 200, response.body
            self.home = json.loads(response.body)
        finally:
            server.server_close()


def _changed_files() -> set[str]:
    done = subprocess.run(
        ["git", "status", "--porcelain", "--ignored", "--untracked-files=all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in done.stdout.splitlines() if "__pycache__" not in line}


def test_the_demo_serves_synthetic_stocks_and_writes_nothing_here():
    before = _changed_files()
    ask, out = Ask(), io.StringIO()
    assert launcher.main(["--demo", "2", "--port", "0"], serve=ask, out=out) == 0
    home = ask.home
    assert home["data_class"] == "synthetic" and home["demo"]
    assert [s["ticker"] for s in home["stocks"]] == ["DEMO_CALM", "DEMO_SWING", "DEMO_TREND"]
    assert all(s["name"].endswith("(synthetic)") for s in home["stocks"])
    assert all(s["price"]["available"] and s["price"]["line"] for s in home["stocks"])
    assert all(s["outlook"]["state"] == "fresh" for s in home["stocks"])
    assert home["status"]["level"] == "ok" and home["research_url"] is None
    assert "http://127.0.0.1:" in out.getvalue() and "synthetic demo" in out.getvalue()
    assert _changed_files() == before


def test_the_real_app_wires_the_rules_calendar_the_reader_and_the_exclusions(tmp_path, monkeypatch):
    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "SKHY", at=ny("2026-10-21", "12:00"))
    seen = {}

    def spy(records, **options):
        seen.update(options)
        return {"stocks": []}

    monkeypatch.setattr(launcher, "build_home", spy)
    ask = Ask()
    code = launcher.main(
        [
            "--port",
            "0",
            "--at",
            "2026-10-22T07:40-04:00",
            "--research-url",
            "http://127.0.0.1:8502/",
        ],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        recent=launcher.observed,
        serve=ask,
        out=io.StringIO(),
    )
    assert code == 0
    assert seen["calendar"] is sessions and seen["calendar_id"] == CALENDAR_ID
    assert seen["data_class"] == DataClass.REAL and seen["recent"] is launcher.observed
    assert seen["excluded"] == ("SKHY", "SNDK") and seen["local_zone"] is None
    assert seen["now"].isoformat() == "2026-10-22T11:40:00+00:00"
    assert ask.home["research_url"] == "http://127.0.0.1:8502/"


def test_the_real_app_end_to_end_on_fixture_records(tmp_path):
    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "SNDK", at=ny("2026-10-21", "12:00"))
    lab.forecast(REAL_TEST_TICKER, "2026-10-21", created_at=ny("2026-10-21", "20:31"))
    lab.job(at=ny("2026-10-21", "20:32"), expected="2026-10-21", status="ok", activation=None)
    calls = []

    def recent(symbol, *, count, retrieved_by):
        calls.append((symbol, count))
        return launcher.SimpleNamespace(state="no_store", reason=None, sessions=())

    ask = Ask()
    launcher.main(
        ["--port", "0", "--at", "2026-10-22T07:40-04:00"],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        recent=recent,
        serve=ask,
        out=io.StringIO(),
    )
    (stock,) = ask.home["stocks"]
    assert stock["ticker"] == REAL_TEST_TICKER and calls == [(REAL_TEST_TICKER, 22)]
    assert (
        stock["outlook"]["state"] == "fresh" and stock["price"]["words"] == "No prices stored yet."
    )


def test_a_failing_price_read_is_shown_as_unreadable(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError("/Users/someone/stock-agent-private/... is locked")

    monkeypatch.setattr(launcher, "read_recent", broken)
    found = launcher.observed("NVDA", count=22, retrieved_by=None)
    assert found.state == "unreadable" and found.sessions == ()
    assert "OSError" in capsys.readouterr().err


def test_a_busy_port_is_refused_plainly():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        out = io.StringIO()
        assert launcher.main(["--demo", "2", "--port", str(port)], serve=Ask(), out=out) == 2
    assert f"port {port} can't be used" in out.getvalue()


@pytest.mark.parametrize(
    ("argv", "words"),
    [
        (["--research-url", "https://example.com/"], "a page on this Mac"),
        (["--at", "2026-10-22T07:40"], "needs a time zone"),
        (["--demo", "1"], "2 to 60 sessions"),
    ],
)
def test_bad_options_are_refused(argv, words):
    out = io.StringIO()
    assert launcher.main(argv, serve=Ask(), out=out) == 2
    assert words in out.getvalue()
