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
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
DEFAULT_HEALTH_SUMMARY = launcher.default_health_summary  # before any test replaces it


@pytest.fixture(autouse=True)
def no_keys_from_the_caller(monkeypatch):
    """Tests that use the real environment never see a key set where they run."""

    for name in ("LAB_APP_TOKEN", "LAB_APP_RESEARCH_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_private_health_summary(monkeypatch, tmp_path):
    """Health never reads the owner's Model Health summary in a test."""

    monkeypatch.setattr(launcher, "default_health_summary", lambda: tmp_path / "no-summary.json")


def get(app, path):
    """One GET with this run's key, as the page sends it."""

    return app.handle("GET", path, f"127.0.0.1:{app.port}", {"x-lab-token": app.token})


class Ask:
    """
    A stand-in for serving: one GET of the Home API and one of the first listed stock's
    detail, then the server is closed.
    """

    def __init__(self):
        self.home = self.detail = self.history = self.health = self.strategies = None
        self.missing = None  # the status of a stock that is not listed

    def __call__(self, server):
        app = server.app
        try:
            for name in ("history", "health", "strategies"):
                response = get(app, f"/api/{name}")
                assert response.status == 200, response.body
                setattr(self, name, json.loads(response.body))
            response = get(app, "/api/home")
            assert response.status == 200, response.body
            self.home = json.loads(response.body)
            stocks = self.home.get("stocks") or []
            if stocks:
                response = get(app, f"/api/stock/{stocks[0]['ticker']}")
                assert response.status == 200, response.body
                self.detail = json.loads(response.body)
            self.missing = get(app, "/api/stock/ZZZZ").status
        finally:
            server.server_close()


BYTECODE = re.compile(r"(^|/)__pycache__/[^/]+\.pyc$")  # the interpreter's own cache only
# Folders other tools fill while tests run (the environment, linters, test caches).
NOT_OURS = (".venv", ".ruff_cache", ".pytest_cache", ".mypy_cache")


def _changed_files() -> set[str]:
    done = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--ignored",
            "--untracked-files=all",
            "--",
            ".",
            *[f":(exclude){folder}" for folder in NOT_OURS],
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in done.stdout.splitlines() if not BYTECODE.search(line[3:])}


def test_the_demo_serves_synthetic_stocks_reads_nothing_real_and_writes_nothing_here(
    monkeypatch,
):
    def refuse(*args, **kwargs):
        raise AssertionError("the demo read real recent prices")

    monkeypatch.setattr(launcher, "read_recent", refuse)
    seen, real_build = {}, launcher.build_home

    def spy(records, **options):
        seen.update(options)
        return real_build(records, **options)

    monkeypatch.setattr(launcher, "build_home", spy)
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
    detail = ask.detail
    assert detail["ticker"] == "DEMO_CALM" and detail["data_class"] == "synthetic"
    assert detail["outlook"]["state"] == "fresh" and detail["demo"]
    assert detail["chart"]["available"] and len(detail["chart"]["points"]) == 252
    assert detail["learning"]["url"] is None and ask.missing == 404
    # The replay records the run only, so the demo shows no other context chip.
    assert {chip["key"] for chip in detail["lately"]["chips"]} <= {"run"}
    development = detail["reliable"]["development"]
    assert development["state"] == "ok"
    assert "synthetic demo stocks" in " ".join(development["sheet"]["paragraphs"])
    assert "http://127.0.0.1:" in out.getvalue() and "synthetic demo" in out.getvalue()
    assert seen["local_zone"] is None and seen["data_class"] == DataClass.SYNTHETIC
    assert seen["summary"] is None and home["badge"] is None  # the demo has no summary
    history = ask.history
    assert history["data_class"] == "synthetic" and history["demo"]
    assert history["filters"] == ["All", "DEMO_CALM", "DEMO_SWING", "DEMO_TREND"]
    assert history["problem"] is None and history["counts"].endswith("0 missed.")
    # The replay's evenings: the last one waits for its result; the earlier ones are scored.
    assert [e["state"] for e in history["evenings"]] == ["pending", "recorded", "recorded"]
    assert {x["state"] for x in history["evenings"][0]["items"]} == {"pending"}
    assert {x["state"] for e in history["evenings"][1:] for x in e["items"]} == {"scored"}
    health = ask.health
    assert (health["level"], health["line"]) == ("ok", home["status"]["words"])
    assert health["checks"]["note"] == "The demo has no Model Health checks."
    assert health["demo"] and health["link"] is None
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
        serve=ask,
        out=io.StringIO(),
    )
    assert code == 0
    assert seen["calendar"] is sessions and seen["calendar_id"] == CALENDAR_ID
    assert seen["data_class"] == DataClass.REAL and seen["recent"] is launcher.observed
    assert seen["excluded"] == ("SKHY", "SNDK") and seen["local_zone"] is None
    assert seen["now"].isoformat() == "2026-10-22T11:40:00+00:00"
    assert ask.home["research_url"] == "http://127.0.0.1:8502/"
    # The tab's badge reads Model Health's summary (this test's stand-in default) at that time.
    read = seen["summary"]()
    assert read.path == tmp_path / "no-summary.json" and read.missing


def test_the_real_detail_wires_the_rules_calendar_the_reader_and_the_exclusions(
    tmp_path, monkeypatch
):
    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "SKHY", at=ny("2026-10-21", "12:00"))
    seen = {}

    def spy(records, **options):
        seen.update(options)
        return {"ticker": options["symbol"]}

    monkeypatch.setattr(launcher, "build_detail", spy)
    ask = Ask()
    launcher.main(
        [
            "--port",
            "0",
            "--at",
            "2026-10-22T07:40-04:00",
            "--research-url",
            "http://127.0.0.1:8502/",
        ],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        serve=ask,
        out=io.StringIO(),
    )
    assert ask.detail == {"ticker": REAL_TEST_TICKER}
    assert seen["calendar"] is sessions and seen["calendar_id"] == CALENDAR_ID
    assert seen["data_class"] == DataClass.REAL and seen["recent"] is launcher.observed
    assert seen["excluded"] == ("SKHY", "SNDK") and seen["local_zone"] is None
    assert seen["now"].isoformat() == "2026-10-22T11:40:00+00:00"
    assert seen["research_url"] == "http://127.0.0.1:8502/" and "tickers" not in seen


def test_the_real_history_and_health_wire_the_calendar_the_summary_and_the_exclusions(
    tmp_path, monkeypatch
):
    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "SKHY", at=ny("2026-10-21", "12:00"))
    seen = {}

    def history(records, **options):
        seen["history"] = options
        return {"evenings": []}

    def health(records, **options):
        seen["health"] = options
        seen["summary"] = options["summary"]()
        return {"level": "ok"}

    def home(records, **options):
        seen["home"] = options["summary"]()
        return {"stocks": []}

    monkeypatch.setattr(launcher, "build_history", history)
    monkeypatch.setattr(launcher, "build_health", health)
    monkeypatch.setattr(launcher, "build_home", home)
    summary = tmp_path / "health" / "latest.json"
    ask = Ask()
    argv = ["--port", "0", "--at", "2026-10-22T07:40-04:00", "--research-url"]
    launcher.main(
        [*argv, "http://127.0.0.1:8502/", "--health-summary", str(summary)],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        serve=ask,
        out=io.StringIO(),
    )
    assert (ask.history, ask.health) == ({"evenings": []}, {"level": "ok"})
    for name in ("history", "health"):
        options = seen[name]
        assert options["calendar"] is sessions and options["data_class"] == DataClass.REAL
        assert options["excluded"] == ("SKHY", "SNDK") and options["local_zone"] is None
        assert options["now"].isoformat() == "2026-10-22T11:40:00+00:00"
        assert "tickers" not in options
    assert seen["health"]["calendar_id"] == CALENDAR_ID
    assert seen["health"]["research_url"] == "http://127.0.0.1:8502/"
    assert seen["summary"].path == summary and seen["summary"].missing  # read, not there
    assert seen["home"].path == summary  # the tab's badge reads the same file


def test_health_reads_model_healths_local_summary_unless_told_otherwise(tmp_path, monkeypatch):
    # The research pages' own default (scripts/run_portfolio_lab.py); only the path is checked.
    assert (
        DEFAULT_HEALTH_SUMMARY() == Path.home() / "stock-agent-private" / "health" / "latest.json"
    )
    seen = {}

    def health(records, **options):
        seen["summary"] = options["summary"]()
        return {"level": "ok"}

    monkeypatch.setattr(launcher, "build_health", health)
    launcher.main(
        ["--port", "0"],
        records_factory=lambda: ReadOnlyRecords(tmp_path / "none.sqlite"),
        clock=lambda: pd.Timestamp("2026-10-22T11:40:00Z"),
        serve=Ask(),
        out=io.StringIO(),
    )
    assert seen["summary"].path == tmp_path / "no-summary.json"  # this test's stand-in default


class Locked:
    """A stand-in for serving: every data route, asked without the key and with a wrong one."""

    def __init__(self):
        self.answers = {}

    def __call__(self, server):
        app = server.app
        try:
            paths = [*app.api, *(f"{prefix}ZZZZ" for prefix in app.items)]
            for path in paths:
                for key in (None, "w" * 43):
                    headers = {} if key is None else {"x-lab-token": key}
                    found = app.handle("GET", path, f"127.0.0.1:{app.port}", headers)
                    self.answers[(path, key)] = found.status
        finally:
            server.server_close()


@pytest.mark.parametrize("mode", [[], ["--demo", "2"]], ids=["real", "demo"])
def test_every_data_route_refuses_a_request_without_this_runs_key(tmp_path, mode):
    locked = Locked()
    launcher.main(
        [*mode, "--port", "0"],
        records_factory=lambda: ReadOnlyRecords(tmp_path / "none.sqlite"),
        clock=lambda: pd.Timestamp("2026-10-22T11:40:00Z"),
        serve=locked,
        out=io.StringIO(),
    )
    routes = {path for path, _ in locked.answers}
    assert routes == {
        "/api/home",
        "/api/history",
        "/api/health",
        "/api/strategies",
        "/api/stock/ZZZZ",
    }
    assert set(locked.answers.values()) == {401}


def test_the_real_app_end_to_end_on_fixture_records(tmp_path):
    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "SNDK", at=ny("2026-10-21", "12:00"))
    lab.forecast(REAL_TEST_TICKER, "2026-10-21", created_at=ny("2026-10-21", "20:31"))
    lab.job(at=ny("2026-10-21", "20:32"), expected="2026-10-21", status="ok", activation=None)
    calls = []

    def recent(symbol, *, count, retrieved_by):
        calls.append((symbol, count))
        assert retrieved_by == pd.Timestamp("2026-10-22T11:40Z")
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
    assert stock["ticker"] == REAL_TEST_TICKER
    assert calls == [(REAL_TEST_TICKER, 22), (REAL_TEST_TICKER, 252)]  # Home, then the detail
    assert (
        stock["outlook"]["state"] == "fresh" and stock["price"]["words"] == "No prices stored yet."
    )
    assert ask.detail["outlook"]["state"] == "fresh" and ask.detail["chart"]["available"] is False
    assert ask.missing == 404


def test_a_failing_price_read_is_shown_as_unreadable(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError("/Users/someone/stock-agent-private/... is locked")

    monkeypatch.setattr(launcher, "read_recent", broken)
    found = launcher.observed("NVDA", count=22, retrieved_by=None)
    assert found.state == "unreadable" and found.sessions == ()
    err = capsys.readouterr().err
    assert "OSError" in err and "stock-agent-private" not in err


def test_without_at_the_clock_is_read_on_every_request(tmp_path):
    lab = Lab(tmp_path)
    moments = iter(pd.Timestamp("2026-10-22T11:40Z") + pd.Timedelta(minutes=k) for k in range(9))
    seen = []

    def twice(server):
        app = server.app
        try:
            for _ in range(2):
                response = get(app, "/api/home")
                seen.append(json.loads(response.body)["generated_at"])
        finally:
            server.server_close()

    launcher.main(
        ["--port", "0"],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        clock=lambda: next(moments),
        recent=lambda symbol, **_: launcher.SimpleNamespace(state="no_store", sessions=()),
        serve=twice,
        out=io.StringIO(),
    )
    assert seen == ["2026-10-22T11:40:00+00:00", "2026-10-22T11:41:00+00:00"]


def test_a_busy_port_is_refused_plainly(monkeypatch):
    def replay(*args, **kwargs):
        raise AssertionError("the demo replay ran before the port was claimed")

    monkeypatch.setattr(launcher, "demo_api", replay)
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
        (["--port", "70000"], "port 70000 can't be used"),
        (["--demo", "61"], "2 to 60 sessions"),
        (["--research-url", "https://example.com/"], "a page on this Mac"),
        (["--research-url", "http://127.0.0.1:@evil.example/"], "a page on this Mac"),
        (["--research-url", "http://localhost:x@evil.example/"], "a page on this Mac"),
        (["--research-url", "http://127.0.0.1:99999/"], "a page on this Mac"),
        (["--research-url", "http://127.0.0.1:8502/ x"], "a page on this Mac"),
        (["--research-url", "http://127.0.0.1/"], "a page on this Mac"),
        (["--at", "2026-10-22T07:40"], "needs a time zone"),
        (["--demo", "1"], "2 to 60 sessions"),
    ],
)
def test_bad_options_are_refused(argv, words):
    out = io.StringIO()
    assert launcher.main(argv, serve=Ask(), out=out) == 2
    assert words in out.getvalue()


@pytest.mark.parametrize("url", ["http://127.0.0.1:8502/", "http://localhost:8502/research"])
def test_a_research_url_on_this_mac_is_accepted(url):
    assert launcher.local_url(url)


def test_a_polite_stop_removes_the_demo_store_like_control_c(tmp_path):
    script = REPO / "scripts" / "run_lab_app.py"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "TMPDIR": str(tmp_path)}
    run = subprocess.Popen(
        [sys.executable, "-B", str(script), "--demo", "2", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    try:
        assert "Lab app: http://127.0.0.1:" in run.stdout.readline()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith("lab-app-demo-")]
        run.send_signal(signal.SIGTERM)
        assert run.wait(timeout=60) == 0
    finally:
        if run.poll() is None:
            run.kill()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("lab-app-demo-")]


def test_a_detail_whose_store_cannot_be_read_answers_with_the_plain_error(tmp_path, capsys):
    # Not "not on the watchlist": the store's watchlist itself could not be read.
    broken = tmp_path / "lab.sqlite3"
    broken.write_text("not a database")
    seen = {}

    def ask(server):
        app = server.app
        try:
            seen["status"] = get(app, "/api/stock/NVDA")
        finally:
            server.server_close()

    launcher.main(
        ["--port", "0", "--at", "2026-10-22T07:40-04:00"],
        records_factory=lambda: ReadOnlyRecords(broken),
        recent=lambda *args, **kwargs: pytest.fail("prices were read for an unverified stock"),
        serve=ask,
        out=io.StringIO(),
    )
    response = seen["status"]
    assert response.status == 500
    assert json.loads(response.body) == {"error": "The app couldn't prepare this screen."}
    assert "/api/stock/ failed" in capsys.readouterr().err


# --- this run's key (owner decision of 2026-10-23) ---


class Terminal(io.StringIO):
    def isatty(self):
        return True


def keyed_run(argv, **options):
    seen = {}

    def ask(server):
        seen["app"] = server.app
        seen["locked"] = server.app.handle("GET", "/api/home", f"127.0.0.1:{server.app.port}")
        server.server_close()

    options.setdefault("out", io.StringIO())
    code = launcher.main(
        ["--port", "0", *argv],
        recent=lambda symbol, **_: launcher.SimpleNamespace(state="no_store", sessions=()),
        serve=ask,
        **options,
    )
    return code, seen, options["out"].getvalue()


def test_the_key_the_mac_app_passes_locks_the_data_and_is_never_printed():
    key = "m" * 43
    code, seen, printed = keyed_run([], environ={"LAB_APP_TOKEN": key})
    assert code == 0 and seen["app"].token == key
    assert seen["locked"].status == 401
    assert key not in printed


def test_without_a_passed_key_each_run_makes_its_own():
    keys = iter(["a" * 43, "b" * 43])
    first = keyed_run([], environ={}, make_key=lambda: next(keys))[1]["app"].token
    second = keyed_run([], environ={}, make_key=lambda: next(keys))[1]["app"].token
    assert (first, second) == ("a" * 43, "b" * 43)
    assert len(launcher.new_key()) >= 43 and launcher.new_key() != launcher.new_key()


@pytest.mark.parametrize("bad", ["short", "x" * 40 + "#key", "x" * 40 + " ab"])
def test_an_unusable_passed_key_is_refused(bad):
    code, seen, printed = keyed_run([], environ={"LAB_APP_TOKEN": bad})
    assert code == 2 and "LAB_APP_TOKEN must be" in printed and not seen


def test_open_takes_the_browser_to_the_keyed_address(monkeypatch):
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    code, seen, printed = keyed_run(["--open"], environ={"LAB_APP_TOKEN": "o" * 43})
    assert code == 0
    port = seen["app"].port
    assert opened == [f"http://127.0.0.1:{port}/#key={'o' * 43}"]
    assert "o" * 43 not in printed


def test_only_a_terminal_is_shown_the_keyed_address():
    code, _, printed = keyed_run([], environ={"LAB_APP_TOKEN": "t" * 43}, out=Terminal())
    assert code == 0 and f"#key={'t' * 43}" in printed


def test_the_research_link_carries_the_research_pages_own_key():
    seen = {}

    def ask(server):
        response = get(server.app, "/api/home")
        seen["research_url"] = json.loads(response.body)["research_url"]
        server.server_close()

    code = launcher.main(
        ["--port", "0", "--research-url", "http://127.0.0.1:8502/"],
        recent=lambda symbol, **_: launcher.SimpleNamespace(state="no_store", sessions=()),
        serve=ask,
        out=io.StringIO(),
        environ={"LAB_APP_TOKEN": "l" * 43, "LAB_APP_RESEARCH_KEY": "p" * 43},
    )
    assert code == 0 and seen["research_url"] == f"http://127.0.0.1:8502/?key={'p' * 43}"


def test_an_unusable_research_key_is_refused():
    out = io.StringIO()
    code = launcher.main(
        ["--port", "0"], serve=pytest.fail, out=out, environ={"LAB_APP_RESEARCH_KEY": "short"}
    )
    assert code == 2 and "LAB_APP_RESEARCH_KEY must be" in out.getvalue()


@pytest.mark.parametrize("mode", [[], ["--demo", "2"]], ids=["real", "demo"])
def test_strategies_reads_the_reports_folder_only_in_the_real_app(tmp_path, monkeypatch, mode):
    roots = []
    real = launcher.build_strategies

    def spy(root, *, now):
        roots.append(root)
        return real(root, now=now)

    monkeypatch.setattr(launcher, "build_strategies", spy)
    asked = Ask()
    launcher.main(
        [*mode, "--port", "0"],
        records_factory=lambda: ReadOnlyRecords(tmp_path / "none.sqlite"),
        clock=lambda: pd.Timestamp("2026-10-22T11:40:00Z"),
        serve=asked,
        out=io.StringIO(),
    )
    assert asked.strategies["schema"] == "lab_app_strategies"
    if mode:  # the demo: no folder at all, so nothing real is read
        assert roots == [None] and asked.strategies["problem"]["title"] == "Not in the demo."
    else:  # the conftest's empty reports folder: "no report yet"
        assert roots == [launcher.reports_root()]
        assert asked.strategies["problem"]["title"] == "No strategies report yet."
