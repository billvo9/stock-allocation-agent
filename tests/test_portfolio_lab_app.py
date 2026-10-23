"""
The Portfolio Lab app's launcher (scripts/portfolio_lab_app.py): one pair of servers
per app (the Lab app and the research pages), port choice, this launch's two keys
(made fresh, kept owner-only, passed in the environment, the Lab app's opened in the
address's fragment through osascript's standard input, never printed or logged), the
Lab app's proof that it holds its key, the already-running, half-running and stale
cases, the earlier single-server record retired, a clean stop that never signals a
stranger, a key-free environment, and nothing written in the checkout.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "stock_agent_portfolio_lab_app", REPO / "scripts" / "portfolio_lab_app.py"
)
app = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = app  # dataclasses look their module up while the class is built
_spec.loader.exec_module(app)

KEY, RESEARCH_KEY = "k" * 43, "s" * 43  # the keys a new start makes, in that order
OLD_KEY, OLD_RESEARCH_KEY = "r" * 43, "t" * 43  # a recorded launch's keys
STARTED = "Fri Oct 23 12:00:00 2026"
STRANGER = "Sat Oct 24 09:00:00 2026"  # a live process that reused a recorded pid
NONCE = "n" * 32
RESEARCH_PID, LAB_PID = 4242, 4243  # the spawn order: the research pages, then the Lab app

# Stand-ins for the two start scripts. Each answers its health check only when its
# environment is right, so a real start pins who gets which key; the Lab app proves its
# key with HMAC-SHA256(key, nonce). Each exits after two minutes, should a test be killed.
FAKE_SERVER = """\
import hashlib
import hmac
import http.server
import os
import re
import sys
import threading

threading.Timer(120, os._exit, args=(0,)).start()
port = int(sys.argv[sys.argv.index("--port") + 1])
LAB = {lab!r}
KEY = re.compile(r"[A-Za-z0-9_-]{{32,128}}")
env = os.environ
if LAB:
    right = (
        KEY.fullmatch(env.get("LAB_APP_TOKEN", "")) is not None
        and KEY.fullmatch(env.get("LAB_APP_RESEARCH_KEY", "")) is not None
        and env.get("PYTHONPATH", "").endswith("/src")
        and "PORTFOLIO_LAB_RESEARCH_KEY" not in env
    )
else:
    right = (
        KEY.fullmatch(env.get("PORTFOLIO_LAB_RESEARCH_KEY", "")) is not None
        and not {{"LAB_APP_TOKEN", "LAB_APP_RESEARCH_KEY", "PYTHONPATH"}} & set(env)
    )


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        body = b"fake page"
        if right and LAB and path == "/health" and query.startswith("nonce="):
            key = env["LAB_APP_TOKEN"].encode()
            body = hmac.new(key, query[6:].encode(), hashlib.sha256).hexdigest().encode()
        elif right and not LAB and path == "/_stcore/health":
            body = b"ok"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def checkout(root: Path) -> Path:
    """A minimal project folder: a .venv Python (this interpreter) and both start scripts."""

    folder = root / "checkout"
    (folder / ".venv" / "bin").mkdir(parents=True)
    (folder / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (folder / "scripts").mkdir()
    for server in app.SERVERS:
        lab = server.name == app.LAB.name
        (folder / server.script).write_text(FAKE_SERVER.format(lab=lab), encoding="utf-8")
    return folder


def places(root: Path) -> app.Places:
    return app.Places(checkout(root), root / "private" / "portfolio_lab_app")


class Fake:
    """A scripted operating system: processes whose lives and answers the test controls."""

    def __init__(self, *, answers=(True,), alive=True):
        self.answers_left = list(answers)
        self.asked: list[tuple[int, str, bytes]] = []
        self.alive: dict[int, bool] = {}
        self.starts: dict[int, str] = {}  # a live pid reporting another start time: a stranger
        self.default_alive = alive
        self.next_pid = RESEARCH_PID
        self.spawned: list[tuple[list[str], dict[str, str], Path, Path]] = []
        self.opened: list[str] = []
        self.signals: list[tuple[int, int]] = []
        self.time = 0.0
        self.keys = iter([KEY, RESEARCH_KEY, "j" * 43, "i" * 43, "h" * 43, "g" * 43])

    def system(self, **overrides) -> app.System:
        values = {
            "is_free": lambda port: True,
            "answers": self.answer,
            "new_nonce": lambda: NONCE,
            "process_start": self.process_start,
            "spawn": self.spawn,
            "signal_group": self.signal_group,
            "group_alive": lambda group: False,
            "open_url": self.opened.append,
            "new_key": lambda: next(self.keys),
            "clock": lambda: self.time,
            "sleep": self.sleep,
            "now": lambda: datetime(2026, 10, 23, 16, 0, tzinfo=UTC),
        }
        return app.System(**{**values, **overrides})

    def answer(self, port: int, path: str, expect: bytes) -> bool:
        self.asked.append((port, path, expect))
        return self.answers_left.pop(0) if len(self.answers_left) > 1 else self.answers_left[0]

    def process_start(self, pid: int) -> str | None:
        if not self.alive.get(pid, self.default_alive):
            return None
        return self.starts.get(pid, STARTED)

    def spawn(self, command, env, cwd, log) -> int:
        self.spawned.append((command, env, cwd, log))
        pid, self.next_pid = self.next_pid, self.next_pid + 1
        return pid

    def signal_group(self, group: int, number: int) -> None:
        self.signals.append((group, number))
        self.alive[group] = False

    def sleep(self, seconds: float) -> None:
        self.time += seconds


def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(folder)): (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
    }


def record(where, *, lab=(LAB_PID, 8510), research=(RESEARCH_PID, 8502), checkout=None):
    app.private_folder(where)
    state = {
        "kind": app.KIND,
        "checkout": str(checkout or where.checkout),
        "opened_at": "2026-10-23T15:00:00+00:00",
        "key": OLD_KEY,
        "research_key": OLD_RESEARCH_KEY,
        "servers": {
            "lab": {"pid": lab[0], "port": lab[1], "started": STARTED},
            "research": {"pid": research[0], "port": research[1], "started": STARTED},
        },
    }
    app.write_state(where.state, state)
    return state


# --- ports ----------------------------------------------------------------------------


def test_each_server_prefers_its_own_port():
    assert app.choose_port(app.LAB, lambda port: True) == 8510
    assert app.choose_port(app.RESEARCH, lambda port: True) == 8502
    assert app.choose_port(app.LAB, lambda port: port >= 8513) == 8513


def test_no_free_port_is_an_explained_failure():
    with pytest.raises(app.LabError, match="No free port between 8502 and 8509"):
        app.choose_port(app.RESEARCH, lambda port: False)


def test_a_listening_port_is_not_free():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert app.is_free(server.getsockname()[1]) is False


# --- the environments -------------------------------------------------------------------

CALLER = {
    "TIINGO_API_KEY": "secret-key",
    "FRED_API_KEY": "another-secret",
    "PYTHONPATH": "/elsewhere",
    "PATH": "/attacker/bin:/usr/bin",
    "LAB_APP_TOKEN": "a-key-from-the-caller-is-never-passed-on-xxxx",
    "PORTFOLIO_LAB_RESEARCH_KEY": "a-caller-key-for-the-research-pages-xxxxxx",
    "HOME": "/Users/someone",
    "LANG": "en_GB.UTF-8",
}
BASE = {"HOME", "LANG", "PATH", "PYTHONDONTWRITEBYTECODE"}


def test_no_server_inherits_a_key_or_a_variable_of_the_caller():
    env = app.server_environment(CALLER)
    assert set(env) == BASE
    assert env["PATH"] == app.SYSTEM_PATH and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any("secret" in value or "caller" in value for value in env.values())


def test_the_research_pages_get_only_their_own_key():
    env = app.research_environment(RESEARCH_KEY, CALLER)
    assert set(env) == BASE | {"PORTFOLIO_LAB_RESEARCH_KEY"}
    assert env["PORTFOLIO_LAB_RESEARCH_KEY"] == RESEARCH_KEY
    assert not any("secret" in value or "caller" in value for value in env.values())


def test_the_lab_app_gets_this_checkouts_code_its_key_and_the_link_key(tmp_path):
    where = places(tmp_path)
    env = app.lab_environment(where, KEY, RESEARCH_KEY, CALLER)
    assert set(env) == BASE | {"PYTHONPATH", "LAB_APP_TOKEN", "LAB_APP_RESEARCH_KEY"}
    assert env["PYTHONPATH"] == str(where.checkout / "src")
    assert (env["LAB_APP_TOKEN"], env["LAB_APP_RESEARCH_KEY"]) == (KEY, RESEARCH_KEY)
    assert not any("secret" in value or "caller" in value for value in env.values())


def test_the_health_checks_talk_to_the_loopback_address_only():
    assert app.HOST == "127.0.0.1"
    assert app.lab_url(8510) == "http://127.0.0.1:8510/"
    assert app.keyed_url(8510, KEY) == f"http://127.0.0.1:8510/#key={KEY}"
    assert (app.LAB.health, app.RESEARCH.health) == ("/health", "/_stcore/health")


def test_the_lab_app_must_prove_it_holds_its_key():
    state = {"key": KEY}
    system = Fake().system()
    path, expect = app.health_check(app.LAB, state, system)
    assert path == f"/health?nonce={NONCE}"
    assert expect == app.proof(KEY, NONCE) != app.proof(RESEARCH_KEY, NONCE)
    assert app.health_check(app.RESEARCH, state, system) == ("/_stcore/health", b"ok")


# --- the browser --------------------------------------------------------------------------


def test_the_browser_gets_the_address_on_standard_input_not_as_an_argument(monkeypatch):
    seen = {}

    def run(command, **options):
        seen.update(command=command, **options)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(app.subprocess, "run", run)
    app.open_url(app.keyed_url(8510, KEY))
    assert seen["command"] == ["/usr/bin/osascript", "-"]
    assert KEY not in " ".join(seen["command"])
    assert seen["input"] == f'open location "http://127.0.0.1:8510/#key={KEY}"'
    assert seen["capture_output"] is True and seen["env"] == {"PATH": app.SYSTEM_PATH}


@pytest.mark.parametrize(
    "url",
    ['http://127.0.0.1:8510/#key=" & do shell script "x', "http://evil.example/", "javascript:x"],
)
def test_only_the_labs_own_address_is_ever_quoted_into_the_script(monkeypatch, url):
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: pytest.fail("ran osascript"))
    with pytest.raises(app.LabError, match="couldn't open your browser"):
        app.open_url(url)


@pytest.mark.parametrize("failure", ["exit", "missing"])
def test_a_browser_that_fails_is_reported_without_the_address(monkeypatch, failure):
    def run(command, **options):
        if failure == "missing":
            raise FileNotFoundError(2, "No such file or directory", command[0])
        return subprocess.CompletedProcess(command, 1, "", f"error near {options['input']}")

    monkeypatch.setattr(app.subprocess, "run", run)
    with pytest.raises(app.LabError) as raised:
        app.open_url(app.keyed_url(8510, KEY))
    assert KEY not in str(raised.value)


# --- open -------------------------------------------------------------------------------


def test_open_starts_both_servers_once_and_then_only_shows_the_lab(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()

    assert app.open_lab(where, system) == "http://127.0.0.1:8510/"  # printed: no key
    assert app.open_lab(where, system) == "http://127.0.0.1:8510/"

    assert len(fake.spawned) == 2  # never a second pair
    (research, research_env, cwd, research_log), (lab, lab_env, cwd2, lab_log) = fake.spawned
    assert research == [str(where.python), str(where.script(app.RESEARCH)), "--port", "8502"]
    assert lab == [
        str(where.python),
        str(where.script(app.LAB)),
        "--port",
        "8510",
        "--research-url",
        "http://127.0.0.1:8502/",
    ]
    assert KEY not in " ".join(lab + research) and RESEARCH_KEY not in " ".join(lab + research)
    assert research_env == app.research_environment(RESEARCH_KEY)
    assert lab_env == app.lab_environment(where, KEY, RESEARCH_KEY)
    assert {"LAB_APP_TOKEN", "LAB_APP_RESEARCH_KEY", "PYTHONPATH"}.isdisjoint(research_env)
    assert lab_env["LAB_APP_TOKEN"] == KEY and lab_env["PYTHONPATH"] == str(where.checkout / "src")
    assert cwd == cwd2 == where.checkout
    assert (research_log, lab_log) == (where.log(app.RESEARCH), where.log(app.LAB))
    assert fake.asked[:2] == [
        (8502, "/_stcore/health", b"ok"),
        (8510, f"/health?nonce={NONCE}", app.proof(KEY, NONCE)),
    ]
    assert fake.opened == [f"http://127.0.0.1:8510/#key={KEY}"] * 2
    state = app.read_state(where.state)
    assert (state["key"], state["research_key"]) == (KEY, RESEARCH_KEY)
    assert state["servers"] == {
        "research": {"pid": RESEARCH_PID, "port": 8502, "started": STARTED},
        "lab": {"pid": LAB_PID, "port": 8510, "started": STARTED},
    }


def test_each_start_makes_new_keys(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    assert app.stop_lab(where, system) is True
    fake.alive.clear()
    app.open_lab(where, system)
    assert fake.opened == [
        f"http://127.0.0.1:8510/#key={KEY}",
        f"http://127.0.0.1:8510/#key={'j' * 43}",
    ]
    assert app.read_state(where.state)["research_key"] == "i" * 43


@pytest.mark.parametrize("keys", [("short", RESEARCH_KEY), (KEY, KEY)])
def test_unusable_new_keys_start_nothing(tmp_path, keys):
    where, fake = places(tmp_path), Fake()
    made = iter(keys)
    with pytest.raises(app.LabError, match="could not make this launch's keys"):
        app.open_lab(where, fake.system(new_key=lambda: next(made)))
    assert fake.spawned == []


def test_no_key_is_ever_printed_or_logged(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    assert app.main(["open"], where, fake.system()) == 0
    assert app.main(["status"], where, fake.system()) == 0
    printed = capsys.readouterr()
    log = where.launcher_log.read_text(encoding="utf-8")
    for key in (KEY, RESEARCH_KEY):
        assert key not in printed.out + printed.err and key not in log


def test_a_failing_browser_is_explained_without_the_key_and_the_servers_kept(
    tmp_path, capsys, monkeypatch
):
    where, fake = places(tmp_path), Fake()
    original = app.subprocess.run

    def failing_osascript(command, **options):  # the real open_url, with osascript failing
        return subprocess.CompletedProcess(command, 1, "", f"error: {options.get('input')}")

    monkeypatch.setattr(app.subprocess, "run", failing_osascript)
    assert app.main(["open"], where, fake.system(open_url=app.open_url)) == 1
    monkeypatch.setattr(app.subprocess, "run", original)
    err = capsys.readouterr().err
    assert "couldn't open your browser" in err and KEY not in err
    assert KEY not in where.launcher_log.read_text(encoding="utf-8")
    assert app.read_state(where.state)["servers"]["lab"]["pid"] == LAB_PID
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert len(fake.spawned) == 2


def test_open_waits_for_servers_that_are_still_starting(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False, False, True))
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.time == pytest.approx(2 * app.POLL)


def test_a_server_on_the_lab_port_without_its_key_is_never_shown(tmp_path):
    # A program that took the port first answers "ok" but can't prove the key.
    where, fake = places(tmp_path), Fake()
    system = fake.system(answers=lambda port, path, expect: expect == b"ok" or port != 8510)
    with pytest.raises(app.LabError, match="^The Lab did not answer within 60 seconds"):
        app.open_lab(where, system)
    assert fake.opened == []
    assert sorted(fake.signals) == [(RESEARCH_PID, signal.SIGTERM), (LAB_PID, signal.SIGTERM)]


def test_a_record_whose_processes_are_gone_is_forgotten_and_strangers_left_alone(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where, lab=(999, 8510), research=(998, 8502))
    fake.starts.update({999: STRANGER, 998: STRANGER})  # those pids now belong to strangers

    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert len(fake.spawned) == 2 and fake.signals == []


@pytest.mark.parametrize(("gone", "running_on"), [(999, 998), (998, 999)])
def test_half_a_pair_is_stopped_and_restarted_as_a_whole(tmp_path, gone, running_on):
    where, fake = places(tmp_path), Fake()
    record(where, lab=(999, 8510), research=(998, 8502))
    fake.alive[gone] = False  # one died; its partner runs on

    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.signals == [(running_on, signal.SIGTERM)]
    state = app.read_state(where.state)
    assert (state["servers"]["research"]["pid"], state["servers"]["lab"]["pid"]) == (
        RESEARCH_PID,
        LAB_PID,
    )
    assert (state["key"], state["research_key"]) == (KEY, RESEARCH_KEY)  # fresh, not OLD_*
    assert fake.opened == [f"http://127.0.0.1:8510/#key={KEY}"]


def test_a_running_pair_is_shown_with_its_recorded_key(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where)
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.spawned == [] and fake.opened == [f"http://127.0.0.1:8510/#key={OLD_KEY}"]
    assert (8510, f"/health?nonce={NONCE}", app.proof(OLD_KEY, NONCE)) in fake.asked


def test_when_the_research_pages_die_while_starting_the_lab_app_is_stopped_too(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    system = fake.system()
    seen = {RESEARCH_PID: 0}

    def process_start(pid):
        if pid == RESEARCH_PID:
            seen[pid] += 1
            return STARTED if seen[pid] == 1 else None  # recorded at spawn, gone at the wait
        return fake.process_start(pid)

    system.process_start = process_start
    with pytest.raises(
        app.LabError, match="^The research pages stopped while starting.*research.log"
    ):
        app.open_lab(where, system)
    assert fake.signals == [(LAB_PID, signal.SIGTERM)]
    assert not where.state.exists() and fake.opened == []


def test_when_the_lab_app_never_answers_both_are_stopped(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system(answers=lambda port, path, expect: path == app.RESEARCH.health)
    with pytest.raises(app.LabError, match="^The Lab did not answer within 60 seconds"):
        app.open_lab(where, system)
    assert fake.time >= app.START_TIMEOUT
    assert sorted(fake.signals) == [(RESEARCH_PID, signal.SIGTERM), (LAB_PID, signal.SIGTERM)]
    assert not where.state.exists()


def test_a_process_that_dies_at_once_is_reported_and_its_partner_stopped(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    starts = {RESEARCH_PID: STARTED, LAB_PID: None}  # the Lab app dies at once
    system.process_start = lambda pid: starts.get(pid) if fake.alive.get(pid, True) else None

    with pytest.raises(app.LabError, match="^The Lab stopped right after starting. .*lab.log"):
        app.open_lab(where, system)
    assert fake.signals == [(RESEARCH_PID, signal.SIGTERM)]
    assert not where.state.exists()


def test_when_the_research_pages_die_at_once_the_lab_app_never_starts(tmp_path):
    where, fake = places(tmp_path), Fake()
    with pytest.raises(app.LabError, match="^The research pages stopped right after starting"):
        app.open_lab(where, fake.system(process_start=lambda pid: None))
    assert len(fake.spawned) == 1 and fake.signals == []


def test_a_running_server_that_stops_answering_is_explained(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    record(where)
    with pytest.raises(
        app.LabError, match="^The research pages stopped answering, so Portfolio Lab will quit now"
    ):
        app.open_lab(where, fake.system())
    assert fake.spawned == []
    assert fake.time == pytest.approx(app.REUSE_TIMEOUT)


@pytest.mark.parametrize("missing", ["python", "lab", "research"])
def test_a_checkout_without_its_python_or_a_start_script_is_refused(tmp_path, missing):
    where, fake = places(tmp_path), Fake()
    target = where.python if missing == "python" else where.script(getattr(app, missing.upper()))
    target.unlink()
    with pytest.raises(app.LabError, match="missing"):
        app.open_lab(where, fake.system())
    assert fake.spawned == [] and not where.private.exists()


def test_main_shows_an_expected_failure_on_stderr(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    where.python.unlink()
    assert app.main(["open"], where, fake.system()) == 1
    assert "Python environment (.venv) is missing" in capsys.readouterr().err
    log = where.launcher_log.read_text(encoding="utf-8")
    assert "open failed: The project's Python environment (.venv) is missing." in log


def test_busy_preferred_ports_move_each_server_to_its_next_free_one(tmp_path):
    where, fake = places(tmp_path), Fake()
    url = app.open_lab(where, fake.system(is_free=lambda port: port not in (8502, 8510)))
    assert url == "http://127.0.0.1:8511/"
    assert fake.spawned[0][0][-2:] == ["--port", "8503"]
    assert fake.spawned[1][0][-4:] == ["--port", "8511", "--research-url", "http://127.0.0.1:8503/"]


def test_status_names_both_servers(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    assert app.status(where, system) == (
        "The Lab is running at http://127.0.0.1:8510/, with the research pages at "
        "http://127.0.0.1:8502/ (since 2026-10-23T16:00:00+00:00)."
    )


def test_a_large_server_log_is_kept_once_before_a_new_start(tmp_path, monkeypatch):
    where, fake = places(tmp_path), Fake()
    app.private_folder(where)
    where.log(app.LAB).write_text("x" * 20, encoding="utf-8")
    monkeypatch.setattr(app, "LOG_LIMIT", 10)
    app.open_lab(where, fake.system())
    assert (where.private / "lab.log.1").read_text(encoding="utf-8") == "x" * 20


# --- the earlier, single-server app ------------------------------------------------------


def legacy(where, *, checkout=None):
    app.private_folder(where)
    old = {
        "pid": 4000,
        "port": 8502,
        "started": STARTED,
        "checkout": str(checkout or where.checkout),
    }
    app.write_state(where.state, old)
    return old


def test_the_earlier_apps_server_is_retired_before_the_pair_starts(tmp_path):
    where, fake = places(tmp_path), Fake()
    legacy(where)
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.signals[0] == (4000, signal.SIGTERM)
    assert len(fake.spawned) == 2 and app.read_state(where.state)["kind"] == app.KIND
    assert "retired the earlier app's server (pid 4000)" in where.launcher_log.read_text()


def test_stop_retires_the_earlier_apps_server_too(tmp_path):
    where, fake = places(tmp_path), Fake()
    legacy(where)
    assert app.stop_lab(where, fake.system()) is True
    assert fake.signals == [(4000, signal.SIGTERM)] and not where.state.exists()


def test_the_earlier_app_of_another_build_is_neither_retired_nor_overwritten(tmp_path):
    where, fake = places(tmp_path), Fake()
    old = legacy(where, checkout="/elsewhere")
    with pytest.raises(app.LabError, match="built for another project folder is running"):
        app.open_lab(where, fake.system())
    assert app.stop_lab(where, fake.system()) is False
    assert fake.spawned == [] and fake.signals == []
    assert json.loads(where.state.read_text()) == old


def test_a_gone_earlier_server_is_simply_forgotten(tmp_path):
    where, fake = places(tmp_path), Fake()
    legacy(where)
    fake.starts[4000] = STRANGER  # pid 4000 now belongs to another program
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.signals == []


@pytest.mark.parametrize(
    "content",
    ['{"pid": 4000, "port": 0, "started": "x"}', '{"pid": 4000, "port": 8502}', '{"kind": "x"}'],
)
def test_an_unusable_earlier_record_is_not_taken_for_a_server(tmp_path, content):
    path = tmp_path / "server.json"
    path.write_text(content, encoding="utf-8")
    assert app.read_legacy(path) is None


# --- stop -------------------------------------------------------------------------------


def test_stop_terminates_both_process_groups_and_forgets_them(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    exits = iter([True, False, True, False])  # each group exits after one check: no SIGKILL
    system.group_alive = lambda group: next(exits)

    assert app.stop_lab(where, system) is True
    assert fake.signals == [(RESEARCH_PID, signal.SIGTERM), (LAB_PID, signal.SIGTERM)]
    assert not where.state.exists()
    assert app.status(where, system) == "The Lab is not running."


def test_stop_kills_a_group_that_ignores_sigterm(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system(group_alive=lambda group: True)
    app.open_lab(where, system)

    assert app.stop_lab(where, system) is True
    assert (RESEARCH_PID, signal.SIGKILL) in fake.signals and (
        LAB_PID,
        signal.SIGKILL,
    ) in fake.signals
    assert fake.time >= app.STOP_TIMEOUT


def test_stop_never_signals_a_process_that_reused_the_pid(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where)
    fake.starts.update({LAB_PID: STRANGER, RESEARCH_PID: STRANGER})  # alive, but not ours

    assert app.stop_lab(where, fake.system()) is False
    assert fake.signals == [] and not where.state.exists()


def test_stop_without_a_running_lab_does_nothing(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    assert app.main(["stop"], where, fake.system()) == 0
    assert capsys.readouterr().out.strip() == "The Lab was not running."
    assert fake.signals == []


def test_a_running_lab_of_another_build_is_neither_reused_nor_stopped(tmp_path):
    where, fake = places(tmp_path), Fake()
    state = record(where, checkout="/elsewhere")

    with pytest.raises(app.LabError, match="built for another project folder is running"):
        app.open_lab(where, fake.system())
    assert app.stop_lab(where, fake.system()) is False
    assert fake.spawned == [] and fake.signals == []
    assert app.read_state(where.state) == state


def test_another_builds_pair_that_is_gone_no_longer_blocks_opening(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where, checkout="/elsewhere")
    fake.alive.update({LAB_PID: False, RESEARCH_PID: False})  # e.g. after a restart
    assert app.stop_lab(where, fake.system()) is False and not where.state.exists()

    record(where, checkout="/elsewhere")
    fake.alive.clear()
    fake.starts.update({LAB_PID: STRANGER, RESEARCH_PID: STRANGER})
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert len(fake.spawned) == 2 and fake.signals == []


def test_another_builds_pair_with_one_server_running_is_left_alone(tmp_path):
    where, fake = places(tmp_path), Fake()
    state = record(where, checkout="/elsewhere")
    fake.alive[RESEARCH_PID] = False
    with pytest.raises(app.LabError, match="built for another project folder is running"):
        app.open_lab(where, fake.system())
    assert app.stop_lab(where, fake.system()) is False
    assert app.read_state(where.state) == state and fake.signals == []


def test_an_answer_on_the_port_does_not_count_once_our_process_is_gone():
    fake = Fake(alive=False)  # another server answers on the port, ours exited
    state = {"key": KEY, "servers": {"lab": {"pid": LAB_PID, "port": 8510, "started": STARTED}}}
    assert app.wait_until_answering(app.LAB, state, fake.system(), timeout=5) is False


def test_a_planted_private_folder_is_refused(tmp_path):
    where = places(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    where.private.parent.mkdir()
    where.private.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(app.LabError, match="is not a folder of yours"):
        app.private_folder(where)


GOOD = {
    "kind": "lab-app",
    "checkout": "/c",
    "key": KEY,
    "research_key": RESEARCH_KEY,
    "servers": {
        "lab": {"pid": 7, "port": 8510, "started": "x"},
        "research": {"pid": 8, "port": 8502, "started": "x"},
    },
}
LAB_RECORD = GOOD["servers"]["lab"]


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "something else"},
        {"key": "short"},
        {"key": KEY + "#"},
        {"research_key": None},
        {"research_key": KEY},  # the two keys must differ
        {"servers": {"lab": LAB_RECORD}},
        {"servers": {**GOOD["servers"], "extra": LAB_RECORD}},
        {"servers": {**GOOD["servers"], "lab": {**LAB_RECORD, "pid": 1}}},
        {"servers": {**GOOD["servers"], "lab": {**LAB_RECORD, "port": True}}},
        {"servers": {**GOOD["servers"], "lab": {**LAB_RECORD, "pid": 2**40}}},
        {"servers": {**GOOD["servers"], "lab": {"pid": 7, "port": 8510}}},
    ],
)
def test_an_unusable_record_is_ignored(tmp_path, change):
    path = tmp_path / "server.json"
    path.write_text(json.dumps({**GOOD, **change}), encoding="utf-8")
    assert app.read_state(path) is None
    path.write_text(json.dumps(GOOD), encoding="utf-8")
    assert app.read_state(path) == GOOD


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_an_unreadable_record_is_ignored(tmp_path, content):
    path = tmp_path / "server.json"
    path.write_text(content, encoding="utf-8")
    assert app.read_state(path) is None and app.read_legacy(path) is None


def test_the_launcher_the_server_and_the_page_agree_on_the_keys_and_the_health_check():
    from stock_agent.app import run
    from stock_agent.app.web import server

    terminal = importlib.util.spec_from_file_location(
        "stock_agent_run_lab_app_parity", REPO / "scripts" / "run_lab_app.py"
    )
    run_lab_app = importlib.util.module_from_spec(terminal)
    terminal.loader.exec_module(run_lab_app)
    assert app.KEY.pattern == server.KEY.pattern == run_lab_app.KEY.pattern
    assert app.KEY_VARIABLE == run_lab_app.KEY_VARIABLE == "LAB_APP_TOKEN"
    assert app.LINK_KEY_VARIABLE == run_lab_app.LINK_KEY_VARIABLE == "LAB_APP_RESEARCH_KEY"
    assert app.RESEARCH_KEY_VARIABLE == run.RESEARCH_KEY_VARIABLE
    page = (server.STATIC / "app.js").read_text(encoding="utf-8")
    assert "const KEY = /^#key=(" + server.KEY.pattern + ")$/;" in page
    assert app.LAB.health == server.HEALTH
    lab = server.LabApp(port=8510, api={}, static={}, token=KEY)
    answer = lab.handle("GET", f"/health?nonce={NONCE}", "127.0.0.1:8510")
    assert answer.body == app.proof(KEY, NONCE)


# --- where it writes ----------------------------------------------------------------------


def test_the_private_folder_is_outside_the_checkout_and_owner_only(tmp_path):
    assert not app.PRIVATE_DIR.is_relative_to(app.CHECKOUT)
    assert app.PRIVATE_DIR.parent == Path.home() / "stock-agent-private"

    where, fake = places(tmp_path), Fake()
    app.open_lab(where, fake.system())
    assert stat.S_IMODE(where.private.stat().st_mode) == 0o700
    for path in (where.state, where.lock, where.launcher_log):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600  # the record holds the keys


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_real_servers_start_once_stop_and_write_nothing_in_the_checkout(tmp_path, monkeypatch):
    # One kernel-chosen port each, so a concurrent run can't collide with the real ranges.
    research_port, lab_port = _free_port(), _free_port()
    monkeypatch.setattr(
        app,
        "RESEARCH",
        dataclasses.replace(app.RESEARCH, ports=range(research_port, research_port + 1)),
    )
    monkeypatch.setattr(
        app, "LAB", dataclasses.replace(app.LAB, ports=range(lab_port, lab_port + 1))
    )
    monkeypatch.setattr(app, "SERVERS", (app.RESEARCH, app.LAB))
    where = places(tmp_path)
    before = snapshot(where.checkout)
    opened: list[str] = []
    system = app.System(open_url=opened.append)

    url = app.open_lab(where, system)
    try:
        state = app.read_state(where.state)
        lab, research = state["servers"]["lab"], state["servers"]["research"]
        # The Lab app proves its key (its environment held it); the research pages answer.
        nonce = app.new_nonce()
        assert app.answers(lab["port"], f"/health?nonce={nonce}", app.proof(state["key"], nonce))
        assert not app.answers(lab["port"], f"/health?nonce={nonce}", app.proof("x" * 43, nonce))
        assert app.answers(research["port"], app.RESEARCH.health)
        assert app.open_lab(where, system) == url  # the second open reuses the pair
        assert app.read_state(where.state)["servers"] == state["servers"]
        assert opened == [f"{url}#key={state['key']}"] * 2
        assert len(state["key"]) >= 43 and state["key"] != state["research_key"]
        for server in app.SERVERS:
            assert stat.S_IMODE(where.log(server).stat().st_mode) == 0o600
    finally:
        stopped = app.stop_lab(where, system)
    assert stopped is True
    assert not app.group_alive(lab["pid"]) and not app.group_alive(research["pid"])
    assert not app.answers(research["port"], app.RESEARCH.health)
    assert snapshot(where.checkout) == before
    assert sorted(os.listdir(where.private)) == [
        "lab.log",
        "launcher.lock",
        "launcher.log",
        "research.log",
    ]
