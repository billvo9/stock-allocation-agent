"""
The Portfolio Lab app's launcher (scripts/portfolio_lab_app.py): one pair of servers
per app (the Lab app and the research pages), port choice, this launch's key (made
fresh, kept owner-only, passed in the environment and the address's fragment, never
printed or logged), the already-running, half-running and stale cases, the earlier
single-server record retired, a clean stop that never signals a stranger, a key-free
environment, and nothing written in the checkout.
"""

from __future__ import annotations

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

KEY = "k" * 43
STARTED = "Fri Oct 23 12:00:00 2026"
RESEARCH_PID, LAB_PID = 4242, 4243  # the spawn order: the research pages, then the Lab app

# Stand-ins for the two start scripts: each answers its own health check with "ok".
FAKE_SERVER = """\
import http.server
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])
HEALTH = {health!r}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok" if self.path == HEALTH else b"fake page"
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
        script = folder / server.script
        script.write_text(FAKE_SERVER.format(health=server.health), encoding="utf-8")
    return folder


def places(root: Path) -> app.Places:
    return app.Places(checkout(root), root / "private" / "portfolio_lab_app")


class Fake:
    """A scripted operating system: processes whose lives and answers the test controls."""

    def __init__(self, *, answers=(True,), alive=True):
        self.answers_left = list(answers)
        self.alive: dict[int, bool] = {}
        self.default_alive = alive
        self.next_pid = RESEARCH_PID
        self.spawned: list[tuple[list[str], dict[str, str], Path, Path]] = []
        self.opened: list[str] = []
        self.signals: list[tuple[int, int]] = []
        self.time = 0.0
        self.keys = iter([KEY, "j" * 43, "i" * 43])

    def system(self, **overrides) -> app.System:
        values = {
            "is_free": lambda port: True,
            "answers": self.answer,
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

    def answer(self, port: int, path: str) -> bool:
        return self.answers_left.pop(0) if len(self.answers_left) > 1 else self.answers_left[0]

    def process_start(self, pid: int) -> str | None:
        return STARTED if self.alive.get(pid, self.default_alive) else None

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
        "key": KEY,
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
    "HOME": "/Users/someone",
    "LANG": "en_GB.UTF-8",
}


def test_the_research_pages_inherit_no_key_and_no_caller_variable():
    env = app.server_environment(CALLER)
    assert set(env) == {"HOME", "LANG", "PATH", "PYTHONDONTWRITEBYTECODE"}
    assert env["PATH"] == app.SYSTEM_PATH and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any("secret" in value or "caller" in value for value in env.values())


def test_the_lab_app_gets_this_checkouts_code_and_only_this_launchs_key(tmp_path):
    where = places(tmp_path)
    env = app.lab_environment(where, KEY, CALLER)
    assert set(env) == {
        "HOME",
        "LANG",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "LAB_APP_TOKEN",
    }
    assert env["PYTHONPATH"] == str(where.checkout / "src") and env["LAB_APP_TOKEN"] == KEY
    assert not any("secret" in value or "caller" in value for value in env.values())


def test_the_health_checks_talk_to_the_loopback_address_only():
    assert app.HOST == "127.0.0.1"
    assert app.lab_url(8510) == "http://127.0.0.1:8510/"
    assert app.keyed_url(8510, KEY) == f"http://127.0.0.1:8510/#key={KEY}"
    assert (app.LAB.health, app.RESEARCH.health) == ("/health", "/_stcore/health")


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
    assert KEY not in " ".join(lab + research)  # never on a command line
    assert research_env == app.server_environment()
    assert lab_env == app.lab_environment(where, KEY)
    assert cwd == cwd2 == where.checkout
    assert (research_log, lab_log) == (where.log(app.RESEARCH), where.log(app.LAB))
    assert fake.opened == [f"http://127.0.0.1:8510/#key={KEY}"] * 2
    state = app.read_state(where.state)
    assert state["key"] == KEY
    assert state["servers"] == {
        "research": {"pid": RESEARCH_PID, "port": 8502, "started": STARTED},
        "lab": {"pid": LAB_PID, "port": 8510, "started": STARTED},
    }


def test_each_start_makes_a_new_key(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    exits = iter([False] * 10)
    system.group_alive = lambda group: next(exits)
    assert app.stop_lab(where, system) is True
    fake.alive.clear()
    app.open_lab(where, system)
    assert fake.opened == [
        f"http://127.0.0.1:8510/#key={KEY}",
        f"http://127.0.0.1:8510/#key={'j' * 43}",
    ]


def test_the_key_is_never_printed_or_logged(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    assert app.main(["open"], where, fake.system()) == 0
    assert app.main(["status"], where, fake.system()) == 0
    printed = capsys.readouterr()
    assert KEY not in printed.out + printed.err
    assert KEY not in where.launcher_log.read_text(encoding="utf-8")


def test_open_waits_for_servers_that_are_still_starting(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False, False, True))
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.time == pytest.approx(2 * app.POLL)


def test_a_record_whose_processes_are_gone_is_forgotten_and_strangers_left_alone(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where, lab=(999, 8510), research=(998, 8502))
    fake.alive.update({999: False, 998: False})  # those pids now belong to strangers

    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert len(fake.spawned) == 2 and fake.signals == []


def test_half_a_pair_is_stopped_and_restarted_as_a_whole(tmp_path):
    where, fake = places(tmp_path), Fake()
    record(where, lab=(999, 8510), research=(998, 8502))
    fake.alive.update({999: False, 998: True})  # the Lab app died, the research pages run on

    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.signals[0] == (998, signal.SIGTERM)
    assert len(fake.spawned) == 2


def test_a_server_that_exits_while_starting_is_reported_with_its_log(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    system = fake.system()
    starts = iter([STARTED, STARTED, None])  # both recorded, the research pages gone at once
    system.process_start = lambda pid: next(starts, None)

    with pytest.raises(app.LabError, match="research pages stopped while starting. .*research.log"):
        app.open_lab(where, system)
    assert not where.state.exists() and fake.opened == []


def test_a_server_that_never_answers_stops_both_and_is_reported(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    with pytest.raises(app.LabError, match="did not answer within 60 seconds"):
        app.open_lab(where, fake.system())
    assert fake.time >= app.START_TIMEOUT
    assert {group for group, number in fake.signals if number == signal.SIGTERM} == {
        RESEARCH_PID,
        LAB_PID,
    }
    assert not where.state.exists()


def test_a_process_that_dies_at_once_is_reported_and_its_partner_stopped(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    starts = {RESEARCH_PID: STARTED, LAB_PID: None}  # the Lab app dies at once
    system.process_start = lambda pid: starts.get(pid) if fake.alive.get(pid, True) else None

    with pytest.raises(app.LabError, match="The Lab stopped as soon as it started. .*lab.log"):
        app.open_lab(where, system)
    assert (RESEARCH_PID, signal.SIGTERM) in fake.signals
    assert not where.state.exists()


def test_a_running_server_that_stops_answering_is_explained(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    record(where)
    with pytest.raises(app.LabError, match="stopped answering. Portfolio Lab will stop it now"):
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


def test_a_failing_browser_is_explained_and_the_servers_kept_for_the_next_open(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()

    def no_browser(url):
        raise subprocess.CalledProcessError(1, ["/usr/bin/open", "(address)"])

    assert app.main(["open"], where, fake.system(open_url=no_browser)) == 1
    assert "The Lab launcher could not finish" in capsys.readouterr().err
    assert app.read_state(where.state)["servers"]["lab"]["pid"] == LAB_PID
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert len(fake.spawned) == 2


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
    assert fake.signals[0] == (4000, signal.SIGTERM) and not where.state.exists()


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
    fake.alive[4000] = False
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8510/"
    assert fake.signals == []


# --- stop -------------------------------------------------------------------------------


def test_stop_terminates_both_process_groups_and_forgets_them(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)

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
    fake.alive.update({LAB_PID: False, RESEARCH_PID: False})

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


def test_an_answer_on_the_port_does_not_count_once_our_process_is_gone():
    fake = Fake(alive=False)  # another server answers on the port, ours exited
    state = {"pid": LAB_PID, "port": 8510, "started": STARTED}
    assert app.wait_until_answering(state, "/health", fake.system(), timeout=5) is False


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
    "servers": {
        "lab": {"pid": 7, "port": 8510, "started": "x"},
        "research": {"pid": 8, "port": 8502, "started": "x"},
    },
}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        json.dumps({**GOOD, "kind": "something else"}),
        json.dumps({**GOOD, "key": "short"}),
        json.dumps({**GOOD, "key": KEY + "#"}),
        json.dumps({**GOOD, "servers": {"lab": GOOD["servers"]["lab"]}}),
        json.dumps({**GOOD, "servers": {**GOOD["servers"], "extra": GOOD["servers"]["lab"]}}),
        json.dumps(
            {
                **GOOD,
                "servers": {**GOOD["servers"], "lab": {"pid": 1, "port": 8510, "started": "x"}},
            }
        ),
        json.dumps(
            {
                **GOOD,
                "servers": {**GOOD["servers"], "lab": {"pid": 7, "port": True, "started": "x"}},
            }
        ),
        json.dumps(
            {
                **GOOD,
                "servers": {**GOOD["servers"], "lab": {"pid": 2**40, "port": 1, "started": "x"}},
            }
        ),
    ],
)
def test_an_unusable_record_is_ignored(tmp_path, content):
    path = tmp_path / "server.json"
    path.write_text(content, encoding="utf-8")
    assert app.read_state(path) is None
    path.write_text(json.dumps(GOOD), encoding="utf-8")
    assert app.read_state(path) == GOOD


# --- where it writes ----------------------------------------------------------------------


def test_the_private_folder_is_outside_the_checkout_and_owner_only(tmp_path):
    assert not app.PRIVATE_DIR.is_relative_to(app.CHECKOUT)
    assert app.PRIVATE_DIR.parent == Path.home() / "stock-agent-private"

    where, fake = places(tmp_path), Fake()
    app.open_lab(where, fake.system())
    assert stat.S_IMODE(where.private.stat().st_mode) == 0o700
    for path in (where.state, where.lock, where.launcher_log):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600  # the record holds the key


def test_real_servers_start_once_stop_and_write_nothing_in_the_checkout(tmp_path):
    where = places(tmp_path)
    before = snapshot(where.checkout)
    opened: list[str] = []
    system = app.System(open_url=opened.append)

    url = app.open_lab(where, system)
    state = app.read_state(where.state)
    lab, research = state["servers"]["lab"], state["servers"]["research"]
    try:
        assert app.answers(lab["port"], app.LAB.health)
        assert app.answers(research["port"], app.RESEARCH.health)
        assert app.open_lab(where, system) == url  # the second open reuses the pair
        assert app.read_state(where.state)["servers"] == state["servers"]
        assert opened == [f"{url}#key={state['key']}"] * 2
        assert len(state["key"]) >= 43
        for server in app.SERVERS:
            assert stat.S_IMODE(where.log(server).stat().st_mode) == 0o600
    finally:
        assert app.stop_lab(where, system) is True

    assert not app.group_alive(lab["pid"]) and not app.group_alive(research["pid"])
    assert not app.answers(lab["port"], app.LAB.health)
    assert snapshot(where.checkout) == before
    assert sorted(os.listdir(where.private)) == [
        "lab.log",
        "launcher.lock",
        "launcher.log",
        "research.log",
    ]
