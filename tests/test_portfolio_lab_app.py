"""
The Portfolio Lab app's launcher (scripts/portfolio_lab_app.py): one server per
app, port choice, the already-running and stale cases, a clean stop that never
signals a stranger, a key-free environment, and nothing written in the checkout.
"""

from __future__ import annotations

import importlib.util
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

# A stand-in for scripts/run_portfolio_lab.py: answers the health check like Streamlit.
FAKE_LAB = """\
import http.server
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok" if self.path == "/_stcore/health" else b"fake lab"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def checkout(root: Path, *, lab: str = FAKE_LAB) -> Path:
    """A minimal project folder: a .venv Python (this interpreter) and a Lab start script."""

    folder = root / "checkout"
    (folder / ".venv" / "bin").mkdir(parents=True)
    (folder / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (folder / "scripts").mkdir()
    (folder / "scripts" / "run_portfolio_lab.py").write_text(lab, encoding="utf-8")
    return folder


def places(root: Path) -> app.Places:
    return app.Places(checkout(root), root / "private" / "portfolio_lab_app")


class Fake:
    """A scripted operating system: one process (pid 4242) whose life the test controls."""

    def __init__(self, *, answers=(True,), alive=True, started="Wed Oct 21 12:00:00 2026"):
        self.answers_left = list(answers)
        self.alive = alive
        self.started = started
        self.spawned: list[tuple[list[str], dict[str, str], Path, Path]] = []
        self.opened: list[str] = []
        self.signals: list[tuple[int, int]] = []
        self.group_checks = 0
        self.time = 0.0

    def system(self, **overrides) -> app.System:
        values = {
            "is_free": lambda port: True,
            "answers": self.answer,
            "process_start": lambda pid: self.started if self.alive else None,
            "spawn": self.spawn,
            "signal_group": lambda group, number: self.signals.append((group, number)),
            "group_alive": self.group_alive,
            "open_url": self.opened.append,
            "clock": lambda: self.time,
            "sleep": self.sleep,
            "now": lambda: datetime(2026, 10, 21, 16, 0, tzinfo=UTC),
        }
        return app.System(**{**values, **overrides})

    def answer(self, port: int) -> bool:
        return self.answers_left.pop(0) if len(self.answers_left) > 1 else self.answers_left[0]

    def spawn(self, command, env, cwd, log) -> int:
        self.spawned.append((command, env, cwd, log))
        return 4242

    def group_alive(self, group: int) -> bool:
        self.group_checks += 1
        return self.alive

    def sleep(self, seconds: float) -> None:
        self.time += seconds


def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(folder)): (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
    }


# --- port choice ----------------------------------------------------------------------


def test_the_preferred_port_is_8502_when_it_is_free():
    assert app.choose_port(lambda port: True) == 8502


def test_busy_ports_are_skipped():
    assert app.choose_port(lambda port: port >= 8505) == 8505


def test_no_free_port_is_an_explained_failure():
    with pytest.raises(app.LabError, match="No free port between 8502 and 8599"):
        app.choose_port(lambda port: False)


def test_a_listening_port_is_not_free():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert app.is_free(port) is False


# --- the server's environment ---------------------------------------------------------


def test_the_server_inherits_no_key_and_no_caller_variable():
    caller = {
        "TIINGO_API_KEY": "secret-key",
        "FRED_API_KEY": "another-secret",
        "PYTHONPATH": "/elsewhere",
        "PATH": "/attacker/bin:/usr/bin",
        "HOME": "/Users/someone",
        "LANG": "en_GB.UTF-8",
    }
    env = app.server_environment(caller)
    assert set(env) == {"HOME", "LANG", "PATH", "PYTHONDONTWRITEBYTECODE"}
    assert env["PATH"] == app.SYSTEM_PATH
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any("secret" in value for value in env.values())


def test_the_health_check_talks_to_the_loopback_address_only():
    assert app.HOST == "127.0.0.1"
    assert app.lab_url(8502) == "http://127.0.0.1:8502/"


# --- open -----------------------------------------------------------------------------


def test_open_starts_the_lab_once_and_then_only_shows_it(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()

    assert app.open_lab(where, system) == "http://127.0.0.1:8502/"
    assert app.open_lab(where, system) == "http://127.0.0.1:8502/"

    assert len(fake.spawned) == 1  # never a second server
    command, env, cwd, log = fake.spawned[0]
    assert command == [str(where.python), str(where.lab_script), "--port", "8502"]
    assert env == app.server_environment()
    assert (cwd, log) == (where.checkout, where.server_log)
    assert fake.opened == ["http://127.0.0.1:8502/"] * 2
    state = app.read_state(where.state)
    assert (state["pid"], state["port"], state["started"]) == (4242, 8502, fake.started)


def test_open_waits_for_a_server_that_is_still_starting(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False, False, True))
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8502/"
    assert fake.time == pytest.approx(2 * app.POLL)


def test_open_forgets_a_record_whose_process_is_gone(tmp_path):
    where, fake = places(tmp_path), Fake()
    app.private_folder(where)
    app.write_state(where.state, {"pid": 999, "port": 8510, "started": "Mon Oct 19 09:00:00"})
    fake.started = "Wed Oct 21 12:00:00 2026"  # pid 999 now reports another start time

    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8502/"
    assert len(fake.spawned) == 1
    assert fake.signals == []  # the stranger with the reused pid is left alone


def test_a_server_that_exits_while_starting_is_reported_with_its_log(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    system = fake.system()
    starts = iter([fake.started, None])  # running when recorded, gone at the first check
    system.process_start = lambda pid: next(starts, None)

    with pytest.raises(app.LabError, match="stopped while starting. Choose Show Logs"):
        app.open_lab(where, system)
    assert fake.signals == []
    assert not where.state.exists()
    assert fake.opened == []


def test_a_server_that_never_answers_is_stopped_and_reported(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    system = fake.system()
    system.group_alive = lambda group: False

    with pytest.raises(app.LabError, match="did not answer within 60 seconds, so it was stopped"):
        app.open_lab(where, system)
    assert fake.time >= app.START_TIMEOUT
    assert fake.signals == [(4242, signal.SIGTERM)]
    assert not where.state.exists()


def test_a_process_that_dies_at_once_is_reported(tmp_path):
    where, fake = places(tmp_path), Fake(alive=False)
    with pytest.raises(app.LabError, match="stopped as soon as it started"):
        app.open_lab(where, fake.system())
    assert not where.state.exists()


def test_a_running_server_that_stops_answering_is_explained(tmp_path):
    where, fake = places(tmp_path), Fake(answers=(False,))
    app.private_folder(where)
    record = {"pid": 4242, "port": 8502, "started": fake.started, "checkout": str(where.checkout)}
    app.write_state(where.state, record)
    with pytest.raises(app.LabError, match="Portfolio Lab will stop it now"):
        app.open_lab(where, fake.system())
    assert fake.spawned == []
    assert fake.time == pytest.approx(app.REUSE_TIMEOUT)


@pytest.mark.parametrize("missing", ["python", "lab_script"])
def test_a_checkout_without_its_python_or_start_script_is_refused(tmp_path, missing):
    where, fake = places(tmp_path), Fake()
    getattr(where, missing).unlink()
    with pytest.raises(app.LabError, match="missing"):
        app.open_lab(where, fake.system())
    assert fake.spawned == []
    assert not where.private.exists()


def test_main_shows_an_expected_failure_on_stderr(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    where.python.unlink()
    assert app.main(["open"], where, fake.system()) == 1
    assert "Python environment (.venv) is missing" in capsys.readouterr().err
    log = where.launcher_log.read_text(encoding="utf-8")
    assert "open failed: The project's Python environment (.venv) is missing." in log


def test_a_failing_browser_is_explained_and_the_server_kept_for_the_next_open(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()

    def no_browser(url):
        raise subprocess.CalledProcessError(1, ["/usr/bin/open", url])

    assert app.main(["open"], where, fake.system(open_url=no_browser)) == 1
    assert "The Lab launcher could not finish" in capsys.readouterr().err
    assert app.read_state(where.state)["pid"] == 4242  # the next open reuses it
    assert app.open_lab(where, fake.system()) == "http://127.0.0.1:8502/"
    assert len(fake.spawned) == 1


def test_a_busy_8502_moves_the_lab_to_the_next_free_port(tmp_path):
    where, fake = places(tmp_path), Fake()
    url = app.open_lab(where, fake.system(is_free=lambda port: port != 8502))
    assert url == "http://127.0.0.1:8503/"
    assert fake.spawned[0][0][-2:] == ["--port", "8503"]


def test_status_names_the_running_lab(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    assert app.status(where, system) == (
        "The Lab is running at http://127.0.0.1:8502/ (since 2026-10-21T16:00:00+00:00)."
    )


def test_a_large_server_log_is_kept_once_before_a_new_start(tmp_path, monkeypatch):
    where, fake = places(tmp_path), Fake()
    app.private_folder(where)
    where.server_log.write_text("x" * 20, encoding="utf-8")
    monkeypatch.setattr(app, "LOG_LIMIT", 10)
    app.open_lab(where, fake.system())
    assert (where.private / "lab.log.1").read_text(encoding="utf-8") == "x" * 20


# --- stop -----------------------------------------------------------------------------


def test_stop_terminates_the_process_group_and_forgets_it(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)
    exits = iter([True, False])
    system.group_alive = lambda group: next(exits)

    assert app.stop_lab(where, system) is True
    assert fake.signals == [(4242, signal.SIGTERM)]
    assert not where.state.exists()
    assert app.status(where, system) == "The Lab is not running."


def test_stop_kills_a_group_that_ignores_sigterm(tmp_path):
    where, fake = places(tmp_path), Fake()
    system = fake.system()
    app.open_lab(where, system)

    assert app.stop_lab(where, system) is True
    assert fake.signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert fake.time >= app.STOP_TIMEOUT


def test_stop_never_signals_a_process_that_reused_the_pid(tmp_path):
    where, fake = places(tmp_path), Fake()
    app.private_folder(where)
    app.write_state(where.state, {"pid": 4242, "port": 8502, "started": "Mon Oct 19 09:00:00"})

    assert app.stop_lab(where, fake.system()) is False
    assert fake.signals == []
    assert not where.state.exists()


def test_stop_without_a_running_lab_does_nothing(tmp_path, capsys):
    where, fake = places(tmp_path), Fake()
    assert app.main(["stop"], where, fake.system()) == 0
    assert capsys.readouterr().out.strip() == "The Lab was not running."
    assert fake.signals == []


def test_a_running_lab_of_another_build_is_neither_reused_nor_stopped(tmp_path):
    where, fake = places(tmp_path), Fake()
    app.private_folder(where)
    record = {"pid": 4242, "port": 8502, "started": fake.started, "checkout": "/elsewhere"}
    app.write_state(where.state, record)

    with pytest.raises(app.LabError, match="built for another project folder is running"):
        app.open_lab(where, fake.system())
    assert app.stop_lab(where, fake.system()) is False
    assert fake.spawned == [] and fake.signals == []
    assert app.read_state(where.state) == record


def test_an_answer_on_the_port_does_not_count_once_our_process_is_gone(tmp_path):
    fake = Fake(alive=False)  # another server answers on the port, ours exited
    state = {"pid": 4242, "port": 8502, "started": fake.started}
    assert app.wait_until_answering(state, fake.system(), timeout=5) is False


def test_a_planted_private_folder_is_refused(tmp_path):
    where = places(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    where.private.parent.mkdir()
    where.private.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(app.LabError, match="is not a folder of yours"):
        app.private_folder(where)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"pid": 1, "port": 8502, "started": "x"}',
        '{"pid": 7, "port": 0}',
        '{"pid": 7, "port": true, "started": "x"}',
        '{"pid": 99999999999, "port": 8502, "started": "x"}',
    ],
)
def test_an_unusable_record_is_ignored(tmp_path, content):
    path = tmp_path / "server.json"
    path.write_text(content, encoding="utf-8")
    assert app.read_state(path) is None


# --- where it writes ------------------------------------------------------------------


def test_the_private_folder_is_outside_the_checkout_and_owner_only(tmp_path):
    assert not app.PRIVATE_DIR.is_relative_to(app.CHECKOUT)
    assert app.PRIVATE_DIR.parent == Path.home() / "stock-agent-private"

    where, fake = places(tmp_path), Fake()
    app.open_lab(where, fake.system())
    assert stat.S_IMODE(where.private.stat().st_mode) == 0o700
    for path in (where.state, where.lock, where.launcher_log):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_real_lab_process_starts_once_stops_and_writes_nothing_in_the_checkout(tmp_path):
    where = places(tmp_path)
    before = snapshot(where.checkout)
    opened: list[str] = []
    system = app.System(open_url=opened.append)

    url = app.open_lab(where, system)
    state = app.read_state(where.state)
    port = state["port"]
    try:
        assert app.answers(port)
        assert app.open_lab(where, system) == url  # the second open reuses the server
        assert app.read_state(where.state)["pid"] == state["pid"]
        assert opened == [url, url]
        assert stat.S_IMODE(where.server_log.stat().st_mode) == 0o600
    finally:
        assert app.stop_lab(where, system) is True

    assert not app.group_alive(state["pid"])
    assert not app.answers(port)
    assert snapshot(where.checkout) == before
    assert sorted(os.listdir(where.private)) == ["lab.log", "launcher.lock", "launcher.log"]
