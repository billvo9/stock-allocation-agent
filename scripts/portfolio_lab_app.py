"""
Open the Portfolio Lab like a Mac app: start its local server once, then show it.

"Portfolio Lab.app" (built by scripts/build_mac_app.py) runs this script from
the checkout with the checkout's own Python, in isolated mode:

    .venv/bin/python -I -B scripts/portfolio_lab_app.py open | stop | status

- open: when the server this app started is running, only show it again (a
  new browser tab); never start a second one. Otherwise start
  scripts/run_portfolio_lab.py on 127.0.0.1, on port 8502 when it is free (the
  terminal launcher's default, so the browser keeps the Lab's theme setting)
  or the next free port up to 8599. Wait until the server's health endpoint
  answers (up to 60 seconds; it usually takes about one), then open the Lab in
  the default browser.
- stop: stop the server this app started, and nothing else. The recorded
  process must still be the same one (process id and start time) before its
  process group is sent SIGTERM; SIGKILL follows only after 10 seconds.
- status: say whether the Lab is running, and where.

What it never does:
- read, pass or ask for a key or any other secret. The server inherits only a
  short allowlist of variables (`PASSED_VARIABLES`): no TIINGO_API_KEY and no
  other variable of the caller;
- make a network request beyond this Mac. The health check connects to
  127.0.0.1 directly, so no proxy setting applies;
- start a fetch, a forecast or any scheduled job. The only process it starts is
  the Lab. The unattended evening job (`local.stock-lab.daily`) is separate and
  untouched;
- write inside the checkout. The evening job refuses a checkout with untracked
  files, so the state, lock and logs live in
  ~/stock-agent-private/portfolio_lab_app/ (owner-only permissions), and the
  server runs with PYTHONDONTWRITEBYTECODE=1 so no bytecode lands there either.

A failure exits with status 1 and a short explanation on stderr, which the app
shows in a macOS dialog; it is also recorded in launcher.log. Standard library
only.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path.home() / "stock-agent-private"
PRIVATE_DIR = PRIVATE_ROOT / "portfolio_lab_app"
HOST = "127.0.0.1"
PORTS = range(8502, 8600)  # the first free one is used; 8502 is run_portfolio_lab's default
HEALTH_PATH = "/_stcore/health"  # Streamlit answers 200 "ok" once it accepts browsers
START_TIMEOUT = 60.0  # seconds for a new server to answer (it usually takes about one)
REUSE_TIMEOUT = 10.0  # seconds for a recorded server that is still starting
STOP_TIMEOUT = 10.0  # seconds after SIGTERM before SIGKILL
POLL = 0.25
LOG_LIMIT = 5_000_000  # bytes; a larger server log is kept once as lab.log.1
# The only variables the server inherits from the caller. Everything else is dropped,
# so no key (TIINGO_API_KEY among them) can reach the Lab through this launcher.
PASSED_VARIABLES = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
# Files in the private folder are opened without following a symlink planted in their place.
PRIVATE_FLAGS = os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC


SEE_LOG = "Choose Show Logs and open lab.log to see why."


class LabError(Exception):
    """An expected failure, shown to the owner as it is worded."""


@dataclass(frozen=True)
class Places:
    """The checkout the Lab runs from and the private folder this launcher writes to."""

    checkout: Path
    private: Path

    @property
    def python(self) -> Path:
        return self.checkout / ".venv" / "bin" / "python"

    @property
    def lab_script(self) -> Path:
        return self.checkout / "scripts" / "run_portfolio_lab.py"

    @property
    def state(self) -> Path:
        return self.private / "server.json"

    @property
    def lock(self) -> Path:
        return self.private / "launcher.lock"

    @property
    def server_log(self) -> Path:
        return self.private / "lab.log"

    @property
    def launcher_log(self) -> Path:
        return self.private / "launcher.log"


def is_free(port: int) -> bool:
    """Whether a server could listen on 127.0.0.1:port now."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def answers(port: int, timeout: float = 1.0) -> bool:
    """Whether the Lab on 127.0.0.1:port answers its health check (no proxy is consulted)."""

    connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        connection.request("GET", HEALTH_PATH)
        response = connection.getresponse()
        return response.status == 200 and response.read(16).strip() == b"ok"
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _reap(pid: int) -> bool:
    """Collect pid if it is an exited child of this process; True when it was."""

    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return done == pid


def process_start(pid: int) -> str | None:
    """The start time `ps` reports for a running process, or None when it is gone."""

    if _reap(pid):
        return None
    result = subprocess.run(
        ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": SYSTEM_PATH, "LC_ALL": "C"},
    )
    return result.stdout.strip() or None


_STARTED: list[subprocess.Popen] = []


def spawn(command: list[str], env: dict[str, str], cwd: Path, log: Path) -> int:
    """Start command in its own session (process group) with output appended to log."""

    fd = os.open(log, os.O_WRONLY | os.O_APPEND | PRIVATE_FLAGS, 0o600)
    with os.fdopen(fd, "ab") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _STARTED.append(process)  # keeps the handle, so no "still running" warning at exit
    return process.pid


# macOS answers EPERM, not ESRCH, for a process group whose members have exited but are
# not yet collected (zombies), so EPERM counts as "still exiting".
def signal_group(group: int, number: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, number)


def group_alive(group: int) -> bool:
    _reap(group)
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exited, not yet collected
    return True


def open_url(url: str) -> None:
    """Show url in the default browser (macOS `open`)."""

    subprocess.run(["/usr/bin/open", url], check=True, env={"PATH": SYSTEM_PATH})


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class System:
    """The operating-system seams; tests replace them with fakes."""

    is_free: Callable[[int], bool] = is_free
    answers: Callable[[int], bool] = answers
    process_start: Callable[[int], str | None] = process_start
    spawn: Callable[[list[str], dict[str, str], Path, Path], int] = spawn
    signal_group: Callable[[int, int], None] = signal_group
    group_alive: Callable[[int], bool] = group_alive
    open_url: Callable[[str], None] = open_url
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = _now


def lab_url(port: int) -> str:
    return f"http://{HOST}:{port}/"


def server_environment(source: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The server's whole environment: the allowlisted variables and fixed settings."""

    env = {name: source[name] for name in PASSED_VARIABLES if source.get(name)}
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "en_US.UTF-8")
    env["PATH"] = SYSTEM_PATH
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # no __pycache__ inside the checkout
    return env


def choose_port(free: Callable[[int], bool] = is_free) -> int:
    for port in PORTS:
        if free(port):
            return port
    raise LabError(
        f"No free port between {PORTS.start} and {PORTS.stop - 1} on this Mac. "
        "Quit another local server and open Portfolio Lab again."
    )


def private_folder(places: Places) -> None:
    """Create the private folders with owner-only permissions, refusing a planted one."""

    for folder in (places.private.parent, places.private):
        folder.mkdir(mode=0o700, exist_ok=True)
    info = places.private.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise LabError(f"{places.private} is not a folder of yours; move it away and try again.")
    if stat.S_IMODE(info.st_mode) != 0o700:
        places.private.chmod(0o700)


def note(places: Places, system: System, message: str) -> None:
    """Append one line to the launcher's own log."""

    stamp = system.now().isoformat(timespec="seconds")
    fd = os.open(places.launcher_log, os.O_WRONLY | os.O_APPEND | PRIVATE_FLAGS, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as log:
        log.write(f"{stamp} {message}\n")


@contextlib.contextmanager
def locked(places: Places) -> Iterator[None]:
    """One launcher action at a time (an exclusive lock on a file in the private folder)."""

    private_folder(places)
    fd = os.open(places.lock, os.O_RDWR | PRIVATE_FLAGS, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_state(path: Path) -> dict | None:
    """The recorded server, or None when there is none or the record is unusable."""

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    pid, port, started = state.get("pid"), state.get("port"), state.get("started")
    if not (type(pid) is int and 1 < pid < 2**31 and type(port) is int and 0 < port < 65536):
        return None  # type(...) is int also refuses true and false
    if not isinstance(started, str) or not started:
        return None
    return state


def write_state(path: Path, state: dict) -> None:
    """Replace the record atomically through a new owner-only temporary file."""

    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(temporary, path)


def same_checkout(state: dict, places: Places) -> bool:
    """Whether the record is this checkout's (a build with --checkout shares the folder)."""

    return state.get("checkout") == str(places.checkout)


def still_ours(state: dict, system: System) -> bool:
    """Whether the recorded process is still the one this launcher started."""

    return system.process_start(state["pid"]) == state["started"]


def wait_until_answering(state: dict, system: System, timeout: float) -> bool:
    """Poll the health check until it answers (True), the process ends or time runs out."""

    deadline = system.clock() + timeout
    while True:
        # Ours first: a server that failed to bind must not pass for one that answers.
        if not still_ours(state, system):
            return False
        if system.answers(state["port"]):
            return True
        if system.clock() >= deadline:
            return False
        system.sleep(POLL)


def stop_process(state: dict, system: System) -> None:
    """SIGTERM the recorded process group, then SIGKILL whatever is left after the timeout."""

    group = state["pid"]  # a session leader: its process group id is its pid
    system.signal_group(group, signal.SIGTERM)
    deadline = system.clock() + STOP_TIMEOUT
    while system.group_alive(group):
        if system.clock() >= deadline:
            system.signal_group(group, signal.SIGKILL)
            return
        system.sleep(POLL)


def check_checkout(places: Places) -> None:
    if not places.python.is_file():
        raise LabError("The project's Python environment (.venv) is missing.")
    if not places.lab_script.is_file():
        raise LabError("The Lab's start script (scripts/run_portfolio_lab.py) is missing.")


def rotate(log: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        if log.stat().st_size > LOG_LIMIT:
            os.replace(log, log.with_name(log.name + ".1"))


def running_url(places: Places, system: System) -> str | None:
    """The address of the server this app started, when it is still running."""

    state = read_state(places.state)
    if state is None:
        return None
    if not still_ours(state, system):
        places.state.unlink(missing_ok=True)
        note(places, system, f"forgot pid {state['pid']}: no longer running")
        return None
    if not same_checkout(state, places):
        raise LabError(
            "A Portfolio Lab app built for another project folder is running. "
            "Quit it, then open this one again."
        )
    if not wait_until_answering(state, system, REUSE_TIMEOUT):
        raise LabError(
            "The Lab is running but stopped answering. Portfolio Lab will stop it now; "
            "open Portfolio Lab again to restart it."
        )
    return lab_url(state["port"])


def start(places: Places, system: System) -> str:
    port = choose_port(system.is_free)
    rotate(places.server_log)
    command = [str(places.python), str(places.lab_script), "--port", str(port)]
    pid = system.spawn(command, server_environment(), places.checkout, places.server_log)
    state = {
        "pid": pid,
        "port": port,
        "started": system.process_start(pid) or "",
        "checkout": str(places.checkout),
        "opened_at": system.now().isoformat(timespec="seconds"),
    }
    if not state["started"]:
        raise LabError(f"The Lab stopped as soon as it started. {SEE_LOG}")
    write_state(places.state, state)
    note(places, system, f"started pid {pid} on port {port}")
    if wait_until_answering(state, system, START_TIMEOUT):
        return lab_url(port)
    places.state.unlink(missing_ok=True)
    if not still_ours(state, system):
        raise LabError(f"The Lab stopped while starting. {SEE_LOG}")
    stop_process(state, system)
    raise LabError(
        f"The Lab did not answer within {START_TIMEOUT:.0f} seconds, so it was stopped. {SEE_LOG}"
    )


def open_lab(places: Places, system: System) -> str:
    """Show the Lab, starting its server only when this app's server is not running."""

    check_checkout(places)
    with locked(places):
        url = running_url(places, system)
        if url is None:
            url = start(places, system)
        else:
            note(places, system, f"already running at {url}")
        system.open_url(url)
    return url


def stop_lab(places: Places, system: System) -> bool:
    """Stop the server this app started; False when none was running."""

    with locked(places):
        state = read_state(places.state)
        if state is None:
            return False
        if not still_ours(state, system):
            places.state.unlink(missing_ok=True)
            return False
        if not same_checkout(state, places):
            return False  # a running Lab of another build is left alone
        stop_process(state, system)
        note(places, system, f"stopped pid {state['pid']}")
        places.state.unlink(missing_ok=True)
        return True


def status(places: Places, system: System) -> str:
    state = read_state(places.state)
    if state is None or not still_ours(state, system):
        return "The Lab is not running."
    return f"The Lab is running at {lab_url(state['port'])} (since {state.get('opened_at')})."


def main(
    argv: list[str] | None = None, places: Places | None = None, system: System | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("action", choices=("open", "stop", "status"))
    args = parser.parse_args(argv)
    places = places or Places(CHECKOUT, PRIVATE_DIR)
    system = system or System()
    try:
        if args.action == "open":
            print(open_lab(places, system))
        elif args.action == "stop":
            print("Stopped the Lab." if stop_lab(places, system) else "The Lab was not running.")
        else:
            print(status(places, system))
    except (LabError, OSError, subprocess.SubprocessError) as error:
        message = (
            str(error)
            if isinstance(error, LabError)
            else (f"The Lab launcher could not finish: {error}. Choose Show Logs to see its log.")
        )
        with contextlib.suppress(OSError):  # the dialog matters more than the log line
            private_folder(places)
            note(places, system, f"{args.action} failed: {message}")
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
