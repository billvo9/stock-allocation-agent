"""
Open the Portfolio Lab like a Mac app: start its two local servers once, then show it.

"Portfolio Lab.app" (built by scripts/build_mac_app.py) runs this script from
the checkout with the checkout's own Python, in isolated mode:

    .venv/bin/python -I -B scripts/portfolio_lab_app.py open | stop | status

The two servers, both on 127.0.0.1:
- the Lab app (scripts/run_lab_app.py), the simple screens the app shows, on
  port 8510 when it is free or the next free one up to 8599;
- the research pages (scripts/run_portfolio_lab.py, the Streamlit Lab), which
  the Lab app's More screen links to, on 8502 or the next free one up to 8509.

- open: when the servers this app started are running, only show the Lab app
  again (a new browser tab); never start a second pair. Otherwise start both,
  wait until each health check answers (up to 60 seconds; usually a second or
  two), then open the Lab app in the default browser.
- stop: stop the servers this app started, and nothing else. Each recorded
  process must still be the same one (process id and start time) before its
  process group is sent SIGTERM; SIGKILL follows only after 10 seconds.
- status: say whether the Lab is running, and where.

This launch's keys (owner decision of 2026-10-23). Each start makes two new
random keys (`secrets.token_urlsafe`): one for the Lab app and one for the
research pages, so a program posing as either server never learns the
other's. Each server shows its pages only to a browser that has its key, so
another account or program on this Mac that can reach 127.0.0.1 sees no
numbers and can change nothing.
- The keys are kept in the owner-only record (server.json) and passed to the
  servers in their environments, never on a command line, which other accounts
  can list. The Lab app also gets the research pages' key, to build its link to
  them.
- The browser gets the Lab app's key in the address's fragment (`#key=...`),
  which browsers never send to a server. The address goes to the browser through
  osascript's standard input (`open location`, as Python's own webbrowser does
  on macOS), not as an argument.
- Before the browser opens, the Lab app must prove it holds the key: its health
  check answers a fresh nonce with HMAC-SHA256(key, nonce). A program that took
  the port first can't, so it is never shown as the Lab.
- No key is ever printed or logged; a failure is reported without the address.

What it never does:
- read, pass or ask for a vendor key or other credential. The servers inherit
  only a short allowlist of variables (`PASSED_VARIABLES`): no TIINGO_API_KEY
  and no other variable of the caller;
- make a network request beyond this Mac. The health checks connect to
  127.0.0.1 directly, so no proxy setting applies;
- start a fetch, a forecast or any scheduled job. The only processes it starts
  are the two servers. The unattended evening job (`local.stock-lab.daily`) is
  separate and untouched, even when the app runs from its runner checkout;
- write inside the checkout. The evening job refuses a checkout with untracked
  files, so the record, lock and logs live in
  ~/stock-agent-private/portfolio_lab_app/ (owner-only permissions), and the
  servers run with PYTHONDONTWRITEBYTECODE=1 so no bytecode lands there either.

A record left by the earlier, single-server app (the Streamlit Lab alone) is
retired on the next open or stop: that server is stopped when it is still
this checkout's.

A failure exits with status 1 and a short explanation on stderr, which the app
shows in a macOS dialog; it is also recorded in launcher.log. Standard library
only.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
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
START_TIMEOUT = 60.0  # seconds for a new server to answer (it usually takes a second or two)
REUSE_TIMEOUT = 10.0  # seconds for a recorded server that is still starting
STOP_TIMEOUT = 10.0  # seconds after SIGTERM before SIGKILL
POLL = 0.25
LOG_LIMIT = 5_000_000  # bytes; a larger server log is kept once as <name>.1
# The only variables the servers inherit from the caller. Everything else is dropped,
# so no key (TIINGO_API_KEY among them) can reach the Lab through this launcher.
PASSED_VARIABLES = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
KEY_VARIABLE = "LAB_APP_TOKEN"  # the Lab app's key, read by scripts/run_lab_app.py
RESEARCH_KEY_VARIABLE = "PORTFOLIO_LAB_RESEARCH_KEY"  # the research pages' (app/run.py)
LINK_KEY_VARIABLE = "LAB_APP_RESEARCH_KEY"  # the same, for the Lab app's link to them
KEY = re.compile(r"[A-Za-z0-9_-]{32,128}")
KIND = "lab-app"  # the record's format: the Lab app, the research pages and their keys
# Files in the private folder are opened without following a symlink planted in their place.
PRIVATE_FLAGS = os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True)
class Server:
    """One of the two local servers the app starts."""

    name: str
    script: str  # relative to the checkout
    ports: range  # the first free one is used
    health: str  # a path that answers "ok" once the server accepts browsers
    log: str  # its output, in the private folder
    title: str  # in messages


LAB = Server("lab", "scripts/run_lab_app.py", range(8510, 8600), "/health", "lab.log", "The Lab")
RESEARCH = Server(
    "research",
    "scripts/run_portfolio_lab.py",
    range(8502, 8510),
    "/_stcore/health",  # Streamlit answers 200 "ok" once it accepts browsers
    "research.log",
    "The research pages",
)
SERVERS = (RESEARCH, LAB)  # start order: the Lab app links to the research pages

SEE_LOG = "Choose Show Logs and open {log} to see why."


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

    def script(self, server: Server) -> Path:
        return self.checkout / server.script

    def log(self, server: Server) -> Path:
        return self.private / server.log

    @property
    def state(self) -> Path:
        return self.private / "server.json"

    @property
    def lock(self) -> Path:
        return self.private / "launcher.lock"

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


def answers(port: int, path: str, expect: bytes = b"ok", timeout: float = 1.0) -> bool:
    """
    Whether the server on 127.0.0.1:port answers its health check with `expect` (no
    proxy is consulted).
    """

    connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status == 200 and response.read(128).strip() == expect
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


LOCAL_ADDRESS = re.compile(r"http://127\.0\.0\.1:\d{1,5}/(#key=[A-Za-z0-9_-]{32,128})?")
NO_BROWSER = (
    "Portfolio Lab couldn't open your browser. The Lab is running: "
    "open Portfolio Lab again to show it."
)


def open_url(url: str) -> None:
    """
    Show url in the default browser. The address goes to osascript on its standard
    input (`open location`, as Python's webbrowser does on macOS), never as an argument
    that another account could list, and a failure is reported without it: it holds
    this launch's key.
    """

    if not LOCAL_ADDRESS.fullmatch(url):  # nothing else is ever quoted into the script
        raise LabError(NO_BROWSER)
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-"],
            input=f'open location "{url}"',
            text=True,
            capture_output=True,  # osascript may echo its script on an error
            check=False,
            timeout=30,
            env={"PATH": SYSTEM_PATH},
        )
    except (OSError, subprocess.SubprocessError):
        raise LabError(NO_BROWSER) from None
    if result.returncode != 0:
        raise LabError(NO_BROWSER)


def new_key() -> str:
    return secrets.token_urlsafe(32)


def _now() -> datetime:
    return datetime.now(UTC)


def new_nonce() -> str:
    return secrets.token_hex(16)


@dataclass
class System:
    """The operating-system seams; tests replace them with fakes."""

    is_free: Callable[[int], bool] = is_free
    answers: Callable[[int, str, bytes], bool] = answers
    new_nonce: Callable[[], str] = new_nonce
    process_start: Callable[[int], str | None] = process_start
    spawn: Callable[[list[str], dict[str, str], Path, Path], int] = spawn
    signal_group: Callable[[int, int], None] = signal_group
    group_alive: Callable[[int], bool] = group_alive
    open_url: Callable[[str], None] = open_url
    new_key: Callable[[], str] = new_key
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = _now


def lab_url(port: int) -> str:
    return f"http://{HOST}:{port}/"


def keyed_url(port: int, key: str) -> str:
    """The Lab app's address with this launch's key in its fragment (never sent to a server)."""

    return f"{lab_url(port)}#key={key}"


def server_environment(source: Mapping[str, str] = os.environ) -> dict[str, str]:
    """A server's whole environment: the allowlisted variables and fixed settings."""

    env = {name: source[name] for name in PASSED_VARIABLES if source.get(name)}
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("LANG", "en_US.UTF-8")
    env["PATH"] = SYSTEM_PATH
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # no __pycache__ inside the checkout
    return env


def lab_environment(
    places: Places, key: str, research_key: str, source: Mapping[str, str] = os.environ
) -> dict:
    """
    The Lab app's environment: the same allowlist, this checkout's code, its key and the
    research pages' key (for its link to them).
    """

    env = server_environment(source)
    env["PYTHONPATH"] = str(places.checkout / "src")  # this checkout's code, never another's
    env[KEY_VARIABLE] = key
    env[LINK_KEY_VARIABLE] = research_key
    return env


def research_environment(research_key: str, source: Mapping[str, str] = os.environ) -> dict:
    """The research pages' environment: the same allowlist and their own key only."""

    return {**server_environment(source), RESEARCH_KEY_VARIABLE: research_key}


def proof(key: str, nonce: str) -> bytes:
    """What the Lab app's health check answers for `nonce`: HMAC-SHA256(key, nonce), hex."""

    return hmac.new(key.encode("ascii"), nonce.encode("ascii"), hashlib.sha256).hexdigest().encode()


def health_check(server: Server, state: dict, system: System) -> tuple[str, bytes]:
    """The path to ask and the answer that counts: the Lab app must prove it holds its key."""

    if server.name != LAB.name:
        return server.health, b"ok"
    nonce = system.new_nonce()
    return f"{server.health}?nonce={nonce}", proof(state["key"], nonce)


def choose_port(server: Server, free: Callable[[int], bool] = is_free) -> int:
    for port in server.ports:
        if free(port):
            return port
    raise LabError(
        f"No free port between {server.ports.start} and {server.ports.stop - 1} on this Mac "
        f"for {server.title.lower()}. Quit another local server and open Portfolio Lab again."
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
    """Append one line to the launcher's own log (never a key)."""

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


def _process(record) -> bool:
    """Whether a recorded process looks like one this launcher wrote down."""

    if not isinstance(record, dict):
        return False
    pid, port, started = record.get("pid"), record.get("port"), record.get("started")
    return (
        type(pid) is int  # type(...) is int also refuses true and false
        and 1 < pid < 2**31
        and type(port) is int
        and 0 < port < 65536
        and isinstance(started, str)
        and bool(started)
    )


def _load(path: Path) -> dict | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def read_state(path: Path) -> dict | None:
    """The recorded pair of servers and their keys, or None when there is none or it is unusable."""

    state = _load(path)
    if state is None or state.get("kind") != KIND:
        return None
    servers = state.get("servers")
    keys = (state.get("key"), state.get("research_key"))
    if not all(isinstance(key, str) and KEY.fullmatch(key) for key in keys):
        return None
    if keys[0] == keys[1] or not isinstance(servers, dict):
        return None
    if set(servers) != {server.name for server in SERVERS}:
        return None
    if not all(_process(record) for record in servers.values()):
        return None
    return state


def read_legacy(path: Path) -> dict | None:
    """A record of the earlier, single-server app (the Streamlit Lab alone), if usable."""

    state = _load(path)
    if state is None or "kind" in state or not _process(state):
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


def still_ours(record: dict, system: System) -> bool:
    """Whether the recorded process is still the one this launcher started."""

    return system.process_start(record["pid"]) == record["started"]


def wait_until_answering(server: Server, state: dict, system: System, timeout: float) -> bool:
    """Poll the server's health check until it answers (True), its process ends or time runs out."""

    record = state["servers"][server.name]
    deadline = system.clock() + timeout
    while True:
        # Ours first: a server that failed to bind must not pass for one that answers.
        if not still_ours(record, system):
            return False
        path, expect = health_check(server, state, system)
        if system.answers(record["port"], path, expect):
            return True
        if system.clock() >= deadline:
            return False
        system.sleep(POLL)


def stop_process(record: dict, system: System) -> None:
    """SIGTERM the recorded process group, then SIGKILL whatever is left after the timeout."""

    group = record["pid"]  # a session leader: its process group id is its pid
    system.signal_group(group, signal.SIGTERM)
    deadline = system.clock() + STOP_TIMEOUT
    while system.group_alive(group):
        if system.clock() >= deadline:
            system.signal_group(group, signal.SIGKILL)
            return
        system.sleep(POLL)


def stop_ours(state: dict, system: System) -> list[str]:
    """Stop every recorded server that is still ours; the names of those stopped."""

    stopped = []
    for server in SERVERS:
        record = state["servers"][server.name]
        if still_ours(record, system):
            stop_process(record, system)
            stopped.append(server.name)
    return stopped


def check_checkout(places: Places) -> None:
    if not places.python.is_file():
        raise LabError("The project's Python environment (.venv) is missing.")
    for server in SERVERS:
        if not places.script(server).is_file():
            raise LabError(
                f"A start script ({server.script}) is missing. Update the project folder to "
                "the latest main (git pull), then open Portfolio Lab again."
            )


def rotate(log: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        if log.stat().st_size > LOG_LIMIT:
            os.replace(log, log.with_name(log.name + ".1"))


OTHER_BUILD = (
    "A Portfolio Lab app built for another project folder is running. "
    "Quit it, then open this one again."
)


def retire_legacy(places: Places, system: System, *, refuse_other: bool) -> bool:
    """
    Stop and forget the earlier app's single server, when it is still this checkout's;
    True when one was stopped. Another build's running server is left alone (and, with
    `refuse_other`, reported, so its record is never overwritten).
    """

    legacy = read_legacy(places.state)
    if legacy is None:
        return False
    if not still_ours(legacy, system):
        places.state.unlink(missing_ok=True)
        note(places, system, f"forgot the earlier app's pid {legacy['pid']}: no longer running")
        return False
    if not same_checkout(legacy, places):
        if refuse_other:
            raise LabError(OTHER_BUILD)
        return False
    stop_process(legacy, system)
    places.state.unlink(missing_ok=True)
    note(places, system, f"retired the earlier app's server (pid {legacy['pid']})")
    return True


def running(places: Places, system: System) -> dict | None:
    """The record of the servers this app started, when both still run and answer."""

    state = read_state(places.state)
    if state is None:
        retire_legacy(places, system, refuse_other=True)
        return None
    alive = {s.name: still_ours(state["servers"][s.name], system) for s in SERVERS}
    if not same_checkout(state, places):
        if any(alive.values()):
            raise LabError(OTHER_BUILD)
        places.state.unlink(missing_ok=True)  # another build's pair is gone: nothing to keep
        note(places, system, "forgot another build's servers: no longer running")
        return None
    if not all(alive.values()):
        stopped = stop_ours(state, system)  # half a pair is restarted as a whole
        places.state.unlink(missing_ok=True)
        note(places, system, f"forgot the servers: not both running (stopped {stopped or 'none'})")
        return None
    for server in SERVERS:
        if not wait_until_answering(server, state, system, REUSE_TIMEOUT):
            raise LabError(
                f"{server.title} stopped answering, so Portfolio Lab will quit now. "
                "Open Portfolio Lab again to restart it."
            )
    return state


def start(places: Places, system: System) -> dict:
    """Start the research pages and the Lab app with a new key; their record."""

    research_port = choose_port(RESEARCH, system.is_free)
    lab_port = choose_port(LAB, system.is_free)
    key, research_key = system.new_key(), system.new_key()
    if not (KEY.fullmatch(key) and KEY.fullmatch(research_key)) or key == research_key:
        raise LabError("Portfolio Lab could not make this launch's keys.")
    commands = {
        RESEARCH.name: (
            [str(places.python), str(places.script(RESEARCH)), "--port", str(research_port)],
            research_environment(research_key),
        ),
        LAB.name: (
            [
                str(places.python),
                str(places.script(LAB)),
                "--port",
                str(lab_port),
                "--research-url",
                lab_url(research_port),
            ],
            lab_environment(places, key, research_key),
        ),
    }
    state = {
        "kind": KIND,
        "checkout": str(places.checkout),
        "opened_at": system.now().isoformat(timespec="seconds"),
        "key": key,
        "research_key": research_key,
        "servers": {},
    }
    ports = {RESEARCH.name: research_port, LAB.name: lab_port}
    for server in SERVERS:
        rotate(places.log(server))
        command, env = commands[server.name]
        pid = system.spawn(command, env, places.checkout, places.log(server))
        record = {
            "pid": pid,
            "port": ports[server.name],
            "started": system.process_start(pid) or "",
        }
        if not record["started"]:
            _abandon(state, places, system)
            raise LabError(
                f"{server.title} stopped right after starting. {SEE_LOG.format(log=server.log)}"
            )
        state["servers"][server.name] = record
        note(places, system, f"started {server.name} pid {pid} on port {ports[server.name]}")
    write_state(places.state, state)  # once both run: a record always names a whole pair
    for server in SERVERS:
        record = state["servers"][server.name]
        if wait_until_answering(server, state, system, START_TIMEOUT):
            continue
        gone = not still_ours(record, system)
        _abandon(state, places, system)
        if gone:
            raise LabError(
                f"{server.title} stopped while starting. {SEE_LOG.format(log=server.log)}"
            )
        raise LabError(
            f"{server.title} did not answer within {START_TIMEOUT:.0f} seconds, so Portfolio Lab "
            f"was stopped. {SEE_LOG.format(log=server.log)}"
        )
    return state


def _abandon(state: dict, places: Places, system: System) -> None:
    """Stop whatever of a failed start is still ours, and forget it."""

    for record in state["servers"].values():
        if still_ours(record, system):
            stop_process(record, system)
    places.state.unlink(missing_ok=True)


def open_lab(places: Places, system: System) -> str:
    """Show the Lab, starting its servers only when this app's are not running."""

    check_checkout(places)
    with locked(places):
        state = running(places, system)
        if state is None:
            state = start(places, system)
        else:
            note(places, system, "already running")
        port = state["servers"][LAB.name]["port"]
        system.open_url(keyed_url(port, state["key"]))
    return lab_url(port)  # the address without the key: it is printed


def stop_lab(places: Places, system: System) -> bool:
    """Stop the servers this app started; False when none was running."""

    with locked(places):
        state = read_state(places.state)
        if state is None:
            return retire_legacy(places, system, refuse_other=False)
        if not same_checkout(state, places):
            if not any(still_ours(state["servers"][s.name], system) for s in SERVERS):
                places.state.unlink(missing_ok=True)  # gone: nothing of it to keep
            return False  # a running Lab of another build is left alone
        stopped = stop_ours(state, system)
        note(places, system, f"stopped {', '.join(stopped) or 'nothing (already gone)'}")
        places.state.unlink(missing_ok=True)
        return bool(stopped)


def status(places: Places, system: System) -> str:
    state = read_state(places.state)
    if state is None or not all(still_ours(state["servers"][s.name], system) for s in SERVERS):
        return "The Lab is not running."
    lab, research = (state["servers"][s.name]["port"] for s in (LAB, RESEARCH))
    return (
        f"The Lab is running at {lab_url(lab)}, with the research pages at {lab_url(research)} "
        f"(since {state.get('opened_at')})."
    )


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
