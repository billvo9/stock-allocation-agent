"""
DNS-rebinding protection for the local Streamlit servers.

Every launcher binds 127.0.0.1, but binding alone does not stop a web page
whose host name an attacker rebinds to 127.0.0.1: the browser then sends that
name as the Host header, and Streamlit treats a WebSocket whose Origin host
equals its Host as same-origin, so the page could read what the app renders.
Streamlit refuses such a WebSocket only when `server.allowedHosts` lists the
names the server is reached by, so each launcher passes 127.0.0.1 and
localhost, and the repository's .streamlit/config.toml does the same for
launches from the repository root.

The end-to-end tests start each launcher's real server on a free loopback
port (nothing leaves the machine) and open the app's WebSocket the way a
browser would.
"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import itertools
import os
import socket
import subprocess
import time
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHERS = ("run_portfolio_lab", "run_dashboard", "run_model_health")
LOCAL_HOSTS = ["127.0.0.1", "localhost"]
STREAM = "/_stcore/stream"  # the app's WebSocket (Streamlit 1.64)


def _launcher(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"stock_agent_{name}_hosts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flag_values(command: list[str], flag: str) -> list[str]:
    return [value for part, value in itertools.pairwise(command) if part == flag]


@pytest.mark.parametrize("name", LAUNCHERS)
def test_every_launcher_accepts_only_the_local_host_names(name):
    command = _launcher(name).command(8502)
    assert _flag_values(command, "--server.allowedHosts") == LOCAL_HOSTS
    assert _flag_values(command, "--server.address") == ["127.0.0.1"]


def test_the_repository_config_accepts_only_the_local_host_names():
    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["allowedHosts"] == LOCAL_HOSTS
    assert config["server"]["address"] == "127.0.0.1"


# --- end to end: the real server ------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(port: int, path: str, headers: dict[str, str]) -> int:
    """The status of one request to the local server, with these exact headers."""

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        # A Host header given here replaces the one http.client would add.
        connection.request("GET", path, headers=headers)
        return connection.getresponse().status
    finally:
        connection.close()


def _open_stream(port: int, host: str, origin: str) -> int:
    """The status of a browser's WebSocket handshake: 101 accepted, 403 refused."""

    return _get(
        port,
        STREAM,
        {
            "Host": f"{host}:{port}",
            "Origin": origin,
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode("ascii"),
            "Sec-WebSocket-Version": "13",
        },
    )


@pytest.fixture(scope="module", params=LAUNCHERS)
def server_port(request):
    """A free port with the launcher's own Streamlit command serving on it (one per launcher)."""

    module = _launcher(request.param)
    port = _free_port()
    paths = [str(REPO / "src"), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    server = subprocess.Popen(
        [*module.command(port), "--server.fileWatcherType", "none"],
        cwd=REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert server.poll() is None, server.communicate()[0].decode(errors="replace")
            assert time.monotonic() < deadline, "the server did not start within 60 s"
            try:
                if _get(port, "/_stcore/health", {"Host": f"127.0.0.1:{port}"}) == 200:
                    break
            except OSError:
                pass  # not listening yet
            time.sleep(0.1)
        yield port
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def test_the_app_loads_on_the_local_host_names(server_port):
    for host in LOCAL_HOSTS:
        address = f"{host}:{server_port}"
        assert _get(server_port, "/", {"Host": address}) == 200, host
        assert _open_stream(server_port, host, f"http://{address}") == 101, host


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example",  # a rebound name: Origin and Host agree, as in a real attack
        "localhost.attacker.example",  # a local name as a prefix is not a match
        "127.0.0.1.attacker.example",
    ],
)
def test_a_page_rebound_to_this_machine_cannot_open_the_app(server_port, host):
    assert _open_stream(server_port, host, f"http://{host}:{server_port}") == 403


def test_a_foreign_page_cannot_open_the_app_by_its_local_name(server_port):
    # Cross-site WebSocket hijacking: the right Host, another site's Origin.
    assert _open_stream(server_port, "127.0.0.1", "http://attacker.example") == 403
