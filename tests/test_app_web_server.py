"""
The Lab app's local server (docs/product/lab_app_design.md, "Security"): this
Mac only, the DNS-rebinding check, read-only routes, strict headers and plain
errors. The request core is tested without sockets; one end-to-end test runs
the real server on a free loopback port (nothing leaves the machine).
"""

from __future__ import annotations

import http.client
import json
import socket
import threading

import pytest

from stock_agent.app.web import server

PORT = 8510
PAGE = (b"<!doctype html><title>Portfolio Lab</title>", "text/html; charset=utf-8")


def app(api=None, port=PORT):
    return server.LabApp(
        port=port,
        api=api if api is not None else {"/api/home": lambda: {"hello": "world"}},
        static={"/": PAGE, "/static/app.js": (b"// js", "text/javascript; charset=utf-8")},
    )


@pytest.mark.parametrize("host", ["127.0.0.1:8510", "localhost:8510", "LOCALHOST:8510"])
def test_the_local_names_are_served(host):
    response = app().handle("GET", "/", host)
    assert (response.status, response.body, response.content_type) == (200, *PAGE)


@pytest.mark.parametrize(
    "host", [None, "", "evil.example:8510", "127.0.0.1", "127.0.0.1:8511", "localhost:8510 x"]
)
def test_any_other_host_gets_nothing(host):
    for target in ("/", "/api/home", "/nowhere"):
        response = app().handle("GET", target, host)
        assert response.status == 421 and response.body == b""


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE"])
def test_only_reading_is_allowed(method):
    response = app().handle(method, "/api/home", "127.0.0.1:8510")
    assert response.status == 405 and ("Allow", "GET, HEAD") in response.headers


def test_unknown_paths_are_not_found_and_never_reach_the_file_system():
    for target in ("/nowhere", "/static/../server.py", "/api/home/", "/static/", "//"):
        assert app().handle("GET", target, "127.0.0.1:8510").status == 404


@pytest.mark.parametrize("target", ["http://evil.example/api/home", "http://[/", "x://[a/", "*"])
def test_a_target_that_is_not_a_path_is_refused_without_crashing(target):
    response = app().handle("GET", target, "127.0.0.1:8510")
    assert response.status == 400


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/api/home", {"sec-fetch-site": "cross-site"}, 403),
        ("/api/home", {"sec-fetch-site": "same-site"}, 403),
        ("/static/app.js", {"sec-fetch-site": "cross-site", "sec-fetch-mode": "no-cors"}, 403),
        ("/api/home", {"origin": "http://evil.example"}, 403),
        ("/api/home", {"sec-fetch-site": "same-origin"}, 200),
        ("/api/home", {"sec-fetch-site": "none"}, 200),
        ("/api/home", {"origin": "http://127.0.0.1:8510"}, 200),
        ("/", {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate"}, 200),
        ("/", {"sec-fetch-site": "cross-site", "sec-fetch-mode": "no-cors"}, 403),
    ],
)
def test_requests_other_websites_start_are_refused_before_any_work(path, headers, status):
    calls = []

    def home():
        calls.append(1)
        return {}

    response = app({"/api/home": home}).handle("GET", path, "127.0.0.1:8510", headers)
    assert response.status == status
    assert calls == ([1] if status == 200 and path == "/api/home" else [])


def test_the_query_string_does_not_change_the_route():
    response = app().handle("GET", "/api/home?x=1", "127.0.0.1:8510")
    assert response.status == 200 and json.loads(response.body) == {"hello": "world"}


def test_the_api_answers_strict_json():
    response = app().handle("GET", "/api/home", "localhost:8510")
    assert response.content_type == "application/json; charset=utf-8"
    assert json.loads(response.body) == {"hello": "world"}
    nan = app({"/api/home": lambda: {"x": float("nan")}}).handle(
        "GET", "/api/home", "localhost:8510"
    )
    assert nan.status == 500


def test_a_failing_screen_says_so_plainly(capsys):
    def broken():
        raise RuntimeError("/Users/someone/secret/path exploded")

    response = app({"/api/home": broken}).handle("GET", "/api/home", "127.0.0.1:8510")
    assert response.status == 500
    body = json.loads(response.body)
    assert body == {"error": "The app couldn't prepare this screen."}
    assert "secret" not in response.body.decode() and "Traceback" not in response.body.decode()
    assert "RuntimeError" in capsys.readouterr().err  # the type only, for the log


def test_the_server_binds_this_mac_only():
    with pytest.raises(ValueError, match="127.0.0.1"):
        server.make_server(app(), host="0.0.0.0")


# --- end to end: the real server on a free loopback port ---


@pytest.fixture
def running():
    lab = app(port=0)
    httpd = server.make_server(lab)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def fetch(port, method, path, host):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, headers={"Host": host})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_every_response_carries_the_security_headers(running):
    for method, path, host in (
        ("GET", "/", f"127.0.0.1:{running}"),
        ("HEAD", "/api/home", f"localhost:{running}"),
        ("GET", "/nowhere", f"127.0.0.1:{running}"),
        ("GET", "/", "evil.example"),
        ("POST", "/api/home", f"127.0.0.1:{running}"),
    ):
        _, headers, _ = fetch(running, method, path, host)
        for name, value in server.SECURITY_HEADERS:
            assert headers.get(name) == value, (method, path, name)
        assert "Python" not in headers.get("Server", "")


def test_the_real_server_serves_the_page_and_the_api(running):
    status, headers, body = fetch(running, "GET", "/", f"127.0.0.1:{running}")
    assert (status, body) == (200, PAGE[0]) and headers["Content-Type"] == PAGE[1]
    status, _, body = fetch(running, "GET", "/api/home", f"localhost:{running}")
    assert status == 200 and json.loads(body) == {"hello": "world"}
    status, headers, body = fetch(running, "HEAD", "/api/home", f"localhost:{running}")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0
    assert fetch(running, "GET", "/", "rebound.example")[0] == 421


def test_the_base_servers_own_errors_carry_the_security_headers(running):
    connection = http.client.HTTPConnection("127.0.0.1", running, timeout=10)
    try:
        connection.request("BREW", "/", headers={"Host": f"127.0.0.1:{running}"})
        response = connection.getresponse()
        response.read()
        assert response.status == 501
        for name, value in server.SECURITY_HEADERS:
            assert response.getheader(name) == value, name
    finally:
        connection.close()


def test_an_http_0_9_request_gets_nothing(running):
    with socket.create_connection(("127.0.0.1", running), timeout=10) as raw:
        raw.sendall(f"GET /api/home\r\nHost: 127.0.0.1:{running}\r\n\r\n".encode())
        assert raw.recv(4096) == b""


def test_a_silent_connection_is_dropped_after_a_while():
    httpd = server.make_server(app(port=0))
    try:
        assert httpd.RequestHandlerClass.timeout == 15
    finally:
        httpd.server_close()
