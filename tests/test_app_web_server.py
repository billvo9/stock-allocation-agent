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


def test_a_target_naming_another_host_is_not_served():
    assert app().handle("GET", "//evil.example/api/home", "127.0.0.1:8510").status == 404


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
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "secret" not in err  # the type only, for the log


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


def test_two_host_headers_get_nothing(running):
    connection = http.client.HTTPConnection("127.0.0.1", running, timeout=10)
    try:
        connection.putrequest("GET", "/api/home", skip_host=True)
        connection.putheader("Host", f"127.0.0.1:{running}")
        connection.putheader("Host", "evil.example")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 421 and response.read() == b""
    finally:
        connection.close()


def test_no_cookie_and_no_cross_origin_permission_is_ever_sent(running):
    for path in ("/", "/api/home", "/static/app.js", "/nowhere"):
        _, headers, _ = fetch(running, "GET", path, f"127.0.0.1:{running}")
        assert "Set-Cookie" not in headers
        assert not [name for name in headers if name.lower().startswith("access-control-")]


# --- one stock's screen, by its ticker ---


LONGEST = "A" * 20  # the longest name the route accepts


def stocks(calls, known=("NVDA", "DEMO_CALM", "BRK-B", LONGEST)):
    def find(name):
        calls.append(name)
        return {"ticker": name} if name in known else None

    return server.LabApp(port=PORT, api={}, static={}, items={"/api/stock/": find})


@pytest.mark.parametrize("name", ["NVDA", "DEMO_CALM", "BRK-B", LONGEST])
def test_a_stock_on_the_list_is_served_by_its_ticker(name):
    calls = []
    response = stocks(calls).handle("GET", f"/api/stock/{name}?x=1", "127.0.0.1:8510")
    assert response.status == 200 and json.loads(response.body) == {"ticker": name}
    assert response.content_type == "application/json; charset=utf-8" and calls == [name]


def test_a_ticker_the_view_does_not_know_is_not_found():
    calls = []
    response = stocks(calls).handle("GET", "/api/stock/AMD", "127.0.0.1:8510")
    assert (response.status, response.body, calls) == (404, b"Not found", ["AMD"])


@pytest.mark.parametrize(
    "target",
    [
        "/api/stock/",
        "/api/stock/nvda",
        "/api/stock/NVDA/",
        "/api/stock/NVDA/x",
        "/api/stock/../home",
        "/api/stock/%2E%2E",
        "/api/stock/NV%44A",
        "/api/stock/-NVDA",
        "/api/stock/_NVDA",
        "/api/stock/NVDA.",
        "/api/stock/NV DA",
        "/api/stock/NVDA%0A",
        "/api/stock/" + "A" * 21,
        "/api/stockNVDA",
        "/api/stocks/NVDA",
    ],
)
def test_a_name_that_is_not_a_ticker_never_reaches_the_view(target):
    calls = []
    assert stocks(calls).handle("GET", target, "127.0.0.1:8510").status == 404
    assert calls == []


def test_a_stock_request_another_website_starts_is_refused_before_any_work():
    calls = []
    headers = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "cors"}
    response = stocks(calls).handle("GET", "/api/stock/NVDA", "127.0.0.1:8510", headers)
    assert response.status == 403 and calls == []
    for other in (
        {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate"},  # only "/" may be visited
        {"origin": "http://evil.example"},  # no Fetch metadata (an older browser)
    ):
        assert stocks(calls).handle("GET", "/api/stock/NVDA", "127.0.0.1:8510", other).status == 403
    assert stocks(calls).handle("GET", "/api/stock/NVDA", "evil.example:8510").status == 421
    assert stocks(calls).handle("POST", "/api/stock/NVDA", "127.0.0.1:8510").status == 405
    assert calls == []


def test_a_failing_stock_screen_says_so_plainly(capsys):
    def broken(name):
        raise RuntimeError(f"/Users/someone/secret/{name} exploded")

    lab = server.LabApp(port=PORT, api={}, static={}, items={"/api/stock/": broken})
    response = lab.handle("GET", "/api/stock/NVDA", "127.0.0.1:8510")
    assert response.status == 500
    assert json.loads(response.body) == {"error": "The app couldn't prepare this screen."}
    err = capsys.readouterr().err
    assert "/api/stock/ failed (RuntimeError)" in err and "secret" not in err and "NVDA" not in err


def test_the_real_server_serves_one_stock():
    calls = []
    lab = stocks(calls)
    lab.port = 0
    httpd = server.make_server(lab)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        status, headers, body = fetch(port, "GET", "/api/stock/NVDA", f"127.0.0.1:{port}")
        assert status == 200 and json.loads(body) == {"ticker": "NVDA"}
        for name, value in server.SECURITY_HEADERS:
            assert headers.get(name) == value, name
        assert fetch(port, "GET", "/api/stock/AMD", f"127.0.0.1:{port}")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_head_on_a_stock_answers_like_get_without_a_body_at_the_socket():
    calls = []
    response = stocks(calls).handle("HEAD", "/api/stock/NVDA", "127.0.0.1:8510")
    assert response.status == 200 and calls == ["NVDA"]  # the handler drops the body for HEAD


# --- the per-launch key (owner decision of 2026-10-23) ---

KEY = "k" * 43


def keyed():
    return server.LabApp(
        port=PORT,
        api={"/api/home": lambda: {"hello": "world"}},
        static={"/": PAGE},
        items={"/api/stock/": lambda name: {"ticker": name}},
        token=KEY,
    )


@pytest.mark.parametrize("given", [None, "", "k" * 42, "K" * 43, KEY + "x", "kéy" * 15])
@pytest.mark.parametrize("target", ["/api/home", "/api/stock/NVDA"])
def test_the_data_needs_this_launchs_key(given, target):
    headers = {} if given is None else {"x-lab-token": given}
    response = keyed().handle("GET", target, "127.0.0.1:8510", headers)
    assert response.status == 401 and json.loads(response.body) == server.LOCKED


def test_with_the_key_the_data_is_served():
    found = keyed().handle("GET", "/api/stock/NVDA", "127.0.0.1:8510", {"x-lab-token": KEY})
    assert found.status == 200 and json.loads(found.body) == {"ticker": "NVDA"}


def test_the_page_and_the_health_check_need_no_key_and_hold_no_data():
    assert keyed().handle("GET", "/", "127.0.0.1:8510").status == 200
    health = keyed().handle("GET", "/health", "127.0.0.1:8510")
    assert (health.status, health.body, health.content_type) == (200, b"ok", server.TEXT)


def test_a_short_or_unusual_key_is_refused_at_start():
    for bad in ("short", "x" * 31, "k" * 40 + "#ab", "k" * 40 + " ab"):
        with pytest.raises(ValueError, match="key"):
            server.LabApp(port=PORT, api={}, static={}, token=bad)


def test_the_key_is_still_checked_after_the_host_and_site_checks():
    other_site = {"x-lab-token": KEY, "sec-fetch-site": "cross-site"}
    assert keyed().handle("GET", "/api/home", "evil.example:8510", other_site).status == 421
    assert keyed().handle("GET", "/api/home", "127.0.0.1:8510", other_site).status == 403


def test_the_real_server_answers_its_health_check_and_keeps_the_data_locked():
    lab = keyed()
    lab.port = 0
    httpd = server.make_server(lab)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        assert fetch(port, "GET", "/health", f"127.0.0.1:{port}")[::2] == (200, b"ok")
        assert fetch(port, "GET", "/api/home", f"127.0.0.1:{port}")[0] == 401
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request(
                "GET", "/api/home", headers={"Host": f"127.0.0.1:{port}", "X-Lab-Token": KEY}
            )
            response = connection.getresponse()
            assert response.status == 200 and json.loads(response.read()) == {"hello": "world"}
        finally:
            connection.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_more_than_four_screens_at_once_get_a_plain_busy_answer():
    lab = keyed()
    held = [lab.working.acquire(blocking=False) for _ in range(server.AT_ONCE)]
    try:
        assert all(held)
        busy = lab.handle("GET", "/api/home", "127.0.0.1:8510", {"x-lab-token": KEY})
        assert busy.status == 503 and json.loads(busy.body) == server.BUSY
        assert lab.handle("GET", "/health", "127.0.0.1:8510").status == 200  # never queued
    finally:
        for _ in held:
            lab.working.release()
    after = lab.handle("GET", "/api/home", "127.0.0.1:8510", {"x-lab-token": KEY})
    assert after.status == 200
