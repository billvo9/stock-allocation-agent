"""
The Lab app's local server: Python's standard-library HTTP server, on this Mac
only, read-only.

- It binds 127.0.0.1 and refuses any other address.
- DNS-rebinding protection: a request whose Host header is not exactly
  `127.0.0.1:<port>` or `localhost:<port>` (one Host header, any letter case)
  gets 421 and no content, so a web page whose name an attacker rebinds to
  this machine can read nothing.
- Only GET and HEAD (405 otherwise), over HTTP/1.0 or 1.1, with a path
  target (400 otherwise); fixed routes; the static files are read once at start
  from a fixed list, so no part of a request reaches the file system.
- A request another website starts in the background (by its Fetch metadata or
  Origin) gets 403 before any work; only a top-level visit to the page may come
  from a link elsewhere.
- A connection that sends nothing is dropped after 15 seconds.
- Every response carries the same strict headers (`SECURITY_HEADERS`): the
  page may load only its own script, stylesheet, font and data, and may not
  be framed. No cookies, no CORS, nothing cached.
- A screen that fails answers 500 with one plain sentence; the error's type,
  never its message or a traceback, goes to the log.

`LabApp.handle` is the whole request logic, with no socket, so it is tested
directly; the handler only copies a request in and a response out.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
LOCAL_NAMES = ("127.0.0.1", "localhost")
SECURITY_HEADERS = (
    (
        "Content-Security-Policy",
        (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
            "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'"
        ),
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cache-Control", "no-store"),
)
TEXT = "text/plain; charset=utf-8"
JSON = "application/json; charset=utf-8"
FAILED = {"error": "The app couldn't prepare this screen."}

STATIC = Path(__file__).with_name("static")
FONT = Path(__file__).parents[1] / "static" / "fonts" / "SourceSans3VF-Upright.woff2"
MARK = Path(__file__).parents[4] / "assets" / "brand" / "portfolio-lab-mark.svg"  # the logo
# Route -> (file, content type): the only files the server can send.
FILES = {
    "/": (STATIC / "index.html", "text/html; charset=utf-8"),
    "/static/app.css": (STATIC / "app.css", "text/css; charset=utf-8"),
    "/static/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
    "/static/mark.svg": (MARK, "image/svg+xml"),
    "/static/source-sans-3.woff2": (FONT, "font/woff2"),
}


def static_files() -> dict[str, tuple[bytes, str]]:
    """The page and its assets, read once, by route."""

    return {route: (path.read_bytes(), kind) for route, (path, kind) in FILES.items()}


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


class LabApp:
    """The routes: `api` maps a path to a function returning JSON-ready data."""

    def __init__(
        self,
        *,
        port: int,
        api: Mapping[str, Callable[[], Mapping]],
        static: Mapping[str, tuple[bytes, str]],
    ) -> None:
        self.port = port
        self.api = dict(api)
        self.static = dict(static)

    def allowed(self, host: str | None) -> bool:
        if not host:
            return False
        return host.lower() in {f"{name}:{self.port}" for name in LOCAL_NAMES}

    def cross_site(self, path: str, headers: Mapping[str, str]) -> bool:
        """
        A request another website started in the background (an image, a script, a
        fetch), by its Fetch metadata or its Origin: refused before any work. Only a
        top-level visit to the page itself may come from elsewhere (a link).
        """

        if path == "/" and headers.get("sec-fetch-mode") == "navigate":
            return False
        site = headers.get("sec-fetch-site")
        if site is not None and site not in ("same-origin", "none"):
            return True
        origin = headers.get("origin")
        own = {f"http://{name}:{self.port}" for name in LOCAL_NAMES}
        return origin is not None and origin.lower() not in own

    def handle(
        self,
        method: str,
        target: str,
        host: str | None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """`headers`: the request's other headers, by lower-case name."""

        if not self.allowed(host):
            return Response(421, TEXT, b"")
        if method not in ("GET", "HEAD"):
            return Response(405, TEXT, b"", (("Allow", "GET, HEAD"),))
        if not target.startswith("/"):  # a full URL, "*" or anything else: not a path here
            return Response(400, TEXT, b"Bad request")
        path = target.split("?", 1)[0].split("#", 1)[0]
        if self.cross_site(path, headers or {}):
            return Response(403, TEXT, b"")
        if path in self.static:
            body, kind = self.static[path]
            return Response(200, kind, body)
        if path in self.api:
            return self._api(path)
        return Response(404, TEXT, b"Not found")

    def _api(self, path: str) -> Response:
        try:
            body = json.dumps(self.api[path](), allow_nan=False, ensure_ascii=False)
        except Exception as error:  # noqa: BLE001 - one failing screen never stops the app
            print(f"lab app: {path} failed ({type(error).__name__})", file=sys.stderr)
            return Response(500, JSON, json.dumps(FAILED).encode())
        return Response(200, JSON, body.encode())


class _Handler(BaseHTTPRequestHandler):
    server_version = "PortfolioLab"
    sys_version = ""
    timeout = 15  # seconds: a connection that sends nothing does not hold a thread forever

    def _serve(self) -> None:
        if self.request_version not in ("HTTP/1.0", "HTTP/1.1"):
            self.close_connection = True  # HTTP/0.9 has no headers to carry: nothing is sent
            return
        hosts = self.headers.get_all("Host") or []
        host = hosts[0] if len(hosts) == 1 else None
        others = {name.lower(): value for name, value in self.headers.items()}
        response = self.server.app.handle(self.command, self.path, host, others)
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        for name, value in response.headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response.body)

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = _serve
    do_OPTIONS = do_TRACE = do_CONNECT = _serve

    def end_headers(self) -> None:
        # Every response, the base class's own errors included (an unknown method,
        # a malformed request line), carries the same strict headers.
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        super().end_headers()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app: LabApp) -> None:
        self.app = app
        super().__init__(address, _Handler)

    def handle_error(self, request, client_address) -> None:
        # The error's type only: no traceback, so no path reaches the log.
        print(f"lab app: a request failed ({sys.exc_info()[0].__name__})", file=sys.stderr)


def make_server(app: LabApp, *, host: str = HOST) -> ThreadingHTTPServer:
    """The server for `app` on 127.0.0.1 (port 0 picks a free one and records it on `app`)."""

    if host != HOST:
        raise ValueError(f"the Lab app binds {HOST} only, not {host!r}")
    server = _Server((host, app.port), app)
    app.port = server.server_address[1]
    return server
