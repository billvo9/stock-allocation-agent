"""
Open the Lab app, the Portfolio Lab's simple screens, on this Mac only.

It serves the Lab app (src/stock_agent/app/web; docs/product/lab_app_design.md)
from Python's standard-library web server on 127.0.0.1, read-only: it opens
the Stock Lab's store read-only, reads each stock's recent observed prices
through the isolated descriptive reader (decision A6), and passes both, with
the NYSE rules calendar, to the app's Home and Stock detail views. It never
runs the evening job, writes nothing, sends nothing anywhere and reads no
credential.

Modes
    (default)   the real Stock Lab
    --demo [N]  a synthetic demo: replay N synthetic sessions (default 30) into
                a temporary store that is removed when the app stops; nothing
                real is read
Options
    --port N            the port (default 8510; the Streamlit Lab keeps 8502)
    --at TIME           show the app as of TIME (ISO 8601 with a time zone), for
                        a preview; only records that existed by then are read
    --research-url URL  where More > Research pages links, for example
                        http://127.0.0.1:8502/ when the Streamlit Lab runs
    --open              open the app in the default browser

Usage:
    .venv/bin/python scripts/run_lab_app.py --demo --open
    .venv/bin/python scripts/run_lab_app.py --open
Stop it with Control-C.
"""

from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import webbrowser
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from stock_agent.app.web.detail import build_detail
from stock_agent.app.web.home import build_home
from stock_agent.app.web.server import HOST, LabApp, make_server, static_files
from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.inference.recent import read_recent
from stock_agent.stock_lab.prices import session_returns
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, utc_now

DEFAULT_PORT = 8510
EXCLUDED = ("SKHY", "SNDK")  # the owner's choice: kept off the watchlist views
DEMO_SESSIONS = 30
DEMO_CALENDAR = "synthetic_weekdays"
STOCK_ROUTE = "/api/stock/"  # one stock's detail: /api/stock/NVDA


def observed(symbol: str, *, count: int, retrieved_by: pd.Timestamp):
    """A real stock's recent observed prices as stored by `retrieved_by` (decision A6)."""

    try:
        return read_recent(symbol, count=count, retrieved_by=retrieved_by)
    except Exception as error:  # noqa: BLE001 - one stock's prices never stop the app
        print(f"lab app: {symbol} prices failed ({type(error).__name__})", file=sys.stderr)
        return SimpleNamespace(state="unreadable", reason=type(error).__name__, sessions=())


def local_url(text: str) -> bool:
    """Whether `text` is a plain http address on this Mac: 127.0.0.1 or localhost, a port."""

    if any(c.isspace() or c == "\\" or not c.isprintable() for c in text):
        return False
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname in ("127.0.0.1", "localhost")
        and parts.username is None
        and parts.password is None
        and "@" not in parts.netloc
        and port is not None
        and 0 < port < 65536
    )


def _moment(text: str | None, clock: Callable[[], pd.Timestamp]) -> datetime:
    moment = clock() if text is None else pd.Timestamp(text)
    if moment.tzinfo is None:
        raise ValueError("--at needs a time zone, for example 2026-10-22T07:40-04:00")
    return moment.tz_convert("UTC").to_pydatetime()


def real_api(
    *,
    records_factory: Callable[[], ReadOnlyRecords],
    clock: Callable[[], datetime],
    recent: Callable,
    research_url: str | None,
) -> tuple[dict, dict]:
    """The routes (Home) and the named routes (one stock's detail, by its ticker)."""

    def home() -> dict:
        view = build_home(
            records_factory(),
            now=clock(),
            calendar=sessions,
            calendar_id=CALENDAR_ID,
            data_class=DataClass.REAL,
            recent=recent,
            local_zone=None,  # this Mac's own zone, at each moment
            excluded=EXCLUDED,
        )
        return {**view, "research_url": research_url}

    def stock(symbol: str) -> dict | None:
        return build_detail(
            records_factory(),
            symbol=symbol,
            now=clock(),
            calendar=sessions,
            calendar_id=CALENDAR_ID,
            data_class=DataClass.REAL,
            recent=recent,
            local_zone=None,
            excluded=EXCLUDED,
            research_url=research_url,
        )

    return {"/api/home": home}, {STOCK_ROUTE: stock}


# --- the synthetic demo ---


def weekdays(start, end) -> pd.DatetimeIndex:
    """The synthetic demo's calendar: every weekday (the replay's)."""

    return pd.bdate_range(start, end, tz="UTC")


def demo_prices(records: ReadOnlyRecords) -> Callable:
    """The synthetic stocks' prices in the recent-prices reader's shape (synthetic only)."""

    def recent(symbol: str, *, count: int, retrieved_by: pd.Timestamp):
        stored = records.bars(symbol, as_of=retrieved_by)
        if stored is None:
            return SimpleNamespace(state="no_data", reason=None, sessions=())
        frame = session_returns(stored.bars.drop(columns=["batch_id"])).tail(count)
        rows = tuple(
            SimpleNamespace(
                session=pd.Timestamp(item.session).strftime("%Y-%m-%d"),
                close=None if np.isnan(item.close) else float(item.close),
                level=None if np.isnan(item.level) else float(item.level),
                change=None if np.isnan(item.price_return) else float(item.price_return),
                notes=(),
            )
            for item in frame.itertuples()
        )
        return SimpleNamespace(state="observed", reason=None, sessions=rows)

    return recent


def demo_api(home_dir: Path, sessions_count: int, at: str | None, research_url: str | None):
    """Replay synthetic sessions into `home_dir`; the routes and the named routes."""

    from stock_agent.stock_lab.job import record_run
    from stock_agent.stock_lab.replay import run_replay  # research side: the demo only
    from stock_agent.stock_lab.synthetic import DEMO_IDS, DEMOS

    path = home_dir / DB_NAME

    def record_day(*, as_of, store, record, model):
        day = pd.Timestamp(as_of)
        record_run(
            store,
            run_id=f"replay-{day.strftime('%Y%m%d')}",
            started=day + pd.Timedelta(hours=21, minutes=10),
            finished=day + pd.Timedelta(hours=21, minutes=11),
            status="ok",
            reason=None,
            expected=day,
            tickers=list(DEMO_IDS),
            data_class=DataClass.SYNTHETIC,
            health_inputs=home_dir / "health_inputs",
            sessions=weekdays,
            calendar_id=DEMO_CALENDAR,
        )

    days = run_replay(StockLabStore(path), sessions=sessions_count, record_day=record_day)
    last = pd.Timestamp(days[-1].as_of)
    morning = (last + pd.offsets.BDay(1)).tz_localize(None).tz_localize("America/New_York")
    default = morning + pd.Timedelta(hours=7, minutes=40)
    now = _moment(at, lambda: default)
    records = ReadOnlyRecords(path)
    names = {spec.stock_id: spec.name for spec in DEMOS}

    def home() -> dict:
        view = build_home(
            records,
            now=now,
            calendar=weekdays,
            calendar_id=DEMO_CALENDAR,
            data_class=DataClass.SYNTHETIC,
            recent=demo_prices(records),
            local_zone=None,  # this Mac's own zone, at each moment
            tickers=list(DEMO_IDS),
            names=names,
        )
        return {**view, "research_url": research_url}

    def stock(symbol: str) -> dict | None:
        return build_detail(
            records,
            symbol=symbol,
            now=now,
            calendar=weekdays,
            calendar_id=DEMO_CALENDAR,
            data_class=DataClass.SYNTHETIC,
            recent=demo_prices(records),
            local_zone=None,
            tickers=list(DEMO_IDS),
            names=names,
            research_url=research_url,
        )

    return {"/api/home": home}, {STOCK_ROUTE: stock}


# --- serving ---


def serve_forever(server) -> None:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _bind(port: int, out: TextIO):
    """The app and its bound server, before any slow work; None when the port is busy."""

    app = LabApp(port=port, api={}, static=static_files())
    try:
        return app, make_server(app)
    except (OSError, OverflowError) as error:  # busy, not allowed, or not a port at all
        print(
            f"refused: port {port} can't be used "
            f"({getattr(error, 'strerror', None) or type(error).__name__}); "
            "is the Lab app already open? Pass --port to use another.",
            file=out,
        )
        return None


def _serve(app: LabApp, server, routes, args, *, serve: Callable, out: TextIO) -> int:
    api, items = routes
    app.api, app.items = dict(api), dict(items)
    url = f"http://{HOST}:{app.port}/"
    kind = "a synthetic demo" if args.demo is not None else "this Mac's Stock Lab, read-only"
    print(f"Lab app: {url} ({kind}). Press Control-C to stop.", file=out, flush=True)
    if args.open:
        webbrowser.open(url)
    serve(server)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    records_factory: Callable[[], ReadOnlyRecords] = ReadOnlyRecords,
    clock: Callable[[], pd.Timestamp] = utc_now,
    recent: Callable = observed,
    serve: Callable = serve_forever,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--demo", type=int, nargs="?", const=DEMO_SESSIONS, metavar="N")
    parser.add_argument("--at", metavar="TIME")
    parser.add_argument("--research-url", metavar="URL")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if args.research_url and not local_url(args.research_url):
        print("refused: --research-url must be a page on this Mac (http://127.0.0.1:...)", file=out)
        return 2
    if args.demo is not None and not 2 <= args.demo <= 60:
        print("refused: --demo takes 2 to 60 sessions", file=out)
        return 2
    try:
        fixed = None if args.at is None else _moment(args.at, clock)
    except ValueError as error:
        print(f"refused: {error}", file=out)
        return 2
    bound = _bind(args.port, out)
    if bound is None:
        return 2
    app, server = bound
    try:
        if args.demo is None:
            routes = real_api(
                records_factory=records_factory,
                clock=(lambda: fixed) if fixed else (lambda: _moment(None, clock)),
                recent=recent,
                research_url=args.research_url,
            )
            return _serve(app, server, routes, args, serve=serve, out=out)
        with tempfile.TemporaryDirectory(prefix="lab-app-demo-") as home_dir:
            routes = demo_api(Path(home_dir), args.demo, args.at, args.research_url)
            return _serve(app, server, routes, args, serve=serve, out=out)
    except ValueError as error:
        server.server_close()
        print(f"refused: {error}", file=out)
        return 2


def _stop(signum, frame) -> None:
    raise KeyboardInterrupt  # a polite stop (SIGTERM) ends like Control-C: the demo store goes too


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    raise SystemExit(main())
