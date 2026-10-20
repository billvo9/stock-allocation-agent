"""
Prepare the Stock Lab's premarket briefing from stored records (owner-run,
private, local; read-only on the store).

It opens the Stock Lab's store read-only (the watchlist, the recorded
forecasts, their scores, the evaluations and the daily job's runs) and, for
real stocks, each one's latest observed close through the inference store's
isolated descriptive reader (decision A6; only batches stored by the
preparation time), prepares
the briefing for the next NYSE session (the target is 08:30 New York time
that morning) and prints it as plain text. It never retrains, never runs the
evening job, never writes to the store, sends nothing anywhere and reads no
credential. A stock without a stored forecast is shown as unavailable, with
its reason; no probability is ever filled in.

Modes
    (default)       print the plain-text briefing for the watchlist
    --preview       also write its HTML, text and JSON, once per market date,
                    into the private briefing directory
                    (STOCK_AGENT_STOCK_LAB_BRIEFING_HOME, or
                    ~/stock-agent-private/stock_lab_briefings/; refused inside
                    any Git working tree); open the HTML file locally
    --demo [N]      a synthetic demo: replay N synthetic sessions (default 8)
                    into a temporary store that is removed afterwards, and
                    brief the demo stocks; nothing real is read
Options
    --at TIME       prepare as of TIME (ISO 8601 with a time zone), for a
                    preview: only records that existed by then are read
    --include-closed-days
                    prepare on a weekend or market holiday (for the next
                    session); a scheduled run skips those days
    --tickers ...   brief these tickers instead of the watchlist
    --exclude ...   leave these out (default: SKHY SNDK, the owner's choice)

Nothing is scheduled or installed here: scripts/stock_lab_schedule.py
--briefing prints an 08:30 launchd agent that runs `--preview`, for the
owner to install once approved, and --email-dry-run describes an email
adapter without one existing. The Portfolio Lab's Premarket Briefing page
shows the written previews, read-only. Printed paths show the home folder
as ~.

Usage:
    .venv/bin/python scripts/stock_lab_briefing.py --demo
    .venv/bin/python scripts/stock_lab_briefing.py --demo --preview
    .venv/bin/python scripts/stock_lab_briefing.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab.briefing.delivery import (
    DeliveryLog,
    DeliveryRefused,
    PreviewAdapter,
    RenderedBriefing,
    default_directory,
    deliver_once,
)
from stock_agent.stock_lab.briefing.render import render_html, render_text
from stock_agent.stock_lab.briefing.report import (
    Briefing,
    ReadOnlyRecords,
    briefing_document,
    build_briefing,
    subject,
)
from stock_agent.stock_lab.briefing.timing import (
    CLOSED_DAY,
    PREMARKET,
    TimingRefused,
    briefing_timing,
    long_date,
    scheduled_at,
)
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.inference.recent import read_recent
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, StoreRefused, utc_now
from stock_agent.ui import theme

EXCLUDED = ("SKHY", "SNDK")  # the owner's choice: kept off the briefing
DEMO_SESSIONS = 8
DEMO_CALENDAR = "synthetic_weekdays"


def briefing_theme() -> dict:
    """The Lab's palette for the HTML briefing, light and dark (stock_agent.ui.theme)."""

    fonts = {key: theme.LAB_TYPE[key] for key in ("serif", "sans", "mono")}
    return {
        "light": {
            "colours": theme.LAB_COLOURS,
            "status": theme.STATUS,
            "direction": theme.DIRECTION_COLOURS,
        },
        "dark": {
            "colours": theme.LAB_DARK_COLOURS,
            "status": theme.LAB_DARK_STATUS,
            "direction": theme.DIRECTION_DARK_COLOURS,
        },
        "fonts": fonts,
    }


def weekdays(start, end) -> pd.DatetimeIndex:
    """The synthetic demo's calendar: every weekday (the replay's)."""

    return pd.bdate_range(start, end, tz="UTC")


def shown(path: Path) -> str:
    """A path with the home folder as ~."""

    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text


def rendered(briefing: Briefing) -> RenderedBriefing:
    label = "premarket"
    if briefing.state != PREMARKET:
        moment = pd.Timestamp(briefing.generated_at).strftime("%H%M%S")
        label = f"{briefing.state.replace('_', '-')}-preview-{moment}"
    return RenderedBriefing(
        market_date=briefing.market_date,
        data_class=briefing.data_class,
        subject=subject(briefing),
        text=render_text(briefing),
        html=render_html(briefing, briefing_theme()),
        document=json.dumps(briefing_document(briefing), allow_nan=False, sort_keys=True, indent=1),
        label=label,
    )


def _emit(briefing: Briefing, *, preview: bool, directory: Path | None, out: TextIO) -> int:
    """
    Print the text; with --preview also write the files. Only a premarket briefing goes
    through the duplicate-send log (once per market date): a preview prepared at any other
    time is written under its own name and never counts as that day's delivery.
    """

    output = rendered(briefing)
    out.write(output.text)
    if not preview:
        return 0
    try:
        adapter = PreviewAdapter(directory or default_directory())
        if briefing.state == PREMARKET:
            result = deliver_once(output, adapter, DeliveryLog(adapter.directory))
            status, detail = result.status, result.detail
        else:
            status, detail = "written", adapter.deliver(output) + "; a preview, not a delivery"
    except DeliveryRefused as error:
        print(f"refused: {error}", file=out)
        return 2
    except (OSError, sqlite3.Error) as error:  # the type only: no path or message is printed
        print(f"failed: the preview could not be written ({type(error).__name__})", file=out)
        return 1
    print(f"\npreview: {status} ({detail})", file=out)
    if status in ("sent", "written"):
        page = adapter.paths(output)["html"]
        print(f"open the HTML page: open {shown(page)}", file=out)
    return 1 if status == "failed" else 0


def _timestamp(text: str | None, clock: Callable[[], pd.Timestamp]) -> pd.Timestamp:
    if text is None:
        return clock()
    moment = pd.Timestamp(text)
    if moment.tzinfo is None:
        raise ValueError("--at needs a time zone, for example 2026-10-19T08:30-04:00")
    return moment.tz_convert("UTC")


def _demo(args, *, clock: Callable[[], pd.Timestamp], out: TextIO) -> int:
    from stock_agent.stock_lab.job import record_run
    from stock_agent.stock_lab.replay import run_replay  # research side: the demo only
    from stock_agent.stock_lab.synthetic import DEMO_IDS

    if not 1 <= args.demo <= 60:
        print("refused: --demo takes 1 to 60 sessions", file=out)
        return 2
    with tempfile.TemporaryDirectory(prefix="stock-lab-briefing-demo-") as home:
        os.chmod(home, 0o700)
        path = Path(home) / DB_NAME

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
                health_inputs=Path(home) / "health_inputs",
                sessions=weekdays,
                calendar_id=DEMO_CALENDAR,
            )

        days = run_replay(StockLabStore(path), sessions=args.demo, record_day=record_day)
        last = pd.Timestamp(days[-1].as_of).date()
        following = [
            d.date() for d in weekdays(last, last + pd.Timedelta(days=7)) if d.date() > last
        ]
        default = pd.Timestamp(scheduled_at(following[0])).tz_convert("UTC")
        now = _timestamp(args.at, lambda: default)
        plan = briefing_timing(now.to_pydatetime(), weekdays)
        briefing = build_briefing(
            ReadOnlyRecords(path),
            timing=plan,
            calendar=weekdays,
            calendar_id=DEMO_CALENDAR,
            data_class=DataClass.SYNTHETIC,
            tickers=list(DEMO_IDS),
        )
        return _emit(briefing, preview=args.preview, directory=args.directory, out=out)


def observed(symbol: str, *, retrieved_by: pd.Timestamp):
    """A real stock's recent observed prices as stored by `retrieved_by` (decision A6)."""

    return read_recent(symbol, retrieved_by=retrieved_by)


def main(
    argv: Sequence[str] | None = None,
    *,
    records_factory: Callable[[], ReadOnlyRecords] = ReadOnlyRecords,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
    recent: Callable = observed,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--demo", type=int, nargs="?", const=DEMO_SESSIONS, metavar="N")
    parser.add_argument("--at", metavar="TIME")
    parser.add_argument("--include-closed-days", action="store_true")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--exclude", nargs="*", default=list(EXCLUDED), metavar="TICKER")
    parser.add_argument("--directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.demo is not None:
            return _demo(args, clock=clock, out=out)
        now = _timestamp(args.at, clock)
        plan = briefing_timing(now.to_pydatetime(), sessions)
    except (ValueError, TimingRefused, CalendarRefused) as error:
        print(f"refused: {error}", file=out)
        return 2
    if plan.state == CLOSED_DAY and not args.include_closed_days:
        today = now.tz_convert("America/New_York").date()
        print(
            f"skipped: the market is closed on {long_date(today)} (weekend or holiday); the next "
            f"briefing is for {long_date(plan.market_date)}. Pass --include-closed-days to "
            "preview it.",
            file=out,
        )
        return 0
    try:
        briefing = build_briefing(
            records_factory(),
            timing=plan,
            calendar=sessions,
            calendar_id=CALENDAR_ID,
            data_class=DataClass.REAL,
            tickers=args.tickers,
            excluded=args.exclude,
            recent=recent,
        )
    except (StoreRefused, CalendarRefused) as error:
        print(f"failed: {error}", file=out)
        return 1
    return _emit(briefing, preview=args.preview, directory=args.directory, out=out)


if __name__ == "__main__":
    raise SystemExit(main())
