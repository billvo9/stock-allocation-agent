"""
Fetch the trailing inference window of the Stock Lab's watchlist from Tiingo
into the separate inference store (owner-run, private, local).

Prepared, not yet run. It requests protected-period prices (2025 on), so it
runs only after the owner has configured TIINGO_API_KEY locally and
authorized this exact scope (decisions A1 and A2,
docs/research/stock_probability_lab.md, section 7). Without --authorized it
makes no request and prints the plan (exit 0).

Scope (one run):
    N requests  one prices request per ticker of the saved watchlist whose
                latest listing check in the research store is "supported" (at
                most 30; a longer watchlist is refused before any request),
                for the INFERENCE_SESSIONS (252) NYSE sessions ending at the
                last complete session (a close counts as complete from 20:00
                America/New_York): raw open, high, low, close, volume, divCash
                and splitFactor. No date before 2025 and no listing is
                requested: the research fetch's listing check is used, so
                this runs only after the research history is stored.

Read-only by default: with --authorized it prints each ticker's coverage and
corporate-action audit (stock_lab.audit), never a price series or the key,
and writes nothing. With --authorized --store (a separate approval) it writes
one write-once batch per ready ticker into the inference store
(stock_lab.inference: its own private file and directory, outside every
repository and apart from the research store), then reads the window back
through the frozen-model reader as a check. The research store is only read
(the watchlist and its checks), never written.

A ticker is ready when its audit is and the provider served every session of
the window. A session it did not serve is held for the owner's review, not
stored: a split or dividend on it would be invisible to the audit (the batch
would record none there), and there is deliberately no override yet.

Exit codes: 0 success; 1 a refused request, response or write; 2 refused
before any request.

Usage:
    .venv/bin/python scripts/stock_lab_fetch_recent.py                        # the plan only
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized           # read-only run
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized --store   # store (approval)
The tickers are always the saved watchlist (add or remove them in the Lab).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.data.vendors.tiingo import TiingoRequestFailed
from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.contracts import DataClass, SymbolStatus
from stock_agent.stock_lab.inference.schema import (
    INFERENCE_SESSIONS,
    PROTECTED_START,
    InferenceRefused,
    Refusal,
    close_available_at,
)
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.inference.window import load_inference_window
from stock_agent.stock_lab.store import StockLabStore, utc_now
from stock_agent.stock_lab.symbols import parse_ticker, refused
from stock_agent.stock_lab.tiingo import (
    SOURCE_ID,
    VINTAGE_NOTE,
    NotAuthorized,
    StockAdapter,
    StockDataRefused,
    prices_url,
)

MAX_TICKERS = 30  # the authorized scope (A1); never raised by an option


def shown(path: Path) -> str:
    """A path for printed output, with the home folder as ~ (output may be pasted publicly)."""

    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


CALENDAR_SPAN = pd.Timedelta(days=2 * INFERENCE_SESSIONS)


def last_complete_session(now: pd.Timestamp) -> pd.Timestamp:
    """The last NYSE session whose close is available at `now` (20:00 America/New_York)."""

    today = pd.Timestamp(now.tz_convert("America/New_York").date())
    candidates = sessions(today - pd.Timedelta(days=10), today)
    if now < close_available_at(candidates[-1]):
        candidates = candidates[:-1]
    return candidates[-1]


def request_window(now: pd.Timestamp) -> pd.DatetimeIndex:
    """The INFERENCE_SESSIONS sessions ending at the last complete session."""

    end = last_complete_session(now)
    return sessions(end - CALENDAR_SPAN, end)[-INFERENCE_SESSIONS:]


def on_request(bars: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Bars with one row per session of the request; a session not served is a row without prices."""

    outside = ~bars["session"].isin(days)
    if outside.any():
        raise StockDataRefused("non_session", f"a bar on {bars.loc[outside, 'session'].iloc[0]}")
    full = bars.set_index("session").reindex(days)
    full["split_factor"] = full["split_factor"].fillna(1.0)
    full["div_cash"] = full["div_cash"].fillna(0.0)
    return full.rename_axis("session").reset_index()


def plan(
    tickers: Sequence[str],
    supported: Sequence[str],
    days: pd.DatetimeIndex,
    research: StockLabStore,
    inference: InferenceStore,
    out: TextIO,
) -> None:
    first, last = days[0].date(), days[-1].date()
    print("Stock Lab recent-window plan (no request made; pass --authorized to run):", file=out)
    saved = ", ".join(tickers) or "(none saved)"
    print(f"watchlist ({len(tickers)} of at most {MAX_TICKERS}): {saved}", file=out)
    print(
        f"window: the {len(days)} NYSE sessions {first} to {last} (protected period: inputs to "
        "a frozen model only, never research, training, tuning or backtests)",
        file=out,
    )
    print("requests, in order (each at least 2 seconds apart; the key is sent only to", file=out)
    print("api.tiingo.com, in the Authorization header, never in a URL):", file=out)
    for number, ticker in enumerate(supported, start=1):
        print(f"  {number}. GET {prices_url(ticker, first, last)}", file=out)
    if not supported:
        print("  none: no watched ticker has a supported listing check in the research", file=out)
        print("  store yet (scripts/stock_lab_fetch.py --authorized --store stores it)", file=out)
    skipped = [ticker for ticker in tickers if ticker not in supported]
    if skipped:
        print(f"  not requested (no supported listing check): {', '.join(skipped)}", file=out)
    print(
        "local side effects of --authorized without --store: none (no record is written;", file=out
    )
    print(f"  the watchlist and its checks are read from {shown(research.path)})", file=out)
    print(
        "with --store (a separate approval): one batch per ticker whose audit is ready is", file=out
    )
    print(f"  written to the inference store {shown(inference.path)}; the research store", file=out)
    print("  is never written; a ticker with an unserved session is held for review", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[..., StockAdapter] = StockAdapter,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    inference_factory: Callable[[], InferenceStore] = InferenceStore,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    args = parser.parse_args(argv)
    research = store_factory()

    entries = [parse_ticker(text) for text in research.watched()]
    for entry in entries:
        if not entry.ok:
            print(f"refused {entry.entered}: {refused(entry).reason.value}", file=out)
    tickers = list(dict.fromkeys(entry.ticker for entry in entries if entry.ok))
    if len(tickers) > MAX_TICKERS:
        print(f"refused: {len(tickers)} tickers; at most {MAX_TICKERS} per run", file=out)
        return 2
    if args.store and not args.authorized:
        print("refused: --store needs --authorized", file=out)
        return 2
    days = request_window(clock())
    if days[0].date() < PROTECTED_START:
        print("refused: the window would reach development data (before 2025)", file=out)
        return 2
    checks = research.latest_checks()
    supported = [
        ticker
        for ticker in tickers
        if ticker in checks and checks[ticker].status is SymbolStatus.SUPPORTED
    ]
    try:
        inference = inference_factory()
    except InferenceRefused as error:
        print(f"refused: the inference store's location ({error})", file=out)
        return 2
    if not args.authorized:
        plan(tickers, supported, days, research, inference, out)
        return 0
    if not supported:
        print("nothing to fetch: no watched ticker has a supported listing check yet", file=out)
        return 0

    first, last = days[0].date(), days[-1].date()
    adapter = adapter_factory(authorized=True, max_requests=len(supported))
    ready: list[str] = []
    review: list[str] = []
    try:
        for ticker in supported:
            response = adapter.prices(ticker, first, last)
            bars = on_request(response.bars, days)
            report = audit_bars(ticker, bars)
            for line in audit_lines(report):
                print(line, file=out)
            unserved = int(bars["close"].isna().sum())
            if unserved:
                print(
                    f"  {unserved} session(s) not served: a split or dividend there would be "
                    "invisible, so the ticker is held for review",
                    file=out,
                )
            storable = report["ready"] and not unserved
            (ready if storable else review).append(ticker)
            if not args.store:
                continue
            if not storable:
                print("  not stored: held for review first", file=out)
                continue
            batch = inference.record_batch(
                ticker,
                bars,
                source=SOURCE_ID,
                retrieved_at=response.retrieved_at,
                request_start=days[0],
                request_end=days[-1],
                sessions=sessions,
                calendar_id=CALENDAR_ID,
                data_class=DataClass.REAL,
                vintage_note=VINTAGE_NOTE,
            )
            print(f"  stored batch {batch.batch_id}", file=out)
            try:
                window = load_inference_window(
                    inference.path,
                    ticker,
                    days[-1],
                    retrieved_by=clock(),
                    sessions=sessions,
                    calendar_id=CALENDAR_ID,
                )
            except InferenceRefused as error:
                if error.reason is not Refusal.AFTER_NEXT_OPEN:
                    raise
                print(f"  read back: the {last} window is past its decision time", file=out)
            else:
                print(
                    f"  read back: the {last} window has {len(window.sessions)} sessions "
                    f"and {len(window.gaps)} gaps",
                    file=out,
                )
    except (TiingoRequestFailed, StockDataRefused, NotAuthorized, InferenceRefused) as error:
        print(f"stopped: {error}", file=out)
        return 1
    print(
        f"audit summary: {len(ready)} ready ({', '.join(ready) or 'none'}); "
        f"{len(review)} to review ({', '.join(review) or 'none'})",
        file=out,
    )
    if not args.store:
        print("nothing was written; storing is a separate, explicit step (--store)", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
