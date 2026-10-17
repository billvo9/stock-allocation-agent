"""
Fetch development history for the Stock Lab's watchlist from Tiingo
(owner-run, private, local).

Prepared, not yet run: it contacts Tiingo, so run it only after the owner
has configured TIINGO_API_KEY locally and authorized this exact scope.
Without --authorized it makes no request and prints the plan (exit 0).

Scope (one run):
    1 request    the supported-tickers listing (apimedia.tiingo.com; no key
                 is sent to that host). It classifies each ticker: US
                 exchange, stock, USD, recent coverage (symbols.classify).
                 The listing's dates are current (they reach past 2024): they
                 are used only to classify tickers, never as a model input or
                 in an evaluation.
    N requests   one prices request per supported ticker of the saved
                 watchlist (at most 30; a longer watchlist is refused before
                 any request) for 2003-01-02 through 2024-12-31 only: raw
                 open, high, low, close, volume, divCash and splitFactor. No
                 date after 2024 is requested.

Read-only by default: without --store it prints the coverage and
corporate-action audit of every ticker (stock_lab.audit: coverage, missing
NYSE sessions, null fields, low/high sanity, every split with its raw and
split-consistent close ratios and a verdict, ex-dividend dates and the
largest yield, large moves without a split, unavailable returns), never a
price series or the key, and writes nothing. With --store it writes the
whole listing snapshot (every row, for the classification's audit trail),
the checks and one write-once price batch per ticker into the private Stock
Lab store (stock_lab.store; outside the repository), only for tickers whose
audit is ready (the others are reported, not stored). Sessions of the NYSE
calendar that Tiingo did not serve are stored as rows without prices, so a
gap stays a gap.

Usage:
    .venv/bin/python scripts/stock_lab_fetch.py                         # the plan only
    .venv/bin/python scripts/stock_lab_fetch.py --authorized            # read-only run
    .venv/bin/python scripts/stock_lab_fetch.py --authorized --store    # store the batch
The tickers are always the saved watchlist (add or remove them in the Lab).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import TextIO

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.vendors.tiingo import TiingoRequestFailed
from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.contracts import DEVELOPMENT_END, DataClass, SymbolStatus
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.symbols import classify, parse_ticker, refused
from stock_agent.stock_lab.tiingo import (
    LISTING_URL,
    SOURCE_ID,
    VINTAGE_NOTE,
    NotAuthorized,
    StockAdapter,
    StockDataRefused,
    prices_url,
)

HISTORY_START = date(2003, 1, 2)  # the frozen NYSE calendar's first session
MAX_TICKERS = 30  # the authorized scope (A1); never raised by an option
CUTOFF_WEEKDAYS = 3  # a current listing reaches within 3 weekdays of the listing's newest date


def weekdays_after(day: date, until: date) -> int:
    """Weekdays after `day` through `until` (a stand-in for sessions; holidays not removed)."""

    return int(np.busday_count(day + timedelta(days=1), until + timedelta(days=1)))


def on_nyse_calendar(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Bars with a row for every NYSE session in their span; and how many were added."""

    if bars.empty:
        return bars, 0
    calendar = nyse.sessions(bars["session"].iloc[0].date(), bars["session"].iloc[-1].date())
    outside = ~bars["session"].isin(calendar)
    if outside.any():
        raise StockDataRefused("non_session", f"a bar on {bars.loc[outside, 'session'].iloc[0]}")
    full = bars.set_index("session").reindex(calendar)
    full["split_factor"] = full["split_factor"].fillna(1.0)
    full["div_cash"] = full["div_cash"].fillna(0.0)
    return full.rename_axis("session").reset_index(), len(calendar) - len(bars)


def plan(tickers: Sequence[str], out: TextIO, store: StockLabStore) -> None:
    print("Stock Lab fetch plan (no request made; pass --authorized to run):", file=out)
    saved = ", ".join(tickers) or "(none saved)"
    print(f"watchlist ({len(tickers)} of at most {MAX_TICKERS}): {saved}", file=out)
    print("requests, in order (each at least 2 seconds apart; the key is sent only to", file=out)
    print("api.tiingo.com, in the Authorization header, never in a URL):", file=out)
    print(f"  1. GET {LISTING_URL} (no key)", file=out)
    for number, ticker in enumerate(tickers, start=2):
        print(f"  {number}. GET {prices_url(ticker, HISTORY_START, DEVELOPMENT_END)}", file=out)
    print("  (a ticker the listing does not classify as a supported US stock is skipped)", file=out)
    print(
        "local side effects of --authorized without --store: none (no record is written;", file=out
    )
    print(f"  the watchlist is read from {store.path}, which may create SQLite's", file=out)
    print("  -shm/-wal companion files beside it)", file=out)
    print("with --store (a separate approval): the listing, the checks and one batch per", file=out)
    print(f"  ticker whose audit is ready are written to {store.path}", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[..., StockAdapter] = StockAdapter,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    args = parser.parse_args(argv)
    store = store_factory()

    entries = [parse_ticker(text) for text in store.watched()]
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
    if not args.authorized:
        plan(tickers, out, store)
        return 0
    if not tickers:
        print("nothing to fetch: the watchlist is empty", file=out)
        return 0

    adapter = adapter_factory(authorized=True, max_requests=len(tickers) + 1)
    try:
        listing = adapter.listing()
        ends = [row.end_date for row in listing.rows if row.end_date is not None]
        if not ends:
            raise StockDataRefused("listing_dates", "no row has an end date")
        newest = max(ends)
        cutoff = date.fromordinal(newest.toordinal())
        while weekdays_after(cutoff, newest) < CUTOFF_WEEKDAYS:
            cutoff -= timedelta(days=1)
        checks = [
            classify(
                ticker,
                listing.rows,
                recent_cutoff=cutoff,
                sessions_since=lambda day: weekdays_after(day, newest),
            )
            for ticker in tickers
        ]
        snapshot_id = None
        audited: list[dict] = []
        if args.store:
            snapshot_id = store.record_listing(
                listing.rows,
                source=SOURCE_ID,
                retrieved_at=listing.retrieved_at,
                data_class=DataClass.REAL,
            )
            store.record_checks(checks, snapshot_id=snapshot_id)
        print(f"listing: {len(listing.rows)} rows; newest end date {newest}", file=out)
        for check in checks:
            print(f"  {check.ticker}: {check.status.value} ({check.reason.value})", file=out)
        for check in checks:
            if check.status is not SymbolStatus.SUPPORTED:
                continue
            response = adapter.prices(check.ticker, HISTORY_START, DEVELOPMENT_END)
            bars, _ = on_nyse_calendar(response.bars)
            if bars.empty:
                print(f"{check.ticker}: no rows served", file=out)
                continue
            report = audit_bars(check.ticker, bars)
            for line in audit_lines(report):
                print(line, file=out)
            audited.append(report)
            if args.store and not report["ready"]:
                print("  not stored: the audit needs review first", file=out)
                continue
            if args.store:
                batch = store.record_prices(
                    check.ticker,
                    bars,
                    source=SOURCE_ID,
                    retrieved_at=response.retrieved_at,
                    data_class=DataClass.REAL,
                    vintage_note=VINTAGE_NOTE,
                )
                print(f"  stored batch {batch.batch_id}", file=out)
    except (TiingoRequestFailed, StockDataRefused, NotAuthorized) as error:
        print(f"stopped: {error}", file=out)
        return 1
    ready = [r["ticker"] for r in audited if r["ready"]]
    review = [r["ticker"] for r in audited if not r["ready"]]
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
