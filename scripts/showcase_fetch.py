"""
Fetch the exploratory showcase's 2003-2024 history from Tiingo into the
separate showcase store (owner-run, private, local).

Prepared, not yet run: it contacts Tiingo, so run it only after the owner has
authorized this exact scope. Without --authorized it makes no request and
prints the plan (exit 0).

Scope (one run, from configs/showcase/real_rebalancing_v1.yaml):
    1 request    the supported-tickers listing (apimedia.tiingo.com; no key is
                 sent to that host). Each ticker is classified as the asset
                 type it is registered as (a stock, or an ETF for the sector
                 funds): US exchange, USD, recent coverage (symbols.classify).
                 The listing's dates are used only to classify tickers.
    N requests   one prices request per supported registered ticker (at most
                 31) for 2003-01-02 through 2024-12-31 only: raw open, high,
                 low, close, volume, divCash and splitFactor. No date after
                 2024 is requested, and the frozen calendar refuses one.

That is at most 32 requests, under the free plan's 50 an hour. The 20:30
New York evening job uses the same key, so an authorized run is refused on
weekdays from 19:30 to 20:45 New York: running twice in that hour could
leave the evening without requests, and a missed evening is a permanent gap.

Read-only by default: with --authorized it prints each ticker's coverage and
corporate-action audit (stock_lab.audit), never a price series or the key,
and writes nothing. With --authorized --store (a separate approval) it audits
and stores in the same run, from the same responses, so the hourly limit is
not spent twice: the listing snapshot, the checks and one write-once batch
per ticker whose audit is ready go into the showcase store
(stock_agent.showcase.store: ~/stock-agent-private/showcase_store/, apart
from every Stock Lab store). A ticker whose audit asks for review is reported
and not stored. The Stock Lab's watchlist, stores, evening job and frozen
model are never touched. A run stopped part-way (a refused response, the
request budget) keeps what it already stored, each record write-once; a later
run stores the rest, and an identical retrieval is not stored twice. A served
session outside the request (2025, say) refuses that response: nothing from it
is stored.

The key. By default it comes from TIINGO_API_KEY. With --keychain it is read
from the macOS Keychain (the item the evening job uses), once, just before the
first request, and handed to the adapter in memory only.

Exit codes: 0 success; 1 a refused request, response or write; 2 refused
before any request.

Usage:
    .venv/bin/python scripts/showcase_fetch.py                                  # the plan only
    .venv/bin/python scripts/showcase_fetch.py --authorized --keychain          # read-only audit
    .venv/bin/python scripts/showcase_fetch.py --authorized --store --keychain  # audit and store
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import date, time, timedelta
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.vendors.keychain import KeychainRefused, add_command, read_key
from stock_agent.data.vendors.tiingo import TiingoRequestFailed
from stock_agent.showcase.config import MAX_TICKERS, load_config
from stock_agent.showcase.store import ShowcaseStoreRefused, open_store
from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.contracts import DataClass, SymbolStatus
from stock_agent.stock_lab.store import StockLabStore, utc_now
from stock_agent.stock_lab.symbols import ETF_ASSET_TYPES, STOCK_ASSET_TYPES, classify
from stock_agent.stock_lab.tiingo import (
    LISTING_URL,
    SOURCE_ID,
    VINTAGE_NOTE,
    NotAuthorized,
    StockAdapter,
    StockDataRefused,
    prices_url,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "showcase" / "real_rebalancing_v1.yaml"
CUTOFF_WEEKDAYS = 3  # a current listing reaches within 3 weekdays of the listing's newest date
EVENING_QUIET = (time(19, 30), time(20, 45))  # New York, weekdays: the evening job's hour
ASSET_TYPES = {"etf": ETF_ASSET_TYPES, "stock": STOCK_ASSET_TYPES}


def shown(path: Path) -> str:
    """A path for printed output, with the home folder as ~ (output may be pasted publicly)."""

    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def weekdays_after(day: date, until: date) -> int:
    """Weekdays after `day` through `until` (a stand-in for sessions; holidays not removed)."""

    return int(np.busday_count(day + timedelta(days=1), until + timedelta(days=1)))


def in_evening_quiet(now: pd.Timestamp) -> bool:
    """True on a weekday from 19:30 to 20:45 New York, the evening job's hour."""

    local = now.tz_convert("America/New_York")
    return local.dayofweek < 5 and EVENING_QUIET[0] <= local.time() < EVENING_QUIET[1]


def on_nyse_calendar(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars with a row for every NYSE session in their span; a session not served stays a gap."""

    if bars.empty:
        return bars
    calendar = nyse.sessions(bars["session"].iloc[0].date(), bars["session"].iloc[-1].date())
    outside = ~bars["session"].isin(calendar)
    if outside.any():
        raise StockDataRefused("non_session", f"a bar on {bars.loc[outside, 'session'].iloc[0]}")
    full = bars.set_index("session").reindex(calendar)
    full["split_factor"] = full["split_factor"].fillna(1.0)
    full["div_cash"] = full["div_cash"].fillna(0.0)
    return full.rename_axis("session").reset_index()


def same_day_split_and_dividend(bars: pd.DataFrame) -> list[str]:
    """Sessions with both a split and a dividend: F5.1 needs a share basis the data lacks."""

    both = (bars["split_factor"] != 1.0) & (bars["div_cash"] > 0)
    return [pd.Timestamp(s).strftime("%Y-%m-%d") for s in bars.loc[both, "session"]]


def plan(config, out: TextIO, store: StockLabStore, keychain: bool) -> None:
    tickers = config.tickers()
    print("Showcase fetch plan (no request made; pass --authorized to run):", file=out)
    print(
        f"configuration {config.name} ({len(tickers)} of at most {MAX_TICKERS} tickers)", file=out
    )
    print("requests, in order (each at least 2 seconds apart; the key is sent only to", file=out)
    print("api.tiingo.com, in the Authorization header, never in a URL):", file=out)
    print(f"  1. GET {LISTING_URL} (no key)", file=out)
    for number, ticker in enumerate(tickers, start=2):
        print(f"  {number}. GET {prices_url(ticker, config.start, config.end)}", file=out)
    print(f"  {len(tickers) + 1} requests in all, under the free plan's 50 an hour", file=out)
    print("  (a ticker the listing does not classify as its registered type is skipped)", file=out)
    source = "the macOS Keychain" if keychain else "TIINGO_API_KEY"
    print(f"the key: read from {source}, never printed", file=out)
    print("refused on weekdays from 19:30 to 20:45 New York (the evening job's hour)", file=out)
    print("local side effects of --authorized without --store: none", file=out)
    print("with --store (a separate approval): the listing, the checks and one batch per", file=out)
    print(f"  ticker whose audit is ready are written to {shown(store.path)}", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[..., StockAdapter] = StockAdapter,
    store_factory: Callable[[], StockLabStore] = open_store,
    key_reader: Callable[[], str] = read_key,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--keychain", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(CONFIG)  # refuses a window past 2024-12-31 (the sealed holdout)
    if args.store and not args.authorized:
        print("refused: --store needs --authorized", file=out)
        return 2
    try:
        store = store_factory()
    except ShowcaseStoreRefused as error:
        print(f"refused: the showcase store's location ({error})", file=out)
        return 2
    if not args.authorized:
        plan(config, out, store, args.keychain)
        return 0
    if in_evening_quiet(clock()):
        print(
            "refused: the 20:30 evening job shares the key's 50 requests an hour; run this "
            "before 19:30 or after 20:45 New York",
            file=out,
        )
        return 2

    key_source = {}
    if args.keychain:
        try:
            key = key_reader()
        except KeychainRefused as error:
            print(
                f"refused: the Tiingo key could not be read from the macOS Keychain ({error}); "
                f"nothing was requested. To add it, type it at the prompt of: {add_command()}",
                file=out,
            )
            return 2
        key_source["token_provider"] = lambda: key

    kinds = config.kinds()
    tickers = config.tickers()
    adapter = adapter_factory(authorized=True, max_requests=len(tickers) + 1, **key_source)
    audited: list[dict] = []
    stored: list[str] = []
    try:
        adapter.check_key()  # before any request or write
        listing = adapter.listing()
        ends = [row.end_date for row in listing.rows if row.end_date is not None]
        if not ends:
            raise StockDataRefused("listing_dates", "no row has an end date")
        newest = max(ends)
        cutoff = newest
        while weekdays_after(cutoff, newest) < CUTOFF_WEEKDAYS:
            cutoff -= timedelta(days=1)
        checks = [
            classify(
                ticker,
                listing.rows,
                recent_cutoff=cutoff,
                sessions_since=lambda day: weekdays_after(day, newest),
                asset_types=ASSET_TYPES[kinds[ticker]],
            )
            for ticker in tickers
        ]
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
            response = adapter.prices(check.ticker, config.start, config.end)
            bars = on_nyse_calendar(response.bars)
            if bars.empty:
                print(f"{check.ticker}: no rows served", file=out)
                continue
            report = audit_bars(check.ticker, bars)
            for line in audit_lines(report):
                print(line, file=out)
            both = same_day_split_and_dividend(bars)
            if both:
                print(
                    f"  a split and a dividend on the same session ({', '.join(both)}): that "
                    "return is unavailable (no share basis), and a portfolio holding it stops",
                    file=out,
                )
            audited.append(report)
            if not args.store:
                continue
            if not report["ready"]:
                print("  not stored: the audit needs review first", file=out)
                continue
            batch = store.record_prices(
                check.ticker,
                bars,
                source=SOURCE_ID,
                retrieved_at=response.retrieved_at,
                data_class=DataClass.REAL,
                vintage_note=VINTAGE_NOTE,
            )
            stored.append(check.ticker)
            print(f"  stored batch {batch.batch_id}", file=out)
    except (TiingoRequestFailed, StockDataRefused, NotAuthorized) as error:
        print(f"stopped: {error}", file=out)
        return 1
    print(f"requests made: {adapter.requests_made}", file=out)
    ready = [r["ticker"] for r in audited if r["ready"]]
    review = [r["ticker"] for r in audited if not r["ready"]]
    skipped = [c.ticker for c in checks if c.status is not SymbolStatus.SUPPORTED]
    print(
        f"audit summary: {len(ready)} ready ({', '.join(ready) or 'none'}); "
        f"{len(review)} to review ({', '.join(review) or 'none'}); "
        f"{len(skipped)} not fetched ({', '.join(skipped) or 'none'})",
        file=out,
    )
    if args.store:
        print(f"stored {len(stored)} of {len(tickers)} tickers in {shown(store.path)}", file=out)
    else:
        print("nothing was written; storing is a separate, explicit step (--store)", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
