"""
Fetch the sealed holdout's history from Tiingo into the private holdout store
(owner-run, once; stock_lab.holdout).

Prepared, not yet run. It requests protected-period prices, so it runs only
after the owner has authorized this exact scope (the holdout evaluation,
owner decision of 2026-10-26). Without --authorized it makes no request,
reads no key and prints the plan (exit 0).

Scope (one run):
    4 requests  one prices request per protocol stock (AMD, MRVL, MU, NVDA)
                whose latest listing check in the research store is
                "supported", for every NYSE session from FETCH_START
                (2024-01-02, the first session of the first decision's
                252-session window) through HOLDOUT_END (2026-10-14): raw
                open, high, low, close, volume, divCash and splitFactor. A
                stock already in the holdout store is not requested again.

Read-only by default: with --authorized it prints each stock's coverage and
corporate-action audit (stock_lab.audit) and how its 2024 sessions compare
with the research store's (the development history the model was frozen
from), never a price series or the key, and writes nothing. With
--authorized --store (the same authorization) it writes one batch per ready
stock into the holdout store (its own private file and directory, outside
every repository). The research store is only read, never written.

A stock is ready when its audit is, the provider served every session, and
its 2024 closes, dividends and split factors equal the research store's. Any
other stock is held for the owner's review, not stored.

The key. By default it comes from TIINGO_API_KEY. With --keychain it is read
instead from the macOS Keychain item the evening job uses
(stock_agent.data.vendors.keychain: service stock-agent-tiingo, account
stock-lab), once, just before the first request, and handed to the adapter in
memory only: never an environment variable, an argument, a file or a log. An
item that cannot be read refuses the run before any request (exit 2), naming
the reason, never the value. Nothing reads the key when nothing is requested.
The key is held only in this process while it runs: an ordinary traceback
shows no local values, but a debugger or a traceback that prints locals
would show it.

Exit codes: 0 success; 1 a refused request, response or write; 2 refused
before any request.

Usage:
    .venv/bin/python scripts/stock_lab_holdout_fetch.py                        # the plan only
    .venv/bin/python scripts/stock_lab_holdout_fetch.py --authorized           # read-only run
    .venv/bin/python scripts/stock_lab_holdout_fetch.py --authorized --store   # fetch and store
    .venv/bin/python scripts/stock_lab_holdout_fetch.py --authorized --store --keychain
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.data.vendors.keychain import (
    TIINGO_ACCOUNT,
    TIINGO_SERVICE,
    KeychainRefused,
    add_command,
    read_key,
)
from stock_agent.data.vendors.tiingo import TiingoRequestFailed
from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.contracts import DEVELOPMENT_END, SymbolStatus
from stock_agent.stock_lab.holdout.protocol import PROTOCOL, HoldoutRefused, Protocol
from stock_agent.stock_lab.holdout.store import HoldoutStore
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.tiingo import (
    SOURCE_ID,
    VINTAGE_NOTE,
    NotAuthorized,
    StockAdapter,
    StockDataRefused,
    prices_url,
)

COMPARED = ("close", "div_cash", "split_factor")


def shown(path: Path) -> str:
    """A path for printed output, with the home folder as ~ (output may be pasted publicly)."""

    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def on_request(bars: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Bars with one row per session of the request; a session not served is a row without prices."""

    outside = ~bars["session"].isin(days)
    if outside.any():
        raise StockDataRefused("non_session", f"a bar on {bars.loc[outside, 'session'].iloc[0]}")
    full = bars.set_index("session").reindex(days)
    full["split_factor"] = full["split_factor"].fillna(1.0)
    full["div_cash"] = full["div_cash"].fillna(0.0)
    return full.rename_axis("session").reset_index()


def development_differences(bars: pd.DataFrame, research: StockLabStore, ticker: str) -> int | None:
    """
    How many development sessions (through 2024-12-31) of `bars` differ from the research
    store's history in close, dividend or split factor; None when the research store has none.
    """

    stored = research.bars(ticker)
    if stored is None:
        return None
    cutoff = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
    fetched = bars[bars["session"] <= cutoff].set_index("session")[list(COMPARED)]
    known = stored.bars.set_index("session")[list(COMPARED)].reindex(fetched.index)
    a, b = fetched.to_numpy(dtype="float64"), known.to_numpy(dtype="float64")
    same = np.isclose(a, b, rtol=1e-9, atol=0.0) | (np.isnan(a) & np.isnan(b))
    return int((~same.all(axis=1)).sum())


def plan(
    protocol: Protocol,
    supported: Sequence[str],
    stored: Sequence[str],
    days: pd.DatetimeIndex,
    research: StockLabStore,
    holdout: HoldoutStore,
    out: TextIO,
    *,
    keychain: bool = False,
) -> None:
    first, last = days[0].date(), days[-1].date()
    print("Stock Lab holdout fetch plan (no request made; pass --authorized to run):", file=out)
    print(
        f"stocks: {', '.join(protocol.tickers)}; sessions {first} to {last} ({len(days)} NYSE "
        "sessions; 2024 inputs to the first windows, then the sealed holdout)",
        file=out,
    )
    print("requests, in order (each at least 2 seconds apart; the key is sent only to", file=out)
    print("api.tiingo.com, in the Authorization header, never in a URL):", file=out)
    requested = [ticker for ticker in supported if ticker not in stored]
    for number, ticker in enumerate(requested, start=1):
        print(f"  {number}. GET {prices_url(ticker, first, last)}", file=out)
    if not requested:
        print("  none", file=out)
    if stored:
        print(f"  already stored, not requested again: {', '.join(stored)}", file=out)
    missing = [t for t in protocol.tickers if t not in supported and t not in stored]
    if missing:
        print(f"  not requested (no supported listing check): {', '.join(missing)}", file=out)
    print(
        "local side effects of --authorized without --store: none (no record is written;", file=out
    )
    print(f"  the listing checks and 2024 history are read from {shown(research.path)})", file=out)
    print("with --store: one batch per ready stock is written to the holdout store", file=out)
    print(f"  {shown(holdout.path)}; a stock that is not ready is held for review", file=out)
    if keychain:
        print(
            f"the key: read from the macOS Keychain (service {TIINGO_SERVICE}, account "
            f"{TIINGO_ACCOUNT}) just before the first request; not read now",
            file=out,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[..., StockAdapter] = StockAdapter,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    holdout_factory: Callable[[], HoldoutStore] = HoldoutStore,
    protocol: Protocol = PROTOCOL,
    calendar: Callable = sessions,
    calendar_id: str = CALENDAR_ID,
    out: TextIO = sys.stdout,
    key_reader: Callable[[], str] = read_key,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--keychain", action="store_true")
    args = parser.parse_args(argv)
    if args.store and not args.authorized:
        print("refused: --store needs --authorized", file=out)
        return 2
    research = store_factory()
    try:
        holdout = holdout_factory()
        days = protocol.fetch_sessions(calendar)
        stored = [batch.ticker for batch in holdout.batches()]
        spent = holdout.evaluation() is not None
    except HoldoutRefused as error:
        print(f"refused: the holdout store ({error})", file=out)
        return 2
    if spent:
        print("refused: the holdout is spent (evaluated once); nothing is fetched", file=out)
        return 2
    checks = research.latest_checks()
    supported = [
        ticker
        for ticker in protocol.tickers
        if ticker in checks and checks[ticker].status is SymbolStatus.SUPPORTED
    ]
    if not args.authorized:
        plan(protocol, supported, stored, days, research, holdout, out, keychain=args.keychain)
        return 0
    requested = [ticker for ticker in supported if ticker not in stored]
    if not requested:
        print("nothing to fetch: every supported protocol stock is stored already", file=out)
        return 0
    first, last = days[0].date(), days[-1].date()
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
    adapter = adapter_factory(authorized=True, max_requests=len(requested), **key_source)
    ready: list[str] = []
    review: list[str] = []
    try:
        for ticker in requested:
            response = adapter.prices(ticker, first, last)
            bars = on_request(response.bars, days)
            report = audit_bars(ticker, bars)
            for line in audit_lines(report):
                print(line, file=out)
            unserved = int(bars["close"].isna().sum())
            if unserved:
                print(f"  {unserved} session(s) not served: held for review", file=out)
            differ = development_differences(bars, research, ticker)
            if differ is None:
                print("  2024 sessions: no research history to compare: held for review", file=out)
            else:
                print(
                    f"  2024 sessions: {differ} differ from the research store in close, "
                    "dividend or split factor",
                    file=out,
                )
            storable = report["ready"] and not unserved and differ == 0
            (ready if storable else review).append(ticker)
            if not args.store:
                continue
            if not storable:
                print("  not stored: held for review first", file=out)
                continue
            batch = holdout.record_batch(
                ticker,
                bars,
                source=SOURCE_ID,
                retrieved_at=response.retrieved_at,
                sessions=calendar,
                calendar_id=calendar_id,
                vintage_note=VINTAGE_NOTE,
            )
            print(f"  stored batch {batch.batch_id}", file=out)
    except (TiingoRequestFailed, StockDataRefused, NotAuthorized, HoldoutRefused) as error:
        print(f"stopped: {error}", file=out)
        return 1
    print(
        f"audit summary: {len(ready)} ready ({', '.join(ready) or 'none'}); "
        f"{len(review)} to review ({', '.join(review) or 'none'})",
        file=out,
    )
    if not args.store:
        print("nothing was written; storing is the explicit step (--store)", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
