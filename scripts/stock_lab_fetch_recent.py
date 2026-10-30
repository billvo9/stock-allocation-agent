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

The key. By default it comes from TIINGO_API_KEY. With --keychain it is read
instead from the macOS Keychain (stock_agent.data.vendors.keychain: service
stock-agent-tiingo, account stock-lab), once, just before the first request,
and handed to the adapter in memory only, so an unattended run (the evening
agent printed by scripts/stock_lab_schedule.py) needs no variable holding it.
An item that cannot be read refuses the run before any request (exit 2),
naming the reason, never the value.

Unattended (--unattended, the evening agent's run, which may start the moment
the Mac wakes; no wait or retry starts that would end at or after the open of
the session after the window, when the evening can no longer be recorded):
    power    on battery, a warning and one local notification
             (stock_lab.notify, "on_battery"): run.sh's caffeinate -s holds
             the Mac awake on AC power only. The run carries on.
    network  before the key is read and before the first request, a wait of
             at most 5 minutes until api.tiingo.com resolves and accepts a TCP
             connection (stock_lab.tiingo.reach: no request, nothing sent, no
             key). If it never does, exit 4 with nothing requested.
    retries  a transport error (no response: the network dropped or timed
             out) is retried up to TRANSPORT_RETRIES times in the whole run,
             after 10, 30 and 90 seconds. Each retry is a request of the run's
             budget, so the evening makes at most MAX_TICKERS +
             TRANSPORT_RETRIES requests, under the key's HOURLY_REQUESTS. An
             HTTP refusal (4xx, 429 included), a server error, a refused
             response or an audit hold is never retried.

Exit codes: 0 success; 1 a refused request, response or write; 2 refused
before any request; 4 (--unattended) the network never came up, nothing
requested.

Usage:
    .venv/bin/python scripts/stock_lab_fetch_recent.py                        # the plan only
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized           # read-only run
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized --store   # store (approval)
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized --store --keychain
    .venv/bin/python scripts/stock_lab_fetch_recent.py --authorized --store --keychain --unattended
The tickers are always the saved watchlist (add or remove them in the Lab).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.data.vendors.keychain import (
    TIINGO_ACCOUNT,
    TIINGO_SERVICE,
    KeychainRefused,
    add_command,
    read_key,
)
from stock_agent.data.vendors.tiingo import TiingoFailure, TiingoRequestFailed
from stock_agent.stock_lab.audit import audit_bars, audit_lines
from stock_agent.stock_lab.contracts import DataClass, SymbolStatus
from stock_agent.stock_lab.forecast import session_open
from stock_agent.stock_lab.inference.schema import (
    INFERENCE_SESSIONS,
    PROTECTED_START,
    InferenceRefused,
    Refusal,
    close_available_at,
)
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.inference.window import load_inference_window
from stock_agent.stock_lab.notify import NOTICES, notify
from stock_agent.stock_lab.power import BATTERY
from stock_agent.stock_lab.power import source as power_source
from stock_agent.stock_lab.store import StockLabStore, utc_now
from stock_agent.stock_lab.symbols import parse_ticker, refused
from stock_agent.stock_lab.tiingo import (
    NETWORK_WAIT,
    PRICES_HOST,
    SOURCE_ID,
    VINTAGE_NOTE,
    NotAuthorized,
    PriceResponse,
    StockAdapter,
    StockDataRefused,
    prices_url,
    reach,
    wait_for_network,
)

MAX_TICKERS = 30  # the authorized scope (A1); never raised by an option
TRANSPORT_RETRIES = 3  # --unattended: extra requests in a run, after transport errors only
RETRY_DELAYS = (10.0, 30.0, 90.0)  # seconds before each retry
HOURLY_REQUESTS = 50  # the key's plan: the evening's at most MAX_TICKERS + TRANSPORT_RETRIES
NO_NETWORK = 4  # exit status: the network never came up within the wait; nothing requested


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


def recording_ends(last: pd.Timestamp) -> pd.Timestamp:
    """The 09:30 New York open of the session after `last`: no evening of `last` after it."""

    following = sessions(last + pd.Timedelta(days=1), last + pd.Timedelta(days=10))[0]
    return pd.Timestamp(session_open(following.strftime("%Y-%m-%d"))).tz_convert("UTC")


class Retries:
    """A run's transport-error retries: at most `allowed`, none ending at or after `until`."""

    def __init__(
        self,
        allowed: int,
        until: pd.Timestamp | None,
        clock: Callable[[], pd.Timestamp],
        sleep: Callable[[float], None],
    ) -> None:
        self.allowed, self.until, self.clock, self.sleep = allowed, until, clock, sleep
        self.used = 0

    def delay(self, error: TiingoRequestFailed) -> float | None:
        """The wait before retrying after `error`, or None when it is not retried."""

        if error.reason is not TiingoFailure.TRANSPORT_ERROR or self.used >= self.allowed:
            return None
        delay = RETRY_DELAYS[self.used]
        if self.until is None or self.clock() + pd.Timedelta(seconds=delay) >= self.until:
            return None
        self.used += 1
        return delay


def fetch_prices(
    adapter: StockAdapter, ticker: str, days: pd.DatetimeIndex, retries: Retries, out: TextIO
) -> PriceResponse:
    """The ticker's window from the adapter, retried after a transport error (Retries)."""

    while True:
        try:
            return adapter.prices(ticker, days[0].date(), days[-1].date())
        except TiingoRequestFailed as error:
            delay = retries.delay(error)
            if delay is None:
                raise
            print(
                f"  {ticker}: {error}; retry {retries.used} of {retries.allowed} in {delay:g} s",
                file=out,
            )
            retries.sleep(delay)


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
    *,
    keychain: bool = False,
    unattended: bool = False,
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
    if keychain:
        print(
            f"the key: read from the macOS Keychain (service {TIINGO_SERVICE}, account "
            f"{TIINGO_ACCOUNT}) just before the first request; not read now",
            file=out,
        )
    if unattended:
        print(
            f"unattended: waits up to {NETWORK_WAIT / 60:g} minutes for {PRICES_HOST} before the "
            "first request (a DNS lookup and a TCP connection: no request, no key), retries a "
            f"transport error up to {TRANSPORT_RETRIES} times (at most "
            f"{len(supported) + TRANSPORT_RETRIES} requests), none ending after the "
            f"{recording_ends(days[-1]).tz_convert('America/New_York'):%Y-%m-%d %H:%M} New York "
            "open, and warns on battery; nothing checked now",
            file=out,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    adapter_factory: Callable[..., StockAdapter] = StockAdapter,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    inference_factory: Callable[[], InferenceStore] = InferenceStore,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
    key_reader: Callable[[], str] = read_key,
    network: Callable[[], str | None] = reach,
    sleep: Callable[[float], None] = time.sleep,
    power: Callable[[], str] = power_source,
    notifier: Callable[[str], object] = notify,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--keychain", action="store_true")
    parser.add_argument("--unattended", action="store_true")
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
        plan(tickers, supported, days, research, inference, out, keychain=args.keychain,
             unattended=args.unattended)  # fmt: skip
        return 0
    if args.unattended and power() == BATTERY:
        print(f"warning: {NOTICES['on_battery']}", file=out)
        notifier("on_battery")
    if not supported:
        print("nothing to fetch: no watched ticker has a supported listing check yet", file=out)
        return 0

    last = days[-1].date()
    retries = Retries(0, None, clock, sleep)  # by hand: no wait and no retry
    if args.unattended:
        retries = Retries(TRANSPORT_RETRIES, recording_ends(days[-1]), clock, sleep)
        waited = wait_for_network(network, clock=clock, sleep=sleep, until=retries.until)
        if not waited.reachable:
            print(
                f"stopped: {PRICES_HOST} was not reachable after {waited.waited:g} s of waiting "
                f"({waited.checks} check{'s' * (waited.checks > 1)}; last {waited.reason}); "
                "nothing was requested",
                file=out,
            )
            return NO_NETWORK
        if waited.checks > 1:
            print(
                f"network: {PRICES_HOST} reachable after {waited.waited:g} s ({waited.checks} "
                f"checks; first {waited.first})",
                file=out,
            )
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
    budget = len(supported) + retries.allowed
    adapter = adapter_factory(authorized=True, max_requests=budget, **key_source)
    ready: list[str] = []
    review: list[str] = []
    try:
        for ticker in supported:
            response = fetch_prices(adapter, ticker, days, retries, out)
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
