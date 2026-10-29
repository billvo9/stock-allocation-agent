"""
Backfill the 3-month Treasury rate (FRED DGS3MO) with every ALFRED vintage published
through 2024-12-31, into the private vintage-aware rates store (owner-run, local).

The owner authorized it on 2026-10-26 with group C ("Approve all C1–C9 (Recommended)",
which included a one-time FRED download of 2003-2024 T-bill history; C2's backfill
range). Without --authorized it makes no request and prints the plan (exit 0).

Scope (one run):
    FRED's series/observations for DGS3MO: observations dated 2002-12-02 to 2024-12-30,
    real-time periods 1776-07-04 to 2024-12-31 (every vintage published through the end
    of 2024; output_type 1), one request per 100,000 rows (about 6,000 to 10,000 rows are
    expected, so one request). The value for 2024-12-31 is posted in 2025 and is not
    requested; no vintage published after 2024-12-31 is requested or kept (the holdout),
    and a served row from 2025 refuses the whole response.
    FRED's series/vintagedates for DGS3MO, real-time 1776-07-04 to 2024-12-31 (C2 asks for
    it separately; it cannot be combined with the observations' real-time bounds), one
    request: a cross-check that every listed vintage date is seen in the observations. A
    listed date after 2024-12-31 refuses the run.

The key is never printed. By default it comes from FRED_API_KEY; with --keychain it is
read from the macOS Keychain (service stock-agent-fred, account rates) just before
the request and kept in memory only. FRED's version 1 API takes the key as a query
parameter, which the existing client keeps out of every message and traceback.

Read-only by default: --authorized downloads, checks the vintages
(stock_agent.data.rates.vintages) and prints their diagnostics, writing nothing. With
--store it also saves them as a new file in ~/stock-agent-private/rates/ (or
STOCK_AGENT_RATES_HOME), outside every Git working tree, append-only.

Exit codes: 0 success; 1 a refused request or response; 2 refused before any request.

Usage:
    .venv/bin/python scripts/rates_backfill.py                                  # the plan only
    .venv/bin/python scripts/rates_backfill.py --authorized --keychain          # check, no write
    .venv/bin/python scripts/rates_backfill.py --authorized --store --keychain  # check and store
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.config import load_risk_free_rate_config
from stock_agent.data.fred_source import FredDataSource
from stock_agent.data.rates.vintages import (
    LAST_VINTAGE,
    VintageRefused,
    default_home,
    diagnostics,
    save_vintages,
    vintages_from_alfred,
)
from stock_agent.data.vendors.keychain import KeychainRefused, add_command, read_key

CONFIG = Path(__file__).resolve().parents[1] / "config" / "rates.yaml"
SERIES = "DGS3MO"
OBSERVATION_START = "2002-12-02"
OBSERVATION_END = "2024-12-30"
REALTIME_START = "1776-07-04"  # FRED's "every vintage" start
REALTIME_END = LAST_VINTAGE.isoformat()
FRED_SERVICE = "stock-agent-fred"
FRED_ACCOUNT = "rates"
REQUEST = (
    f"GET https://api.stlouisfed.org/fred/series/observations?series_id={SERIES}"
    f"&observation_start={OBSERVATION_START}&observation_end={OBSERVATION_END}"
    f"&realtime_start={REALTIME_START}&realtime_end={REALTIME_END}"
    "&output_type=1&file_type=json&limit=100000&sort_order=asc (the key is added, never shown)"
)
VINTAGE_REQUEST = (
    f"GET https://api.stlouisfed.org/fred/series/vintagedates?series_id={SERIES}"
    f"&realtime_start={REALTIME_START}&realtime_end={REALTIME_END}&file_type=json"
    "&limit=10000&sort_order=asc (the key is added, never shown)"
)


def shown(path: Path) -> str:
    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def plan(out: TextIO, keychain: bool, home: Path) -> None:
    print("DGS3MO vintage backfill plan (no request made; pass --authorized to run):", file=out)
    print(f"  1. {REQUEST}", file=out)
    print(f"  2. {VINTAGE_REQUEST}", file=out)
    print("  one request each per page (100,000 rows; 10,000 dates); two are expected", file=out)
    source = f"the macOS Keychain ({FRED_SERVICE})" if keychain else "FRED_API_KEY"
    print(f"the key: read from {source}, never printed", file=out)
    print("nothing after 2024-12-31 is requested or kept (the holdout)", file=out)
    print("--authorized without --store writes nothing", file=out)
    print(f"--store saves a new file in {shown(home)}", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    source_factory: Callable[[str], object] = lambda key: FredDataSource(api_key=key),
    key_reader: Callable[[], str] = lambda: read_key(FRED_SERVICE, FRED_ACCOUNT),
    clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    home_factory: Callable[[], Path] = default_home,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--keychain", action="store_true")
    args = parser.parse_args(argv)
    home = home_factory()
    if args.store and not args.authorized:
        print("refused: --store needs --authorized", file=out)
        return 2
    if not args.authorized:
        plan(out, args.keychain, home)
        return 0

    if args.keychain:
        try:
            key = key_reader()
        except KeychainRefused as error:
            print(
                f"refused: the FRED key could not be read from the macOS Keychain ({error}); "
                "nothing was requested. To add it, type it at the prompt of: "
                f"{add_command(FRED_SERVICE, FRED_ACCOUNT)}",
                file=out,
            )
            return 2
    else:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            print("refused: FRED_API_KEY is not set; nothing was requested", file=out)
            return 2

    spec = load_risk_free_rate_config(CONFIG).spec  # the rate_id its callers already use
    if spec.provider_series_id != SERIES:
        print(f"refused: the configured risk-free series is not {SERIES}", file=out)
        return 2
    try:
        source = source_factory(key)
        observations = source.get_series_observations(
            series_id=SERIES,
            observation_start=OBSERVATION_START,
            observation_end=OBSERVATION_END,
            realtime_start=REALTIME_START,
            realtime_end=REALTIME_END,
        )
        dates = pd.to_datetime(observations["date"])
        if dates.max() > pd.Timestamp(OBSERVATION_END) or dates.min() < pd.Timestamp(
            OBSERVATION_START
        ):
            raise VintageRefused("an observation outside the requested dates")
        listed = source.get_series_vintage_dates(
            series_id=SERIES, realtime_start=REALTIME_START, realtime_end=REALTIME_END
        )
        if any(pd.Timestamp(day).date() > LAST_VINTAGE for day in listed):
            raise VintageRefused("a listed vintage date after 2024-12-31 (the holdout)")
        frame = vintages_from_alfred(
            observations,
            rate_id=spec.rate_id,
            provider_series_id=SERIES,
            currency=spec.currency,
            tenor=spec.tenor,
            quote_convention=spec.quote_convention,
            retrieved_at=clock(),
        )
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        # A request failure carries only the series and the status (the client's rule).
        print(f"stopped: {type(error).__name__}: {error}", file=out)
        return 1
    except Exception as error:  # noqa: BLE001 - anything else: its type only, never its text
        print(f"stopped: {type(error).__name__}", file=out)
        return 1
    stats = diagnostics(frame)
    seen = set(frame["vintage_date"].dt.strftime("%Y-%m-%d"))
    stats["listed_vintage_dates"] = len(listed)
    stats["listed_not_seen"] = len(set(listed) - seen)  # a cross-check, reported, not corrected
    print("DGS3MO vintages (first releases book wealth; revisions are kept as new rows):", file=out)
    for name, value in stats.items():
        print(f"  {name}: {value}", file=out)
    if not args.store:
        print("nothing was written; storing is a separate, explicit step (--store)", file=out)
        return 0
    try:
        path = save_vintages(frame, spec.rate_id, home)
    except (VintageRefused, OSError) as error:
        print(f"stopped: not stored ({type(error).__name__}: {error})", file=out)
        return 1
    if path is None:
        print("nothing new to store: every vintage is already in the store", file=out)
        return 0
    print(f"stored: {shown(path)}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
