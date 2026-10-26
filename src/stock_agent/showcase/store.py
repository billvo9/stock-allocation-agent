"""
The showcase's own private store, kept apart from every Stock Lab store.

It reuses the Stock Lab's write-once SQLite store (`stock_lab.store.
StockLabStore`: listing snapshots, symbol checks and price batches as
vintages, each checked against its row count and content hash when read) at
a separate location, for research only:

    ~/stock-agent-private/showcase_store/showcase.sqlite3
    (or the directory named by STOCK_AGENT_SHOWCASE_HOME)

Isolation rules, checked before the store is opened:

- the file is named `showcase.sqlite3`;
- it is outside every Git working tree (the runner clone and this
  repository included; no data/raw/ exception);
- it is not in or under the Stock Lab research store's directory;
- neither its directory nor any folder above it (below the home folder) holds
  another SQLite store (`*.sqlite3`), so it can never sit beside or inside
  any other store's folder, the Stock Lab's included. This module names no
  other store: the Stock Lab's inference store stays unnamed outside its own
  package (tests/test_stock_lab_inference_boundaries.py).

The showcase never adds to a watchlist, so the Stock Lab's watchlist, the
evening job and the frozen model are never touched.

Bars read for a run must lie inside the registered window: a session after
2024-12-31 (the sealed holdout) refuses the read rather than being dropped.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import store as research_store
from stock_agent.stock_lab.store import StockLabStore, StoredBars

HOME_ENV = "STOCK_AGENT_SHOWCASE_HOME"
DIRECTORY = "showcase_store"
DB_NAME = "showcase.sqlite3"


class ShowcaseStoreRefused(ValueError):
    """A store location or a read that the isolation rules refuse."""


def default_path() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).expanduser() / DB_NAME
    return Path.home() / "stock-agent-private" / DIRECTORY / DB_NAME


def _other_stores(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.sqlite3") if p.name != DB_NAME)


def check_location(path: Path) -> Path:
    """The resolved path, after the isolation rules (module docstring)."""

    resolved = Path(path).expanduser().resolve()
    if resolved.name != DB_NAME:
        raise ShowcaseStoreRefused(f"the showcase store's file is named {DB_NAME}")
    if any((folder / ".git").exists() for folder in resolved.parents):
        raise ShowcaseStoreRefused("the showcase store must be outside every Git working tree")
    directory = resolved.parent
    research = research_store.default_path().expanduser().resolve().parent
    if directory.is_relative_to(research):
        raise ShowcaseStoreRefused(
            "the showcase store may not be in or under a Stock Lab store's directory"
        )
    home = Path.home().resolve()
    for folder in (directory, *directory.parents):
        if folder == home or folder == folder.parent:
            break
        others = _other_stores(folder)
        if others:
            raise ShowcaseStoreRefused(f"{folder.name} holds another store ({', '.join(others)})")
    return resolved


def open_store(path: Path | None = None) -> StockLabStore:
    """The showcase store at `path` (default `default_path()`), after `check_location`."""

    return StockLabStore(check_location(path or default_path()))


def window_bars(store: StockLabStore, ticker: str, start: date, end: date) -> StoredBars:
    """
    The ticker's stored bars, refused when it has none, when a stored session
    lies outside [start, end] (a 2025 session is never silently dropped), or
    when they are not real data.
    """

    stored = store.bars(ticker)
    if stored is None:
        raise ShowcaseStoreRefused(f"{ticker}: no stored bars")
    if stored.data_class != "real":
        raise ShowcaseStoreRefused(f"{ticker}: the showcase reads real bars only")
    sessions = pd.DatetimeIndex(stored.bars["session"])
    first, last = sessions.min().date(), sessions.max().date()
    if first < start or last > end:
        raise ShowcaseStoreRefused(
            f"{ticker}: stored sessions {first} to {last} leave the window {start} to {end}"
        )
    return stored
