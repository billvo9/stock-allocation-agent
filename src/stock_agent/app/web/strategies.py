"""
The Lab app's Strategies view model: a small, read-only screen under More for
the exploratory real-data showcase (owner decision of 2026-10-26: "Yes, small
and read-only (Recommended)"). JSON-ready, with display-ready words.

Where it comes from
    Only the newest saved showcase results (`results.json`, written by
    scripts/showcase_run.py into ~/stock-agent-private/showcase/<run_id>/, or
    under STOCK_AGENT_SHOWCASE_REPORTS). Nothing is recomputed: every number
    is one the report already holds, rounded for display only. The file is
    read with the standard library (the app never imports the showcase
    package), strictly: a non-finite number, a file over `MAX_BYTES`, another
    artifact or schema, or a missing field gives an "unreadable" screen with
    one plain sentence, never a partial one.

What it shows
    For each list of holdings: growth of $10,000 for every registered rule on
    one log-scale chart (sampled weekly for drawing), and a table of what
    $10,000 became, the total return, the worst fall, the rebalances, the
    costs paid and, when the report has one, the rebalancing-day range (F7.2).
    The report's own caveat labels come first, unchanged. Rules appear in the
    report's registered order, never ranked.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.app.web import words

SCHEMA = "lab_app_strategies"
VERSION = 1
REPORTS_ENV = "STOCK_AGENT_SHOWCASE_REPORTS"
ARTIFACT = "showcase_results"
MAX_BYTES = 32 * 1024 * 1024
STEP = 5  # chart points: one session in five (weekly), and the last
RUN = re.compile(r"[a-z][a-z0-9_]*-(\d{8}T\d{6}Z)")

TITLE = "Strategies"
LEDE = (
    "What simple rebalancing rules would have done with $10,000 on real prices from 2003 to "
    "2024. An exploratory look back, not a recommendation."
)
NONE_TITLE = "No strategies report yet."
NONE_WORDS = "It appears here after the showcase report is made on this Mac."
UNREADABLE_TITLE = "The strategies report can't be read."
UNREADABLE_WORDS = "No number is shown from it, and nothing was changed."
DEMO_TITLE = "Not in the demo."
DEMO_WORDS = "The Strategies screen shows only your own private report, never made-up numbers."
CHART_NOTE = "Log scale: equal steps are equal percentage changes. Each line is one rule."
DAYS_NOTE = (
    "Days: the gap, in percentage points, between the highest and the lowest total return "
    "across every possible rebalancing day of the rule's cycle. The row's own result is the "
    "registered day; the gap shows how much of it is the luck of the day."
)
COLUMNS = ["Rule", "$10,000 became", "Total return", "Worst fall", "Rebalances", "Costs", "Days"]


def reports_root() -> Path:
    configured = os.environ.get(REPORTS_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "stock-agent-private" / "showcase"


class Unreadable(ValueError):
    """A results file this screen will not show (module docstring)."""


def _refuse_constant(token: str) -> None:
    raise Unreadable(f"a non-finite number ({token})")


def newest_results(root: Path) -> Path | None:
    """The newest run's results.json under `root`, by the run's UTC stamp; None if none."""

    if not root.is_dir():
        return None
    runs = []
    for folder in root.iterdir():
        found = RUN.fullmatch(folder.name)
        if found and folder.is_dir() and (folder / "results.json").is_file():
            runs.append((found[1], folder.name, folder / "results.json"))
    return max(runs)[2] if runs else None


def read_results(path: Path) -> dict:
    """The results record at `path`, read strictly (module docstring)."""

    with path.open("rb") as handle:  # one handle: a file that grows meanwhile is still capped
        body = handle.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise Unreadable("the file is too large")
    try:
        record = json.loads(body.decode("utf-8"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Unreadable(type(error).__name__) from None
    if not isinstance(record, dict) or record.get("artifact") != ARTIFACT:
        raise Unreadable("not a showcase results file")
    version = record.get("schema_version")
    if type(version) is not int or version != 1:
        raise Unreadable("an unknown schema version")
    return record


def _whole_percent(r: float | None) -> str:
    if r is None or not math.isfinite(r):
        return words.DASH
    sign = "+" if r > 0 else words.MINUS if r < 0 else ""
    return f"{sign}{abs(r) * 100:,.0f}%"


def _dollars(value: float | None) -> str:
    return words.DASH if value is None or not math.isfinite(value) else f"${value:,.0f}"


def _fall(drawdown: Mapping) -> str:
    value = drawdown.get("value")
    if value is None:
        return words.DASH
    text = f"{words.MINUS}{value * 100:.0f}%"
    if drawdown.get("measure") != "maximum_drawdown":
        text += " (largest seen; some days missing)"
    return text


def _days(family: Mapping | None) -> str:
    if not family or not family.get("applies"):
        return "Never rebalances" if family else words.DASH
    spread = family["total_return"].get("spread")
    if spread is None:
        return words.DASH
    return f"{spread * 100:,.0f} points across {family['k']} days"


def _short_dollars(value: float) -> str:
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= size:
            return f"${value / size:g}{unit}"
    return f"${value:,.0f}"


def _ticks(values: list[float]) -> list[dict]:
    """Round 1-2-5 dollar levels inside the drawn range, at most five, for the axis."""

    if not values:
        return []
    low, high = min(values), max(values)
    levels = [
        m * 10.0**e
        for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        for m in (1, 2, 5)
        if low <= m * 10.0**e <= high
    ]
    while len(levels) > 5:
        levels = levels[::2]
    return [{"value": level, "words": _short_dollars(level)} for level in levels]


def _chart(universe: Mapping) -> dict:
    dates = list(universe["dates"])
    picks = list(range(0, len(dates), STEP))
    if picks and picks[-1] != len(dates) - 1:
        picks.append(len(dates) - 1)
    series = []
    for strategy in universe["strategies"]:
        values = strategy["wealth"]
        if len(values) != len(dates):
            raise Unreadable("a wealth path does not match its dates")
        series.append(
            {
                "label": strategy["label"],
                "rule": strategy["rule"],
                "every": strategy["every"],
                "values": [None if values[i] is None else float(values[i]) for i in picks],
                "words": [_dollars(values[i]) for i in picks],
            }
        )
    return {
        "label": f"Growth of $10,000 for each rule, {universe['label']}, log scale",
        "days": [words.short_day(dates[i]).split(", ", 1)[1] + f", {dates[i][:4]}" for i in picks],
        "series": series,
        "ticks": _ticks([v for s in series for v in s["values"] if v is not None and v > 0]),
        "note": CHART_NOTE,
    }


def _rows(universe: Mapping) -> list[dict]:
    rows = []
    for s in universe["strategies"]:
        stopped = None if s["complete"] else f"Stopped early ({s['status_reason']})"
        rows.append(
            {
                "label": s["label"],
                "rule": s["rule"],
                "every": s["every"],
                "description": s["description"],
                "cells": [
                    _dollars(s["final_wealth"]),
                    _whole_percent(s["total_return"]),
                    _fall(s["drawdown"]),
                    f"{s['turnover']['rebalances']:,}",
                    _dollars(s["costs"]["paid_dollars"]),
                    _days(s.get("offset_family")),
                ],
                "stopped": stopped,
            }
        )
    return rows


def _universe(universe: Mapping) -> dict:
    members = ", ".join(m["ticker"] for m in universe["members"])
    return {
        "key": universe["key"],
        "title": universe["label"],
        "members": members,
        "selection": universe["selection"],
        "span": f"From {universe['build']} to {universe['dates'][-1]}, after trading costs.",
        "chart": _chart(universe),
        "columns": COLUMNS,
        "rows": _rows(universe),
    }


def _base(now: datetime) -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "generated_at": pd.Timestamp(now).tz_convert("UTC").isoformat(),
        "title": TITLE,
        "lede": LEDE,
        "problem": None,
        "labels": [],
        "source": None,
        "universes": [],
        "days_note": DAYS_NOTE,
    }


def strategies_view(results: Mapping, *, now: datetime) -> dict:
    """The screen for one results record (raises `Unreadable` for a record it can't show)."""

    try:
        universes = [_universe(u) for u in results["universes"]]
        labels = [str(label) for label in results["labels"]]
        commit = results.get("code_sha")
        source = (
            f"Report made {str(results['generated_at'])[:10]} from configuration "
            f"{results['config']['name']}"
            + (f", code {commit[:7]}" if commit else "")
            + f". {results['variants_run']} variants, none chosen afterwards."
        )
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as error:
        raise Unreadable(f"a missing or malformed field ({type(error).__name__})") from None
    return {**_base(now), "labels": labels, "source": source, "universes": universes}


def build_strategies(root: Path | None, *, now: datetime) -> dict:
    """The Strategies screen from the newest results under `root` (None: the demo, no data)."""

    if root is None:
        return {**_base(now), "problem": {"title": DEMO_TITLE, "words": DEMO_WORDS}}
    try:
        path = newest_results(root)
        if path is None:
            return {**_base(now), "problem": {"title": NONE_TITLE, "words": NONE_WORDS}}
        return strategies_view(read_results(path), now=now)
    except (Unreadable, OSError) as error:  # OSError: its type only, never its path
        detail = error if isinstance(error, Unreadable) else type(error).__name__
        problem = {
            "title": UNREADABLE_TITLE,
            "words": UNREADABLE_WORDS,
            "technical": words.TECHNICAL.format(detail=detail),
        }
        return {**_base(now), "problem": problem}
