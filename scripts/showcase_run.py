"""
Run every registered variant of the exploratory showcase on the stored real
prices and write the private report (owner-run, private, local; no network).

It reads only the showcase store (stock_agent.showcase.store), never a Stock
Lab store, and refuses a universe with a ticker that has no stored bars, or
bars outside 2003-01-02 to 2024-12-31: a missing holding is never silently
dropped. Every variant of configs/showcase/real_rebalancing_v2.yaml runs, in
registered order, at 0x, 1x and 2x the registered costs, and each rebalancing
rule's registered offset family (F7.2) at 1x. Version 2 is version 1 plus that
family; the fetch reads version 1, whose tickers are the same.

It writes, once, into a new folder ~/stock-agent-private/showcase/<run_id>/
(or under STOCK_AGENT_SHOWCASE_REPORTS), outside every Git working tree:

    results.json   every measure, the configuration's sha256, the code's commit
                   and each price batch used (strict JSON)
    report.html    the private report (open it in a browser)
    report.md      the same tables as Markdown
    fonts/         the Lab's bundled fonts and their licence, for the page

Usage:
    .venv/bin/python scripts/showcase_run.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from stock_agent.showcase.config import load_config
from stock_agent.showcase.report import check_wording, render_html, render_markdown
from stock_agent.showcase.returns import window_sessions
from stock_agent.showcase.run import ShowcaseRunRefused, run_showcase
from stock_agent.showcase.store import ShowcaseStoreRefused, open_store, window_bars
from stock_agent.stock_lab.store import StockLabStore

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "showcase" / "real_rebalancing_v2.yaml"
LOGO = REPO / "assets" / "brand" / "portfolio-lab-mark.svg"
FONTS = REPO / "src" / "stock_agent" / "app" / "static" / "fonts"
REPORTS_ENV = "STOCK_AGENT_SHOWCASE_REPORTS"


def reports_root() -> Path:
    configured = os.environ.get(REPORTS_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "stock-agent-private" / "showcase"


def shown(path: Path) -> str:
    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def code_sha(repo: Path = REPO) -> str | None:
    """The checkout's commit, or None when it has uncommitted changes (or is not Git)."""

    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return None if dirty or not re.fullmatch(r"[0-9a-f]{40}", sha) else sha


def logo_markup(path: Path = LOGO) -> str:
    """The Lab's mark as inline SVG at 40 pixels (no XML prolog or comments)."""

    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<\?xml.*?\?>|<!--.*?-->", "", text, flags=re.DOTALL).strip()
    return text.replace('width="1024" height="1024"', 'width="40" height="40" aria-hidden="true"')


def write_once(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], StockLabStore] = open_store,
    root_factory: Callable[[], Path] = reports_root,
    sha_reader: Callable[[], str | None] = code_sha,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.parse_args(argv)
    config = load_config(CONFIG)
    try:
        store = store_factory()
    except ShowcaseStoreRefused as error:
        print(f"refused: the showcase store's location ({error})", file=out)
        return 2
    loaded, refusals = {}, []
    for ticker in config.tickers():
        try:
            loaded[ticker] = window_bars(store, ticker, config.start, config.end)
        except ShowcaseStoreRefused as error:
            refusals.append(str(error))
    if refusals:
        for refusal in refusals:
            print(f"refused: {refusal}", file=out)
        print("nothing was run: every registered holding needs its stored bars", file=out)
        return 2

    root = root_factory().expanduser().resolve()
    if any((folder / ".git").exists() for folder in (root, *root.parents)):
        print("refused: the report folder must be outside every Git working tree", file=out)
        return 2
    generated = now().astimezone(UTC)
    sha = sha_reader()
    try:
        results = run_showcase(
            config,
            loaded.__getitem__,
            sessions=window_sessions(config.start, config.end),
            code_sha=sha,
            generated_at=generated,
        )
    except ShowcaseRunRefused as error:
        print(f"refused: {error}; nothing was written", file=out)
        return 1
    page = render_html(results, logo_svg=logo_markup(), font_dir="fonts")
    markdown = render_markdown(results)
    words = check_wording(page + markdown)
    if words:
        print(f"refused: the report would use ranking words {words}", file=out)
        return 1

    run_id = f"{config.name}-{generated.strftime('%Y%m%dT%H%M%SZ')}"
    folder = root / run_id
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        folder.mkdir(mode=0o700)  # a new folder per run: nothing is overwritten
    except FileExistsError:
        print(f"refused: {shown(folder)} already exists; nothing was overwritten", file=out)
        return 1
    write_once(
        folder / "results.json",
        json.dumps(results, indent=1, sort_keys=True, allow_nan=False) + "\n",
    )
    write_once(folder / "report.html", page)
    write_once(folder / "report.md", markdown)
    (folder / "fonts").mkdir(mode=0o700)
    for name in ("SourceSerif4Variable-Roman.woff2", "SourceSans3VF-Upright.woff2",
                 "SourceSans3VF-Italic.woff2", "OFL.txt"):  # fmt: skip
        shutil.copyfile(FONTS / name, folder / "fonts" / name)
    if sha is None:
        print("note: the checkout has uncommitted changes, so no commit is recorded", file=out)
    stopped = [
        f"{u['key']}/{s['key']}"
        for u in results["universes"]
        for s in u["strategies"]
        if not s["complete"]
    ]
    if stopped:
        print(f"note: stopped early (see the report): {', '.join(stopped)}", file=out)
    print(
        f"variants run: {results['variants_run']} ({results['simulations_run']} simulations)",
        file=out,
    )
    print(f"report: {shown(folder / 'report.html')}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
