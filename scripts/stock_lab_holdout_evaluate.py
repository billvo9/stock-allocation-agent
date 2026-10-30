"""
The one sealed holdout evaluation of the frozen Stock Lab model (owner-run,
once; stock_lab.holdout).

Prepared, not yet run. Without --run it prints the plan and the holdout
store's state (which stocks are fetched, whether the evaluation exists) and
reads no bars (exit 0).

    --run   loads the protocol's frozen model (MODEL_ID) from the research
            store (a registered real model, re-checked on load: its id, the
            feature version and the environment), then computes the record
            inside the holdout store's one evaluation (`evaluate_once`), which
            stores it before anything is printed. The bars are read only
            there. A real run needs a clean checkout whose own stock_agent
            package is imported (`code_state`): its commit is recorded with
            the result. Prints the report.
    --show  prints the stored record's report (read-only; after the run).

Nothing here is written to the repository, and the research store is only
read. A second --run is refused: the holdout is spent after the first.

Exit codes: 0 success; 1 refused during the run; 2 refused before it.

Usage:
    .venv/bin/python scripts/stock_lab_holdout_evaluate.py          # the plan only
    PYTHONPATH=src .venv/bin/python scripts/stock_lab_holdout_evaluate.py --run    # the run
    .venv/bin/python scripts/stock_lab_holdout_evaluate.py --show   # the stored result
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

import stock_agent
from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab import registry
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.frozen_model import RegistryRefused
from stock_agent.stock_lab.holdout import evaluate
from stock_agent.stock_lab.holdout.protocol import PROTOCOL, HoldoutRefused, Protocol
from stock_agent.stock_lab.holdout.report import lines
from stock_agent.stock_lab.holdout.store import HoldoutStore
from stock_agent.stock_lab.store import StockLabStore, utc_now

PROJECT = Path(__file__).resolve().parents[1]


def _git(*arguments: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def code_state() -> dict | None:
    """
    The checked-out commit and whether it is the code that runs: clean when no tracked file
    has uncommitted changes, nothing untracked lies under src/ or scripts/, and the
    stock_agent package is imported from this checkout (not from another one sharing its
    environment). None when Git cannot say.
    """

    sha = (_git("rev-parse", "HEAD") or "").strip()
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts")
    if len(sha) != 40 or tracked is None or untracked is None:
        return None
    imported = Path(stock_agent.__file__).resolve().parents[2] == PROJECT
    return {"sha": sha, "clean": not tracked.strip() and not untracked.strip() and imported}


def shown(path: Path) -> str:
    """A path for printed output, with the home folder as ~ (output may be pasted publicly)."""

    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def plan(protocol: Protocol, holdout: HoldoutStore, out: TextIO) -> None:
    stored = {batch.ticker: batch for batch in holdout.batches()}
    print("Stock Lab sealed holdout evaluation plan (nothing read; pass --run to run):", file=out)
    print(f"  model {protocol.model_id}", file=out)
    print(f"  protocol {protocol.sha256()}", file=out)
    print(
        f"  decisions {protocol.first_decision} to five sessions before {protocol.holdout_end}; "
        f"outcomes {protocol.holdout_start} to {protocol.holdout_end}",
        file=out,
    )
    print(
        f"  deciding: {protocol.deciding_score} on {protocol.deciding_target}, the model against "
        f"{protocol.comparator}, fold-block t over {len(protocol.blocks)} blocks, one-sided at "
        f"{protocol.alpha:.1%}",
        file=out,
    )
    for ticker in protocol.tickers:
        batch = stored.get(ticker)
        state = f"fetched ({batch.batch_id})" if batch else "not fetched yet"
        print(f"  {ticker}: {state}", file=out)
    print(f"  the holdout store: {shown(holdout.path)}", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    holdout_factory: Callable[[], HoldoutStore] = HoldoutStore,
    protocol: Protocol = PROTOCOL,
    calendar: Callable = sessions,
    calendar_id: str = CALENDAR_ID,
    clock: Callable[[], pd.Timestamp] = utc_now,
    code: Callable[[], dict | None] = code_state,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    try:
        holdout = holdout_factory()
        stored = holdout.evaluation()
    except HoldoutRefused as error:
        print(f"refused: the holdout store ({error})", file=out)
        return 2
    if args.show:
        if stored is None:
            print("no evaluation yet: the holdout has not been evaluated", file=out)
            return 0
        print("\n".join(lines(stored)), file=out)
        return 0
    if not args.run:
        plan(protocol, holdout, out)
        if stored is not None:
            print("  evaluated already: the holdout is spent (--show prints it)", file=out)
        return 0
    if stored is not None:
        print("refused: the holdout is spent (evaluated once); --show prints the result", file=out)
        return 2
    model_record = store_factory().frozen_model(protocol.model_id)
    if model_record is None or model_record.get("data_class") != DataClass.REAL.value:
        print(f"refused: no registered real frozen model {protocol.model_id}", file=out)
        return 2
    try:
        model = registry.load(model_record)
    except RegistryRefused as error:
        print(f"refused: the frozen model does not load here ({error})", file=out)
        return 2
    state = code()
    if state is None or not state["clean"]:
        print(
            "refused: the run needs a clean checkout whose own code is imported (its commit "
            "is recorded; run it with PYTHONPATH=src from that checkout)",
            file=out,
        )
        return 2
    created_at = clock()

    def run(histories, batches):
        return evaluate.run(
            histories,
            model,
            protocol=protocol,
            sessions=calendar,
            calendar_id=calendar_id,
            batches=[batch.lineage() for batch in batches],
            created_at=created_at,
            code=state,
        )

    try:
        record = holdout.evaluate_once(run, calendar_id=calendar_id)
    except (HoldoutRefused, RegistryRefused) as error:
        print(f"stopped: {error} (nothing was recorded)", file=out)
        return 1
    print("\n".join(lines(record)), file=out)
    print(f"\nstored as the holdout's one evaluation: {record['record_id']}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
