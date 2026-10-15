"""
Check the owner's private distribution-anchors file offline (no network).

It loads the file the audit and the collection would load (`--anchors`, else
`STOCK_AGENT_ANCHORS`, else `data/raw/issuer_sources/anchors.csv`) with the
same path checks and validation, and prints, for the research and the
prospective scope, the file's sha256, row counts and per-fund anchor counts.
It never prints an amount and never contacts Tiingo.

Exit codes: 0 valid, or missing (an empty allowlist); 2 refused (the reason
names the row and column, never a value).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/check_anchors.py
    PYTHONPATH=src .venv/bin/python scripts/check_anchors.py --anchors ~/stock-agent-private/anchors.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from stock_agent.data.private_paths import GitRunner, run_git_check_ignore
from stock_agent.data.vendors.anchors import AnchorScope, AnchorsRefused, anchors_path, load_anchors
from stock_agent.data.vendors.tiingo import utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(
    argv: list[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    clock: Callable[[], pd.Timestamp] = utc_now,
    git_runner: GitRunner = run_git_check_ignore,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--anchors",
        help="the private anchors CSV (default: $STOCK_AGENT_ANCHORS, else "
        "data/raw/issuer_sources/anchors.csv)",
    )
    args = parser.parse_args(argv)
    path = anchors_path(args.anchors, project_root=project_root)
    today = clock()
    for scope in (AnchorScope.RESEARCH, AnchorScope.PROSPECTIVE):
        try:
            anchors = load_anchors(
                path, scope, project_root=project_root, today=today, git_runner=git_runner
            )
        except AnchorsRefused as error:
            print(f"Refused: anchors {error}", file=sys.stderr)
            return 2
        print(f"{scope.value} scope")
        for line in anchors.summary_lines():
            print(f"  {line}")
    print("valid: no amount was printed and nothing was contacted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
