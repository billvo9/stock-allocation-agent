"""
Simulate the coverage of the Stock Lab briefing's matched-comparison interval
(the block t of briefing.reliability) at a nominal 90%, per score, and how often
its persistence flag fires; write the record and print its summary (local,
synthetic series only: no network, no stored data).

The design is fixed in stock_lab.comparison_coverage. The interval is a
reporting convention, never a decision test: whether a challenger beats B1 is
decided on development data only (A7).

Usage:
    .venv/bin/python scripts/stock_lab_comparison_coverage.py \
        --out docs/research/stock_lab_comparison_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from stock_agent.stock_lab import comparison_coverage as cc
from stock_agent.stock_lab.briefing.reliability import PERSISTENCE_THRESHOLD

SCHEMA = "stock_lab_comparison_coverage"
SCHEMA_VERSION = "1.1"  # 1.1 adds Brier and log loss, the break series and the flag rates


def record(design: cc.CoverageDesign) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "data_class": "synthetic",
        "use": "coverage of a reporting interval; never a decision test",
        **cc.simulate(design),
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _label(cell: Mapping) -> str:
    if "phi" in cell:
        return f"{cell['series']} φ = {cell['phi']}"
    if "delta" in cell:
        return f"{cell['series']} δ = {cell['delta']}"
    return f"{cell['series']} {cell['score']} (target {cell['target']})"


def _head(sizes: Sequence[int]) -> list[str]:
    return [
        "| Series | " + " | ".join(f"T = {n}" for n in sizes) + " |",
        "|---|" + "---|" * len(sizes),
    ]


def summary(found: Mapping, sizes: Sequence[int] = (100, 101, 119, 120, 200, 300, 500)) -> str:
    """Coverage by series and size as Markdown: the current rule, the previous in brackets."""

    lines = _head(sizes)
    for cell in found["cells"]:
        by_size = {row["dates"]: row for row in cell["sizes"]}
        row = []
        for n in sizes:
            current, previous = by_size[n]["current"], by_size[n]["previous"]
            text = _pct(current["coverage"])
            if abs(current["coverage"] - previous["coverage"]) >= 0.0005:
                text += f" ({_pct(previous['coverage'])})"
            row.append(text)
        lines.append(f"| {_label(cell)} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def flag_summary(found: Mapping, sizes: Sequence[int] = (120, 200, 300, 500)) -> str:
    """How often SE40 / SE20 exceeds the adopted threshold (1.2 in brackets), as Markdown."""

    lines = [f"Share of records with SE40 / SE20 above {PERSISTENCE_THRESHOLD} (above 1.2):", ""]
    lines += _head(sizes)
    for cell in found["cells"]:
        by_size = {row["dates"]: row["persistence"]["flag_rate"] for row in cell["sizes"]
                   if row["persistence"] is not None}  # fmt: skip
        row = [
            f"{_pct(by_size[n][str(PERSISTENCE_THRESHOLD)])} ({_pct(by_size[n]['1.2'])})"
            for n in sizes
        ]
        lines.append(f"| {_label(cell)} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON record")
    args = parser.parse_args(argv)
    found = record(cc.CoverageDesign())
    args.out.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    print(summary(found), file=stdout)
    print(file=stdout)
    print(flag_summary(found), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
