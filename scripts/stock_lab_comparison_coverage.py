"""
Simulate the coverage of the Stock Lab briefing's matched-comparison interval
(the block t of briefing.reliability) at a nominal 90%, write the record and
print its summary (local, synthetic series only: no network, no stored data).

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

SCHEMA = "stock_lab_comparison_coverage"
SCHEMA_VERSION = "1.0"


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
    return f"{cell['series']} (target {cell['target']})"


def summary(found: Mapping, sizes: Sequence[int] = (100, 101, 119, 120, 200, 300, 500)) -> str:
    """Coverage by series and size as Markdown: the current rule, the previous in brackets."""

    lines = [
        "| Series | " + " | ".join(f"T = {n}" for n in sizes) + " |",
        "|---|" + "---|" * len(sizes),
    ]
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


def main(argv: Sequence[str] | None = None, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON record")
    args = parser.parse_args(argv)
    found = record(cc.CoverageDesign())
    args.out.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    print(summary(found), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
