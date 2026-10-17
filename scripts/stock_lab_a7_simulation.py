"""
Owner decision A7: simulate the candidate decision tests' size and power under
the Stock Lab's planned conditions, apply the pre-registered criterion, and
write the record (local, synthetic panels only: no network, no stored data).

The design and the criterion are fixed in stock_lab.planned_simulation and
were pre-registered in docs/research/stock_lab_a7_decision.md before the
first run. Nothing here changes the evaluation's rule: B1 stays the default.

`--set FIELD=VALUE ...` makes a sensitivity run instead (for example
constant volatility, `--set arch=0.0 garch=0.0`): it is labelled as such and
reported only; the pre-registered choice is the default run's.

A tuple field takes comma-separated numbers (`--set deltas=0.00025,0.0005`).

`--evaluation-check` instead runs the evaluation's own recorded size check
(size_simulation: the neighbour placebos, iid returns, the three reported
tests, 1,000 panels) at the planned shapes: each universe size, 19 test years.
Reported only; `--set nominal=0.025` changes its level.

Usage:
    .venv/bin/python scripts/stock_lab_a7_simulation.py \
        --out docs/research/stock_lab_a7_simulation.json --jobs 4
    .venv/bin/python scripts/stock_lab_a7_simulation.py --set arch=0.0 garch=0.0 \
        --out docs/research/stock_lab_a7_sensitivity.json --jobs 4
    .venv/bin/python scripts/stock_lab_a7_simulation.py --evaluation-check \
        --out docs/research/stock_lab_a7_evaluation_check.json --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from pathlib import Path
from typing import TextIO

from stock_agent.stock_lab import decision_tests as dt
from stock_agent.stock_lab import evaluation, size_simulation
from stock_agent.stock_lab import planned_simulation as ps

SCHEMA = "stock_lab_a7_simulation"
SCHEMA_VERSION = "1.0"


def _run(args: tuple[int, Mapping]) -> dict:
    names, overrides = args
    return ps.simulate_planned(ps.planned_design(names, **overrides))


def _evaluation_check(args: tuple[int, Mapping]) -> dict:
    names, changed = args
    config = evaluation.EvaluationConfig()
    design = evaluation.size_design(config, names=names, years=ps.PlannedDesign.years)
    return size_simulation.simulate_size(replace(design, **changed))


def _parse_sets(parser: argparse.ArgumentParser, items: Sequence[str], allowed: set[str]) -> dict:
    """FIELD=VALUE pairs: a number, or comma-separated numbers for a tuple field."""

    changed = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in allowed:
            parser.error(f"--set: {name!r} cannot be changed here")
        numbers = [float(v) if "." in v or "e" in v else int(v) for v in value.split(",")]
        changed[name] = tuple(float(v) for v in numbers) if "," in value else numbers[0]
    return changed


def evaluation_table(sizes: Sequence[Mapping]) -> str:
    """The evaluation's size check at each universe size, as Markdown (range over half-lives)."""

    names = [size["design"]["names"] for size in sizes]
    lines = [
        "| Placebo | Test | " + " | ".join(f"N = {n}" for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for placebo in size_simulation.PLACEBOS:
        for method in size_simulation.METHODS:
            row = []
            for size in sizes:
                rates = [
                    r["rate"]
                    for r in size["rates"]
                    if r["placebo"] == placebo and r["method"] == method
                ]
                row.append(f"{_pct(min(rates))}–{_pct(max(rates))}")
            lines.append(f"| {placebo} | `{method}` | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def tables(record: Mapping) -> str:
    """The packet's summary tables, as Markdown."""

    sizes = [size["design"]["names"] for size in record["sizes"]]
    cells = [cell for size in record["sizes"] for cell in size["cells"]]
    lines = []

    def by(name: str, names: int) -> list[Mapping]:
        return [c for c in cells if c["procedure"] == name and c["names"] == names]

    header = "| Procedure | " + " | ".join(f"N = {n}" for n in sizes) + " |"
    rule = "|---|" + "---|" * len(sizes)
    for null, title in (
        ("size_own", "Size, own-history null (range over the half-lives)"),
        ("size_symmetric", "Size, symmetric null (range over the half-lives)"),
    ):
        lines += [f"**{title}**", "", header, rule]
        for name in dt.PROCEDURES:
            row = []
            for n in sizes:
                rates = [c[null]["rate"] for c in by(name, n)]
                row.append(f"{_pct(min(rates))}–{_pct(max(rates))}")
            lines.append(f"| `{name}` | " + " | ".join(row) + " |")
        lines.append("")
    for delta in record["deltas"]:
        lines += [f"**Power at δ = {delta} (mean over the half-lives)**", "", header, rule]
        for name in dt.PROCEDURES:
            row = []
            for n in sizes:
                rates = [
                    p["rate"] for c in by(name, n) for p in c["power_own"] if p["delta"] == delta
                ]
                row.append(_pct(sum(rates) / len(rates)))
            lines.append(f"| `{name}` | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["**Family: Holm false selection (range over the half-lives)**", "", header, rule]
    for name in dt.PROCEDURES:
        row = []
        for n in sizes:
            rates = [
                f["replaced"]["rate"] for c in by(name, n) for f in c["family"] if f["true"] is None
            ]
            row.append(f"{_pct(min(rates))}–{_pct(max(rates))}")
        lines.append(f"| `{name}` | " + " | ".join(row) + " |")
    lines.append("")
    choice = record["choice"]
    pre_registered = record.get("role") == "pre-registered"
    lines += [
        "**Pre-registered choice**"
        if pre_registered
        else "**The criterion applied to this sensitivity run (reported only)**",
        "",
        "| Procedure | Distortion | Mean power |",
        "|---|---|---|",
    ]
    for name in dt.PROCEDURES:
        mark = " (chosen)" if name == choice["chosen"] else ""
        lines.append(
            f"| `{name}`{mark} | {100 * choice['distortion'][name]:.1f} pp | "
            f"{_pct(choice['mean_power'][name])} |"
        )
    lines += ["", f"Chosen: `{choice['chosen']}` ({choice['reason']})."]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO = sys.stdout,
    overrides: Mapping | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON record")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(ps.UNIVERSE_SIZES))
    parser.add_argument("--reps", type=int, default=ps.REPS)
    parser.add_argument("--jobs", type=int, default=1, help="universe sizes simulated at once")
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "a sensitivity run: numeric design fields changed, a tuple field as comma-separated "
            "numbers (reported only, never chosen on); with --evaluation-check only nominal"
        ),
    )
    parser.add_argument(
        "--evaluation-check",
        action="store_true",
        help="the evaluation's own size check at the planned shapes (reported only)",
    )
    args = parser.parse_args(argv)
    if args.evaluation_check:
        changed = _parse_sets(parser, args.set, {"nominal"})
        work = [(names, changed) for names in args.sizes]
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                checks = list(pool.map(_evaluation_check, work))
        else:
            checks = [_evaluation_check(item) for item in work]
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "data_class": "synthetic",
            "role": "the evaluation's size check at the planned shapes (reported only)",
            "changed": changed,
            "sizes": checks,
        }
        args.out.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        print(evaluation_table(checks), file=out)
        return 0
    numeric = {
        f.name for f in fields(ps.PlannedDesign) if f.type in ("float", "int", "tuple[float, ...]")
    }
    changed = _parse_sets(parser, args.set, numeric - {"names", "seed"})
    pre_registered = not changed and not overrides
    pre_registered &= args.sizes == list(ps.UNIVERSE_SIZES) and args.reps == ps.REPS
    settings = {"reps": args.reps, **changed, **(overrides or {})}
    work = [(names, settings) for names in args.sizes]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            sizes = list(pool.map(_run, work))
    else:
        sizes = [_run(item) for item in work]
    cells = [cell for size in sizes for cell in size["cells"]]
    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "data_class": "synthetic",
        "use": "owner decision A7; B1 stays the default; no real data",
        "role": "pre-registered" if pre_registered else "sensitivity (reported only)",
        "changed": changed,
        "procedures": list(dt.PROCEDURES),
        "deltas": list(sizes[0]["design"]["deltas"]),
        "sizes": sizes,
        "choice": ps.choose_procedure(cells),
    }
    args.out.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(tables(record), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
