"""
The zero-volatility floor check for the prospective ETF forecasters (docs/research/zero_vol_floor.md).

Subcommands, run from the repository root (nothing here contacts a network,
freezes, registers or records a forecast):

    examples       the worked examples of the floor (no data)
        .venv/bin/python scripts/check_zero_vol_floor.py examples
    registration   the registration schedule, test levels and minimum
                   detectable effects (no data; docs/research/prospective_registration.md)
        .venv/bin/python scripts/check_zero_vol_floor.py registration
    check          the development-data check on the stored F1 research batch
        PYTHONPATH=src .venv/bin/python scripts/check_zero_vol_floor.py check \
            --research-store ~/stock-agent-private/fund_store

`check` reads the F1 research store through its fail-closed reader only, so a
prospective store (or anything else that is not F1 batches) is refused, and
it refuses any price, distribution or split dated after 2024-12-31 before
computing anything. It develops the forecasters under the rule in force and
under the research-only option B2 (never freezing either), evaluates the
pre-stated criteria Z1-Z4 in order and stops at the first failure. It writes
report.json and report.md once, under a new timestamped directory of
--output-dir (default reports/ml/prospective/zero_vol_check, ignored by Git).

Exit codes: 0 every criterion passed; 1 a criterion failed (the run stopped
there: report it, never switch rules); 2 the inputs were refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.reader import read_store
from stock_agent.data.funds.schema import FundDataRefused
from stock_agent.forecasting.prospective import zero_vol, zero_vol_check
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.forecasting.provenance import code_provenance, environment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "prospective" / "etf_prospective_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "ml" / "prospective" / "zero_vol_check"
SUMMARY_TABLES = ("window_totals", "coverage")


def _now() -> datetime:
    return datetime.now(UTC)


def _refused(message: str) -> int:
    print(f"Refused: {message}", file=sys.stderr)
    return 2


def _check(args, clock: Callable[[], datetime]) -> int:
    config = load_config(args.config)
    root = Path(args.research_store).expanduser()
    try:
        store = read_store(root)
    except FundDataRefused as error:
        return _refused(
            f"{root} is not an F1 research store ({error}); a prospective store, or any "
            "directory that is not F1 batches, is refused"
        )
    started = clock()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    directory = Path(args.output_dir).expanduser() / stamp
    if directory.exists():
        return _refused(f"{directory} already exists; a report is written once")
    try:
        report = zero_vol_check.run_check(
            store.prices, store.distributions, store.splits, store.batches, config
        )
    except zero_vol_check.CheckRefused as error:
        return _refused(str(error))
    context = {
        "run_at": started.isoformat(),
        "research_store": str(root),
        "config": config.experiment,
        "config_sha256": config.sha256,
        "training_end": str(config.training_end.date()),
        "code": code_provenance(PROJECT_ROOT),
        "environment": environment(),
    }
    directory.mkdir(parents=True)
    document = zero_vol_check.report_document(report, context)
    (directory / "report.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "report.md").write_text(
        zero_vol_check.render_report(report, {"run_at": context["run_at"]}), encoding="utf-8"
    )
    print(f"zero-volatility check: {report.status.upper()} ({report.data_class})")
    for label in report.labels:
        print(f"- {label}")
    criteria = pd.DataFrame(report.criteria).loc[:, ["id", "status", "measured"]]
    print(zero_vol_check.markdown_table(criteria))
    for name in SUMMARY_TABLES:
        table = report.tables[name]
        if "target" in table:
            table = table.loc[table["target"] == zero_vol_check.VOL]
        print(f"\n{name}\n{zero_vol_check.markdown_table(table)}")
    print(f"\nreport: {directory / 'report.json'} and report.md")
    if report.status != zero_vol_check.PASS:
        print(
            "A criterion failed and the run stopped there. Report it to the owner; the rule "
            f"in force stays {zero_vol.CURRENT} and is never switched in response."
        )
        return 1
    return 0


def main(argv: list[str] | None = None, *, clock: Callable[[], datetime] = _now) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("examples")
    commands.add_parser("registration")
    checking = commands.add_parser("check")
    checking.add_argument("--research-store", required=True)
    checking.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    checking.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "examples":
        print(zero_vol_check.render_examples())
        return 0
    if args.command == "registration":
        print(zero_vol_check.render_registration())
        return 0
    return _check(args, clock)


if __name__ == "__main__":
    raise SystemExit(main())
