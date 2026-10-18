"""
Build a Model Health summary (docs/operations/model_health.md).

Two commands:

    ci     public-safe: repository code and synthetic fixtures only. Writes
           <output-dir>/health_summary.json and <output-dir>/evidence/. This is
           what the daily GitHub Actions workflow runs and uploads.
    local  private: the ci checks plus the owner's private stores and other
           tracks' outputs, each only when it exists. Writes
           <output-dir>/latest.json, history/ and evidence/ under a private
           directory (default ~/stock-agent-private/health), which the
           Portfolio Lab's Model Health page reads. Refuses to run in GitHub
           Actions and refuses an output directory Git would track. With
           --github it records the latest GitHub Actions runs on main with
           `gh run list` (the only network step; without it, or without gh,
           those checks are insufficient evidence).

Real data and real forecast monitoring run only through `local`, on the
owner's machine, manually or on a schedule the owner installs
(docs/operations/model_health.md, "The private local monitoring path"). Alerts are informational: this script never trades,
retrains, promotes, deletes or changes configuration. Exit status: 0 the
overall state is not failed, 1 it is failed, 2 the run was refused or the
arguments are wrong.

Usage:
    .venv/bin/python scripts/health_check.py ci --output-dir OUT [--run-verify | --verify-log F]
    .venv/bin/python scripts/health_check.py local [--output-dir DIR] [--run-verify |
        --verify-log F] [--github | --ci-runs F] [--fund-store DIR] [--health-inputs DIR]
        [--registry DIR [--ml-artifacts DIR]] [--ml-root DIR] [--rl-runs DIR]
        [--ledger FILE] [--ledger-head-backup FILE] [--prospective-store DIR]
        [--anchors FILE]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from stock_agent.forecasting.prospective import ledger
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.health import collect
from stock_agent.health.ledger import RulesCalendar
from stock_agent.health.model import CONDITION_PRECEDENCE, Condition, HealthState
from stock_agent.health.prospective import ProspectiveSources

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE_CONFIG = PROJECT_ROOT / "configs" / "prospective" / "etf_prospective_v1.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    commands = parser.add_subparsers(dest="command", required=True)
    ci = commands.add_parser("ci", help="public-safe checks on synthetic fixtures")
    local = commands.add_parser("local", help="private checks on this machine")
    ci.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--output-dir", type=Path, default=collect.DEFAULT_PRIVATE_DIR)
    for command in (ci, local):
        verify = command.add_mutually_exclusive_group()
        verify.add_argument("--run-verify", action="store_true", help="run scripts/verify.py")
        verify.add_argument(
            "--verify-log", type=Path, help="a verify log this script recorded (dated inside)"
        )
    runs = local.add_mutually_exclusive_group()
    runs.add_argument("--ci-runs", type=Path, help="gh run list --json output (runbook)")
    runs.add_argument(
        "--github",
        action="store_true",
        help="record the latest runs on main with `gh run list` (needs gh signed in)",
    )
    local.add_argument("--fund-store", type=Path, help="a private F1 fund-store root")
    local.add_argument(
        "--health-inputs",
        type=Path,
        default=PROJECT_ROOT / "reports" / "health_inputs",
        help="producers' health inputs, <dir>/<artifact>/<run_id>.json (default reports/...)",
    )
    local.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ml" / "registry",
        help="the model-registry root (default reports/ml/registry)",
    )
    local.add_argument("--ml-artifacts", type=Path, help="its artifacts root (default: parent)")
    local.add_argument(
        "--ml-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ml",
        help="the experiment runner's root, holding ledger.jsonl (default reports/ml)",
    )
    local.add_argument(
        "--rl-runs",
        type=Path,
        default=PROJECT_ROOT / "models" / "rl",
        help="RL run directories with evaluation reports (default models/rl)",
    )
    local.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / ledger.DEFAULT_LEDGER,
        help="the prospective forecast ledger (default data/raw/prospective_forecasts/...)",
    )
    local.add_argument(
        "--ledger-head-backup",
        type=Path,
        help="a file outside the ledger whose first line is the head hash the job printed",
    )
    local.add_argument("--prospective-store", type=Path, help="the private prospective store")
    local.add_argument(
        "--anchors",
        type=Path,
        help="the private anchors CSV (default $STOCK_AGENT_ANCHORS, else data/raw/...)",
    )
    return parser


def _prospective(arguments, environment: collect.Environment):
    """
    The rules calendar and the prospective sources for the local run.

    This script is a reviewed consumer of the prospective side
    (tests/test_prospective_boundaries.py, SCRIPT_CONSUMERS): it imports only
    the store's reader, the window constants and the rules calendar, and the
    anchors loader, and passes them to the health checks, which never import
    them. Nothing here reads a value; the checks keep counts and hashes only.
    """

    from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
    from stock_agent.data.prospective.inference import load_inference_window
    from stock_agent.data.prospective.window import H_END, WINDOW_START
    from stock_agent.data.vendors.anchors import AnchorScope, anchors_path, load_anchors

    calendar = RulesCalendar(sessions, CALENDAR_ID)
    anchors_file = anchors_path(arguments.anchors, project_root=PROJECT_ROOT)

    def anchors():
        return load_anchors(
            anchors_file,
            AnchorScope.PROSPECTIVE,
            project_root=PROJECT_ROOT,
            today=pd.Timestamp(environment.now()),
        )

    config = load_config(PROSPECTIVE_CONFIG)
    sources = ProspectiveSources(
        store_root=arguments.prospective_store,
        assets=tuple(config.assets),
        window_start=WINDOW_START,
        h_end=H_END,
        calendar=calendar,
        load_window=load_inference_window,
        load_anchors=anchors,
        lookback=config.prospective.lookback_sessions,
    )
    return calendar, sources


def _verify(arguments, environment: collect.Environment, *, public: bool):
    if arguments.run_verify:
        return collect.run_verify(environment, public=public)
    if arguments.verify_log is not None:
        if not arguments.verify_log.is_file():
            raise collect.LocalRunRefused(f"--verify-log {arguments.verify_log} is not a file")
        return collect.read_verify_log(arguments.verify_log)
    return None


def main(argv: list[str] | None = None, environment: collect.Environment | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environment = environment or collect.Environment(root=PROJECT_ROOT, env=dict(os.environ))
    try:
        if arguments.command == "ci":
            output = arguments.output_dir.resolve()
            verify = _verify(arguments, environment, public=True)
            summary, path = collect.run_ci(environment, output, verify=verify)
        else:
            output = arguments.output_dir.expanduser()
            collect.check_private_output(environment, output)
            verify = _verify(arguments, environment, public=False)
            calendar, sources = _prospective(arguments, environment)
            inputs = collect.PrivateInputs(
                ci_runs=arguments.ci_runs,
                github=arguments.github,
                fund_store=arguments.fund_store,
                health_inputs=arguments.health_inputs,
                registry=arguments.registry,
                ml_artifacts=arguments.ml_artifacts,
                ml_root=arguments.ml_root,
                rl_runs=arguments.rl_runs,
                ledger=arguments.ledger,
                ledger_head_backup=arguments.ledger_head_backup,
                calendar=calendar,
                prospective=sources,
            )
            summary, path = collect.run_local(environment, output, verify=verify, inputs=inputs)
    except (collect.LocalRunRefused, collect.PublicSummaryRefused) as error:
        print(f"health_check: refused: {error}", file=sys.stderr)
        return 2

    # Counts by operator condition, so stale, not activated and no evidence stay apart.
    conditions = Counter(check.condition for check in summary.checks)
    counts = ", ".join(
        f"{item.value} {conditions[item]}" for item in CONDITION_PRECEDENCE if conditions[item]
    )
    print(f"health_check: overall {summary.overall_state.value} ({counts})")
    rank = {item: index for index, item in enumerate(CONDITION_PRECEDENCE)}
    for check in sorted(summary.checks, key=lambda check: rank[check.condition]):
        if check.condition is not Condition.HEALTHY:
            print(f"  {check.condition.value:<14} {check.check_id}: {check.summary}")
    print(f"health_check: {len(summary.alerts)} informational alerts; summary at {path}")
    return 1 if summary.overall_state is HealthState.FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
