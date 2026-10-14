"""
Build a Model Health summary (docs/operations/model_health.md).

Two commands:

    ci     public-safe: repository code and synthetic fixtures only. Writes
           <output-dir>/health_summary.json and <output-dir>/evidence/. This is
           what the daily GitHub Actions workflow runs and uploads.
    local  private: the ci checks plus the owner's private stores and other
           tracks' outputs, each only when its path is given. Writes
           <output-dir>/latest.json, history/ and evidence/ under a private
           directory (default ~/stock-agent-private/health). Refuses to run in
           GitHub Actions and refuses an output directory Git would track.

Alerts are informational: this script never trades, retrains, promotes,
deletes or changes configuration. Exit status: 0 the overall state is not
failed, 1 it is failed, 2 the run was refused or the arguments are wrong.

Usage:
    .venv/bin/python scripts/health_check.py ci --output-dir OUT [--run-verify | --verify-log F]
    .venv/bin/python scripts/health_check.py local [--output-dir DIR] [--run-verify |
        --verify-log F] [--ci-runs F] [--fund-store DIR] [--health-inputs DIR]
        [--registry DIR [--ml-artifacts DIR]]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stock_agent.health import collect
from stock_agent.health.model import HealthState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        verify.add_argument("--verify-log", type=Path, help="an existing verify.py output")
    local.add_argument("--ci-runs", type=Path, help="gh run list --json output (runbook)")
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
    return parser


def _verify(arguments, environment: collect.Environment, evidence_log: Path, *, public: bool):
    if arguments.run_verify:
        return collect.run_verify(environment, evidence_log, public=public)
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
            log = output / collect.EVIDENCE_DIR / "verify.log"
            verify = _verify(arguments, environment, log, public=True)
            summary, path = collect.run_ci(environment, output, verify=verify)
        else:
            output = arguments.output_dir.expanduser()
            collect.check_private_output(environment, output)
            log = output.resolve() / collect.EVIDENCE_DIR / "verify-latest.log"
            verify = _verify(arguments, environment, log, public=False)
            inputs = collect.PrivateInputs(
                ci_runs=arguments.ci_runs,
                fund_store=arguments.fund_store,
                health_inputs=arguments.health_inputs,
                registry=arguments.registry,
                ml_artifacts=arguments.ml_artifacts,
            )
            summary, path = collect.run_local(environment, output, verify=verify, inputs=inputs)
    except (collect.LocalRunRefused, collect.PublicSummaryRefused) as error:
        print(f"health_check: refused: {error}", file=sys.stderr)
        return 2

    counts = ", ".join(f"{state} {n}" for state, n in summary.state_counts().items() if n)
    print(f"health_check: overall {summary.overall_state.value} ({counts})")
    for check in summary.checks:
        if check.state is not HealthState.HEALTHY:
            print(f"  {check.state.value:<22} {check.check_id}: {check.summary}")
    print(f"health_check: {len(summary.alerts)} informational alerts; summary at {path}")
    return 1 if summary.overall_state is HealthState.FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
