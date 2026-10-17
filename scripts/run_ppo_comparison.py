"""
Run the registered synthetic comparison of the baselines against PPO.

Synthetic scenarios only: nothing this writes is evidence of real investment
performance. Every strategy runs through the approved T5a simulator (D1-D8,
approved 2026-10-15) with the same schedule, costs, build and registered
constraint set; see docs/research/ppo_comparison.md.

Usage (from the repository root):
    .venv/bin/python scripts/run_ppo_comparison.py train
        --config configs/comparison/ppo_comparison_v1.yaml
        [--output-root models/comparison] [--workers 4]
    .venv/bin/python scripts/run_ppo_comparison.py evaluate
        --config configs/comparison/ppo_comparison_v1.yaml
        --run-dir models/comparison/<run_id> [--fold test|dev]
        [--reports-root reports/comparison]
        [--health-inputs-root reports/health_inputs] [--record-repeat-look]

`train` writes <output-root>/<run_id>/ (ignored by Git): every learned
member's dev-selected checkpoint and the frozen selection.json. The test
fold is never built while training.

`evaluate` records the look in the configuration's private evaluation log,
~/stock-agent-private/comparison/looks/<config_sha256>.jsonl (write-once,
hash-chained; the same file whatever --reports-root is), before evaluating,
then writes the versioned artifact
<reports-root>/<artifact_id>/comparison.json and one backtest_summary per
(constraint set, strategy) under <health-inputs-root>/backtest_summary/
(all ignored by Git). The test fold is evaluated once per registered
configuration; a second look is refused unless --record-repeat-look records
it, counted and labelled, as a repeat.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from train_ppo import git_code_version

from stock_agent.comparison.ledger import EvaluationLog
from stock_agent.comparison.protocol import load_comparison_config
from stock_agent.comparison.runner import evaluate, train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "comparison"
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "reports" / "comparison"
DEFAULT_HEALTH_INPUTS_ROOT = PROJECT_ROOT / "reports" / "health_inputs"


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_version: Callable[[], dict] = git_code_version,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train and dev-select every learned member")
    training.add_argument("--config", required=True, type=Path)
    training.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    training.add_argument("--workers", type=int, default=1)
    evaluation = commands.add_parser("evaluate", help="evaluate the frozen selection once")
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--run-dir", required=True, type=Path)
    evaluation.add_argument("--fold", choices=("test", "dev"), default="test")
    evaluation.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    evaluation.add_argument("--health-inputs-root", type=Path, default=DEFAULT_HEALTH_INPUTS_ROOT)
    evaluation.add_argument("--record-repeat-look", action="store_true")
    args = parser.parse_args(argv)

    config = load_comparison_config(args.config)
    code = code_version()
    if args.command == "train":
        now = clock()
        run_dir = args.output_root / f"{now:%Y%m%dT%H%M%SZ}-{config.sha256()[:12]}"
        print(f"training {config.name} ({config.sha256()[:12]}) into {run_dir}", flush=True)
        path = train(config, run_dir=run_dir, code=code, clock=clock, workers=args.workers)
        print(f"wrote {path}")
        return 0
    path = evaluate(
        config,
        run_dir=args.run_dir,
        fold_role=args.fold,
        reports_root=args.reports_root,
        health_inputs_root=args.health_inputs_root,
        code=code,
        clock=clock,
        repeat_look=args.record_repeat_look,
    )
    log = EvaluationLog.for_config(config.sha256())
    print(f"wrote {path}")
    print(f"look recorded in {log.path} ({log.looks('test')} test look(s) of this configuration)")
    print("Synthetic scenarios only — not evidence of real investment performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
