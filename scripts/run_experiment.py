"""
Run one ML research experiment from a YAML config (configs/ml/*.yaml).

Every output is versioned and written once under the output root (default
reports/ml, ignored by Git): the T2 contract run, the experiment record and
tables, one out-of-fold forecast artifact per candidate, one new registry
version per candidate, and a ledger line per attempt. Only synthetic data
sources exist; the 2025+ lockbox is never touched. Afterwards the registry's
health export is refreshed under --health-inputs (default
reports/health_inputs; --no-health-export skips it).

Usage (from the repository root):
    .venv/bin/python scripts/run_experiment.py configs/ml/equity_xs_planted_linear.yaml
        [--output-root reports/ml] [--health-inputs reports/health_inputs]
        [--no-health-export]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.forecasting.config import load_config
from stock_agent.forecasting.experiment import REGISTRY_DIRECTORY, run_experiment
from stock_agent.forecasting.provenance import code_provenance, environment
from stock_agent.registry import open_registry
from stock_agent.registry.export import build_export, write_export

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_COLUMNS = (
    "mean_rank_ic",
    "mean_rank_ic_t",
    "rank_ic_coverage",
    "coverage_weighted_rank_ic",
    "coverage_weighted_rank_ic_t",
    "oos_r2_vs_zero",
    "permutation_p_value",
    "holm_adjusted_p",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "reports" / "ml")
    parser.add_argument(
        "--health-inputs", type=Path, default=PROJECT_ROOT / "reports" / "health_inputs"
    )
    parser.add_argument("--no-health-export", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    started_at = datetime.now(UTC)
    provenance = {"code": code_provenance(PROJECT_ROOT), "environment": environment()}
    result = run_experiment(
        config, output_root=args.output_root, provenance=provenance, started_at=started_at
    )
    registry = open_registry(args.output_root / REGISTRY_DIRECTORY, args.output_root)
    print(f"experiment {config.experiment} run {result.run_id}")
    print(
        f"config sha256 {config.sha256}; results sha256 {result.record['outputs']['results_sha256']}"
    )
    print(f"record: {result.directory / 'record.json'}")
    print(f"checks: {result.record['checks_summary']}")
    rows = []
    for entry in result.entries:
        assessment = registry.assessment(entry.ref)
        rows.append(
            {
                "entry": str(entry.ref),
                "stage": assessment.stage,
                "blockers": ",".join(sorted({c.reason for c in assessment.blockers})),
                **{key: entry.metrics.get(key) for key in SUMMARY_COLUMNS},
            }
        )
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    if not args.no_health_export:
        document = build_export(
            registry, generated_at=datetime.now(UTC), code_sha=provenance["code"]["git_sha"]
        )
        print(f"registry health export -> {write_export(document, args.health_inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
