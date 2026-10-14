"""
Write the registry's health export for the Model Health Center.

The registry stays the source of truth; this writes a derived, write-once
snapshot (stages computed from re-verified evidence at export time) to
<health-inputs>/model_registry/<run_id>.json as strict JSON (agreed input
format v1.0; docs/research/ml_registry.md, "Health export"). Run it after
an activation or a rollback; scripts/run_experiment.py writes one after
every run.

Usage (from the repository root):
    .venv/bin/python scripts/export_registry_health.py
        [--registry reports/ml/registry] [--health-inputs reports/health_inputs]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.forecasting.provenance import code_provenance
from stock_agent.registry import open_registry
from stock_agent.registry.export import build_export, write_export

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export(registry_root: Path, health_inputs: Path, *, code_sha: str | None) -> Path:
    reader = open_registry(registry_root)
    document = build_export(reader, generated_at=datetime.now(UTC), code_sha=code_sha)
    return write_export(document, health_inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--registry", type=Path, default=PROJECT_ROOT / "reports" / "ml" / "registry"
    )
    parser.add_argument(
        "--health-inputs", type=Path, default=PROJECT_ROOT / "reports" / "health_inputs"
    )
    args = parser.parse_args(argv)
    path = export(
        args.registry, args.health_inputs, code_sha=code_provenance(PROJECT_ROOT)["git_sha"]
    )
    print(f"registry health export -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
