"""
Development diagnostics run: the T2 null models, controls and canary, and
with --with-linear-models the T4 Ridge, Lasso and Elastic Net candidates
(nested purged selection) and the OLS diagnostic.

Builds the feature panel, restricts it to the config universe, truncates it
before the 2025-01-01 lockbox and only then computes labels, builds the
development folds, runs model_diagnostics.runner.run_diagnostics, and writes
reports/runs/<run_id>/ plus a line in reports/ledger.jsonl (both ignored by
Git). Development mode only: this script never builds holdout folds.

Usage (from the repository root: the feature SQL reads data/raw/prices.parquet
by a relative path):
    .venv/bin/python scripts/run_null_diagnostics.py [--output-root reports]
        [--permutation-draws 49] [--bootstrap-reps 999] [--seed 0]
        [--label {equal_weight_excess,raw}] [--with-linear-models]

The default label is the equal-weight excess return (owner decision,
2026-10-10); the raw return is stored beside it. The comparators follow the
label (nested.comparison_settings).
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from stock_agent.config import load_asset_symbols
from stock_agent.features.build import SQL_PATH, build_model_dataset
from stock_agent.features.training import MODEL_EXCESS_LABEL_SPEC, MODEL_LABEL_SPEC
from stock_agent.model_diagnostics.artifacts import append_ledger, write_run
from stock_agent.model_diagnostics.nested import comparison_settings, linear_models
from stock_agent.model_diagnostics.runner import (
    DiagnosticsConfig,
    default_models,
    prepare_development_frame,
    run_diagnostics,
)
from stock_agent.model_validation.audit import file_sha256
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    make_expanding_folds,
    make_test_windows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"
PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "prices.parquet"
DEVELOPMENT_START = "2019-01-01"
BLOCK_SESSIONS = 63
DIAGNOSTIC_FEATURES = ("daily_return", "momentum_20d", "volatility_20d")
PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "duckdb",
    "yfinance",
    "edgartools",
    "scikit-learn",
    "scipy",
)

UNIVERSE_NOTE = (
    "Config universe chosen with hindsight (semiconductors, survivorship-biased); "
    "results verify the pipeline and are not product evidence."
)
ADJUSTED_PRICE_DISCLOSURE = (
    "adjusted_close levels before the lockbox embed later (2025+) dividend and split "
    "factors. Same-symbol ratios (returns, momentum, volatility, labels) cancel the "
    "common factor, so no holdout value enters a feature or label; outputs may differ "
    "in the last bits after a re-download."
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout


def code_provenance() -> dict[str, object]:
    status = _git("status", "--porcelain")
    dirty = bool(status.strip())
    provenance: dict[str, object] = {
        "git_sha": _git("rev-parse", "HEAD").strip(),
        "git_dirty": dirty,
    }
    if dirty:
        diff = _git("diff", "HEAD", "--binary")
        untracked = _git("ls-files", "--others", "--exclude-standard")
        digest = hashlib.sha256((diff + "\0" + untracked).encode("utf-8")).hexdigest()
        provenance["git_diff_sha256"] = digest
    return provenance


def environment() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--permutation-draws", type=int, default=49)
    parser.add_argument("--bootstrap-reps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label", choices=("equal_weight_excess", "raw"), default="equal_weight_excess"
    )
    parser.add_argument("--with-linear-models", action="store_true")
    args = parser.parse_args(argv)
    label_spec = (
        MODEL_EXCESS_LABEL_SPEC if args.label == "equal_weight_excess" else MODEL_LABEL_SPEC
    )
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error(
            f"run from the repository root ({PROJECT_ROOT}); the feature SQL uses relative paths"
        )

    started_at = datetime.now(UTC)
    candidates = ("ridge", "lasso", "elastic_net") if args.with_linear_models else ()
    defaults = DiagnosticsConfig()
    config = replace(
        defaults,
        permutation_draws=args.permutation_draws,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        permutation_models=(*defaults.permutation_models, *candidates),
        stale_models=(*defaults.stale_models, *candidates),
        **comparison_settings(label_spec, with_linear=args.with_linear_models),
    )
    models = default_models()
    if args.with_linear_models:
        models += linear_models(label_spec, DIAGNOSTIC_FEATURES)
    ledger_entry: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "argv": sys.argv[1:] if argv is None else argv,
        "mode": "development",
    }
    try:
        symbols = load_asset_symbols(CONFIG_PATH)
        panel = build_model_dataset()
        labeled, truncation = prepare_development_frame(
            panel, symbols=symbols, label_spec=label_spec, holdout_start=MODEL_HOLDOUT_START
        )
        truncation["holdout_rows_loaded"] = True  # the feature SQL scans the whole price file
        universe = sorted(set(labeled["symbol"]))
        sessions = labeled["date"].drop_duplicates().sort_values()
        windows = make_test_windows(
            sessions,
            start=DEVELOPMENT_START,
            end=MODEL_HOLDOUT_START,
            block_sessions=BLOCK_SESSIONS,
        )
        folds = make_expanding_folds(labeled, windows, holdout_start=MODEL_HOLDOUT_START)
        provenance = {
            "code": code_provenance(),
            "environment": environment(),
            "inputs": {
                "prices_parquet_sha256": file_sha256(PRICES_PATH),
                "prices_parquet_bytes": PRICES_PATH.stat().st_size,
                "feature_sql_sha256": file_sha256(SQL_PATH),
                "assets_config_sha256": file_sha256(CONFIG_PATH),
                "retrieval_time": None,  # not recorded by the downloader (follow-up)
            },
            "truncation": truncation,
            "fold_params": {
                "start": DEVELOPMENT_START,
                "end": MODEL_HOLDOUT_START.isoformat(),
                "block_sessions": BLOCK_SESSIONS,
                "train_start": None,
                "mode": "development",
            },
            "universe_note": UNIVERSE_NOTE,
            "adjusted_price_disclosure": ADJUSTED_PRICE_DISCLOSURE,
        }
        run = run_diagnostics(
            labeled,
            folds,
            universe=universe,
            label_spec=label_spec,
            diagnostic_features=DIAGNOSTIC_FEATURES,
            provenance=provenance,
            started_at=started_at,
            models=models,
            config=config,
        )
        finished_at = datetime.now(UTC)
        directory = write_run(
            run,
            args.output_root,
            extra_record={"finished_at": finished_at.isoformat(), "status": "completed"},
        )
    except Exception as error:
        append_ledger(
            args.output_root,
            {**ledger_entry, "status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    append_ledger(
        args.output_root,
        {
            **ledger_entry,
            "status": "completed",
            "run_id": run.run_id,
            "spec_sha256": run.record["spec_sha256"],
            "results_sha256": run.record["outputs"]["results_sha256"],
            "variants": run.record["spec"]["registered_variants"],
            "git_sha": provenance["code"]["git_sha"],
            "git_dirty": provenance["code"]["git_dirty"],
            "finished_at": finished_at.isoformat(),
        },
    )
    summary = run.record["checks_summary"]
    print(f"run {run.run_id} -> {directory}")
    print(f"checks: {summary}")
    pd.set_option("display.width", 200)
    print(run.tables["checks"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
