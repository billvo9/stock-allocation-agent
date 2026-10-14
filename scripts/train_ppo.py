"""
Train PPO, and the linear-softmax CEM baseline, on the synthetic scenarios.

ENGINEERING ONLY until the owner approves decisions D1-D8 and Track A's
`portfolio_step` merges: the only transition available is the labelled
stand-in for the proposed rules, and every artifact carries the label
`engineering_only_unapproved_transition`. A completed run is an engineering
result, not evidence that any policy is a good investment.

Usage (from the repository root, with the optional RL dependencies
installed, see docs/research/rl_ppo.md):
    .venv/bin/python scripts/train_ppo.py --config configs/rl/ppo_smoke.yaml
        [--output-root models/rl]

Writes <output-root>/<run_id>/ (ignored by Git):
    run.json                  configuration, code version, libraries, labels
    training_health.json      rl_training_health 1.0.0 for every seed
    cem_baseline.json         the CEM baseline (train fold, dev-selected)
    seed_<k>/best_dev.zip     the dev-selected PPO checkpoint
    seed_<k>/checkpoint.json  normalizer, config hash, code SHA, scenario
                              version, library versions, model hash
and the Model Health Center's input (agreed format v1.0, ignored by Git):
    <health-inputs-root>/rl_training_health/<run_id>.json
A run that raises still writes that input, with status "failed".

Reproducibility: the same configuration, code SHA and library versions give
identical parameters (CPU, one torch thread, seeded episodes). Tuning and
checkpoint selection use the dev fold only; the test fold is evaluated by
scripts/evaluate_rl.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.rl import reference_transition
from stock_agent.rl.config import load_run_config
from stock_agent.rl.health import (
    failed_health_input_record,
    health_input_record,
    training_health_report,
    write_health_input,
)
from stock_agent.rl.learned import train_linear_softmax
from stock_agent.rl.observation import ObservationBuilder, fit_normalizer
from stock_agent.rl.ppo import library_versions, train_ppo_seed
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.scenarios import ASSET_NAMES, CASH_INDEX, GENERATOR_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "rl"
DEFAULT_HEALTH_INPUTS_ROOT = PROJECT_ROOT / "reports" / "health_inputs"
RUN_SCHEMA = "rl_run"
RUN_SCHEMA_VERSION = "1.0.0"

TRANSITIONS = {
    "proposed_reference": (
        reference_transition.proposed_portfolio_step,
        reference_transition.TRANSITION_INFO,
    ),
}


def git_code_version(root: Path = PROJECT_ROOT) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_version: Callable[[], dict] = git_code_version,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--health-inputs-root", type=Path, default=DEFAULT_HEALTH_INPUTS_ROOT)
    parser.add_argument("--transition", choices=sorted(TRANSITIONS), default="proposed_reference")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    transition, transition_info = TRANSITIONS[args.transition]
    label = "approved_transition" if transition_info.approved else transition_info.status
    now = clock()
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{config.sha256()[:12]}"
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    code = code_version()
    print(f"run {run_id}: {label}", flush=True)
    try:
        health = _train(config, transition, transition_info, label, run_id, created_at, code, out)
    except Exception as error:
        write_health_input(
            failed_health_input_record(
                run_id=run_id,
                generated_at=generated_at,
                code_sha=code.get("git_sha"),
                scenario_version=GENERATOR_VERSION,
                transition=transition_info,
                seeds=len(config.training.seeds),
                detail=f"{type(error).__name__}: {error}",
            ),
            args.health_inputs_root,
        )
        raise
    path = write_health_input(
        health_input_record(
            health,
            generated_at=generated_at,
            code_sha=code.get("git_sha"),
            scenario_version=GENERATOR_VERSION,
            thresholds=config.health,
        ),
        args.health_inputs_root,
    )
    print(f"  training health: {health['status']}", flush=True)
    for record in health["seeds"]:
        print(
            f"    seed {record['seed']}: {record['status']} {record['reasons'] + record['warnings']}"
        )
    print(f"wrote {out}")
    print(f"wrote {path}")
    return 0


def _train(config, transition, transition_info, label, run_id, created_at, code, out) -> dict:
    normalizer = fit_normalizer(config.plan.train, config.features)
    builder = ObservationBuilder(config.features, normalizer, ASSET_NAMES, CASH_INDEX)

    seeds = []
    for seed in config.training.seeds:
        print(f"  training PPO seed {seed}", flush=True)
        seeds.append(
            train_ppo_seed(
                config,
                seed,
                builder=builder,
                transition=transition,
                transition_info=transition_info,
                out_dir=out / f"seed_{seed}",
                run_id=run_id,
                code=code,
                created_at=created_at,
            )
        )

    print("  training the linear-softmax CEM baseline", flush=True)
    cem = train_linear_softmax(
        config.plan,
        builder=builder,
        transition=transition,
        transition_info=transition_info,
        env_config=config.env,
        config=config.cem,
    )
    _write(out / "cem_baseline.json", {**cem.to_dict(), "run_id": run_id})

    health = training_health_report(
        run_id=run_id,
        config_sha256=config.sha256(),
        created_at=created_at,
        transition=transition_info,
        reward=REWARD_INFO,
        thresholds=config.health,
        seeds=[run.health for run in seeds],
    )
    _write(out / "training_health.json", health)
    _write(
        out / "run.json",
        {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": created_at,
            "label": label,
            "transition_key": next(k for k, v in TRANSITIONS.items() if v[1] == transition_info),
            "transition": transition_info.to_dict(),
            "reward": REWARD_INFO.to_dict(),
            "config": config.to_dict(),
            "config_sha256": config.sha256(),
            "code": code,
            "libraries": library_versions(),
            "seeds": list(config.training.seeds),
            "normalizer_fitted_on": normalizer.fitted_on,
        },
    )
    return health


if __name__ == "__main__":
    sys.exit(main())
