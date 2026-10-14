"""
Evaluate a training run with the comparison harness.

Every method runs on the same fold windows, assets, costs and schedule,
through the same environment and transition:
    buy_and_hold, equal_weight, fixed_balanced, inverse_volatility,
    mean_variance, linear_softmax_cem, ppo_seed_<k> (no forecasts),
    guarded:ppo_seed_<k> (health gate and fallback), and ppo_with_forecasts
    (reported unavailable until Track B's forecast artifacts exist).

Usage (from the repository root):
    .venv/bin/python scripts/evaluate_rl.py --run-dir models/rl/<run_id>
        [--fold test|dev]

Writes <run-dir>/evaluation_<fold>.json (rl_engineering_report: fairness
checks, finiteness, fallbacks; no performance figure). The strategy
comparison table, comparison_<fold>.json, is written only when the run's
transition is owner-approved; until then it is withheld.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train_ppo import TRANSITIONS

from stock_agent.rl import forecasts
from stock_agent.rl.config import run_config_from_dict
from stock_agent.rl.harness import PolicyEntry, baseline_entries, run_comparison
from stock_agent.rl.health import model_is_healthy
from stock_agent.rl.learned import LinearSoftmaxParams, LinearSoftmaxPolicy
from stock_agent.rl.ppo import load_ppo_checkpoint
from stock_agent.rl.safety import GuardedPolicy


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fold", choices=("test", "dev"), default="test")
    args = parser.parse_args(argv)

    run = _read(args.run_dir / "run.json")
    config = run_config_from_dict(run["config"])
    transition, transition_info = TRANSITIONS[run["transition_key"]]
    if transition_info.to_dict() != run["transition"]:
        raise ValueError("The run was trained with a different transition than this code provides.")
    health = _read(args.run_dir / "training_health.json")
    cem = _read(args.run_dir / "cem_baseline.json")

    loaded = {seed: load_ppo_checkpoint(args.run_dir / f"seed_{seed}") for seed in run["seeds"]}
    builder = next(iter(loaded.values())).builder
    if any(c.builder.normalizer != builder.normalizer for c in loaded.values()):
        raise ValueError("The run's checkpoints disagree on the frozen normalizer.")
    params = LinearSoftmaxParams.from_dict(cem["params"])

    entries = [
        *baseline_entries(),
        PolicyEntry(
            "linear_softmax_cem", lambda: LinearSoftmaxPolicy(params, builder), kind="learned"
        ),
    ]
    for seed, checkpoint in loaded.items():
        healthy = model_is_healthy(health, seed)
        entries.append(
            PolicyEntry(f"ppo_seed_{seed}", lambda c=checkpoint: c.policy, kind="ppo", group="ppo")
        )
        entries.append(
            PolicyEntry(
                f"guarded:ppo_seed_{seed}",
                lambda c=checkpoint, h=healthy: GuardedPolicy(
                    c.policy, gate=config.gate, model_healthy=h
                ),
                kind="ppo_guarded",
                group="ppo_guarded",
            )
        )
    entries.append(
        PolicyEntry(
            "ppo_with_forecasts",
            forecasts.unavailable("Track B registry"),
            kind="ppo_with_forecasts",
        )
    )

    fold = config.plan.test if args.fold == "test" else config.plan.dev
    comparison = run_comparison(
        entries,
        fold,
        builder=builder,
        transition=transition,
        transition_info=transition_info,
        env_config=config.env,
    )
    report = {**comparison.engineering_report(), "run_id": run["run_id"], "fold_role": fold.role}
    _write(args.run_dir / f"evaluation_{fold.role}.json", report)
    print(
        f"{run['run_id']} on the {fold.role} fold: {report['label']}; fairness {report['fairness']}"
    )
    print(f"  comparison tables: {report['comparison_tables']}")
    for name, reason in report["unavailable"].items():
        print(f"  unavailable: {name}: {reason}")
    if transition_info.approved:
        _write(args.run_dir / f"comparison_{fold.role}.json", comparison.comparison_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
