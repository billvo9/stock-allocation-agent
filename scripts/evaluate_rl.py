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

Every evaluation is write-once and counted: it writes
<run-dir>/evaluation_<fold>_<n>.json (rl_engineering_report: fairness checks,
finiteness, fallbacks, provenance; no performance figure), where n numbers
that fold's evaluations, and appends one line to <run-dir>/evaluations.jsonl,
so every look at the test fold is on record. The strategy comparison table,
comparison_<fold>_<n>.json, is written only when the run's transition is
owner-approved; until then it is withheld.

Checkpoints from another scenario generator or config are refused; a
different code SHA warns and is recorded in the report's provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from train_ppo import TRANSITIONS, git_code_version

from stock_agent.rl import forecasts
from stock_agent.rl.config import run_config_from_dict
from stock_agent.rl.harness import PolicyEntry, baseline_entries, run_comparison
from stock_agent.rl.health import model_is_healthy
from stock_agent.rl.learned import LinearSoftmaxParams, LinearSoftmaxPolicy
from stock_agent.rl.ppo import load_ppo_checkpoint
from stock_agent.rl.safety import GuardedPolicy


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, record: dict) -> None:
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _ledger(run_dir: Path) -> list[dict]:
    path = run_dir / "evaluations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    code_version: Callable[[], dict] = git_code_version,
) -> int:
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

    current = code_version()
    loaded = {
        seed: load_ppo_checkpoint(
            args.run_dir / f"seed_{seed}",
            expected_config_sha256=run["config_sha256"],
            expected_code_sha=current.get("git_sha"),
        )
        for seed in run["seeds"]
    }
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
    number = 1 + sum(1 for entry in _ledger(args.run_dir) if entry["fold"] == fold.role)
    evaluated_at = clock().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    first = next(iter(loaded.values())).provenance
    report = {
        **comparison.engineering_report(),
        "run_id": run["run_id"],
        "fold_role": fold.role,
        "fold_evaluation_number": number,
        "evaluated_at": evaluated_at,
        "provenance": {
            "trained_code_sha": first["trained_code_sha"],
            "current_code_sha": current.get("git_sha"),
            "current_git_dirty": current.get("git_dirty"),
            "code_sha_matches": all(c.provenance["code_sha_matches"] for c in loaded.values()),
            "config_sha256": run["config_sha256"],
            "scenario_generator_version": first["scenario_generator_version"],
        },
    }
    name = f"evaluation_{fold.role}_{number}.json"
    _write_once(args.run_dir / name, report)
    if transition_info.approved:
        _write_once(
            args.run_dir / f"comparison_{fold.role}_{number}.json", comparison.comparison_table()
        )
    with (args.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as ledger:
        entry = {
            "fold": fold.role,
            "number": number,
            "file": name,
            "evaluated_at": evaluated_at,
            "code_sha": current.get("git_sha"),
            "git_dirty": current.get("git_dirty"),
        }
        ledger.write(json.dumps(entry, sort_keys=True) + "\n")
    print(
        f"{run['run_id']} on the {fold.role} fold (evaluation {number} of that fold): "
        f"{report['label']}; fairness {report['fairness']}"
    )
    print(f"  comparison tables: {report['comparison_tables']}")
    for method, reason in report["unavailable"].items():
        print(f"  unavailable: {method}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
