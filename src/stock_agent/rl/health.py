"""
The training-health report for RL runs (schema `rl_training_health`).

One JSON document per training run, for the Model Health Center and for the
deployment gate (`safety.GuardedPolicy(model_healthy=...)`). Built from
plain records, so it needs no learning library:

- per update (one PPO optimisation phase): approximate KL, clip fraction,
  entropy, value loss, explained variance, action standard deviation, and
  the largest absolute parameter and count of non-finite parameters after
  the update;
- per rollout: the environment rewards (before any value bootstrap), the
  count of non-finite rewards and of imputed network inputs;
- dev-fold checkpoint evaluations.

Status per seed:
- unhealthy: no_updates, non_finite_reward, non_finite_loss,
  non_finite_parameters, parameter_explosion, non_finite_dev_score;
- warning: approx_kl_spike, high_clip_fraction, action_std_collapse;
- healthy otherwise.
The run is as bad as its worst seed. `seed_spread` reports every seed's
selected dev score, never only the best.

`health_input_record` condenses a report into the agreed v1.0 input for the
Model Health Center (`<health inputs root>/rl_training_health/<run_id>.json`;
the root is the caller's, `scripts/train_ppo.py` by default):
the common envelope plus the checks nan_or_inf, exploding_values, approx_kl,
clip_fraction, entropy_collapse, value_loss, reward_trend and seed_spread,
each with this module's own status and threshold.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from stock_agent.rl.transition import TransitionInfo

HEALTH_SCHEMA = "rl_training_health"
HEALTH_SCHEMA_VERSION = "1.0.0"

_LOSSES = ("approx_kl", "clip_fraction", "entropy", "value_loss")


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(v)]


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = _finite(values)
    if not finite:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _series(updates: list[dict], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in updates if key in row]
    finite = _finite(values)
    return {
        "first": values[0] if values and math.isfinite(values[0]) else None,
        "last": values[-1] if values and math.isfinite(values[-1]) else None,
        "mean": statistics.fmean(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _usable(score) -> bool:
    return score is not None and math.isfinite(score)


def _selection(evaluations: list[dict]) -> dict[str, object]:
    """The best finite dev score (ties: the earliest); missing scores never crash."""

    usable = [i for i, row in enumerate(evaluations) if _usable(row["score"])]
    best = max(usable, key=lambda i: (evaluations[i]["score"], -i)) if usable else None
    return {
        "metric": "mean dev-fold log growth (deterministic policy)",
        "evaluations": [dict(row) for row in evaluations],
        "selected_timesteps": None if best is None else evaluations[best]["timesteps"],
        "selected_score": None if best is None else evaluations[best]["score"],
        "checkpoints_compared": len(evaluations),
        "non_finite_scores": len(evaluations) - len(usable),
    }


def seed_health(
    seed: int,
    *,
    updates: list[dict],
    rollouts: list[dict],
    evaluations: list[dict],
    thresholds,
) -> dict[str, object]:
    rewards = [float(r) for rollout in rollouts for r in rollout["rewards"]]
    non_finite_rewards = sum(int(rollout["non_finite_rewards"]) for rollout in rollouts)
    imputed = sum(int(rollout.get("imputed_features", 0)) for rollout in rollouts)
    features = sum(int(rollout.get("features", 0)) for rollout in rollouts)

    reasons: list[str] = []
    warnings: list[str] = []
    if not updates:
        reasons.append("no_updates")
    if non_finite_rewards or any(not math.isfinite(r) for r in rewards):
        reasons.append("non_finite_reward")
    if any(not math.isfinite(float(row[key])) for row in updates for key in _LOSSES):
        reasons.append("non_finite_loss")
    if any(int(row["param_non_finite"]) for row in updates):
        reasons.append("non_finite_parameters")
    if any(float(row["param_max_abs"]) > thresholds.max_param_abs for row in updates):
        reasons.append("parameter_explosion")
    if any(not _usable(row["score"]) for row in evaluations):
        reasons.append("non_finite_dev_score")

    kl = _series(updates, "approx_kl")
    clip = _series(updates, "clip_fraction")
    std = _series(updates, "action_std")
    if kl["max"] is not None and kl["max"] > thresholds.kl_warning:
        warnings.append("approx_kl_spike")
    if clip["mean"] is not None and clip["mean"] > thresholds.clip_fraction_warning:
        warnings.append("high_clip_fraction")
    if std["last"] is not None and std["last"] < thresholds.min_action_std:
        warnings.append("action_std_collapse")

    status = "unhealthy" if reasons else "warning" if warnings else "healthy"
    return {
        "seed": int(seed),
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "updates": len(updates),
        "reward": {**_stats(rewards), "non_finite": non_finite_rewards},
        "approx_kl": kl,
        "clip_fraction": clip,
        "entropy": _series(updates, "entropy"),
        "value_loss": _series(updates, "value_loss"),
        "explained_variance": _series(updates, "explained_variance"),
        "action_std": std,
        "parameters": {
            "max_abs": max((float(row["param_max_abs"]) for row in updates), default=None),
            "non_finite": sum(int(row["param_non_finite"]) for row in updates),
        },
        "imputed_feature_share": imputed / features if features else 0.0,
        "dev_selection": _selection(evaluations),
    }


def seed_spread(seeds: list[dict]) -> dict[str, object]:
    values = [record["dev_selection"]["selected_score"] for record in seeds]
    finite = _finite([v for v in values if v is not None])
    return {
        "metric": "selected checkpoint's mean dev-fold log growth, one value per seed",
        "values": values,
        "n": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "std": statistics.pstdev(finite) if finite else None,
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def training_health_report(
    *,
    run_id: str,
    config_sha256: str,
    created_at: str,
    transition: TransitionInfo,
    reward,
    thresholds,
    seeds: list[dict],
) -> dict[str, object]:
    statuses = {record["status"] for record in seeds}
    status = (
        "unhealthy"
        if "unhealthy" in statuses or not seeds
        else "warning"
        if "warning" in statuses
        else "healthy"
    )
    return {
        "schema": HEALTH_SCHEMA,
        "schema_version": HEALTH_SCHEMA_VERSION,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "created_at": created_at,
        "label": "approved_transition" if transition.approved else transition.status,
        "transition": transition.to_dict(),
        "reward": reward.to_dict(),
        "thresholds": {
            "max_param_abs": thresholds.max_param_abs,
            "kl_warning": thresholds.kl_warning,
            "clip_fraction_warning": thresholds.clip_fraction_warning,
            "min_action_std": thresholds.min_action_std,
        },
        "status": status,
        "seeds": seeds,
        "seed_spread": seed_spread(seeds),
    }


def model_is_healthy(report: dict, seed: int) -> bool:
    """True when the seed's training record exists and is not unhealthy."""

    for record in report.get("seeds", []):
        if record.get("seed") == seed:
            return record.get("status") in ("healthy", "warning")
    return False


# --- the agreed Model Health Center input (health-input-formats-agreed.md, v1.0) ---

HEALTH_INPUT_ARTIFACT = "rl_training_health"
HEALTH_INPUT_SCHEMA_VERSION = "1.0"
VALUE_LOSS_GROWTH_WARNING = 10.0
MIN_SEEDS_FOR_SPREAD = 3
CHECK_IDS = (
    "nan_or_inf",
    "exploding_values",
    "approx_kl",
    "clip_fraction",
    "entropy_collapse",
    "value_loss",
    "reward_trend",
    "seed_spread",
)
_HEX = set("0123456789abcdef")


def _check(check_id, status, value, unit, threshold, detail) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": status,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "detail": detail,
    }


def _envelope(
    *, run_id, generated_at, code_sha, git_dirty, scenario_version, transition_status, seeds
):
    if code_sha is not None and (len(code_sha) != 40 or not set(code_sha) <= _HEX):
        raise ValueError("code_sha must be a 40-character lowercase hex SHA or None.")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise ValueError("git_dirty must be a bool or None.")
    return {
        "artifact": HEALTH_INPUT_ARTIFACT,
        "schema_version": HEALTH_INPUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "code_sha": code_sha,
        "git_dirty": git_dirty,
        "synthetic": True,
        "run_id": run_id,
        "algorithm": "ppo",
        "transition_status": transition_status,
        "seeds": int(seeds),
        "scenario_version": scenario_version,
    }


def _transition_status(status: str) -> str:
    return "approved" if status == "approved" else "engineering_only_unapproved_transition"


def _extreme(values, pick):
    finite = _finite([v for v in values if v is not None])
    return pick(finite) if finite else None


def health_input_record(
    report: dict,
    *,
    generated_at: str,
    code_sha: str | None,
    scenario_version: str,
    thresholds,
    git_dirty: bool | None = None,
    status: str = "completed",
) -> dict[str, object]:
    """
    The agreed v1.0 summary of a `training_health_report` for the Model Health
    Center. Track C owns every status and threshold; the reader recomputes
    nothing. Dev scores enter only as training diagnostics (each seed's
    deterministic policy against its own untrained start), never as a
    strategy comparison.
    """

    seeds = report["seeds"]
    non_finite = (
        sum(int(s["reward"]["non_finite"]) for s in seeds)
        + sum(int(s["parameters"]["non_finite"]) for s in seeds)
        + sum(int(s["dev_selection"].get("non_finite_scores", 0)) for s in seeds)
    )
    loss_seeds = [s["seed"] for s in seeds if "non_finite_loss" in s["reasons"]]
    nan_value = non_finite + len(loss_seeds)
    checks = [
        _check(
            "nan_or_inf",
            "fail" if nan_value else "pass",
            nan_value,
            "count",
            0,
            "non-finite rewards, parameters and dev scores, plus seeds with a non-finite loss"
            + (f" (loss: seeds {loss_seeds})" if loss_seeds else ""),
        )
    ]

    largest = _extreme([s["parameters"]["max_abs"] for s in seeds], max)
    checks.append(
        _check(
            "exploding_values",
            "not_run"
            if largest is None
            else "fail"
            if largest > thresholds.max_param_abs
            else "pass",
            largest,
            "absolute parameter value",
            thresholds.max_param_abs,
            "largest absolute policy or value parameter after any update, over seeds",
        )
    )

    for check_id, key, stat, pick, threshold, unit, worse in (
        ("approx_kl", "approx_kl", "max", max, thresholds.kl_warning, "nats", "above"),
        (
            "clip_fraction",
            "clip_fraction",
            "mean",
            max,
            thresholds.clip_fraction_warning,
            "fraction",
            "above",
        ),
        (
            "entropy_collapse",
            "action_std",
            "last",
            min,
            thresholds.min_action_std,
            "logit standard deviation",
            "below",
        ),
    ):
        value = _extreme([s[key][stat] for s in seeds], pick)
        bad = value is not None and (value > threshold if worse == "above" else value < threshold)
        checks.append(
            _check(
                check_id,
                "not_run" if value is None else "warn" if bad else "pass",
                value,
                unit,
                threshold,
                f"{'largest' if pick is max else 'smallest'} per-seed {key} ({stat} over "
                f"updates); warn when {worse} the threshold",
            )
        )

    growth = []
    for s in seeds:
        first, last = s["value_loss"]["first"], s["value_loss"]["last"]
        if first is not None and last is not None and first > 0:
            growth.append(last / first)
    largest_growth = max(growth) if growth else None
    checks.append(
        _check(
            "value_loss",
            "fail"
            if loss_seeds
            else "not_run"
            if largest_growth is None
            else "warn"
            if largest_growth > VALUE_LOSS_GROWTH_WARNING
            else "pass",
            largest_growth,
            "ratio of the last to the first update's value loss",
            VALUE_LOSS_GROWTH_WARNING,
            "largest per-seed growth of the value loss; a growing value loss means the "
            "critic is diverging",
        )
    )

    deltas = []
    for s in seeds:
        scores = [e["score"] for e in s["dev_selection"]["evaluations"]]
        if len(scores) >= 2 and scores[0] is not None and scores[-1] is not None:
            deltas.append(scores[-1] - scores[0])
    trend = statistics.median(deltas) if deltas else None
    checks.append(
        _check(
            "reward_trend",
            "not_run" if trend is None else "warn" if trend < 0 else "pass",
            trend,
            "change in mean dev-fold log growth",
            0.0,
            "median over seeds of the deterministic policy's final minus initial dev score; "
            "a training diagnostic, not a strategy comparison",
        )
    )

    spread = report["seed_spread"]
    checks.append(
        _check(
            "seed_spread",
            "warn" if spread["n"] < MIN_SEEDS_FOR_SPREAD else "pass",
            spread["std"],
            "standard deviation of the selected dev score across seeds",
            None,
            f"{spread['n']} seeds, every one counted (the dev scores themselves stay in the "
            f"run's training report, never in a published view); fewer than "
            f"{MIN_SEEDS_FOR_SPREAD} seeds cannot show a spread",
        )
    )

    return {
        **_envelope(
            run_id=report["run_id"],
            generated_at=generated_at,
            code_sha=code_sha,
            git_dirty=git_dirty,
            scenario_version=scenario_version,
            transition_status=_transition_status(report["transition"]["status"]),
            seeds=len(seeds),
        ),
        "status": status,
        "checks": checks,
    }


def failed_health_input_record(
    *,
    run_id: str,
    generated_at: str,
    code_sha: str | None,
    scenario_version: str,
    transition: TransitionInfo,
    seeds: int,
    detail: str,
    git_dirty: bool | None = None,
) -> dict[str, object]:
    return {
        **_envelope(
            run_id=run_id,
            generated_at=generated_at,
            code_sha=code_sha,
            git_dirty=git_dirty,
            scenario_version=scenario_version,
            transition_status=_transition_status(transition.status),
            seeds=seeds,
        ),
        "status": "failed",
        "checks": [
            _check(check_id, "not_run", None, None, None, f"training failed: {detail}")
            for check_id in CHECK_IDS
        ],
    }


def write_health_input(record: dict, root: Path) -> Path:
    """Write `<root>/rl_training_health/<run_id>.json` once (strict JSON)."""

    directory = Path(root) / HEALTH_INPUT_ARTIFACT
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['run_id']}.json"
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path
