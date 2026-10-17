"""
The comparison artifact (`ppo_comparison` 1.1.0) and the `backtest_summary`
health inputs it feeds.

The artifact is written once to `<root>/<artifact_id>/comparison.json`
(locally `reports/comparison/`, ignored by Git), as strict JSON with a
`content_sha256` over the canonical JSON of every other field, so a reader
can refuse an edited file. It holds:

- labels: "Synthetic scenarios only — not evidence of real investment
  performance", "engineering evaluation", the D4 (a) cash label, and the
  provisional D2 rates;
- the registered configuration and its hash, the fold, the look number and
  the log entry from the configuration's private evaluation log, the frozen selection's hash, the code version, the
  approved transition and reward, and each strategy's registry stage (at most
  `development_evaluated`; synthetic runs are `engineering_only`);
- `rows`: every (constraint set, strategy, member seed, scenario, scenario
  seed) with its approved measures and its 0x, 1x and 2x cost sensitivity;
- `summary`: per (constraint set, strategy, scenario), the spread of each
  measure across the scenario's test paths (and, for learned strategies,
  across seeds): n, min, quartiles, median and max (linear interpolation);
- `seed_variability`: for PPO and CEM, each seed's mean terminal wealth over
  the scenario's paths, every seed listed, and their min, quartiles and max:
  never only the best seed;
- `build`: per (constraint set, strategy), the executed weights at the build,
  uninvested cash, incomplete constructions and the sessions until fully
  built (D5, confirmed 2026-10-17);
- `fairness` and `selection`: what the contract and the no-test-selection
  check verified.

No CAGR, Sharpe ratio or confidence interval (group D is not approved), no
ranking and no claim of skill or profitability.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stock_agent.comparison.measures import UNITS
from stock_agent.comparison.protocol import (
    HEADLINE_COST_MULTIPLIER,
    PROTOCOL,
    STRATEGIES,
    ComparisonConfig,
)
from stock_agent.portfolio import backtest_summary
from stock_agent.portfolio.engine import CASH_RETURN_LABEL
from stock_agent.portfolio.schedule import execution_lag
from stock_agent.registry.stages import FLOOR, STAGES
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.scenarios import ASSET_NAMES, BASE_COST_RATES, CASH_INDEX, GENERATOR_VERSION
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import APPROVED_TRANSITION

ARTIFACT_SCHEMA = "ppo_comparison"
ARTIFACT_SCHEMA_VERSION = "1.1.0"
ARTIFACT_FILE = "comparison.json"
SCOPE_LABEL = "Synthetic scenarios only — not evidence of real investment performance"
EVALUATION_LABEL = "engineering evaluation"
EVIDENCE_STATUS = "synthetic_engineering_only"
MAX_STAGE = "development_evaluated"
STAGE_BASIS = (
    "Synthetic scenarios only: the registry's development_evaluated stage needs a registered, "
    "point-in-time research panel, so every strategy here stays at engineering_only."
)
COSTS_LABEL = (
    "D2 provisional all-in rates (5 bp the equity and bond funds, 2 bp the short-Treasury fund, "
    "0 the cash account), times each scenario's cost regime"
)
NOT_REPORTED = {
    "measures": ["CAGR", "Sharpe ratio", "confidence intervals"],
    "reason": "group D formulas of t5_portfolio_formulas.md are not approved",
}
NO_CLAIM = (
    "Nothing here ranks the strategies or claims skill or profitability: the scenarios are "
    "synthetic and the comparison is an engineering evaluation of the simulator and protocol."
)


def multiplier_key(multiplier: float) -> str:
    return f"{multiplier:g}x"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(document: Mapping[str, object]) -> str:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def describe(values: Sequence[float]) -> dict[str, float | int] | None:
    """n, min, quartiles, median and max (linear interpolation); None for no value."""

    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(array.max()),
    }


def seed_values(
    rows: Sequence[Mapping[str, object]], constraint_set: str, strategy: str, scenario: str
) -> dict[int, float]:
    """Each member seed's mean terminal wealth over one scenario's evaluation paths."""

    by_seed: dict[int, list[float]] = {}
    for row in rows:
        if (row["constraint_set"], row["strategy"], row["scenario"]) == (
            constraint_set,
            strategy,
            scenario,
        ):
            by_seed.setdefault(int(row["member_seed"]), []).append(float(row["terminal_wealth"]))
    return {seed: statistics.fmean(values) for seed, values in sorted(by_seed.items())}


def _summary(rows: list[dict]) -> dict[str, object]:
    def spread(read: Callable[[dict], float]) -> dict | None:
        return describe([read(row) for row in rows])

    reasons: Counter = Counter()
    for row in rows:
        reasons.update(row["fallback_reasons"])
    first = rows[0]
    return {
        "constraint_set": first["constraint_set"],
        "strategy": first["strategy"],
        "scenario": first["scenario"],
        "episodes": len(rows),
        "terminal_wealth": spread(lambda r: r["terminal_wealth"]),
        "total_return": spread(lambda r: r["total_return"]),
        "drawdown": {
            "value": spread(lambda r: r["drawdown"]["value"]),
            "incomplete_histories": sum(not r["drawdown"]["history_complete"] for r in rows),
            "recovered": sum(r["drawdown"]["recovered"] for r in rows),
            "duration_sessions": spread(lambda r: r["drawdown"]["duration_sessions"]),
        },
        "turnover_one_way": spread(lambda r: r["turnover_one_way"]),
        "construction_turnover_one_way": spread(lambda r: r["construction_turnover_one_way"]),
        "rebalance_turnover_one_way": spread(lambda r: r["rebalance_turnover_one_way"]),
        "turnover_per_rebalance": describe(
            [r["turnover_per_rebalance"] for r in rows if r["turnover_per_rebalance"] is not None]
        ),
        "span_sessions": spread(lambda r: r["span_sessions"]),
        "traded_notional": spread(lambda r: r["traded_notional"]),
        "costs_paid": spread(lambda r: r["costs_paid"]),
        "cost_sensitivity": {
            key: spread(lambda r, key=key: r["cost_sensitivity"][key]["terminal_wealth"])
            for key in first["cost_sensitivity"]
        },
        "interventions": {
            name: sum(r["interventions"][name] for r in rows)
            for name in ("targets", "cap_hits", "turnover_limit_hits")
        }
        | {
            "distance_one_way_total": float(
                sum(r["interventions"]["distance_one_way_total"] for r in rows)
            )
        },
        "fallback_events": sum(r["fallback_events"] for r in rows),
        "fallback_reasons": dict(sorted(reasons.items())),
        "invalid_targets": sum(r["invalid_targets"] for r in rows),
    }


def _build_summary(rows: list[dict]) -> dict[str, object]:
    builds = [row["build"] for row in rows if row["build"] is not None]
    completed = [b["sessions_until_fully_built"] for b in builds if b["fully_built_session"]]
    executed = np.asarray([b["executed_weights"] for b in builds], dtype=np.float64)
    first = rows[0]
    return {
        "constraint_set": first["constraint_set"],
        "strategy": first["strategy"],
        "episodes": len(rows),
        "builds_executed": len(builds),
        "executed_weights": {
            "assets": list(ASSET_NAMES),
            "median": [float(v) for v in np.median(executed, axis=0)] if builds else None,
            "min": [float(v) for v in executed.min(axis=0)] if builds else None,
            "max": [float(v) for v in executed.max(axis=0)] if builds else None,
        },
        "uninvested_cash": describe([b["uninvested_cash"] for b in builds]),
        "incomplete_construction": sum(b["incomplete_construction"] for b in builds),
        "never_fully_built": sum(b["fully_built_session"] is None for b in builds),
        "sessions_until_fully_built": describe(completed),
        "delayed_builds": sum(b["delay_sessions"] > 0 for b in builds),
        "terminal_cash_weight": describe([row["terminal_cash_weight"] for row in rows]),
    }


def _grouped(rows: Sequence[dict], *keys: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def assemble(
    *,
    config: ComparisonConfig,
    fold: FoldSpec,
    look: Mapping[str, object],
    selection_sha256: str,
    selection: Mapping[str, object],
    code: Mapping[str, object],
    generated_at: datetime,
    artifact_id: str,
    rows: list[dict],
    contract: Mapping[str, object],
) -> dict[str, object]:
    """The artifact document (module docstring), with its `content_sha256`."""

    sets = [s.name for s in config.constraint_sets]
    scenarios = list(fold.scenario_ids)
    by_group = _grouped(rows, "constraint_set", "strategy", "scenario")
    summary = [
        _summary(by_group[(cs, key, scenario)])
        for cs in sets
        for key in config.strategies
        for scenario in scenarios
    ]
    variability = []
    for cs in sets:
        for key in config.strategies:
            if not STRATEGIES[key].is_learned:
                continue
            for scenario in scenarios:
                values = seed_values(rows, cs, key, scenario)
                variability.append(
                    {
                        "constraint_set": cs,
                        "strategy": key,
                        "scenario": scenario,
                        "statistic": "mean terminal wealth over the scenario's evaluation "
                        "paths, per training seed",
                        "seeds": sorted(values),
                        "per_seed_mean_terminal_wealth": [values[s] for s in sorted(values)],
                        "distribution": describe(list(values.values())),
                    }
                )
    by_strategy = _grouped(rows, "constraint_set", "strategy")
    builds = [_build_summary(by_strategy[(cs, key)]) for cs in sets for key in config.strategies]
    members = selection["members"]
    strategies = []
    for key in config.strategies:
        spec = STRATEGIES[key]
        strategies.append(
            {
                "key": key,
                "label": spec.label,
                "kind": spec.kind,
                "rule": spec.rule,
                "learned": spec.learned,
                "member_seeds": None
                if not spec.is_learned
                else {
                    cs: [
                        m["seed"]
                        for m in members
                        if m["kind"] == spec.learned and m["constraint_set"] == cs
                    ]
                    for cs in sets
                },
                "registry_stage": FLOOR,
                "stage_basis": STAGE_BASIS,
            }
        )
    variants = {
        kind: {
            "members": sum(1 for m in members if m["kind"] == kind),
            "checkpoints_compared_on_dev": sum(
                int(m["selection"]["checkpoints_compared"]) for m in members if m["kind"] == kind
            ),
            "candidates_evaluated_on_train": sum(
                int(m.get("variants_evaluated", 0)) for m in members if m["kind"] == kind
            ),
        }
        for kind in config.learned_kinds()
    }
    document = {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "artifact_id": artifact_id,
        "generated_at": generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "synthetic": True,
        "evidence_status": EVIDENCE_STATUS,
        "labels": {
            "scope": SCOPE_LABEL,
            "evaluation": EVALUATION_LABEL,
            "cash": CASH_RETURN_LABEL,
            "costs": COSTS_LABEL,
            "claims": NO_CLAIM,
        },
        "not_reported": NOT_REPORTED,
        "code": {"git_sha": code.get("git_sha"), "git_dirty": code.get("git_dirty")},
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "selection_sha256": selection_sha256,
        "fold": fold.to_dict(),
        "look": {
            "number": look["look_number"],
            "repeat": look["repeat"],
            "log_entry_sha256": look["entry_sha256"],
            "log_sequence": look["sequence"],
            "log_file": f"{look['config_sha256']}.jsonl",
            "test_looks_recorded": look["test_looks"],
            "dev_looks_recorded": look["dev_looks"],
        },
        "transition": APPROVED_TRANSITION.to_dict(),
        "reward": REWARD_INFO.to_dict(),
        "execution_lag": execution_lag(),
        "schedule": {
            "rebalance_every": config.rl.env.rebalance_every,
            "rebalance_offset": config.rl.env.rebalance_offset,
            "build": "100% cash until the common build close, one session after the first "
            "decision (D3)",
        },
        "assets": list(ASSET_NAMES),
        "cash_index": CASH_INDEX,
        "base_cost_rates": dict(zip(ASSET_NAMES, BASE_COST_RATES, strict=True)),
        "cost_multipliers": [multiplier_key(m) for m in config.cost_multipliers],
        "headline_cost_multiplier": multiplier_key(HEADLINE_COST_MULTIPLIER),
        "scenario_generator_version": GENERATOR_VERSION,
        "constraint_sets": [s.to_dict() for s in config.constraint_sets],
        "strategies": strategies,
        "units": dict(UNITS),
        "variants": variants,
        "fairness": dict(contract),
        "rows": rows,
        "summary": summary,
        "seed_variability": variability,
        "build": builds,
    }
    for strategy in strategies:
        if STAGES.index(strategy["registry_stage"]) > STAGES.index(MAX_STAGE):
            raise ValueError("A comparison shows registry stages up to development_evaluated.")
    document["content_sha256"] = content_sha256(document)
    return document


def write_artifact(document: Mapping[str, object], root: Path) -> Path:
    """Write the artifact once to `<root>/<artifact_id>/comparison.json`."""

    directory = Path(root) / str(document["artifact_id"])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / ARTIFACT_FILE
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


HEALTH_UNITS = {
    "turnover": (
        "largest one-way turnover of a single execution over every evaluation episode, the build "
        "included (a full switch is 1), rounded to 12 decimals (the engine's dust, 1e-12), so "
        "floating-point residue at a binding turnover limit is not read as a breach"
    ),
    "costs_bps": (
        "largest, over the evaluation episodes, summed trading cost of every execution, each in "
        "basis points of its pre-trade wealth (not annualized)"
    ),
    "max_weight": (
        "largest non-cash weight after a trade or at a valued close, any episode; drift above "
        "the cap between rebalances is allowed and reported (D5)"
    ),
    "max_drawdown": (
        "largest F6.5 maximum drawdown over the evaluation episodes whose history is complete; "
        "episodes with unvalued sessions are left out and counted in assumptions"
    ),
}


def health_records(
    document: Mapping[str, object], *, clock: Callable[[], datetime]
) -> list[dict[str, object]]:
    """
    One `backtest_summary` (agreed v1.0) per (constraint set, strategy), each
    measure the worst over the strategy's 1x evaluation episodes (all seeds),
    with the optional constraint-violation and fallback counts.
    """

    sets = {s["name"]: s for s in document["constraint_sets"]}
    by_strategy = _grouped(document["rows"], "constraint_set", "strategy")
    builds = {(b["constraint_set"], b["strategy"]): b for b in document["build"]}
    fold = document["fold"]["role"]
    records = []
    for (cs, key), rows in by_strategy.items():
        complete = [r["drawdown"]["value"] for r in rows if r["drawdown"]["history_complete"]]
        reasons: Counter = Counter()
        for row in rows:
            reasons.update(row["fallback_reasons"])
        generated_at = clock()
        records.append(
            {
                "artifact": backtest_summary.ARTIFACT,
                "schema_version": backtest_summary.SCHEMA_VERSION,
                "generated_at": generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "code_sha": document["code"]["git_sha"],
                "synthetic": True,
                "run_id": f"{document['artifact_id']}-{cs}-{key}",
                "policy": f"{document['config']['name']}/{fold}/{cs}/{key}",
                "evidence_status": EVIDENCE_STATUS,
                "transition_status": backtest_summary.TRANSITION_STATUS,
                "constraints": {
                    "max_weight": sets[cs]["max_weight"],
                    "max_turnover": sets[cs]["max_turnover"],
                    "max_cost_bps": None,
                    "max_drawdown": None,
                },
                "measures": {
                    "turnover": round(max(r["largest_execution_turnover"] for r in rows), 12),
                    "costs_bps": max(r["costs_bps_summed"] for r in rows),
                    "max_weight": max(r["max_weight"] for r in rows),
                    "max_drawdown": max(complete) if complete else None,
                },
                "units": dict(HEALTH_UNITS),
                "assumptions": {
                    "cash_return": CASH_RETURN_LABEL,
                    "cost_rates": document["base_cost_rates"],
                    "cost_multiplier": 1.0,
                    "lag": document["execution_lag"],
                    "schedule": document["schedule"],
                    "episodes": len(rows),
                    "episodes_with_incomplete_history": len(rows) - len(complete),
                    "comparison_artifact": document["artifact_id"],
                    "scope": SCOPE_LABEL,
                },
                "constraint_violations": sum(r["invalid_targets"] for r in rows),
                "fallback_events": sum(r["fallback_events"] for r in rows),
                "fallback_reasons": dict(sorted(reasons.items())),
                "build": builds[(cs, key)],
                "status": "complete",
                "status_reason": None,
            }
        )
    return records


def write_health_inputs(records: Sequence[Mapping[str, object]], root: Path) -> list[Path]:
    return [backtest_summary.write_summary(record, root) for record in records]
