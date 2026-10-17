"""
The registered comparison configuration: what is compared, under which
constraint sets and cost multipliers, with which learned seeds, before any
result exists.

A configuration is a validated, frozen `ComparisonConfig` loaded from YAML
(`configs/comparison/`). Unknown keys are errors, and `sha256()` identifies
it in the artifact and in the evaluation log, where every look at a fold is
counted per configuration.

Registered rules (validated here, so a configuration cannot weaken them):

- `cost_multipliers` are D2's 0x, 1x and 2x (`SENSITIVITY_MULTIPLIERS`),
  never chosen; 1x is the headline run and 0x and 2x replay its decisions.
- `constraint_sets` hold at least two sets: `plain`, the D5 defaults (cap
  1.0, no turnover limit), and at least one stricter set. Each set applies
  identically to every strategy (D5 as confirmed by the owner on
  2026-10-17), and learned policies train and select under the set they are
  evaluated in. The RL section may not register constraints of its own.
- Learned strategies are reported over at least `MIN_LEARNED_SEEDS` seeds:
  PPO over `rl.training.seeds`, CEM over `cem_seeds` (each member uses the
  RL section's CEM settings with its own seed).
- The `rl` section is a complete RL run configuration (`configs/rl/`
  schema, without `schema_version` and `name`): the fold plan, features,
  schedule, PPO and CEM settings and the health gate shared by every
  strategy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from stock_agent.portfolio.transition import SENSITIVITY_MULTIPLIERS, Constraints
from stock_agent.rl.config import CONFIG_SCHEMA_VERSION, RunConfig, run_config_from_dict
from stock_agent.rl.env import EnvConfig
from stock_agent.rl.policies import (
    ConstantMix,
    InverseVolatility,
    MeanVariance,
    buy_and_hold,
    equal_weight,
    fixed_balanced,
)
from stock_agent.rl.scenarios import ASSET_NAMES

SCHEMA_VERSION = 1
PROTOCOL = "registered_synthetic_comparison"
MIN_LEARNED_SEEDS = 5
PLAIN = "plain"
HEADLINE_COST_MULTIPLIER = 1.0
_NAME = re.compile(r"[a-z][a-z0-9_]*")


class ProtocolError(ValueError):
    """A configuration breaks a registered rule of the comparison protocol."""


@dataclass(frozen=True)
class ConstraintSet:
    """One registered D5 set, applied identically to every strategy."""

    name: str
    max_weight: float = 1.0
    max_turnover: float | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ProtocolError(f"Constraint set name {self.name!r} must be a lowercase word.")
        Constraints(0, self.max_weight, self.max_turnover)  # validates the values

    @property
    def is_plain_default(self) -> bool:
        return self.max_weight == 1.0 and self.max_turnover is None

    def env(self, base: EnvConfig) -> EnvConfig:
        return dataclasses.replace(base, max_weight=self.max_weight, max_turnover=self.max_turnover)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_weight": self.max_weight,
            "max_turnover": self.max_turnover,
        }


@dataclass(frozen=True)
class StrategySpec:
    """
    A registered strategy. `label` and `rule` are shown in the Lab; `kind`
    is rule, optimizer or learned; learned strategies name the trainer
    (`learned`: "cem" or "ppo") and form a group reported over seeds.
    """

    key: str
    label: str
    kind: str
    rule: str
    learned: str | None = None

    @property
    def is_learned(self) -> bool:
        return self.learned is not None


def _all_short_treasury() -> ConstantMix:
    weights = np.zeros(len(ASSET_NAMES))
    weights[ASSET_NAMES.index("short_treasury_fund")] = 1.0
    return ConstantMix("all_shv", weights)


STRATEGIES: dict[str, StrategySpec] = {
    spec.key: spec
    for spec in (
        StrategySpec(
            "buy_and_hold",
            "60/30/10, held without rebalancing",
            "rule",
            "Builds 60% equity fund, 30% bond fund and 10% short-Treasury fund once, then "
            "holds the drifting position.",
        ),
        StrategySpec(
            "equal_weight",
            "Equal weight",
            "rule",
            "One third in each fund, no cash, rebalanced at every decision.",
        ),
        StrategySpec(
            "fixed_balanced",
            "60/30/10, rebalanced",
            "rule",
            "60% equity fund, 30% bond fund, 10% short-Treasury fund at every decision.",
        ),
        StrategySpec(
            "inverse_volatility",
            "Inverse volatility",
            "rule",
            "Each fund in proportion to one over its 60-session volatility; no cash.",
        ),
        StrategySpec(
            "mean_variance",
            "Mean-variance",
            "optimizer",
            "Long-only mean-variance (risk aversion 5) on 60-session sample estimates, "
            "solved exactly; cash riskless.",
        ),
        StrategySpec(
            "all_shv",
            "100% short-Treasury fund",
            "rule",
            "The short-Treasury fund held as a fund (2 bp to trade, its own return). "
            "Uninvested cash earns 0% (D4 (a), synthetic engineering only).",
        ),
        StrategySpec(
            "linear_softmax_cem",
            "Linear-softmax (CEM)",
            "learned",
            "Five-parameter learned baseline trained by the cross-entropy method; "
            "checkpoint chosen on development data only.",
            learned="cem",
        ),
        StrategySpec(
            "ppo",
            "PPO without forecast inputs",
            "learned",
            "PPO's deterministic policy (the Gaussian mean); checkpoint chosen on "
            "development data only.",
            learned="ppo",
        ),
        StrategySpec(
            "ppo_guarded",
            "PPO with the health gate",
            "learned",
            "The same PPO checkpoints behind the health gate: the registered fallback "
            "decides on stale, non-finite or out-of-range inputs.",
            learned="ppo",
        ),
    )
}

RULE_FACTORIES: dict[str, Callable[[], object]] = {
    "buy_and_hold": buy_and_hold,
    "equal_weight": equal_weight,
    "fixed_balanced": fixed_balanced,
    "inverse_volatility": InverseVolatility,
    "mean_variance": MeanVariance,
    "all_shv": _all_short_treasury,
}


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    strategies: tuple[str, ...]
    constraint_sets: tuple[ConstraintSet, ...]
    cost_multipliers: tuple[float, ...]
    cem_seeds: tuple[int, ...]
    rl: RunConfig

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ProtocolError(f"Configuration name {self.name!r} must be a lowercase word.")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ProtocolError("A comparison needs distinct strategies.")
        unknown = sorted(set(self.strategies) - set(STRATEGIES))
        if unknown:
            raise ProtocolError(f"Unknown strategies: {unknown}")
        names = [s.name for s in self.constraint_sets]
        if len(names) < 2 or len(set(names)) != len(names):
            raise ProtocolError("Register at least two distinct constraint sets.")
        plain = [s for s in self.constraint_sets if s.name == PLAIN]
        if not plain or not plain[0].is_plain_default:
            raise ProtocolError(
                f"The {PLAIN!r} set must be the D5 defaults (cap 1.0, no turnover limit)."
            )
        if not any(not s.is_plain_default for s in self.constraint_sets):
            raise ProtocolError("Register at least one stricter constraint set.")
        if tuple(self.cost_multipliers) != SENSITIVITY_MULTIPLIERS:
            raise ProtocolError(
                f"Cost multipliers are D2's {SENSITIVITY_MULTIPLIERS}, never chosen; "
                f"got {tuple(self.cost_multipliers)}."
            )
        env = self.rl.env
        if env.max_weight != 1.0 or env.max_turnover is not None:
            raise ProtocolError(
                "The RL section may not register constraints: every set comes from "
                "constraint_sets and applies to every strategy."
            )
        learned = {STRATEGIES[key].learned for key in self.strategies} - {None}
        if "ppo" in learned:
            _require_seeds(self.rl.training.seeds, "PPO (rl.training.seeds)")
        if "cem" in learned:
            _require_seeds(self.cem_seeds, "CEM (cem_seeds)")

    @property
    def ppo_seeds(self) -> tuple[int, ...]:
        return tuple(self.rl.training.seeds)

    def learned_kinds(self) -> tuple[str, ...]:
        kinds = {STRATEGIES[key].learned for key in self.strategies} - {None}
        return tuple(sorted(kinds))

    def seeds_for(self, kind: str) -> tuple[int, ...]:
        return self.ppo_seeds if kind == "ppo" else tuple(self.cem_seeds)

    def entry_names(self) -> tuple[str, ...]:
        """Every evaluated entry, in registered order: each rule, each learned seed."""

        names: list[str] = []
        for key in self.strategies:
            spec = STRATEGIES[key]
            if spec.is_learned:
                names.extend(entry_name(key, seed) for seed in self.seeds_for(spec.learned))
            else:
                names.append(key)
        return tuple(names)

    def constraint_set(self, name: str) -> ConstraintSet:
        for constraint_set in self.constraint_sets:
            if constraint_set.name == name:
                return constraint_set
        raise KeyError(name)

    def run_config(self, constraint_set: ConstraintSet) -> RunConfig:
        """The RL run configuration of one set: the shared settings with its D5 set."""

        return dataclasses.replace(self.rl, env=constraint_set.env(self.rl.env))

    def to_dict(self) -> dict[str, object]:
        rl = self.rl.to_dict()
        rl.pop("schema_version")
        rl.pop("name")
        rl["cem"] = {k: v for k, v in rl["cem"].items() if k != "seed"}  # cem_seeds rules
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "name": self.name,
            "strategies": list(self.strategies),
            "constraint_sets": [s.to_dict() for s in self.constraint_sets],
            "cost_multipliers": list(self.cost_multipliers),
            "cem_seeds": list(self.cem_seeds),
            "rl": rl,
        }

    def sha256(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entry_name(strategy: str, seed: int | None = None) -> str:
    """An evaluated entry's name: the strategy, and its training seed when learned."""

    return strategy if seed is None else f"{strategy}/seed_{seed}"


def _require_seeds(seeds, what: str) -> None:
    seeds = tuple(seeds)
    if len(seeds) < MIN_LEARNED_SEEDS or len(set(seeds)) != len(seeds):
        raise ProtocolError(
            f"{what} needs at least {MIN_LEARNED_SEEDS} distinct seeds, reported as a "
            f"distribution; got {list(seeds)}."
        )


TOP_LEVEL = {
    "schema_version",
    "protocol",
    "name",
    "strategies",
    "constraint_sets",
    "cost_multipliers",
    "cem_seeds",
    "rl",
}


def comparison_config_from_dict(record: dict) -> ComparisonConfig:
    unknown = sorted(set(record) - TOP_LEVEL)
    if unknown:
        raise ProtocolError(f"Unknown top-level keys: {unknown}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(
            f"Unsupported schema_version {record.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    if record.get("protocol", PROTOCOL) != PROTOCOL:
        raise ProtocolError(f"Only the {PROTOCOL!r} protocol is defined.")
    rl = dict(record["rl"])
    clash = sorted({"schema_version", "name"} & set(rl))
    if clash:
        raise ProtocolError(f"The rl section takes its {clash} from the top level.")
    if "seed" in (rl.get("cem") or {}):
        raise ProtocolError("CEM seeds are registered in cem_seeds, not rl.cem.seed.")
    name = str(record["name"])
    run_config = run_config_from_dict({"schema_version": CONFIG_SCHEMA_VERSION, "name": name, **rl})
    sets = []
    for entry in record["constraint_sets"]:
        extra = sorted(set(entry) - {"name", "max_weight", "max_turnover"})
        if extra:
            raise ProtocolError(f"Unknown constraint-set keys: {extra}")
        sets.append(
            ConstraintSet(
                name=str(entry["name"]),
                max_weight=float(entry.get("max_weight", 1.0)),
                max_turnover=None
                if entry.get("max_turnover") is None
                else float(entry["max_turnover"]),
            )
        )
    return ComparisonConfig(
        name=name,
        strategies=tuple(str(key) for key in record["strategies"]),
        constraint_sets=tuple(sets),
        cost_multipliers=tuple(float(m) for m in record["cost_multipliers"]),
        cem_seeds=tuple(int(seed) for seed in record.get("cem_seeds", ())),
        rl=run_config,
    )


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ProtocolError("A comparison configuration must be a YAML mapping.")
    return comparison_config_from_dict(record)
