"""
The comparison harness: every method on the same dates, assets, costs and
constraints.

`run_comparison` builds, for every (scenario, seed) episode of one fold, a
fresh environment with the same window, schedule, transition and
observation builder for every method, runs each method through
`replay.rollout`, and then `check_fairness` verifies the records agree on
everything but the policy: scenario fingerprint, window, configuration,
transition and reward identity, booked sessions and cost rates.

Publication rule. Until the injected transition is owner-approved, results
are engineering facts only: `engineering_report` contains no performance
figure, and `comparison_table` raises `UnapprovedTransitionError`. Once it
is approved, the table reports the mean episode log growth (the sum of the
D8 rewards) and trade and fallback counts per method and scenario. Risk and
performance metrics (drawdown, CAGR, Sharpe) are owner-gated formulas and
join the table only from the approved modules (`portfolio.descriptive`,
T5c); the episodes' engineering diagnostics never do. Learned methods
trained with several seeds form a group, reported as a distribution over
seeds, never as the best seed.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from stock_agent.rl.env import AllocationEnv, EnvConfig
from stock_agent.rl.forecasts import ForecastUnavailableError
from stock_agent.rl.observation import ObservationBuilder, ScenarioLoader
from stock_agent.rl.policies import (
    InverseVolatility,
    MeanVariance,
    buy_and_hold,
    equal_weight,
    fixed_balanced,
)
from stock_agent.rl.replay import EpisodeRecord, rollout
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.scenarios import generate_scenario
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import PortfolioStep, TransitionInfo

REPORT_SCHEMA = "rl_engineering_report"
REPORT_SCHEMA_VERSION = "1.0.0"

Episode = tuple[str, int]


class FairnessError(AssertionError):
    """Two methods were not run under identical conditions."""


class UnapprovedTransitionError(RuntimeError):
    """Comparison tables need an owner-approved transition."""


@dataclass(frozen=True)
class PolicyEntry:
    name: str
    factory: Callable[[], object]
    kind: str = "rule"
    group: str | None = None


def baseline_entries() -> list[PolicyEntry]:
    return [
        PolicyEntry("buy_and_hold", buy_and_hold),
        PolicyEntry("equal_weight", equal_weight),
        PolicyEntry("fixed_balanced", fixed_balanced),
        PolicyEntry("inverse_volatility", InverseVolatility),
        PolicyEntry("mean_variance", MeanVariance, kind="optimizer"),
    ]


def _label(info: TransitionInfo) -> str:
    return "approved_transition" if info.approved else info.status


def check_fairness(records: dict[str, dict[Episode, EpisodeRecord]]) -> None:
    names = sorted(records)
    if not names:
        return
    reference = records[names[0]]
    for name in names[1:]:
        if set(records[name]) != set(reference):
            raise FairnessError(f"{name} ran different episodes from {names[0]}.")
        for episode, record in records[name].items():
            other = reference[episode]
            if record.identity != other.identity:
                raise FairnessError(f"{name} and {names[0]} differ in setup on {episode}.")
            if [r.session for r in record.ledger] != [r.session for r in other.ledger]:
                raise FairnessError(f"{name} booked different sessions on {episode}.")
            if [d.decision_session for d in record.decisions] != [
                d.decision_session for d in other.decisions
            ]:
                raise FairnessError(f"{name} decided on different sessions on {episode}.")
            for mine, theirs in zip(record.ledger, other.ledger, strict=True):
                if not np.array_equal(mine.cost_rates, theirs.cost_rates):
                    raise FairnessError(f"{name} paid different cost rates on {episode}.")


def _valid(target) -> bool:
    return target is None or (
        bool(np.all(np.isfinite(target)))
        and bool(np.all(target >= 0))
        and abs(float(target.sum()) - 1.0) <= 1e-9
    )


def _distribution(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


@dataclass
class ComparisonRun:
    fold: FoldSpec
    entries: list[PolicyEntry]
    records: dict[str, dict[Episode, EpisodeRecord]]
    transition_info: TransitionInfo
    env_config: EnvConfig
    unavailable: dict[str, str] = field(default_factory=dict)

    def _header(self) -> dict[str, object]:
        return {
            "label": _label(self.transition_info),
            "transition": self.transition_info.to_dict(),
            "reward": REWARD_INFO.to_dict(),
            "fold": self.fold.to_dict(),
            "env": self.env_config.to_dict(),
        }

    def engineering_report(self) -> dict[str, object]:
        """Facts about the runs: no performance figure of any method."""

        methods: dict[str, object] = {}
        for entry in self.entries:
            if entry.name not in self.records:
                continue
            episodes = list(self.records[entry.name].values())
            reasons = Counter(
                reason
                for record in episodes
                for d in record.decisions
                for reason in d.fallback_reasons
            )
            methods[entry.name] = {
                "kind": entry.kind,
                "group": entry.group,
                "episodes": len(episodes),
                "decisions": sum(len(record.decisions) for record in episodes),
                "trades": sum(row.cost > 0 for record in episodes for row in record.ledger),
                "all_finite": all(
                    bool(np.all(np.isfinite(record.wealth_path)) and np.all(record.wealth_path > 0))
                    for record in episodes
                ),
                "invalid_targets": sum(
                    not _valid(d.target) for record in episodes for d in record.decisions
                ),
                "fallback_decisions": sum(
                    bool(d.fallback_reasons) for record in episodes for d in record.decisions
                ),
                "fallback_reasons": dict(sorted(reasons.items())),
            }
        return {
            "schema": REPORT_SCHEMA,
            "schema_version": REPORT_SCHEMA_VERSION,
            **self._header(),
            "fairness": "passed",
            "comparison_tables": "available"
            if self.transition_info.approved
            else "withheld: the injected transition is not owner-approved",
            "methods": methods,
            "unavailable": dict(self.unavailable),
        }

    def comparison_table(self) -> dict[str, object]:
        if not self.transition_info.approved:
            raise UnapprovedTransitionError(
                "Strategy comparison tables are withheld until the owner approves the transition."
            )
        scenarios = sorted({scenario for scenario, _ in self.fold.episodes()})
        rows = []
        means: dict[tuple[str, str], float] = {}
        for name, by_episode in self.records.items():
            for scenario in scenarios:
                records = [
                    record
                    for (episode_scenario, _), record in sorted(by_episode.items())
                    if episode_scenario == scenario
                ]
                row = {
                    "method": name,
                    "scenario": scenario,
                    "episodes": len(records),
                    "log_growth": statistics.fmean(r.total_reward for r in records),
                    "trades": statistics.fmean(
                        sum(row.cost > 0 for row in r.ledger) for r in records
                    ),
                    "fallbacks": statistics.fmean(
                        sum(bool(d.fallback_reasons) for d in r.decisions) for r in records
                    ),
                }
                means[(name, scenario)] = row["log_growth"]
                rows.append(row)
        groups: dict[str, object] = {}
        for group in sorted({e.group for e in self.entries if e.group}):
            members = [e.name for e in self.entries if e.group == group and e.name in self.records]
            groups[group] = {
                "members": members,
                "metric": "mean log growth per member (training seed)",
                "by_scenario": {
                    scenario: _distribution([means[(m, scenario)] for m in members])
                    for scenario in scenarios
                },
            }
        return {**self._header(), "rows": rows, "groups": groups}


def run_comparison(
    entries: list[PolicyEntry],
    fold: FoldSpec,
    *,
    builder: ObservationBuilder,
    transition: PortfolioStep,
    transition_info: TransitionInfo,
    env_config: EnvConfig | None = None,
    loader: ScenarioLoader = generate_scenario,
) -> ComparisonRun:
    env_config = env_config or EnvConfig()
    data = {episode: loader(*episode) for episode in fold.episodes()}
    records: dict[str, dict[Episode, EpisodeRecord]] = {}
    unavailable: dict[str, str] = {}
    for entry in entries:
        try:
            policy = entry.factory()
        except ForecastUnavailableError as error:
            unavailable[entry.name] = str(error)
            continue
        records[entry.name] = {}
        for episode in fold.episodes():
            env = AllocationEnv(
                data[episode],
                start=fold.start,
                stop=fold.stop,
                transition=transition,
                transition_info=transition_info,
                builder=builder,
                config=env_config,
            )
            records[entry.name][episode] = rollout(env, policy)
    check_fairness(records)
    return ComparisonRun(
        fold=fold,
        entries=list(entries),
        records=records,
        transition_info=transition_info,
        env_config=env_config,
        unavailable=unavailable,
    )
