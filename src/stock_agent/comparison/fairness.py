"""
The fairness contract of the registered comparison, checked on every record
before any measure is reported, and the check that no result of the
evaluation fold selected anything.

Every strategy is evaluated through one API (`replay.rollout` on
`AllocationEnv`), and `check_contract` verifies, for every strategy and
episode, that the record matches the registered configuration exactly:

- API: the record is an `EpisodeRecord` and its identity (scenario
  fingerprint, window, schedule and constraint set, transition and reward)
  is the registered environment's;
- transition: the approved `portfolio_step` (D1-D8): every booked step is
  replayed through it from the step's own inputs and must agree bit for bit
  (weights, cost, returns, turnover, traded notional), wealth compounds by
  each net return and each reward is log(1 + R_net) (D8 (a));
- timing (D1): a decision at complete session c_j executes at c_{j+1}; no
  other session carries a target;
- schedule (D7): the decisions are the registered schedule's, every
  strategy books the same sessions;
- costs (D2): the provisional rates, times the scenario's cost regime, times
  the registered sensitivity multiplier, at every booked session;
- build (D3): 100% cash until the common build close, where every
  strategy's first decision executes;
- constraint set (D5): every target passed through `feasible_target` with
  the registered set, and the transition received its result.

`check_no_test_selection` verifies that the reported learned members are
exactly the members frozen (dev-selected) before the evaluation fold was
built, with their recorded checkpoints, and that every seed distribution
is recomputed from all of them.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping

import numpy as np

from stock_agent.portfolio.schedule import execution_lag
from stock_agent.portfolio.transition import feasible_target, portfolio_step
from stock_agent.rl.env import ENV_EXECUTION_LAG, AllocationEnv, EnvConfig
from stock_agent.rl.harness import FairnessError, check_fairness
from stock_agent.rl.observation import ObservationBuilder
from stock_agent.rl.replay import EpisodeRecord
from stock_agent.rl.scenarios import BASE_COST_RATES, ScenarioData
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import APPROVED_TRANSITION

__all__ = [
    "FairnessError",
    "SelectionError",
    "check_contract",
    "check_no_test_selection",
]

Episode = tuple[str, int]
_STEP_FIELDS = ("gross_return", "cost", "net_return", "turnover_one_way", "traded_notional")


class SelectionError(AssertionError):
    """A reported result depends on the evaluation fold (selection on test results)."""


def _fail(name: str, episode: Episode, what: str) -> None:
    raise FairnessError(f"{name} on {episode}: {what}")


def _reference(
    data: ScenarioData, fold: FoldSpec, env_config: EnvConfig, builder: ObservationBuilder
) -> AllocationEnv:
    return AllocationEnv(
        data,
        start=fold.start,
        stop=fold.stop,
        transition=portfolio_step,
        transition_info=APPROVED_TRANSITION,
        builder=builder,
        config=env_config,
    )


def _check_record(
    name: str,
    episode: Episode,
    record: object,
    reference: AllocationEnv,
    cost_multiplier: float,
) -> int:
    if type(record) is not EpisodeRecord:
        _fail(name, episode, f"evaluated through another API ({type(record).__name__})")
    data = reference.data
    if record.identity != reference.identity():
        _fail(name, episode, "its scenario, window, schedule, constraint set or transition differs")
    expected_rates = tuple(rate * cost_multiplier for rate in BASE_COST_RATES)
    if not np.array_equal(data.base_cost_rates, expected_rates):
        _fail(name, episode, "its data carries other cost rates than the registered D2 rates")

    every = reference.config.rebalance_every
    first = reference.first_decision_index
    decisions = list(range(first, reference.final_index, every))
    if [d.decision_index for d in record.decisions] != decisions:
        _fail(name, episode, "decided on another schedule than the registered one")
    if [d.decision_session for d in record.decisions] != [
        int(reference.complete[j]) for j in decisions
    ]:
        _fail(name, episode, "decided on other sessions than the registered schedule's")
    expected_sessions = [int(s) for s in reference.complete[: reference.final_index + 1]]
    if [row.session for row in record.ledger] != expected_sessions:
        _fail(name, episode, "booked other sessions than the registered window's")
    if [row.index for row in record.ledger] != list(range(len(record.ledger))):
        _fail(name, episode, "its ledger is out of order")

    targets = {d.decision_index + ENV_EXECUTION_LAG: d.target for d in record.decisions}
    constraints = reference.config.constraints(data.cash_index)
    cash = np.zeros(data.n_assets)
    cash[data.cash_index] = 1.0
    checked = 0
    previous = None
    for row in record.ledger:
        target = targets.get(row.index)
        if (row.target is None) != (target is None) or (
            target is not None and not np.array_equal(row.target, target)
        ):
            _fail(name, episode, f"a target executes at ledger row {row.index}, off the lag")
        if row.index <= first and not np.array_equal(row.w_pre, cash):
            _fail(name, episode, "did not start from 100% cash at the common build (D3)")
        if previous is not None and not np.array_equal(row.w_pre, previous.w_next):
            _fail(name, episode, f"its holdings jump between rows {previous.index} and {row.index}")
        rates = data.cost_rates_at(row.session)
        if not np.array_equal(row.cost_rates, rates):
            _fail(name, episode, f"paid other cost rates at session {row.session}")
        executed = None
        if row.target is not None:
            if row.feasibility is None:
                _fail(name, episode, f"skipped the D5 feasibility map at row {row.index}")
            expected = feasible_target(row.target, row.w_pre, constraints)
            if not np.array_equal(expected.weights, row.feasibility.weights):
                _fail(name, episode, f"ran under another constraint set at row {row.index}")
            executed = np.array(row.feasibility.weights)
        elif row.feasibility is not None:
            _fail(name, episode, f"a feasibility record without a target at row {row.index}")
        step = portfolio_step(
            row.w_pre,
            executed,
            data.span_returns(row.session, row.next_session),
            rates,
            data.cash_index,
        )
        same = np.array_equal(step.w_post_trade, row.w_post_trade) and np.array_equal(
            step.w_next, row.w_next
        )
        same = same and all(getattr(step, f) == getattr(row, f) for f in _STEP_FIELDS)
        if not same:
            _fail(name, episode, f"row {row.index} is not the approved portfolio_step's result")
        if row.wealth_after != row.wealth_before * (1.0 + row.net_return):
            _fail(name, episode, f"wealth does not compound by the net return at row {row.index}")
        if row.reward != math.log1p(row.net_return):
            _fail(name, episode, f"the reward is not log(1 + R_net) at row {row.index} (D8 (a))")
        if previous is not None and row.wealth_before != previous.wealth_after:
            _fail(name, episode, f"wealth jumps between rows {previous.index} and {row.index}")
        previous = row
        checked += 1
    if not np.array_equal(record.wealth_path, [1.0, *(r.wealth_after for r in record.ledger)]):
        _fail(name, episode, "its wealth path is not its ledger's")
    return checked


def check_contract(
    records: Mapping[str, Mapping[Episode, EpisodeRecord]],
    *,
    data: Mapping[Episode, ScenarioData],
    fold: FoldSpec,
    env_config: EnvConfig,
    builder: ObservationBuilder,
    cost_multiplier: float,
) -> dict[str, int]:
    """
    Raise `FairnessError` unless every strategy's record of every episode
    of `fold` was produced under the registered contract (module docstring).
    Returns how many strategies, episodes and booked steps were checked.
    """

    if execution_lag() != ENV_EXECUTION_LAG:
        raise FairnessError("The environment's lag is not the registered execution lag (D1).")
    episodes = set(fold.episodes())
    steps = 0
    for name, by_episode in records.items():
        if set(by_episode) != episodes:
            raise FairnessError(f"{name} ran other episodes than the fold's.")
        for episode, record in by_episode.items():
            if type(record) is not EpisodeRecord:
                _fail(name, episode, f"evaluated through another API ({type(record).__name__})")
    check_fairness({name: dict(by_episode) for name, by_episode in records.items()})
    references = {
        episode: _reference(data[episode], fold, env_config, builder)
        for episode in sorted(episodes)
    }
    for name, by_episode in sorted(records.items()):
        for episode, record in sorted(by_episode.items()):
            steps += _check_record(name, episode, record, references[episode], cost_multiplier)
    return {"strategies": len(records), "episodes": len(episodes), "steps": steps}


def _spread(values: list[float]) -> dict[str, float | int]:
    # Recomputed here, independently of the report's own code.
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


def check_no_test_selection(artifact: Mapping[str, object], selection: Mapping[str, object]) -> int:
    """
    Raise `SelectionError` unless the artifact reports every member frozen in
    `selection` (dev-selected before the evaluation fold existed) on every
    episode of the fold, each with its recorded checkpoint, and every seed
    distribution and learned summary is recomputed, here, from all members'
    rows. Returns the members checked.
    """

    frozen = {
        (m["constraint_set"], m["kind"], int(m["seed"])): m["checkpoint"]["sha256"]
        for m in selection["members"]
    }
    learned = {s["key"]: s["learned"] for s in artifact["strategies"] if s["learned"]}
    fold = artifact["fold"]
    episodes = {
        (scenario, int(seed)) for scenario in fold["scenario_ids"] for seed in fold["seeds"]
    }
    reported: dict[tuple[str, str, int], set[tuple[str, int]]] = {}
    wealth: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    for row in artifact["rows"]:
        if row["strategy"] not in learned:
            continue
        seed = int(row["member_seed"])
        key = (row["constraint_set"], learned[row["strategy"]], seed)
        if key not in frozen:
            raise SelectionError(f"{row['strategy']} reports a member never frozen: {key}.")
        if row["checkpoint_sha256"] != frozen[key]:
            raise SelectionError(f"{row['strategy']} {key} reports another checkpoint.")
        reported.setdefault((row["constraint_set"], row["strategy"], seed), set()).add(
            (row["scenario"], int(row["scenario_seed"]))
        )
        group = (row["constraint_set"], row["strategy"], row["scenario"])
        wealth.setdefault(group, {}).setdefault(seed, []).append(float(row["terminal_wealth"]))
    for strategy, kind in learned.items():
        for constraint_set in [s["name"] for s in artifact["constraint_sets"]]:
            expected = sorted(
                seed for (cs, k, seed) in frozen if cs == constraint_set and k == kind
            )
            for seed in expected:
                if reported.get((constraint_set, strategy, seed)) != episodes:
                    raise SelectionError(
                        f"{strategy} under {constraint_set}: seed {seed} is not reported on "
                        "every episode of the fold; every frozen seed is reported."
                    )
            for scenario in fold["scenario_ids"]:
                per_seed = wealth.get((constraint_set, strategy, scenario), {})
                if sorted(per_seed) != expected:
                    raise SelectionError(f"{strategy}: not every frozen seed is reported.")
    variability = {
        (e["constraint_set"], e["strategy"], e["scenario"]): e for e in artifact["seed_variability"]
    }
    if set(variability) != set(wealth):
        raise SelectionError("A learned strategy's seed spread is missing for some scenario.")
    for group, per_seed in wealth.items():
        entry = variability[group]
        means = [statistics.fmean(per_seed[seed]) for seed in sorted(per_seed)]
        if entry["seeds"] != sorted(per_seed) or entry["per_seed_mean_terminal_wealth"] != means:
            raise SelectionError(f"{group[1]}: the per-seed values are not every seed's.")
        if entry["distribution"] != _spread(means):
            raise SelectionError(f"{group[1]}: the seed distribution is not all seeds'.")
    for summary in artifact["summary"]:
        group = (summary["constraint_set"], summary["strategy"], summary["scenario"])
        if group in wealth:
            values = [v for seed in sorted(wealth[group]) for v in wealth[group][seed]]
            if summary["terminal_wealth"] != _spread(values):
                raise SelectionError(f"{group[1]}: its summary does not span every seed's paths.")
    return len(frozen)
