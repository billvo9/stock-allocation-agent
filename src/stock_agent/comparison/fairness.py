"""
The fairness contract of the registered comparison, checked on every record
before any measure is reported, and the check that no result of the
evaluation fold selected anything.

The registered setup is derived here, from the registered configuration
alone (`RegisteredSetup`), never from what the runner was given: per
constraint set and cost multiplier, ONE common setup of transition, lag, D2
cost rates, schedule and offset, build date, evaluation window and the full
D5 set (cap and turnover limit), with the scenario data regenerated from the
registered generator. `ContractCheck` then checks every strategy's record of
every episode against it, and `close()` compares all strategies together:

- API: the record is an `EpisodeRecord` whose identity (scenario
  fingerprint with its cost rates, window, schedule and constraint set,
  transition and reward) is the registered setup's;
- transition: the approved `portfolio_step` (D1-D8): every booked step is
  replayed through it, from the registered data, and must agree bit for bit
  (weights, cost, returns, turnover, traded notional); wealth compounds by
  each net return and each reward is log(1 + R_net) (D8 (a));
- timing (D1): a decision at complete session c_j executes at c_{j+1}; no
  other session carries a target;
- schedule and window (D7): the decisions and booked sessions are the
  registered schedule's over the registered window;
- costs (D2): the provisional rates, times the scenario's cost regime, times
  the registered multiplier, at every booked session;
- build (D3): 100% cash until the registered build close, where every
  strategy's first decision executes;
- constraint set (D5): every target passed through `feasible_target` with
  the registered set, and the transition received its result;
- together: every registered entry (each rule, each learned seed) was
  checked exactly once, and all share one fingerprint, the registered one.

`check_no_test_selection` verifies that the reported learned members are
exactly the members frozen (dev-selected) before the evaluation fold was
built, with their recorded checkpoints, and that every seed distribution
is recomputed from all of them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from stock_agent.comparison.protocol import ComparisonConfig
from stock_agent.portfolio.schedule import execution_lag
from stock_agent.portfolio.transition import feasible_target, portfolio_step
from stock_agent.rl.env import ENV_EXECUTION_LAG, AllocationEnv, EnvConfig
from stock_agent.rl.harness import FairnessError
from stock_agent.rl.observation import ObservationBuilder, ScenarioLoader
from stock_agent.rl.replay import EpisodeRecord
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.scenarios import BASE_COST_RATES, ScenarioData, generate_scenario
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import APPROVED_TRANSITION

__all__ = [
    "ContractCheck",
    "FairnessError",
    "RegisteredSetup",
    "SelectionError",
    "check_contract",
    "check_no_test_selection",
    "scaled_costs",
]

Episode = tuple[str, int]
_STEP_FIELDS = ("gross_return", "cost", "net_return", "turnover_one_way", "traded_notional")


class SelectionError(AssertionError):
    """A reported result depends on the evaluation fold (selection on test results)."""


def _fail(name: str, episode: Episode, what: str) -> None:
    raise FairnessError(f"{name} on {episode}: {what}")


def scaled_costs(data: ScenarioData, multiplier: float) -> ScenarioData:
    """The same scenario with its base cost rates times a D2 sensitivity multiplier."""

    rates = tuple(rate * multiplier for rate in data.base_cost_rates)
    return dataclasses.replace(data, base_cost_rates=rates)


@dataclass(frozen=True)
class RegisteredSetup:
    """
    The one common setup of a registered constraint set at one cost
    multiplier on one fold, derived from the registered configuration only.
    """

    config: ComparisonConfig
    constraint_set: str
    fold_role: str
    cost_multiplier: float
    loader: ScenarioLoader = generate_scenario
    _data: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.config.constraint_set(self.constraint_set)  # KeyError for an unregistered set
        if self.fold_role not in ("test", "dev"):
            raise ValueError("The comparison is evaluated on the test or the dev fold.")
        if self.cost_multiplier not in self.config.cost_multipliers:
            raise ValueError(f"{self.cost_multiplier} is not a registered cost multiplier.")

    @property
    def fold(self) -> FoldSpec:
        plan = self.config.rl.plan
        return plan.test if self.fold_role == "test" else plan.dev

    @property
    def env_config(self) -> EnvConfig:
        return self.config.run_config(self.config.constraint_set(self.constraint_set)).env

    def data(self, episode: Episode) -> ScenarioData:
        if episode not in self._data:
            self._data[episode] = scaled_costs(self.loader(*episode), self.cost_multiplier)
        return self._data[episode]

    def reference(self, episode: Episode, builder: ObservationBuilder) -> AllocationEnv:
        fold = self.fold
        return AllocationEnv(
            self.data(episode),
            start=fold.start,
            stop=fold.stop,
            transition=portfolio_step,
            transition_info=APPROVED_TRANSITION,
            builder=builder,
            config=self.env_config,
        )

    def identity(self) -> dict[str, object]:
        """The common identity every strategy shares (episode-independent)."""

        env = self.env_config
        fold = self.fold
        return {
            "config_sha256": self.config.sha256(),
            "constraint_set": {
                "name": self.constraint_set,
                "max_weight": env.max_weight,
                "max_turnover": env.max_turnover,
            },
            "transition": APPROVED_TRANSITION.to_dict(),
            "reward": REWARD_INFO.to_dict(),
            "execution_lag": ENV_EXECUTION_LAG,
            "base_cost_rates": [rate * self.cost_multiplier for rate in BASE_COST_RATES],
            "cost_multiplier": self.cost_multiplier,
            "schedule": {"every": env.rebalance_every, "offset": env.rebalance_offset},
            "build": "100% cash until one session after the first decision (D3)",
            "window": [fold.start, fold.stop],
            "fold": self.fold_role,
        }


def _digest(values: Sequence[object]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def _fingerprint(record: EpisodeRecord) -> str:
    """What every strategy of one episode shares: setup, sessions, schedule, build, rates."""

    first = record.decisions[0].decision_index + ENV_EXECUTION_LAG
    return _digest(
        [
            record.identity,
            [row.session for row in record.ledger],
            [d.decision_session for d in record.decisions],
            record.ledger[first].session,
            [row.cost_rates.tolist() for row in record.ledger],
        ]
    )


def _reference_fingerprint(reference: AllocationEnv) -> str:
    first = reference.first_decision_index
    decisions = range(first, reference.final_index, reference.config.rebalance_every)
    booked = [int(s) for s in reference.complete[: reference.final_index + 1]]
    return _digest(
        [
            reference.identity(),
            booked,
            [int(reference.complete[j]) for j in decisions],
            int(reference.complete[first + ENV_EXECUTION_LAG]),
            [reference.data.cost_rates_at(s).tolist() for s in booked],
        ]
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


class ContractCheck:
    """
    The contract for one registered setup: `add` checks each strategy's
    records against the registered reference; `close` checks every strategy
    together, once all are added.
    """

    def __init__(self, setup: RegisteredSetup, builder: ObservationBuilder) -> None:
        if execution_lag() != ENV_EXECUTION_LAG:
            raise FairnessError("The environment's lag is not the registered execution lag (D1).")
        self.setup = setup
        self.builder = builder
        self.episodes = sorted(setup.fold.episodes())
        self._references = {episode: setup.reference(episode, builder) for episode in self.episodes}
        self._expected = {
            episode: _reference_fingerprint(reference)
            for episode, reference in self._references.items()
        }
        self._fingerprints: dict[str, dict[Episode, str]] = {}
        self.steps = 0
        self.closed = False

    def add(self, name: str, records: Mapping[Episode, EpisodeRecord]) -> None:
        if self.closed:
            raise FairnessError("The contract was already closed.")
        if name in self._fingerprints:
            raise FairnessError(
                f"{name} was evaluated twice on the {self.setup.fold_role} fold (an uncounted look)."
            )
        if set(records) != set(self.episodes):
            raise FairnessError(f"{name} ran other episodes than the registered fold's.")
        for episode, record in records.items():
            if type(record) is not EpisodeRecord:
                _fail(name, episode, f"evaluated through another API ({type(record).__name__})")
        fingerprints = {}
        for episode in self.episodes:
            record = records[episode]
            self.steps += _check_record(
                name, episode, record, self._references[episode], self.setup.cost_multiplier
            )
            fingerprints[episode] = _fingerprint(record)
        self._fingerprints[name] = fingerprints

    def close(self, expected_names: Sequence[str]) -> dict[str, object]:
        """
        Every registered entry was checked once, and all of them share the
        registered fingerprint of every episode.
        """

        if set(self._fingerprints) != set(expected_names) or len(expected_names) != len(
            set(expected_names)
        ):
            missing = sorted(set(expected_names) - set(self._fingerprints))
            extra = sorted(set(self._fingerprints) - set(expected_names))
            raise FairnessError(
                f"Not every registered entry was checked: missing {missing}, extra {extra}."
            )
        for episode in self.episodes:
            shared = {self._fingerprints[name][episode] for name in expected_names}
            if shared != {self._expected[episode]}:
                raise FairnessError(
                    f"The strategies do not share the registered setup on {episode}."
                )
        self.closed = True
        return {
            "setup": self.setup.identity(),
            "strategies": len(expected_names),
            "episodes": len(self.episodes),
            "steps": self.steps,
        }


def check_contract(
    records: Mapping[str, Mapping[Episode, EpisodeRecord]],
    *,
    setup: RegisteredSetup,
    builder: ObservationBuilder,
    expected_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Check every strategy's records against the registered setup, then all
    of them together (`expected_names`: the registered entries; by default
    the strategies given). Raises `FairnessError` on any deviation.
    """

    check = ContractCheck(setup, builder)
    for name, by_episode in records.items():
        check.add(name, by_episode)
    return check.close(list(records) if expected_names is None else list(expected_names))


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
