"""
The fairness contract of the registered comparison: every strategy uses the
same API, transition (`portfolio_step`), decision timing (decide at d,
execute at the next close), D2 costs, schedule and offset (D7), build from
cash (D3) and registered constraint set (D5).

The honest registered rules pass under both registered sets and at every
cost multiplier. Each mutation probe runs one strategy with exactly one
advantage or deviation, through the same environment API wherever one can
be planted there, and the contract must refuse it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from stock_agent.comparison import fairness
from stock_agent.comparison.fairness import FairnessError, SelectionError, check_contract
from stock_agent.comparison.protocol import RULE_FACTORIES
from stock_agent.comparison.runner import CEMKind, scaled_costs
from stock_agent.portfolio import engine
from stock_agent.portfolio.transition import Constraints, portfolio_step
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import scenarios as sc
from stock_agent.rl.config import run_config_from_dict
from stock_agent.rl.policies import fixed_balanced
from stock_agent.rl.replay import rollout
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import APPROVED_TRANSITION

FOLD = FoldSpec("test", ("calm", "missing_closes"), (20,), 540, 660)
PLAIN = rl_env.EnvConfig(rebalance_every=5)
STRICT = rl_env.EnvConfig(rebalance_every=5, max_weight=0.5, max_turnover=0.25)
SPEC = ob.FeatureSpec()
BUILDER = ob.ObservationBuilder(
    SPEC,
    ob.FeatureNormalizer.identity(SPEC.market_names(sc.ASSET_NAMES, sc.CASH_INDEX)),
    sc.ASSET_NAMES,
    sc.CASH_INDEX,
)
DATA = {episode: sc.generate_scenario(*episode) for episode in FOLD.episodes()}


def run(
    factory=fixed_balanced,
    *,
    env_cls=rl_env.AllocationEnv,
    config=STRICT,
    data=DATA,
    transition=portfolio_step,
):
    return {
        episode: rollout(
            env_cls(
                data[episode],
                start=FOLD.start,
                stop=FOLD.stop,
                transition=transition,
                transition_info=APPROVED_TRANSITION,
                builder=BUILDER,
                config=config,
            ),
            factory(),
        )
        for episode in FOLD.episodes()
    }


def contract(records, *, data=DATA, config=STRICT, multiplier=1.0):
    return check_contract(
        records,
        data=data,
        fold=FOLD,
        env_config=config,
        builder=BUILDER,
        cost_multiplier=multiplier,
    )


@pytest.mark.parametrize("config", [PLAIN, STRICT], ids=["plain", "strict"])
def test_every_registered_rule_passes_the_contract(config):
    records = {key: run(factory, config=config) for key, factory in RULE_FACTORIES.items()}
    checked = contract(records, config=config)
    assert checked["strategies"] == len(RULE_FACTORIES) and checked["episodes"] == 2
    assert checked["steps"] > 100 * len(RULE_FACTORIES)


@pytest.mark.parametrize("multiplier", [0.0, 1.0, 2.0])
def test_cost_sensitivity_replays_pass_with_their_own_multiplier_only(multiplier):
    data = {episode: scaled_costs(d, multiplier) for episode, d in DATA.items()}
    records = {"fixed_balanced": run(data=data)}
    contract(records, data=data, multiplier=multiplier)
    other = 1.0 if multiplier != 1.0 else 2.0
    with pytest.raises(FairnessError, match="registered D2 rates"):
        contract(records, data=data, multiplier=other)


def test_the_registered_rules_differ_only_in_their_targets():
    records = {key: run(factory) for key, factory in RULE_FACTORIES.items()}
    reference = records["equal_weight"]
    for by_episode in records.values():
        for episode, record in by_episode.items():
            other = reference[episode]
            assert record.identity == other.identity
            assert [r.session for r in record.ledger] == [r.session for r in other.ledger]
            first = record.decisions[0].decision_index + 1
            assert record.ledger[first].session == other.ledger[first].session  # common build
            assert np.array_equal(record.ledger[first].w_pre, [0, 0, 0, 1])  # from cash


# --- mutation probes: one deviation each, which the contract must refuse ---


def half_cost_step(w_pre, target, r_next, cost_rate, cash_index):
    return portfolio_step(w_pre, target, r_next, np.asarray(cost_rate) * 0.5, cash_index)


def free_build_step(w_pre, target, r_next, cost_rate, cash_index):
    free = np.asarray(w_pre)[cash_index] == 1.0
    rates = np.zeros_like(cost_rate) if free else cost_rate
    return portfolio_step(w_pre, target, r_next, rates, cash_index)


class SameCloseEnv(rl_env.AllocationEnv):
    """Lag 0: the decision at c_j trades at the close of c_j itself."""

    def step(self, target):
        row = self.ledger.pop()  # row j, booked before the decision existed
        self._post_trade.pop()
        self._wealth.pop()
        self._w_pre = row.w_pre
        self._book(self._j, target=target, decision_session=int(self.complete[self._j]))
        return super().step(None)


class InvestedEnv(rl_env.AllocationEnv):
    """Starts already invested instead of building from cash."""

    def reset(self):
        self._w_pre = np.array([0.6, 0.3, 0.1, 0.0])
        self._wealth, self.ledger, self._post_trade = [1.0], [], []
        for m in range(self.first_decision_index + 1):
            self._book(m, target=None, decision_session=None)
        self._j, self._ready = self.first_decision_index, True
        return self._observe(self._j), self._info(self._j, None)


class LooseConstraintEnv(rl_env.AllocationEnv):
    """Claims the registered set but maps targets with the defaults."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = Constraints(self.data.cash_index)


class DoubleCountEnv(rl_env.AllocationEnv):
    """Subtracts the cost again from the reward (D8 (b)-like double counting)."""

    def _book(self, m, *, target, decision_session):
        row = super()._book(m, target=target, decision_session=decision_session)
        self.ledger[-1] = dataclasses.replace(row, reward=row.reward - row.cost)
        return self.ledger[-1]


def engine_api(factory=fixed_balanced):
    records = {}
    for episode, data in DATA.items():
        complete = data.complete_sessions(FOLD.start, FOLD.stop)
        returns = np.full((len(complete), 4), np.nan)
        for k in range(1, len(complete)):
            returns[k] = data.span_returns(int(complete[k - 1]), int(complete[k]))
        records[episode] = engine.simulate(
            returns,
            {0: factory().weights},
            lag=1,
            cost_rate=np.array(sc.BASE_COST_RATES),
            constraints=STRICT.constraints(sc.CASH_INDEX),
            initial_wealth=1.0,
        )
    return records


def edit_records(edit):
    return {episode: edit(record) for episode, record in run().items()}


def drop_feasibility(record):
    ledger = tuple(dataclasses.replace(row, feasibility=None) for row in record.ledger)
    return dataclasses.replace(record, ledger=ledger)


def inflate_wealth(record):
    return dataclasses.replace(record, wealth_path=record.wealth_path * 1.01)


MUTANTS = {
    "another_api": (engine_api, "another API"),
    "cheaper_transition": (lambda: run(transition=half_cost_step), "approved portfolio_step"),
    "free_build": (lambda: run(transition=free_build_step), "approved portfolio_step"),
    "same_close_execution": (lambda: run(env_cls=SameCloseEnv), "off the lag"),
    "daily_schedule": (
        lambda: run(config=dataclasses.replace(STRICT, rebalance_every=1)),
        "differ",
    ),
    "other_offset": (
        lambda: run(config=dataclasses.replace(STRICT, rebalance_offset=1)),
        "differ",
    ),
    "cheaper_cost_model": (
        lambda: run(data={e: scaled_costs(d, 0.5) for e, d in DATA.items()}),
        "differ",
    ),
    "starts_invested": (lambda: run(env_cls=InvestedEnv), "100% cash"),
    "another_constraint_set": (lambda: run(config=PLAIN), "differ"),
    "constraints_not_applied": (lambda: run(env_cls=LooseConstraintEnv), "constraint set"),
    "feasibility_skipped": (lambda: edit_records(drop_feasibility), "feasibility map"),
    "fabricated_wealth": (lambda: edit_records(inflate_wealth), "wealth path"),
    "double_counted_cost": (lambda: run(env_cls=DoubleCountEnv), r"log\(1 \+ R_net\)"),
    "missing_episode": (lambda: dict(list(run().items())[:1]), "other episodes"),
}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_a_strategy_with_any_deviation_fails_the_contract(mutant):
    make, message = MUTANTS[mutant]
    honest = run(RULE_FACTORIES["equal_weight"])
    contract({"equal_weight": honest})  # the honest strategy alone passes
    with pytest.raises(FairnessError, match=message):
        contract({"equal_weight": honest, mutant: make()})


def test_a_lag_other_than_the_registered_one_is_refused(monkeypatch):
    monkeypatch.setattr(fairness, "execution_lag", lambda: 0)
    with pytest.raises(FairnessError, match="registered execution lag"):
        contract({"fixed_balanced": run()})


# --- learned members load only as frozen, observing through the same normalizer ---


def _cem_member(tmp_path, *, normalizer, env_config):
    record = {
        "normalizer": normalizer.to_dict(),
        "env_config": env_config.to_dict(),
        "params": {"bias": [0.0, 0.0, 0.0], "beta_momentum": 0.0, "beta_volatility": 0.0},
    }
    path = tmp_path / "cem_seed_0.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"checkpoint": {"path": path.name, "sha256": sha}, "seed": 0}


def _run_config(env_config):
    record = {
        "schema_version": 1,
        "name": "x",
        "training": {
            "seeds": [0],
            "total_timesteps": 1,
            "n_envs": 1,
            "episode_sessions": 60,
            "eval_every": 1,
        },
    }
    return dataclasses.replace(run_config_from_dict(record), env=env_config)


def test_a_cem_member_loads_only_with_its_frozen_hash_normalizer_and_constraint_set(tmp_path):
    kind = CEMKind()
    member = _cem_member(tmp_path, normalizer=BUILDER.normalizer, env_config=STRICT)
    policy = kind.load(member, run_dir=tmp_path, builder=BUILDER, run_config=_run_config(STRICT))()
    assert policy.name == "linear_softmax_cem"
    with pytest.raises(FairnessError, match="another constraint set"):
        kind.load(member, run_dir=tmp_path, builder=BUILDER, run_config=_run_config(PLAIN))
    other = dataclasses.replace(BUILDER.normalizer, mean=(0.5,) * len(BUILDER.normalizer.names))
    member = _cem_member(tmp_path, normalizer=other, env_config=STRICT)
    with pytest.raises(FairnessError, match="another normalizer"):
        kind.load(member, run_dir=tmp_path, builder=BUILDER, run_config=_run_config(STRICT))
    member["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(SelectionError, match="frozen"):
        kind.load(member, run_dir=tmp_path, builder=BUILDER, run_config=_run_config(STRICT))
