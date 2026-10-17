"""
The registered comparison configuration: its content is pinned by hash, and
no configuration can weaken a registered rule (D2 multipliers, the D5 sets
applied to every strategy, at least five learned seeds).
"""

from __future__ import annotations

import copy

import pytest
from comparison_helpers import REGISTERED, registered_record

from stock_agent.comparison import protocol
from stock_agent.comparison.protocol import (
    ProtocolError,
    comparison_config_from_dict,
    load_comparison_config,
)
from stock_agent.portfolio.transition import SENSITIVITY_MULTIPLIERS
from stock_agent.rl.splits import EVALUATION_SCENARIOS

# A change to the registered file is a new registration (a new name and hash), never an
# edit of this one after results exist.
REGISTERED_SHA256 = "4f45f7d084164913625b090e125d921eb441cf06e7a3798bf19b8d945e94a417"


def test_the_registered_configuration_is_pinned_by_its_hash():
    config = load_comparison_config(REGISTERED)
    assert config.name == "ppo_comparison_v1"
    assert config.sha256() == REGISTERED_SHA256


def test_the_registered_configuration_covers_the_brief():
    config = load_comparison_config(REGISTERED)
    assert config.strategies == (
        "buy_and_hold",
        "equal_weight",
        "fixed_balanced",
        "inverse_volatility",
        "mean_variance",
        "all_shv",
        "linear_softmax_cem",
        "ppo",
        "ppo_guarded",
    )
    sets = {s.name: (s.max_weight, s.max_turnover) for s in config.constraint_sets}
    assert sets == {"plain": (1.0, None), "strict": (0.5, 0.25)}
    assert config.cost_multipliers == SENSITIVITY_MULTIPLIERS == (0.0, 1.0, 2.0)
    assert len(config.ppo_seeds) >= 5 and len(config.cem_seeds) >= 5
    plan = config.rl.plan
    stresses = {"calm", "crash", "high_correlation", "extreme_volatility"}
    stresses |= {"stale_observations", "high_costs"}
    assert stresses <= set(plan.test.scenario_ids) == set(EVALUATION_SCENARIOS)
    assert set(plan.dev.scenario_ids) == set(plan.test.scenario_ids)
    assert not set(plan.test.seeds) & (set(plan.dev.seeds) | set(plan.train.seeds))


def test_each_set_changes_only_the_constraints_of_the_shared_run_configuration():
    config = load_comparison_config(REGISTERED)
    plain, strict = (config.run_config(s) for s in config.constraint_sets)
    assert (strict.env.max_weight, strict.env.max_turnover) == (0.5, 0.25)
    assert (plain.env.max_weight, plain.env.max_turnover) == (1.0, None)
    first, second = plain.to_dict(), strict.to_dict()
    first.pop("env"), second.pop("env")
    assert first == second


def test_the_configuration_round_trips():
    config = load_comparison_config(REGISTERED)
    again = comparison_config_from_dict(config.to_dict())
    assert again == config and again.sha256() == config.sha256()


def _broken(edit) -> dict:
    record = copy.deepcopy(registered_record())
    edit(record)
    return record


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.update(cost_multipliers=[1.0, 2.0]), "never chosen"),
        (lambda r: r.update(cost_multipliers=[0.0, 1.0, 3.0]), "never chosen"),
        (lambda r: r["constraint_sets"].pop(), "at least two"),
        (lambda r: r["constraint_sets"][0].update(max_weight=0.9), "D5 defaults"),
        (lambda r: r["constraint_sets"][0].update(name="loose"), "D5 defaults"),
        (lambda r: r["constraint_sets"][1].update(max_weight=1.0, max_turnover=None), "stricter"),
        (lambda r: r["rl"]["env"].update(max_weight=0.5), "may not register constraints"),
        (lambda r: r["rl"]["env"].update(max_turnover=0.1), "may not register constraints"),
        (lambda r: r["rl"]["training"].update(seeds=[0, 1, 2, 3]), "at least 5"),
        (lambda r: r.update(cem_seeds=[0, 1, 2, 3]), "at least 5"),
        (lambda r: r.update(cem_seeds=[0, 0, 1, 2, 3]), "at least 5"),
        (lambda r: r["rl"]["cem"].update(seed=3), "cem_seeds"),
        (lambda r: r["strategies"].append("momentum"), "Unknown strategies"),
        (lambda r: r["strategies"].append("ppo"), "distinct"),
        (lambda r: r.update(extra=1), "Unknown top-level"),
        (lambda r: r.update(protocol="another"), "protocol"),
        (lambda r: r["rl"].update(name="x"), "top level"),
        (lambda r: r["constraint_sets"][1].update(max_turnover=1.5), "tau_max"),
    ],
)
def test_a_configuration_that_breaks_a_registered_rule_is_refused(edit, message):
    with pytest.raises((ProtocolError, ValueError), match=message):
        comparison_config_from_dict(_broken(edit))


def test_a_rules_only_comparison_needs_no_learned_seeds():
    record = _broken(lambda r: r.update(strategies=["equal_weight", "all_shv"], cem_seeds=[]))
    record["rl"]["training"]["seeds"] = [0]
    config = comparison_config_from_dict(record)
    assert config.learned_kinds() == ()


def test_every_strategy_is_registered_with_a_rule_or_a_trainer():
    for key, spec in protocol.STRATEGIES.items():
        assert spec.is_learned or key in protocol.RULE_FACTORIES
        assert spec.label and spec.rule
    shv = protocol.RULE_FACTORIES["all_shv"]()
    assert list(shv.weights) == [0.0, 0.0, 1.0, 0.0]  # the short-Treasury fund, held as a fund


def test_the_registered_entries_are_every_rule_and_every_learned_seed():
    config = load_comparison_config(REGISTERED)
    names = config.entry_names()
    assert names[:6] == (
        "buy_and_hold",
        "equal_weight",
        "fixed_balanced",
        "inverse_volatility",
        "mean_variance",
        "all_shv",
    )
    for key in ("linear_softmax_cem", "ppo", "ppo_guarded"):
        assert [n for n in names if n.startswith(f"{key}/")] == [
            protocol.entry_name(key, seed) for seed in range(5)
        ]
    assert len(names) == len(set(names)) == 6 + 3 * 5
