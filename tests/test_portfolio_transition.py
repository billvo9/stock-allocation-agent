"""
The T5a engine contract (`portfolio.transition`, `portfolio.schedule`,
`portfolio.engine`, `environment.reward.log_growth_reward`).

Two kinds of test:

- Structure (passing now): data types validate their inputs, the
  `Transition` seam has the F8.1 signature, nothing imports the product
  front ends, and every economic function still refuses to run.
- Economics (strict xfails, `raises=NotImplementedError`): the packet's
  hand examples, seeded invariants, shares-ledger conservation, adversarial
  timing and replay equivalence against the reference oracle. They wait for
  the owner's OD8 group B approval (D1-D8). Any other exception, or an
  unexpected pass, fails the suite, so the stubs cannot be filled in without
  flipping these tests deliberately.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from reference import t5a_reference as oracle
from reference import timing
from reference.replay import ReplayPath, ReplayScenario, oracle_runner, replay

from stock_agent.environment.reward import log_growth_reward
from stock_agent.features.training import MODEL_LABEL_SPEC
from stock_agent.portfolio import engine, schedule
from stock_agent.portfolio.descriptive import DISTRIBUTION_IN_GAP
from stock_agent.portfolio.transition import (
    CAP,
    NO_TARGET,
    NO_TRADE,
    PENDING_APPROVAL,
    TRADED,
    TURNOVER_LIMIT,
    Constraints,
    CostModel,
    FeasibleTarget,
    InvalidTarget,
    StepResult,
    Transition,
    feasible_target,
    portfolio_step,
)

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stock_agent"
NAN = float("nan")
TOLERANCE = 1e-12

PENDING = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=PENDING_APPROVAL)


def _step(**overrides):
    fields = {
        "w_post_trade": [0.4, 0.6],
        "w_next": [0.5, 0.5],
        "gross_return": 0.01,
        "cost": 0.0,
        "net_return": 0.01,
        "turnover_one_way": 0.0,
        "traded_notional": 0.0,
        "reason": NO_TARGET,
    }
    return StepResult(**{**fields, **overrides})


def engine_runner(scenario: ReplayScenario) -> ReplayPath:
    """The engine behind the replay harness's `EpisodeRunner` seam."""

    result = engine.simulate(
        np.asarray(scenario.returns, dtype=float),
        {
            d: None if target is None else np.asarray(target, dtype=float)
            for d, target in scenario.targets.items()
        },
        lag=scenario.lag,
        cost_rate=np.asarray(scenario.cost_rate, dtype=float),
        constraints=Constraints(scenario.cash_index, scenario.w_max, scenario.tau_max),
        initial_wealth=scenario.initial_wealth,
        has_close=scenario.has_close,
    )
    return ReplayPath(tuple(result.sessions), tuple(result.wealth), result.status, result.reason)


# ======================================================================
# Structure (passing now)
# ======================================================================


def test_portfolio_step_has_the_f8_1_signature_of_the_transition_seam():
    expected = ["w_pre", "target", "r_next", "cost_rate", "cash_index"]
    assert list(inspect.signature(portfolio_step).parameters) == expected
    seam = list(inspect.signature(Transition.__call__).parameters)
    assert seam == ["self", *expected]


def test_a_step_result_is_read_only_and_validated():
    step = _step()
    with pytest.raises(ValueError):
        step.w_next[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.cost = 0.5
    with pytest.raises(ValueError, match="same shape"):
        _step(w_next=[0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="finite"):
        _step(net_return=NAN)
    with pytest.raises(ValueError, match="non-negative"):
        _step(cost=-1e-9)
    with pytest.raises(ValueError, match="reason"):
        _step(reason="deferred")
    assert {TRADED, NO_TARGET, NO_TRADE} == {
        _step(reason=r).reason for r in (TRADED, NO_TARGET, NO_TRADE)
    }


def test_cost_models_and_constraints_validate_structure_only():
    model = CostModel({"VTI": 0.0005, "CASH": 0})
    with pytest.raises(TypeError):
        model.rates["VTI"] = 0.0  # read-only mapping
    with pytest.raises(ValueError, match="non-negative"):
        CostModel({"VTI": -0.0001})
    with pytest.raises(ValueError, match="at least one"):
        CostModel({})
    with pytest.raises(ValueError, match="multiplier"):
        CostModel({"VTI": 0.0005}, multiplier=-1)
    assert Constraints(2) == Constraints(cash_index=2, w_max=1.0, tau_max=None)
    with pytest.raises(ValueError, match="w_max"):
        Constraints(2, w_max=0.0)
    with pytest.raises(ValueError, match="tau_max"):
        Constraints(2, tau_max=1.5)
    with pytest.raises(TypeError):
        Constraints(True)


def test_a_feasible_target_records_how_far_the_map_moved_the_request():
    record = FeasibleTarget([0.7, 0.2, 0.1], [0.5, 0.2, 0.3], 0.2, (CAP,))
    assert record.alpha == 1.0 and record.binding == (CAP,)
    with pytest.raises(ValueError, match="binding"):
        FeasibleTarget([0.5, 0.5], [0.5, 0.5], 0.0, ("budget",))
    with pytest.raises(ValueError, match="alpha"):
        FeasibleTarget([0.5, 0.5], [0.5, 0.5], 0.0, alpha=1.5)


def test_schedule_types_validate_structure_only():
    assert schedule.ScheduleSpec(every=5, offset=4, lag=1).first_decision == 0
    with pytest.raises(ValueError, match="offset"):
        schedule.ScheduleSpec(every=5, offset=5, lag=1)
    with pytest.raises(ValueError, match="every"):
        schedule.ScheduleSpec(every=0, offset=0, lag=1)
    with pytest.raises(TypeError):
        schedule.ScheduleSpec(every=2, offset=0, lag=1.0)
    with pytest.raises(ValueError, match="reason"):
        schedule.ScheduledExecution(3, None, schedule.REBALANCE)
    with pytest.raises(ValueError, match="precede"):
        schedule.ScheduledExecution(3, 2, schedule.REBALANCE)
    assert DISTRIBUTION_IN_GAP in schedule.REASONS


def test_a_simulation_result_states_why_it_stopped():
    with pytest.raises(ValueError, match="reason"):
        engine.SimulationResult((1,), (1.0,), (), (), engine.STOPPED)
    with pytest.raises(ValueError, match="increasing"):
        engine.SimulationResult((2, 1), (1.0, 1.0), (), (), engine.COMPLETE)
    stopped = engine.SimulationResult((1,), (1.0,), (), (), engine.STOPPED, DISTRIBUTION_IN_GAP)
    assert stopped.reason == oracle.DISTRIBUTION_IN_GAP


SPEC = schedule.ScheduleSpec(every=1, offset=0, lag=1)
UNAPPROVED = {
    "portfolio_step": lambda: portfolio_step(
        np.array([0.5, 0.5]), None, np.zeros(2), np.zeros(2), 1
    ),
    "feasible_target": lambda: feasible_target(
        np.array([0.5, 0.5]), np.array([0.5, 0.5]), Constraints(1)
    ),
    "CostModel.rates_for": lambda: CostModel({"A": 0.0}).rates_for(["A"]),
    "execution_lag": schedule.execution_lag,
    "decision_sessions": lambda: schedule.decision_sessions(SPEC, 4),
    "execution_plan": lambda: schedule.execution_plan(SPEC, 4),
    "offset_family": lambda: schedule.offset_family(SPEC),
    "next_executable_session": lambda: schedule.next_executable_session(
        1, [0], lambda session, asset: True, 4
    ),
    "collect_targets": lambda: engine.collect_targets(lambda h: None, np.zeros((3, 2)), SPEC),
    "simulate": lambda: engine.simulate(
        np.zeros((3, 2)),
        {0: np.array([0.5, 0.5])},
        lag=1,
        cost_rate=np.zeros(2),
        constraints=Constraints(1),
        initial_wealth=1.0,
    ),
    "log_growth_reward": lambda: log_growth_reward(_step()),
}


@pytest.mark.parametrize("name", sorted(UNAPPROVED))
def test_no_unapproved_formula_runs(name):
    with pytest.raises(NotImplementedError, match="awaiting OD8 group B approval"):
        UNAPPROVED[name]()


def _imports(path: Path) -> set[str]:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


@pytest.mark.parametrize("name", ["transition", "schedule", "engine"])
def test_the_engine_never_imports_a_front_end_or_a_learner(name):
    imported = _imports(PACKAGE / "portfolio" / f"{name}.py")
    forbidden = ("app", "ui", "dashboard", "rl", "model_diagnostics", "publication", "environment")
    for module in imported:
        assert not any(module.startswith(f"stock_agent.{part}") for part in forbidden), module


# ======================================================================
# Economics (strict xfails until OD8 group B is approved)
# ======================================================================

# --- hand examples (packet D2, D3; spec T2.1-T2.5, T8.1) ---

REBALANCE = (
    np.array([0.55, 0.25, 0.20]),
    np.array([0.40, 0.40, 0.20]),
    np.array([0.02, -0.01, 0.0001]),
    np.array([0.001, 0.001, 0.0]),
    2,
)


def _assert_matches_ledger(step, w_pre, target, r_next, cost_rate, cash_index):
    ledger = oracle.ledger_step(w_pre, target, r_next, cost_rate, cash_index)
    np.testing.assert_allclose(step.w_post_trade, ledger["w_post_trade"], rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(step.w_next, ledger["w_next"], rtol=0, atol=TOLERANCE)
    for name in ("gross_return", "cost", "net_return", "turnover_one_way", "traded_notional"):
        assert getattr(step, name) == pytest.approx(ledger[name], rel=0, abs=TOLERANCE), name


@PENDING
def test_t8_1_the_rebalance_example():
    step = portfolio_step(*REBALANCE)
    assert step.cost == pytest.approx(0.0003, abs=1e-15)
    assert step.gross_return == pytest.approx(0.00402, abs=1e-14)
    assert step.net_return == pytest.approx(0.0037187940, abs=TOLERANCE)
    assert 100_000 * (1 + step.net_return) == pytest.approx(100_371.8794, abs=1e-6)
    assert [round(w, 5) for w in step.w_next] == [0.40637, 0.39441, 0.19922]
    assert step.turnover_one_way == pytest.approx(0.15, abs=1e-15)
    assert step.traded_notional == pytest.approx(0.30, abs=1e-15)
    assert step.reason == TRADED
    _assert_matches_ledger(step, *REBALANCE)


@PENDING
def test_t2_1_selling_into_cash_charges_the_risky_leg_only():
    step = portfolio_step(
        np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros(2), np.array([0.001, 0.0]), 1
    )
    assert 100_000 * (1 + step.net_return) == pytest.approx(99_900.0, abs=1e-9)


@PENDING
def test_t2_2_the_build_from_cash_costs_and_turns_over_fully():
    step = portfolio_step(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.5, 0.0]),
        np.zeros(3),
        np.array([0.001, 0.001, 0.0]),
        2,
    )
    assert 100_000 * (1 + step.net_return) == pytest.approx(99_900.0, abs=1e-9)
    assert step.turnover_one_way == pytest.approx(1.0, abs=1e-15)
    assert step.traded_notional == pytest.approx(1.0, abs=1e-15)


@PENDING
def test_t2_3_costs_multiply():
    step = portfolio_step(
        np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.10, 0.0]), np.array([0.001, 0.0]), 1
    )
    assert 1 + step.net_return == pytest.approx(1.0989, abs=TOLERANCE)


@PENDING
def test_t2_5_cross_class_moves():
    rates = CostModel({"VTI": 0.0005, "SHV": 0.0002, "AGG": 0.0005, "CASH": 0.0}).rates_for(
        ["VTI", "SHV", "AGG", "CASH"]
    )
    w_pre = np.array([0.8, 0.0, 0.0, 0.2])
    to_shv = portfolio_step(w_pre, np.array([0.6, 0.2, 0.0, 0.2]), np.zeros(4), rates, 3)
    to_agg = portfolio_step(w_pre, np.array([0.6, 0.0, 0.2, 0.2]), np.zeros(4), rates, 3)
    assert to_shv.cost * 1e4 == pytest.approx(1.4, abs=TOLERANCE)
    assert to_agg.cost * 1e4 == pytest.approx(2.0, abs=TOLERANCE)


@PENDING
def test_cost_rates_follow_the_asset_order_and_the_sensitivity_multiplier():
    model = CostModel({"VTI": 0.0005, "SHV": 0.0002, "CASH": 0.0}, multiplier=2.0)
    np.testing.assert_allclose(model.rates_for(["SHV", "VTI", "CASH"]), [0.0004, 0.001, 0.0])
    np.testing.assert_array_equal(
        CostModel(dict(model.rates), multiplier=0.0).rates_for(["VTI", "CASH"]), [0.0, 0.0]
    )
    with pytest.raises(KeyError):
        model.rates_for(["VTI", "QQQ"])


# --- D5: the feasibility map ---


@PENDING
def test_d5_the_turnover_limit_trades_part_way_and_says_so():
    record = feasible_target(
        np.array([0.30, 0.50, 0.20]), np.array([0.70, 0.10, 0.20]), Constraints(2, tau_max=0.25)
    )
    np.testing.assert_allclose(record.weights, [0.45, 0.35, 0.20], rtol=0, atol=TOLERANCE)
    assert record.alpha == pytest.approx(0.625, abs=TOLERANCE)
    assert record.binding == (TURNOVER_LIMIT,)
    assert record.distance_one_way == pytest.approx(0.15, abs=TOLERANCE)


@PENDING
def test_d5_the_cap_sends_the_excess_to_cash_and_says_so():
    record = feasible_target(
        np.array([0.70, 0.20, 0.10]), np.array([0.0, 0.0, 1.0]), Constraints(2, w_max=0.5)
    )
    np.testing.assert_allclose(record.weights, [0.50, 0.20, 0.30], rtol=0, atol=TOLERANCE)
    assert record.binding == (CAP,) and record.alpha == 1.0
    assert record.distance_one_way == pytest.approx(0.20, abs=TOLERANCE)


@PENDING
def test_d5_both_constraints_apply_in_order_like_the_oracle():
    requested, w_pre = np.array([0.70, 0.20, 0.10]), np.array([0.2, 0.2, 0.6])
    record = feasible_target(requested, w_pre, Constraints(2, w_max=0.5, tau_max=0.1))
    expected = oracle.feasible(requested.tolist(), w_pre.tolist(), 2, w_max=0.5, tau_max=0.1)
    np.testing.assert_allclose(record.weights, expected, rtol=0, atol=TOLERANCE)
    assert record.binding == (CAP, TURNOVER_LIMIT)


@PENDING
def test_d5_a_target_inside_the_set_passes_unchanged():
    requested = np.array([0.3, 0.3, 0.4])
    record = feasible_target(requested, np.array([0.2, 0.4, 0.4]), Constraints(2, 0.5, 0.5))
    np.testing.assert_array_equal(record.weights, requested)
    assert record.binding == () and record.distance_one_way == 0.0


@PENDING
@pytest.mark.parametrize("requested", [[0.6, 0.6, -0.2], [0.5, 0.5, 0.1], [NAN, 0.5, 0.5]])
def test_d5_a_rule_breaking_target_is_an_error(requested):
    with pytest.raises(InvalidTarget):
        feasible_target(np.array(requested), np.array([0.0, 0.0, 1.0]), Constraints(2))


# --- seeded invariants (spec F8.1, T8.2) ---


def _random_step(rng):
    n = int(rng.randint(2, 7))
    cash = int(rng.randint(n))
    w_pre = rng.dirichlet(np.ones(n) * rng.choice([0.3, 1.0, 5.0]))
    kind = rng.choice(["none", "same", "dense", "sparse"])
    if kind == "none":
        target = None
    elif kind == "same":
        target = w_pre.copy()
    else:
        target = rng.dirichlet(np.ones(n))
        if kind == "sparse":
            target[rng.rand(n) < 0.4] = 0.0
            target = target / target.sum() if target.sum() > 0 else np.eye(n)[cash]
    r_next = rng.uniform(-0.5, 0.5, size=n)
    r_next[cash] = rng.uniform(0.0, 0.001)
    rates = rng.uniform(0.0, 0.005, size=n)
    rates[cash] = 0.0
    return w_pre, target, r_next, rates, cash


@PENDING
@pytest.mark.parametrize("seed", range(4))
def test_f8_1_invariants_hold_on_random_steps(seed):
    rng = np.random.RandomState(seed)  # legacy stream: stable across numpy versions
    for _ in range(250):
        w_pre, target, r_next, rates, cash = _random_step(rng)
        step = portfolio_step(w_pre, target, r_next, rates, cash)
        for weights in (step.w_post_trade, step.w_next):
            assert (weights >= 0).all()
            assert abs(weights.sum() - 1.0) <= TOLERANCE
        assert 0.0 <= step.cost < 1.0
        assert step.net_return <= step.gross_return + 1e-15
        if target is None or np.array_equal(target, w_pre):
            assert step.cost == 0.0 and step.turnover_one_way == 0.0
            assert step.traded_notional == 0.0
            np.testing.assert_array_equal(step.w_post_trade, w_pre)
            assert step.reason == (NO_TARGET if target is None else NO_TRADE)
        else:
            assert step.reason == TRADED
        _assert_matches_ledger(step, w_pre, target, r_next, rates, cash)


@PENDING
@pytest.mark.parametrize("seed", range(3))
def test_an_all_cash_portfolio_earns_the_cash_return(seed):
    rng = np.random.RandomState(100 + seed)
    for _ in range(50):
        _, _, r_next, rates, cash = _random_step(rng)
        all_cash = np.eye(len(r_next))[cash]
        for target in (None, all_cash):
            step = portfolio_step(all_cash, target, r_next, rates, cash)
            assert step.net_return == pytest.approx(r_next[cash], rel=0, abs=1e-15)
            np.testing.assert_array_equal(step.w_next, all_cash)


@PENDING
def test_a_missing_return_on_a_held_asset_raises_and_on_an_unheld_one_does_not():
    r_next = np.array([NAN, 0.01, 0.0])
    with pytest.raises(ValueError):
        portfolio_step(np.array([0.5, 0.5, 0.0]), None, r_next, np.zeros(3), 2)
    step = portfolio_step(np.array([0.0, 0.5, 0.5]), None, r_next, np.zeros(3), 2)
    assert step.w_next[0] == 0.0 and np.isfinite(step.w_next).all()


# --- conservation: weights-based steps equal the shares ledger (T8.3) ---


def _random_path(seed, sessions=30):
    rng = np.random.RandomState(1_000 + seed)
    n = int(rng.randint(2, 6))
    cash = int(rng.randint(n))
    returns = rng.uniform(-0.1, 0.1, size=(sessions, n))
    returns[:, cash] = rng.uniform(0.0, 0.0005, size=sessions)
    returns[0] = NAN
    rates = rng.uniform(0.0, 0.003, size=n)
    rates[cash] = 0.0
    targets = {
        d: None if rng.rand() < 0.3 else rng.dirichlet(np.ones(n)) for d in range(sessions - 1)
    }
    targets[0] = rng.dirichlet(np.ones(n))
    return returns, targets, rates, cash


@PENDING
@pytest.mark.parametrize("seed", range(6))
def test_chained_steps_conserve_wealth_against_the_shares_ledger(seed):
    returns, targets, rates, cash = _random_path(seed)
    ledger = oracle.run(
        returns,
        {d: None if t is None else t.tolist() for d, t in targets.items()},
        lag=1,
        cost_rate=rates.tolist(),
        cash_index=cash,
    )
    weights = np.eye(len(rates))[cash]
    growth = 1.0
    for session in range(1, len(returns) - 1):
        assert ledger.wealth_at(session) == pytest.approx(growth, rel=0, abs=TOLERANCE)
        np.testing.assert_allclose(weights, ledger.weights[session - 1], rtol=0, atol=TOLERANCE)
        step = portfolio_step(weights, targets[session - 1], returns[session + 1], rates, cash)
        growth *= 1 + step.net_return
        weights = step.w_next
    assert growth == pytest.approx(ledger.wealth[-1] / ledger.wealth[0], rel=0, abs=TOLERANCE)


# --- the schedule (D1, D7; spec T1.3, T7.1-T7.3) ---


@PENDING
def test_t1_3_the_execution_lag_is_bound_to_the_label_entry_lag():
    assert schedule.execution_lag() == MODEL_LABEL_SPEC.entry_lag == 1


@PENDING
def test_t7_1_offsets_share_the_build_decision_and_phase_the_rebalances():
    assert schedule.decision_sessions(schedule.ScheduleSpec(2, 0, 1), 8) == (0, 2, 4, 6)
    assert schedule.decision_sessions(schedule.ScheduleSpec(2, 1, 1), 8) == (0, 1, 3, 5, 7)
    later_build = schedule.ScheduleSpec(every=2, offset=0, lag=1, first_decision=3)
    assert schedule.decision_sessions(later_build, 10) == (3, 5, 7, 9)  # phase from the build
    plan = schedule.execution_plan(schedule.ScheduleSpec(2, 1, 1), 8)
    assert [(e.decision, e.execution, e.kind, e.reason) for e in plan] == [
        (0, 1, schedule.BUILD, None),
        (1, 2, schedule.REBALANCE, None),
        (3, 4, schedule.REBALANCE, None),
        (5, 6, schedule.REBALANCE, None),
        (7, None, schedule.REBALANCE, schedule.NO_EXECUTION_SESSION),
    ]


@PENDING
def test_t7_2_all_offsets_form_one_family():
    spec = schedule.ScheduleSpec(every=5, offset=2, lag=1, first_decision=3)
    family = schedule.offset_family(spec)
    assert [member.offset for member in family] == [0, 1, 2, 3, 4]
    assert {(m.every, m.lag, m.first_decision) for m in family} == {(5, 1, 3)}
    assert spec in family


@PENDING
def test_d6_deferral_finds_the_first_session_every_involved_asset_closes():
    closes = {(2, 0): False, (3, 0): False}

    def has_close(session, asset):
        return closes.get((session, asset), True)

    assert schedule.next_executable_session(2, [0, 1], has_close, 6) == 4
    assert schedule.next_executable_session(2, [1], has_close, 6) == 2
    assert schedule.next_executable_session(2, [0], lambda s, a: False, 6) is None
    assert schedule.next_executable_session(5, [1], has_close, 6) is None  # no following session


# --- the engine: timing, build, gaps, offsets (D1, D3, D6, D7) ---


@PENDING
def test_t1_1_the_engine_executes_the_switch_at_the_next_close():
    assert timing.final_wealth(engine_runner, timing.switch_scenario(0.10)) == pytest.approx(
        0.950, abs=TOLERANCE
    )
    assert not timing.earns_before_execution(engine_runner)


@PENDING
def test_a_same_close_engine_is_detected():
    def same_close(scenario):
        return engine_runner(dataclasses.replace(scenario, lag=0))

    assert timing.final_wealth(same_close, timing.switch_scenario(0.10)) == pytest.approx(1.045)
    assert timing.earns_before_execution(same_close)


@PENDING
def test_t1_2_the_engine_reproduces_the_label_tie():
    result = engine.simulate(
        np.array([[NAN, NAN], [0.10, 0.0], [0.10, 0.0], [-0.10, 0.0]]),
        {0: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
        lag=1,
        cost_rate=np.zeros(2),
        constraints=Constraints(1),
        initial_wealth=1.0,
    )
    assert result.wealth[-1] - 1 == pytest.approx(108.9 / 110 - 1, abs=TOLERANCE)


@PENDING
def test_t1_4_a_decision_on_the_final_session_is_not_executed():
    result = engine.simulate(
        np.array([[NAN] * 2, [0.01, 0.0], [0.02, 0.0]]),
        {0: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
        lag=1,
        cost_rate=np.zeros(2),
        constraints=Constraints(1),
        initial_wealth=1.0,
    )
    assert result.executions[-1] == schedule.ScheduledExecution(
        2, None, schedule.REBALANCE, schedule.NO_EXECUTION_SESSION
    )
    assert len(result.steps) == 1


@PENDING
def test_d3_the_engine_builds_from_cash_at_a_cost():
    result = engine.simulate(
        np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3]),
        {0: np.array([0.5, 0.5, 0.0])},
        lag=1,
        cost_rate=np.array([0.001, 0.001, 0.0]),
        constraints=Constraints(2),
        initial_wealth=100_000.0,
    )
    assert result.sessions == (1, 2)
    assert result.wealth == pytest.approx((100_000.0, 99_900.0), abs=1e-9)
    assert result.executions[0].kind == schedule.BUILD
    assert result.steps[0].turnover_one_way == pytest.approx(1.0, abs=1e-15)


def _gap_returns():
    return np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [NAN, 0.01, 0.0], [0.03, 0.01, 0.0], [0.0] * 3]
    )


@PENDING
def test_d6_the_engine_values_across_a_gap_and_defers_the_rebalance():
    result = engine.simulate(
        _gap_returns(),
        {0: np.array([0.5, 0.5, 0.0]), 1: np.array([0.2, 0.8, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=100.0,
    )
    assert result.sessions[:4] == (1, 2, 3, 4)
    assert result.wealth[1] is None and result.wealth[2] is None
    assert result.wealth[3] == pytest.approx(103.01505, abs=TOLERANCE)
    assert result.executions[1] == schedule.ScheduledExecution(
        1, 4, schedule.REBALANCE, schedule.UNEXECUTABLE_MISSING_CLOSE
    )


@PENDING
def test_d6_the_engine_stops_on_an_unavailable_return_of_a_held_asset():
    returns = _gap_returns()
    returns[4, 0] = NAN
    closes = np.isfinite(returns)
    closes[0] = True
    closes[4, 0] = True
    result = engine.simulate(
        returns,
        {0: np.array([0.5, 0.5, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=1.0,
        has_close=closes,
    )
    assert (result.status, result.reason) == (engine.STOPPED, DISTRIBUTION_IN_GAP)


def _new_asset_gap_returns():
    # Held B closes every session; A, not yet held, has no close at 2 (its 1 -> 3 return is +5%).
    return np.array([[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [0.05, 0.01, 0.0], [0.02, 0.01, 0.0]])


@PENDING
def test_d6_a_trade_waits_for_the_close_of_an_asset_it_buys():
    result = engine.simulate(
        _new_asset_gap_returns(),
        {0: np.array([0.0, 1.0, 0.0]), 1: np.array([0.5, 0.5, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=1.0,
    )
    assert result.executions[1] == schedule.ScheduledExecution(
        1, 3, schedule.REBALANCE, schedule.UNEXECUTABLE_MISSING_CLOSE
    )
    assert result.wealth[-1] == pytest.approx(1.01**2 * (0.5 * 1.02 + 0.5 * 1.01), abs=TOLERANCE)


@PENDING
def test_t7_3_the_offset_example():
    returns = np.array([[NAN] * 3, [0.0] * 3] + [[r, 0.0, 0.0] for r in [0.10, -0.10] * 3])
    ends = {}
    for offset in (0, 1):
        spec = schedule.ScheduleSpec(every=2, offset=offset, lag=1)
        targets = engine.collect_targets(lambda history: np.array([0.5, 0.5, 0.0]), returns, spec)
        result = engine.simulate(
            returns,
            targets,
            lag=spec.lag,
            cost_rate=np.zeros(3),
            constraints=Constraints(2),
            initial_wealth=1.0,
        )
        ends[offset] = result.wealth[-1]
    assert (round(ends[0], 5), round(ends[1], 5)) == (0.98507, 0.98755)


# --- causality: no decision sees the return its holdings will earn ---


@PENDING
def test_collect_targets_shows_each_decision_only_its_own_past():
    returns = _random_path(5, sessions=12)[0]
    cash = 0
    spec = schedule.ScheduleSpec(every=1, offset=0, lag=1)

    def collect(policy, returns):
        return engine.collect_targets(policy, returns, spec)

    assert timing.future_sensitive_decisions(collect, timing.peeking_policy(cash), returns) == []

    seen = {}

    def recorder(history):
        seen[len(history) - 1] = history.copy()
        history[:] = 0.0  # a policy cannot alter the caller's returns

    collect(recorder, returns)
    for d, history in seen.items():
        np.testing.assert_array_equal(history, returns[: d + 1])


# --- replay equivalence: the engine and the oracle give one wealth path ---


@PENDING
@pytest.mark.parametrize("seed", range(4))
def test_the_engine_replays_the_oracle(seed):
    returns, targets, rates, cash = _random_path(seed, sessions=25)
    scenario = ReplayScenario(
        returns,
        {d: None if t is None else t.tolist() for d, t in targets.items()},
        cash,
        rates.tolist(),
        lag=1,
        initial_wealth=100_000.0,
        w_max=0.6,
        tau_max=0.5 if seed % 2 else None,
    )
    reports = replay(
        {"oracle": oracle_runner, "engine": engine_runner}, scenario, reference="oracle"
    )
    assert reports["engine"].equivalent, reports["engine"]


@PENDING
def test_the_engine_replays_the_oracle_across_a_gap():
    scenario = ReplayScenario(
        _gap_returns(), {0: [0.5, 0.5, 0.0], 1: [0.2, 0.8, 0.0]}, 2, [0.001, 0.0005, 0.0], lag=1
    )
    reports = replay(
        {"oracle": oracle_runner, "engine": engine_runner}, scenario, reference="oracle"
    )
    assert reports["engine"].equivalent, reports["engine"]


@PENDING
def test_the_engine_replays_the_oracle_when_a_bought_asset_has_no_close():
    scenario = ReplayScenario(
        _new_asset_gap_returns(),
        {0: [0.0, 1.0, 0.0], 1: [0.5, 0.5, 0.0]},
        2,
        [0.001] * 2 + [0.0],
        lag=1,
    )
    reports = replay(
        {"oracle": oracle_runner, "engine": engine_runner}, scenario, reference="oracle"
    )
    assert reports["engine"].equivalent, reports["engine"]


# --- D8: the reward ---


@PENDING
def test_d8_the_reward_is_log_growth_and_counts_costs_once():
    step = portfolio_step(*REBALANCE)
    reward = log_growth_reward(step)
    assert reward == pytest.approx(0.0037118964, abs=1e-10)
    assert reward == pytest.approx(math.log(1 + step.net_return), abs=1e-15)
    assert abs(reward - (math.log(1 + step.net_return) - step.cost)) > 1e-4
