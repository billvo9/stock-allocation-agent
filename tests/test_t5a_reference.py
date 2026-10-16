"""
The T5a reference oracle (`tests/reference/`) against every number in the
decision packet (`docs/research/t5a_open_decisions.md`), and the timing
detectors and replay harness against planted defects.

They check test-only code: the oracle implements the packet's options that
the owner approved on 2026-10-15 (D1-D8). The production engine is held to
the same numbers by `test_portfolio_transition.py`.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from reference import t5a_reference as oracle
from reference import timing
from reference.replay import ReplayPath, ReplayScenario, compare_paths, oracle_runner, replay

REFERENCE = Path(__file__).resolve().parent / "reference"
NAN = float("nan")


def _imports(path: Path) -> set[str]:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_the_oracle_is_independent_of_the_engine():
    imported = _imports(REFERENCE / "t5a_reference.py")
    assert imported <= {"__future__", "math", "collections.abc", "dataclasses"}, imported
    for name in ("replay.py", "timing.py"):
        assert not any(m.startswith("stock_agent") for m in _imports(REFERENCE / name)), name


# --- D1: execution timing ---


def test_d1_the_switch_earns_0_950_at_lag_1_and_1_045_at_the_same_close():
    assert timing.final_wealth(oracle_runner, timing.switch_scenario(0.10)) == pytest.approx(
        0.950, abs=1e-12
    )
    same_close = timing.final_wealth(oracle_runner, timing.switch_scenario(0.10, lag=0))
    assert same_close == pytest.approx(1.045, abs=1e-12)


def test_d1_the_label_tie_lag_1_equals_the_label_and_lag_0_does_not():
    # P = 100, 110, 121, 108.9; enter on decision 0, exit to cash on decision 2.
    returns = np.array([[NAN, NAN], [0.10, 0.0], [0.10, 0.0], [-0.10, 0.0]])
    targets = {0: [1.0, 0.0], 2: [0.0, 1.0]}
    label = 108.9 / 110 - 1

    lagged = oracle.run(returns, targets, lag=1, cost_rate=[0.0, 0.0], cash_index=1)
    assert lagged.wealth[-1] - 1 == pytest.approx(label, abs=1e-12)
    assert label == pytest.approx(-0.0100, abs=1e-12)

    same_close = oracle.run(returns, targets, lag=0, cost_rate=[0.0, 0.0], cash_index=1)
    assert same_close.wealth_at(3) - 1 == pytest.approx(0.21, abs=1e-12)


# --- D2: costs ---


def test_d2_the_rebalance_example():
    step = oracle.ledger_step(
        [0.55, 0.25, 0.20], [0.40, 0.40, 0.20], [0.02, -0.01, 0.0001], [0.001, 0.001, 0.0], 2
    )
    assert step["cost"] == pytest.approx(0.0003, abs=1e-15)
    assert step["gross_return"] == pytest.approx(0.00402, abs=1e-14)
    assert step["net_return"] == pytest.approx(0.0037187940, abs=1e-12)
    assert 100_000 * (1 + step["net_return"]) == pytest.approx(100_371.8794, abs=1e-6)
    assert [round(w, 5) for w in step["w_next"]] == [0.40637, 0.39441, 0.19922]
    assert step["turnover_one_way"] == pytest.approx(0.15, abs=1e-15)
    assert step["traded_notional"] == pytest.approx(0.30, abs=1e-15)
    # The additive alternative (W(1 + R - C)) is not the recommended rule.
    assert 100_000 * (1 + 0.00402 - 0.0003) == pytest.approx(100_372.00, abs=1e-6)
    assert abs(100_000 * step["net_return"] - 100_000 * (0.00402 - 0.0003)) > 0.1


def test_d2_selling_everything_into_cash_costs_the_risky_leg_only():
    step = oracle.ledger_step([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.001, 0.0], 1)
    assert 100_000 * (1 + step["net_return"]) == pytest.approx(99_900.0, abs=1e-9)
    # Charging the cash leg too counts the one trade twice: 99,800.
    assert 100_000 * (1 - 0.001 * (1.0 + 1.0)) == pytest.approx(99_800.0)
    with pytest.raises(ValueError, match="funding leg"):
        oracle.ledger_step([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.001, 0.001], 1)


def test_d2_costs_multiply_rather_than_subtract():
    # T2.3: C = 0.001 and R = +10% give 1.0989, not the additive 1.0990.
    step = oracle.ledger_step([0.0, 1.0], [1.0, 0.0], [0.10, 0.0], [0.001, 0.0], 1)
    assert 1 + step["net_return"] == pytest.approx(1.0989, abs=1e-12)
    assert abs((1 + step["net_return"]) - 1.0990) > 1e-9


def test_d2_cross_class_moves_cost_both_legs_at_their_own_rates():
    # Assets (VTI 5 bp, SHV 2 bp, AGG 5 bp, cash); move 20% of wealth out of VTI.
    rates = [0.0005, 0.0002, 0.0005, 0.0]
    to_shv = oracle.ledger_step([0.8, 0.0, 0.0, 0.2], [0.6, 0.2, 0.0, 0.2], [0.0] * 4, rates, 3)
    to_agg = oracle.ledger_step([0.8, 0.0, 0.0, 0.2], [0.6, 0.0, 0.2, 0.2], [0.0] * 4, rates, 3)
    assert to_shv["cost"] * 1e4 == pytest.approx(1.4, abs=1e-12)
    assert to_agg["cost"] * 1e4 == pytest.approx(2.0, abs=1e-12)


# --- D3: the build ---


def test_d3_every_run_builds_from_cash_and_pays_for_it():
    returns = np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3])
    result = oracle.run(
        returns,
        {0: [0.5, 0.5, 0.0]},
        lag=1,
        cost_rate=[0.001, 0.001, 0.0],
        cash_index=2,
        initial_wealth=100_000,
    )
    assert result.sessions == [1, 2]  # nothing is valued before the build's close
    assert result.wealth == pytest.approx([100_000.0, 99_900.0], abs=1e-9)
    (build,) = result.fills
    assert build.drifted == [0.0, 0.0, 1.0]
    assert build.cost == pytest.approx(0.001, abs=1e-15)
    assert build.turnover_one_way == pytest.approx(1.0, abs=1e-15)
    assert build.traded_notional(2) == pytest.approx(1.0, abs=1e-15)


# --- D5: constraints ---


def test_d5_the_turnover_limit_trades_part_way():
    executed = oracle.feasible([0.30, 0.50, 0.20], [0.70, 0.10, 0.20], 2, tau_max=0.25)
    assert executed == pytest.approx([0.45, 0.35, 0.20], abs=1e-12)
    assert 0.25 / 0.40 == pytest.approx(0.625)


def test_d5_the_cap_sends_the_excess_to_cash():
    capped = oracle.feasible([0.70, 0.20, 0.10], [0.0, 0.0, 1.0], 2, w_max=0.5)
    assert capped == pytest.approx([0.50, 0.20, 0.30], abs=1e-12)
    # The proportional alternative would give (0.50, 0.333, 0.167).
    proportional = [0.50, 0.20 + 0.20 * 0.20 / 0.30, 0.10 + 0.20 * 0.10 / 0.30]
    assert [round(w, 3) for w in proportional] == [0.5, 0.333, 0.167]
    assert not np.allclose(capped, proportional)


def test_d5_a_target_inside_the_set_is_unchanged():
    target = [0.3, 0.3, 0.4]
    assert oracle.feasible(target, [0.2, 0.4, 0.4], 2, w_max=0.5, tau_max=0.5) == target


@pytest.mark.parametrize("target", [[0.6, 0.6, -0.2], [0.5, 0.5, 0.1], [NAN, 0.5, 0.5], [0.5, 0.5]])
def test_d5_a_rule_breaking_target_is_an_error_never_repaired(target):
    with pytest.raises(oracle.InvalidTarget):
        oracle.feasible(target, [0.0, 0.0, 1.0], 2)


def test_d5_drift_above_the_cap_is_kept_between_rebalances():
    returns = np.array([[NAN] * 3, [0.0] * 3, [0.20, 0.0, 0.0], [0.0] * 3])
    result = oracle.run(
        returns, {0: [0.5, 0.5, 0.0]}, lag=1, cost_rate=[0.0] * 3, cash_index=2, w_max=0.5
    )
    assert result.weights[result.sessions.index(2)][0] == pytest.approx(0.6 / 1.1)
    assert len(result.fills) == 1


# --- D6: missing prices ---


def _gap_returns():
    # Build at close 1 (t0); A has no close at 2 and 3; A's 1 -> 4 return is +3%.
    return np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [NAN, 0.01, 0.0], [0.03, 0.01, 0.0], [0.0] * 3]
    )


def test_d6_holdings_are_valued_exactly_across_a_gap():
    result = oracle.run(
        _gap_returns()[:5],
        {0: [0.5, 0.5, 0.0]},
        lag=1,
        cost_rate=[0.0] * 3,
        cash_index=2,
        initial_wealth=100,
    )
    assert result.sessions == [1, 2, 3, 4]
    assert result.wealth[1] is None and result.wealth[2] is None  # never carried flat
    assert result.wealth_at(4) == pytest.approx(50 * 1.03 + 50 * 1.01**3, abs=1e-12)
    assert result.wealth_at(4) == pytest.approx(103.01505, abs=1e-12)
    assert oracle.log_rewards(result) == [(4, pytest.approx(math.log(1.0301505), abs=1e-14))]


def test_d6_a_rebalance_waits_for_every_involved_close():
    result = oracle.run(
        _gap_returns(),
        {0: [0.5, 0.5, 0.0], 1: [0.2, 0.8, 0.0]},
        lag=1,
        cost_rate=[0.0] * 3,
        cash_index=2,
    )
    assert [(e.decision, e.planned, e.executed, e.reason) for e in result.events] == [
        (0, 1, 1, None),
        (1, 2, 4, oracle.UNEXECUTABLE_MISSING_CLOSE),
    ]
    assert result.fills[-1].session == 4


def test_d6_an_unavailable_return_on_a_held_asset_stops_the_run():
    returns = _gap_returns()
    returns[4, 0] = NAN  # A closes at 4, but its span return is unavailable
    closes = np.isfinite(returns)
    closes[0] = True
    closes[4, 0] = True
    held = oracle.run(
        returns, {0: [0.5, 0.5, 0.0]}, lag=1, cost_rate=[0.0] * 3, cash_index=2, has_close=closes
    )
    assert (held.status, held.reason) == (oracle.STOPPED, oracle.DISTRIBUTION_IN_GAP)
    assert held.sessions == [1, 2, 3]
    not_held = oracle.run(
        returns, {0: [0.0, 1.0, 0.0]}, lag=1, cost_rate=[0.0] * 3, cash_index=2, has_close=closes
    )
    assert not_held.status == oracle.COMPLETE


def test_d6_a_trade_waits_for_the_close_of_an_asset_it_buys():
    # B is held and closes; A, bought by decision 1, has no close at 2.
    returns = np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [0.05, 0.01, 0.0], [0.02, 0.01, 0.0]]
    )
    result = oracle.run(
        returns, {0: [0.0, 1.0, 0.0], 1: [0.5, 0.5, 0.0]}, lag=1, cost_rate=[0.0] * 3, cash_index=2
    )
    assert result.events[1] == oracle.Event(1, 2, 3, oracle.UNEXECUTABLE_MISSING_CLOSE)
    assert result.wealth_at(2) == pytest.approx(1.01, abs=1e-15)  # B alone is valued at 2
    # A's +5% from close 1 to close 3 is never earned: it was bought at close 3.
    assert result.wealth[-1] == pytest.approx(1.01**2 * (0.5 * 1.02 + 0.5 * 1.01), abs=1e-12)


def test_d6_a_newer_hold_leaves_a_deferred_target_waiting():
    # A hold (None) trades nothing and is not a newer target, so it supersedes nothing.
    returns = _gap_returns()
    returns[5] = [0.02, -0.01, 0.0]
    result = oracle.run(
        returns,
        {0: [0.5, 0.5, 0.0], 1: [0.2, 0.8, 0.0], 2: None},
        lag=1,
        cost_rate=[0.0] * 3,
        cash_index=2,
    )
    assert result.events[1:] == [
        oracle.Event(2, 3, 3, None),
        oracle.Event(1, 2, 4, oracle.UNEXECUTABLE_MISSING_CLOSE),
    ]
    at_t3 = 0.5 * 1.03 + 0.5 * 1.01**3
    assert result.wealth[-1] == pytest.approx(at_t3 * (1 + 0.2 * 0.02 - 0.8 * 0.01), abs=1e-12)


def test_d6_a_newer_decision_supersedes_a_deferred_target():
    # Approved 2026-10-15: the newer target replaces the deferred one (superseded_deferred_target).
    returns = _gap_returns()
    returns[5] = [0.02, -0.01, 0.0]
    result = oracle.run(
        returns,
        {0: [0.5, 0.5, 0.0], 1: [0.2, 0.8, 0.0], 2: [0.4, 0.6, 0.0]},
        lag=1,
        cost_rate=[0.0] * 3,
        cash_index=2,
    )
    assert result.events[1:] == [
        oracle.Event(1, 2, None, oracle.SUPERSEDED_DEFERRED_TARGET),
        oracle.Event(2, 3, 4, oracle.UNEXECUTABLE_MISSING_CLOSE),
    ]
    at_t3 = 0.5 * 1.03 + 0.5 * 1.01**3
    assert result.wealth[-1] == pytest.approx(at_t3 * (1 + 0.4 * 0.02 - 0.6 * 0.01), abs=1e-12)


# --- D7: schedule and offsets ---


def _alternating():
    # A alternates +10% / -10% for six sessions after the build at close 1; B is flat.
    return np.array([[NAN] * 3, [0.0] * 3] + [[r, 0.0, 0.0] for r in [0.10, -0.10] * 3])


def test_d7_offsets_share_the_build_and_differ_only_in_phase():
    assert oracle.schedule_decisions(0, 2, 0, 8) == [0, 2, 4, 6]
    assert oracle.schedule_decisions(0, 2, 1, 8) == [0, 1, 3, 5, 7]
    assert oracle.schedule_decisions(3, 2, 0, 10) == [3, 5, 7, 9]  # phase from the build
    ends = {}
    for offset in (0, 1):
        decisions = oracle.schedule_decisions(0, 2, offset, 8)
        result = oracle.run(
            _alternating(),
            {d: [0.5, 0.5, 0.0] for d in decisions},
            lag=1,
            cost_rate=[0.0] * 3,
            cash_index=2,
        )
        assert result.fills[0].session == 1
        ends[offset] = result.wealth[-1]
    assert ends[0] == pytest.approx(0.995**3, abs=1e-12)
    assert ends[1] == pytest.approx(1.05 * 0.995**2 * 0.95, abs=1e-12)
    assert (round(ends[0], 5), round(ends[1], 5)) == (0.98507, 0.98755)
    assert round(ends[1] - ends[0], 4) == 0.0025
    hold = oracle.run(
        _alternating(), {0: [0.5, 0.5, 0.0]}, lag=1, cost_rate=[0.0] * 3, cash_index=2
    )
    assert round(hold.wealth[-1], 5) == 0.98515


def test_d7_a_decision_past_the_data_is_recorded_unexecuted():
    decisions = oracle.schedule_decisions(0, 2, 1, 8)
    targets = {d: [0.5, 0.5, 0.0] for d in decisions}
    result = oracle.run(_alternating(), targets, lag=1, cost_rate=[0.0] * 3, cash_index=2)
    assert result.events[-1] == oracle.Event(7, 8, None, oracle.NO_EXECUTION_SESSION)


# --- D8: the reward ---


def test_d8_the_reward_is_log_growth_with_costs_counted_once():
    step = oracle.ledger_step(
        [0.55, 0.25, 0.20], [0.40, 0.40, 0.20], [0.02, -0.01, 0.0001], [0.001, 0.001, 0.0], 2
    )
    assert step["reward"] == pytest.approx(0.0037118964, abs=1e-10)
    assert step["reward"] == pytest.approx(math.log(1 + step["net_return"]), abs=1e-15)
    double_counted = step["reward"] - step["cost"]
    assert double_counted == pytest.approx(0.0034118964, abs=1e-10)
    assert abs(step["reward"] - double_counted) > 1e-4


def test_d8_the_drawdown_variant_is_a_different_objective():
    # Alternative (b), not the contract: a -1.5% day deepening drawdown 2.0% -> 3.5%, lambda 2.
    variant = math.log(1 - 0.015) - 2.0 * max(0.0, 0.035 - 0.020)
    assert round(variant, 5) == -0.04511
    assert abs(variant - math.log(1 - 0.015)) > 0.01


# --- the oracle's two paths agree with each other ---


def _random_case(seed, sessions=12, assets=4):
    rng = np.random.RandomState(seed)  # legacy stream: stable across numpy versions
    cash = int(rng.randint(assets))
    returns = rng.uniform(-0.08, 0.08, size=(sessions, assets))
    returns[:, cash] = rng.uniform(0.0, 0.0005, size=sessions)
    returns[0] = NAN
    rates = rng.uniform(0.0, 0.002, size=assets)
    rates[cash] = 0.0
    targets = {d: rng.dirichlet(np.ones(assets)).tolist() for d in range(sessions - 1)}
    return returns, targets, rates.tolist(), cash


@pytest.mark.parametrize("seed", range(5))
def test_the_run_loop_agrees_with_chained_single_step_ledgers(seed):
    returns, targets, rates, cash = _random_case(seed)
    result = oracle.run(returns, targets, lag=1, cost_rate=rates, cash_index=cash)
    weights = [1.0 if i == cash else 0.0 for i in range(len(rates))]
    wealth = 1.0
    for session in range(1, len(returns) - 1):
        assert result.wealth_at(session) == pytest.approx(wealth, rel=0, abs=1e-12)
        step = oracle.ledger_step(weights, targets[session - 1], returns[session + 1], rates, cash)
        wealth *= 1 + step["net_return"]
        weights = step["w_next"]
    assert result.wealth[-1] == pytest.approx(wealth, rel=0, abs=1e-12)


# --- the timing detectors catch planted defects ---


def test_the_lag_detector_passes_the_oracle_and_flags_a_same_close_runner():
    def same_close(scenario):
        return oracle_runner(dataclasses.replace(scenario, lag=0))

    assert not timing.earns_before_execution(oracle_runner)
    assert timing.earns_before_execution(same_close)


def test_the_causality_detector_passes_the_honest_caller_and_flags_a_peek():
    returns, _, _, cash = _random_case(7)
    decisions = list(range(1, len(returns) - 1))
    policy = timing.peeking_policy(cash)

    def honest(policy, returns):
        return oracle.decide(policy, returns, decisions)

    def passes_r_next(policy, returns):
        return {d: policy([list(row) for row in returns[: d + 2]]) for d in decisions}

    assert timing.future_sensitive_decisions(honest, policy, returns) == []
    assert timing.future_sensitive_decisions(passes_r_next, policy, returns) == decisions


# --- the replay harness ---


def _scenario(seed=3):
    returns, targets, rates, cash = _random_case(seed)
    return ReplayScenario(returns, targets, cash, rates, lag=1, initial_wealth=100_000.0)


def test_replay_finds_a_runner_equivalent_to_itself():
    (report,) = replay(
        {"oracle": oracle_runner, "again": oracle_runner}, _scenario(), reference="oracle"
    ).values()
    assert report.equivalent and report.max_difference == 0.0 and report.first_divergence is None


def test_replay_locates_the_first_divergence():
    def doubled_costs(scenario):
        rates = [2 * rate for rate in scenario.cost_rate]
        return oracle_runner(dataclasses.replace(scenario, cost_rate=rates))

    report = replay(
        {"oracle": oracle_runner, "mutant": doubled_costs}, _scenario(), reference="oracle"
    )
    assert not report["mutant"].equivalent
    assert report["mutant"].first_divergence == 2  # the build at close 1 is first felt at 2


def test_replay_rejects_different_sessions_status_or_availability():
    base = ReplayPath((1, 2, 3), (1.0, None, 1.1))
    assert compare_paths(base, base).equivalent
    assert not compare_paths(base, ReplayPath((1, 2), (1.0, None))).equivalent
    assert not compare_paths(base, ReplayPath((1, 2, 3), (1.0, 1.05, 1.1))).equivalent
    stopped = ReplayPath((1, 2, 3), (1.0, None, 1.1), oracle.STOPPED, oracle.DISTRIBUTION_IN_GAP)
    assert not compare_paths(base, stopped).equivalent
    assert not compare_paths(base, ReplayPath((1, 2, 3), (1.0, None, 1.1 + 1e-9))).equivalent
