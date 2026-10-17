"""
The approved T5a engine (D1-D8, 2026-10-15) beyond the packet's hand
examples (`test_portfolio_transition.py`):

- equivalence with the shares-and-cash oracle on random gappy paths, with
  deferrals, superseded targets, holds, caps, turnover limits and stops;
- equivalence of the backtest engine and the RL environment on the same
  targets and returns, so no policy gains an execution or cost advantage
  from the API it is run through;
- execution-delay mutants (lag 0, same-close earning, a peeking caller);
- drift, the cap, cost sensitivity at 0x, 1x and 2x, and deterministic
  replay.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from reference import t5a_reference as oracle
from reference import timing
from reference.replay import ReplayScenario, compare_paths, oracle_runner, replay
from t5a_runners import engine_runner, rl_env_runner

from stock_agent.portfolio import engine, schedule
from stock_agent.portfolio.transition import (
    BROAD_ETF,
    CASH_ACCOUNT,
    PROVISIONAL_RATES,
    SENSITIVITY_MULTIPLIERS,
    SINGLE_STOCK,
    TBILL_FUND,
    Constraints,
    CostModel,
    feasible_target,
    portfolio_step,
)

NAN = float("nan")
TOLERANCE = 1e-12


# --- random gappy paths: the engine against the oracle ---


def gappy_scenario(seed: int, sessions: int = 40) -> ReplayScenario:
    """
    Missing closes (and, rarely, a close whose return is unavailable) on the
    risky assets, a decision at nearly every session with sparse targets and
    holds, so deferrals, superseded targets and stops all occur.
    """

    rng = np.random.RandomState(5_000 + seed)
    n = int(rng.randint(3, 6))
    cash = int(rng.randint(n))
    daily = np.expm1(rng.normal(0.0, 0.02, size=(sessions, n)))
    daily[:, cash] = rng.choice([0.0, 0.0002])
    closes = rng.rand(sessions, n) > 0.15
    closes[:, cash] = True
    closes[0] = True
    returns = np.full((sessions, n), NAN)
    growth = np.ones(n)
    for t in range(1, sessions):
        growth *= 1.0 + daily[t]
        returns[t, closes[t]] = growth[closes[t]] - 1.0
        growth[closes[t]] = 1.0
    broken = (rng.rand(sessions, n) < 0.01) & closes
    broken[:, cash] = False
    broken[0] = False
    returns[broken] = NAN
    targets = {}
    for d in range(sessions - 1):
        draw = rng.rand()
        if draw < 0.2 and d > 0:
            continue
        if draw < 0.35 and d > 0:
            targets[d] = None
            continue
        weights = rng.dirichlet(np.ones(n))
        weights[rng.rand(n) < 0.35] = 0.0
        targets[d] = (weights / weights.sum()).tolist() if weights.sum() else np.eye(n)[cash]
    rates = rng.uniform(0.0, 0.002, size=n)
    rates[cash] = 0.0
    return ReplayScenario(
        returns=returns,
        targets={d: None if t is None else list(t) for d, t in targets.items()},
        cash_index=cash,
        cost_rate=rates.tolist(),
        lag=1,
        initial_wealth=100_000.0,
        has_close=closes,
        w_max=float(rng.choice([1.0, 0.6, 0.45])),
        tau_max=None if rng.rand() < 0.5 else 0.3,
    )


def _engine_result(scenario: ReplayScenario, **overrides) -> engine.SimulationResult:
    arguments = {
        "lag": scenario.lag,
        "cost_rate": np.asarray(scenario.cost_rate),
        "constraints": Constraints(scenario.cash_index, scenario.w_max, scenario.tau_max),
        "initial_wealth": scenario.initial_wealth,
        "has_close": scenario.has_close,
        **overrides,
    }
    targets = {d: None if t is None else np.asarray(t) for d, t in scenario.targets.items()}
    return engine.simulate(np.asarray(scenario.returns), targets, **arguments)


def _oracle_result(scenario: ReplayScenario) -> oracle.OracleRun:
    return oracle.run(
        scenario.returns,
        scenario.targets,
        lag=scenario.lag,
        cost_rate=scenario.cost_rate,
        cash_index=scenario.cash_index,
        initial_wealth=scenario.initial_wealth,
        has_close=scenario.has_close,
        w_max=scenario.w_max,
        tau_max=scenario.tau_max,
    )


@pytest.mark.parametrize("seed", range(40))
def test_the_engine_equals_the_shares_ledger_on_random_gappy_paths(seed):
    scenario = gappy_scenario(seed)
    (report,) = replay(
        {"oracle": oracle_runner, "engine": engine_runner}, scenario, reference="oracle"
    ).values()
    assert report.equivalent, report
    expected = sorted((e.decision, e.executed, e.reason) for e in _oracle_result(scenario).events)
    result = _engine_result(scenario)
    assert [(e.decision, e.execution, e.reason) for e in result.executions] == expected


def test_the_random_paths_exercise_every_d6_rule():
    reasons, statuses, unavailable = set(), set(), False
    for seed in range(40):
        result = _engine_result(gappy_scenario(seed))
        reasons |= {e.reason for e in result.executions}
        statuses.add(result.status)
        unavailable |= any(value is None for value in result.wealth)
    assert {
        schedule.UNEXECUTABLE_MISSING_CLOSE,
        schedule.SUPERSEDED_DEFERRED_TARGET,
        schedule.NO_EXECUTION_SESSION,
    } <= reasons
    assert statuses == {engine.COMPLETE, engine.STOPPED}
    assert unavailable


@pytest.mark.parametrize("seed", range(12))
def test_wealth_telescopes_and_every_span_is_booked_once(seed):
    scenario = gappy_scenario(seed)
    result = _engine_result(scenario)
    valued = [(s, w) for s, w in zip(result.sessions, result.wealth, strict=True) if w is not None]
    growth = np.prod([1.0 + step.net_return for step in result.steps])
    assert valued[-1][1] / valued[0][1] == pytest.approx(growth, rel=1e-13)
    assert [end for _, end in result.spans] == [s for s, _ in valued[1:]]
    assert [start for start, _ in result.spans] == [s for s, _ in valued[:-1]]
    for step in result.steps:
        for weights in (step.w_post_trade, step.w_next):
            assert (weights >= 0).all() and abs(weights.sum() - 1.0) <= TOLERANCE
        assert 0.0 <= step.cost < 1.0 and step.net_return <= step.gross_return + 1e-15


def test_a_hold_during_a_deferral_leaves_the_target_waiting():
    returns = np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [NAN, 0.01, 0.0], [0.03, 0.01, 0.0]]
        + [[0.02, -0.01, 0.0]]
    )
    targets = {0: np.array([0.5, 0.5, 0.0]), 1: np.array([0.2, 0.8, 0.0]), 2: None}
    result = engine.simulate(
        returns,
        targets,
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=1.0,
    )
    assert result.executions[1:] == (
        schedule.ScheduledExecution(1, 4, schedule.REBALANCE, schedule.UNEXECUTABLE_MISSING_CLOSE),
        schedule.ScheduledExecution(2, 3, schedule.REBALANCE),
    )
    at_t3 = 0.5 * 1.03 + 0.5 * 1.01**3
    assert result.wealth[-1] == pytest.approx(at_t3 * (1 + 0.2 * 0.02 - 0.8 * 0.01), abs=1e-12)


def test_a_gap_in_an_unheld_asset_leaves_wealth_available():
    returns = np.array([[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [0.05, 0.01, 0.0]])
    result = engine.simulate(
        returns,
        {0: np.array([0.0, 1.0, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=1.0,
    )
    assert result.wealth == pytest.approx((1.0, 1.01, 1.01**2), abs=1e-15)


def test_gap_valuation_is_exact_and_never_carried_flat():
    # D6's example: W = 100 split 50/50; A has no close at t1, t2 (t0 -> t3 is +3%); B +1% a day.
    returns = np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [NAN, 0.01, 0.0], [0.03, 0.01, 0.0]]
    )
    result = engine.simulate(
        returns,
        {0: np.array([0.5, 0.5, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=100.0,
    )
    assert result.wealth[:3] == (100.0, None, None)
    assert result.wealth[3] == pytest.approx(103.01505, abs=1e-12)
    assert result.spans == ((1, 4),)  # one step books the whole span


# --- the backtest engine and the RL environment: one wealth path ---


def rl_scope_scenario(
    seed: int, *, w_max: float = 1.0, tau_max: float | None = None, sessions: int = 80
):
    """The RL scope: decisions on complete sessions, every fund held once built."""

    rng = np.random.RandomState(seed)
    n, cash = 4, 3
    daily = np.zeros((sessions, n))
    daily[:, :cash] = np.expm1(rng.normal(0.0003, 0.015, size=(sessions, cash)))
    missing = rng.random_sample((sessions, cash)) < 0.08
    missing[:3] = False
    closes = np.ones((sessions, n), dtype=bool)
    closes[:, :cash] = ~missing
    returns = np.full((sessions, n), NAN)
    returns[:, cash] = 0.0
    for i in range(cash):
        growth = 1.0
        for t in range(1, sessions):
            growth *= 1.0 + daily[t, i]
            if closes[t, i]:
                returns[t, i] = growth - 1.0
                growth = 1.0
    complete = [t for t in range(1, sessions) if closes[t].all()]
    targets = {}
    for position, t in enumerate(complete[:-2]):
        if position == 0 or rng.rand() < 0.5:
            targets[t] = list(rng.dirichlet(np.ones(n)))
        else:
            targets[t] = None
    return ReplayScenario(
        returns=returns,
        targets=targets,
        cash_index=cash,
        cost_rate=(5e-4, 5e-4, 2e-4, 0.0),
        lag=1,
        initial_wealth=100_000.0,
        has_close=closes,
        w_max=w_max,
        tau_max=tau_max,
    )


RUNNERS = {"oracle": oracle_runner, "engine": engine_runner, "rl_env": rl_env_runner}


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(("w_max", "tau_max"), [(1.0, None), (0.4, None), (1.0, 0.25)])
def test_the_engine_and_the_rl_environment_give_one_wealth_path(seed, w_max, tau_max):
    scenario = rl_scope_scenario(seed, w_max=w_max, tau_max=tau_max)
    reports = replay(RUNNERS, scenario, reference="engine")
    assert reports["rl_env"].equivalent, reports["rl_env"]
    assert reports["oracle"].equivalent, reports["oracle"]


def test_the_rl_scenarios_exercise_gaps_holds_and_binding_constraints():
    scenario = rl_scope_scenario(0, w_max=0.4)
    result = _engine_result(scenario)
    assert any(value is None for value in result.wealth)
    assert any(target is None for target in scenario.targets.values())
    assert any(trade.feasibility.binding for trade in result.trades)


def test_an_api_advantage_is_detected():
    # Same targets and returns; a simulator that earned the execution session's return
    # (lag 0) or charged half the cost would differ from the engine.
    scenario = rl_scope_scenario(3)
    reference = engine_runner(scenario)

    def cheaper(w_pre, target, r_next, cost_rate, cash_index):
        return portfolio_step(w_pre, target, r_next, 0.5 * cost_rate, cash_index)

    assert not compare_paths(reference, rl_env_runner(scenario, transition=cheaper)).equivalent
    same_close = engine_runner(dataclasses.replace(scenario, lag=0))
    assert not compare_paths(reference, same_close).equivalent


# --- execution delay: mutants that earn too early or decide on the future ---


def same_close_runner(scenario: ReplayScenario):
    """
    A mutant engine whose holdings earn the return dated at their execution
    close (r_s instead of r_{s+1}): the returns are fed one row late.
    """

    late = np.vstack([np.full((1, scenario.returns.shape[1]), NAN), scenario.returns[:-1]])
    late[1] = np.where(np.isfinite(late[1]), late[1], 0.0)
    return engine_runner(dataclasses.replace(scenario, returns=late))


def test_the_engine_passes_the_lag_detector_and_both_mutants_fail_it():
    assert not timing.earns_before_execution(engine_runner)
    assert timing.earns_before_execution(lambda s: engine_runner(dataclasses.replace(s, lag=0)))
    assert timing.earns_before_execution(same_close_runner)
    assert timing.final_wealth(engine_runner, timing.switch_scenario(0.10)) == pytest.approx(
        0.950, abs=TOLERANCE
    )


def test_a_caller_that_shows_a_decision_its_next_return_is_detected():
    returns = gappy_scenario(2).returns.copy()
    returns[~np.isfinite(returns)] = 0.0
    spec = schedule.ScheduleSpec(every=1, offset=0, lag=1)
    policy = timing.peeking_policy(cash_index=0)

    def honest(policy, returns):
        return engine.collect_targets(policy, returns, spec)

    def leaky(policy, returns):
        sessions = schedule.decision_sessions(spec, len(returns))
        return {d: np.asarray(policy(returns[: d + 2].tolist())) for d in sessions}

    assert timing.future_sensitive_decisions(honest, policy, returns) == []
    assert timing.future_sensitive_decisions(leaky, policy, returns) != []


# --- drift and the cap (D5) ---


def test_drift_above_the_cap_is_kept_and_reported_never_sold():
    # A doubles while held at the 50% cap; with no decision in between nothing is sold.
    returns = np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3, [1.0, 0.0, 0.0]] + [[0.0] * 3] * 3)
    result = engine.simulate(
        returns,
        {0: np.array([0.7, 0.3, 0.0]), 4: np.array([0.7, 0.3, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2, w_max=0.5),
        initial_wealth=1.0,
    )
    build, rebalance = result.trades
    np.testing.assert_allclose(build.feasibility.weights, [0.5, 0.3, 0.2], atol=TOLERANCE)
    assert build.feasibility.distance_one_way == pytest.approx(0.2, abs=TOLERANCE)
    drifted = result.steps[2].w_next  # the close of session 4, before the rebalance there
    assert drifted[0] == pytest.approx(1.0 / 1.5, abs=TOLERANCE)  # 0.5 * 2 / (0.5 * 2 + 0.5)
    np.testing.assert_allclose(rebalance.feasibility.weights, [0.5, 0.3, 0.2], atol=TOLERANCE)
    assert (rebalance.session, rebalance.feasibility.binding) == (5, ("w_max",))


@pytest.mark.parametrize("seed", range(3))
def test_drift_matches_the_shares_ledger_and_ignores_costs(seed):
    rng = np.random.RandomState(300 + seed)
    for _ in range(100):
        n = int(rng.randint(2, 6))
        cash = int(rng.randint(n))
        w_pre, target = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        r_next = rng.uniform(-0.3, 0.3, n)
        rates = rng.uniform(0.0, 0.01, n)
        rates[cash] = 0.0
        step = portfolio_step(w_pre, target, r_next, rates, cash)
        free = portfolio_step(w_pre, target, r_next, np.zeros(n), cash)
        ledger = oracle.ledger_step(w_pre, target, r_next, rates, cash)
        np.testing.assert_allclose(step.w_next, ledger["w_next"], rtol=0, atol=TOLERANCE)
        np.testing.assert_array_equal(step.w_next, free.w_next)  # costs move wealth, not weights


# --- D2: provisional rates and the 0x / 1x / 2x sensitivity ---


def test_the_provisional_rates_are_the_approved_assumptions():
    assert dict(PROVISIONAL_RATES) == {
        SINGLE_STOCK: 0.0010,
        BROAD_ETF: 0.0005,
        TBILL_FUND: 0.0002,
        CASH_ACCOUNT: 0.0,
    }
    assert SENSITIVITY_MULTIPLIERS == (0.0, 1.0, 2.0)
    model = CostModel.provisional(
        {"VTI": BROAD_ETF, "SHV": TBILL_FUND, "AAPL": SINGLE_STOCK, "CASH": CASH_ACCOUNT}
    )
    rates = model.rates_for(["VTI", "SHV", "AAPL", "CASH"])
    np.testing.assert_array_equal(rates, [0.0005, 0.0002, 0.0010, 0.0])
    step = portfolio_step(
        np.array([0.5, 0.0, 0.3, 0.2]), np.array([0.3, 0.2, 0.3, 0.2]), np.zeros(4), rates, 3
    )
    assert step.cost * 1e4 == pytest.approx(1.4, abs=TOLERANCE)  # VTI -> SHV: 0.2 x (5 + 2) bp
    with pytest.raises(ValueError, match="asset classes"):
        CostModel.provisional({"X": "crypto"})


def test_cost_sensitivity_scales_every_charge_and_nothing_else():
    scenario = gappy_scenario(7)
    assets = [f"a{i}" for i in range(len(scenario.cost_rate))]
    model = CostModel(dict(zip(assets, scenario.cost_rate, strict=True)))
    assert [m.multiplier for m in model.sensitivity()] == [0.0, 1.0, 2.0]
    runs = engine.simulate_cost_sensitivity(
        np.asarray(scenario.returns),
        {d: None if t is None else np.asarray(t) for d, t in scenario.targets.items()},
        assets=assets,
        cost_model=model,
        lag=1,
        constraints=Constraints(scenario.cash_index, scenario.w_max, scenario.tau_max),
        initial_wealth=1.0,
        has_close=scenario.has_close,
    )
    zero, one, two = runs[0.0], runs[1.0], runs[2.0]
    assert zero.sessions == one.sessions == two.sessions
    assert all(step.cost == 0.0 for step in zero.steps)
    assert sum(step.cost for step in one.steps) > 0
    for a, b, c in zip(zero.steps, one.steps, two.steps, strict=True):
        np.testing.assert_array_equal(a.w_next, b.w_next)
        assert b.gross_return == a.gross_return and c.gross_return == a.gross_return
        assert c.cost == pytest.approx(2 * b.cost, rel=1e-12, abs=1e-18)
        assert a.net_return == pytest.approx(a.gross_return, abs=1e-15)
    final = [run.wealth[-1] for run in (zero, one, two) if run.wealth[-1] is not None]
    assert final == sorted(final, reverse=True)


# --- D7 offsets and deterministic replay ---


def test_every_offset_shares_the_build_and_reports_as_one_family():
    returns = rl_scope_scenario(1, sessions=60).returns.copy()
    returns[~np.isfinite(returns)] = 0.0
    family = schedule.offset_family(schedule.ScheduleSpec(every=5, offset=0, lag=1))
    results = []
    for spec in family:
        targets = engine.collect_targets(lambda h: np.array([0.3, 0.3, 0.2, 0.2]), returns, spec)
        results.append(
            engine.simulate(
                returns,
                targets,
                lag=spec.lag,
                cost_rate=np.array([5e-4, 5e-4, 2e-4, 0.0]),
                constraints=Constraints(3),
                initial_wealth=1.0,
            )
        )
    assert len(results) == 5
    assert {r.executions[0] for r in results} == {schedule.ScheduledExecution(0, 1, schedule.BUILD)}
    assert len({r.sessions for r in results}) == 1
    assert len({round(r.wealth[-1], 12) for r in results}) > 1  # the phases differ


def test_a_simulation_replays_bit_for_bit():
    scenario = gappy_scenario(11)
    first, second = _engine_result(scenario), _engine_result(scenario)
    assert first.sessions == second.sessions and first.wealth == second.wealth
    assert first.executions == second.executions
    for a, b in zip(first.steps, second.steps, strict=True):
        assert a.net_return == b.net_return and a.cost == b.cost
        np.testing.assert_array_equal(a.w_next, b.w_next)
    policy = timing.peeking_policy(0)
    returns = np.nan_to_num(scenario.returns)
    spec = schedule.ScheduleSpec(every=3, offset=1, lag=1)
    once = engine.collect_targets(lambda h: np.asarray(policy(h.tolist())), returns, spec)
    again = engine.collect_targets(lambda h: np.asarray(policy(h.tolist())), returns, spec)
    assert once.keys() == again.keys()
    assert all(np.array_equal(once[d], again[d]) for d in once)


def test_the_cash_label_states_the_d4_a_scope():
    assert engine.UNINVESTED_CASH_RETURN == 0.0
    assert engine.CASH_RETURN_LABEL == (
        "uninvested cash earns 0% (synthetic engineering only; group C pending)"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"lag": -1},
        {"initial_wealth": 0.0},
        {"cost_rate": np.zeros(2)},
    ],
)
def test_simulate_rejects_malformed_inputs(bad):
    scenario = gappy_scenario(0)
    with pytest.raises(ValueError):
        _engine_result(scenario, **bad)


# --- one source of portfolio economics ---


def test_no_production_module_uses_the_retired_pre_t5a_helpers():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in [*(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(name.endswith("portfolio_math") for name in names):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_a_rebalance_that_sells_a_held_asset_waits_for_its_close():
    # The target sells A entirely, so only B is targeted; A is held and has no close at 2 or 3,
    # so the trade cannot be sized until A closes again (D6: held OR targeted).
    returns = np.array(
        [[NAN] * 3, [0.0] * 3, [NAN, 0.01, 0.0], [NAN, 0.01, 0.0], [0.03, 0.01, 0.0], [0.0] * 3]
    )
    result = engine.simulate(
        returns,
        {0: np.array([0.5, 0.5, 0.0]), 1: np.array([0.0, 1.0, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=100.0,
    )
    assert result.executions[1] == schedule.ScheduledExecution(
        1, 4, schedule.REBALANCE, schedule.UNEXECUTABLE_MISSING_CLOSE
    )
    assert result.trades[1].session == 4
    assert result.wealth[-1] == pytest.approx(103.01505, abs=1e-12)


@pytest.mark.parametrize("cash_rate", [0.0, 0.001, 0.5])
def test_the_cash_entry_of_cost_rate_is_never_charged(cash_rate):
    # D2: the cash account is the free funding leg, whatever its cost_rate entry says.
    rng = np.random.RandomState(42)
    for _ in range(50):
        w_pre, target = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
        r_next = rng.uniform(-0.1, 0.1, 4)
        free = portfolio_step(w_pre, target, r_next, np.array([1e-3, 5e-4, 2e-4, 0.0]), 3)
        charged = portfolio_step(w_pre, target, r_next, np.array([1e-3, 5e-4, 2e-4, cash_rate]), 3)
        assert charged.cost == free.cost and charged.net_return == free.net_return
    returns = np.array([[NAN] * 2, [0.0] * 2, [0.0] * 2, [0.0] * 2])
    sold = engine.simulate(
        returns,
        {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])},
        lag=1,
        cost_rate=np.array([0.001, cash_rate]),
        constraints=Constraints(1),
        initial_wealth=100_000.0,
    )
    assert sold.wealth[-1] == pytest.approx(100_000.0 * 0.999 * 0.999)  # build, then the sale


def test_interpretation_b_the_turnover_limit_also_limits_the_build():
    # Confirmed by the owner on 2026-10-17: one feasibility map for every target, the build
    # included. A 50/50 build from cash with tau_max = 0.25 executes a quarter of the way.
    record = feasible_target(
        np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 1.0]), Constraints(2, tau_max=0.25)
    )
    np.testing.assert_allclose(record.weights, [0.125, 0.125, 0.75], rtol=0, atol=TOLERANCE)
    assert record.alpha == pytest.approx(0.25) and record.binding == ("tau_max",)
    result = engine.simulate(
        np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3]),
        {0: np.array([0.5, 0.5, 0.0])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2, tau_max=0.25),
        initial_wealth=1.0,
    )
    np.testing.assert_allclose(result.steps[0].w_post_trade, [0.125, 0.125, 0.75], atol=TOLERANCE)


def test_interpretation_a_a_hold_does_not_supersede_a_deferred_build():
    # Confirmed by the owner on 2026-10-17: a deferred build survives a hold.
    # A has no close at the build's execution close (1) or at 2; the holds are due at 2 and 3.
    returns = np.array([[NAN] * 3, [NAN, 0.0, 0.0], [NAN, 0.01, 0.0], [0.02, 0.01, 0.0], [0.0] * 3])
    result = engine.simulate(
        returns,
        {0: np.array([0.5, 0.5, 0.0]), 1: None, 2: None},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2),
        initial_wealth=1.0,
    )
    build = result.executions[0]
    assert (build.execution, build.reason) == (3, schedule.UNEXECUTABLE_MISSING_CLOSE)
    assert result.trades[0].session == 3
