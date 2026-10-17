"""
Build reporting (D5 as confirmed by the owner on 2026-10-17): the build's
executed weights, its uninvested cash, and whether a registered turnover cap
left the construction incomplete, with the sessions until it was fully built.
The engine and the RL environment report the same build from the same
targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from stock_agent.portfolio import construction, engine
from stock_agent.portfolio.backtest_summary import summary_record
from stock_agent.portfolio.transition import (
    CAP,
    TURNOVER_LIMIT,
    Constraints,
    feasible_target,
)
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import replay as rl_replay
from stock_agent.rl import scenarios as sc
from stock_agent.rl import transition as tr
from stock_agent.rl.policies import BuyAndHold, ConstantMix

NAN = float("nan")
TOLERANCE = 1e-12
FIFTY_FIFTY = np.array([0.5, 0.5, 0.0])


def flat_returns(sessions: int, n: int = 3) -> np.ndarray:
    table = np.zeros((sessions, n))
    table[0] = NAN
    return table


def simulate(targets, *, w_max=1.0, tau_max=None, sessions=12):
    return engine.simulate(
        flat_returns(sessions),
        targets,
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2, w_max, tau_max),
        initial_wealth=1.0,
    )


def test_a_build_the_turnover_limit_binds_is_partial_and_names_its_completion():
    # The packet's interpretation (b), confirmed 2026-10-17: a 50/50 build from cash
    # with tau_max = 0.25 executes (0.125, 0.125, 0.75 cash). Rebalancing to 50/50
    # every two sessions on flat returns then moves 1/3, 1/2 and finally all of the
    # remaining way: the fourth execution (session 7) is the first the limit allows.
    targets = {d: FIFTY_FIFTY for d in (0, 2, 4, 6)}
    report = simulate(targets, tau_max=0.25).build_report(cash_index=2)
    assert report.session == 1
    np.testing.assert_allclose(report.executed, [0.125, 0.125, 0.75], atol=TOLERANCE)
    assert report.requested == (0.5, 0.5, 0.0)
    assert report.uninvested_cash == pytest.approx(0.75)
    assert report.binding == (TURNOVER_LIMIT,) and report.alpha == pytest.approx(0.25)
    assert report.incomplete_construction
    assert (report.fully_built_session, report.sessions_until_fully_built) == (7, 6)


def test_a_portfolio_that_trades_once_under_the_limit_is_never_fully_built():
    report = simulate({0: FIFTY_FIFTY, 2: None, 4: None}, tau_max=0.25).build_report(2)
    assert report.incomplete_construction
    assert report.fully_built_session is None and report.sessions_until_fully_built is None
    assert report.uninvested_cash == pytest.approx(0.75)


def test_without_a_turnover_limit_the_build_is_complete_at_once():
    report = simulate({0: FIFTY_FIFTY}).build_report(2)
    assert not report.incomplete_construction and report.binding == ()
    assert (report.fully_built_session, report.sessions_until_fully_built) == (1, 0)
    assert report.uninvested_cash == 0.0 and report.alpha == 1.0


def test_the_caps_excess_is_reported_as_uninvested_cash():
    # D5: excess above the cap goes to cash. (0.7, 0.3) under w_max = 0.5 executes
    # (0.5, 0.3, 0.2 cash); the cap alone never makes the construction incomplete.
    report = simulate({0: np.array([0.7, 0.3, 0.0])}, w_max=0.5).build_report(2)
    np.testing.assert_allclose(report.executed, [0.5, 0.3, 0.2], atol=TOLERANCE)
    assert report.uninvested_cash == pytest.approx(0.2)
    assert report.binding == (CAP,) and not report.incomplete_construction
    assert report.sessions_until_fully_built == 0


def test_a_portfolio_that_never_trades_has_no_build():
    assert construction.build_report([], cash_index=2) is None
    assert simulate({0: None}).build_report(2) is None


def test_trades_out_of_order_or_a_wrong_cash_index_are_refused():
    record = feasible_target(FIFTY_FIFTY, np.array([0.0, 0.0, 1.0]), Constraints(2))
    with pytest.raises(ValueError, match="execution order"):
        construction.build_report([(5, record), (3, record)], 2)
    with pytest.raises(ValueError, match="outside"):
        construction.build_report([(5, record)], 3)


def test_the_report_rejects_inconsistent_fields():
    with pytest.raises(ValueError, match="incomplete exactly"):
        construction.BuildReport(1, (1.0, 0.0), (1.0, 0.0), 0.0, (), 1.0, True, None, None)
    with pytest.raises(ValueError, match="complete at once"):
        construction.BuildReport(1, (1.0, 0.0), (1.0, 0.0), 0.0, (), 1.0, False, 3, 2)
    with pytest.raises(ValueError, match="go together"):
        construction.BuildReport(1, (1.0, 0.0), (1.0, 0.0), 0.0, (), 1.0, False, 1, None)


def test_the_backtest_summary_carries_the_build_report():
    from datetime import UTC, datetime

    result = simulate({0: FIFTY_FIFTY, 2: None}, tau_max=0.25)
    record = summary_record(
        result,
        constraints=Constraints(2, tau_max=0.25),
        run_id="build-test",
        policy="fifty_fifty_once",
        evidence_status="synthetic_engineering_only",
        synthetic=True,
        generated_at=datetime(2026, 10, 17, tzinfo=UTC),
        code_sha=None,
        assumptions={},
    )
    assert record["build"]["incomplete_construction"] is True
    assert record["build"]["sessions_until_fully_built"] is None
    assert record["build"]["uninvested_cash"] == pytest.approx(0.75)
    np.testing.assert_allclose(record["build"]["executed_weights"], [0.125, 0.125, 0.75])


def _env_build(policy, *, max_turnover):
    data = sc.generate_scenario("calm", 3)
    env = rl_env.AllocationEnv(
        data,
        start=400,
        stop=520,
        transition=tr.portfolio_step,
        transition_info=tr.APPROVED_TRANSITION,
        builder=ob.ObservationBuilder(
            ob.FeatureSpec(),
            ob.FeatureNormalizer.identity(
                ob.FeatureSpec().market_names(sc.ASSET_NAMES, sc.CASH_INDEX)
            ),
            sc.ASSET_NAMES,
            sc.CASH_INDEX,
        ),
        config=rl_env.EnvConfig(rebalance_every=5, max_weight=0.5, max_turnover=max_turnover),
    )
    record = rl_replay.rollout(env, policy)
    trades = [(row.session, row.feasibility) for row in record.ledger if row.feasibility]
    return construction.build_report(trades, sc.CASH_INDEX), record


@pytest.mark.parametrize("max_turnover", [None, 0.25])
def test_the_rl_environment_and_the_engine_report_the_same_build(max_turnover):
    weights = [0.6, 0.3, 0.1, 0.0]
    env_report, record = _env_build(ConstantMix("mix", weights), max_turnover=max_turnover)
    data = sc.generate_scenario("calm", 3)
    complete = data.complete_sessions(400, 520)
    # The same targets through the engine, on the environment's complete sessions.
    sessions = [int(s) for s in complete[: record.ledger[-1].index + 2]]
    returns = np.full((len(sessions), 4), NAN)
    for k in range(1, len(sessions)):
        returns[k] = data.span_returns(sessions[k - 1], sessions[k])
    targets = {d.decision_index: d.target for d in record.decisions}
    result = engine.simulate(
        returns,
        targets,
        lag=1,
        cost_rate=np.array(sc.BASE_COST_RATES),
        constraints=Constraints(sc.CASH_INDEX, 0.5, max_turnover),
        initial_wealth=1.0,
    )
    engine_report = result.build_report(sc.CASH_INDEX)
    np.testing.assert_allclose(env_report.executed, engine_report.executed, atol=TOLERANCE)
    assert env_report.session == sessions[engine_report.session]
    assert env_report.incomplete_construction == engine_report.incomplete_construction
    assert env_report.incomplete_construction == (max_turnover is not None)
    assert env_report.uninvested_cash == pytest.approx(engine_report.uninvested_cash)
    if max_turnover is not None:
        assert env_report.fully_built_session == sessions[engine_report.fully_built_session]


def test_a_buy_and_hold_under_a_turnover_cap_stays_partly_in_cash_in_the_environment():
    report, _ = _env_build(BuyAndHold("bh", [0.6, 0.3, 0.1, 0.0]), max_turnover=0.25)
    # Capped to (0.5, 0.3, 0.1, 0.1 cash), then a quarter-turnover build from cash:
    # tau = 0.9, alpha = 0.25 / 0.9.
    alpha = 0.25 / 0.9
    np.testing.assert_allclose(
        report.executed, [0.5 * alpha, 0.3 * alpha, 0.1 * alpha, 1 - 0.9 * alpha], atol=1e-12
    )
    assert report.binding == (CAP, TURNOVER_LIMIT)
    assert report.incomplete_construction and report.fully_built_session is None
