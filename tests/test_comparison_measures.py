"""
The approved measures of one evaluated episode, against hand-computed
values on a deterministic scenario: terminal wealth and total return, the
F6.5 drawdown with peak, trough, recovery and duration (and the incomplete-
history label), F2.4 turnover and traded notional, D2 costs, and the build.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.comparison import measures
from stock_agent.portfolio.transition import CAP, TURNOVER_LIMIT
from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import scenarios as sc
from stock_agent.rl.policies import BuyAndHold, ConstantMix
from stock_agent.rl.replay import rollout
from stock_agent.rl.transition import APPROVED_TRANSITION, portfolio_step

SESSIONS = 200
START, STOP = 80, 160
SPEC = ob.FeatureSpec()
BUILDER = ob.ObservationBuilder(
    SPEC,
    ob.FeatureNormalizer.identity(SPEC.market_names(sc.ASSET_NAMES, sc.CASH_INDEX)),
    sc.ASSET_NAMES,
    sc.CASH_INDEX,
)


def scenario(*, bond_gap_at: int | None = None) -> sc.ScenarioData:
    """Equity +1% a session, except -2% at sessions 100-104; bond, fund and cash flat."""

    returns = np.zeros((SESSIONS, 4))
    returns[:, 0] = 0.01
    returns[100:105, 0] = -0.02
    has_close = np.ones((SESSIONS, 4), dtype=bool)
    if bond_gap_at is not None:
        has_close[bond_gap_at, 1] = False
        returns[bond_gap_at, 1] = np.nan
    return sc.ScenarioData(
        scenario_id="calm",
        generator_version=sc.GENERATOR_VERSION,
        seed=999,
        returns=returns,
        has_close=has_close,
        obs_lag=np.zeros((SESSIONS, 4), dtype=np.int64),
        cost_multiplier=np.ones(SESSIONS),
        regime=np.array(["calm"] * SESSIONS, dtype=object),
        latent_returns=np.nan_to_num(returns),
    )


def episode(policy, *, data=None, max_weight=1.0, max_turnover=None):
    env = rl_env.AllocationEnv(
        data or scenario(),
        start=START,
        stop=STOP,
        transition=portfolio_step,
        transition_info=APPROVED_TRANSITION,
        builder=BUILDER,
        config=rl_env.EnvConfig(
            rebalance_every=5, max_weight=max_weight, max_turnover=max_turnover
        ),
    )
    return rollout(env, policy)


def all_equity():
    return BuyAndHold("all_equity", [1.0, 0.0, 0.0, 0.0])


def test_terminal_wealth_and_total_return_by_hand():
    # Complete sessions 80..159; the schedule's last observation is index 75, so the
    # ledger books sessions 80..155 and the last valued close is 156. The build
    # executes at 81 (one session after the first decision at 80) and pays 5 bp.
    record = episode(all_equity())
    result = measures.episode_measures(record, sc.CASH_INDEX)
    assert (result["start_session"], result["end_session"], result["span_sessions"]) == (
        81,
        156,
        75,
    )
    expected = (1 - 0.0005) * 1.01**70 * 0.98**5  # 75 sessions earned, 5 of them -2%
    assert result["terminal_wealth"] == pytest.approx(expected, rel=1e-12)
    assert result["total_return"] == pytest.approx(expected - 1.0, rel=1e-12)
    assert measures.terminal_measures(record)["terminal_wealth"] == result["terminal_wealth"]


def test_the_drawdown_is_the_approved_f6_5_with_its_peak_trough_and_recovery():
    drawdown = measures.episode_measures(episode(all_equity()), sc.CASH_INDEX)["drawdown"]
    # Peak at the close of 99, five -2% sessions to the trough at 104; back above the
    # peak after n sessions of +1% with 1.01**n >= 1 / 0.98**5, n = 11: session 115.
    assert drawdown["measure"] == measures.MAXIMUM_DRAWDOWN and drawdown["history_complete"]
    assert drawdown["value"] == pytest.approx(1 - 0.98**5, rel=1e-12)
    assert (drawdown["peak_session"], drawdown["trough_session"]) == (99, 104)
    assert drawdown["recovery_session"] == 115 and drawdown["recovered"]
    assert drawdown["duration_sessions"] == 16 and drawdown["missing_sessions"] == 0
    assert math.ceil(math.log(1 / 0.98**5) / math.log(1.01)) == 11


def test_an_unvalued_session_makes_the_history_incomplete_and_is_labelled():
    drawdown = measures.episode_measures(
        episode(all_equity(), data=scenario(bond_gap_at=120)), sc.CASH_INDEX
    )["drawdown"]
    # The environment values only sessions where every asset closes (D6, universe-
    # complete), so 120 has no wealth: the result is only the largest observed decline.
    assert drawdown["measure"] == measures.OBSERVED_DECLINE
    assert not drawdown["history_complete"] and drawdown["missing_sessions"] == 1


def test_turnover_traded_notional_and_costs_of_a_single_build():
    result = measures.episode_measures(episode(all_equity()), sc.CASH_INDEX)
    assert result["turnover_one_way"] == pytest.approx(1.0)  # D3: a build from cash is 1
    assert result["build_turnover_one_way"] == pytest.approx(1.0)
    assert result["rebalance_turnover_per_year"] == 0.0
    assert result["traded_notional"] == pytest.approx(1.0)  # the cash leg is not notional
    assert result["costs_paid"] == pytest.approx(0.0005)  # 5 bp of the initial wealth
    assert result["costs_bps_summed"] == pytest.approx(5.0)
    assert result["trades"] == 1 and result["holds"] == result["decisions"] - 1
    assert result["largest_execution_turnover"] == pytest.approx(1.0)


def test_rebalance_turnover_is_annualized_per_252_session_synthetic_year():
    record = episode(ConstantMix("half_half", [0.5, 0.5, 0.0, 0.0]))
    result = measures.episode_measures(record, sc.CASH_INDEX)
    # Five +1% sessions of equity drift between executions: 0.5 * 1.01**5 / (0.5 * 1.01**5
    # + 0.5) = 0.512433 of equity; returning to 0.5 is one-way turnover 0.012433.
    drifted = 1.01**5 / (1.01**5 + 1)
    first_rebalance = [row for row in record.ledger if row.target is not None][1]
    assert first_rebalance.turnover_one_way == pytest.approx(drifted - 0.5, rel=1e-12)
    after_build = sum(row.turnover_one_way for row in record.ledger) - 1.0
    assert result["rebalance_turnover_per_year"] == pytest.approx(after_build * 252 / 75)
    assert result["traded_notional"] == pytest.approx(1.0 + 2 * after_build)


def test_interventions_and_the_build_under_a_stricter_set():
    result = measures.episode_measures(
        episode(all_equity(), max_weight=0.5, max_turnover=0.25), sc.CASH_INDEX
    )
    build = result["build"]
    # Capped to (0.5, 0, 0, 0.5 cash), then a quarter-turnover build from cash:
    # tau = 0.5, alpha = 0.5, so 25% equity and 75% uninvested cash; held after that.
    assert build["executed_weights"] == pytest.approx([0.25, 0.0, 0.0, 0.75])
    assert build["uninvested_cash"] == pytest.approx(0.75)
    assert build["binding"] == [CAP, TURNOVER_LIMIT]
    assert build["incomplete_construction"] and build["fully_built_session"] is None
    assert build["delay_sessions"] == 0 and build["session"] == 81
    assert result["interventions"] == {
        "targets": 1,
        "cap_hits": 1,
        "turnover_limit_hits": 1,
        "distance_one_way_total": pytest.approx(0.75),
        "distance_one_way_max": pytest.approx(0.75),
    }
    assert result["invalid_targets"] == 0 and result["fallback_events"] == 0


def test_a_rebalanced_build_under_the_limit_completes_and_says_when():
    result = measures.episode_measures(
        episode(ConstantMix("half_half", [0.5, 0.5, 0.0, 0.0]), max_turnover=0.25),
        sc.CASH_INDEX,
    )
    build = result["build"]
    assert build["incomplete_construction"]
    # Executions at 81, 86, 91: a quarter of the way from cash, then the limit again
    # (equity drift lifts the remaining distance); at 96 the remaining 0.234 is within
    # 0.25, so the portfolio first reaches 50/50 there, 15 sessions after the build.
    assert build["fully_built_session"] == 96
    assert build["sessions_until_fully_built"] == 15


def test_every_reported_measure_has_a_unit():
    assert set(measures.UNITS) >= {
        "terminal_wealth",
        "total_return",
        "drawdown",
        "turnover_one_way",
        "rebalance_turnover_per_year",
        "traded_notional",
        "costs_paid",
        "cost_sensitivity",
    }
    assert "252" in measures.UNITS["rebalance_turnover_per_year"]
