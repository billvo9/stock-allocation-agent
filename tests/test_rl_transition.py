"""
The per-step transition the RL environment injects.

The environment injects the approved `portfolio.transition.portfolio_step`
(T5a D1-D8, approved 2026-10-15). These tests pin it, through the RL seam,
to the specification's worked examples (t5_portfolio_formulas.md sections 2
and 8), and check that `check_step_result` rejects a transition that breaks
the shared invariants. (They pinned the engineering stand-in to the same
numbers before the approval; the stand-in is retired.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stock_agent.rl import transition as tr

STEP = tr.portfolio_step
BP = 1e-4


def _vec(*values):
    return np.array(values, dtype=np.float64)


def test_the_spec_rebalance_example_t2_4_and_t8_1():
    result = STEP(
        _vec(0.55, 0.25, 0.20),
        _vec(0.40, 0.40, 0.20),
        _vec(0.02, -0.01, 0.0001),
        _vec(10 * BP, 10 * BP, 0.0),
        2,
    )
    assert result.cost == pytest.approx(0.0003, abs=1e-15)
    assert result.gross_return == pytest.approx(0.00402, abs=1e-15)
    assert result.net_return == pytest.approx(0.0037187940, abs=1e-12)
    assert 100_000 * (1 + result.net_return) == pytest.approx(100_371.8794, abs=1e-6)
    assert result.turnover_one_way == pytest.approx(0.15, abs=1e-15)
    assert result.traded_notional == pytest.approx(0.30, abs=1e-15)
    np.testing.assert_allclose(result.w_post_trade, [0.40, 0.40, 0.20], atol=1e-15)
    np.testing.assert_allclose(result.w_next, [0.40637, 0.39441, 0.19922], atol=5e-6)
    assert result.w_next.sum() == pytest.approx(1.0, abs=1e-12)


def test_costing_from_the_previous_target_instead_of_drifted_weights_would_differ():
    # T2.4's negative: the trade is from the drifted weights, so a target equal to
    # the last target but different from the drifted weights still costs.
    drifted = _vec(0.55, 0.25, 0.20)
    result = STEP(drifted, _vec(0.40, 0.40, 0.20), np.zeros(3), _vec(10 * BP, 10 * BP, 0), 2)
    assert result.cost > 0


def test_stocks_to_cash_costs_only_the_risky_leg_t2_1():
    result = STEP(_vec(1.0, 0.0), _vec(0.0, 1.0), np.zeros(2), _vec(10 * BP, 0.0), 1)
    assert 100_000 * (1 + result.net_return) == pytest.approx(99_900.0, abs=1e-9)


def test_building_from_cash_costs_once_with_unit_turnover_t2_2():
    result = STEP(
        _vec(0.0, 0.0, 1.0), _vec(0.5, 0.5, 0.0), np.zeros(3), _vec(10 * BP, 10 * BP, 0), 2
    )
    assert 100_000 * (1 + result.net_return) == pytest.approx(99_900.0, abs=1e-9)
    assert result.turnover_one_way == pytest.approx(1.0, abs=1e-15)


def test_costs_are_multiplicative_not_additive_t2_3():
    result = STEP(_vec(0.0, 1.0), _vec(1.0, 0.0), _vec(0.10, 0.0), _vec(10 * BP, 0.0), 1)
    assert 1 + result.net_return == pytest.approx(1.0989, abs=1e-12)
    assert abs((1 + result.net_return) - 1.0990) > 1e-9


def test_cross_class_costs_t2_5():
    to_bill_fund = STEP(
        _vec(0.2, 0.0, 0.0, 0.8),
        _vec(0.0, 0.2, 0.0, 0.8),
        np.zeros(4),
        _vec(5 * BP, 2 * BP, 5 * BP, 0),
        3,
    )
    to_treasury = STEP(
        _vec(0.2, 0.0, 0.0, 0.8),
        _vec(0.0, 0.0, 0.2, 0.8),
        np.zeros(4),
        _vec(5 * BP, 2 * BP, 5 * BP, 0),
        3,
    )
    assert to_bill_fund.cost == pytest.approx(1.4 * BP, abs=1e-15)
    assert to_treasury.cost == pytest.approx(2.0 * BP, abs=1e-15)


def test_no_target_or_an_unchanged_target_is_pure_drift():
    w = _vec(0.5, 0.3, 0.2)
    r = _vec(0.01, -0.02, 0.0)
    hold = STEP(w, None, r, _vec(10 * BP, 10 * BP, 0), 2)
    same = STEP(w, w.copy(), r, _vec(10 * BP, 10 * BP, 0), 2)
    for result in (hold, same):
        assert result.cost == 0.0
        assert result.net_return == pytest.approx(result.gross_return, abs=1e-15)
        np.testing.assert_array_equal(result.w_post_trade, w)


def test_an_all_cash_portfolio_earns_the_cash_return():
    result = STEP(_vec(0.0, 1.0), None, _vec(0.05, 0.0002), _vec(5 * BP, 0), 1)
    assert result.net_return == pytest.approx(0.0002, abs=1e-15)


def test_a_target_that_does_not_sum_to_one_is_rejected_not_repaired():
    with pytest.raises(ValueError, match="sum to 1"):
        STEP(_vec(0.5, 0.5), _vec(0.6, 0.6), np.zeros(2), _vec(5 * BP, 0), 1)


def test_a_negative_target_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        STEP(_vec(0.5, 0.5), _vec(1.2, -0.2), np.zeros(2), _vec(5 * BP, 0), 1)


def test_a_non_finite_return_on_a_held_asset_raises():
    with pytest.raises(ValueError, match="held asset"):
        STEP(_vec(0.5, 0.5), None, _vec(np.nan, 0.0), _vec(5 * BP, 0), 1)


def test_a_non_finite_return_on_an_unheld_asset_is_ignored():
    result = STEP(_vec(0.0, 1.0), None, _vec(np.nan, 0.001), _vec(5 * BP, 0), 1)
    assert result.net_return == pytest.approx(0.001, abs=1e-15)
    assert result.w_next[0] == 0.0


def test_a_cost_of_one_or_more_raises():
    with pytest.raises(ValueError, match="cost"):
        STEP(_vec(1.0, 0.0), _vec(0.0, 1.0), np.zeros(2), _vec(1.5, 0.0), 1)


def test_seeded_property_invariants_t8_2():
    rng = np.random.RandomState(20261014)
    rates = _vec(5 * BP, 5 * BP, 2 * BP, 0.0)
    for _ in range(300):
        w_pre = rng.dirichlet(np.ones(4))
        target = None if rng.rand() < 0.3 else rng.dirichlet(np.ones(4))
        r = np.expm1(rng.normal(0.0, 0.03, size=4))
        result = STEP(w_pre, target, r, rates, 3)
        tr.check_step_result(result, n_assets=4)
        assert 0.0 <= result.cost < 1.0
        assert result.net_return <= result.gross_return + 1e-15
        assert abs(result.w_next.sum() - 1.0) <= 1e-12


def test_chained_steps_telescope_to_terminal_wealth_t8_3():
    rng = np.random.RandomState(7)
    rates = _vec(5 * BP, 2 * BP, 0.0)
    w = _vec(0.0, 0.0, 1.0)
    wealth = 1.0
    log_growth = 0.0
    for _ in range(3):
        result = STEP(w, rng.dirichlet(np.ones(3)), np.expm1(rng.normal(0, 0.02, 3)), rates, 2)
        wealth *= 1 + result.net_return
        log_growth += math.log1p(result.net_return)
        w = result.w_next
    assert math.exp(log_growth) == pytest.approx(wealth, rel=1e-14)


def test_check_step_result_rejects_a_transition_that_breaks_the_simplex():
    good = STEP(_vec(0.5, 0.5), None, np.zeros(2), _vec(5 * BP, 0), 1)
    broken = tr.StepResult(
        w_post_trade=good.w_post_trade,
        w_next=_vec(0.7, 0.7),
        gross_return=good.gross_return,
        cost=good.cost,
        net_return=good.net_return,
        turnover_one_way=good.turnover_one_way,
        traded_notional=good.traded_notional,
        reason=good.reason,
    )
    with pytest.raises(tr.TransitionContractError, match="w_next"):
        tr.check_step_result(broken, n_assets=2)


def test_check_step_result_rejects_a_net_return_above_gross():
    good = STEP(_vec(0.5, 0.5), None, _vec(0.01, 0.0), _vec(5 * BP, 0), 1)
    broken = tr.StepResult(**{**good.__dict__, "net_return": good.gross_return + 0.01})
    with pytest.raises(tr.TransitionContractError, match="net_return"):
        tr.check_step_result(broken, n_assets=2)


def test_the_injected_transition_is_the_approved_engine_function():
    from stock_agent.portfolio import transition as engine_transition

    assert tr.portfolio_step is engine_transition.portfolio_step
    assert tr.StepResult is engine_transition.StepResult
    assert tr.APPROVED_TRANSITION.approved and tr.APPROVED_TRANSITION.status == "approved"
    assert tr.ENGINEERING_ONLY == "engineering_only_unapproved_transition"
    assert not tr.TransitionInfo("x", "0", tr.ENGINEERING_ONLY).approved


def test_the_engineering_stand_in_is_retired():
    import importlib.util

    assert importlib.util.find_spec("stock_agent.rl.reference_transition") is None
