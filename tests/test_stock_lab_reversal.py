"""
The reversal analyzer (stock_agent.stock_lab.reversal) and the trap rule
(stock_agent.stock_lab.traps): event definitions, censoring, Kaplan-Meier
identities, thin-evidence refusals, and failed breakouts on staged data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import StableBand
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.reversal import (
    INSUFFICIENT,
    LOW,
    OK,
    after_move,
    after_streak,
    kaplan_meier,
    move_events,
    next_session,
    proportion,
    reversal_durations,
    streak_events,
)
from stock_agent.stock_lab.runs import RunRule, run_table
from stock_agent.stock_lab.traps import (
    CONFIRMED,
    HELD,
    UNRESOLVED,
    TrapRule,
    evaluate_traps,
    trap_events,
)


def bars_from_returns(returns, start="2020-01-02"):
    level = 100.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    close = np.concatenate([[100.0], level])
    n = len(close)
    return pd.DataFrame(
        {
            "session": pd.bdate_range(start, periods=n, tz="UTC").as_unit("us"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1e6),
            "div_cash": np.zeros(n),
            "split_factor": np.ones(n),
        }
    )


@pytest.fixture(scope="module")
def trend():
    return build_features(synthetic.demo_bars("DEMO_TREND"))


def test_a_streak_is_counted_once_when_it_first_reaches_k():
    # down down down down up : a 4-session fall reaches 3 exactly once.
    frame = build_features(bars_from_returns([-0.01, -0.01, -0.01, -0.01, 0.02]))
    assert streak_events(frame, 3, "decrease").tolist() == [3]
    assert streak_events(frame, 4, "decrease").tolist() == [4]
    assert streak_events(frame, 5, "decrease").tolist() == []
    shares = next_session(frame, streak_events(frame, 4, "decrease"))
    assert shares["increase"].count == 1 and shares["increase"].n == 1


def test_a_proportion_is_refused_or_flagged_when_evidence_is_thin():
    assert proportion(np.ones(5)).status == INSUFFICIENT
    assert math.isnan(proportion(np.ones(5)).estimate)
    assert proportion(np.r_[np.ones(15), np.zeros(15)]).status == LOW
    big = proportion(np.r_[np.ones(40), np.zeros(60)])
    assert big.status == OK and big.estimate == pytest.approx(0.4)
    assert big.ci_low < 0.4 < big.ci_high
    weighted = proportion(np.r_[np.ones(40), np.zeros(60)], weights=np.linspace(0.1, 1, 100))
    assert weighted.n_eff < 100


def test_moves_are_declustered_so_outcomes_never_overlap():
    returns = pd.Series([0.0, 0.05, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.0])
    assert move_events(returns, 0.04, +1, decluster=5).tolist() == [1, 8]
    assert move_events(-returns, 0.04, -1, decluster=5).tolist() == [1, 8]
    with pytest.raises(ValueError):
        move_events(returns, 0.0, +1)


def test_outcomes_after_the_cutoff_are_pending_never_resolved(trend):
    band = StableBand()
    cutoff = trend["session"].iloc[-200]
    summary = after_move(trend, 0.02, +1, band, outcome_end=cutoff)
    full = after_move(trend, 0.02, +1, band)
    assert summary.events == full.events and summary.censored > full.censored
    resolved = summary.table.query("horizon == 5 and regime == 'all'")["n"].iloc[0]
    assert resolved == summary.events - summary.censored


def test_outcome_shares_sum_to_one_and_use_the_scaled_band(trend):
    summary = after_streak(trend, 2, "increase", StableBand())
    table = summary.table.query("regime == 'all' and status != 'insufficient'")
    for _, group in table.groupby("horizon"):
        assert group["estimate"].sum() == pytest.approx(1.0)
    assert set(summary.table["regime"]) == {"all", "calm", "middle", "volatile"}


def test_kaplan_meier_hazard_is_one_minus_the_continuation_rate():
    lengths = np.array([1, 1, 2, 3, 3, 3, 5, 2, 4, 6])
    censored = np.zeros(10, dtype=bool)
    survival = kaplan_meier(lengths, censored)
    for length in range(1, 6):
        reached = (lengths >= length).sum()
        continued = (lengths > length).sum()
        assert survival.hazard[length - 1] == pytest.approx(1 - continued / reached)
    assert survival.median == 3.0


def test_kaplan_meier_treats_open_runs_as_lasting_at_least_their_length():
    lengths = np.array([2, 2, 5, 5])
    complete = kaplan_meier(lengths, np.zeros(4, dtype=bool))
    open_ = kaplan_meier(lengths, np.array([False, False, True, True]))
    assert open_.survival[-1] > complete.survival[-1]
    assert open_.median is None or open_.median >= complete.median
    assert open_.censored == 2


def test_reversal_durations_follow_only_immediate_opposite_runs():
    returns = [-0.01, -0.01, -0.01, 0.02, 0.02, -0.01, 0.0, 0.02, -0.01, -0.01, -0.01, 0.0]
    frame = build_features(bars_from_returns(returns))
    durations = reversal_durations(frame, 3, "decrease", RunRule())
    # Two 3-falls: the first is followed by a 2-rise; the second by a Stable session.
    assert durations.runs == 1 and durations.lengths.tolist() == [1, 2]
    table = run_table(frame["session"], frame["category"])
    assert (table["direction"] == "decrease").sum() == 3


def test_staged_failed_breakouts_are_confirmed_traps(trend):
    rule = TrapRule()
    events = trap_events(trend, rule)
    setups = events[events["is_setup"]]
    assert len(setups) >= 3
    assert (setups["status"] == CONFIRMED).sum() >= 3
    evaluation = evaluate_traps(events, rule)
    assert evaluation.setup_rate.n == int((setups["status"] != UNRESOLVED).sum())
    assert 0 <= evaluation.false_alarm_rate <= 1 and 0 <= evaluation.recall <= 1


def test_a_setup_inside_its_window_is_never_confirmed():
    returns = [0.001] * 70 + [-0.012] * 12 + [0.016] * 10 + [0.0] * 3
    frame = build_features(bars_from_returns(returns))
    events = trap_events(frame, TrapRule())
    latest = events[events["is_setup"]].iloc[-1]
    assert latest["status"] == UNRESOLVED and np.isnan(latest["failed"])
    cutoff_events = trap_events(frame, TrapRule(), outcome_end=frame["session"].iloc[60])
    assert set(cutoff_events["status"]) <= {UNRESOLVED, HELD, CONFIRMED}
    assert (
        cutoff_events[cutoff_events["session"] > frame["session"].iloc[50]]["status"] == UNRESOLVED
    ).all()


def test_bear_traps_mirror_bull_traps():
    returns = [0.001] * 70 + [-0.012] * 12 + [0.016] * 10 + [-0.02] * 10
    frame = build_features(bars_from_returns(returns))
    mirrored = build_features(bars_from_returns([1 / (1 + r) - 1 for r in returns]))
    bull = trap_events(frame, TrapRule())
    bear = trap_events(mirrored, TrapRule(), kind="bear")
    assert bull["session"].tolist() == bear["session"].tolist()
    assert bull["status"].tolist() == bear["status"].tolist()


def test_the_rule_is_validated():
    with pytest.raises(ValueError):
        TrapRule(decline=1.2)
    with pytest.raises(ValueError):
        TrapRule(failure_window=0)
    with pytest.raises(ValueError):
        trap_events(build_features(bars_from_returns([0.01] * 30)), kind="sideways")


def test_an_open_run_is_at_risk_only_below_its_length():
    survival = kaplan_meier(np.array([3, 3]), np.array([False, True]))
    # Length 3: the complete run ends there; the open one is not at risk at 3.
    assert survival.at_risk.tolist() == [2, 2, 1]
    assert survival.survival[-1] == 0.0


def test_a_run_ending_on_the_cutoff_is_cut_not_finished():
    returns = [-0.01, -0.01, -0.01, 0.02, 0.02, -0.01, 0.0, 0.0]
    frame = build_features(bars_from_returns(returns))
    cutoff = frame["session"].iloc[5]  # the 2-session rise ends at row 5
    durations = reversal_durations(frame, 3, "decrease", RunRule(), outcome_end=cutoff)
    assert durations.runs == 1 and durations.censored == 1


def test_every_setup_is_one_of_the_breakouts_and_its_high_comes_before_its_low(trend):
    events = trap_events(trend, TrapRule())
    assert events["is_breakout"].all()
    evaluation = evaluate_traps(events, TrapRule())
    resolved_setups = events[events["is_setup"] & (events["status"] != UNRESOLVED)]
    assert evaluation.precision == pytest.approx(resolved_setups["failed"].mean())
    assert np.isfinite(evaluation.matched_base_rate)
    # A recovery to a new high with no fall in between is no setup.
    rising = build_features(bars_from_returns([0.004] * 120))
    assert not trap_events(rising, TrapRule())["is_setup"].any()
