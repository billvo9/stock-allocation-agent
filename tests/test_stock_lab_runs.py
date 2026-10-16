"""
Directional runs (stock_agent.stock_lab.runs): both Stable-session rules,
gaps, censoring, and agreement between the per-session view and the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab.runs import BREAKS, PAUSES, RunRule, current_runs, run_table

U, S, D = "increase", "stable", "decrease"


def series(*values):
    return pd.Series(list(values), dtype=object)


def dates(n):
    return pd.bdate_range("2024-01-01", periods=n, tz="UTC")


def lengths(table):
    return list(zip(table["direction"], table["length"], table["censored"], strict=True))


def test_breaks_counts_maximal_blocks_and_censors_the_open_run():
    cats = series(U, U, S, U, D, D, D)
    table = run_table(dates(7), cats)
    assert lengths(table) == [(U, 2, False), (S, 1, False), (U, 1, False), (D, 3, True)]
    now = current_runs(cats)
    assert now["run_length"].tolist() == [1, 2, 1, 1, 1, 2, 3]
    assert now["run_direction"].tolist() == [U, U, S, U, D, D, D]


def test_pauses_lets_one_stable_session_pause_a_streak():
    rule = RunRule(PAUSES, max_pause=1)
    cats = series(U, U, S, U, D, S, S, D)
    table = run_table(dates(8), cats, rule)
    assert lengths(table) == [(U, 3, False), (D, 1, False), (S, 2, False), (D, 1, True)]
    assert table["paused"].tolist() == [1, 0, 0, 0]
    now = current_runs(cats, rule)
    assert now["run_direction"].tolist() == [U, U, U, U, D, D, S, D]
    assert now["run_length"].tolist() == [1, 2, 2, 3, 1, 1, 2, 1]
    assert now["paused"].tolist() == [0, 0, 1, 1, 0, 1, 0, 0]


def test_pauses_with_a_longer_allowance():
    rule = RunRule(PAUSES, max_pause=2)
    table = run_table(dates(5), series(D, S, S, D, U), rule)
    assert lengths(table) == [(D, 2, False), (U, 1, True)]
    assert table["paused"].tolist() == [2, 0]


def test_a_stable_stretch_before_the_opposite_direction_is_its_own_run():
    table = run_table(dates(3), series(U, S, D), RunRule(PAUSES))
    assert lengths(table) == [(U, 1, False), (S, 1, False), (D, 1, True)]


def test_a_gap_ends_every_run_and_censors_it():
    cats = series(D, D, None, D, np.nan, U)
    table = run_table(dates(6), cats)
    assert lengths(table) == [(D, 2, True), (D, 1, True), (U, 1, True)]
    now = current_runs(cats)
    assert now["run_length"].tolist() == [1, 2, 0, 1, 0, 1]
    assert pd.isna(now["run_direction"].iloc[2]) and pd.isna(now["run_direction"].iloc[4])


@pytest.mark.parametrize("rule", [RunRule(BREAKS), RunRule(PAUSES, 1), RunRule(PAUSES, 3)])
def test_every_run_row_matches_the_per_session_view_at_its_end(rule):
    rng = np.random.RandomState(4)
    values = rng.choice([U, S, D, None], size=400, p=[0.4, 0.25, 0.33, 0.02])
    cats = pd.Series(values, dtype=object)
    sessions = dates(400)
    table = run_table(sessions, cats, rule)
    now = current_runs(cats, rule)
    position = {day: i for i, day in enumerate(sessions)}
    for row in table.itertuples():
        if rule.policy == PAUSES and row.direction == S and row.length <= rule.max_pause:
            # A short Stable stretch read as a pause at the time, until the opposite
            # direction (or a gap) showed it ended the run: the per-session view is
            # past-only, so it cannot know that yet.
            continue
        at_end = now.iloc[position[row.end]]
        assert at_end["run_direction"] == row.direction
        assert at_end["run_length"] == row.length
    # Every directional session belongs to exactly one run's span under "breaks".
    if rule.policy == BREAKS:
        known = int(cats.notna().sum())
        assert int(table["length"].sum()) == known


def test_the_rule_is_validated():
    with pytest.raises(ValueError):
        RunRule("ignore")
    with pytest.raises(ValueError):
        RunRule(PAUSES, 0)
    with pytest.raises(TypeError):
        RunRule(PAUSES, True)
    assert "1 Stable" in RunRule(PAUSES).words()


def test_lengths_must_match():
    with pytest.raises(ValueError):
        run_table(dates(2), series(U))
