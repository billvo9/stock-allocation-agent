"""
The Stock Lab's significance tests and their simulated size
(stock_agent.stock_lab.size_simulation): the three tests reported for every
comparison, and how often each one is fooled by a challenger with no skill.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics.inference import (
    bootstrap_mean,
    draw_circular_blocks,
    fold_block_t,
    newey_west_mean,
)
from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import scores
from stock_agent.stock_lab import size_simulation as size


def design(**changes) -> size.SizeDesign:
    settings = {
        "names": 3,
        "years": 3,
        "half_lives": (21.0, 252.0),
        "reps": 24,
        "seed": 5,
        "nw_lag": 10,
        "block": 20,
        "bootstrap_reps": 99,
        "bootstrap_seed": 7,
    }
    return size.SizeDesign(**{**settings, **changes})


# --- the three tests ---


def test_the_three_tests_are_the_shared_estimators_made_one_sided():
    rng = np.random.RandomState(1)
    values = rng.normal(-0.0004, 0.002, 600)
    blocks = np.arange(600) // 200
    draws = draw_circular_blocks(600, block_length=20, reps=99, seed=7)
    tests = size.one_sided_tests(values, blocks, draws, lag=10)
    assert tuple(tests) == size.METHODS
    shared = {
        "newey_west": newey_west_mean(values, lag=10),
        "fold_block_t": fold_block_t(values, blocks),
        "circular_block_bootstrap": bootstrap_mean(values, draws),
    }
    for method, expected in shared.items():
        assert tests[method].p_value == expected.p_value and tests[method].se == expected.se
        # The challenger is better when its loss differential is negative.
        assert size.one_sided_p(tests[method]) == pytest.approx(expected.p_value / 2)
    assert tests["fold_block_t"].params["df"] == 2


def test_a_positive_differential_is_never_evidence_for_the_challenger():
    values = np.random.RandomState(2).normal(0.001, 0.002, 600)
    draws = draw_circular_blocks(600, block_length=20, reps=99, seed=7)
    tests = size.one_sided_tests(values, np.arange(600) // 200, draws, lag=10)
    for result in tests.values():
        assert size.one_sided_p(result) > 0.5


def test_an_unavailable_test_has_no_p_value():
    draws = draw_circular_blocks(5, block_length=20, reps=9, seed=7)
    tests = size.one_sided_tests(np.full(5, 0.001), np.zeros(5), draws, lag=10)
    for result in tests.values():
        assert result.status == "unavailable" and math.isnan(size.one_sided_p(result))


# --- the no-skill differential ---


def test_the_placebo_differentials_are_the_estimators_own_scores():
    """In RPS on the five daily targets, averaged per date: B1 read from the next
    stock's history minus the stock's own B1 ("other_vs_own"), and minus B1 read
    from the previous stock's history ("other_vs_other"); computed here with
    est.b1 and scores.rps on a panel, row by row."""

    rng = np.random.RandomState(3)
    names, sessions, first, half_life = 4, 160, 40, 21.0
    codes = rng.choice(3, size=(1, sessions, names), p=[0.4, 0.2, 0.4])
    panel = pd.DataFrame(
        {
            "symbol": np.tile([f"S{i}" for i in range(names)], sessions),
            "session": np.repeat(pd.bdate_range("2020-01-01", periods=sessions, tz="UTC"), names),
            "category_code": codes[0].reshape(-1).astype(float),
        }
    )
    params = est.FrequencyParams(alpha=50.0, half_life=half_life, kappa=40.0)
    b1, _ = est.b1(panel, params)
    b1 = b1.reshape(sessions, names, 3)
    expected = {"other_vs_own": [], "other_vs_other": []}
    for t in range(first, sessions - 5):
        values = {key: [] for key in expected}
        for i in range(names):
            for h in range(1, 6):
                y = np.array([codes[0, t + h, i]])
                following = scores.rps(b1[t, (i + 1) % names][None], y)[0]
                own = scores.rps(b1[t, i][None], y)[0]
                previous = scores.rps(b1[t, (i - 1) % names][None], y)[0]
                values["other_vs_own"].append(following - own)
                values["other_vs_other"].append(following - previous)
        for key, series in expected.items():
            series.append(np.mean(values[key]))
    got = size.placebo_differentials(codes, (half_life,), alpha_prior=50.0, kappa=40.0, first=first)
    assert size.PLACEBOS == ("other_vs_own", "other_vs_other")
    assert got.shape == (2, 1, 1, sessions - 5 - first)
    for index, key in enumerate(size.PLACEBOS):
        np.testing.assert_allclose(got[index, 0, 0], expected[key], rtol=1e-10, atol=1e-14)


def test_the_symmetric_placebo_changes_sign_when_the_stocks_are_listed_backwards():
    """Reversing the stocks' order swaps every stock's next and previous
    neighbour, so "other_vs_other" is exactly antisymmetric: with stocks alike,
    its distribution is symmetric around zero. "other_vs_own" is not."""

    codes = np.random.RandomState(8).choice(3, size=(2, 400, 5), p=[0.4, 0.2, 0.4])
    kwargs = {"alpha_prior": 50.0, "kappa": 40.0, "first": 100}
    forward = size.placebo_differentials(codes, (21.0, 63.0), **kwargs)
    backward = size.placebo_differentials(codes[:, :, ::-1], (21.0, 63.0), **kwargs)
    np.testing.assert_allclose(backward[1], -forward[1], atol=1e-15)
    assert not np.allclose(backward[0], -forward[0])


def test_null_returns_are_correlated_across_names_and_categorized_like_the_lab():
    rng = np.random.RandomState(4)
    returns = size.null_returns(rng, 2, 4000, 6, correlation=0.4, daily_vol=0.015, t_df=5.0)
    pairs = [np.corrcoef(returns[0, :, 0], returns[0, :, j])[0, 1] for j in range(1, 6)]
    assert np.mean(pairs) == pytest.approx(0.4, abs=0.05)
    assert returns.std() == pytest.approx(0.015, rel=0.05)
    codes = size.categorize(np.array([-0.006, -0.005, 0.0, 0.005, 0.0051]), 0.005)
    assert codes.tolist() == [0, 1, 1, 1, 2]  # band edges are Stable, as in prices.categorize


# --- the simulation ---


def test_the_simulation_records_every_placebo_method_and_half_life_and_repeats_exactly():
    first = size.simulate_size(design())
    assert first == size.simulate_size(design())  # same seed, same answer
    assert [(r["placebo"], r["half_life"], r["method"]) for r in first["rates"]] == [
        (p, h, m) for p in size.PLACEBOS for h in (21.0, 252.0) for m in size.METHODS
    ]
    for row in first["rates"]:
        assert 0.0 <= row["rate"] <= 1.0 and row["reps"] == 24
        assert row["rate"] == row["rejections"] / 24
        assert row["exceeds_nominal"] == (row["rate"] > first["design"]["limit"])
    assert first["design"]["names"] == 3 and first["design"]["dates"] == 3 * 252
    assert set(first["design"]["placebos"]) == set(size.PLACEBOS)
    assert "never used" in first["use"]
    first["rates"].clear()  # a caller's copy: the cached result is unchanged
    assert size.simulate_size(design())["rates"]


def test_both_placebos_have_mean_zero_and_the_symmetric_one_rejects_half_at_nominal_half():
    """Every date's differential has mean zero under both placebos (neither
    history says anything about the stock's next sessions). The symmetric
    placebo is also symmetric in distribution, so a one-sided test at a nominal
    50% rejects about half the time."""

    result = size.simulate_size(design(nominal=0.5, reps=40))
    for row in result["rates"]:
        if row["placebo"] == "other_vs_other":
            assert 0.25 <= row["rate"] <= 0.75, row
    for row in result["null_mean"]:
        assert abs(row["mean"]) < 4 * row["se"], row


def test_fewer_than_three_stocks_have_no_two_neighbours_so_no_size_is_reported():
    for names in (1, 2):
        result = size.simulate_size(design(names=names))
        assert result["status"] == "unavailable" and result["reason"] == "needs_three_stocks"
        assert result["rates"] == [] and result["null_mean"] == []
    assert size.simulate_size(design())["status"] == "ok"
