"""
The transparent estimators (stock_agent.stock_lab.estimators), checked by
hand and on data whose true probabilities are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab.contracts import StableBand


def panel_of(codes_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for symbol, codes in codes_by_symbol.items():
        sessions = pd.bdate_range("2020-01-01", periods=len(codes), tz="UTC")
        for session, code in zip(sessions, codes, strict=True):
            rows.append({"symbol": symbol, "session": session, "category_code": code})
    return pd.DataFrame(rows)


def test_b0_is_counts_plus_a_pooled_prior_by_hand():
    panel = panel_of({"A": [2, 2, 0], "B": [1, 1, 1]})
    params = est.FrequencyParams(alpha=6.0)
    p = est.b0(panel, params)
    # A at its third session: counts (1, 0, 2) of 3; pooled through that date over
    # both stocks: D 1, S 3, I 2 of 6.
    row = panel.index[(panel["symbol"] == "A")][2]
    expected = (np.array([1, 0, 2]) + 6.0 * np.array([1, 3, 2]) / 6) / (3 + 6.0)
    np.testing.assert_allclose(p[row], expected)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_b1_matches_its_definition_and_its_effective_sample_size():
    rng = np.random.RandomState(0)
    codes = rng.choice([0, 1, 2], size=400).astype(float)
    codes[50] = np.nan
    panel = panel_of({"A": list(codes)})
    params = est.FrequencyParams(alpha=50.0, half_life=21.0, kappa=40.0)
    prior = est.b0(panel, params)
    p, n_eff = est.b1(panel, params, prior)
    lam = 2 ** (-1 / 21)
    d = 399
    ages = d - np.arange(d + 1)
    known = np.isfinite(codes[: d + 1])
    w = lam**ages * known
    sums = np.array([(w * (codes[: d + 1] == c)).sum() for c in range(3)])
    expected = (sums + 40.0 * prior[d]) / (w.sum() + 40.0)
    np.testing.assert_allclose(p[d], expected, rtol=1e-10)
    assert n_eff[d] == pytest.approx(w.sum() ** 2 / (w**2).sum())
    # Long-run Kish size of geometric weights: (1 + lam) / (1 - lam), about 2.885 H.
    assert n_eff[d] == pytest.approx((1 + lam) / (1 - lam), rel=0.01)


def test_a_zero_tilt_is_exactly_b1_at_every_horizon():
    rng = np.random.RandomState(1)
    base = rng.dirichlet([3, 3, 3], size=50)
    states = rng.randint(-1, est.N_STATES, size=50)
    marginals = est.b2_marginals(base, states, est.Tilt.zero())
    for h in range(5):
        np.testing.assert_allclose(marginals[:, h], base, atol=1e-12)


def test_chain_marginals_are_probabilities_and_match_brute_force_paths():
    rng = np.random.RandomState(2)
    base = rng.dirichlet([2, 2, 2], size=3)
    tilt = est.Tilt(rng.normal(0, 0.5, size=(est.N_STATES, 3)) * np.array([1, 0, 1]), ridge=1.0)
    states = np.array([0, 5, 11])
    marginals = est.b2_marginals(base, states, tilt)
    np.testing.assert_allclose(marginals.sum(axis=2), 1.0)
    step = est.tilt_step(base, tilt)
    # Brute force: enumerate every 3-session path from each start.
    import itertools

    for row in range(3):
        third = np.zeros(3)
        for path in itertools.product(range(3), repeat=3):
            state, prob = states[row], 1.0
            for code in path:
                prob *= step[row, state, code]
                state = est.NEXT[state, code]
            third[path[-1]] += prob
        np.testing.assert_allclose(marginals[row, 2], third, atol=1e-12)
    paths = est.b2_path_probabilities(base, states, tilt)
    joint = sum(paths[f"joint_{a}{b}"] for a in range(3) for b in range(3))
    np.testing.assert_allclose(joint, 1.0)
    np.testing.assert_allclose(
        sum(paths[f"joint_{a}0"] for a in range(3)), marginals[:, 1, 0], atol=1e-12
    )


def test_the_tilt_recovers_a_known_run_length_effect():
    # True model: after 2+ falling sessions the next session rises with probability 0.5
    # instead of the base 1/3 (a reversal tilt); otherwise the base holds.
    rng = np.random.RandomState(3)
    n = 60_000
    base = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    states = rng.randint(0, est.N_STATES, size=n)
    reversal_states = [est.STATE_INDEX[("decrease", k)] for k in (2, 3, 4)]
    truth = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    truth[np.isin(states, reversal_states)] = [0.25, 0.25, 0.5]
    outcomes = np.array([rng.choice(3, p=p) for p in truth]).astype(float)
    tilt = est.fit_tilt(base, states, outcomes, ridge=1.0)
    fitted = est.tilt_step(base[:1], tilt)[0]
    for s in reversal_states:
        np.testing.assert_allclose(fitted[s], [0.25, 0.25, 0.5], atol=0.02)
    np.testing.assert_allclose(fitted[est.STATE_INDEX[("increase", 1)]], [1 / 3] * 3, atol=0.02)


def test_the_tilt_shrinks_states_with_few_rows_toward_b1():
    base = np.tile([0.3, 0.4, 0.3], (3, 1))
    states = np.array([0, 0, 0])
    outcomes = np.array([2.0, 2.0, 2.0])
    loose = est.fit_tilt(base, states, outcomes, ridge=0.01)
    tight = est.fit_tilt(base, states, outcomes, ridge=100.0)
    assert abs(tight.theta[0, 2]) < abs(loose.theta[0, 2])
    assert abs(tight.theta[0, 2]) < 0.05
    assert est.fit_tilt(base, np.array([-1, -1, -1]), outcomes).theta.sum() == 0


def test_b3_matches_the_normal_probabilities_for_normal_errors():
    rng = np.random.RandomState(4)
    z = np.sort(rng.normal(size=200_000))
    errors = est.ErrorDistribution(1, z, {"A": z}, kappa=250.0)
    band = StableBand(0.005)
    p = est.b3(errors, ["A", "A"], np.array([0.01, 0.03]), band)
    # 2 Phi(0.5) - 1 = 0.3829; 2 Phi(1/6) - 1 = 0.1324
    np.testing.assert_allclose(p[:, 1], [0.3829, 0.1324], atol=0.004)
    np.testing.assert_allclose(p[:, 0], p[:, 2], atol=0.01)  # sampling noise of the ECDF
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.isnan(est.b3(errors, ["A"], np.array([np.nan]), band)).all()


def test_b3_shrinks_a_short_history_toward_the_pooled_distribution():
    pooled = np.sort(np.random.RandomState(5).normal(size=10_000))
    own = np.array([-0.01, 0.0, 0.01])  # almost every move inside the band
    errors = est.ErrorDistribution(1, pooled, {"A": own}, kappa=250.0)
    p = est.b3(errors, ["A"], np.array([0.01]), StableBand(0.005))
    weight = 3 / 253
    assert p[0, 1] == pytest.approx(weight * 1.0 + (1 - weight) * 0.383, abs=0.01)


def test_standardized_returns_use_the_volatility_known_h_sessions_earlier():
    frame = pd.DataFrame(
        {"price_return": [0.01, 0.02, -0.03, 0.04], "ewma_vol": [0.01, 0.02, 0.03, 0.04]}
    )
    z = est.standardized(frame, 2)
    np.testing.assert_allclose(z, [np.nan, np.nan, -3.0, 2.0])


def test_floor_keeps_rows_summing_to_one():
    p = est.floor(np.array([[0.0, 0.0, 1.0]]), 0.001)
    assert p.sum() == pytest.approx(1.0) and p.min() > 0


# --- review additions: no lookahead in estimator state, and hand-written chain facts ---


def _sorted_panel(codes_by_symbol):
    return panel_of(codes_by_symbol).sort_values(["session", "symbol"]).reset_index(drop=True)


def test_frequency_state_is_prefix_stable_across_stocks_and_gaps():
    rng = np.random.RandomState(0)
    codes = {s: rng.choice(3, size=300).astype(float) for s in ("A", "B")}
    codes["A"][[50, 51]] = np.nan
    panel = _sorted_panel(codes)
    params = est.FrequencyParams(alpha=50.0, half_life=21.0, kappa=40.0)
    full0 = est.b0(panel, params)
    full1, full_n = est.b1(panel, params)
    sessions = panel["session"].drop_duplicates().to_numpy()
    for cut in (10, 50, 51, 120, 298):
        mask = (panel["session"] <= sessions[cut]).to_numpy()
        part = panel[mask].reset_index(drop=True)
        np.testing.assert_allclose(full0[mask], est.b0(part, params), rtol=0, atol=1e-15)
        p1, n1 = est.b1(part, params)
        np.testing.assert_allclose(full1[mask], p1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(full_n[mask], n1, rtol=0, atol=1e-12)


def test_frozen_chain_and_volatility_forecasts_use_no_later_row():
    from stock_agent.stock_lab.features import build_features
    from stock_agent.stock_lab.synthetic import demo_bars

    frame = build_features(demo_bars("DEMO_SWING")).assign(symbol="DEMO_SWING")
    tilt = est.Tilt(np.random.RandomState(1).normal(0, 0.3, (est.N_STATES, 3)) * [1, 0, 1], 1.0)
    errors = est.ErrorDistribution(1, np.sort(np.random.RandomState(2).normal(size=5000)), {})
    base = np.tile([0.3, 0.4, 0.3], (len(frame), 1))
    full_b2 = est.b2_marginals(base, est.states_of(frame), tilt)
    full_b3 = est.b3(errors, frame["symbol"], frame["ewma_vol"].to_numpy(), StableBand())
    for cut in (300, 899, 900, 1200):
        part = frame.iloc[: cut + 1]
        np.testing.assert_array_equal(est.states_of(part), est.states_of(frame)[: cut + 1])
        np.testing.assert_allclose(
            est.b2_marginals(base[: cut + 1], est.states_of(part), tilt), full_b2[: cut + 1]
        )
        np.testing.assert_allclose(
            est.b3(errors, part["symbol"], part["ewma_vol"].to_numpy(), StableBand()),
            full_b3[: cut + 1],
        )


def test_the_chain_moves_between_states_as_written_by_hand():
    up, stable, down = 2, 1, 0
    expected = {
        (("increase", 1), up): ("increase", 2),
        (("increase", 3), up): ("increase", 4),
        (("increase", 4), up): ("increase", 4),  # the cap
        (("increase", 4), down): ("decrease", 1),
        (("increase", 2), stable): ("stable", 1),
        (("stable", 2), stable): ("stable", 3),
        (("stable", 4), stable): ("stable", 4),
        (("decrease", 1), down): ("decrease", 2),
        (("decrease", 3), up): ("increase", 1),
    }
    for (state, code), after in expected.items():
        assert est.STATES[est.NEXT[est.STATE_INDEX[state], code]] == after


def test_the_all_rising_path_probability_by_hand():
    base = np.array([[0.3, 0.3, 0.4]])
    theta = np.zeros((est.N_STATES, 3))
    for length, tilt in ((1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)):
        theta[est.STATE_INDEX[("increase", length)], 2] = tilt
    model = est.Tilt(theta, ridge=1.0)
    start = np.array([est.STATE_INDEX[("increase", 1)]])
    paths = est.b2_path_probabilities(base, start, model)

    def p_up(t):
        weights = np.array([0.3, 0.3, 0.4 * np.exp(t)])
        return weights[2] / weights.sum()

    expected = p_up(0.1) * p_up(0.2) * p_up(0.3) * p_up(0.4) * p_up(0.4)
    assert paths["all_increase"][0] == pytest.approx(expected)


def test_b3_puts_the_band_edge_in_stable():
    z = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # tau / sigma = 0.5 exactly
    errors = est.ErrorDistribution(1, z, {"A": z}, kappa=0.0)
    p = est.b3(errors, ["A"], np.array([0.01]), StableBand(0.005))
    np.testing.assert_allclose(p[0], [0.2, 0.6, 0.2])  # -0.5, 0 and 0.5 are Stable


def test_standardized_returns_on_the_panel_ignore_rows_outside_a_training_set():
    from stock_agent.stock_lab.features import build_features
    from stock_agent.stock_lab.synthetic import demo_bars

    frame = build_features(demo_bars("DEMO_CALM")).assign(symbol="DEMO_CALM")
    z = est.panel_standardized(frame, 3)
    np.testing.assert_allclose(z, est.standardized(frame, 3))
    rows = np.zeros(len(frame), dtype=bool)
    rows[100:400] = True
    rows[200] = False  # a hole in the training rows never shifts the alignment
    errors = est.errors_from_panel(frame, z, rows, 3)
    finite = z[rows][np.isfinite(z[rows])]
    np.testing.assert_allclose(errors.by_stock["DEMO_CALM"], np.sort(finite))
