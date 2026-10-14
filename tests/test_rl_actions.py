"""The deterministic logits-to-simplex action map and where the likelihood lives."""

from __future__ import annotations

import numpy as np
import pytest

from stock_agent.rl import actions


def test_softmax_weights_are_long_only_and_sum_to_one():
    rng = np.random.RandomState(0)
    for _ in range(200):
        weights = actions.logits_to_weights(rng.normal(0, 3, size=4))
        assert weights.shape == (4,)
        assert np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-15)


def test_the_map_is_deterministic_and_shift_invariant():
    logits = np.array([0.3, -1.2, 2.0, 0.0])
    first = actions.logits_to_weights(logits)
    np.testing.assert_array_equal(first, actions.logits_to_weights(logits.copy()))
    np.testing.assert_allclose(actions.logits_to_weights(logits + 7.5), first, atol=1e-15)


def test_extreme_logits_do_not_overflow():
    weights = actions.logits_to_weights(np.array([1000.0, -1000.0, 0.0, 0.0]))
    np.testing.assert_array_equal(weights, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("bad", [[np.nan, 0.0], [np.inf, 0.0], [[0.0, 1.0]]])
def test_non_finite_or_misshaped_logits_are_rejected(bad):
    with pytest.raises(ValueError):
        actions.logits_to_weights(np.array(bad))


def test_the_likelihood_is_a_density_over_logits_not_over_weights():
    """
    Two logit samples one constant apart give identical weights but different
    Gaussian densities. A density over weights cannot recover either number,
    so PPO's ratio must be computed on the logits that were sampled.
    """

    mean = np.zeros(4)
    log_std = np.zeros(4)
    sample = np.array([0.4, -0.2, 0.1, 0.0])
    shifted = sample + 1.0
    np.testing.assert_allclose(
        actions.logits_to_weights(sample), actions.logits_to_weights(shifted), atol=1e-15
    )
    assert actions.gaussian_log_prob(sample, mean, log_std) != pytest.approx(
        actions.gaussian_log_prob(shifted, mean, log_std)
    )


def test_gaussian_log_prob_matches_the_closed_form():
    sample = np.array([0.5, -1.0])
    mean = np.array([0.0, 0.5])
    log_std = np.array([0.0, np.log(2.0)])
    expected = sum(
        -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
        for x, m, s in zip(sample, mean, [1.0, 2.0], strict=True)
    )
    assert actions.gaussian_log_prob(sample, mean, log_std) == pytest.approx(expected, rel=1e-14)
