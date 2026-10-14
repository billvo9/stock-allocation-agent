"""
Actions: unconstrained logits mapped deterministically to long-only weights.

A learned policy outputs a logit z_i for each of the n assets, cash
included. The environment holds

    w = softmax(z),   w_i = exp(z_i - max z) / sum_j exp(z_j - max z),

so every action is long-only and sums to 1 without any projection or
repair step.

Why the PPO ratio stays valid. PPO samples z ~ pi_theta(. | s), a diagonal
Gaussian, and scores

    ratio = pi_theta(z | s) / pi_theta_old(z | s)

on the SAME z it sampled. The map f(z) = softmax(z) is deterministic and
belongs to the environment: the reward depends on z only through f(z). The
score-function gradient E[grad log pi(z|s) * A] is unbiased for any
deterministic f, because the expectation is over z, whose density is the one
being differentiated. Scoring a transformed action g(z) under pi_theta
instead would evaluate the density of a point that was never sampled; for a
projection onto the simplex's boundary the pushed-forward distribution even
has point masses where pi_theta has none, so the ratio is meaningless.
Hence: never project a sampled action and then score the projected action.

Two consequences:
- softmax is shift-invariant, so one direction in logit space (adding a
  constant to every logit) never changes the reward. That direction gets no
  gradient signal, which is harmless for the ratio. The environment's action
  space is unbounded, so no clipping breaks the shift invariance.
- After approval, Track A's `feasible_target` (caps, turnover limits) will
  compose with the map: w = feasible(softmax(z)). The ratio stays valid, but
  where a cap binds, moving z within the capped region changes nothing the
  environment sees: those directions carry zero signal and only variance.
  The policy then learns nothing about how far past the cap it "wants" to
  be, and evaluation should report how often caps bind.
"""

from __future__ import annotations

import math

import numpy as np


def logits_to_weights(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError(f"Logits must be a non-empty vector, got shape {z.shape}.")
    if not np.all(np.isfinite(z)):
        raise ValueError("Logits must be finite.")
    shifted = np.exp(z - z.max())
    return shifted / shifted.sum()


def gaussian_log_prob(sample: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> float:
    """log N(sample; mean, diag(exp(log_std))^2), the density PPO scores on logits."""

    x = np.asarray(sample, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(log_std, dtype=np.float64)
    standardized = (x - m) / np.exp(s)
    return float(np.sum(-0.5 * standardized**2 - s - 0.5 * math.log(2.0 * math.pi)))
