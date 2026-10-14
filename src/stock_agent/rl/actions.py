"""
Actions: logits mapped deterministically to long-only weights.

A learned policy outputs one logit per fund; the cash account is the
reference asset with its logit pinned at 0. With z the n logits (cash's
inserted as 0), the environment holds

    w = softmax(z),   w_i = exp(z_i - max z) / sum_j exp(z_j - max z),

so every action is long-only and sums to 1 without any projection or
repair step. `logits_to_weights` is the map on all n logits;
`fund_logits_to_weights` pins cash and applies the learner's action bound.

Why the PPO ratio stays valid. PPO samples a ~ pi_theta(. | s), a diagonal
Gaussian, and scores

    ratio = pi_theta(a | s) / pi_theta_old(a | s)

on the SAME a it sampled (stable-baselines3 stores the raw sample and
scores it). The map f(a) = softmax(pad_cash(clip(a))) is deterministic and
belongs to the environment: the reward depends on a only through f(a). The
score-function gradient E[grad log pi(a|s) * A] is unbiased for any
deterministic f, because the expectation is over a, whose density is the one
being differentiated. Scoring a transformed action g(a) under pi_theta
instead would evaluate the density of a point that was never sampled; for a
projection onto the simplex's boundary the pushed-forward distribution even
has point masses where pi_theta has none, so the ratio is meaningless.
Hence: never project a sampled action and then score the projected action.

Consequences:
- Why cash is pinned. softmax is shift-invariant: adding a constant to every
  logit never changes the weights, so with n free logits one direction gets
  no gradient signal and can drift. stable-baselines3 requires a finite
  action box and clips to it, and clipping is not shift-invariant, so a
  drifting common mode would eventually distort the weights. Pinning cash at
  0 removes that direction: n - 1 logits identify the weights exactly.
- The bound. Fund logits are clipped to [-ACTION_BOUND, ACTION_BOUND]. The
  clip binds only at saturation (a fund-to-cash weight ratio beyond
  e^10, about 22,000), where the weights barely move; there the action has
  no gradient signal, exactly like a binding cap.
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

ACTION_BOUND = 10.0


def logits_to_weights(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError(f"Logits must be a non-empty vector, got shape {z.shape}.")
    if not np.all(np.isfinite(z)):
        raise ValueError("Logits must be finite.")
    shifted = np.exp(z - z.max())
    return shifted / shifted.sum()


def fund_logits_to_weights(
    fund_logits: np.ndarray, cash_index: int, bound: float = ACTION_BOUND
) -> np.ndarray:
    """Clip the n - 1 fund logits, insert cash's 0 at `cash_index`, softmax."""

    a = np.asarray(fund_logits, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("Fund logits must be a finite vector.")
    return logits_to_weights(np.insert(np.clip(a, -bound, bound), cash_index, 0.0))


def gaussian_log_prob(sample: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> float:
    """log N(sample; mean, diag(exp(log_std))^2), the density PPO scores on logits."""

    x = np.asarray(sample, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(log_std, dtype=np.float64)
    standardized = (x - m) / np.exp(s)
    return float(np.sum(-0.5 * standardized**2 - s - 0.5 * math.log(2.0 * math.pi)))
