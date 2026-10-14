"""
A health gate and a deterministic fallback for learned policies.

Before a learned policy acts, `HealthGate.check` inspects the observation
and the model's recorded health. Any reason below switches the decision to
the registered fallback, and the reason is recorded with the decision:

    stale_inputs         the data feed lags more than `max_obs_age` sessions
    non_finite_features  a feature could not be computed (missing history)
    out_of_distribution  a market feature lies beyond `z_limit` training
                         standard deviations (state drift)
    model_unhealthy      the model's training-health record failed
    invalid_action       the policy returned a non-simplex action
    policy_input_error   the policy refused the observation

The fallback is the registered fixed balanced allocation, except on stale
inputs, where it holds the current positions: the product rule is never to
trade on stale prices (vision_and_prd.md, section 15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_agent.rl.observation import Observation
from stock_agent.rl.policies import FIXED_BALANCED_WEIGHTS, PolicyInputError

FALLBACK_REASONS = (
    "stale_inputs",
    "non_finite_features",
    "out_of_distribution",
    "model_unhealthy",
    "invalid_action",
    "policy_input_error",
)


@dataclass(frozen=True)
class HealthGate:
    max_obs_age: int = 2
    z_limit: float = 8.0

    def check(self, observation: Observation, model_healthy: bool = True) -> tuple[str, ...]:
        reasons: list[str] = []
        if observation.max_obs_age > self.max_obs_age:
            reasons.append("stale_inputs")
        if not observation.finite:
            reasons.append("non_finite_features")
        elif observation.max_abs_z > self.z_limit:
            reasons.append("out_of_distribution")
        if not model_healthy:
            reasons.append("model_unhealthy")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {"max_obs_age": self.max_obs_age, "z_limit": self.z_limit}


class FallbackPolicy:
    name = "fallback_reference"

    def __init__(self, reference=FIXED_BALANCED_WEIGHTS, hold_on_stale: bool = True) -> None:
        self.reference = np.asarray(reference, dtype=np.float64).copy()
        self.hold_on_stale = hold_on_stale

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation, reasons: tuple[str, ...] = ()) -> np.ndarray | None:
        if self.hold_on_stale and "stale_inputs" in reasons:
            return None
        return self.reference.copy()


def _valid_action(action, n_assets: int) -> bool:
    if action is None:
        return True
    array = np.asarray(action, dtype=np.float64)
    return (
        array.shape == (n_assets,)
        and bool(np.all(np.isfinite(array)))
        and bool(np.all(array >= 0))
        and abs(float(array.sum()) - 1.0) <= 1e-9
    )


class GuardedPolicy:
    """Run `primary` only when the gate passes; otherwise the fallback decides."""

    def __init__(
        self,
        primary,
        *,
        gate: HealthGate | None = None,
        fallback: FallbackPolicy | None = None,
        model_healthy: bool = True,
    ) -> None:
        self.primary = primary
        self.gate = gate or HealthGate()
        self.fallback = fallback or FallbackPolicy()
        self.model_healthy = model_healthy
        self.name = f"guarded:{primary.name}"
        self.last_reasons: tuple[str, ...] = ()

    def reset(self) -> None:
        self.primary.reset()
        self.fallback.reset()
        self.last_reasons = ()

    def decide(self, observation: Observation) -> np.ndarray | None:
        reasons = self.gate.check(observation, self.model_healthy)
        if not reasons:
            try:
                action = self.primary.decide(observation)
            except PolicyInputError:
                reasons = ("policy_input_error",)
            else:
                if _valid_action(action, observation.weights.shape[0]):
                    self.last_reasons = ()
                    return action
                reasons = ("invalid_action",)
        self.last_reasons = reasons
        return self.fallback.decide(observation, reasons)
