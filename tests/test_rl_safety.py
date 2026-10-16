"""The health gate and the deterministic fallback policy."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from stock_agent.rl import env as rl_env
from stock_agent.rl import observation as ob
from stock_agent.rl import policies as pol
from stock_agent.rl import replay, safety
from stock_agent.rl import scenarios as sc
from stock_agent.rl import transition as tr

SPEC = ob.FeatureSpec()


def make_env(data, start=60, stop=700):
    names = SPEC.market_names(data.asset_names, data.cash_index)
    builder = ob.ObservationBuilder(
        SPEC, ob.FeatureNormalizer.identity(names), data.asset_names, data.cash_index
    )
    return rl_env.AllocationEnv(
        data,
        start=start,
        stop=stop,
        transition=tr.portfolio_step,
        transition_info=tr.APPROVED_TRANSITION,
        builder=builder,
    )


@pytest.fixture(scope="module")
def clean_obs():
    obs, _ = make_env(sc.generate_scenario("calm", seed=1)).reset()
    return obs


class Fixed:
    name = "fixed"

    def __init__(self, action):
        self.action = action

    def reset(self):
        pass

    def decide(self, observation):
        return self.action


class Raises:
    name = "raises"

    def reset(self):
        pass

    def decide(self, observation):
        raise pol.PolicyInputError("no history")


GATE = safety.HealthGate(max_obs_age=2, z_limit=8.0)


def test_a_clean_observation_passes(clean_obs):
    assert GATE.check(clean_obs) == ()


def test_stale_inputs_are_flagged(clean_obs):
    assert GATE.check(dataclasses.replace(clean_obs, max_obs_age=3)) == ("stale_inputs",)


def test_non_finite_features_are_flagged(clean_obs):
    vector = np.array(clean_obs.vector)
    vector[0] = np.nan
    assert "non_finite_features" in GATE.check(dataclasses.replace(clean_obs, vector=vector))


def test_out_of_distribution_features_are_flagged(clean_obs):
    z = np.array(clean_obs.z_market)
    z[2] = 9.0
    assert GATE.check(dataclasses.replace(clean_obs, z_market=z)) == ("out_of_distribution",)


def test_an_unhealthy_model_is_flagged(clean_obs):
    assert GATE.check(clean_obs, model_healthy=False) == ("model_unhealthy",)


def test_the_registered_fallback_is_the_fixed_balanced_reference(clean_obs):
    fallback = safety.FallbackPolicy()
    np.testing.assert_array_equal(fallback.decide(clean_obs), pol.FIXED_BALANCED_WEIGHTS)
    assert fallback.decide(clean_obs, reasons=("stale_inputs",)) is None
    assert (
        safety.FallbackPolicy(hold_on_stale=False).decide(clean_obs, reasons=("stale_inputs",))
        is not None
    )


def test_the_guard_uses_the_primary_when_healthy(clean_obs):
    guarded = safety.GuardedPolicy(Fixed(np.full(4, 0.25)), gate=GATE)
    np.testing.assert_array_equal(guarded.decide(clean_obs), np.full(4, 0.25))
    assert guarded.last_reasons == ()


@pytest.mark.parametrize(
    "action",
    [np.array([0.5, 0.5, 0.5, -0.5]), np.array([np.nan, 0.5, 0.25, 0.25]), np.full(3, 1 / 3)],
)
def test_an_invalid_primary_action_falls_back(clean_obs, action):
    guarded = safety.GuardedPolicy(Fixed(action), gate=GATE)
    np.testing.assert_array_equal(guarded.decide(clean_obs), pol.FIXED_BALANCED_WEIGHTS)
    assert guarded.last_reasons == ("invalid_action",)


def test_a_policy_input_error_falls_back(clean_obs):
    guarded = safety.GuardedPolicy(Raises(), gate=GATE)
    np.testing.assert_array_equal(guarded.decide(clean_obs), pol.FIXED_BALANCED_WEIGHTS)
    assert guarded.last_reasons == ("policy_input_error",)


def test_an_unhealthy_model_never_acts(clean_obs):
    guarded = safety.GuardedPolicy(Fixed(np.full(4, 0.25)), gate=GATE, model_healthy=False)
    np.testing.assert_array_equal(guarded.decide(clean_obs), pol.FIXED_BALANCED_WEIGHTS)
    assert guarded.last_reasons == ("model_unhealthy",)


def test_on_a_stale_feed_the_guard_holds_and_records_why():
    data = sc.generate_scenario("stale_observations", seed=2)
    guarded = safety.GuardedPolicy(Fixed(np.full(4, 0.25)), gate=GATE)
    record = replay.rollout(make_env(data, 60, 700), guarded)
    stale = [d for d in record.decisions if "stale_inputs" in d.fallback_reasons]
    healthy = [d for d in record.decisions if not d.fallback_reasons]
    assert stale and healthy
    assert all(d.target is None for d in stale)
    for decision in healthy:
        np.testing.assert_array_equal(decision.target, np.full(4, 0.25))
