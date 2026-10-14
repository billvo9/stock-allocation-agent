"""The inspectable learned baseline: a linear-softmax policy trained by CEM."""

from __future__ import annotations

import numpy as np
import pytest

from stock_agent.rl import env as rl_env
from stock_agent.rl import learned, splits
from stock_agent.rl import observation as ob
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import scenarios as sc

SPEC = ob.FeatureSpec()


@pytest.fixture(scope="module")
def plan():
    return splits.SplitPlan(
        train=splits.FoldSpec("train", ("mixed",), (0, 1), 60, 200),
        dev=splits.FoldSpec("dev", ("calm", "crash"), (10,), 260, 400),
        test=splits.FoldSpec("test", ("calm",), (20,), 460, 600),
        embargo=60,
    )


@pytest.fixture(scope="module")
def builder(plan):
    norm = ob.fit_normalizer(plan.train, SPEC)
    return ob.ObservationBuilder(SPEC, norm, sc.ASSET_NAMES, sc.CASH_INDEX)


def _envs(fold, builder):
    return rl_env.fold_envs(
        fold,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        builder=builder,
    )


def test_fold_envs_cover_every_scenario_seed_pair_over_the_fold_window(plan, builder):
    envs = _envs(plan.dev, builder)
    assert [(e.data.scenario_id, e.data.seed) for e in envs] == plan.dev.episodes()
    assert all((e.start, e.stop) == (plan.dev.start, plan.dev.stop) for e in envs)


def test_zero_parameters_give_equal_weights_over_funds_and_cash(plan, builder):
    obs, _ = _envs(plan.dev, builder)[0].reset()
    policy = learned.LinearSoftmaxPolicy(learned.LinearSoftmaxParams.zeros(3), builder)
    np.testing.assert_allclose(policy.decide(obs), np.full(4, 0.25), atol=1e-15)


def test_logits_are_bias_plus_momentum_and_volatility_slopes(plan, builder):
    obs, _ = _envs(plan.dev, builder)[0].reset()
    params = learned.LinearSoftmaxParams(
        bias=(0.1, -0.2, 0.3), beta_momentum=0.7, beta_volatility=-0.4
    )
    policy = learned.LinearSoftmaxPolicy(params, builder)
    names = builder.names
    logits = np.zeros(4)
    for i, asset in enumerate(sc.ASSET_NAMES[:3]):
        mom = obs.vector[names.index(f"{asset}:ret_20")]
        vol = obs.vector[names.index(f"{asset}:vol_20")]
        logits[i] = params.bias[i] + 0.7 * mom - 0.4 * vol
    expected = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    np.testing.assert_allclose(policy.decide(obs), expected, rtol=1e-12)


def test_parameters_round_trip(plan):
    params = learned.LinearSoftmaxParams(
        bias=(0.1, -0.2, 0.3), beta_momentum=0.7, beta_volatility=-0.4
    )
    assert learned.LinearSoftmaxParams.from_vector(params.vector(), 3) == params
    assert learned.LinearSoftmaxParams.from_dict(params.to_dict()) == params


def test_missing_features_raise_a_policy_input_error(plan, builder):
    early = builder.build(ob.snapshot_at(sc.generate_scenario("calm", 0), 10), np.eye(4)[3], 0.0)
    policy = learned.LinearSoftmaxPolicy(learned.LinearSoftmaxParams.zeros(3), builder)
    with pytest.raises(learned.PolicyInputError):
        policy.decide(early)


def test_cem_finds_the_maximum_of_a_known_objective():
    target = np.array([0.5, -1.0, 2.0])
    history = learned.cross_entropy_method(
        lambda theta: -float(np.sum((theta - target) ** 2)),
        dimension=3,
        config=learned.CEMConfig(iterations=25, population=40, seed=1),
    )
    np.testing.assert_allclose(history[-1].mean, target, atol=0.05)


def test_cem_is_deterministic_given_its_seed():
    def objective(theta):
        return -float(np.sum(theta**2))

    config = learned.CEMConfig(iterations=3, population=8, seed=4)
    first = learned.cross_entropy_method(objective, 2, config)
    second = learned.cross_entropy_method(objective, 2, config)
    assert first == second


def test_training_selects_the_checkpoint_on_dev_and_never_builds_test(plan, builder, monkeypatch):
    seen_roles = []
    original = rl_env.fold_envs

    def spy(fold, **kwargs):
        seen_roles.append(fold.role)
        return original(fold, **kwargs)

    monkeypatch.setattr(learned, "fold_envs", spy)
    result = learned.train_linear_softmax(
        plan.train,
        plan.dev,
        builder=builder,
        transition=ref.proposed_portfolio_step,
        transition_info=ref.TRANSITION_INFO,
        config=learned.CEMConfig(iterations=2, population=4, seed=0),
    )
    assert sorted(seen_roles) == ["dev", "train"]
    assert result.selected_iteration == int(np.argmax(result.dev_scores))
    assert result.params == result.history[result.selected_iteration].mean_params(3)
    assert result.variants_evaluated == 2 * 4
    assert result.checkpoints_compared == 2


def test_training_refuses_folds_in_the_wrong_role(plan, builder):
    with pytest.raises(splits.LeakageError):
        learned.train_linear_softmax(
            plan.test,
            plan.dev,
            builder=builder,
            transition=ref.proposed_portfolio_step,
            transition_info=ref.TRANSITION_INFO,
        )
    with pytest.raises(splits.LeakageError):
        learned.train_linear_softmax(
            plan.train,
            plan.test,
            builder=builder,
            transition=ref.proposed_portfolio_step,
            transition_info=ref.TRANSITION_INFO,
        )
