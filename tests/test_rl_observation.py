"""
Observations contain only information available at decision session d.

The negative tests prove leakage is rejected: future returns cannot reach a
feature, a snapshot holding rows after d cannot be built, data the feed has
not delivered yet is invisible, and normalization cannot be fitted on
anything but the training fold.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from stock_agent.rl import observation as ob
from stock_agent.rl import scenarios as sc
from stock_agent.rl import splits

SPEC = ob.FeatureSpec()


@pytest.fixture(scope="module")
def calm():
    return sc.generate_scenario("calm", seed=2)


def _with(data, **arrays):
    return dataclasses.replace(data, **{k: np.array(v) for k, v in arrays.items()})


def test_a_snapshot_holds_rows_up_to_d_only(calm):
    snap = ob.snapshot_at(calm, 100)
    assert snap.returns.shape == (101, calm.n_assets)
    assert snap.decision_session == 100


def test_a_snapshot_with_rows_after_the_decision_session_is_rejected(calm):
    with pytest.raises(splits.LeakageError, match="after the decision session"):
        ob.MarketSnapshot(
            decision_session=100,
            returns=np.array(calm.returns[:102]),
            obs_age=np.zeros(calm.n_assets, dtype=np.int64),
            cost_multiplier=1.0,
        )


def test_features_at_d_ignore_every_return_after_d(calm):
    d = 300
    base = ob.market_features(ob.snapshot_at(calm, d), SPEC, calm.cash_index)
    future = np.array(calm.returns)
    future[d + 1 :, :3] = 0.5  # r_{d+1}, r_{d+2}, ... replaced by absurd values
    changed = _with(calm, returns=future)
    np.testing.assert_array_equal(
        ob.market_features(ob.snapshot_at(changed, d), SPEC, calm.cash_index), base
    )


def test_features_at_d_change_when_r_d_changes(calm):
    d = 300
    base = ob.market_features(ob.snapshot_at(calm, d), SPEC, calm.cash_index)
    present = np.array(calm.returns)
    present[d, 0] += 0.01
    changed = ob.market_features(ob.snapshot_at(_with(calm, returns=present), d), SPEC, 3)
    assert not np.array_equal(changed, base)


def test_a_stale_feed_hides_returns_the_feed_has_not_delivered(calm):
    d, lag = 300, 3
    obs_lag = np.array(calm.obs_lag)
    obs_lag[d, 0] = lag
    lagged = _with(calm, obs_lag=obs_lag)
    snap = ob.snapshot_at(lagged, d)
    assert np.all(np.isnan(snap.returns[d - lag + 1 :, 0]))
    assert snap.obs_age[0] == lag
    base = ob.market_features(snap, SPEC, 3)
    hidden = np.array(calm.returns)
    hidden[d - lag + 1 : d + 1, 0] = 0.4
    again = ob.market_features(ob.snapshot_at(_with(lagged, returns=hidden), d), SPEC, 3)
    np.testing.assert_array_equal(again, base)


def test_trailing_return_and_volatility_by_hand(calm):
    d = 200
    feats = dict(
        zip(
            SPEC.market_names(calm.asset_names, calm.cash_index),
            ob.market_features(ob.snapshot_at(calm, d), SPEC, calm.cash_index),
            strict=True,
        )
    )
    window = calm.returns[d - 19 : d + 1, 0]
    assert feats["equity_etf:ret_20"] == pytest.approx(np.prod(1 + window) - 1, rel=1e-12)
    assert feats["equity_etf:vol_20"] == pytest.approx(window.std(ddof=1), rel=1e-12)


def test_a_feature_spec_round_trips():
    spec = ob.FeatureSpec(return_windows=(10,), vol_windows=(30,), include_cost_regime=False)
    assert ob.FeatureSpec.from_dict(spec.to_dict()) == spec


def test_feature_names_are_stable():
    names = SPEC.market_names(sc.ASSET_NAMES, sc.CASH_INDEX)
    assert names[:5] == (
        "equity_etf:ret_5",
        "equity_etf:ret_20",
        "equity_etf:ret_60",
        "equity_etf:vol_20",
        "equity_etf:vol_60",
    )
    assert names[-1] == "cost_multiplier"
    assert len(names) == 3 * 5 + 1
    assert SPEC.lookback == 60


def test_the_cost_regime_is_the_one_known_at_d_not_at_execution(calm):
    d = 300
    multiplier = np.array(calm.cost_multiplier)
    multiplier[d + 1 :] = 9.0
    feats = ob.market_features(ob.snapshot_at(_with(calm, cost_multiplier=multiplier), d), SPEC, 3)
    assert feats[-1] == 1.0


def test_an_unobservable_cost_regime_is_left_out(calm):
    hidden = dataclasses.replace(calm, cost_regime_observable=False)
    snap = ob.snapshot_at(hidden, 300)
    assert snap.cost_multiplier is None
    spec = ob.FeatureSpec(include_cost_regime=False)
    assert len(ob.market_features(snap, spec, 3)) == 15
    with pytest.raises(ValueError, match="cost regime"):
        ob.market_features(snap, SPEC, 3)


def test_too_little_visible_history_gives_nan_not_a_guess(calm):
    feats = ob.market_features(ob.snapshot_at(calm, 10), SPEC, 3)
    names = SPEC.market_names(calm.asset_names, calm.cash_index)
    assert np.isnan(feats[names.index("equity_etf:ret_60")])
    assert np.isfinite(feats[names.index("equity_etf:ret_5")])


@pytest.mark.parametrize(
    "kwargs",
    [{"return_windows": (0, 20)}, {"vol_windows": (1,)}, {"return_windows": (-5,)}],
)
def test_invalid_windows_are_rejected(kwargs):
    with pytest.raises(ValueError, match="window"):
        ob.FeatureSpec(**kwargs)


def _small_plan():
    return splits.SplitPlan(
        train=splits.FoldSpec("train", ("calm", "crash"), (0, 1), 60, 400),
        dev=splits.FoldSpec("dev", ("calm",), (10,), 460, 700),
        test=splits.FoldSpec("test", ("calm",), (20,), 760, 1000),
        embargo=60,
    )


def test_normalization_is_fitted_on_the_training_fold_only():
    plan = _small_plan()
    norm = ob.fit_normalizer(plan.train, SPEC)
    assert norm.fitted_on["role"] == "train"
    assert norm.fitted_on["seeds"] == [0, 1]
    assert norm.fitted_on["window"] == [60, 400]
    for fold in (plan.dev, plan.test):
        with pytest.raises(splits.LeakageError, match="training fold"):
            ob.fit_normalizer(fold, SPEC)


def test_normalization_never_reads_sessions_after_the_training_window():
    plan = _small_plan()
    base = ob.fit_normalizer(plan.train, SPEC)

    def poisoned(scenario_id, seed):
        data = sc.generate_scenario(scenario_id, seed)
        returns = np.array(data.returns)
        returns[plan.train.stop :, :3] = 0.9
        return dataclasses.replace(data, returns=returns)

    again = ob.fit_normalizer(plan.train, SPEC, loader=poisoned)
    assert again.mean == base.mean and again.scale == base.scale


def test_a_window_without_enough_history_is_refused():
    fold = splits.FoldSpec("train", ("calm",), (0,), 30, 400)
    with pytest.raises(ValueError, match="lookback"):
        ob.fit_normalizer(fold, SPEC)


def test_a_frozen_normalizer_round_trips_and_never_updates():
    norm = ob.fit_normalizer(_small_plan().train, SPEC)
    assert ob.FeatureNormalizer.from_dict(norm.to_dict()) == norm
    with pytest.raises(dataclasses.FrozenInstanceError):
        norm.mean = ()  # type: ignore[misc]
    raw = np.zeros(len(norm.names))
    before = norm.to_dict()
    norm.transform(raw)
    assert norm.to_dict() == before


def test_a_constant_training_feature_is_centered_not_divided_by_zero():
    norm = ob.fit_normalizer(_small_plan().train, SPEC)
    index = norm.names.index("cost_multiplier")
    assert norm.scale[index] == 1.0
    assert "cost_multiplier" in norm.fitted_on["constant_features"]


def test_transform_clips_extremes_but_reports_the_unclipped_size():
    norm = ob.FeatureNormalizer.identity(("a", "b"), clip=5.0)
    clipped, z = norm.transform(np.array([100.0, -1.0]))
    np.testing.assert_array_equal(clipped, [5.0, -1.0])
    np.testing.assert_array_equal(z, [100.0, -1.0])


def test_the_builder_appends_weights_and_drawdown_after_the_market_block(calm):
    norm = ob.fit_normalizer(_small_plan().train, SPEC)
    builder = ob.ObservationBuilder(SPEC, norm, calm.asset_names, calm.cash_index)
    weights = np.array([0.5, 0.3, 0.1, 0.1])
    obs = builder.build(ob.snapshot_at(calm, 300), weights, drawdown=0.07)
    assert obs.vector.shape == (builder.dimension,)
    np.testing.assert_array_equal(obs.vector[-5:-1], weights)
    assert obs.vector[-1] == 0.07
    assert obs.max_obs_age == 0 and obs.finite
    assert builder.names[-5:] == (
        "weight:equity_etf",
        "weight:bond_etf",
        "weight:short_treasury_fund",
        "weight:cash",
        "drawdown",
    )


def test_a_builder_refuses_a_normalizer_for_other_features(calm):
    norm = ob.FeatureNormalizer.identity(("x",))
    with pytest.raises(ValueError, match="normalizer"):
        ob.ObservationBuilder(SPEC, norm, calm.asset_names, calm.cash_index)


def test_a_normalizer_for_another_feature_spec_or_generator_is_refused():
    plan = _small_plan()
    norm = ob.fit_normalizer(plan.train, SPEC)
    ob.require_fitted_on(norm, plan.train, SPEC)
    other_spec = ob.FeatureSpec(return_windows=(5, 20, 60), vol_windows=(20, 60), min_fraction=0.6)
    with pytest.raises(splits.LeakageError, match="feature spec"):
        ob.require_fitted_on(norm, plan.train, other_spec)
    stale = dataclasses.replace(norm, fitted_on={**norm.fitted_on, "generator_versions": ["1.0.0"]})
    with pytest.raises(splits.LeakageError, match="generator"):
        ob.require_fitted_on(stale, plan.train, SPEC)
