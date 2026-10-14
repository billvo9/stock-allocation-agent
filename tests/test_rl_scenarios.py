"""The deterministic, versioned synthetic scenario library."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from stock_agent.rl import scenarios as sc

REQUIRED = (
    "calm",
    "crash",
    "high_correlation",
    "extreme_volatility",
    "stale_observations",
    "high_costs",
)


@pytest.fixture(scope="module")
def library():
    return {name: sc.generate_scenario(name, seed=3) for name in sc.SCENARIOS}


def test_the_library_covers_every_required_regime():
    assert set(REQUIRED) <= set(sc.SCENARIOS)
    assert "missing_closes" in sc.SCENARIOS


def test_the_universe_is_etf_like_with_a_cash_account_last():
    assert sc.ASSET_NAMES == ("equity_etf", "bond_etf", "short_treasury_fund", "cash")
    assert sc.CASH_INDEX == 3
    np.testing.assert_allclose(sc.BASE_COST_RATES, [5e-4, 5e-4, 2e-4, 0.0])


@pytest.mark.parametrize("name", sorted(sc.SCENARIOS))
def test_generation_replays_exactly_from_id_version_and_seed(name):
    first = sc.generate_scenario(name, seed=11)
    second = sc.generate_scenario(name, seed=11, generator_version=sc.GENERATOR_VERSION)
    for field in ("returns", "has_close", "obs_lag", "cost_multiplier", "latent_returns"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != sc.generate_scenario(name, seed=12).fingerprint()


def test_scenarios_with_the_same_seed_differ():
    assert (
        sc.generate_scenario("calm", seed=5).fingerprint()
        != sc.generate_scenario("crash", seed=5).fingerprint()
    )


@pytest.mark.parametrize("version", ["0.9.0", "1.0.0"])
def test_an_unsupported_generator_version_is_refused_rather_than_approximated(version):
    with pytest.raises(ValueError, match="generator version"):
        sc.generate_scenario("calm", seed=0, generator_version=version)


def test_the_generator_version_is_1_1_0():
    assert sc.GENERATOR_VERSION == "1.1.0"
    assert sc.SUPPORTED_GENERATOR_VERSIONS == ("1.1.0",)


GOLDEN_SCENARIOS: dict[str, tuple[str, list[float]]] = {
    "calm": ("9e4e3d0a385cb119", [0.736495756176, 0.221370700766, 0.110284392593, 0.0]),
    "crash": ("9a58f4d4f58de783", [-0.557887554885, -0.126718766916, 0.082989396319, 0.0]),
    "extreme_volatility": (
        "9dd606b0a19585b7",
        [-0.382874764282, 0.30635467977, 0.107087731396, 0.0],
    ),
    "high_correlation": ("044be0f14b9fdfdb", [1.19891985694, 0.385731945602, 0.128736974577, 0.0]),
    "high_costs": ("351c2fffc2b502e6", [0.043641913529, 0.033147788244, 0.091783731496, 0.0]),
    "missing_closes": ("d450202f10b1b7d9", [0.734824803449, 0.142364140126, 0.113270656172, 0.0]),
    "mixed": ("24cd6dd4f1c1cceb", [-0.72644937922, 0.101484969295, 0.105012984643, 0.0]),
    "stale_observations": (
        "a9774781d636565c",
        [0.64383688675, 0.344770720275, 0.116872194327, 0.0],
    ),
}


def _exact_digest(data):
    digest = hashlib.sha256()
    for array in (data.has_close, data.obs_lag, data.cost_multiplier):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update("|".join(data.regime.tolist()).encode("utf-8"))
    return digest.hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(GOLDEN_SCENARIOS))
def test_golden_pins_for_every_scenario(name):
    """
    Integer and boolean arrays (closes, feed lags, cost multipliers, regimes)
    come from exact uniform draws, so they are pinned bitwise. Float returns
    pass through platform maths libraries, so they are pinned by per-asset
    sums to 1e-9 rather than by bytes.
    """

    data = sc.generate_scenario(name, seed=0)
    digest, sums = GOLDEN_SCENARIOS[name]
    assert _exact_digest(data) == digest
    np.testing.assert_allclose(np.nansum(data.returns, axis=0), sums, rtol=0, atol=1e-9)


def test_an_unknown_scenario_is_refused():
    with pytest.raises(KeyError):
        sc.generate_scenario("bull_market_forever", seed=0)


def test_a_golden_draw_pins_the_generator():
    data = sc.generate_scenario("calm", seed=0)
    np.testing.assert_allclose(data.returns[1, :3], GOLDEN_CALM_SEED0_SESSION1, rtol=0, atol=1e-12)


GOLDEN_CALM_SEED0_SESSION1 = [0.01128441641962181, -0.0005631654284133033, -3.365694687966892e-05]


def test_arrays_are_read_only(library):
    with pytest.raises(ValueError):
        library["calm"].returns[5, 0] = 0.0


def test_returns_exceed_minus_one_and_are_nan_exactly_where_no_close(library):
    for data in library.values():
        observed = data.returns[data.has_close]
        assert np.all(np.isfinite(observed)) and np.all(observed > -1.0)
        assert np.all(np.isnan(data.returns[~data.has_close]))


def test_cash_earns_zero_and_always_has_a_close_d4a(library):
    for data in library.values():
        assert np.all(data.has_close[:, sc.CASH_INDEX])
        assert np.all(data.returns[:, sc.CASH_INDEX] == 0.0)
        assert np.all(data.obs_lag[:, sc.CASH_INDEX] == 0)


def test_a_return_after_a_gap_compounds_the_whole_gap(library):
    data = library["missing_closes"]
    assert (~data.has_close).sum() > 10
    for t, i in zip(*np.nonzero(~data.has_close[:-5]), strict=True):
        nxt = t + 1
        while not data.has_close[nxt, i]:
            nxt += 1
        prev = t - 1
        while not data.has_close[prev, i]:
            prev -= 1
        expected = np.prod(1.0 + data.latent_returns[prev + 1 : nxt + 1, i]) - 1.0
        assert data.returns[nxt, i] == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_calm_has_no_gaps_no_staleness_and_unit_costs(library):
    data = library["calm"]
    assert data.has_close.all()
    assert not data.obs_lag.any()
    assert np.all(data.cost_multiplier == 1.0)


def test_the_stale_feed_lags_risky_assets_only_inside_stale_segments(library):
    data = library["stale_observations"]
    stale = data.regime == "stale_feed"
    assert data.obs_lag[stale, :3].max() >= 2
    assert data.obs_lag[~stale & (np.arange(data.sessions) > 200), :3].max() <= 5
    assert np.all(data.obs_lag <= np.arange(data.sessions)[:, None])


def test_high_costs_multiply_the_base_rates_in_their_segments(library):
    data = library["high_costs"]
    high = data.regime == "high_costs"
    assert np.all(data.cost_multiplier[high] == 5.0)
    assert np.all(data.cost_multiplier[~high] == 1.0)
    t = int(np.flatnonzero(high)[0])
    np.testing.assert_allclose(data.cost_rates_at(t), 5.0 * np.asarray(sc.BASE_COST_RATES))


def test_the_crash_regime_is_reliably_severe():
    declines = []
    for seed in range(100):
        data = sc.generate_scenario("crash", seed)
        crash = np.flatnonzero(data.regime == "crash")
        for segment in np.split(crash, np.flatnonzero(np.diff(crash) > 1) + 1):
            declines.append(np.prod(1.0 + data.latent_returns[segment, 0]) - 1.0)
    declines = np.array(declines)
    assert len(declines) == 500
    assert (declines >= 0).mean() < 0.01
    assert -0.40 < np.median(declines) < -0.20


def test_regime_moments_have_the_intended_signs():
    crash = sc.generate_scenario("crash", seed=1)
    in_crash = crash.regime == "crash"
    assert crash.latent_returns[in_crash, 0].mean() < -0.002

    corr = sc.generate_scenario("high_correlation", seed=1)
    high = corr.regime == "high_correlation"
    rho = np.corrcoef(corr.latent_returns[high, 0], corr.latent_returns[high, 1])[0, 1]
    calm_rho = np.corrcoef(corr.latent_returns[~high, 0], corr.latent_returns[~high, 1])[0, 1]
    assert rho > 0.75 and calm_rho < 0.1

    vol = sc.generate_scenario("extreme_volatility", seed=1)
    extreme = vol.regime == "extreme_volatility"
    ratio = vol.latent_returns[extreme, 0].std() / vol.latent_returns[~extreme, 0].std()
    assert ratio > 3.0


def test_every_fold_window_of_a_stress_scenario_contains_its_stress_regime(library):
    for name, regime in [
        ("crash", "crash"),
        ("high_correlation", "high_correlation"),
        ("extreme_volatility", "extreme_volatility"),
        ("stale_observations", "stale_feed"),
        ("missing_closes", "gappy"),
        ("high_costs", "high_costs"),
    ]:
        data = library[name]
        for start, stop in [(60, 900), (960, 1200), (1260, 1500)]:
            assert (data.regime[start:stop] == regime).any(), (name, start)


def test_complete_sessions_are_those_where_every_asset_has_a_close(library):
    data = library["missing_closes"]
    complete = data.complete_sessions(100, 400)
    assert np.all(data.has_close[complete].all(axis=1))
    expected = [t for t in range(100, 400) if data.has_close[t].all()]
    np.testing.assert_array_equal(complete, expected)


def test_span_returns_compound_each_assets_closes_between_complete_sessions(library):
    data = library["missing_closes"]
    complete = data.complete_sessions(0, data.sessions)
    gaps = np.flatnonzero(np.diff(complete) > 1)
    a, b = int(complete[gaps[0]]), int(complete[gaps[0] + 1])
    span = data.span_returns(a, b)
    expected = np.prod(1.0 + data.latent_returns[a + 1 : b + 1], axis=0) - 1.0
    np.testing.assert_allclose(span, expected, rtol=1e-12, atol=1e-15)


def test_span_returns_refuse_incomplete_endpoints(library):
    data = library["missing_closes"]
    incomplete = int(np.flatnonzero(~data.has_close.all(axis=1))[0])
    with pytest.raises(ValueError, match="complete"):
        data.span_returns(incomplete - 1 if incomplete > 0 else 0, incomplete)
    with pytest.raises(ValueError, match="after"):
        data.span_returns(10, 10)


def test_a_regime_with_a_non_psd_correlation_is_rejected():
    calm = sc.REGIMES["calm"]
    with pytest.raises(ValueError, match="positive definite"):
        dataclasses.replace(calm, correlation=(0.99, -0.99, 0.99))


def test_a_regime_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="probability"):
        dataclasses.replace(sc.REGIMES["calm"], gap_probability=1.0)
