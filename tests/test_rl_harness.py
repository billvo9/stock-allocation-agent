"""
The comparison harness: every method on the same dates, assets, costs and
constraints, and no comparison table until the transition is approved.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stock_agent.rl import env as rl_env
from stock_agent.rl import forecasts, harness, replay, splits
from stock_agent.rl import observation as ob
from stock_agent.rl import policies as pol
from stock_agent.rl import reference_transition as ref
from stock_agent.rl import scenarios as sc
from stock_agent.rl import transition as tr

SPEC = ob.FeatureSpec()
FOLD = splits.FoldSpec("dev", ("calm", "high_costs"), (10, 11), 260, 420)
ENV_CONFIG = rl_env.EnvConfig(rebalance_every=5)
APPROVED = tr.TransitionInfo(name="approved.portfolio_step", version="1.0.0", status="approved")


@pytest.fixture(scope="module")
def builder():
    train = splits.FoldSpec("train", ("mixed",), (0,), 60, 240)
    return ob.ObservationBuilder(SPEC, ob.fit_normalizer(train, SPEC), sc.ASSET_NAMES, 3)


def _entries():
    return [
        *harness.baseline_entries(),
        harness.PolicyEntry("ppo_seed_0", pol.equal_weight, kind="ppo", group="ppo"),
        harness.PolicyEntry("ppo_seed_1", pol.fixed_balanced, kind="ppo", group="ppo"),
        harness.PolicyEntry(
            "ppo_with_forecasts", forecasts.unavailable("Track B"), kind="ppo_with_forecasts"
        ),
    ]


def _run(builder, info=ref.TRANSITION_INFO):
    return harness.run_comparison(
        _entries(),
        FOLD,
        builder=builder,
        transition=ref.proposed_portfolio_step,
        transition_info=info,
        env_config=ENV_CONFIG,
    )


@pytest.fixture(scope="module")
def comparison(builder):
    return _run(builder)


def test_every_available_method_runs_every_episode(comparison):
    names = {entry.name for entry in _entries()} - {"ppo_with_forecasts"}
    assert set(comparison.records) == names
    for records in comparison.records.values():
        assert set(records) == set(FOLD.episodes())


def test_the_baselines_cover_the_registered_methods():
    assert [entry.name for entry in harness.baseline_entries()] == [
        "buy_and_hold",
        "equal_weight",
        "fixed_balanced",
        "inverse_volatility",
        "mean_variance",
    ]


def test_methods_share_dates_assets_costs_and_constraints(comparison):
    for episode in FOLD.episodes():
        records = [comparison.records[name][episode] for name in comparison.records]
        first = records[0]
        for record in records[1:]:
            assert record.identity == first.identity
            assert [r.session for r in record.ledger] == [r.session for r in first.ledger]
            for a, b in zip(record.ledger, first.ledger, strict=True):
                np.testing.assert_array_equal(a.cost_rates, b.cost_rates)


def test_the_fairness_check_rejects_a_method_run_on_a_different_window(builder):
    data = sc.generate_scenario("calm", 10)

    def record(start):
        env = rl_env.AllocationEnv(
            data,
            start=start,
            stop=420,
            transition=ref.proposed_portfolio_step,
            transition_info=ref.TRANSITION_INFO,
            builder=builder,
            config=ENV_CONFIG,
        )
        return replay.rollout(env, pol.equal_weight())

    with pytest.raises(harness.FairnessError):
        harness.check_fairness({"a": {("calm", 10): record(260)}, "b": {("calm", 10): record(265)}})


def test_unavailable_methods_are_reported_not_silently_dropped(comparison):
    assert "ppo_with_forecasts" in comparison.unavailable
    assert "Track B" in comparison.unavailable["ppo_with_forecasts"]


def test_the_engineering_report_has_no_performance_figures(comparison):
    report = comparison.engineering_report()
    assert report["label"] == "engineering_only_unapproved_transition"
    assert report["comparison_tables"].startswith("withheld")
    text = json.dumps(report, allow_nan=False)
    for word in ("log_growth", "terminal_wealth", "drawdown", "sharpe", "return"):
        assert word not in text
    entry = report["methods"]["mean_variance"]
    assert entry["episodes"] == 4 and entry["all_finite"] and entry["invalid_targets"] == 0
    assert report["fairness"] == "passed"


def test_no_comparison_table_without_an_approved_transition(comparison):
    with pytest.raises(harness.UnapprovedTransitionError):
        comparison.comparison_table()


def test_with_an_approved_transition_learned_seeds_are_a_distribution(builder):
    table = _run(builder, info=APPROVED).comparison_table()
    rows = {(row["method"], row["scenario"]): row for row in table["rows"]}
    assert ("equal_weight", "calm") in rows
    group = table["groups"]["ppo"]
    assert group["members"] == ["ppo_seed_0", "ppo_seed_1"]
    calm = group["by_scenario"]["calm"]
    assert calm["n"] == 2 and calm["min"] <= calm["median"] <= calm["max"]
    assert table["label"] == "approved_transition"


def test_comparison_tables_never_carry_engineering_diagnostics(builder):
    table = _run(builder, info=APPROVED).comparison_table()
    for row in table["rows"]:
        assert set(row) == {"method", "scenario", "episodes", "log_growth", "trades", "fallbacks"}
    record = next(iter(next(iter(_run(builder, info=APPROVED).records.values())).values()))
    rows = {(r["method"], r["scenario"]): r for r in table["rows"]}
    assert rows[("buy_and_hold", "calm")]["log_growth"] == pytest.approx(
        np.mean(
            [
                rec.total_reward
                for (scenario, _), rec in _run(builder, info=APPROVED)
                .records["buy_and_hold"]
                .items()
                if scenario == "calm"
            ]
        )
    )
    assert record.engineering_diagnostics()["kind"].startswith("engineering")


def test_forecast_providers_must_be_certified_out_of_fold_and_point_in_time():
    class Provider:
        artifact = forecasts.ForecastArtifact(
            "fc-1", "1.0.0", out_of_fold=False, point_in_time=True
        )

        def forecasts_at(self, scenario_id, seed, decision_session):
            return np.zeros(3)

    with pytest.raises(splits.LeakageError, match="out-of-fold"):
        forecasts.require_certified(Provider())
    Provider.artifact = forecasts.ForecastArtifact("fc-1", "1.0.0", True, True)
    assert forecasts.require_certified(Provider()).artifact.artifact_id == "fc-1"
