"""Feature contracts, the three feature families and missingness diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.asof import asof_fundamentals, current_fundamentals
from stock_agent.features.contracts import (
    FeatureContractError,
    validate_feature_contracts,
)
from stock_agent.features.families import (
    CONTRACTS_BY_NAME,
    FUNDAMENTALS_AVAILABLE_AT,
    build_feature_frame,
    family_features,
)
from stock_agent.features.frozen_universe import (
    UniverseRule,
    assert_membership_independent,
    freeze_universe,
)
from stock_agent.features.missingness import missingness_table
from stock_agent.forecasting.synthetic import SyntheticEquitySpec, synthetic_equity_panel
from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.checks import assert_prefix_stable
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows

FAMILIES = ("relative_technical", "risk_liquidity", "fundamental_quality")
SPEC = SyntheticEquitySpec(n_assets=8, n_sessions=420, late_listings=2, delistings=1, seed=5)


@pytest.fixture(scope="module")
def panel():
    return synthetic_equity_panel(SPEC)


@pytest.fixture(scope="module")
def built(panel):
    universe = freeze_universe(
        panel.prices, UniverseRule(60), universe_id="test", survivorship_free=True
    )
    frame, contracts = build_feature_frame(
        panel.prices, universe, FAMILIES, fundamentals=panel.fundamentals
    )
    return universe, frame, contracts


def build_at(prices, fundamentals, cutoff):
    """Features built from only the data known before `cutoff`."""

    prices = prices.loc[prices["date"] < cutoff]
    fundamentals = fundamentals.loc[fundamentals["available_at"] < cutoff]
    universe = freeze_universe(prices, UniverseRule(60), universe_id="t", survivorship_free=True)
    return build_feature_frame(prices, universe, FAMILIES, fundamentals=fundamentals)[0]


# --- contracts ---


def test_every_family_feature_has_availability_semantics(built):
    _, frame, contracts = built
    assert set(contracts) == set(family_features(FAMILIES))
    assert {c.family for c in contracts.values()} == set(FAMILIES)
    assert validate_feature_contracts(frame, list(contracts), contracts) == list(contracts)


def test_a_feature_without_a_contract_is_refused(built):
    _, frame, contracts = built
    frame = frame.assign(mystery_signal=1.0)
    with pytest.raises(FeatureContractError, match="no availability semantics"):
        validate_feature_contracts(frame, [*contracts, "mystery_signal"], contracts)


def test_contracts_must_be_complete():
    base = CONTRACTS_BY_NAME["volatility_20d"]
    for change in (
        {"family": "alpha"},
        {"availability": "whenever"},
        {"missingness": "drop_row"},
        {"availability_column": "x_available_at"},
        {"lookback_sessions": -1},
        {"cross_sectional": "across_dates"},
    ):
        with pytest.raises(FeatureContractError):
            replace(base, **change)
    with pytest.raises(FeatureContractError, match="availability_column"):
        replace(CONTRACTS_BY_NAME["quality_leverage"], availability_column=None)


def test_a_contract_does_not_launder_a_label_or_level(built):
    _, frame, _ = built
    leaky = replace(CONTRACTS_BY_NAME["volatility_20d"], name="target_return_shadow")
    frame = frame.assign(target_return_shadow=0.0)
    with pytest.raises(ValueError, match="label, key, timestamp"):
        validate_feature_contracts(frame, ["target_return_shadow"], {"target_return_shadow": leaky})


def test_values_used_before_their_availability_are_rejected(built):
    _, frame, contracts = built
    names = ["quality_gross_profitability"]
    late = frame.copy()
    late.loc[10, FUNDAMENTALS_AVAILABLE_AT] = late.loc[10, "date"] + pd.Timedelta(hours=21)
    with pytest.raises(LeakageError, match="before they were available"):
        validate_feature_contracts(late, names, contracts)
    unstamped = frame.copy()
    unstamped.loc[10, FUNDAMENTALS_AVAILABLE_AT] = pd.NaT
    with pytest.raises(LeakageError, match="without an availability time"):
        validate_feature_contracts(unstamped, names, contracts)
    same_day = frame.copy()
    same_day.loc[10, FUNDAMENTALS_AVAILABLE_AT] = same_day.loc[10, "date"]
    validate_feature_contracts(same_day, names, contracts)  # 00:00 UTC of d is allowed


def test_missingness_policies_are_enforced(built):
    _, frame, contracts = built
    refusing = {"volatility_60d": replace(contracts["volatility_60d"], missingness="refuse")}
    with pytest.raises(FeatureContractError, match="policy refuse"):
        validate_feature_contracts(frame, ["volatility_60d"], refusing)
    native = {"volatility_60d": replace(contracts["volatility_60d"], missingness="native_missing")}
    with pytest.raises(FeatureContractError, match="handles NaN"):
        validate_feature_contracts(frame, ["volatility_60d"], native)
    validate_feature_contracts(frame, ["volatility_60d"], native, estimator_handles_missing=True)


# --- families ---


def test_every_family_is_prefix_stable(panel, built):
    _, full, contracts = built
    names = list(contracts)
    dates = full["date"].drop_duplicates().sort_values()
    for cutoff in dates.iloc[[10, 90, 200, 340]]:
        partial = build_at(panel.prices, panel.fundamentals, cutoff)
        before = full.loc[full["date"] < cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial[["date", "symbol"]], before[["date", "symbol"]])
        np.testing.assert_array_equal(
            partial[names].to_numpy(np.float64), before[names].to_numpy(np.float64)
        )


def test_the_prefix_check_passes_the_price_families_and_catches_a_future_read(panel):
    names = family_features(("relative_technical", "risk_liquidity"))
    cutoffs = panel.prices["date"].drop_duplicates().sort_values().iloc[[90, 200, 330]]

    def honest(prices):
        universe = freeze_universe(
            prices, UniverseRule(60), universe_id="t", survivorship_free=True
        )
        frame, _ = build_feature_frame(prices, universe, ("relative_technical", "risk_liquidity"))
        return frame.loc[:, ["date", "symbol", *names]]

    def leaky(prices):  # a "feature" that reads the next close
        frame = honest(prices)
        ordered = prices.sort_values(["symbol", "date"])
        ahead = ordered.groupby("symbol")["close"].shift(-1) / ordered["close"] - 1.0
        nxt = ordered.loc[:, ["date", "symbol"]].assign(volatility_20d=ahead.to_numpy())
        return frame.drop(columns="volatility_20d").merge(nxt, on=["date", "symbol"])

    assert_prefix_stable(honest, panel.prices, cutoffs=cutoffs, value_columns=names)
    with pytest.raises(LeakageError, match="volatility_20d"):
        assert_prefix_stable(leaky, panel.prices, cutoffs=cutoffs, value_columns=names)


def test_relative_features_rank_within_the_date_only(built):
    _, frame, _ = built
    for name in ("rel_return_5d", "rel_momentum_20d"):
        by_date = frame.groupby("date")[name]
        assert np.allclose(by_date.min().dropna(), -0.5)
        assert np.allclose(by_date.max().dropna(), 0.5)
        assert np.allclose(by_date.mean().dropna(), 0.0, atol=1e-12)


def test_fundamentals_use_the_latest_period_then_its_latest_version():
    day = pd.Timestamp("2020-01-01", tz="UTC")
    observations = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "period_end": [day - pd.Timedelta(days=90), day, day - pd.Timedelta(days=90), day],
            "available_at": [
                day + pd.Timedelta(days=10, hours=21),  # Q3 original
                day + pd.Timedelta(days=40, hours=21),  # Q4 original
                day + pd.Timedelta(days=50, hours=21),  # late Q3 amendment: never current
                day + pd.Timedelta(days=60, hours=21),  # Q4 amendment: current from day 61
            ],
            "gross_profit": [1.0, 2.0, 9.0, 3.0],
            "total_assets": [10.0] * 4,
            "total_liabilities": [5.0] * 4,
        }
    )
    assert list(current_fundamentals(observations)["gross_profit"]) == [1.0, 2.0, 3.0]
    keys = pd.DataFrame(
        {"date": [day + pd.Timedelta(days=d) for d in (10, 11, 41, 55, 61)], "symbol": "A"}
    )
    values = asof_fundamentals(keys, observations)["gross_profit"].tolist()
    assert np.isnan(values[0])  # filed at 21:30 UTC on day 10: usable from day 11
    assert values[1:] == [1.0, 2.0, 2.0, 3.0]
    midnight = observations.iloc[[0]].assign(available_at=day + pd.Timedelta(days=10))
    on_time = asof_fundamentals(keys.iloc[[0]], midnight)["gross_profit"].tolist()
    assert on_time == [1.0]  # available at 00:00 UTC of d: usable on d


def test_ambiguous_same_instant_versions_are_refused():
    stamp = pd.Timestamp("2020-02-01T21:30", tz="UTC")
    observations = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "period_end": [pd.Timestamp("2019-12-31", tz="UTC")] * 2,
            "available_at": [stamp, stamp],
            "gross_profit": [1.0, 2.0],
            "total_assets": [10.0, 10.0],
            "total_liabilities": [5.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        current_fundamentals(observations)


# --- missingness ---


def test_missingness_has_a_reason_for_every_missing_value(panel, built):
    universe, _, contracts = built
    late_filings = panel.fundamentals.loc[
        panel.fundamentals["available_at"] > pd.Timestamp("2012-03-15", tz="UTC")
    ]
    later, _ = build_feature_frame(panel.prices, universe, FAMILIES, fundamentals=late_filings)
    table = missingness_table(later, contracts)
    overall = table[(table["scope"] == "panel") & (table["group"] == "all")].set_index("feature")
    reasons = ["n_warmup", "n_not_yet_available", "n_source_missing"]
    assert (overall[reasons].sum(axis=1) == overall["n_missing"]).all()
    assert overall.loc["rel_momentum_120_20d", "n_warmup"] > 0
    assert overall.loc["quality_gross_profitability", "n_not_yet_available"] > 0
    assert overall.loc["volatility_20d", "n_missing"] == 0
    assert (overall["n_rows"] == len(later)).all()


def test_missingness_flags_without_dropping_and_never_moves_membership(built):
    universe, frame, contracts = built
    holed = frame.copy()
    holed.loc[holed.index[::2], "volatility_20d"] = np.nan
    before = holed.copy()
    table = missingness_table(holed, contracts, warning_rate=0.25)
    row = table[(table["feature"] == "volatility_20d") & (table["group"] == "all")].iloc[0]
    assert row["status"] == "warning" and row["reason"] == "high_missing_rate"
    assert row["n_source_missing"] == row["n_missing"]
    pd.testing.assert_frame_equal(holed, before)
    assert_membership_independent(holed, universe)
    with pytest.raises(LeakageError, match="membership"):
        assert_membership_independent(holed.dropna(subset=["volatility_20d"]), universe)


def test_missingness_is_reported_per_fold_and_scope(built):
    _, frame, contracts = built
    labeled = frame.assign(
        target_return=0.0,
        target_start_date=frame["date"] + pd.Timedelta(days=1),
        target_end_date=frame["date"] + pd.Timedelta(days=8),
    )
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(
        sessions, start=sessions.iloc[250], end=sessions.iloc[340], block_sessions=30
    )
    folds = make_expanding_folds(labeled, windows, holdout_start=sessions.iloc[340])
    table = missingness_table(frame, contracts, folds=folds)
    training = table[(table["scope"] == "training") & (table["feature"] == "volatility_60d")]
    assert list(training["fold_id"]) == [f.fold_id for f in folds]
    assert list(training["n_rows"]) == [len(f.train_rows) for f in folds]
    evaluation = table[table["scope"] == "evaluation"]
    assert set(evaluation["fold_id"]) == {f.fold_id for f in folds}
