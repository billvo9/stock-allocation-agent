"""Frozen eligible universes, prefix-invariant labels and the synthetic generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.asof import asof_fundamentals
from stock_agent.features.frozen_universe import (
    UniverseRule,
    eligible_keys,
    freeze_universe,
    verify_universe,
)
from stock_agent.features.invariant_labels import (
    assert_labels_prefix_invariant,
    frozen_universe_labels,
)
from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    LabelSpec,
    add_forward_return_target,
)
from stock_agent.forecasting.synthetic import SyntheticEquitySpec, synthetic_equity_panel
from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.checks import assert_prefix_stable

SPEC = SyntheticEquitySpec(n_assets=8, n_sessions=400, late_listings=2, delistings=2, seed=11)
RULE = UniverseRule(min_history_sessions=40)
RAW = LabelSpec(horizon=10, entry_lag=1, price_column="close")
EXCESS = replace(RAW, benchmark="equal_weight")


@pytest.fixture(scope="module")
def panel():
    return synthetic_equity_panel(SPEC)


def freeze(prices):
    return freeze_universe(prices, RULE, universe_id="test", survivorship_free=True)


def gappy(prices):
    """One symbol misses every 7th session, so its labels end later than its peers'."""

    first = prices["symbol"] == prices["symbol"].min()
    rank = prices.groupby("symbol").cumcount()
    return prices.loc[~(first & (rank % 7 == 3))].reset_index(drop=True)


def cutoffs(panel, positions=(60, 150, 240, 330, 390)):
    return panel.prices["date"].drop_duplicates().sort_values().iloc[list(positions)]


# --- frozen universe ---


def test_membership_uses_only_earlier_rows_of_the_same_symbol(panel):
    universe = freeze(panel.prices)
    membership = universe.membership
    first = membership.groupby("symbol")["date"].transform("min")
    rank = membership.groupby("symbol").cumcount() + 1
    assert (membership["history_sessions"] == rank).all()
    assert (membership["eligible"] == (rank >= RULE.min_history_sessions)).all()
    early = membership.loc[~membership["eligible"], "reason"].unique().tolist()
    assert early == ["insufficient_history"]
    assert (membership.loc[membership["eligible"], "date"] > first[membership["eligible"]]).all()
    assert_prefix_stable(
        lambda prices: freeze(prices).membership,
        panel.prices,
        cutoffs=cutoffs(panel),
        value_columns=["history_sessions", "eligible", "reason"],
    )


def test_invalid_prices_are_ineligible_with_a_reason(panel):
    prices = panel.prices.copy()
    prices.loc[500, "close"] = -1.0
    membership = freeze(prices).membership.set_index(["date", "symbol"])
    key = (prices.loc[500, "date"], prices.loc[500, "symbol"])
    assert not membership.loc[key, "eligible"]
    assert membership.loc[key, "reason"] == "invalid_price"


def test_a_frozen_universe_refuses_a_changed_membership(panel):
    universe = freeze(panel.prices)
    verify_universe(universe)
    edited = universe.membership.copy()
    edited.loc[edited.index[-1], "eligible"] = not edited.loc[edited.index[-1], "eligible"]
    with pytest.raises(LeakageError, match="frozen hash"):
        verify_universe(replace(universe, membership=edited))
    assert freeze(panel.prices).sha256 == universe.sha256
    other_rule = freeze_universe(
        panel.prices, UniverseRule(41), universe_id="test", survivorship_free=True
    )
    assert other_rule.sha256 != universe.sha256


# --- labels ---


@pytest.mark.parametrize("gaps", [False, True], ids=["aligned", "gappy"])
@pytest.mark.parametrize("spec", [RAW, EXCESS], ids=["raw", "excess"])
def test_later_data_never_changes_an_earlier_label(panel, spec, gaps):
    def build(prices):
        return frozen_universe_labels(prices, freeze(prices), spec)

    prices = gappy(panel.prices) if gaps else panel.prices
    assert_labels_prefix_invariant(build, prices, cutoffs=cutoffs(panel))


def test_a_survivor_benchmark_is_caught_as_label_leakage(panel):
    def survivors_only(prices):  # benchmark over names whose label is known: look-ahead
        labels = frozen_universe_labels(prices, freeze(prices), RAW)
        mean = labels.groupby("date")[TARGET_RETURN_COLUMN].transform("mean")
        labels[TARGET_RETURN_COLUMN] = labels[TARGET_RETURN_COLUMN] - mean
        labels[TARGET_END_COLUMN] = labels.groupby("date")[TARGET_END_COLUMN].transform("max")
        return labels

    with pytest.raises(LeakageError, match="Prefix instability"):
        assert_labels_prefix_invariant(survivors_only, gappy(panel.prices), cutoffs=cutoffs(panel))


def test_a_full_sample_normalized_label_is_caught(panel):
    def normalized(prices):  # divides by a volatility estimated on all the data
        labels = frozen_universe_labels(prices, freeze(prices), RAW)
        labels[TARGET_RETURN_COLUMN] /= labels[TARGET_RETURN_COLUMN].std()
        return labels

    with pytest.raises(LeakageError, match="target_return"):
        assert_labels_prefix_invariant(normalized, panel.prices, cutoffs=cutoffs(panel))


def test_withholding_matured_labels_near_the_data_end_is_caught(panel):
    def cautious(prices):  # blanks the last 30 sessions of whatever data it is given
        labels = frozen_universe_labels(prices, freeze(prices), RAW)
        dates = labels["date"].drop_duplicates().sort_values()
        recent = labels["date"] >= dates.iloc[-min(30, len(dates))]
        labels.loc[recent, TARGET_RETURN_COLUMN] = np.nan
        return labels

    with pytest.raises(LeakageError, match="matured before the cutoff"):
        assert_labels_prefix_invariant(cautious, panel.prices, cutoffs=cutoffs(panel))


def test_excess_labels_use_the_eligible_members_and_fail_closed(panel):
    universe = freeze(panel.prices)
    excess = frozen_universe_labels(panel.prices, universe, EXCESS)
    raw = add_forward_return_target(
        panel.prices, horizon=RAW.horizon, price_column="close", entry_lag=RAW.entry_lag
    )
    raw["date"] = raw["date"].dt.as_unit("ns")
    merged = eligible_keys(universe).merge(raw, on=["date", "symbol"], how="left")
    complete = merged.groupby("date")[TARGET_RETURN_COLUMN].transform(
        lambda values: bool(np.isfinite(values).all())
    )
    expected = (
        merged[TARGET_RETURN_COLUMN]
        - merged.groupby("date")[TARGET_RETURN_COLUMN].transform("mean")
    ).where(complete.astype(bool))
    np.testing.assert_allclose(
        excess[TARGET_RETURN_COLUMN].to_numpy(), expected.to_numpy(), equal_nan=True
    )
    assert (excess.groupby("date")[TARGET_RETURN_COLUMN].sum().abs() < 1e-12).all()
    # A delisted member blanks every date whose label it would cover (fail closed).
    last = panel.prices.groupby("symbol")["date"].max()
    delisted_end = last[last < panel.prices["date"].max()].min()
    blanked = excess.loc[
        (excess["date"] <= delisted_end) & (excess["date"] > delisted_end - pd.Timedelta(days=12))
    ]
    assert blanked[TARGET_RETURN_COLUMN].isna().all()


def test_labels_cover_exactly_the_eligible_keys_whatever_the_features(panel):
    universe = freeze(panel.prices)
    labels = frozen_universe_labels(panel.prices, universe, EXCESS)
    pd.testing.assert_frame_equal(labels[["date", "symbol"]], eligible_keys(universe))


def test_labels_refuse_a_price_column_other_than_the_universe_s(panel):
    with pytest.raises(ValueError, match="price column"):
        frozen_universe_labels(panel.prices, freeze(panel.prices), LabelSpec(10, 1))


# --- synthetic generator ---


def test_the_generator_is_deterministic_and_stays_before_the_lockbox():
    first, second = synthetic_equity_panel(SPEC), synthetic_equity_panel(SPEC)
    pd.testing.assert_frame_equal(first.prices, second.prices)
    pd.testing.assert_frame_equal(first.fundamentals, second.fundamentals)
    assert first.prices["date"].max() < pd.Timestamp("2025-01-01", tz="UTC")
    with pytest.raises(ValueError, match="lockbox"):
        synthetic_equity_panel(replace(SPEC, start="2024-01-01"))
    assert not synthetic_equity_panel(replace(SPEC, seed=12)).prices.equals(first.prices)


def test_the_planted_drift_uses_only_observable_information(panel):
    observed = asof_fundamentals(panel.truth[["date", "symbol"]], panel.fundamentals)
    profitability = observed["gross_profit"] / observed["total_assets"]
    frame = panel.truth.assign(q=profitability.to_numpy())
    by_date = frame.groupby("date")["q"]
    z = (frame["q"] - by_date.transform("mean")) / by_date.transform(lambda v: v.std(ddof=0))
    np.testing.assert_allclose(frame["drift"], SPEC.signal_strength * z.fillna(0.0), atol=1e-15)


def test_null_and_nonlinear_drifts_have_the_documented_shape():
    null = synthetic_equity_panel(replace(SPEC, signal="none"))
    assert (null.truth["drift"] == 0.0).all()
    bent = synthetic_equity_panel(replace(SPEC, signal="nonlinear"))
    observed = asof_fundamentals(bent.truth[["date", "symbol"]], bent.fundamentals)
    q = (observed["gross_profit"] / observed["total_assets"]).to_numpy()
    frame = bent.truth.assign(q=q)
    by_date = frame.groupby("date")
    assert (by_date["drift"].sum().abs() < 1e-15).all()
    linear_fit = by_date.apply(
        lambda g: np.corrcoef(g["q"], g["drift"])[0, 1], include_groups=False
    )
    assert linear_fit.abs().median() < 0.5  # mostly symmetric: little linear content
