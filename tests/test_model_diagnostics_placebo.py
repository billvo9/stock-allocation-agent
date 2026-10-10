"""
Placebo transforms: values move only between rows of the same date
(permutation) or backward within a symbol (stale), never forward, never
from labels, and never depending on later rows.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics.controls import CONTROL_SYMBOL_CODE, add_control_columns
from stock_agent.model_diagnostics.placebo import (
    block_ids,
    block_permuted_features,
    stale_features,
)
from stock_agent.model_validation.checks import assert_prefix_stable

UNIVERSE = ("A", "B", "C", "D")
FEATURES = ["feature_a", "feature_b", CONTROL_SYMBOL_CODE]


@pytest.fixture
def panel(labeled_panel):
    return add_control_columns(
        labeled_panel(symbols=UNIVERSE, sessions=200, horizon=5, seed=2), UNIVERSE
    )


def _permute(frame, draw=0, **kwargs):
    return block_permuted_features(frame, FEATURES, seed=11, draw=draw, block_length=20, **kwargs)


def test_permutation_preserves_each_dates_cross_section_and_the_labels(panel):
    permuted = _permute(panel)
    for date, original in panel.groupby("date"):
        moved = permuted[permuted["date"] == date]
        before = sorted(map(tuple, original[FEATURES].to_numpy()))
        after = sorted(map(tuple, moved[FEATURES].to_numpy()))
        assert before == after  # same feature vectors, whole rows kept together
    label_columns = ["date", "symbol", "target_return", "target_start_date", "target_end_date"]
    pd.testing.assert_frame_equal(permuted[label_columns], panel[label_columns])
    assert (permuted[CONTROL_SYMBOL_CODE] != panel[CONTROL_SYMBOL_CODE]).any()


def test_permutation_is_constant_within_each_block(panel):
    permuted = _permute(panel)
    blocks = block_ids(panel, block_length=20)
    mapping = pd.DataFrame(
        {"block": blocks, "symbol": panel["symbol"], "source": permuted[CONTROL_SYMBOL_CODE]}
    )
    per_block = mapping.groupby(["block", "symbol"])["source"].nunique()
    assert (per_block == 1).all()
    assert mapping.groupby("block")["source"].apply(tuple).nunique() > 1  # blocks differ


def test_permutation_ignores_labels(panel):
    relabeled = panel.assign(target_return=-panel["target_return"] * 9.0)
    pd.testing.assert_frame_equal(_permute(panel)[FEATURES], _permute(relabeled)[FEATURES])


def test_permutation_is_prefix_stable(panel):
    cutoffs = [panel["date"].iloc[300], panel["date"].iloc[600]]
    assert_prefix_stable(
        lambda frame: _permute(frame), panel, cutoffs=cutoffs, value_columns=FEATURES
    )


def test_permutation_pinned_mapping_and_seed_dependence(panel):
    first_block = panel[block_ids(panel, block_length=20) == 0]
    first_date = first_block[first_block["date"] == first_block["date"].min()]
    mapped = _permute(panel).loc[first_date.index, CONTROL_SYMBOL_CODE].tolist()

    def key(symbol):  # independent oracle of the documented SHA-256 rule
        digest = hashlib.sha256(f"11:block_permutation:0:0:{symbol}".encode()).digest()
        return int.from_bytes(digest[:8], "big") >> 11

    by_key = sorted(UNIVERSE, key=key)  # i-th name takes the i-th smallest key's row
    assert mapped == [UNIVERSE.index(symbol) for symbol in by_key]
    assert mapped == [0, 3, 2, 1]  # pinned: no numpy RNG involved
    assert not _permute(panel, draw=1)[FEATURES].equals(_permute(panel)[FEATURES])


def test_permutation_handles_listings_and_single_name_dates(labeled_panel):
    raw = labeled_panel(symbols=UNIVERSE, sessions=60, horizon=3, listing={"D": (25, 60)})
    raw = raw[~((raw["symbol"] != "A") & (raw["date"] == raw["date"].min()))]
    frame = add_control_columns(raw.reset_index(drop=True), UNIVERSE)
    permuted = block_permuted_features(frame, FEATURES, seed=3, draw=0, block_length=20)
    lone = frame["date"] == frame["date"].min()
    pd.testing.assert_frame_equal(permuted.loc[lone, FEATURES], frame.loc[lone, FEATURES])
    for _, rows in permuted.groupby("date"):  # a bijection on the symbols present
        present = set(frame.loc[rows.index, CONTROL_SYMBOL_CODE])
        assert set(rows[CONTROL_SYMBOL_CODE]) == present


def test_stale_features_shift_backward_on_the_session_calendar_without_wrapping(labeled_panel):
    raw = labeled_panel(symbols=UNIVERSE, sessions=40, horizon=3, drop_fraction=0.1, seed=4)
    frame = add_control_columns(raw, UNIVERSE)
    sessions = np.unique(frame["date"])
    frame["origin"] = (
        np.searchsorted(sessions, frame["date"].to_numpy()) * 100 + frame[CONTROL_SYMBOL_CODE]
    ).astype(float)
    stale = stale_features(frame, ["origin"], lag_sessions=3)
    for row, value in zip(frame.itertuples(), stale["origin"], strict=True):
        session = int(np.searchsorted(sessions, row.date))
        source = (
            frame[(frame["symbol"] == row.symbol) & (frame["date"] == sessions[session - 3])]
            if session >= 3
            else frame.iloc[0:0]
        )
        if len(source):
            assert value == (session - 3) * 100 + getattr(row, CONTROL_SYMBOL_CODE)
        else:
            assert np.isnan(value)  # no wrap-around, no nearest-row substitute
    assert stale.loc[np.searchsorted(sessions, frame["date"].to_numpy()) < 3, "origin"].isna().all()


@pytest.mark.parametrize("lag", [0, -1, 1.5, True])
def test_stale_features_reject_non_backward_shifts(panel, lag):
    with pytest.raises(ValueError, match="backward"):
        stale_features(panel, ["feature_a"], lag_sessions=lag)


def test_stale_features_are_prefix_stable_and_label_blind(panel):
    build = lambda frame: stale_features(frame, ["feature_a"], lag_sessions=7)
    cutoffs = [panel["date"].iloc[400], panel["date"].iloc[700]]
    assert_prefix_stable(build, panel, cutoffs=cutoffs, value_columns=["feature_a"])
    relabeled = panel.assign(target_return=panel["target_return"] + 1.0)
    pd.testing.assert_series_equal(build(panel)["feature_a"], build(relabeled)["feature_a"])


def test_blocks_are_anchored_at_the_first_test_session_and_stay_prefix_stable(panel):
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    anchor = sessions.iloc[50]
    blocks = block_ids(panel, block_length=20, anchor=anchor)
    by_date = pd.Series(blocks, index=panel["date"]).groupby(level=0).first()
    assert by_date.loc[anchor] == 0
    assert by_date.loc[sessions.iloc[49]] == -1
    assert by_date.loc[sessions.iloc[70]] == 1
    cutoffs = [sessions.iloc[80], sessions.iloc[150]]
    assert_prefix_stable(
        lambda frame: _permute(frame, anchor=anchor),
        panel,
        cutoffs=cutoffs,
        value_columns=FEATURES,
    )
