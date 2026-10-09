"""
Feature diagnostics: training scope reads only fold training rows,
evaluation scope reads test features but never labels, nothing is mutated,
and undefined statistics carry a reason instead of a misleading number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics import feature_diagnostics as fd
from stock_agent.model_validation.audit import frame_content_sha256
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows

FEATURES = ["feature_a", "feature_b"]


def _setup(labeled_panel):
    panel = labeled_panel(symbols=("A", "B", "C", "D"), sessions=200, horizon=5, seed=3)
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    holdout = sessions.iloc[-1] + pd.offsets.BDay(1)
    windows = make_test_windows(
        sessions, start=sessions.iloc[100], end=sessions.iloc[190], block_sessions=30
    )
    folds = make_expanding_folds(panel, windows, holdout_start=holdout, lockbox_start=holdout)
    return panel, folds


def _value(stats, fold_id, feature, statistic):
    row = stats[
        (stats["fold_id"] == fold_id)
        & (stats["feature"] == feature)
        & (stats["statistic"] == statistic)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_training_statistics_read_only_the_folds_training_rows(labeled_panel):
    panel, folds = _setup(labeled_panel)
    stats, pairs = fd.training_feature_diagnostics(panel, folds, FEATURES)
    poisoned = panel.copy()
    outside = np.setdiff1d(np.arange(len(panel)), folds[0].train_rows)
    poisoned.loc[outside, FEATURES] = 1e9  # test rows, purge band, later rows
    stats_p, pairs_p = fd.training_feature_diagnostics(poisoned, folds[:1], FEATURES)
    first = stats[stats["fold_id"] == folds[0].fold_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, stats_p)
    pd.testing.assert_frame_equal(pairs[pairs["fold_id"] == 0].reset_index(drop=True), pairs_p)
    train = panel.loc[folds[0].train_rows, "feature_a"]
    assert _value(stats, 0, "feature_a", "median")["value"] == pytest.approx(train.median())
    assert _value(stats, 0, "feature_a", "n")["value"] == len(train)
    assert set(stats["scope"]) == {fd.TRAINING_SCOPE}


def test_diagnostics_never_mutate_their_inputs(labeled_panel):
    panel, folds = _setup(labeled_panel)
    columns = [*FEATURES, "target_return", "target_end_date"]
    before = frame_content_sha256(panel, columns)
    fd.training_feature_diagnostics(panel, folds, FEATURES)
    fd.evaluation_feature_drift(panel, folds, FEATURES)
    assert frame_content_sha256(panel, columns) == before


def test_evaluation_drift_reads_features_only_and_never_labels(labeled_panel):
    panel, folds = _setup(labeled_panel)
    drift = fd.evaluation_feature_drift(panel, folds, FEATURES)
    relabeled = panel.assign(target_return=panel["target_return"] * -4.0)
    pd.testing.assert_frame_equal(drift, fd.evaluation_feature_drift(relabeled, folds, FEATURES))
    assert set(drift["scope"]) == {fd.EVALUATION_SCOPE}
    shifted = panel.copy()
    test_rows = folds[0].test_rows
    shifted.loc[test_rows, "feature_a"] += 3.0
    moved = fd.evaluation_feature_drift(shifted, folds[:1], FEATURES)
    assert _value(moved, 0, "feature_a", "psi")["value"] > 1.0
    assert _value(moved, 0, "feature_a", "standardized_mean_difference")["value"] > 2.0
    assert _value(drift, 0, "feature_a", "psi")["value"] < 0.5


def test_psi_and_ks_are_zero_for_identical_samples():
    x = np.random.RandomState(0).randn(500)
    assert fd._psi(x, x) == pytest.approx(0.0)
    assert fd._ks(x, x) == 0.0
    assert fd._ks(x, x + 100.0) == 1.0


def test_missing_and_infinite_values_are_counted_separately():
    values = np.array([1.0, np.nan, np.inf, -np.inf, 2.0, 3.0])
    out = fd._univariate(values, np.arange(6), extreme_z=5.0, near_constant_share=0.99)
    assert out["n_missing"][0] == 1
    assert out["n_nonfinite"][0] == 2
    assert out["median"][0] == 2.0


def test_zero_mad_and_constant_features_report_reasons():
    out = fd._univariate(
        np.r_[np.zeros(50), 5.0], np.arange(51), extreme_z=5.0, near_constant_share=0.99
    )
    assert out["n_extreme_robust_z"][1:] == ("unavailable", "zero_mad")
    constant = fd._univariate(
        np.full(10, 0.1), np.arange(10), extreme_z=5.0, near_constant_share=0.99
    )
    assert constant["near_constant"][0] == 1.0
    assert constant["skew"][2] == "constant_feature"


def test_date_variance_share_separates_date_common_from_within_date_variation():
    dates = np.repeat(np.arange(50), 4)
    common = np.repeat(np.random.RandomState(1).randn(50), 4)
    within = np.tile([1.0, -1.0, 2.0, -2.0], 50)
    pick = lambda v: fd._univariate(v, dates, extreme_z=5.0, near_constant_share=0.99)
    assert pick(common)["date_variance_share"][0] == pytest.approx(1.0)
    assert pick(within)["date_variance_share"][0] == pytest.approx(0.0)


def _collinearity(matrix):
    rows = fd._collinearity(0, [f"x{i}" for i in range(matrix.shape[1])], matrix)
    return {(r["feature"], r["statistic"]): r for r in rows}


def test_vif_matches_the_closed_form_for_two_correlated_features():
    rng = np.random.RandomState(2)
    a = rng.randn(2000)
    b = rng.randn(2000)
    a = (a - a.mean()) / a.std()
    b -= b.mean()
    b -= (b @ a) / (a @ a) * a  # Gram-Schmidt: exactly orthogonal to a
    b /= b.std()
    x2 = 0.8 * a + 0.6 * b  # correlation exactly 0.8
    out = _collinearity(np.column_stack([a, x2]))
    assert out[("x0", "vif")]["value"] == pytest.approx(1 / (1 - 0.64))
    assert out[("__design__", "condition_number")]["value"] == pytest.approx(3.0)


def test_condition_number_is_one_for_an_orthogonal_design():
    q, _ = np.linalg.qr(np.random.RandomState(3).randn(100, 3))
    out = _collinearity(q - q.mean(axis=0))
    assert out[("__design__", "condition_number")]["value"] == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        (np.random.RandomState(4).randn(20, 1), "single_feature"),
        (np.random.RandomState(4).randn(3, 2), "too_few_rows"),
        (np.column_stack([np.ones(20), np.arange(20.0)]), "constant_feature"),
        (np.column_stack([np.arange(20.0), 2 * np.arange(20.0)]), "singular"),
    ],
)
def test_undefined_collinearity_statistics_carry_reasons(matrix, reason):
    out = _collinearity(matrix)
    design = out[("__design__", "condition_number")]
    assert design["status"] == "unavailable"
    assert np.isnan(design["value"])
    assert design["reason"] == reason


def test_lockbox_rows_outside_every_fold_never_reach_the_statistics(labeled_panel):
    panel, folds = _setup(labeled_panel)
    clean_stats, _ = fd.training_feature_diagnostics(panel, folds, FEATURES)
    clean_drift = fd.evaluation_feature_drift(panel, folds, FEATURES)
    # Rows after the last fold (stand-ins for lockbox rows) get poison values.
    later = np.setdiff1d(
        np.arange(len(panel)),
        np.concatenate([np.concatenate([f.train_rows, f.test_rows]) for f in folds]),
    )
    later = later[panel.loc[later, "date"] >= folds[-1].test_end]
    assert len(later)
    poisoned = panel.copy()
    poisoned.loc[later, FEATURES] = np.nan
    stats, _ = fd.training_feature_diagnostics(poisoned, folds, FEATURES)
    pd.testing.assert_frame_equal(stats, clean_stats)
    pd.testing.assert_frame_equal(
        fd.evaluation_feature_drift(poisoned, folds, FEATURES), clean_drift
    )


def test_fold_rows_are_the_only_access_path(labeled_panel):
    panel, folds = _setup(labeled_panel)
    empty = dataclasses.replace(folds[0], train_rows=folds[0].train_rows[:3])
    stats, _ = fd.training_feature_diagnostics(panel, [empty], FEATURES)
    assert _value(stats, 0, "feature_a", "n")["value"] == 3
    assert _value(stats, 0, "__design__", "n_complete_cases")["value"] == 3
