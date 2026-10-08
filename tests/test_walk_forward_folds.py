"""
Purged expanding walk-forward folds: membership against a row-by-row oracle,
purge boundaries, global (cross-symbol) purging, the development lockbox,
date-level boundary enforcement, nested folds, and the independent fold
invariant checker.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_validation.checks import LeakageError, assert_fold_is_leakage_safe
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    make_expanding_folds,
    make_test_windows,
    restrict_to_symbols,
)


def _ts(value):
    return pd.Timestamp(value, tz="UTC")


def _rows(spec):
    """
    Hand-built labeled rows: (date, symbol, target_end_date or None, return).
    Entry is the next business day after the row date (entry lag 1).
    """

    frame = pd.DataFrame(
        [
            {
                "date": _ts(date),
                "symbol": symbol,
                "target_return": value,
                "target_start_date": _ts(date) + pd.offsets.BDay(1) if end else pd.NaT,
                "target_end_date": _ts(end) if end else pd.NaT,
                "feature_a": 0.0,
            }
            for date, symbol, end, value in spec
        ]
    )
    for column in ("target_start_date", "target_end_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _oracle(frame, start, end, holdout):
    """Plain per-row restatement of the fold rules."""

    groups = {"train": set(), "purged": set(), "test": set(), "held_out": set()}
    for position, row in enumerate(frame.itertuples(index=False)):
        labeled = np.isfinite(row.target_return) and pd.notna(row.target_end_date)
        if not labeled:
            continue
        if row.target_end_date < start:
            groups["train"].add(position)
        elif row.date < start:
            groups["purged"].add(position)
        elif row.date < end:
            groups["held_out" if row.target_end_date >= holdout else "test"].add(position)
    return groups


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("horizon, entry_lag", [(1, 1), (5, 1), (20, 1), (3, 2)])
def test_fold_membership_matches_row_by_row_oracle(labeled_panel, seed, horizon, entry_lag):
    panel = labeled_panel(
        symbols=("A", "B", "C", "D"),
        sessions=160,
        horizon=horizon,
        entry_lag=entry_lag,
        seed=seed,
        drop_fraction=0.1,
        listing={"C": (30, 160), "D": (0, 110)},
    )
    panel = panel.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sessions = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    holdout = sessions.iloc[150]
    windows = make_test_windows(sessions, start=sessions.iloc[60], end=holdout, block_sessions=30)

    folds = make_expanding_folds(panel, windows, holdout_start=holdout)

    assert len(folds) == 3
    for fold in folds:
        expected = _oracle(panel, fold.test_start, fold.test_end, holdout)
        assert set(fold.train_rows) == expected["train"]
        assert set(fold.purged_rows) == expected["purged"]
        assert set(fold.test_rows) == expected["test"]
        assert set(fold.held_out_rows) == expected["held_out"]
        assert_fold_is_leakage_safe(panel, fold)


def _single_window_fold(frame, start="2024-03-04", end="2024-03-11", holdout="2024-06-03"):
    (fold,) = make_expanding_folds(frame, [(start, end)], holdout_start=holdout)
    return fold


BOUNDARY_ROWS = [
    ("2024-02-26", "A", "2024-03-01", 0.01),  # 0: label ends the session before test_start
    ("2024-02-27", "A", "2024-03-04", 0.02),  # 1: label ends exactly on test_start
    ("2024-02-28", "A", "2024-03-06", 0.03),  # 2: label ends inside the window
    ("2024-03-04", "A", "2024-03-12", 0.04),  # 3: row dated test_start
    ("2024-02-28", "B", "2024-03-05", 0.05),  # 4: B has no test rows at all
    ("2024-02-29", "C", None, float("nan")),  # 5: label not yet matured
    ("2024-03-08", "C", "2024-03-15", 0.06),  # 6: C listed inside the window
]


def test_label_ending_before_test_start_is_trained():
    fold = _single_window_fold(_rows(BOUNDARY_ROWS))
    assert list(fold.train_rows) == [0]


def test_label_ending_on_or_after_test_start_is_purged_for_every_symbol():
    fold = _single_window_fold(_rows(BOUNDARY_ROWS))
    # Row 1 ends exactly on test_start: the label is known only after that
    # close, so it is not available to the decision. Row 4 is purged although
    # B has no test rows: purging is global across symbols.
    assert sorted(fold.purged_rows) == [1, 2, 4]


def test_row_dated_test_start_is_scored_not_trained():
    fold = _single_window_fold(_rows(BOUNDARY_ROWS))
    assert sorted(fold.test_rows) == [3, 6]


def test_unmatured_labels_are_never_trained_or_scored():
    fold = _single_window_fold(_rows(BOUNDARY_ROWS))
    assert list(fold.unlabeled_rows) == [5]


@pytest.mark.parametrize(
    "value, end",
    [(float("nan"), "2024-02-29"), (float("inf"), "2024-02-29"), (0.01, None)],
)
def test_rows_without_a_finite_matured_label_are_excluded(value, end):
    frame = _rows([("2024-02-26", "A", "2024-03-01", 0.01), ("2024-02-27", "A", end, value)])
    frame = pd.concat([frame, _rows([("2024-03-05", "A", "2024-03-12", 0.02)])])
    fold = _single_window_fold(frame.reset_index(drop=True))
    assert list(fold.train_rows) == [0]
    assert list(fold.unlabeled_rows) == [1]


def test_purge_uses_the_recorded_label_end_across_a_trading_gap():
    # Three sessions after 2024-02-26 is 2024-02-29, before test_start, but
    # the symbol was halted, so its recorded exit is 2024-03-05: purge it.
    frame = _rows(
        [
            ("2024-02-26", "A", "2024-03-05", 0.01),
            ("2024-02-20", "A", "2024-02-23", 0.02),
            ("2024-03-04", "B", "2024-03-07", 0.03),
        ]
    )
    fold = _single_window_fold(frame)
    assert list(fold.purged_rows) == [0]
    assert list(fold.train_rows) == [1]


def test_development_folds_never_score_labels_that_mature_in_the_holdout():
    frame = _rows(
        [
            ("2024-05-13", "A", "2024-05-17", 0.01),
            ("2024-05-20", "A", "2024-05-24", 0.02),  # matures before the holdout
            ("2024-05-28", "A", "2024-06-03", 0.03),  # matures on the holdout start
            ("2024-05-31", "B", "2024-06-07", 0.04),  # matures inside the holdout
        ]
    )
    (fold,) = make_expanding_folds(
        frame, [("2024-05-20", "2024-06-03")], holdout_start="2024-06-03"
    )
    assert list(fold.test_rows) == [1]
    assert sorted(fold.held_out_rows) == [2, 3]


def test_development_windows_cannot_reach_into_the_holdout():
    frame = _rows(BOUNDARY_ROWS)
    with pytest.raises(ValueError, match="holdout"):
        make_expanding_folds(frame, [("2024-03-04", "2024-03-11")], holdout_start="2024-03-08")


def test_holdout_folds_must_start_inside_the_holdout():
    frame = _rows(BOUNDARY_ROWS)
    with pytest.raises(ValueError, match="before"):
        make_expanding_folds(
            frame, [("2024-03-04", "2024-03-11")], holdout_start="2024-03-05", mode="holdout"
        )
    (fold,) = make_expanding_folds(
        frame, [("2024-03-04", "2024-03-11")], holdout_start="2024-03-04", mode="holdout"
    )
    assert fold.mode == "holdout"
    assert list(fold.held_out_rows) == []


@pytest.mark.parametrize(
    "windows, holdout",
    [
        ([("2024-03-04 09:30", "2024-03-11")], "2024-06-03"),
        ([("2024-03-04", "2024-03-11 16:00")], "2024-06-03"),
        ([("2024-03-04", "2024-03-11")], "2024-06-03 13:00"),
        ([("2024-03-04 00:00-05:00", "2024-03-11")], "2024-06-03"),
    ],
)
def test_intraday_boundaries_are_rejected(windows, holdout):
    with pytest.raises(ValueError, match="00:00 UTC"):
        make_expanding_folds(_rows(BOUNDARY_ROWS), windows, holdout_start=holdout)


def test_frames_must_hold_utc_session_dates():
    naive = _rows(BOUNDARY_ROWS)
    naive["date"] = naive["date"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        _single_window_fold(naive)

    intraday = _rows(BOUNDARY_ROWS)
    intraday.loc[0, "date"] = _ts("2024-02-26 21:00")
    with pytest.raises(ValueError, match="00:00 UTC"):
        _single_window_fold(intraday)

    # A label end stored as Tokyo midnight is the previous day in UTC and
    # would otherwise be trained a day early.
    tokyo = _rows(BOUNDARY_ROWS)
    tokyo["target_end_date"] = (
        tokyo["target_end_date"].dt.tz_convert(None).dt.tz_localize("Asia/Tokyo")
    )
    with pytest.raises(ValueError, match="UTC"):
        _single_window_fold(tokyo)


@pytest.mark.parametrize(
    "windows",
    [
        [("2024-03-11", "2024-03-18"), ("2024-03-04", "2024-03-11")],
        [("2024-03-04", "2024-03-12"), ("2024-03-11", "2024-03-18")],
        [("2024-03-11", "2024-03-04")],
        [],
    ],
)
def test_windows_must_be_sorted_disjoint_and_non_empty(windows):
    with pytest.raises(ValueError):
        make_expanding_folds(_rows(BOUNDARY_ROWS), windows, holdout_start="2024-06-03")


def test_same_close_labels_are_rejected():
    # Owner decision: labels enter after the feature date (entry lag >= 1).
    same_close = _rows(BOUNDARY_ROWS)
    same_close.loc[0, "target_start_date"] = same_close.loc[0, "date"]
    with pytest.raises(ValueError, match="enter after its feature date"):
        _single_window_fold(same_close)


def test_invalid_frames_are_rejected():
    exit_before_entry = _rows([("2024-02-26", "A", "2024-02-27", 0.01)])
    with pytest.raises(ValueError, match="after target_start_date"):
        _single_window_fold(exit_before_entry)

    missing_entry = _rows(BOUNDARY_ROWS).drop(columns="target_start_date")
    with pytest.raises(ValueError, match="missing required columns"):
        _single_window_fold(missing_entry)

    duplicated = _rows(BOUNDARY_ROWS[:1] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        _single_window_fold(duplicated)

    shifted_index = _rows(BOUNDARY_ROWS)
    shifted_index.index = shifted_index.index + 1
    with pytest.raises(ValueError, match="RangeIndex"):
        _single_window_fold(shifted_index)


def test_fold_without_training_rows_is_rejected():
    frame = _rows([("2024-03-04", "A", "2024-03-12", 0.04)])
    with pytest.raises(ValueError, match="no training rows"):
        _single_window_fold(frame)


def test_training_sets_expand_and_each_row_is_scored_at_most_once(labeled_panel):
    panel = labeled_panel(sessions=200, seed=3)
    sessions = panel["date"].drop_duplicates().sort_values()
    windows = make_test_windows(
        sessions, start=sessions.iloc[50], end=sessions.iloc[190], block_sessions=20
    )
    folds = make_expanding_folds(panel, windows, holdout_start=sessions.iloc[190])

    for earlier, later in pairwise(folds):
        assert set(earlier.train_rows) <= set(later.train_rows)
    scored = np.concatenate([fold.test_rows for fold in folds])
    assert len(scored) == len(set(scored))


def test_train_start_drops_older_rows(labeled_panel):
    panel = labeled_panel(sessions=120)
    sessions = panel["date"].drop_duplicates().sort_values()
    floor = sessions.iloc[30]
    (fold,) = make_expanding_folds(
        panel,
        [(sessions.iloc[80], sessions.iloc[100])],
        holdout_start=sessions.iloc[110],
        train_start=floor,
    )
    assert (panel.loc[fold.train_rows, "date"] >= floor).all()
    assert_fold_is_leakage_safe(panel, fold)


def test_test_windows_are_half_open_blocks_of_sessions():
    sessions = pd.bdate_range("2024-01-01", periods=30, tz="UTC")
    windows = make_test_windows(sessions, start="2024-01-06", end=sessions[23], block_sessions=5)

    assert windows[0][0] == _ts("2024-01-08")  # Saturday start -> next session
    for (_, end), (next_start, _) in pairwise(windows):
        assert end == next_start
    assert windows[-1][1] == sessions[23]
    covered = [s for s in sessions if windows[0][0] <= s < windows[-1][1]]
    assert len(covered) == 18


def test_nested_inner_folds_stay_inside_the_outer_training_set(labeled_panel):
    panel = labeled_panel(sessions=240, seed=5)
    sessions = panel["date"].drop_duplicates().sort_values()
    (outer,) = make_expanding_folds(
        panel, [(sessions.iloc[200], sessions.iloc[220])], holdout_start=sessions.iloc[230]
    )

    # Everything dated before the outer window, including rows the outer fold
    # purged: only holdout_start = outer test_start keeps inner folds clean.
    inner_frame = panel[panel["date"] < outer.test_start].reset_index(drop=True)
    inner_sessions = inner_frame["date"].drop_duplicates().sort_values()
    inner_windows = make_test_windows(
        inner_sessions,
        start=inner_sessions.iloc[100],
        end=outer.test_start,
        block_sessions=20,
    )
    inner_folds = make_expanding_folds(inner_frame, inner_windows, holdout_start=outer.test_start)

    outer_keys = set(
        zip(panel.loc[outer.train_rows, "date"], panel.loc[outer.train_rows, "symbol"])
    )
    for fold in inner_folds:
        scored = inner_frame.iloc[fold.test_rows]
        trained = inner_frame.iloc[fold.train_rows]
        assert (scored["target_end_date"] < outer.test_start).all()
        assert set(zip(scored["date"], scored["symbol"])) <= outer_keys
        assert set(zip(trained["date"], trained["symbol"])) <= outer_keys
    assert len(inner_folds[-1].held_out_rows) > 0  # outer-purged rows are withheld


def test_restrict_to_symbols_drops_benchmarks_and_requires_every_symbol():
    frame = pd.DataFrame({"symbol": ["A", "B", "SP500", "A"], "value": [1, 2, 3, 4]})

    kept = restrict_to_symbols(frame, ["A", "B"])
    assert list(kept["symbol"]) == ["A", "B", "A"]
    assert kept.index.equals(pd.RangeIndex(3))

    with pytest.raises(ValueError, match="absent"):
        restrict_to_symbols(frame, ["A", "NVDA"])
    with pytest.raises(ValueError):
        restrict_to_symbols(frame, ["A", "A"])


def test_fold_positions_are_read_only():
    fold = _single_window_fold(_rows(BOUNDARY_ROWS))
    with pytest.raises(ValueError):
        fold.train_rows[0] = 3


# --- the independent invariant checker catches tampered folds ---


@pytest.fixture
def valid_fold():
    frame = _rows(BOUNDARY_ROWS)
    return frame, _single_window_fold(frame)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda f: {"train_rows": np.append(f.train_rows, 1)}, "ending at or after"),
        (lambda f: {"train_rows": np.array([], dtype=np.int64)}, "full purged set"),
        (lambda f: {"test_rows": np.append(f.test_rows, 0)}, "outside its test window"),
        (lambda f: {"test_rows": np.append(f.test_rows, 5)}, "without a matured label"),
        (lambda f: {"test_end": _ts("2024-06-10")}, "into the holdout"),
        (lambda f: {"mode": "holdout"}, "starts before"),
        (lambda f: {"test_rows": f.test_rows[:1]}, "not every scorable row"),
        (lambda f: {"test_rows": np.append(f.test_rows, f.test_rows[0])}, "duplicate"),
        (lambda f: {"train_rows": np.append(f.train_rows, 99)}, "out of bounds"),
        (lambda f: {"holdout_start": _ts("2025-06-02")}, "after the lockbox start"),
    ],
)
def test_invariant_checker_rejects_tampered_folds(valid_fold, tamper, message):
    frame, fold = valid_fold
    tampered = dataclasses.replace(fold, **tamper(fold))
    with pytest.raises(LeakageError, match=message):
        assert_fold_is_leakage_safe(frame, tampered)


def test_invariant_checker_rejects_a_fold_from_another_frame(valid_fold):
    frame, fold = valid_fold
    other = frame.copy()
    other.loc[0, "symbol"] = "Z"
    with pytest.raises(LeakageError, match="different frame"):
        assert_fold_is_leakage_safe(other, fold)


def test_invariant_checker_rejects_scored_labels_that_mature_in_the_holdout():
    frame = _rows(
        [
            ("2024-05-13", "A", "2024-05-17", 0.01),
            ("2024-05-20", "A", "2024-05-24", 0.02),
            ("2024-05-28", "A", "2024-06-03", 0.03),
        ]
    )
    (fold,) = make_expanding_folds(
        frame, [("2024-05-20", "2024-06-03")], holdout_start="2024-06-03"
    )
    tampered = dataclasses.replace(fold, test_rows=np.array([1, 2]))
    with pytest.raises(LeakageError, match="mature in the holdout"):
        assert_fold_is_leakage_safe(frame, tampered)


def test_invariant_checker_rejects_same_close_labels(valid_fold):
    frame, fold = valid_fold
    relabeled = frame.copy()
    relabeled.loc[0, "target_start_date"] = relabeled.loc[0, "date"]
    with pytest.raises(LeakageError, match="own feature date"):
        assert_fold_is_leakage_safe(relabeled, fold)


def test_lockbox_is_the_owner_decision_and_bounds_development_folds():
    assert _ts("2025-01-01") == MODEL_HOLDOUT_START
    frame = _rows(BOUNDARY_ROWS)
    with pytest.raises(ValueError, match="lockbox"):
        make_expanding_folds(frame, [("2024-03-04", "2024-03-11")], holdout_start="2025-03-03")
    # Earlier holdouts (e.g. inner folds of nested selection) are allowed.
    make_expanding_folds(frame, [("2024-03-04", "2024-03-11")], holdout_start="2024-04-01")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"block_sessions": True}, TypeError),
        ({"block_sessions": 0}, ValueError),
        ({"start": "2024-02-01", "end": "2024-01-01"}, ValueError),
        ({"start": "2030-01-01", "end": "2030-02-01"}, ValueError),
    ],
)
def test_make_test_windows_rejects_invalid_requests(kwargs, error):
    sessions = pd.bdate_range("2024-01-01", periods=30, tz="UTC")
    arguments = {"start": "2024-01-01", "end": "2024-02-01", "block_sessions": 5, **kwargs}
    with pytest.raises(error):
        make_test_windows(sessions, **arguments)


def test_restrict_to_symbols_rejects_an_empty_universe():
    with pytest.raises(ValueError):
        restrict_to_symbols(pd.DataFrame({"symbol": ["A"]}), [])
