"""
The point-in-time equal-weight excess label (T4 primary target): values,
availability of a cross-sectional label, fail-closed incomplete dates, the
benchmark universe, and invariance to truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import (
    EXCESS_AVAILABILITY_RULE,
    MODEL_EXCESS_LABEL_SPEC,
    MODEL_LABEL_SPEC,
    RAW_TARGET_RETURN_COLUMN,
    LabelSpec,
)
from stock_agent.model_validation.checks import validate_feature_columns

SPEC = LabelSpec(horizon=2, entry_lag=1, benchmark="equal_weight")


def _prices(rows):
    """rows: symbol -> list of prices on consecutive sessions (None = no row)."""

    dates = pd.bdate_range("2024-01-01", periods=max(len(p) for p in rows.values()), tz="UTC")
    records = [
        {"date": dates[index], "symbol": symbol, "adjusted_close": price, "feature_a": 0.0}
        for symbol, prices in rows.items()
        for index, price in enumerate(prices)
        if price is not None
    ]
    return pd.DataFrame(records)


def _by(frame, symbol, day):
    rows = frame[(frame["symbol"] == symbol) & (frame["date"].dt.day == day)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_the_excess_is_the_raw_return_minus_the_date_mean():
    # Entry at t+1, exit at t+3. On 2024-01-01: A 100->110 (+10%), B 50->45 (-10%),
    # C 20->23 (+15%); mean +5%, so the excess is +5%, -15%, +10%.
    frame = _prices(
        {
            "A": [100, 100, 105, 110, 111],
            "B": [50, 50, 48, 45, 44],
            "C": [20, 20, 21, 23, 24],
        }
    )
    labeled = SPEC.apply(frame, universe=["A", "B", "C"])
    first = labeled[labeled["date"].dt.day == 1].set_index("symbol")
    np.testing.assert_allclose(first[RAW_TARGET_RETURN_COLUMN], [0.10, -0.10, 0.15])
    np.testing.assert_allclose(first["target_return"], [0.05, -0.15, 0.10])
    sums = labeled.dropna(subset=["target_return"]).groupby("date")["target_return"].sum()
    np.testing.assert_allclose(sums, 0.0, atol=1e-15)


def test_every_row_on_a_date_is_known_only_when_its_whole_cross_section_is():
    # B has no row on 2024-01-03, so its label on 2024-01-01 exits a session later
    # than A's. The excess on 2024-01-01 is known only at B's exit.
    frame = _prices({"A": [100, 101, 102, 103, 104, 105], "B": [50, 51, None, 52, 53, 54]})
    raw = LabelSpec(horizon=2, entry_lag=1).apply(frame)
    labeled = SPEC.apply(frame, universe=["A", "B"])
    a_raw, b_raw = _by(raw, "A", 1), _by(raw, "B", 1)
    assert b_raw["target_end_date"] > a_raw["target_end_date"]
    for symbol in ("A", "B"):
        row = _by(labeled, symbol, 1)
        assert row["target_end_date"] == b_raw["target_end_date"]  # latest member end
        assert row["target_start_date"] == a_raw["target_start_date"]  # earliest start
    assert EXCESS_AVAILABILITY_RULE.startswith("An equal-weight excess label")


def test_an_incomplete_cross_section_leaves_the_whole_date_unlabeled():
    # On the last labelable date A has an exit price and B does not: no row of that
    # date may carry a label, rather than a benchmark of A alone.
    frame = _prices({"A": [100, 101, 102, 103, 104], "B": [50, 51, 52, 53]})
    labeled = SPEC.apply(frame, universe=["A", "B"])
    day2 = labeled[labeled["date"].dt.day == 2]
    assert len(day2) == 2 and day2["target_return"].isna().all()
    assert day2["target_end_date"].isna().all() and day2[RAW_TARGET_RETURN_COLUMN].isna().all()
    assert labeled[labeled["date"].dt.day == 1]["target_return"].notna().all()


def test_a_date_with_too_few_names_is_unlabeled():
    frame = _prices({"A": [100, 101, 102, 103, 104], "B": [None, None, 52, 53, 54]})
    labeled = SPEC.apply(frame, universe=["A", "B"])
    assert labeled[labeled["date"].dt.day == 1]["target_return"].isna().all()  # A alone
    three = LabelSpec(horizon=2, entry_lag=1, benchmark="equal_weight", min_cross_section=3)
    labeled = three.apply(_prices({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}), universe=["A", "B"])
    assert labeled["target_return"].isna().all()


def test_the_benchmark_universe_is_enforced():
    frame = _prices({"A": [100, 101, 102, 103], "SP500": [10, 11, 12, 13]})
    with pytest.raises(ValueError, match="outside the benchmark universe"):
        SPEC.apply(frame, universe=["A"])
    with pytest.raises(ValueError, match="needs the universe"):
        SPEC.apply(frame)


def test_truncation_never_changes_a_matured_excess_label():
    rng = np.random.RandomState(3)
    prices = {s: list(100 * np.exp(np.cumsum(0.01 * rng.randn(40)))) for s in "ABCD"}
    prices["C"][17] = None  # an interior gap shifts C's exits near it
    frame = _prices(prices)
    full = SPEC.apply(frame, universe=list("ABCD"))
    cutoff = frame["date"].drop_duplicates().sort_values().iloc[25]
    truncated = SPEC.apply(frame[frame["date"] < cutoff], universe=list("ABCD"))
    matured = truncated.dropna(subset=["target_return"])
    assert len(matured) > 40
    assert (matured["target_end_date"] < cutoff).all()
    merged = matured.merge(full, on=["date", "symbol"], suffixes=("", "_full"))
    for column in ("target_return", RAW_TARGET_RETURN_COLUMN, "target_end_date"):
        assert (merged[column] == merged[f"{column}_full"]).all(), column


def test_the_raw_return_is_stored_but_never_a_feature():
    frame = SPEC.apply(_prices({"A": [1, 2, 3, 4], "B": [2, 3, 4, 5]}), universe=["A", "B"])
    assert RAW_TARGET_RETURN_COLUMN in frame.columns
    with pytest.raises(ValueError):
        validate_feature_columns(frame, ["feature_a", RAW_TARGET_RETURN_COLUMN])


def test_label_identity_and_validation():
    assert (
        MODEL_LABEL_SPEC.label_id == "forward_simple_return:adjusted_close:horizon=20:entry_lag=1"
    )
    assert MODEL_EXCESS_LABEL_SPEC.label_id == (
        "equal_weight_excess_forward_simple_return:adjusted_close:horizon=20:entry_lag=1"
        ":min_cross_section=2"
    )
    with pytest.raises(ValueError, match="benchmark"):
        LabelSpec(horizon=2, entry_lag=1, benchmark="cap_weight")
    with pytest.raises(ValueError):
        LabelSpec(horizon=2, entry_lag=1, benchmark="equal_weight", min_cross_section=1)
