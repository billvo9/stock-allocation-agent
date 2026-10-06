"""
Forward-label timing: explicit entry lag, the symbol's own session calendar,
independence from other symbols, and maturity-based prefix stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from stock_agent.features.training import (
    MODEL_LABEL_SPEC,
    LabelSpec,
    add_forward_return_target,
)

SPEC = LabelSpec(horizon=5, entry_lag=1)


def _series(prices, symbol="MU", start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(prices), tz="UTC")
    return pd.DataFrame({"date": dates, "symbol": symbol, "adjusted_close": prices})


def _random_walk(seed, sessions=40, symbol="A"):
    rng = np.random.RandomState(seed)
    frame = _series(100.0 * np.exp(np.cumsum(0.01 * rng.randn(sessions))), symbol=symbol)
    return frame


def test_entry_lag_enters_next_close_and_exits_horizon_sessions_later():
    frame = _series([100.0, 101.0, 103.0, 106.0, 110.0, 115.0])

    out = add_forward_return_target(frame, horizon=2, entry_lag=1)

    assert out.loc[0, "target_return"] == pytest.approx(106.0 / 101.0 - 1.0)
    assert out.loc[0, "target_start_date"] == out.loc[1, "date"]
    assert out.loc[0, "target_end_date"] == out.loc[3, "date"]
    assert out["target_return"].iloc[3:].isna().all()
    assert out["target_end_date"].iloc[3:].isna().all()


def test_default_entry_lag_keeps_the_same_close_definition():
    out = add_forward_return_target(_series([100.0, 110.0, 121.0]), horizon=1)

    assert out.loc[0, "target_return"] == pytest.approx(0.10)
    assert out.loc[1, "target_return"] == pytest.approx(0.10)
    assert (out["target_start_date"] == out["date"]).all()


def test_model_label_spec_enters_at_the_next_close():
    assert MODEL_LABEL_SPEC == LabelSpec(horizon=20, entry_lag=1)
    with pytest.raises(TypeError):
        LabelSpec()  # timing must always be stated explicitly


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"horizon": True}, TypeError),
        ({"horizon": 1.5}, TypeError),
        ({"horizon": 0}, ValueError),
        ({"entry_lag": -1}, ValueError),
        ({"entry_lag": True}, TypeError),
        ({"entry_lag": 1.0}, TypeError),
    ],
)
def test_invalid_label_timing_raises(kwargs, error):
    with pytest.raises(error):
        add_forward_return_target(_series([1.0, 2.0, 3.0]), **kwargs)


def test_offsets_use_the_symbols_own_sessions_not_the_union_calendar():
    calendar = pd.bdate_range("2024-01-02", periods=6, tz="UTC")
    a = pd.DataFrame({"date": calendar, "symbol": "A", "adjusted_close": np.arange(1.0, 7.0)})
    b = pd.DataFrame(
        {"date": calendar[[0, 2, 4, 5]], "symbol": "B", "adjusted_close": [10.0, 20.0, 40.0, 80.0]}
    )

    out = add_forward_return_target(pd.concat([a, b]), horizon=1, entry_lag=1)

    first_b = out[(out["symbol"] == "B") & (out["date"] == calendar[0])].iloc[0]
    assert first_b["target_start_date"] == calendar[2]
    assert first_b["target_end_date"] == calendar[4]
    assert first_b["target_return"] == pytest.approx(40.0 / 20.0 - 1.0)


@pytest.mark.parametrize("seed", range(3))
def test_labels_do_not_depend_on_other_symbols_or_row_order(seed):
    a = _random_walk(seed, symbol="A")
    b = _random_walk(seed + 100, symbol="B")
    expected = SPEC.apply(a).reset_index(drop=True)

    altered_b = b.sample(frac=0.6, random_state=seed).assign(
        adjusted_close=lambda frame: frame["adjusted_close"] * 3.0
    )
    mixed = pd.concat([altered_b, a]).sample(frac=1.0, random_state=seed + 1)
    out = SPEC.apply(mixed)

    actual = out[out["symbol"] == "A"].sort_values("date").reset_index(drop=True)
    pdt.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("cutoff_index", [10, 20, 33])
def test_matured_labels_do_not_change_when_later_prices_are_added(cutoff_index):
    frame = _random_walk(7)
    cutoff = frame["date"].iloc[cutoff_index]

    full = SPEC.apply(frame)
    truncated = SPEC.apply(frame[frame["date"] < cutoff])

    matured = full[full["target_end_date"] < cutoff].reset_index(drop=True)
    known = truncated[truncated["target_end_date"].notna()].reset_index(drop=True)
    pdt.assert_frame_equal(matured, known)


def test_trailing_return_feature_equals_an_earlier_label():
    """
    Why any non-forward scheme (K-fold, CPCV) needs an embargo: the trailing
    (horizon)-session return at row r + entry_lag + horizon is exactly row r's
    label, so a training row dated after a test row can contain its label.
    """

    frame = _random_walk(3, sessions=60)
    labeled = LabelSpec(horizon=20, entry_lag=1).apply(frame)
    trailing = frame["adjusted_close"] / frame["adjusted_close"].shift(20) - 1.0

    lag = 1 + 20
    np.testing.assert_allclose(
        trailing.iloc[lag:].to_numpy(),
        labeled["target_return"].iloc[: len(frame) - lag].to_numpy(),
    )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"horizon": 0, "entry_lag": 1}, ValueError),
        ({"horizon": 20, "entry_lag": -1}, ValueError),
        ({"horizon": 20.0, "entry_lag": 1}, TypeError),
    ],
)
def test_label_spec_is_validated_when_built(kwargs, error):
    with pytest.raises(error):
        LabelSpec(**kwargs)
