"""
Daily bars and split-consistent price returns (stock_agent.stock_lab.prices),
checked against hand-computed values.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.stock_lab.contracts import (
    CATEGORY_CODES,
    DEVELOPMENT_END,
    Category,
    ReturnReason,
    StableBand,
)
from stock_agent.stock_lab.prices import (
    BarsRefused,
    categorize,
    on_calendar,
    session_returns,
    split_level,
    stable_share,
    validate_bars,
)

SESSIONS = pd.bdate_range("2024-03-04", periods=6, tz="UTC").as_unit("us")


def bars(close, *, split=None, dividend=None, sessions=SESSIONS):
    n = len(close)
    return pd.DataFrame(
        {
            "session": sessions[:n],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000.0] * n,
            "div_cash": dividend or [0.0] * n,
            "split_factor": split or [1.0] * n,
        }
    )


def test_returns_are_close_to_close_by_hand():
    out = session_returns(validate_bars(bars([100.0, 101.0, 99.99, 99.99])))
    assert out["reason"].tolist() == ["first_close", "ok", "ok", "ok"]
    assert math.isnan(out["price_return"].iloc[0])
    assert out["price_return"].iloc[1] == pytest.approx(0.01)
    assert out["price_return"].iloc[2] == pytest.approx(99.99 / 101.0 - 1)
    assert out["price_return"].iloc[3] == 0.0


def test_a_split_is_applied_to_the_previous_close_so_it_never_shows_as_a_fall():
    # 2-for-1 on the third session: raw close 102 -> 51.5 is +1% split-consistently.
    frame = validate_bars(bars([100.0, 102.0, 51.5, 52.0], split=[1.0, 1.0, 2.0, 1.0]))
    out = session_returns(frame)
    assert out["price_return"].iloc[2] == pytest.approx(51.5 * 2 / 102.0 - 1)
    assert out["split_applied"].tolist() == [False, False, True, False]
    assert not out["split_unverified"].any()
    # The level puts every close on the latest share basis' multiple: ratios are returns.
    level = split_level(frame)
    assert level.tolist() == pytest.approx([100.0, 102.0, 103.0, 104.0])


def test_a_large_split_the_closes_do_not_show_is_unavailable_not_guessed():
    # The vendor reports 2-for-1 but the raw closes did not halve (already adjusted?).
    out = session_returns(validate_bars(bars([100.0, 102.0, 103.0], split=[1.0, 1.0, 2.0])))
    assert out["reason"].iloc[2] == ReturnReason.SPLIT_PRICE_MISMATCH.value
    assert math.isnan(out["price_return"].iloc[2])


def test_a_small_split_is_applied_and_flagged_unverifiable():
    out = session_returns(validate_bars(bars([100.0, 100.0, 96.0], split=[1.0, 1.0, 1.05])))
    assert out["price_return"].iloc[2] == pytest.approx(96.0 * 1.05 / 100.0 - 1)
    assert out["split_unverified"].iloc[2]


def test_dividends_are_excluded_from_the_price_return_and_flagged():
    out = session_returns(validate_bars(bars([100.0, 99.0], dividend=[0.0, 1.0])))
    assert out["price_return"].iloc[1] == pytest.approx(-0.01)
    assert out["total_return_reference"].iloc[1] == pytest.approx(0.0)
    assert out["ex_dividend"].tolist() == [False, True]


def test_a_huge_move_without_a_split_is_flagged_never_corrected():
    out = session_returns(validate_bars(bars([100.0, 49.0, 50.0])))
    assert out["large_move"].tolist() == [False, True, False]
    assert out["price_return"].iloc[1] == pytest.approx(-0.51)


def test_a_missing_session_breaks_the_return_on_both_sides():
    frame = on_calendar(bars([100.0, 101.0, 102.0], sessions=SESSIONS[[0, 1, 3]]), SESSIONS[:4])
    assert len(frame) == 4 and math.isnan(frame["close"].iloc[2])
    out = session_returns(frame)
    assert out["reason"].tolist() == [
        "first_close",
        "ok",
        ReturnReason.MISSING_CLOSE.value,
        ReturnReason.PREVIOUS_CLOSE_MISSING.value,
    ]
    assert out["price_return"].isna().tolist() == [True, False, True, True]


def test_a_bar_on_a_non_session_is_refused():
    with pytest.raises(BarsRefused, match="non-session"):
        on_calendar(bars([1.0, 2.0]), SESSIONS[2:])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"close": [1.0, -2.0]}, "close"),
        ({"close": [1.0, 0.0]}, "close"),
        ({"volume": [1.0, -1.0]}, "volume"),
        ({"div_cash": [0.0, -0.1]}, "div_cash"),
        ({"split_factor": [1.0, 0.0]}, "split_factor"),
        ({"split_factor": [1.0, np.inf]}, "split_factor"),
    ],
)
def test_invalid_values_are_refused(change, message):
    frame = bars([1.0, 2.0])
    for column, values in change.items():
        frame[column] = values
    with pytest.raises(BarsRefused, match=message):
        validate_bars(frame)


def test_sessions_must_be_unique_increasing_session_dates():
    frame = bars([1.0, 2.0])
    frame["session"] = [SESSIONS[1], SESSIONS[0]]
    with pytest.raises(BarsRefused, match="increasing"):
        validate_bars(frame)
    frame["session"] = [SESSIONS[0] + pd.Timedelta(hours=16), SESSIONS[1]]
    with pytest.raises(BarsRefused, match="00:00"):
        validate_bars(frame)
    with pytest.raises(BarsRefused, match="missing columns"):
        validate_bars(bars([1.0]).drop(columns=["split_factor"]))


def test_the_band_edges_are_stable():
    band = StableBand(0.005)
    returns = pd.Series([0.005, -0.005, 0.0050001, -0.0050001, 0.0, np.nan])
    assert categorize(returns, band).tolist() == [
        "stable",
        "stable",
        "increase",
        "decrease",
        "stable",
        None,
    ]
    assert band.categorize(0.005) is Category.STABLE
    assert band.categorize(float("nan")) is None
    assert stable_share(categorize(returns, band)) == pytest.approx(3 / 5)
    assert stable_share(pd.Series([None, None], dtype=object)) is None


def test_category_codes_order_decrease_stable_increase():
    assert CATEGORY_CODES[Category.DECREASE] < CATEGORY_CODES[Category.STABLE]
    assert CATEGORY_CODES[Category.STABLE] < CATEGORY_CODES[Category.INCREASE]


@pytest.mark.parametrize("tau", [0.0, -0.01, 0.2, True])
def test_the_band_is_bounded(tau):
    with pytest.raises((ValueError, TypeError)):
        StableBand(tau)


def test_development_end_is_the_day_before_the_owners_lockbox():
    from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

    assert pd.Timestamp(DEVELOPMENT_END, tz="UTC") + pd.Timedelta(days=1) == MODEL_HOLDOUT_START
