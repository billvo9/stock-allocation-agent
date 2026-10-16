"""
Past-only features (stock_agent.stock_lab.features): prefix stability, values
recomputed independently, split invariance, and no use of the predicted return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.checks import assert_prefix_stable
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import StableBand
from stock_agent.stock_lab.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    NUMERIC_FEATURES,
    build_features,
    ewma_volatility,
    latest_row,
)
from stock_agent.stock_lab.runs import PAUSES, RunRule


@pytest.fixture(scope="module")
def swing() -> pd.DataFrame:
    return synthetic.demo_bars("DEMO_SWING")


@pytest.fixture(scope="module")
def swing_features(swing) -> pd.DataFrame:
    return build_features(swing)


def test_every_feature_column_is_present_and_versioned(swing_features):
    assert set(FEATURE_COLUMNS) <= set(swing_features.columns)
    assert (swing_features["feature_version"] == FEATURE_VERSION).all()
    late = swing_features.iloc[300:]
    for column in NUMERIC_FEATURES:
        assert late[column].notna().all() or column == "gap", column


@pytest.mark.parametrize("rule", [RunRule(), RunRule(PAUSES)])
def test_features_are_prefix_stable(swing, rule):
    columns = [c for c in NUMERIC_FEATURES] + ["run_direction"]

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw.rename(columns={"date": "session"}), rule=rule)
        frame = frame.rename(columns={"session": "date"})
        frame["symbol"] = "X"
        frame["run_direction"] = frame["run_direction"].fillna("none")
        return frame[["date", "symbol", *columns]]

    raw = swing.rename(columns={"session": "date"}).iloc[:1100]  # crosses the split at 900
    cutoffs = raw["date"].iloc[[100, 260, 400, 899, 900, 905, 1099]]
    assert_prefix_stable(build, raw, cutoffs=cutoffs, rtol=1e-12, atol=1e-15)


def test_the_prefix_check_catches_a_feature_that_peeks_ahead(swing):
    def leaky(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw.rename(columns={"date": "session"}))
        frame = frame.rename(columns={"session": "date"})
        frame["symbol"] = "X"
        frame["next_return"] = frame["price_return"].shift(-1)  # the predicted return
        return frame[["date", "symbol", "next_return"]]

    raw = swing.rename(columns={"session": "date"}).iloc[:300]
    with pytest.raises(LeakageError):
        assert_prefix_stable(leaky, raw, cutoffs=raw["date"].iloc[[200]])


def test_changing_the_future_never_changes_a_row(swing):
    base = build_features(swing.iloc[:800])
    changed = swing.iloc[:800].copy()
    changed.loc[500:, ["open", "high", "low", "close"]] *= 1.37
    after = build_features(changed)
    pd.testing.assert_frame_equal(
        base.iloc[:500][list(NUMERIC_FEATURES)], after.iloc[:500][list(NUMERIC_FEATURES)]
    )


def test_returns_and_windows_match_an_independent_computation(swing, swing_features):
    close = swing["close"].to_numpy()
    factor = swing["split_factor"].to_numpy()
    r = close[1:] * factor[1:] / close[:-1] - 1
    r = np.concatenate([[np.nan], r])
    level = close * np.cumprod(factor)
    d = 1000
    f = swing_features.iloc[d]
    assert f["price_return"] == pytest.approx(r[d])
    assert f["ret_5"] == pytest.approx(np.prod(1 + r[d - 4 : d + 1]) - 1)
    assert f["ret_60"] == pytest.approx(level[d] / level[d - 60] - 1)
    assert f["vol_20"] == pytest.approx(np.std(r[d - 19 : d + 1], ddof=1))
    assert f["ma_gap_20"] == pytest.approx(level[d] / level[d - 19 : d + 1].mean() - 1)
    sma = [level[i - 19 : i + 1].mean() for i in (d, d - 5)]
    assert f["ma_slope_20"] == pytest.approx(sma[0] / sma[1] - 1)
    assert f["drawdown_60"] == pytest.approx(level[d] / level[d - 59 : d + 1].max() - 1)
    assert f["bounce_20"] == pytest.approx(level[d] / level[d - 19 : d + 1].min() - 1)
    dv = close * swing["volume"].to_numpy()
    assert f["rel_dollar_volume"] == pytest.approx(dv[d] / np.median(dv[d - 20 : d]))
    assert f["gap"] == pytest.approx(swing["open"].iloc[d] * factor[d] / close[d - 1] - 1)
    ret5 = [np.prod(1 + r[i - 4 : i + 1]) - 1 for i in (d, d - 5)]
    assert f["momentum_accel"] == pytest.approx(ret5[0] - ret5[1])
    cats = np.where(r > 0.005, 1, np.where(r < -0.005, -1, 0))[d - 19 : d + 1]
    assert f["up_count_20"] == (cats == 1).sum()
    assert f["stable_count_20"] == (cats == 0).sum()
    assert f["band_over_vol"] == pytest.approx(0.005 / f["ewma_vol"])


def test_ewma_volatility_matches_its_recursion():
    rng = np.random.RandomState(1)
    r = pd.Series(rng.normal(0, 0.01, 80))
    r.iloc[30] = np.nan
    sigma = ewma_volatility(r)
    variance, seen, expected = None, 0, []
    for value in r:
        if not np.isnan(value):
            seen += 1
            variance = value**2 if variance is None else 0.94 * variance + 0.06 * value**2
        expected.append(np.sqrt(variance) if seen >= 20 else np.nan)
    np.testing.assert_allclose(sigma.to_numpy(), np.array(expected), rtol=1e-12)


def test_a_split_changes_no_feature(swing):
    # The same economic history, with and without a 3-for-1 split in the raw data.
    unsplit = swing.copy()
    after = unsplit.index >= 900
    unsplit.loc[after, ["open", "high", "low", "close"]] *= 3.0
    unsplit.loc[after, "volume"] /= 3.0
    unsplit["split_factor"] = 1.0
    split, plain = build_features(swing), build_features(unsplit)
    for column in NUMERIC_FEATURES:
        np.testing.assert_allclose(
            split[column].to_numpy(),
            plain[column].to_numpy(),
            rtol=1e-9,
            atol=1e-12,
            err_msg=column,
        )


def test_a_gap_blanks_windows_that_touch_it_rather_than_shrinking_them(swing):
    sessions = swing["session"]
    holed = swing.drop(index=[600]).reset_index(drop=True)
    frame = build_features(holed, sessions=sessions)
    assert frame["session"].equals(sessions)
    assert np.isnan(frame.loc[600, "close"])
    # Returns of 600 (no close) and 601 (no previous close) are missing, so every
    # 20-session window ending 600..620 is incomplete; 621's window starts at 602.
    assert frame.loc[600:620, "vol_20"].isna().all()
    assert frame.loc[621, "vol_20"] > 0
    assert frame.loc[600:601, "run_length"].tolist() == [0, 0]
    assert frame.loc[602, "run_length"] == 1


def test_the_band_feeds_only_band_dependent_features(swing):
    narrow = build_features(swing, band=StableBand(0.0025))
    wide = build_features(swing, band=StableBand(0.01))
    assert narrow["ret_20"].equals(wide["ret_20"])
    assert (narrow["stable_count_20"].dropna() <= wide["stable_count_20"].dropna()).all()


def test_latest_row_is_the_last_session_with_a_close(swing):
    frame = build_features(swing)
    assert latest_row(frame)["session"] == swing["session"].iloc[-1]
    assert latest_row(frame.iloc[0:0]) is None


# (feature, window of returns it needs ending at d, or of levels) -> the first row with a
# value is the window's length after the first return (row 1), and a missing close at G
# blanks every row whose window reaches it.
WINDOWS = {
    "ret_1": ("returns", 1),
    "ret_5": ("returns", 5),
    "ret_20": ("returns", 20),
    "ret_60": ("returns", 60),
    "vol_20": ("returns", 20),
    "vol_60": ("returns", 60),
    "up_count_20": ("returns", 20),
    "stable_count_20": ("returns", 20),
    "down_count_20": ("returns", 20),
    "longest_up_60": ("returns", 60),
    "longest_down_60": ("returns", 60),
    "ma_gap_20": ("levels", 20),
    "ma_gap_50": ("levels", 50),
    "drawdown_20": ("levels", 20),
    "drawdown_60": ("levels", 60),
    "drawdown_252": ("levels", 252),
    "bounce_20": ("levels", 20),
    "bounce_60": ("levels", 60),
}


@pytest.mark.parametrize("feature", sorted(WINDOWS))
def test_each_window_is_strict_from_the_start_and_around_a_gap(swing, feature):
    kind, window = WINDOWS[feature]
    frame = build_features(swing.iloc[:700])
    first = window if kind == "returns" else window - 1  # returns start at row 1
    assert frame[feature].iloc[:first].isna().all(), feature
    assert frame[feature].iloc[first:].notna().all(), feature
    gap = 500
    holed = build_features(swing.iloc[:700].drop(index=[gap]), sessions=swing["session"].iloc[:700])
    # Returns of the gap and the session after it are missing; levels only at the gap.
    blank_from = gap
    blank_to = gap + window if kind == "returns" else gap + window - 1
    assert holed[feature].iloc[blank_from : blank_to + 1].isna().all(), feature
    assert holed[feature].iloc[blank_to + 1 :].notna().all(), feature


def test_a_split_the_closes_do_not_show_breaks_every_level_window(swing):
    wrong = swing.iloc[:400].copy()
    wrong.loc[200, "split_factor"] = 2.0  # the vendor says 2-for-1; the closes did not halve
    frame = build_features(wrong)
    assert frame.loc[200, "reason"] == "split_price_mismatch" and np.isnan(frame.loc[200, "level"])
    for feature, (kind, window) in WINDOWS.items():
        span = frame[feature].iloc[200 : 200 + window - (kind == "levels")]
        assert span.isna().all(), feature
    # Windows wholly after the break are back, on one consistent basis.
    assert frame.loc[399, "ma_gap_20"] == pytest.approx(
        build_features(swing.iloc[:400]).loc[399, "ma_gap_20"]
    )


def test_relative_dollar_volume_compares_with_the_sessions_before_only(swing):
    spiked = swing.iloc[:300].copy()
    spiked.loc[250, "volume"] *= 10
    frame = build_features(spiked)
    dv = (spiked["close"] * spiked["volume"]).to_numpy()
    assert frame.loc[250, "rel_dollar_volume"] == pytest.approx(dv[250] / np.median(dv[230:250]))
    assert frame.loc[250, "rel_dollar_volume"] > 5
    # The spike enters the next sessions' baselines only as one of 20 values.
    assert frame.loc[251, "rel_dollar_volume"] == pytest.approx(dv[251] / np.median(dv[231:251]))


def test_volatility_uses_the_sample_standard_deviation(swing, swing_features):
    r = swing_features["price_return"].to_numpy()
    assert swing_features["vol_60"].iloc[800] == pytest.approx(np.std(r[741:801], ddof=1))
