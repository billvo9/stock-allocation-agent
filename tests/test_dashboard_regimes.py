"""
Regime labels: point-in-time regimes use only information known at each
date (prefix-stable, thresholds from earlier dates only); retrospective
episodes are labelled ex post and kept apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.dashboard import regimes


def _inputs(days=400, seed=0, symbols="ABCD"):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2019-01-01", periods=days, tz="UTC")
    rows = [
        {
            "date": date,
            "symbol": symbol,
            "volatility_20d": 0.2 + 0.05 * np.sin(index / 40) + 0.02 * rng.randn(),
            "momentum_20d": 0.05 * np.sin(index / 25) + 0.01 * rng.randn(),
        }
        for index, date in enumerate(dates)
        for symbol in symbols
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("builder", [regimes.volatility_regime, regimes.trend_regime])
def test_point_in_time_regimes_never_change_when_later_data_arrive(builder):
    inputs = _inputs()
    full = builder(inputs)
    for cut in (70, 150, 260, 399):
        cutoff = full.index[cut]
        partial = builder(inputs[inputs["date"] <= cutoff])
        pd.testing.assert_series_equal(partial, full[full.index <= cutoff])


def test_later_values_cannot_move_an_earlier_regime():
    inputs = _inputs()
    before = regimes.volatility_regime(inputs)
    shocked = inputs.copy()
    late = shocked["date"] > before.index[200]
    shocked.loc[late, "volatility_20d"] *= 10
    after = regimes.volatility_regime(shocked)
    pd.testing.assert_series_equal(after[: before.index[200]], before[: before.index[200]])


def test_the_threshold_uses_strictly_earlier_dates():
    dates = pd.bdate_range("2020-01-01", periods=80, tz="UTC")
    level = np.full(80, 0.2)
    level[70] = 0.25  # a spike is high against earlier dates only
    inputs = pd.DataFrame({"date": dates, "symbol": "A", "volatility_20d": level})
    labels = regimes.volatility_regime(inputs, min_history=63)
    assert (labels.iloc[:63] == regimes.INSUFFICIENT_HISTORY).all()
    assert labels.iloc[70] == "high"
    assert labels.iloc[69] == "normal" and labels.iloc[71] == "normal"


def test_a_date_never_counts_in_its_own_threshold():
    # 64 earlier dates alternate 0.1 / 0.3 (median 0.2); date t is 0.25. Against
    # earlier dates t is high; if t joined its own window the median would be
    # 0.25 and t would read normal.
    dates = pd.bdate_range("2020-01-01", periods=65, tz="UTC")
    level = np.array([0.1, 0.3] * 32 + [0.25])
    inputs = pd.DataFrame({"date": dates, "symbol": "A", "volatility_20d": level})
    assert regimes.volatility_regime(inputs, min_history=64).iloc[-1] == "high"


def test_trend_is_the_sign_of_the_mean_momentum():
    dates = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
    inputs = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "symbol": ["A", "B"] * 3,
            "momentum_20d": [0.1, -0.05, -0.1, 0.05, 0.1, -0.1],
        }
    )
    assert regimes.trend_regime(inputs).tolist() == ["up", "down", "flat"]


def test_retrospective_episodes_are_labelled_ex_post_and_kept_apart():
    dates = pd.Series(
        pd.to_datetime(
            ["2020-02-18", "2020-02-19", "2020-03-23", "2022-06-01", "2023-01-03"], utc=True
        )
    )
    labels = regimes.ex_post_episode(dates)
    assert labels.iloc[0] == regimes.OUTSIDE_EPISODES
    assert labels.iloc[1].startswith("COVID-19 crash, ex post")
    assert labels.iloc[2].startswith("COVID-19 crash, ex post")
    assert labels.iloc[3].startswith("2022 bear market, ex post")
    assert labels.iloc[4] == regimes.OUTSIDE_EPISODES
    assert all("ex post" in label for label, _, _ in regimes.EX_POST_EPISODES)
    kinds = {definition.key: definition.point_in_time for definition in regimes.DEFINITIONS}
    assert kinds == {
        "universe_volatility_pit": True,
        "universe_trend_pit": True,
        "drawdown_episode_ex_post": False,
    }
    assert "retrospective" in regimes.DRAWDOWN_EPISODES.label


def test_the_regime_table_reports_why_a_regime_is_unavailable():
    inputs = _inputs()
    table, reasons = regimes.regime_table(inputs, inputs["date"])
    assert reasons == {}
    assert list(table.columns) == ["date", *(d.key for d in regimes.DEFINITIONS)]
    bare = inputs.drop(columns=["volatility_20d"])
    table, reasons = regimes.regime_table(bare, bare["date"])
    assert "volatility_20d" in reasons["universe_volatility_pit"]
    assert set(table["universe_volatility_pit"]) == {regimes.UNAVAILABLE}
    assert regimes.UNAVAILABLE not in set(table["universe_trend_pit"])
