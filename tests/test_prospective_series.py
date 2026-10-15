"""
The prospective track's registered config and per-asset series: F5.1 daily
log returns, the ETF track's features on them, and the two targets.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from forecast_prospective_helpers import REGISTERED, registered_raw

from stock_agent.calendars import nyse
from stock_agent.forecasting import etf_data
from stock_agent.forecasting.config import ConfigError
from stock_agent.forecasting.prospective import bands, series, synthetic
from stock_agent.forecasting.prospective.config import SKILL_LABEL, load_config, parse_config
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

# --- the registered config ---


PROSPECTIVE_CONFIGS = REGISTERED.parent


@pytest.mark.parametrize("path", sorted(PROSPECTIVE_CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_prospective_config_parses(path):
    config = load_config(path)
    assert config.training_end < MODEL_HOLDOUT_START
    assert config.sha256 == load_config(path).sha256


def test_the_registered_config_pre_registers_the_owners_choices():
    config = parse_config(registered_raw())
    assert config.assets == ("VTI", "AGG", "SHV")
    assert config.horizons == (5, 20, 30)
    assert config.targets == ("total_log_return", "log_realized_volatility")
    assert config.training_end == pd.Timestamp("2024-12-31", tz="UTC")
    assert config.h_end == pd.Timestamp("2026-10-14", tz="UTC")
    assert config.prospective.lookback_sessions == 63
    assert config.band.method == bands.METHOD and config.band.level == 0.8
    assert (config.band.lower_quantile, config.band.upper_quantile) == (0.1, 0.9)
    assert SKILL_LABEL in config.labels
    assert config.sha256 == parse_config(copy.deepcopy(registered_raw())).sha256


def _changed(path: str, value):
    raw = copy.deepcopy(registered_raw())
    target = raw
    *parents, last = path.split("__")
    for key in parents:
        target = target[key]
    target[last] = value
    return raw


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("data__training_end", "2025-01-02", "before the 2025-01-01 lockbox"),
        ("models__ridge_grid", [0.1, 1.0], "pre-registered ridge grid"),
        ("band__level", 0.9, "upper - lower = level"),
        ("labels", ["no label"], "unvalidated, no demonstrated skill"),
        ("horizons", [20, 5], "distinct and increasing"),
        ("prospective__h_end", "2024-12-31", "on or after the lockbox"),
        ("entry_lag", 0, "entry_lag"),
        ("features", ["momentum"], "subset"),
    ],
)
def test_the_config_refuses_unregistered_choices(path, value, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(_changed(path, value))


def test_the_config_refuses_unknown_and_missing_keys():
    raw = copy.deepcopy(registered_raw())
    raw["band"]["width"] = 2
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config(raw)
    raw = copy.deepcopy(registered_raw())
    del raw["evaluation"]
    with pytest.raises(ConfigError, match="missing"):
        parse_config(raw)


def test_any_change_is_a_new_config_hash():
    base = parse_config(registered_raw()).sha256
    assert parse_config(_changed("band__min_calibration", 101)).sha256 != base


# --- series ---


def _prices(closes, sessions, fund="X"):
    return pd.DataFrame(
        {
            "fund_id": fund,
            "session_date": sessions,
            "close": closes,
            "adjustment_basis": "unadjusted",
        }
    )


EMPTY_CASH = pd.DataFrame(columns=["fund_id", "ex_date", "amount", "amount_basis"])
EMPTY_SPLITS = pd.DataFrame(columns=["fund_id", "effective_date", "ratio_new_per_old"])


def _no_distribution_series(n=260, seed=1):
    sessions = nyse.sessions("2022-01-03", "2024-12-31")[:n]
    rng = np.random.RandomState(seed)
    closes = 100.0 * np.exp(np.cumsum(0.01 * rng.randn(n)))
    return sessions, closes


def test_without_cash_the_features_equal_the_etf_tracks_price_features():
    sessions, closes = _no_distribution_series()
    returns = series.daily_log_returns(
        sessions, _prices(closes, sessions), EMPTY_CASH, EMPTY_SPLITS, "X"
    )
    mine = series.asset_frame(returns, "X")
    theirs = etf_data.etf_features(
        pd.DataFrame({"date": sessions.as_unit("ns"), "symbol": "X", "close": closes})
    )
    for name in series.FEATURES:
        np.testing.assert_allclose(mine[name], theirs[name], rtol=1e-12, equal_nan=True)


def test_without_cash_the_targets_equal_the_etf_tracks_labels():
    sessions, closes = _no_distribution_series()
    returns = series.daily_log_returns(
        sessions, _prices(closes, sessions), EMPTY_CASH, EMPTY_SPLITS, "X"
    )
    prices = pd.DataFrame({"date": sessions.as_unit("ns"), "symbol": "X", "close": closes})
    for horizon in (5, 20):
        total = series.labels_from_log_returns(
            returns["date"],
            returns["log_return"],
            horizon=horizon,
            entry_lag=1,
            target="total_log_return",
        )
        simple = etf_data.etf_labels(prices, horizon=horizon, entry_lag=1, target="forward_return")
        np.testing.assert_allclose(
            total["target_return"], np.log1p(simple["target_return"]), rtol=1e-10, equal_nan=True
        )
        volatility = series.labels_from_log_returns(
            returns["date"],
            returns["log_return"],
            horizon=horizon,
            entry_lag=1,
            target="log_realized_volatility",
        )
        theirs = etf_data.etf_labels(
            prices, horizon=horizon, entry_lag=1, target="log_realized_volatility"
        )
        np.testing.assert_allclose(
            volatility["target_return"], theirs["target_return"], rtol=1e-10, equal_nan=True
        )
        assert (total["target_end_date"].dropna() == simple["target_end_date"].dropna()).all()


def test_cash_distributions_and_splits_enter_the_total_return_exactly():
    batch = synthetic.fixture_batch(
        ["VTI"],
        start=pd.Timestamp("2008-01-02", tz="UTC"),
        end_exclusive=pd.Timestamp("2009-01-01", tz="UTC"),
        retrieved_at=pd.Timestamp("2026-10-14T12:00:00Z"),
    )
    sessions = nyse.sessions("2008-01-02", "2008-12-31")
    returns = series.daily_log_returns(
        sessions, batch.prices, batch.distributions, batch.splits, "VTI"
    )
    assert len(batch.distributions) and len(batch.splits)  # both cases are present
    closes = batch.prices["close"].to_numpy()
    cash = dict(zip(batch.distributions["ex_date"], batch.distributions["amount"], strict=True))
    split = dict(
        zip(batch.splits["effective_date"], batch.splits["ratio_new_per_old"], strict=True)
    )
    for t in range(1, len(sessions)):
        day = batch.prices["session_date"].iloc[t]
        expected = (split.get(day, 1.0) * closes[t] + cash.get(day, 0.0)) / closes[t - 1] - 1.0
        assert returns["log_return"].iloc[t] == pytest.approx(np.log1p(expected), rel=1e-12)


def test_a_missing_close_leaves_its_return_features_and_labels_unavailable():
    sessions, closes = _no_distribution_series()
    prices = _prices(closes, sessions).drop(index=100)
    returns = series.daily_log_returns(sessions, prices, EMPTY_CASH, EMPTY_SPLITS, "X")
    assert np.isnan(returns["log_return"].iloc[100])  # no close
    assert returns["sessions_spanned"].iloc[101] == 2  # labelled multi-session, never spread
    assert np.isnan(returns["log_return"].iloc[101])
    features = series.asset_frame(returns, "X")
    assert features.loc[100:104, "log_rv_5d"].isna().all()
    labels = series.labels_from_log_returns(
        returns["date"], returns["log_return"], horizon=5, entry_lag=1, target="total_log_return"
    )
    # rows whose (entry, exit] window covers sessions 100 or 101 are unlabeled
    assert labels.loc[94:99, "target_return"].isna().all()
    assert labels["target_return"].iloc[93] == pytest.approx(
        returns["log_return"].iloc[95:100].sum()
    )


def test_features_are_prefix_stable_and_labels_need_their_whole_window():
    sessions, closes = _no_distribution_series()
    full = series.daily_log_returns(
        sessions, _prices(closes, sessions), EMPTY_CASH, EMPTY_SPLITS, "X"
    )
    cut = 200
    head = series.daily_log_returns(
        sessions[:cut], _prices(closes[:cut], sessions[:cut]), EMPTY_CASH, EMPTY_SPLITS, "X"
    )
    pd.testing.assert_frame_equal(
        series.asset_frame(head, "X"), series.asset_frame(full, "X").iloc[:cut], check_exact=True
    )
    # and a feature depends, bit for bit, only on its own window of returns
    window = slice(cut - 61, cut)  # 61 closes: the 60 returns the longest feature needs
    tail = series.daily_log_returns(
        sessions[window], _prices(closes[window], sessions[window]), EMPTY_CASH, EMPTY_SPLITS, "X"
    )
    pd.testing.assert_series_equal(
        series.asset_frame(tail, "X").iloc[-1][list(series.FEATURES)],
        series.asset_frame(full, "X").iloc[cut - 1][list(series.FEATURES)],
        check_exact=True,
        check_names=False,
    )
    labels = series.labels_from_log_returns(
        head["date"], head["log_return"], horizon=20, entry_lag=1, target="total_log_return"
    )
    assert labels["target_return"].iloc[-21:].isna().all()  # exit after the last session


def test_adjusted_closes_and_split_adjusted_amounts_are_never_used():
    sessions, closes = _no_distribution_series(n=80)
    adjusted = _prices(closes, sessions).assign(
        adjustment_basis="split_and_distribution_adjusted_as_of_retrieval"
    )
    returns = series.daily_log_returns(sessions, adjusted, EMPTY_CASH, EMPTY_SPLITS, "X")
    assert returns["log_return"].isna().all()  # no unadjusted close: nothing is computed
    cash = pd.DataFrame(
        {
            "fund_id": ["X"],
            "ex_date": [sessions[10]],
            "amount": [0.5],
            "amount_basis": ["split_adjusted_as_of_retrieval"],
        }
    )
    with pytest.raises(series.SeriesRefused, match="per share as paid"):
        series.daily_log_returns(sessions, _prices(closes, sessions), cash, EMPTY_SPLITS, "X")


def test_two_vintages_of_one_close_are_refused_not_chosen():
    sessions, closes = _no_distribution_series(n=80)
    prices = _prices(closes, sessions)
    with pytest.raises(series.SeriesRefused, match="vintages"):
        series.daily_log_returns(
            sessions, pd.concat([prices, prices.iloc[:1]]), EMPTY_CASH, EMPTY_SPLITS, "X"
        )


# --- bands ---


def test_conformal_quantiles_use_the_finite_sample_ranks():
    values = np.arange(1.0, 20.0)  # 19 scores
    assert bands.conformal_quantiles(values, 0.1, 0.9) == (2.0, 18.0)  # ranks 2 and 18
    assert bands.conformal_quantiles(values[:8], 0.1, 0.9) is None  # rank 9 > 8: too few
    assert bands.conformal_quantiles(np.array([]), 0.1, 0.9) is None


def test_sequential_bands_use_only_earlier_matured_folds():
    dates = pd.bdate_range("2021-01-04", periods=60, tz="UTC").as_unit("ns")
    frame = pd.DataFrame(
        {
            "fold_id": np.repeat([0, 1, 2], 20),
            "date": dates,
            "symbol": "X",
            "target_end_date": dates + pd.Timedelta(days=3),
            "target_return": np.linspace(-1, 1, 60),
            "point": 0.0,
            "scale": 1.0,
        }
    )
    starts = {0: dates[0], 1: dates[20], 2: dates[40]}
    banded = bands.sequential_bands(frame, starts, lower_q=0.1, upper_q=0.9, min_calibration=10)
    assert banded.loc[frame["fold_id"] == 0, "band_low"].isna().all()  # nothing earlier
    second = banded.loc[frame["fold_id"] == 1]
    # fold 1 calibrates on fold 0's rows whose labels ended before its start (purge)
    known = (frame["fold_id"] < 1) & (frame["target_end_date"] < starts[1])
    assert second["n_calibration"].iloc[0] == known.sum() < 20
    expected = bands.conformal_quantiles(frame.loc[known, "target_return"].to_numpy(), 0.1, 0.9)
    assert second["band_low"].iloc[0] == expected[0]
    assert second["band_high"].iloc[0] == expected[1]
