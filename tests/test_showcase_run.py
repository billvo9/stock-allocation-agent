"""
The showcase's returns and runs on fixtures: F5.1 from raw bars, cadences,
buy-and-hold drift against rebalancing, cost accounting, the funding leg never
earning, no lookahead, and every variant reported in registered order.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from showcase_fixtures import (
    bars,
    config,
    default_frames,
    growing,
    loader,
    noisy,
    record,
    sessions_2024,
)

from stock_agent.portfolio.descriptive import SHARE_BASIS_UNKNOWN
from stock_agent.portfolio.engine import collect_targets
from stock_agent.portfolio.schedule import ScheduleSpec
from stock_agent.portfolio.transition import PROVISIONAL_RATES
from stock_agent.showcase import run as showcase_run
from stock_agent.showcase.report import render_html
from stock_agent.showcase.returns import FUNDING_LEG, return_table, ticker_returns
from stock_agent.showcase.run import (
    ShowcaseRunRefused,
    run_showcase,
    run_strategy,
    universe_data,
)
from stock_agent.showcase.strategies import make_policy

SESSIONS = sessions_2024()


def _results(frames=None, **changes):
    frames = frames or default_frames(SESSIONS)
    return run_showcase(config(**changes), loader(frames), sessions=SESSIONS, code_sha=None)


def _universe(results, key):
    return next(u for u in results["universes"] if u["key"] == key)


def _strategy(universe, key):
    return next(s for s in universe["strategies"] if s["key"] == key)


# --- F5.1 returns from raw bars ---


def test_an_ex_dividend_session_earns_the_dividend_and_a_split_is_not_a_loss():
    closes = np.full(len(SESSIONS), 100.0)
    closes[1], closes[2] = 99.5, 100.495  # the spec's ex-date example (T5.1)
    closes[3:] = 25.25  # a 4:1 split at session 3: 100.495 -> 25.25 (T5.2-like)
    frame = ticker_returns(SESSIONS, bars(SESSIONS, closes, div={1: 1.0}, split={3: 4.0}))
    r = frame["total_return"].to_numpy()
    assert r[1] == pytest.approx(0.005, abs=1e-12)  # not price-only -0.5%
    assert r[2] == pytest.approx(0.01, abs=1e-12)
    assert r[3] == pytest.approx(4 * 25.25 / 100.495 - 1, abs=1e-12)  # not -74.9%


def test_an_unserved_session_stays_a_gap_with_a_labelled_multi_session_return():
    closes = growing(SESSIONS, 0.01)
    closes[5] = np.nan
    table = return_table(SESSIONS, {"XAA": bars(SESSIONS, closes), "XBB": bars(SESSIONS, closes)})
    assert not table.has_close[5, 0] and np.isnan(table.returns[5, 0])
    assert table.returns[6, 0] == pytest.approx(1.01**2 - 1)  # two sessions in one return
    assert table.frames["XAA"]["sessions_spanned"].iloc[6] == 2
    assert table.missing_sessions["XAA"] == 1 and table.unavailable["XAA"] == {}
    assert table.assets[-1] == FUNDING_LEG and (table.returns[:, -1] == 0).all()


def test_a_split_and_a_dividend_on_one_session_are_unavailable_not_guessed():
    closes = growing(SESSIONS, 0.0)
    closes[40:] /= 2
    frame = bars(SESSIONS, closes, div={40: 0.5}, split={40: 2.0})
    table = return_table(SESSIONS, {"XAA": frame, "XBB": bars(SESSIONS, growing(SESSIONS, 0))})
    assert np.isnan(table.returns[40, 0]) and table.has_close[40, 0]
    assert table.unavailable["XAA"] == {SHARE_BASIS_UNKNOWN: 1}


def test_a_held_asset_with_an_unavailable_return_stops_the_portfolio_and_says_so():
    frames = default_frames(SESSIONS)
    closes = noisy(SESSIONS, 7, 0.01)
    closes[150:] /= 2
    frames["XAA"] = bars(SESSIONS, closes, div={150: 0.2}, split={150: 2.0})
    universe = _universe(_results(frames), "funds")
    for strategy in universe["strategies"]:
        assert not strategy["complete"] and strategy["status_reason"] == "distribution_in_gap"
        assert strategy["end"] < SESSIONS[150].strftime("%Y-%m-%d")  # nothing past the stop
        assert strategy["final_wealth"] is None and strategy["total_return"] is None
        assert strategy["cost_sensitivity"]["0x"]["final_wealth"] is None
        assert strategy["stop_detail"] == {
            "session": SESSIONS[150].strftime("%Y-%m-%d"),
            "reasons": {"XAA": SHARE_BASIS_UNKNOWN},  # the real cause, not the engine's label
        }
    page = render_html({**_results(frames), "universes": [universe]})
    assert f"stopped on {SESSIONS[150]:%Y-%m-%d}: XAA {SHARE_BASIS_UNKNOWN}" in page


# --- schedules and the common build ---


def test_every_variant_builds_at_the_same_close_and_rebalances_on_its_cadence():
    universe = _universe(_results(), "funds")
    first, lag = 20, 1
    build = SESSIONS[first + lag].strftime("%Y-%m-%d")
    assert universe["build"] == build
    expected = {"equal_weight_weekly": 5, "equal_weight_monthly": 21, "equal_weight_quarterly": 63}
    last = len(SESSIONS) - 1
    for strategy in universe["strategies"]:
        assert strategy["start"] == build
        every = expected.get(strategy["key"])
        if every is None:
            continue
        executions = [d + lag for d in range(first + every, last + 1, every) if d + lag <= last - 1]
        assert strategy["turnover"]["rebalances"] == len(executions)
        assert strategy["executions"]["trades"] == len(executions) + 1  # and the build


def test_the_rebalance_sessions_are_the_registered_schedule():
    data = universe_data(config().universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    policy = make_policy("equal_weight", 3, 2, window=20)
    targets = collect_targets(policy, data.table.returns, ScheduleSpec(21, 0, 1, 20))
    assert sorted(targets) == list(range(20, len(SESSIONS), 21))


# --- buy-and-hold drift against rebalancing ---


def zigzag(sessions, swing: float = 0.01) -> np.ndarray:
    """A level price that alternates around 100: volatile, with no trend."""

    return 100.0 * (1.0 + swing * (-1.0) ** np.arange(len(sessions)))


def _two_assets(first: np.ndarray, second: np.ndarray):
    return {
        "XAA": bars(SESSIONS, first),
        "XBB": bars(SESSIONS, second),
        "SAA": bars(SESSIONS, first),
        "SBB": bars(SESSIONS, second),
    }


BUILD = 21  # first_decision 20 + lag 1


def test_buy_and_hold_drifts_while_equal_weight_trades_back():
    up, level = growing(SESSIONS, 0.002), zigzag(SESSIONS)
    universe = _universe(_results(_two_assets(up, level)), "funds")
    hold = _strategy(universe, "buy_and_hold")
    rate = PROVISIONAL_RATES["broad_etf"]
    expected = 10000 * (1 - rate) * (0.5 * up[-1] / up[BUILD] + 0.5 * level[-1] / level[BUILD])
    assert hold["final_wealth"] == pytest.approx(expected, rel=1e-12)
    assert hold["turnover"]["rebalances"] == 0
    assert hold["turnover"]["build_one_way"] == pytest.approx(1.0)  # a full switch from cash
    weekly = _strategy(universe, "equal_weight_weekly")
    assert weekly["turnover"]["rebalances"] > 0 and weekly["turnover"]["rebalance_one_way"] > 0
    # Rebalancing sells the rising asset: on this fixture it ends below buy-and-hold.
    assert weekly["final_wealth"] < hold["final_wealth"]


def test_with_no_relative_move_rebalancing_trades_nothing_and_pays_only_the_build():
    level = zigzag(SESSIONS)
    universe = _universe(_results(_two_assets(level, level.copy())), "funds")
    for strategy in universe["strategies"]:
        assert strategy["costs"]["paid_dollars"] == pytest.approx(10000 * 0.0005)
        assert strategy["turnover"]["rebalance_one_way"] == pytest.approx(0.0, abs=1e-12)


# --- costs ---


def test_costs_are_charged_at_the_registered_rates_and_scale_with_the_multiplier():
    universe = _universe(_results(), "stocks")
    for strategy in universe["strategies"]:
        sensitivity = strategy["cost_sensitivity"]
        assert sensitivity["0x"]["final_wealth"] > sensitivity["1x"]["final_wealth"]
        assert sensitivity["1x"]["final_wealth"] > sensitivity["2x"]["final_wealth"]
        assert sensitivity["1x"]["final_wealth"] == pytest.approx(strategy["final_wealth"])
        # The build alone costs the single-stock rate on the whole portfolio.
        assert strategy["costs"]["paid_dollars"] >= 10000 * PROVISIONAL_RATES["single_stock"]
        assert strategy["costs"]["paid_fraction_of_initial"] == pytest.approx(
            strategy["costs"]["paid_dollars"] / 10000
        )
    hold = _strategy(universe, "buy_and_hold")
    assert hold["costs"]["paid_dollars"] == pytest.approx(10000 * PROVISIONAL_RATES["single_stock"])
    assert hold["costs"]["summed_bps"] == pytest.approx(10.0)


def test_the_cost_sensitivity_replays_the_same_decisions():
    cfg = config()
    data = universe_data(cfg.universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    weekly = run_strategy(cfg, data, cfg.strategies[1])
    assert {k: v["status"] for k, v in weekly["cost_sensitivity"].items()} == {
        "0x": "complete",
        "1x": "complete",
        "2x": "complete",
    }
    # Same decisions: 0x pays nothing, so its growth is the gross growth of the same trades.
    assert weekly["cost_sensitivity"]["0x"]["final_wealth"] > weekly["final_wealth"]


# --- the funding leg never earns, and no decision sees the future ---


def test_the_funding_legs_return_never_reaches_a_result(monkeypatch):
    baseline = _results()
    original = return_table

    def with_cash_return(sessions, frames):
        table = original(sessions, frames)
        table.returns[:, table.cash_index] = 0.05  # 5% a session if cash were ever held
        return table

    monkeypatch.setattr(showcase_run, "return_table", with_cash_return)
    changed = _results()
    for before, after in zip(baseline["universes"], changed["universes"], strict=True):
        for a, b in zip(before["strategies"], after["strategies"], strict=True):
            assert a["wealth"] == b["wealth"] and a["final_wealth"] == b["final_wealth"]


def test_inverse_volatility_uses_one_over_the_trailing_volatility_and_never_the_future():
    data = universe_data(config().universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    returns = data.table.returns
    policy = make_policy("inverse_volatility", 3, 2, window=20)
    target = policy(returns[:21].copy())
    sigma = np.nanstd(returns[1:21, :2], axis=0, ddof=1)
    expected = (1 / sigma) / (1 / sigma).sum()
    assert target[:2] == pytest.approx(expected) and target[2] == 0.0
    future = returns.copy()
    future[21:, :2] *= 5.0  # change everything after the decision close
    schedule = ScheduleSpec(21, 0, 1, 20)
    before = collect_targets(make_policy("inverse_volatility", 3, 2, window=20), returns, schedule)
    after = collect_targets(make_policy("inverse_volatility", 3, 2, window=20), future, schedule)
    assert np.array_equal(before[20], after[20])


def test_every_target_is_fully_invested():
    universe = _universe(_results(), "stocks")
    for strategy in universe["strategies"]:
        assert strategy["executions"]["by_reason"].get("unexecutable_missing_close", 0) == 0
    data = universe_data(config().universe("stocks"), SESSIONS, loader(default_frames(SESSIONS)))
    for rule in ("buy_and_hold", "equal_weight", "inverse_volatility"):
        target = make_policy(rule, 3, 2, window=20)(data.table.returns[:21].copy())
        assert target[2] == 0.0 and target.sum() == pytest.approx(1.0)


# --- the record ---


def test_every_variant_is_reported_in_registered_order_as_strict_json():
    results = _results()
    assert all(s["stop_detail"] is None for u in results["universes"] for s in u["strategies"])
    json.dumps(results, allow_nan=False)
    assert results["variants_run"] == 2 * 5 and results["simulations_run"] == 30
    for universe in results["universes"]:
        keys = [s["key"] for s in universe["strategies"]]
        assert keys == [s["key"] for s in record()["strategies"]]
        assert all(len(s["wealth"]) == len(universe["dates"]) for s in universe["strategies"])
        assert universe["dates"][0] == universe["build"]
        assert universe["lineage"]["XAA" if universe["key"] == "funds" else "SAA"][0]["batch_id"]


def test_drawdown_and_calendar_years_use_the_approved_definitions():
    universe = _universe(_results(), "funds")
    hold = _strategy(universe, "buy_and_hold")
    wealth = pd.Series(hold["wealth"], dtype="float64")
    expected = float((1 - wealth / wealth.cummax()).max())
    assert hold["drawdown"]["measure"] == "maximum_drawdown"
    assert hold["drawdown"]["value"] == pytest.approx(expected)
    (year,) = hold["calendar_years"]
    assert year["year"] == 2024 and year["partial"]  # starts at the build close, not 2023's end
    assert year["total_return"] == pytest.approx(hold["total_return"])


def test_each_holdings_own_growth_has_no_cost():
    up, level = growing(SESSIONS, 0.002), zigzag(SESSIONS)
    universe = _universe(_results(_two_assets(up, level)), "funds")
    (rising, flat) = universe["components"]
    assert rising["total_return"] == pytest.approx(up[-1] / up[BUILD] - 1)
    assert flat["total_return"] == pytest.approx(level[-1] / level[BUILD] - 1)
    assert math.isclose(rising["drawdown"]["value"], 0.0, abs_tol=1e-12)


# --- review follow-ups: deferred builds, rules that cannot decide, prefix stability, gaps ---


def test_a_build_that_cannot_execute_at_the_registered_close_refuses_the_run():
    frames = default_frames(SESSIONS)
    frames["XAA"].loc[BUILD, ["open", "high", "low", "close", "volume"]] = np.nan
    with pytest.raises(ShowcaseRunRefused, match="build could not execute"):
        _results(frames)


def test_a_rule_that_cannot_decide_refuses_the_run_with_a_reason():
    level = zigzag(SESSIONS)
    frames = _two_assets(level, level.copy())
    frames["XBB"] = bars(SESSIONS, np.full(len(SESSIONS), 100.0))  # zero volatility
    with pytest.raises(ShowcaseRunRefused, match="inverse_volatility_monthly: the rule cannot"):
        _results(frames)


def test_no_later_inverse_volatility_decision_sees_its_future():
    data = universe_data(config().universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    returns = data.table.returns
    schedule = ScheduleSpec(21, 0, 1, 20)
    for decision in (41, 83, 209):
        future = returns.copy()
        future[decision + 1 :, :2] *= 5.0
        before = collect_targets(
            make_policy("inverse_volatility", 3, 2, window=20), returns, schedule
        )
        after = collect_targets(
            make_policy("inverse_volatility", 3, 2, window=20), future, schedule
        )
        assert np.array_equal(before[decision], after[decision])
        window = returns[decision - 19 : decision + 1, :2]
        sigma = window.std(axis=0, ddof=1)
        assert before[decision][:2] == pytest.approx((1 / sigma) / (1 / sigma).sum())


def test_changing_prices_after_a_session_leaves_every_path_up_to_it_unchanged():
    frames = default_frames(SESSIONS)
    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    k = 150
    for frame in changed.values():
        frame.loc[k + 1 :, ["open", "high", "low", "close"]] *= 1.5
    before, after = _results(frames), _results(changed)
    cut = SESSIONS[k].strftime("%Y-%m-%d")
    for a, b in zip(before["universes"], after["universes"], strict=True):
        upto = a["dates"].index(cut) + 1
        for x, y in zip(a["strategies"], b["strategies"], strict=True):
            assert x["wealth"][:upto] == y["wealth"][:upto]
            assert x["wealth"] != y["wealth"]


def test_a_mid_run_gap_leaves_wealth_unavailable_and_the_drawdown_labelled_incomplete():
    frames = default_frames(SESSIONS)
    frames["XAA"].loc[100, ["open", "high", "low", "close", "volume"]] = np.nan
    universe = _universe(_results(frames), "funds")
    gap = universe["dates"].index(SESSIONS[100].strftime("%Y-%m-%d"))
    for strategy in universe["strategies"]:
        assert strategy["complete"] and strategy["wealth"][gap] is None
        assert strategy["drawdown"]["measure"] == "largest_observed_decline_incomplete_history"
        assert strategy["drawdown"]["missing_sessions"] == 1
    page = render_html({**_results(frames), "universes": [universe]})
    chart = page[page.index('data-chart="funds"') :]
    path = re.search(r'<path class="line rule-buy_and_hold" d="([^"]+)"', chart).group(1)
    assert path.count("M") == 2  # the line breaks at the gap instead of bridging it
