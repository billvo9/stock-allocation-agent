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


# --- F7.2: the rebalance-offset sensitivity family ---


@pytest.fixture(scope="module")
def with_family():
    frames = default_frames(SESSIONS)
    return _results(frames, offset_family=True), _results(frames)


def test_the_offset_family_never_changes_or_chooses_the_headline(with_family):
    family, plain = with_family
    assert family["variants_run"] == plain["variants_run"] == 10  # offsets are not variants
    for a, b in zip(family["universes"], plain["universes"], strict=True):
        for x, y in zip(a["strategies"], b["strategies"], strict=True):
            assert {k: v for k, v in x.items() if k != "offset_family"} == {
                k: v for k, v in y.items() if k != "offset_family"
            }  # every headline measure, not only the wealth path
            assert y["offset_family"] is None
            if x["rule"] == "buy_and_hold":
                assert x["offset_family"] == {
                    "applies": False,
                    "k": 21,
                    "reason": "no_rebalances_after_the_build",
                }
                continue
            fam = x["offset_family"]
            assert fam["applies"] and fam["k"] == x["every"] and fam["registered_offset"] == 0
            assert set(fam) == {
                "applies",
                "k",
                "registered_offset",
                "total_return",
                "stopped_offsets",
            }
            total = fam["total_return"]
            assert set(total) == {"registered", "spread"}  # F7.2: no min, max or best day
            assert total["registered"] == x["total_return"] and total["spread"] >= 0
            assert fam["stopped_offsets"] == 0


def test_every_offset_is_run_and_the_days_change_the_result(with_family):
    family, _ = with_family
    assert family["offset_family"] is True
    assert family["offset_family_simulations"] == 2 * (4 + 20 + 62 + 20)  # k - 1 per rule
    weekly = _strategy(_universe(family, "funds"), "equal_weight_weekly")
    assert weekly["offset_family"]["total_return"]["spread"] > 0


def test_an_offset_runs_the_registered_schedule_shifted_by_its_phase():
    cfg = config(offset_family=True)
    data = universe_data(cfg.universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    strategy = cfg.strategies[2]  # equal weight, monthly
    _, registered = showcase_run._targets(cfg, data.table, strategy, 0)
    _, shifted = showcase_run._targets(cfg, data.table, strategy, 7)
    assert min(registered) == min(shifted) == 20  # the common build decision (D3)
    assert sorted(registered)[1:3] == [41, 62] and sorted(shifted)[1:3] == [27, 48]


def test_the_spread_is_unavailable_when_any_offset_stopped():
    assert showcase_run._spread([0.1, None, 0.3], 0.1) == {"registered": 0.1, "spread": None}
    assert showcase_run._spread([0.1, 0.4, 0.3], 0.1)["spread"] == pytest.approx(0.3)


def _offset_wealth(cfg, data, strategy, offset):
    from stock_agent.portfolio.engine import simulate
    from stock_agent.portfolio.transition import CASH_ACCOUNT, Constraints, CostModel

    classes = {**cfg.asset_classes(), FUNDING_LEG: CASH_ACCOUNT}
    costs = CostModel.provisional({asset: classes[asset] for asset in data.table.assets})
    _, targets = showcase_run._targets(cfg, data.table, strategy, offset)
    return simulate(
        data.table.returns,
        targets,
        lag=1,
        cost_rate=costs.rates_for(data.table.assets),
        constraints=Constraints(cash_index=data.table.cash_index),
        initial_wealth=10000.0,
        has_close=data.table.has_close,
    ).final_wealth


def test_the_spread_matches_an_independent_replay_of_every_offset(with_family):
    family, _ = with_family
    cfg = config(offset_family=True)
    data = universe_data(cfg.universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    strategy = cfg.strategies[1]  # equal weight, weekly: k = 5
    totals = [_offset_wealth(cfg, data, strategy, o) / 10000 - 1 for o in range(5)]
    reported = _strategy(_universe(family, "funds"), strategy.key)["offset_family"]
    assert reported["total_return"]["spread"] == pytest.approx(max(totals) - min(totals), abs=1e-12)
    assert reported["total_return"]["registered"] == pytest.approx(totals[0], abs=1e-12)


def test_buy_and_hold_is_the_same_at_every_offset_which_is_why_it_has_no_family():
    cfg = config(offset_family=True)
    data = universe_data(cfg.universe("funds"), SESSIONS, loader(default_frames(SESSIONS)))
    hold = cfg.strategies[0]
    finals = {_offset_wealth(cfg, data, hold, o) for o in (0, 3, 11, 20)}
    assert len(finals) == 1


def test_every_non_registered_offset_is_actually_simulated(monkeypatch):
    calls = []
    real = showcase_run.simulate

    def counting(returns, targets, **kwargs):
        calls.append(min(d for d in targets if d > 20))  # the first rebalance decision
        return real(returns, targets, **kwargs)

    monkeypatch.setattr(showcase_run, "simulate", counting)
    results = _results(offset_family=True)
    assert len(calls) == results["offset_family_simulations"] == 2 * (4 + 20 + 62 + 20)
    weekly = sorted(calls[:4])  # the first universe's weekly rule: offsets 1..4
    assert weekly == [21, 22, 23, 24]


def test_a_family_whose_offsets_stop_reports_no_spread():
    frames = default_frames(SESSIONS)
    closes = noisy(SESSIONS, 7, 0.01)
    closes[150:] /= 2
    frames["XAA"] = bars(SESSIONS, closes, div={150: 0.2}, split={150: 2.0})
    universe = _universe(_results(frames, offset_family=True), "funds")
    weekly = _strategy(universe, "equal_weight_weekly")["offset_family"]
    assert weekly["stopped_offsets"] == 5 and weekly["total_return"]["spread"] is None
    assert weekly["total_return"]["registered"] is None


# --- group D measures (approved 2026-10-26) ---

GROUP_D = {
    "stress_windows": [
        {"key": "spring", "label": "Spring", "start": "2024-03-01", "end": "2024-05-31"},
        {"key": "covid_crash", "label": "COVID crash", "start": "2020-02-19", "end": "2020-03-23"},
    ],
    "participation": {"notional": 1000000, "threshold": 0.01, "window": 20},
}


@pytest.fixture(scope="module")
def with_measures():
    frames = default_frames(SESSIONS)
    return _results(frames, measures=GROUP_D), _results(frames)


def test_the_group_d_measures_never_change_the_headline(with_measures):
    measured, plain = with_measures
    assert measured["group_d"] is True and plain["group_d"] is False
    for a, b in zip(measured["universes"], plain["universes"], strict=True):
        for x, y in zip(a["strategies"], b["strategies"], strict=True):
            assert {k: v for k, v in x.items() if k != "group_d"} == {
                k: v for k, v in y.items() if k != "group_d"
            }
            assert y["group_d"] is None and x["group_d"] is not None


def test_the_group_d_measures_agree_with_the_headline_record(with_measures):
    from stock_agent.portfolio.performance import annual_volatility, cagr

    measured, _ = with_measures
    for universe in measured["universes"]:
        for s in universe["strategies"]:
            g = s["group_d"]
            expected = cagr(
                10_000, s["final_wealth"], pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
            )
            assert {k: g["cagr"][k] for k in expected} == expected  # under a year here
            assert not g["cagr"]["annualized"] and g["cagr"]["end"] == s["end"]
            wealth = pd.Series(s["wealth"], dtype="float64")
            daily = (wealth / wealth.shift(1) - 1).dropna().to_list()
            # The saved path is rounded to the cent; the measure uses the unrounded one.
            assert g["volatility"]["value"] == pytest.approx(
                annual_volatility(daily, [1] * len(daily))["value"], rel=1e-5
            )
            turnover = g["turnover_by_year"]
            yearly = sum(row["turnover"] for row in turnover["years"])
            total = turnover["construction"]["turnover"] + yearly
            assert total == pytest.approx(s["turnover"]["total_one_way"])
            assert turnover["construction"]["executions"] == 1  # no turnover limit: the build
            assert turnover["years"] == [
                {
                    "year": 2024,
                    "turnover": pytest.approx(yearly),
                    "traded_notional": pytest.approx(turnover["years"][0]["traded_notional"]),
                    "partial": True,
                }
            ]
            assert turnover["annual_turnover"] is None  # no full calendar year in the data
            spring, covid = g["stress_windows"]
            assert spring["status"] == "ok" and covid["reason"] == "outside_window"
            participation = g["participation"]
            assert participation["rebalances"]["executions"] == s["turnover"]["rebalances"]


def test_the_stress_window_return_is_the_wealth_ratio_inside_it(with_measures):
    measured, _ = with_measures
    universe = measured["universes"][0]
    dates = universe["dates"]
    for s in universe["strategies"]:
        spring = s["group_d"]["stress_windows"][0]
        start, end = dates.index("2024-03-01"), dates.index("2024-05-31")
        cents = s["wealth"][end] / s["wealth"][start] - 1  # the saved path is rounded to cents
        assert spring["return"] == pytest.approx(cents, abs=1e-5)


def test_participation_uses_the_decision_session_the_trade_and_the_pre_trade_wealth(monkeypatch):
    from stock_agent.portfolio import performance

    captured = []
    real = performance.participation

    def capture(trades, *args, **kwargs):
        captured.append(list(trades))
        return real(trades, *args, **kwargs)

    monkeypatch.setattr(performance, "participation", capture)
    results = _results(measures=GROUP_D)
    funds = _universe(results, "funds")
    for s, trades in zip(funds["strategies"], captured[: len(funds["strategies"])], strict=True):
        decisions = [decision for decision, _, _ in trades]
        assert decisions[0] == 20  # the build decision, not its execution session
        if s["rule"] != "buy_and_hold":
            assert all((d - 20) % s["every"] == 0 for d in decisions)
        # The executed change, not the holdings: half its absolute sum is the turnover.
        halves = sum(0.5 * float(np.abs(delta).sum()) for _, delta, _ in trades)
        assert halves == pytest.approx(s["turnover"]["total_one_way"])
        # Pre-trade wealth per 1.0 of initial cash, at the execution close (decision + 1).
        for decision, _, wealth in trades:
            at = funds["dates"].index(SESSIONS[decision + 1].strftime("%Y-%m-%d"))
            assert wealth == pytest.approx(s["wealth"][at] / 10_000, abs=1e-6)


def test_the_measures_over_a_gap_are_labelled_and_never_bridged():
    frames = default_frames(SESSIONS)
    frames["XAA"].loc[100, ["open", "high", "low", "close", "volume"]] = np.nan
    gap_day = SESSIONS[100].strftime("%Y-%m-%d")
    windows = {
        "stress_windows": [
            {"key": "across", "label": "Across", "start": "2024-04-01", "end": "2024-07-01"},
            {"key": "edge", "label": "Edge", "start": gap_day, "end": "2024-08-01"},
        ],
        "participation": GROUP_D["participation"],
    }
    universe = _universe(_results(frames, measures=windows), "funds")
    for s in universe["strategies"]:
        g = s["group_d"]
        assert not g["volatility"]["complete"]
        # The two-session span counts two return sessions in N and none in n.
        assert g["volatility"]["sessions"] == g["volatility"]["of"] - 2
        across, edge = g["stress_windows"]
        assert across["status"] == "ok" and across["drawdown"]["complete"] is False
        assert edge["reason"] == "window_edge_value_unavailable"
    weekly = _strategy(universe, "equal_weight_weekly")["group_d"]["participation"]
    assert weekly["rebalances"]["unavailable"] > 0  # no volume at the gap: never filled
    page = render_html({**_results(frames, measures=windows), "universes": [universe]})
    assert "without volume data" in page and "can make it read too low" in page


def test_a_stopped_run_has_no_growth_rate_and_still_renders():
    frames = default_frames(SESSIONS)
    closes = noisy(SESSIONS, 7, 0.01)
    closes[150:] /= 2
    frames["XAA"] = bars(SESSIONS, closes, div={150: 0.2}, split={150: 2.0})
    late = {
        "stress_windows": [
            {"key": "late", "label": "Late", "start": "2024-10-01", "end": "2024-11-29"},
        ],
        "participation": GROUP_D["participation"],
    }
    results = _results(frames, measures=late)
    for s in _universe(results, "funds")["strategies"]:
        assert s["group_d"]["cagr"]["value"] is None and s["group_d"]["cagr"]["reason"] == "stopped"
        # Inside the registered span but after the stop: said so, never "outside".
        reason = s["group_d"]["stress_windows"][0]["reason"]
        assert reason == "run_stopped_before_window_end"
    page = render_html(results)
    assert "Growth rate (CAGR)" in page and "unavailable (stopped)" in page


def test_the_growth_rate_runs_to_the_last_valued_close():
    frames = default_frames(SESSIONS)
    frames["XAA"].loc[len(SESSIONS) - 1, ["open", "high", "low", "close", "volume"]] = np.nan
    universe = _universe(_results(frames, measures=GROUP_D), "funds")
    for s in universe["strategies"]:
        assert s["final_wealth"] is None and s["complete"]  # no value on the last session
        cagr = s["group_d"]["cagr"]
        assert cagr["end"] == SESSIONS[-2].strftime("%Y-%m-%d")
        last = s["wealth"][-2]
        assert cagr["value"] == pytest.approx(last / 10_000 - 1, abs=1e-6)  # under a year
