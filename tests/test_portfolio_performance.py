"""
The group D measures that need no new data (M1, M2, M5, M6, M7), checked against the
worked examples of docs/research/group_d_decisions.md (computed there by
scripts/group_cd_packet_numbers.py) and against their edge rules.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.portfolio.performance import (
    EDGE_MISSING,
    OUTSIDE_WINDOW,
    TOO_FEW,
    ZERO,
    Execution,
    StressWindow,
    annual_volatility,
    cagr,
    median_dollar_volume,
    participation,
    stress_window,
    turnover_by_year,
)
from stock_agent.portfolio.transition import TURNOVER_LIMIT, FeasibleTarget, StepResult

day = pd.Timestamp


# --- M1 ---


def test_m1_the_packets_examples():
    one_year = cagr(1.0, 1.10, day("2023-01-02"), day("2024-01-02"))  # 365 days
    assert one_year["value"] == pytest.approx(0.1000718, abs=5e-8) and one_year["annualized"]
    # 2021-01-04 to 2022-12-30 (725 days), +21% after a 5 bp build: the build is inside G.
    with_build = cagr(10_000, 10_000 * (1 - 0.0005) * 1.21, day("2021-01-04"), day("2022-12-30"))
    assert with_build["days"] == 725
    assert round(with_build["value"] * 100, 3) == pytest.approx(10.052)
    without = cagr(10_000, 12_100, day("2021-01-04"), day("2022-12-30"))
    assert round(without["value"] * 100, 3) == pytest.approx(10.080)


def test_m1_a_span_under_a_year_is_never_annualized():
    half = cagr(1.0, 1.05, day("2024-01-02"), day("2024-07-02"))  # 182 days
    assert half == {
        "value": pytest.approx(0.05),
        "annualized": False,
        "label": "partial",
        "days": 182,
    }
    assert cagr(1.0, 1.05, day("2024-01-02"), day("2025-01-01"))["annualized"]  # 365 days
    with pytest.raises(ValueError):
        cagr(0.0, 1.0, day("2024-01-02"), day("2025-01-02"))


# --- M2 ---


def test_m2_is_root_252_times_the_sample_sd_of_single_session_returns():
    returns = [0.01, -0.02, 0.015, 0.0, 0.005]
    result = annual_volatility(returns, [1] * 5)
    assert result["value"] == pytest.approx(math.sqrt(252) * np.std(returns, ddof=1))
    assert result["complete"] and result["sessions"] == result["of"] == 5


def test_m2_over_a_gap_uses_single_sessions_only_and_says_so():
    result = annual_volatility([0.01, 0.30, -0.02, 0.015], [1, 3, 1, 1])
    assert result["value"] == pytest.approx(math.sqrt(252) * np.std([0.01, -0.02, 0.015], ddof=1))
    # N counts every return session: the three-session span adds three.
    assert not result["complete"] and (result["sessions"], result["of"]) == (3, 6)


def test_m2_zero_variance_and_too_few_returns_are_unavailable():
    residue = annual_volatility([1e-16, 2e-16, 1.5e-16], [1, 1, 1])
    assert residue["status"] == "unavailable" and residue["reason"] == ZERO
    assert annual_volatility([0.01], [1])["reason"] == TOO_FEW


# --- M5 ---


def _wealth(values, start="2024-01-02"):
    index = pd.bdate_range(start, periods=len(values), tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


def test_m5_the_approved_f65_example_inside_a_window():
    wealth = _wealth([130, 100, 110, 99, 105, 111, 120])  # the whole path falls 23.8%
    window = StressWindow("w", "Window", wealth.index[1], wealth.index[5])
    found = stress_window(wealth, window)
    assert found["status"] == "ok" and found["return"] == pytest.approx(0.11)
    assert found["drawdown"]["value"] == pytest.approx(0.10)  # 110 -> 99, inside the window only


def test_m5_a_window_outside_the_path_is_never_shortened():
    wealth = _wealth([100, 101, 102])
    window = StressWindow("w", "Window", wealth.index[0] - pd.Timedelta(days=7), wealth.index[2])
    assert stress_window(wealth, window)["reason"] == OUTSIDE_WINDOW
    gapped = _wealth([100, np.nan, 102, 103])
    edge = StressWindow("w", "Window", gapped.index[1], gapped.index[3])
    assert stress_window(gapped, edge)["reason"] == EDGE_MISSING


# --- M6 ---


def _execution(session, turnover, requested, binding=(), notional=None):
    target = np.array(requested, dtype=float)
    record = FeasibleTarget(requested=target, weights=target, distance_one_way=0.0, binding=binding)
    step = StepResult(
        w_post_trade=target,
        w_next=target,
        gross_return=0.0,
        cost=0.0,
        net_return=0.0,
        turnover_one_way=turnover,
        traded_notional=2 * turnover if notional is None else notional,
        reason="traded",
    )
    return Execution(day(session), record, step)


YEARS = {
    2021: (day("2021-01-04"), day("2021-12-31")),
    2022: (day("2022-01-03"), day("2022-12-30")),
    2023: (day("2023-01-03"), day("2023-12-29")),
    2024: (day("2024-01-02"), day("2024-12-31")),
}


def test_m6_the_packets_example():
    target = [0.5, 0.5, 0.0]
    executions = [
        _execution("2021-03-01", 1.0, target),  # the build
        _execution("2021-09-01", 0.25, target),
        _execution("2022-06-01", 0.47, target),
        _execution("2023-06-01", 0.38, target),
        _execution("2024-03-01", 0.10, target),
    ]
    found = turnover_by_year(
        executions,
        first_session=day("2021-03-01"),
        last_session=day("2024-06-28"),
        year_bounds=YEARS,
    )
    assert found["construction"] == {"executions": 1, "turnover": 1.0, "traded_notional": 2.0}
    by_year = {row["year"]: (round(row["turnover"], 10), row["partial"]) for row in found["years"]}
    assert by_year == {
        2021: (0.25, True),
        2022: (0.47, False),
        2023: (0.38, False),
        2024: (0.1, True),
    }
    assert found["annual_turnover"] == pytest.approx(0.425) and found["full_years"] == 2


def test_m6_construction_ends_at_the_target_or_at_a_different_target():
    build = [0.6, 0.4, 0.0]
    limited = (TURNOVER_LIMIT,)
    reached = [
        _execution("2022-01-03", 0.25, build, limited),
        _execution("2022-02-01", 0.25, build, limited),
        _execution("2022-03-01", 0.20, build),  # reaches the build's own target
        _execution("2022-04-01", 0.05, build),
    ]
    found = turnover_by_year(
        reached, first_session=day("2022-01-03"), last_session=day("2022-12-30"), year_bounds=YEARS
    )
    assert found["construction"]["executions"] == 3
    assert found["years"][0]["turnover"] == pytest.approx(0.05)
    changed = [
        _execution("2022-01-03", 0.25, build, limited),
        _execution("2022-02-01", 0.25, [0.3, 0.7, 0.0], limited),  # a different target
    ]
    found = turnover_by_year(
        changed, first_session=day("2022-01-03"), last_session=day("2022-12-30"), year_bounds=YEARS
    )
    assert found["construction"]["executions"] == 1  # bounded: never all construction
    assert found["years"][0]["turnover"] == pytest.approx(0.25)


# --- M7 ---


def test_m7_the_packets_examples():
    mdv = np.full((25, 3), 20_000_000.0)  # $20m a session for every asset
    delta_rebalance = np.array([-0.15, 0.15, 0.0])  # 45% -> 30% on the first asset
    delta_build = np.array([0.30, 0.70, -1.0])
    found = participation(
        [(20, delta_build, 1.0), (24, delta_rebalance, 1.0)],
        mdv,
        2,
        notional=1_000_000,
        threshold=0.01,
        window=20,
    )
    assert found["rebalances"]["max_q"] == pytest.approx(0.0075)  # 0.75%: not flagged
    assert found["rebalances"]["flagged"] == 0
    assert found["build"]["flagged"] == 2  # 30% -> 1.50% and 70% -> 3.50%
    scaled = participation(
        [(20, delta_build, 1.0), (24, delta_rebalance, 1.0)],
        mdv,
        2,
        notional=10_000_000,
        threshold=0.01,
        window=20,
    )
    assert scaled["rebalances"]["max_q"] == pytest.approx(0.075)  # linear in N


def test_m7_the_median_never_includes_the_session_after_the_decision():
    volume = np.arange(1.0, 31.0).reshape(30, 1) * 1e6
    assert median_dollar_volume(volume, 19, 20)[0] == pytest.approx(10.5e6)  # sessions 0..19
    volume[20, 0] = 1e12  # session d + 1: must not count
    assert median_dollar_volume(volume, 19, 20)[0] == pytest.approx(10.5e6)
    assert np.isnan(median_dollar_volume(volume, 18, 20)[0])  # fewer than 20 sessions
    volume[5, 0] = np.nan
    assert np.isnan(median_dollar_volume(volume, 19, 20)[0])
    found = participation(
        [(19, np.array([1.0, -1.0]), 1.0)], volume, 1, notional=1e6, threshold=0.01, window=20
    )
    assert found["build"]["unavailable"] == 1 and found["build"]["assessed"] == 0


def test_boundaries_m1_m6_m7():
    assert not cagr(1.0, 1.05, day("2024-01-02"), day("2024-12-31"))["annualized"]  # 364 days
    target = [0.5, 0.5, 0.0]
    found = turnover_by_year(
        [_execution("2021-03-01", 1.0, target), _execution("2022-06-01", 0.47, target)],
        first_session=day("2021-03-01"),
        last_session=day("2022-12-30"),
        year_bounds=YEARS,
    )
    assert found["years"][1] == {
        "year": 2022, "turnover": 0.47, "traded_notional": 0.94, "partial": False
    }  # fmt: skip
    assert found["annual_turnover"] == pytest.approx(0.47) and found["full_years"] == 1
    mdv = np.full((25, 2), 20_000_000.0)
    exact = participation(
        [(20, np.array([1.0, -1.0]), 1.0), (24, np.array([0.2, -0.2]), 1.0)],
        mdv, 1, notional=1_000_000, threshold=0.01, window=20,
    )  # fmt: skip
    assert exact["rebalances"]["max_q"] == pytest.approx(0.01)
    assert exact["rebalances"]["flagged"] == 0  # flagged only above p
    zero = participation(
        [(20, np.array([1.0, -1.0]), 1.0)], np.zeros((25, 2)), 1,
        notional=1e6, threshold=0.01, window=20,
    )  # fmt: skip
    assert zero["build"]["unavailable"] == 1


def test_m7_scales_with_pre_trade_wealth():
    mdv = np.full((25, 3), 20_000_000.0)
    found = participation(
        [(20, np.array([0.5, 0.5, -1.0]), 1.0), (24, np.array([-0.15, 0.15, 0.0]), 1.2)],
        mdv, 2, notional=1_000_000, threshold=0.01, window=20,
    )  # fmt: skip
    assert found["rebalances"]["max_q"] == pytest.approx(0.009)  # 0.15 x $1.2m / $20m
