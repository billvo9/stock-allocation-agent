"""
The cash account on real data (C1 with C9's booking, C3 as the risk-free rate) and the
Sharpe ratio over it (M2), against docs/research/group_c_decisions.md's examples.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.data.rates.cash import CARRY, CashUnavailable, cash_returns
from stock_agent.data.rates.vintages import as_of, merge_vintages, vintages_from_alfred
from stock_agent.portfolio.performance import ZERO, sharpe_ratio

RATE = "USD_TREASURY_3M"
RETRIEVED = pd.Timestamp("2026-10-27T03:00:00Z")


def alfred(rows, retrieved=RETRIEVED):
    observations = pd.DataFrame(rows, columns=["date", "value", "realtime_start"])
    return vintages_from_alfred(
        observations,
        rate_id=RATE,
        provider_series_id="DGS3MO",
        currency="USD",
        tenor="3M",
        quote_convention="investment_basis",
        retrieved_at=retrieved,
    )


def sessions(*days):
    return pd.DatetimeIndex(list(days), tz="UTC")


def test_c1_friday_to_monday_and_across_a_year_end():
    frame = alfred([("2024-03-08", "5.00", "2024-03-11"), ("2023-12-29", "5.00", "2024-01-02")])
    weekend = cash_returns(frame, sessions("2024-03-08", "2024-03-11"), RATE)
    assert weekend["accrual"].iloc[0] == pytest.approx(0.000409836, abs=1e-9)  # 3/366 in 2024
    year_end = cash_returns(frame, sessions("2023-12-29", "2024-01-02"), RATE)
    assert year_end["accrual"].iloc[0] == pytest.approx(0.000547571, abs=1e-9)  # 3/365 + 1/366


def test_c2_columbus_day_booking_and_c9_exception():
    # Friday 2024-10-11 at 5.10%, posted Tuesday; Monday 2024-10-14 has no value.
    frame = alfred(
        [
            ("2024-10-10", "5.00", "2024-10-11"),
            ("2024-10-11", "5.10", "2024-10-15"),
            ("2024-10-14", ".", "2024-10-15"),
        ]
    )
    booked = cash_returns(frame, sessions("2024-10-11", "2024-10-14", "2024-10-15"), RATE)
    assert booked["accrual"].tolist() == pytest.approx([0.000418033, 0.000139344], abs=1e-9)
    assert booked["reason"].tolist() == [None, CARRY]  # Monday has no value: Friday's carried
    assert booked["stale"].tolist() == [False, False]
    # C9: Monday's booked accrual uses Friday's rate, which as a decision input (C2) does
    # not count until the end of Tuesday.
    monday = pd.Timestamp("2024-10-14 18:00", tz="America/New_York")
    seen = as_of(frame, monday)
    assert pd.Timestamp("2024-10-11") not in set(seen["date"].dt.tz_localize(None))


def test_the_span_uses_the_rate_at_its_start_never_its_end():
    frame = alfred([("2024-06-03", "5.00", "2024-06-04"), ("2024-06-04", "6.00", "2024-06-05")])
    found = cash_returns(frame, sessions("2024-06-03", "2024-06-04"), RATE)
    assert found["accrual"].iloc[0] == pytest.approx(0.05 / 366)  # y_{t-1}, not y_t (0.06)
    assert found["annual_yield"].iloc[0] == pytest.approx(0.05)


def test_a_revision_never_changes_booked_cash_and_a_stale_rate_is_flagged():
    first = alfred([("2024-06-03", "5.00", "2024-06-04")])
    revised = merge_vintages(
        first,
        alfred([("2024-06-03", "5.20", "2024-07-01")], retrieved=RETRIEVED + pd.Timedelta(days=1)),
    )
    days = sessions("2024-06-03", "2024-06-04")
    assert cash_returns(revised, days, RATE)["accrual"].to_numpy().tobytes() == (
        cash_returns(first, days, RATE)["accrual"].to_numpy().tobytes()
    )
    stale = cash_returns(first, sessions("2024-06-12", "2024-06-13"), RATE)
    assert bool(stale["stale"].iloc[0]) and stale["reason"].iloc[0] == CARRY


def test_no_rate_on_or_before_a_span_refuses_the_series():
    frame = alfred([("2024-06-04", "5.00", "2024-06-05")])
    with pytest.raises(CashUnavailable, match="on or before 2024-06-03"):
        cash_returns(frame, sessions("2024-06-03", "2024-06-04"), RATE)


def test_m2_sharpe_over_c3_and_a_cash_portfolio_has_none():
    cash = [0.05 / 365] * 249 + [0.05 * 3 / 365]
    # A 100% cash portfolio: its excess return is 0 up to residue, so no Sharpe ratio (C3).
    residue = [c + 1e-17 for c in cash]
    found = sharpe_ratio(residue, cash, [1] * len(cash))
    assert found["status"] == "unavailable" and found["reason"] == ZERO
    rng = np.random.default_rng(0)
    returns = [c + float(x) for c, x in zip(cash, rng.normal(0.0005, 0.01, len(cash)), strict=True)]
    excess = np.array(returns) - np.array(cash)
    expected = math.sqrt(252) * excess.mean() / excess.std(ddof=1)
    assert sharpe_ratio(returns, cash, [1] * len(cash))["value"] == pytest.approx(expected)
    # Over a gap: single sessions only, labelled.
    gapped = sharpe_ratio(returns[:10], cash[:10], [1, 1, 2, 1, 1, 1, 1, 1, 1, 1])
    assert not gapped["complete"] and (gapped["sessions"], gapped["of"]) == (9, 11)


@pytest.mark.parametrize("start, stale", [("2024-06-10", False), ("2024-06-11", True)])
def test_a_rate_is_stale_after_five_business_days_not_at_five(start, stale):
    # The rate is dated Monday 2024-06-03: 5 business days old on 06-10, 6 on 06-11.
    frame = alfred([("2024-06-03", "5.00", "2024-06-04")])
    end = (pd.Timestamp(start) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    found = cash_returns(frame, sessions(start, end), RATE)
    assert bool(found["stale"].iloc[0]) is stale


def test_a_missing_first_release_is_carried_even_when_a_later_vintage_fills_it():
    frame = alfred(
        [
            ("2024-10-11", "5.10", "2024-10-15"),
            ("2024-10-14", ".", "2024-10-15"),
            ("2024-10-14", "5.05", "2024-11-01"),  # filled later: not the first release
        ]
    )
    found = cash_returns(frame, sessions("2024-10-14", "2024-10-15"), RATE)
    assert (
        found["annual_yield"].iloc[0] == pytest.approx(0.051) and found["reason"].iloc[0] == CARRY
    )


def test_a_pre_alfred_rate_is_flagged_and_other_rate_ids_are_ignored():
    frame = alfred([("2003-01-02", "1.20", "2005-06-28"), ("2005-07-01", "3.05", "2005-07-05")])
    early = cash_returns(frame, sessions("2003-01-02", "2003-01-03"), RATE)
    late = cash_returns(frame, sessions("2005-07-01", "2005-07-05"), RATE)
    assert early["pre_alfred"].tolist() == [True] and late["pre_alfred"].tolist() == [False]
    other = vintages_from_alfred(
        pd.DataFrame(
            [("2003-01-02", "9.99", "2005-06-28")], columns=["date", "value", "realtime_start"]
        ),
        rate_id="OTHER",
        provider_series_id="DGS1MO",
        currency="USD",
        tenor="1M",
        quote_convention="investment_basis",
        retrieved_at=RETRIEVED,
    )
    both = merge_vintages(frame, other)
    assert cash_returns(both, sessions("2003-01-02", "2003-01-03"), RATE)["annual_yield"].iloc[
        0
    ] == (pytest.approx(0.012))
