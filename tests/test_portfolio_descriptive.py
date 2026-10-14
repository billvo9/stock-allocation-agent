"""
OD8 group A (approved 2026-10-14): descriptive fund-history calculations.

Every hand example is from `docs/research/t5_portfolio_formulas.md` and the
group A approval sheet, so a wrong formula fails. A property test checks the
growth index against an independent share-count simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.app.provider import DescriptiveMetrics
from stock_agent.portfolio import descriptive as d


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def days(*texts: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([utc(text) for text in texts])


WEEK = days("2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12")


def tr(sessions, closes, distributions=None, splits=None):
    """Total returns from raw closes and as-paid amounts, the basis stated as every caller must."""

    return d.total_returns(
        sessions,
        closes,
        distributions,
        splits,
        close_basis=d.UNADJUSTED,
        amount_basis=d.PER_SHARE_AS_PAID,
    )


def returns_of(frame: pd.DataFrame) -> pd.Series:
    return frame.set_index("date")["total_return"]


def spans_of(frame: pd.DataFrame) -> pd.Series:
    return frame.set_index("date")["sessions_spanned"]


# --- A1 and R1: total return from raw records ---


def test_ex_date_return_adds_the_distribution_to_the_raw_close():
    closes = pd.Series([100.0, 99.5, 100.495], index=WEEK[:3])
    distributions = pd.DataFrame({"ex_date": [WEEK[1]], "amount": [1.0]})
    frame = tr(WEEK[:3], closes, distributions)
    assert frame["status"].tolist() == ["unavailable", "ok", "ok"]
    assert frame["reason"].iloc[0] == d.NO_PRIOR_CLOSE
    assert frame["total_return"].iloc[1] == pytest.approx(0.005, abs=1e-12)  # not -0.005
    assert frame["total_return"].iloc[2] == pytest.approx(0.01, abs=1e-12)
    assert frame["sessions_spanned"].tolist() == [0, 1, 1]


def test_a_split_scales_the_new_close_by_the_share_ratio():
    closes = pd.Series([400.0, 101.0], index=WEEK[:2])
    splits = pd.DataFrame({"effective_date": [WEEK[1]], "ratio": [4.0]})
    frame = tr(WEEK[:2], closes, splits=splits)
    assert frame["total_return"].iloc[1] == pytest.approx(0.01, abs=1e-12)  # not -0.7475


@pytest.mark.parametrize(("amount", "basis"), [(1.0, d.PRE_SPLIT), (0.25, d.POST_SPLIT)])
def test_a_split_and_distribution_on_one_date_use_one_share_basis(amount, basis):
    closes = pd.Series([400.0, 99.5], index=WEEK[:2])
    splits = pd.DataFrame({"effective_date": [WEEK[1]], "ratio": [4.0]})
    distributions = pd.DataFrame({"ex_date": [WEEK[1]], "amount": [amount], "share_basis": [basis]})
    frame = tr(WEEK[:2], closes, distributions, splits)
    assert frame["total_return"].iloc[1] == pytest.approx(-0.0025, abs=1e-12)  # mixing: -0.4375%


def test_an_unknown_share_basis_on_a_split_date_is_unavailable():
    closes = pd.Series([400.0, 99.5], index=WEEK[:2])
    splits = pd.DataFrame({"effective_date": [WEEK[1]], "ratio": [4.0]})
    distributions = pd.DataFrame({"ex_date": [WEEK[1]], "amount": [1.0]})
    frame = tr(WEEK[:2], closes, distributions, splits)
    assert frame.loc[1, "status"] == "unavailable"
    assert frame.loc[1, "reason"] == d.SHARE_BASIS_UNKNOWN


@pytest.mark.parametrize(
    "kwargs",
    [
        {"close_basis": "split_and_distribution_adjusted_as_of_retrieval"},
        {"close_basis": "split_adjusted_as_of_retrieval"},
        {"amount_basis": "split_adjusted_as_of_retrieval"},
    ],
)
def test_adjusted_inputs_are_refused_so_nothing_is_counted_twice(kwargs):
    closes = pd.Series([100.0, 101.0], index=WEEK[:2])
    stated = {"close_basis": d.UNADJUSTED, "amount_basis": d.PER_SHARE_AS_PAID, **kwargs}
    with pytest.raises(ValueError, match="R1"):
        d.total_returns(WEEK[:2], closes, **stated)


def test_the_input_basis_must_always_be_stated():
    closes = pd.Series([100.0, 101.0], index=WEEK[:2])
    with pytest.raises(TypeError):
        d.total_returns(WEEK[:2], closes)  # no default basis can pass silently


def test_an_unknown_share_basis_is_refused_even_without_a_split():
    closes = pd.Series([100.0, 99.5], index=WEEK[:2])
    distributions = pd.DataFrame(
        {"ex_date": [WEEK[1]], "amount": [1.0], "share_basis": ["split_adjusted_as_of_retrieval"]}
    )
    with pytest.raises(ValueError, match="share_basis"):
        tr(WEEK[:2], closes, distributions)


@pytest.mark.parametrize(
    ("closes", "distributions", "splits"),
    [
        (pd.Series([100.0, -1.0], index=WEEK[:2]), None, None),
        (pd.Series([100.0, np.inf], index=WEEK[:2]), None, None),
        (pd.Series([100.0, 101.0], index=days("2024-01-08", "2024-01-13")), None, None),
        (
            pd.Series([100.0, 101.0], index=WEEK[:2]),
            pd.DataFrame({"ex_date": [WEEK[1]], "amount": [0.0]}),
            None,
        ),
        (
            pd.Series([100.0, 101.0], index=WEEK[:2]),
            None,
            pd.DataFrame({"effective_date": [WEEK[1]], "ratio": [np.nan]}),
        ),
    ],
)
def test_invalid_records_are_refused(closes, distributions, splits):
    with pytest.raises(ValueError):
        tr(WEEK[:2], closes, distributions, splits)


# --- R5a and R5b: sessions without a close ---


def test_a_gap_gives_one_labelled_multi_session_return_never_a_one_day_return():
    closes = pd.Series([100.0, 103.0], index=days("2024-01-08", "2024-01-11"))
    frame = tr(WEEK[:4], closes)
    assert frame["close_observed"].tolist() == [True, False, False, True]
    assert frame["status"].tolist() == ["unavailable"] * 3 + ["ok"]
    assert frame["reason"].tolist()[1:3] == [d.NO_CLOSE, d.NO_CLOSE]
    assert frame["total_return"].iloc[3] == pytest.approx(0.03, abs=1e-12)
    assert frame["sessions_spanned"].iloc[3] == 3  # three sessions, not one
    growth = d.growth_index(returns_of(frame), spans_of(frame))
    assert growth.iloc[0] == 10_000.0
    assert growth.iloc[1:3].isna().all()  # never filled
    assert growth.iloc[3] == pytest.approx(10_300.0, abs=1e-9)


def test_a_split_inside_a_gap_is_reconstructed_from_its_ratio():
    closes = pd.Series([400.0, 101.0], index=days("2024-01-08", "2024-01-10"))
    splits = pd.DataFrame({"effective_date": [WEEK[1]], "ratio": [4.0]})
    frame = tr(WEEK[:3], closes, splits=splits)
    assert frame["total_return"].iloc[2] == pytest.approx(0.01, abs=1e-12)
    assert frame["sessions_spanned"].iloc[2] == 2


def test_a_distribution_inside_a_gap_is_unavailable_and_stops_the_growth():
    closes = pd.Series([100.0, 99.6, 100.0], index=days("2024-01-08", "2024-01-11", "2024-01-12"))
    distributions = pd.DataFrame({"ex_date": [WEEK[1]], "amount": [1.0]})
    frame = tr(WEEK, closes, distributions)
    assert frame.loc[3, "status"] == "unavailable"
    assert frame.loc[3, "reason"] == d.DISTRIBUTION_IN_GAP  # not reinvested through the gap
    assert frame.loc[4, "status"] == "ok"  # the next session's own return exists
    growth = d.growth_index(returns_of(frame), spans_of(frame))
    assert growth.iloc[0] == 10_000.0
    assert growth.iloc[1:].isna().all()  # nothing chains through the unavailable return


# --- A4: hypothetical growth of $10,000 ---


def test_growth_of_ten_thousand_compounds_the_returns():
    returns = pd.Series([np.nan, 0.005, 0.01], index=WEEK[:3])
    spans = pd.Series([0, 1, 1], index=WEEK[:3])
    growth = d.growth_index(returns, spans)
    assert growth.tolist() == pytest.approx([10_000.0, 10_050.0, 10_150.5], abs=1e-9)
    price_only = d.growth_index(pd.Series([np.nan, -0.005, 0.01], index=WEEK[:3]), spans)
    assert price_only.iloc[-1] == pytest.approx(10_049.5, abs=1e-9)


def test_the_start_return_is_never_applied():
    returns = pd.Series([0.5, 0.01], index=WEEK[:2])
    growth = d.growth_index(returns, pd.Series([1, 1], index=WEEK[:2]))
    assert growth.tolist() == pytest.approx([10_000.0, 10_100.0], abs=1e-9)


def test_growth_refuses_a_span_over_sessions_with_a_value():
    returns = pd.Series([np.nan, 0.01, 0.02], index=WEEK[:3])
    with pytest.raises(ValueError, match="multi-session"):
        d.growth_index(returns, pd.Series([0, 1, 2], index=WEEK[:3]))


def _share_count_values(sessions, closes, distributions, splits):
    """
    Independent check by share accounting, apart from the formula: hold shares;
    at a split, multiply them; at an ex-date close, receive cash on the shares
    of the stated basis and buy fractional shares at that close. Values exist
    only at sessions with a close.
    """

    shares = d.GROWTH_START_VALUE / closes.iloc[0]
    values = {closes.index[0]: d.GROWTH_START_VALUE}
    for session in sessions[sessions > closes.index[0]]:
        before = shares
        ratio = float(splits.loc[splits["effective_date"] == session, "ratio"].prod())
        shares *= ratio
        for row in distributions[distributions["ex_date"] == session].itertuples():
            held = before if row.share_basis == d.PRE_SPLIT else shares
            shares += held * row.amount / closes[session]
        if session in closes.index:
            values[session] = shares * closes[session]
    return pd.Series(values)


@pytest.mark.parametrize("seed", range(5))
def test_growth_matches_an_independent_share_count_simulation(seed):
    rng = np.random.RandomState(seed)
    sessions = pd.bdate_range("2020-01-01", periods=300, tz="UTC")
    prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300)), index=sessions)
    ex_positions = sorted(int(i) for i in rng.choice(np.arange(5, 300, 7), size=12, replace=False))
    split_at = ex_positions[6]  # a split on an ex-date: the share basis matters
    prices[prices.index >= sessions[split_at]] /= 2.0
    bases = [d.POST_SPLIT if i == split_at else d.PRE_SPLIT for i in ex_positions]
    distributions = pd.DataFrame(
        {
            "ex_date": sessions[ex_positions],
            "amount": rng.uniform(0.1, 0.6, 12),
            "share_basis": bases,
        }
    )
    splits = pd.DataFrame({"effective_date": [sessions[split_at]], "ratio": [2.0]})
    # Gaps: drop closes on a few sessions that carry no distribution or split, and the
    # session before an ex-date, so the gap-spanning return also ends on an ex-date.
    quiet = np.setdiff1d(np.arange(1, 299), np.array([*ex_positions, split_at]))
    dropped = sorted(int(i) for i in rng.choice(quiet, size=8, replace=False))
    closes = prices.drop(sessions[dropped])
    frame = tr(sessions, closes, distributions, splits)
    growth = d.growth_index(returns_of(frame), spans_of(frame))
    expected = _share_count_values(sessions, closes, distributions, splits)
    assert growth.isna().sum() == len(dropped)  # missing sessions stay empty
    assert np.allclose(growth.dropna().to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-9)
    assert (frame["sessions_spanned"] > 1).sum() >= 1  # at least one multi-session return


def test_appending_later_records_leaves_earlier_returns_bit_identical():
    rng = np.random.RandomState(11)
    sessions = pd.bdate_range("2021-01-01", periods=120, tz="UTC")
    closes = pd.Series(50.0 * np.cumprod(1 + rng.normal(0, 0.01, 120)), index=sessions)
    distributions = pd.DataFrame({"ex_date": sessions[[20, 60, 100]], "amount": [0.2, 0.3, 0.25]})
    full = tr(sessions, closes, distributions)
    for cut in (30, 61, 99):
        prefix = tr(
            sessions[:cut],
            closes.iloc[:cut],
            distributions[distributions["ex_date"] < sessions[cut]],
        )
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut].reset_index(drop=True))


# --- A6: drawdown, recovery and history completeness ---


def growth_series(values, start="2024-01-08"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values), tz="UTC"))


def test_drawdown_example_with_dollar_size_and_recovery():
    growth = growth_series([10_000.0, 11_000.0, 9_900.0, 10_500.0, 11_100.0])
    summary = d.drawdown_summary(growth)
    assert summary.max_drawdown == pytest.approx(0.10, abs=1e-12)
    assert summary.dollar_decline == pytest.approx(1_100.0, abs=1e-9)
    assert (summary.peak_date, summary.trough_date) == (growth.index[1], growth.index[2])
    assert summary.recovery_date == growth.index[4]
    assert summary.duration_sessions == 3
    assert summary.history_complete and summary.missing_sessions == 0


def test_an_open_decline_runs_to_the_last_session_with_a_value():
    growth = growth_series([10_000.0, 11_000.0, 9_900.0, 10_500.0, np.nan])
    summary = d.drawdown_summary(growth)
    assert not summary.recovered
    assert summary.duration_sessions == 2  # peak to the last valued session, not the gap
    assert not summary.history_complete and summary.missing_sessions == 1


def test_a_gap_makes_the_drawdown_an_observed_decline_not_the_full_period_maximum():
    growth = growth_series([10_000.0, 11_000.0, np.nan, 10_500.0, 11_100.0])
    summary = d.drawdown_summary(growth)
    assert summary.max_drawdown == pytest.approx(1 - 10_500 / 11_000, abs=1e-12)
    assert not summary.history_complete  # a deeper low may have happened in the gap
    assert summary.missing_sessions == 1


def test_durations_count_every_session_including_those_without_a_value():
    growth = growth_series([1e4, 1.1e4, np.nan, np.nan, 9.9e3, np.nan, 1.11e4])
    summary = d.drawdown_summary(growth)
    assert summary.recovery_date == growth.index[6]
    assert (summary.duration_sessions, summary.duration_days) == (5, 7)


def test_no_decline_is_recovered_at_once_and_too_little_history_is_unavailable():
    summary = d.drawdown_summary(growth_series([10_000.0, 10_100.0, 10_200.0]))
    assert summary.max_drawdown == 0.0 and summary.recovered and summary.duration_sessions == 0
    with pytest.raises(d.MetricUnavailable) as caught:
        d.drawdown_summary(growth_series([10_000.0, np.nan]))
    assert caught.value.reason == d.INSUFFICIENT_HISTORY


# --- A7 and R5d: calendar-year returns ---

YEAR_ENDS = days("2022-12-30", "2023-12-29", "2024-12-31")


def calendar_growth(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=days(*values))


def test_calendar_years_run_from_year_end_close_to_year_end_close():
    growth = calendar_growth(
        {
            "2022-12-30": 10_000.0,
            "2023-06-30": 9_000.0,
            "2023-12-29": 10_800.0,
            "2024-06-28": 11_000.0,
            "2024-12-31": 10_260.0,
        }
    )
    table = d.calendar_year_returns(growth, YEAR_ENDS)
    assert table["year"].tolist() == [2023, 2024]
    assert table["total_return"].tolist() == pytest.approx([0.08, -0.05], abs=1e-12)
    assert table["losing"].tolist() == [False, True]
    assert table["partial"].tolist() == [False, False]
    assert table["status"].tolist() == ["ok", "ok"]


def test_losing_is_decided_on_the_unrounded_return_and_partial_years_are_labelled():
    growth = calendar_growth(
        {"2023-03-01": 10_000.0, "2023-12-29": 9_996.0, "2024-05-01": 10_100.0}
    )
    table = d.calendar_year_returns(growth, YEAR_ENDS)
    assert table["total_return"].iloc[0] == pytest.approx(-0.0004, abs=1e-12)
    assert table["losing"].tolist() == [True, False]  # -0.04% shows as -0.0% but lost money
    assert table["partial"].tolist() == [True, True]  # starts mid-year; ends before year end


def test_a_missing_year_end_value_makes_both_years_unavailable():
    growth = calendar_growth({"2022-12-30": 10_000.0, "2023-12-29": np.nan, "2024-12-31": 10_260.0})
    table = d.calendar_year_returns(growth, YEAR_ENDS)
    assert table["status"].tolist() == ["unavailable", "unavailable"]
    assert table["reason"].tolist() == [d.YEAR_END_MISSING] * 2
    assert table["total_return"].isna().all() and not table["losing"].any()


def test_a_period_ending_in_a_gap_is_unavailable_at_its_edge_not_at_a_year_end():
    growth = calendar_growth({"2022-12-30": 10_000.0, "2023-06-30": 10_500.0, "2023-07-03": np.nan})
    table = d.calendar_year_returns(growth, YEAR_ENDS)
    assert table["status"].tolist() == ["unavailable"]
    assert table["reason"].tolist() == [d.PERIOD_END_MISSING]
    assert table["partial"].tolist() == [True]


def test_full_calendar_years_compound_to_the_period_growth():
    rng = np.random.RandomState(7)
    sessions = pd.bdate_range("2019-12-31", "2024-12-31", tz="UTC")
    returns = pd.Series(rng.normal(0.0003, 0.01, len(sessions)), index=sessions)
    growth = d.growth_index(returns, pd.Series(1, index=sessions))
    year_ends = pd.DatetimeIndex(sessions.to_series().groupby(sessions.year).max().to_numpy())
    table = d.calendar_year_returns(growth, year_ends)
    assert not table["partial"].any()
    assert np.prod(1 + table["total_return"]) == pytest.approx(growth.iloc[-1] / growth.iloc[0])


# --- R4: the shared start ---


def test_the_shared_start_is_the_first_common_close_after_the_latest_first_close():
    a = days("2024-01-03", "2024-01-04", "2024-01-09", "2024-01-10")
    b = days("2024-01-09", "2024-01-10")
    assert d.common_start({"a": a, "b": b}, utc("2024-01-01")) == utc("2024-01-09")
    a_gap = days("2024-01-03", "2024-01-04", "2024-01-10")
    assert d.common_start({"a": a_gap, "b": b}, utc("2024-01-01")) == utc("2024-01-10")
    assert d.common_start({"a": a, "b": b}, utc("2024-01-10")) == utc("2024-01-10")
    assert d.common_start({"a": a, "b": days("2024-01-11")}, utc("2024-01-01")) is None
    assert d.common_start({}, utc("2024-01-01")) is None


def test_the_calculator_implements_the_explore_interface():
    assert isinstance(d.DescriptiveCalculator(), DescriptiveMetrics)
