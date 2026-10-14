"""
The Portfolio Lab's pure layers: the FundHistory contract, the synthetic
demo funds, the view logic (periods, gaps, multi-session returns, breaks,
provider calls) and display formatting. No Streamlit.

Metric values in the view tests come from a fake provider with hand-written
results, so the view's checks are tested apart from the approved
calculations (tested in `tests/test_portfolio_descriptive.py`).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from stock_agent.app import figures, labels, synthetic, view
from stock_agent.app.history import RECORD_COLUMNS, FundHistory
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.portfolio.descriptive import (
    CALENDAR_COLUMNS,
    GROWTH_START_VALUE,
    DescriptiveCalculator,
    DrawdownSummary,
    MetricUnavailable,
)
from stock_agent.ui import theme

APP = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "app"


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def fund(
    fund_id: str = "fund-a",
    dates: tuple[str, ...] = ("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"),
    returns: tuple[float, ...] | None = None,
    missing: tuple[int, ...] = (),
    **fields,
) -> FundHistory:
    """
    A small hand-built history. Positions in `missing` have no close; the
    first close has no prior close; every later close carries its return
    from `returns` over the sessions since the previous close.
    """

    values = list(returns) if returns is not None else [0.01] * len(dates)
    rows, previous = [], None
    for i, text in enumerate(dates):
        if i in missing:
            rows.append((utc(text), False, np.nan, 0, "unavailable", "missing_input"))
        elif previous is None:
            rows.append((utc(text), True, np.nan, 0, "unavailable", "no_prior_close"))
            previous = i
        else:
            rows.append((utc(text), True, values[i], i - previous, "ok", None))
            previous = i
    records = pd.DataFrame(rows, columns=list(RECORD_COLUMNS))
    records["date"] = pd.DatetimeIndex(records["date"])
    defaults = {
        "fund_id": fund_id,
        "name": f"Fund {fund_id}",
        "role": "equity",
        "records": records,
        "calendar": pd.DatetimeIndex([utc(text) for text in dates]),
        "source_label": "test source",
        "retrieved_label": "test retrieval",
        "vintage_note": "test vintage",
        "limitations": ("synthetic_data",),
    }
    return FundHistory(**{**defaults, **fields})


DRAWDOWN = DrawdownSummary(
    max_drawdown=0.25,
    peak_date=utc("2021-01-05"),
    trough_date=utc("2021-01-06"),
    recovery_date=None,
    duration_sessions=2,
    duration_days=2,
    dollar_decline=2_750.0,
    history_complete=True,
    missing_sessions=0,
)
CALENDAR = pd.DataFrame(
    {
        "year": [2021],
        "total_return": [-0.0004],
        "partial": [True],
        "losing": [True],
        "status": ["ok"],
        "reason": [None],
    }
)


class RecordingMetrics:
    """
    Hand-written results that honour the contract; records what the view
    passes in. Growth is 11,000 after t0 and NaN wherever the return is NaN;
    the drawdown dates are moved onto the series' first and last valued
    sessions so they always exist. No formula is computed.
    """

    def __init__(self, start: pd.Timestamp | None, unavailable: dict[str, str] | None = None):
        self.start = start
        self.unavailable = unavailable or {}
        self.calls: list[tuple] = []

    def common_start(self, available_sessions, window_start):
        self.calls.append(("common_start", available_sessions, window_start))
        return self.start

    def growth_index(self, returns, sessions_spanned, start_value=GROWTH_START_VALUE):
        self.calls.append(("growth_index", returns, sessions_spanned))
        if returns.name in self.unavailable:
            reason = self.unavailable[returns.name]
            raise MetricUnavailable("growth_index", reason, "too few sessions")
        values = np.full(len(returns), 11_000.0)
        values[returns.isna().to_numpy()] = np.nan
        values[0] = start_value
        return pd.Series(values, index=returns.index)

    def drawdown_summary(self, growth):
        valued = growth.dropna().index
        return dataclasses.replace(DRAWDOWN, peak_date=valued[0], trough_date=valued[-1])

    def calendar_year_returns(self, growth, year_end_sessions):
        self.calls.append(("calendar_year_returns", growth, year_end_sessions))
        return CALENDAR.copy()


# --- the FundHistory contract ---


def test_coverage_is_derived_from_the_sessions_with_a_close():
    history = fund(missing=(0, 3))
    assert history.coverage_start == utc("2021-01-05")
    assert history.coverage_end == utc("2021-01-06")
    empty = fund(missing=(0, 1, 2, 3))
    assert empty.coverage_start is None and empty.coverage_end is None
    assert list(history.available_sessions) == [utc("2021-01-05"), utc("2021-01-06")]


def test_year_end_sessions_come_from_the_calendar_not_the_records():
    stale = synthetic.demo_histories()[3]
    ends = stale.year_end_sessions
    assert ends[-1] == utc("2024-12-31")  # the calendar runs on past the stale records
    assert utc("2023-12-29") in ends and utc("2023-06-30") not in ends


def test_between_keeps_no_record_after_the_end():
    history = fund().between(utc("2021-01-05"), utc("2021-01-06"))
    assert history.records["date"].tolist() == [utc("2021-01-05"), utc("2021-01-06")]
    assert history.name == "Fund fund-a" and history.limitations == ("synthetic_data",)


def _records(**changes) -> pd.DataFrame:
    records = fund().records
    for column, values in changes.items():
        records[column] = values
    return records


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": "crypto"}, "role must be"),
        ({"stale": True}, "stale_reason"),
        ({"stale_reason": "old"}, "stale_reason"),
        ({"records": fund().records.drop(columns="reason")}, "record columns"),
        ({"records": _records(date=pd.date_range("2021-01-04", periods=4))}, "UTC"),
        (
            {"records": _records(date=pd.date_range("2021-01-04 12:00", periods=4, tz="UTC"))},
            "00:00 UTC",
        ),
        ({"records": fund().records.iloc[::-1]}, "strictly increasing"),
        ({"records": _records(close_observed=[1, 1, 1, 1])}, "boolean"),
        ({"records": _records(status=["unavailable", "ok", "ok", "stale"])}, "status"),
        ({"records": _records(total_return=[np.nan, np.nan, 0.01, 0.01])}, "finite"),
        ({"records": _records(total_return=[0.0, 0.01, 0.01, 0.01])}, "NaN total_return"),
        ({"records": _records(sessions_spanned=[0, 0, 1, 1])}, "at least one session"),
        ({"records": _records(reason=[None, "x", None, None])}, "carries no reason"),
        ({"records": _records(reason=[None, None, None, None])}, "machine-readable reason"),
        ({"records": _records(sessions_spanned=[0, 1, 2, 1])}, "without a close"),
        (
            {
                "records": fund(missing=(1,)).records.assign(
                    close_observed=[True, False, True, False]
                )
            },
            "observed close",
        ),
        ({"calendar": pd.DatetimeIndex([utc("2021-01-04"), utc("2021-01-07")])}, "contiguous"),
    ],
)
def test_a_malformed_history_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        fund(**change)


def test_a_multi_session_return_must_start_at_a_close():
    records = fund(missing=(1,)).records
    records.loc[2, "sessions_spanned"] = 1  # claims one session, but its origin has no close
    with pytest.raises(ValueError, match="starts at a session with a close"):
        fund(records=records)


def test_limitations_must_be_a_tuple():
    with pytest.raises(TypeError):
        fund(limitations=["synthetic_data"])


def test_evidence_status_is_a_constant_not_a_field():
    assert FundHistory.evidence_status == fund().evidence_status == "descriptive"
    assert "evidence_status" not in {field.name for field in dataclasses.fields(FundHistory)}
    with pytest.raises(TypeError):
        fund(evidence_status="rules_illustration")


def test_a_calendar_that_stops_mid_year_has_no_year_end_for_that_year():
    dates = tuple(str(day.date()) for day in pd.bdate_range("2023-12-27", "2024-05-02"))
    assert list(fund(dates=dates).year_end_sessions) == [utc("2023-12-29")]


# --- the synthetic demo funds ---


def test_the_demo_funds_are_deterministic_and_seeded():
    first, second = synthetic.demo_histories(), synthetic.demo_histories()
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left.records, right.records)
    other = synthetic.demo_histories(seed=1)
    assert not np.allclose(
        first[1].records["total_return"].iloc[1:], other[1].records["total_return"].iloc[1:]
    )


def test_the_demo_funds_are_named_and_labelled_as_synthetic():
    funds = synthetic.demo_histories()
    assert [f.name for f in funds[:3]] == [
        "Synthetic equity fund",
        "Synthetic bond fund",
        "Synthetic cash fund",
    ]
    assert [f.role for f in funds[:3]] == ["equity", "bonds", "cash proxy"]
    for history in funds:
        assert history.source_label == "Synthetic demo data, not real fund performance"
        assert "synthetic_data" in history.limitations
        assert history.evidence_status == "descriptive"
        assert str(synthetic.SEED) in history.vintage_note
    assert labels.LIMITATION_TEXT["synthetic_data"].startswith(
        "Synthetic demo data, not real fund performance"
    )


def test_the_demo_funds_cover_2007_to_2024_weekdays_before_the_lockbox():
    for history in synthetic.demo_histories():
        dates = history.records["date"]
        assert dates.iloc[0] == utc("2007-01-01")
        assert (dates.dt.dayofweek < 5).all()
        assert dates.iloc[-1] < MODEL_HOLDOUT_START
    assert synthetic.demo_histories()[0].records["date"].iloc[-1] == utc("2024-12-31")


def test_the_demo_funds_include_gaps_a_break_a_stale_variant_and_an_unavailable_fund():
    equity, bond, cash, stale, distribution_gap, missing = synthetic.demo_histories()
    gap = equity.records[~equity.records["close_observed"]]
    assert gap["date"].tolist() == list(pd.bdate_range("2015-08-17", "2015-08-21", tz="UTC"))
    assert set(gap["reason"]) == {"missing_input"} and gap["total_return"].isna().all()
    assert view.gaps(equity) == (
        view.Gap(utc("2015-08-17"), utc("2015-08-21"), 5, "missing_input"),
    )
    (span,) = view.spans(equity)
    assert (span.date, span.sessions) == (utc("2015-08-24"), 6)  # one six-session return
    assert view.breaks(equity) == ()
    assert view.breaks(distribution_gap) == (view.Break(utc("2019-03-08"), "distribution_in_gap"),)
    assert not bond.stale and not cash.stale
    assert stale.stale and stale.stale_reason and stale.coverage_end == utc("2023-06-30")
    kept = bond.records[bond.records["date"] <= utc("2023-06-30")].reset_index(drop=True)
    pd.testing.assert_frame_equal(stale.records, kept)
    assert missing.coverage_start is None
    assert (missing.records["status"] == "unavailable").all()


def test_the_synthetic_returns_recompute_from_their_price_paths():
    # The demo records are total returns of a raw price path with distributions and a
    # 2-for-1 split: every one-session return equals the draw it was built from.
    rng = np.random.RandomState(synthetic.SEED)
    dates = synthetic.sessions()
    draws = synthetic._total_return_draws(rng, dates, "equity")
    equity = synthetic.demo_histories()[0].records
    one_session = equity["status"].eq("ok") & equity["sessions_spanned"].eq(1)
    expected = draws[one_session.to_numpy()]
    assert np.allclose(equity.loc[one_session, "total_return"], expected, rtol=0, atol=1e-12)


def test_the_generator_does_no_io():
    tree = ast.parse((APP / "synthetic.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported == {
        "__future__",
        "functools",
        "numpy",
        "pandas",
        "stock_agent.app.history",
        "stock_agent.portfolio.descriptive",
    }
    source = (APP / "synthetic.py").read_text(encoding="utf-8")
    for call in ("open(", "now(", "today(", "read_", "to_parquet", "to_csv"):
        assert call not in source, call


# --- the drawdown summary type ---


def test_recovered_is_derived_from_the_recovery_date():
    assert DRAWDOWN.recovered is False
    assert dataclasses.replace(DRAWDOWN, recovery_date=utc("2021-01-07")).recovered is True
    assert "recovered" not in {field.name for field in dataclasses.fields(DrawdownSummary)}


def test_a_drawdown_summary_must_be_consistent():
    with pytest.raises(ValueError, match="fraction"):
        dataclasses.replace(DRAWDOWN, max_drawdown=1.5)
    with pytest.raises(ValueError, match="recovery cannot precede"):
        dataclasses.replace(DRAWDOWN, recovery_date=utc("2021-01-05"))
    with pytest.raises(ValueError, match="trough"):
        dataclasses.replace(DRAWDOWN, trough_date=utc("2021-01-04"))
    with pytest.raises(ValueError, match="non-negative"):
        dataclasses.replace(DRAWDOWN, dollar_decline=-1.0)
    with pytest.raises(ValueError, match="complete exactly when"):
        dataclasses.replace(DRAWDOWN, missing_sessions=2)


# --- periods ---

COVERAGE = (utc("2007-01-03"), utc("2024-12-31"))


def test_a_period_inside_the_coverage_is_not_clamped():
    period = view.clamp_period(2010, 2012, COVERAGE, "2010-2012")
    assert (period.start, period.end, period.clamped) == (
        utc("2010-01-01"),
        utc("2012-12-31"),
        False,
    )


def test_a_period_is_clamped_to_the_coverage():
    period = view.clamp_period(2005, 2008, COVERAGE, "custom")
    assert (period.start, period.end, period.clamped) == (
        utc("2007-01-03"),
        utc("2008-12-31"),
        True,
    )
    late = view.clamp_period(2024, 2024, (utc("2007-01-03"), utc("2024-06-28")), "2024")
    assert late.end == utc("2024-06-28") and late.clamped


def test_a_period_outside_the_coverage_is_none():
    assert view.clamp_period(2008, 2009, (utc("2010-01-04"), utc("2024-12-31")), "x") is None
    with pytest.raises(ValueError):
        view.clamp_period(2012, 2010, COVERAGE, "x")


def test_presets_ask_for_fixed_calendar_years():
    assert view.preset_years(view.LAST_10, COVERAGE) == (2015, 2024)
    assert view.preset_years(view.CRISIS_2008, COVERAGE) == (2007, 2009)
    assert view.preset_years(view.YEAR_2022, COVERAGE) == (2022, 2022)
    stale_coverage = (utc("2007-01-03"), utc("2023-06-30"))
    assert view.preset_years(view.LAST_10, stale_coverage) == (2014, 2023)
    full = view.preset_period(view.FULL, COVERAGE)
    assert (full.start, full.end, full.clamped) == (*COVERAGE, False)
    assert view.preset_period(view.YEAR_2022, (utc("2010-01-04"), utc("2021-12-31"))) is None
    with pytest.raises(ValueError):
        view.preset_years(view.CUSTOM, COVERAGE)


def test_shared_coverage_is_the_intersection():
    early = fund("early", dates=("2021-01-04", "2021-01-05", "2021-01-06"))
    late = fund("late", dates=("2021-01-05", "2021-01-06", "2021-01-07"))
    assert view.shared_coverage([early, late]) == (utc("2021-01-05"), utc("2021-01-06"))
    disjoint = fund("disjoint", dates=("2021-02-01",))
    assert view.shared_coverage([early, disjoint]) is None
    assert view.shared_coverage([]) is None


def test_gaps_are_runs_of_sessions_without_a_close_and_spans_are_labelled():
    history = fund(
        dates=("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"),
        returns=(0.0, 0.0, 0.0, 0.03, 0.01),
        missing=(1, 2),
    )
    assert view.gaps(history) == (
        view.Gap(utc("2021-01-05"), utc("2021-01-06"), 2, "missing_input"),
    )
    assert view.spans(history) == (view.Span(utc("2021-01-07"), 0.03, 3),)
    assert view.gaps(fund()) == () and view.spans(fund()) == ()
    assert view.breaks(history) == ()  # a fund's first close is not a break


# --- the provider calls ---

PERIOD = view.Period(utc("2021-01-04"), utc("2021-01-06"), "test", clamped=False)


def test_the_common_start_is_delegated_to_the_provider():
    metrics = RecordingMetrics(start=utc("2021-01-05"))
    late = fund("b", missing=(0,))  # first close 2021-01-05
    result = view.build_view([fund("a"), late], PERIOD, metrics)
    assert result.start == utc("2021-01-05")
    _, passed, window_start = metrics.calls[0]
    assert window_start == PERIOD.start
    # The provider gets each fund's available sessions, and none after the period's end.
    assert list(passed) == ["a", "b"]
    assert list(passed["a"]) == [utc("2021-01-04"), utc("2021-01-05"), utc("2021-01-06")]
    assert list(passed["b"]) == [utc("2021-01-05"), utc("2021-01-06")]
    growth_inputs = [call[1] for call in metrics.calls if call[0] == "growth_index"]
    assert [series.index[0] for series in growth_inputs] == [utc("2021-01-05")] * 2
    assert all(series.index[-1] == PERIOD.end for series in growth_inputs)
    calendar_inputs = [call[2] for call in metrics.calls if call[0] == "calendar_year_returns"]
    # The test calendar stops mid-year (2021-01-07), so it has no year end to offer.
    assert calendar_inputs and all(len(ends) == 0 for ends in calendar_inputs)
    assert result.funds[0].drawdown.max_drawdown == DRAWDOWN.max_drawdown
    pd.testing.assert_frame_equal(result.funds[0].calendar, CALENDAR)


def test_gaps_reach_the_provider_as_nan_with_their_spans_and_stay_nan_in_growth():
    metrics = RecordingMetrics(start=utc("2021-01-04"))
    result = view.build_view([fund(missing=(1,))], PERIOD, metrics)
    _, returns, spans = next(call for call in metrics.calls if call[0] == "growth_index")
    assert returns.isna().tolist() == [True, True, False]  # no prior close; no close
    assert spans.tolist() == [0, 0, 2]  # the 2021-01-06 return covers two sessions
    assert result.funds[0].growth.isna().tolist() == [False, True, False]


def test_an_unavailable_input_makes_its_derived_metrics_unavailable():
    metrics = RecordingMetrics(start=utc("2021-01-04"), unavailable={"b": "too_few_sessions"})
    result = view.build_view([fund("a"), fund("b")], PERIOD, metrics)
    available, missing = result.funds
    assert isinstance(available.growth, pd.Series)
    expected = view.Unavailable("too_few_sessions", "too few sessions")
    assert missing.growth == missing.drawdown == missing.calendar == expected


def test_a_break_at_or_before_the_start_is_not_reported():
    gap_fund = synthetic.demo_histories()[4]  # its break is dated 2019-03-08
    period = view.Period(utc("2019-03-04"), utc("2019-12-31"), "from inside the gap", False)
    result = view.build_view([gap_fund], period, DescriptiveCalculator())
    assert result.start == utc("2019-03-08")  # growth starts at the break's close
    assert result.funds[0].breaks == ()
    assert result.funds[0].growth.notna().all()
    full = view.Period(utc("2019-01-01"), utc("2019-12-31"), "2019", False)
    assert view.build_view([gap_fund], full, DescriptiveCalculator()).funds[0].breaks == (
        view.Break(utc("2019-03-08"), "distribution_in_gap"),
    )


def test_no_common_start_is_an_unavailable_state():
    result = view.build_view([fund()], PERIOD, RecordingMetrics(start=None))
    assert result.start == view.Unavailable("no_common_start", labels.NO_COMMON_START)
    assert result.funds[0].growth == result.start


class BrokenMetrics(RecordingMetrics):
    """The compliant fake with one contract violation, named by `broken`."""

    def __init__(self, broken: str):
        super().__init__(start=utc("2021-01-04"))
        self.broken = broken

    def common_start(self, available_sessions, window_start):
        starts = {"start_outside": utc("2021-01-07"), "start_in_gap": utc("2021-01-06")}
        return starts.get(self.broken, self.start)

    def growth_index(self, returns, sessions_spanned, start_value=GROWTH_START_VALUE):
        growth = super().growth_index(returns, sessions_spanned, start_value)
        changed = {
            "index": lambda: growth.iloc[:-1],
            "start_value": lambda: growth * 2,
            "negative": lambda: growth.where(growth.index != utc("2021-01-05"), -1.0),
            "infinite": lambda: growth.where(growth.index != utc("2021-01-05"), np.inf),
            "fills_gap": lambda: growth.fillna(11_000.0),
        }
        return changed.get(self.broken, lambda: growth)()

    def drawdown_summary(self, growth):
        summary = super().drawdown_summary(growth)
        if self.broken == "drawdown_type":
            return dataclasses.asdict(summary)
        if self.broken == "drawdown_outside":
            return dataclasses.replace(summary, recovery_date=utc("2021-01-08"))
        if self.broken == "drawdown_in_gap":
            return dataclasses.replace(summary, trough_date=utc("2021-01-06"))
        return summary

    def calendar_year_returns(self, growth, year_end_sessions):
        changed = {
            "columns": lambda: CALENDAR.drop(columns="status"),
            "duplicate_years": lambda: pd.concat([CALENDAR, CALENDAR]),
            "year_outside": lambda: CALENDAR.assign(year=2020),
            "losing_flag": lambda: CALENDAR.assign(losing=False),
            "zero_is_losing": lambda: CALENDAR.assign(total_return=0.0),
            "unavailable_with_value": lambda: CALENDAR.assign(status="unavailable"),
            "available_without_value": lambda: CALENDAR.assign(total_return=np.nan),
        }
        return changed.get(self.broken, CALENDAR.copy)()


# The fund has no close on 2021-01-06, so the period 2021-01-04..06 ends in a gap.
GAPPED = {"missing": (2,)}


def test_the_compliant_fake_passes_every_check():
    result = view.build_view([fund(**GAPPED)], PERIOD, BrokenMetrics("none"))
    assert result.funds[0].growth.isna().tolist() == [False, False, True]
    # The real calculator passes the same checks.
    real = view.build_view([fund(**GAPPED)], PERIOD, DescriptiveCalculator())
    assert real.funds[0].growth.tolist()[:2] == pytest.approx([10_000.0, 10_100.0])


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ("start_outside", "outside the period"),
        ("start_in_gap", "without a close"),
        ("index", "session dates"),
        ("start_value", "start at the start value"),
        ("negative", "finite and positive"),
        ("infinite", "finite and positive"),
        ("fills_gap", "filled a gap"),
        ("drawdown_outside", "drawdown dates"),
        ("drawdown_in_gap", "drawdown dates"),
        ("columns", "columns"),
        ("duplicate_years", "one row per year"),
        ("year_outside", "outside the growth series"),
        ("losing_flag", "losing must equal"),
        ("zero_is_losing", "losing must equal"),
        ("unavailable_with_value", "finite exactly when available"),
        ("available_without_value", "finite exactly when available"),
    ],
)
def test_a_provider_breaking_the_contract_is_an_error_not_a_state(broken, message):
    with pytest.raises(ValueError, match=message):
        view.build_view([fund(**GAPPED)], PERIOD, BrokenMetrics(broken))


def test_a_drawdown_result_of_the_wrong_type_is_a_type_error():
    with pytest.raises(TypeError, match="must return a DrawdownSummary"):
        view.build_view([fund(**GAPPED)], PERIOD, BrokenMetrics("drawdown_type"))


# --- formatting and tables ---


def test_numbers_are_formatted_for_reading():
    assert view.money(12_345.6) == "$12,346"
    assert view.money(-1_234.0) == "−$1,234"
    assert view.money(10_000.0) == "$10,000"
    assert view.percent(0.3456) == "34.6%"
    assert view.percent(0.08, signed=True) == "+8.0%"
    assert view.percent(-0.05, signed=True) == "−5.0%"
    # Rounding never hides the sign of a loss; an exact zero is not a loss.
    assert view.percent(-0.0004, signed=True) == "−0.0%"
    assert view.percent(0.0, signed=True) == view.percent(-0.0, signed=True) == "+0.0%"
    assert view.day(utc("2021-01-04")) == "2021-01-04"


def test_losing_years_come_from_the_provider_flag_not_the_rounded_text():
    table = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024],
            "total_return": [-0.0004, 0.0, 0.08, np.nan],
            "partial": [True, False, False, False],
            "losing": [True, False, False, False],
            "status": ["ok", "ok", "ok", "unavailable"],
            "reason": [None, None, None, "year_end_value_unavailable"],
        }
    )
    text = view.calendar_table([("Fund A", table)])
    assert text["Year"].tolist() == [2021, 2022, 2023, 2024]
    assert text["Fund A"].tolist() == [
        "−0.0% · lost money · partial year",
        "+0.0%",
        "+8.0%",
        "unavailable",
    ]
    assert view.losing_summary("Fund A", table) == (
        "Fund A: lost money in 2021: 1 of 3 calendar years shown (1 partial; 1 unavailable)."
    )
    none_lost = table.assign(losing=False)
    assert view.losing_summary("Fund A", none_lost) == (
        "Fund A: no calendar year shown lost money (0 of 3; 1 partial; 1 unavailable)."
    )


def test_the_growth_table_lists_the_start_and_each_year_end():
    index = pd.DatetimeIndex([utc(d) for d in ("2020-12-30", "2020-12-31", "2021-01-04")])
    growth = pd.Series([10_000.0, 10_100.0, np.nan], index=index)
    table = view.year_end_table([("Fund A", growth)])
    assert table.to_dict("list") == {
        "Session date": ["2020-12-30", "2020-12-31", "2021-01-04"],
        "Fund A": ["$10,000", "$10,100", "—"],
    }
    assert view.last_value(growth) == (utc("2020-12-31"), 10_100.0)


def test_the_required_honest_labels_are_worded_exactly():
    assert labels.GROWTH_TITLE == "Hypothetical growth of $10,000 — distributions reinvested"
    assert labels.HISTORICAL_BADGE == "Historical market data"
    assert labels.HONEST_LABELS == (
        (
            "Distributions reinvested on the ex-date: a modeling convention, not actual "
            "investor cash flows."
        ),
        "Before taxes and transaction costs.",
        "Fund expenses are already reflected in fund prices and are not deducted again.",
    )
    assert labels.SYNTHETIC_NOTICE.startswith("Synthetic demo data, not real fund performance")


def test_user_facing_copy_names_no_decision_code_and_translates_reason_codes():
    shown = [
        value for name, value in vars(labels).items() if name.isupper() and isinstance(value, str)
    ]
    shown += [*labels.REASON_TEXT.values(), *labels.LIMITATION_TEXT.values()]
    assert not [text for text in shown if "OD8" in text]
    assert labels.reason_text("missing_input") == "no close from the source"
    assert labels.reason_text("late_vendor_file") == "late vendor file"  # unknown: never hidden
    assert labels.FRACTIONAL_SHARES == "Fractional shares assumed."


def test_the_copy_has_no_trading_or_advice_wording():
    text = " ".join(
        value if isinstance(value, str) else " ".join(map(str, value))
        for name, value in vars(labels).items()
        if name.isupper() and isinstance(value, str | tuple)
    ).lower()
    for word in (" buy", " sell", "recommend", "should invest", "outperform"):
        assert word not in text, word


# --- figures ---


def test_fund_identities_are_colour_blind_safe_and_never_colour_alone():
    for colour in figures.FUND_COLOURS:
        assert theme.contrast_ratio(colour, theme.COLOURS["surface"]) >= 3.0, colour
    assert len(set(figures.FUND_COLOURS)) == len(figures.FUND_COLOURS) == 5
    for palette in (figures.FUND_MARKERS, figures.FUND_DASHES):
        assert len(set(palette)) == len(palette) == 6
    styles = figures.fund_styles(synthetic.demo_histories())
    assert len({(s.colour, s.marker, s.dash) for s in styles.values()}) == 6


def test_the_growth_figure_is_log_scaled_labelled_and_keeps_gaps():
    history = fund()
    index = pd.DatetimeIndex([utc(d) for d in ("2021-01-04", "2021-01-05", "2021-01-06")])
    growth = pd.Series([10_000.0, np.nan, 10_200.0], index=index)
    styles = figures.fund_styles([history])
    figure = figures.growth_figure([(history, growth)], styles, start_value=GROWTH_START_VALUE)
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == labels.GROWTH_TITLE
    assert figure.layout.yaxis.type == "log" and "log scale" in figure.layout.yaxis.title.text
    trace = figure.data[0]
    assert trace.connectgaps is False and np.isnan(trace.y[1])
    assert trace.marker.symbol == "circle" and trace.line.dash == "solid"
    assert figure.layout.shapes[0].y0 == GROWTH_START_VALUE  # the $10,000 reference line


def test_the_calendar_figure_hatches_partial_years_on_a_zero_line():
    history = fund()
    table = pd.DataFrame(
        {
            "year": [2021, 2022],
            "total_return": [-0.05, 0.08],
            "partial": [True, False],
            "losing": [True, False],
            "status": ["ok", "ok"],
            "reason": [None, None],
        }
    )
    figure = figures.calendar_year_figure([(history, table)], figures.fund_styles([history]))
    bar = figure.data[0]
    assert list(bar.x) == ["2021", "2022"]
    assert list(bar.marker.pattern.shape) == ["/", ""]
    assert "lost money" in bar.customdata[0] and "partial" in bar.customdata[0]
    assert figure.layout.yaxis.zeroline is True
    assert set(CALENDAR_COLUMNS) == {
        "year",
        "total_return",
        "partial",
        "losing",
        "status",
        "reason",
    }
