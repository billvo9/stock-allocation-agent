"""
The Portfolio Lab's pure layers: the FundHistory contract, the synthetic
demo funds, the provider protocol and its pending implementation, the view
logic (periods, gaps, provider calls) and display formatting. No Streamlit.

Every metric value here comes from a fake provider with hand-written
results: the descriptive formulas (OD8 group A) are not approved, so nothing
in the app computes them.
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
from stock_agent.app.history import FundHistory
from stock_agent.app.provider import (
    PENDING_DETAIL,
    PENDING_REASON,
    DescriptiveMetrics,
    PendingMetrics,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.portfolio.descriptive import (
    CALENDAR_COLUMNS,
    GROWTH_START_VALUE,
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
    reasons: dict[int, str] | None = None,
    **fields,
) -> FundHistory:
    """A small hand-built history; `reasons` marks positions unavailable."""

    reasons = reasons or {}
    values = list(returns) if returns is not None else [0.01] * len(dates)
    records = pd.DataFrame(
        {
            "date": pd.DatetimeIndex([utc(d) for d in dates]),
            "daily_total_return": [
                np.nan if i in reasons else value for i, value in enumerate(values)
            ],
            "status": ["unavailable" if i in reasons else "ok" for i in range(len(dates))],
            "reason": [reasons.get(i) for i in range(len(dates))],
        }
    )
    defaults = {
        "fund_id": fund_id,
        "name": f"Fund {fund_id}",
        "role": "equity",
        "records": records,
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
)
CALENDAR = pd.DataFrame(
    {"year": [2021], "total_return": [-0.0004], "partial": [True], "losing": [True]}
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

    def growth_index(self, daily_total_returns, start_value=GROWTH_START_VALUE):
        self.calls.append(("growth_index", daily_total_returns))
        if daily_total_returns.name in self.unavailable:
            reason = self.unavailable[daily_total_returns.name]
            raise MetricUnavailable("growth_index", reason, "too few sessions")
        values = np.full(len(daily_total_returns), 11_000.0)
        values[daily_total_returns.isna().to_numpy()] = np.nan
        values[0] = start_value
        return pd.Series(values, index=daily_total_returns.index)

    def drawdown_summary(self, growth):
        valued = growth.dropna().index
        return dataclasses.replace(DRAWDOWN, peak_date=valued[0], trough_date=valued[-1])

    def calendar_year_returns(self, growth):
        return CALENDAR.copy()


# --- the FundHistory contract ---


def test_coverage_is_derived_from_the_available_records():
    history = fund(reasons={0: "missing_input", 3: "missing_input"})
    assert history.coverage_start == utc("2021-01-05")
    assert history.coverage_end == utc("2021-01-06")
    empty = fund(reasons={0: "missing_input", 1: "x", 2: "x", 3: "x"})
    assert empty.coverage_start is None and empty.coverage_end is None


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
        ({"records": _records(status=["ok", "ok", "ok", "stale"])}, "status"),
        ({"records": _records(daily_total_return=[0.01, np.nan, 0.01, 0.01])}, "finite"),
        (
            {"records": _records(status=["ok", "unavailable", "ok", "ok"])},
            "NaN daily_total_return",
        ),
        ({"records": _records(reason=["x", None, None, None])}, "carries no reason"),
    ],
)
def test_a_malformed_history_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        fund(**change)


def test_an_unavailable_record_needs_a_reason():
    records = _records(
        status=["ok", "unavailable", "ok", "ok"],
        daily_total_return=[0.01, np.nan, 0.01, 0.01],
    )
    with pytest.raises(ValueError, match="needs a machine-readable reason"):
        fund(records=records)


def test_limitations_must_be_a_tuple():
    with pytest.raises(TypeError):
        fund(limitations=["synthetic_data"])


def test_evidence_status_is_a_constant_not_a_field():
    assert FundHistory.evidence_status == fund().evidence_status == "descriptive"
    assert "evidence_status" not in {field.name for field in dataclasses.fields(FundHistory)}
    with pytest.raises(TypeError):
        fund(evidence_status="rules_illustration")


# --- the synthetic demo funds ---


def test_the_demo_funds_are_deterministic_and_seeded():
    first, second = synthetic.demo_histories(), synthetic.demo_histories()
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left.records, right.records)
    other = synthetic.demo_histories(seed=1)
    assert not np.allclose(
        first[0].available["daily_total_return"], other[0].available["daily_total_return"]
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


def test_the_demo_funds_include_a_gap_a_stale_variant_and_an_unavailable_fund():
    equity, bond, cash, stale, missing = synthetic.demo_histories()
    gap = equity.records[equity.records["status"] == "unavailable"]
    assert gap["date"].tolist() == list(pd.bdate_range("2015-08-17", "2015-08-21", tz="UTC"))
    assert set(gap["reason"]) == {"missing_input"} and gap["daily_total_return"].isna().all()
    assert view.gaps(equity) == (
        view.Gap(utc("2015-08-17"), utc("2015-08-21"), 5, "missing_input"),
    )
    assert not bond.stale and not cash.stale
    assert stale.stale and stale.stale_reason and stale.coverage_end == utc("2023-06-30")
    kept = bond.records[bond.records["date"] <= utc("2023-06-30")].reset_index(drop=True)
    pd.testing.assert_frame_equal(stale.records, kept)
    assert missing.coverage_start is None
    assert (missing.records["status"] == "unavailable").all()


def test_the_generator_does_no_io():
    tree = ast.parse((APP / "synthetic.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported == {"__future__", "numpy", "pandas", "stock_agent.app.history"}
    source = (APP / "synthetic.py").read_text(encoding="utf-8")
    for call in ("open(", "now(", "today(", "read_", "to_parquet", "to_csv"):
        assert call not in source, call


# --- the provider protocol ---


def test_the_pending_provider_marks_every_metric_unavailable():
    pending = PendingMetrics()
    assert isinstance(pending, DescriptiveMetrics)
    returns = pd.Series([0.0], index=pd.DatetimeIndex([utc("2021-01-04")]))
    calls = [
        lambda: pending.common_start({"fund-a": fund().available_sessions}, utc("2021-01-04")),
        lambda: pending.growth_index(returns),
        lambda: pending.drawdown_summary(returns),
        lambda: pending.calendar_year_returns(returns),
    ]
    for call in calls:
        with pytest.raises(MetricUnavailable) as caught:
            call()
        # The machine reason keeps the decision code; the words do not.
        assert caught.value.reason == PENDING_REASON == "pending_od8_group_a"
        assert caught.value.detail == PENDING_DETAIL
    assert PENDING_DETAIL == "awaiting the project owner's approval of the descriptive formulas"


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


def test_gaps_are_runs_of_one_reason():
    history = fund(
        dates=("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"),
        reasons={1: "missing_input", 2: "missing_input", 3: "late_vendor_file"},
    )
    assert view.gaps(history) == (
        view.Gap(utc("2021-01-05"), utc("2021-01-06"), 2, "missing_input"),
        view.Gap(utc("2021-01-07"), utc("2021-01-07"), 1, "late_vendor_file"),
    )
    assert view.gaps(fund()) == ()


# --- the provider calls ---

PERIOD = view.Period(utc("2021-01-04"), utc("2021-01-06"), "test", clamped=False)


def test_the_common_start_is_delegated_to_the_provider():
    metrics = RecordingMetrics(start=utc("2021-01-05"))
    late = fund("b", reasons={0: "missing_input"})  # first available close 2021-01-05
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
    assert result.funds[0].drawdown.max_drawdown == DRAWDOWN.max_drawdown
    pd.testing.assert_frame_equal(result.funds[0].calendar, CALENDAR)


def test_gaps_reach_the_provider_as_nan_and_stay_nan_in_growth():
    metrics = RecordingMetrics(start=utc("2021-01-04"))
    result = view.build_view([fund(reasons={1: "missing_input"})], PERIOD, metrics)
    returns = next(call[1] for call in metrics.calls if call[0] == "growth_index")
    assert returns.isna().tolist() == [False, True, False]
    assert result.funds[0].growth.isna().tolist() == [False, True, False]


def test_an_unavailable_input_makes_its_derived_metrics_unavailable():
    metrics = RecordingMetrics(start=utc("2021-01-04"), unavailable={"b": "too_few_sessions"})
    result = view.build_view([fund("a"), fund("b")], PERIOD, metrics)
    available, missing = result.funds
    assert isinstance(available.growth, pd.Series)
    expected = view.Unavailable("too_few_sessions", "too few sessions")
    assert missing.growth == missing.drawdown == missing.calendar == expected


def test_no_common_start_and_a_pending_provider_are_unavailable_states():
    result = view.build_view([fund()], PERIOD, RecordingMetrics(start=None))
    assert result.start == view.Unavailable("no_common_start", labels.NO_COMMON_START)
    assert result.funds[0].growth == result.start
    pending = view.build_view([fund()], PERIOD, PendingMetrics())
    expected = view.Unavailable(PENDING_REASON, PENDING_DETAIL)
    assert pending.start == expected
    assert pending.funds[0].growth == pending.funds[0].calendar == expected


class BrokenMetrics(RecordingMetrics):
    """The compliant fake with one contract violation, named by `broken`."""

    def __init__(self, broken: str):
        super().__init__(start=utc("2021-01-04"))
        self.broken = broken

    def common_start(self, available_sessions, window_start):
        starts = {"start_outside": utc("2021-01-07"), "start_in_gap": utc("2021-01-06")}
        return starts.get(self.broken, self.start)

    def growth_index(self, daily_total_returns, start_value=GROWTH_START_VALUE):
        growth = super().growth_index(daily_total_returns, start_value)
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

    def calendar_year_returns(self, growth):
        changed = {
            "columns": lambda: CALENDAR.drop(columns="losing"),
            "duplicate_years": lambda: pd.concat([CALENDAR, CALENDAR]),
            "year_outside": lambda: CALENDAR.assign(year=2020),
            "losing_flag": lambda: CALENDAR.assign(losing=False),
            "zero_is_losing": lambda: CALENDAR.assign(total_return=0.0),
        }
        return changed.get(self.broken, CALENDAR.copy)()


# The fund has a gap on 2021-01-06, so the period 2021-01-04..06 ends in it.
GAPPED = {"reasons": {2: "missing_input"}}


def test_the_compliant_fake_passes_every_check():
    result = view.build_view([fund(**GAPPED)], PERIOD, BrokenMetrics("none"))
    assert result.funds[0].growth.isna().tolist() == [False, False, True]


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
            "year": [2021, 2022, 2023],
            "total_return": [-0.0004, 0.0, 0.08],
            "partial": [True, False, False],
            "losing": [True, False, False],
        }
    )
    text = view.calendar_table([("Fund A", table)])
    assert text["Year"].tolist() == [2021, 2022, 2023]
    assert text["Fund A"].tolist() == [
        "−0.0% · lost money · partial year",
        "+0.0%",
        "+8.0%",
    ]
    assert view.losing_summary("Fund A", table) == (
        "Fund A: lost money in 2021: 1 of 3 calendar years shown (1 partial)."
    )
    none_lost = table.assign(losing=False)
    assert "no calendar year shown lost money" in view.losing_summary("Fund A", none_lost)


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
    assert labels.GROWTH_TITLE == "Hypothetical growth of $10,000"
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
    shown += [PENDING_DETAIL, *labels.REASON_TEXT.values(), *labels.LIMITATION_TEXT.values()]
    assert not [text for text in shown if "OD8" in text]
    assert labels.reason_text("missing_input") == "no value from the source"
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
    for palette in (figures.FUND_COLOURS, figures.FUND_MARKERS, figures.FUND_DASHES):
        assert len(set(palette)) == len(palette) == 5
    styles = figures.fund_styles(synthetic.demo_histories())
    assert len({(s.colour, s.marker, s.dash) for s in styles.values()}) == 5


def test_the_growth_figure_is_log_scaled_labelled_and_keeps_gaps():
    history = fund()
    index = pd.DatetimeIndex([utc(d) for d in ("2021-01-04", "2021-01-05", "2021-01-06")])
    growth = pd.Series([10_000.0, np.nan, 10_200.0], index=index)
    styles = figures.fund_styles([history])
    figure = figures.growth_figure([(history, growth)], styles, start_value=GROWTH_START_VALUE)
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Hypothetical growth of $10,000"
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
        }
    )
    figure = figures.calendar_year_figure([(history, table)], figures.fund_styles([history]))
    bar = figure.data[0]
    assert list(bar.x) == ["2021", "2022"]
    assert list(bar.marker.pattern.shape) == ["/", ""]
    assert "lost money" in bar.customdata[0] and "partial" in bar.customdata[0]
    assert figure.layout.yaxis.zeroline is True
    assert set(CALENDAR_COLUMNS) == {"year", "total_return", "partial", "losing"}
