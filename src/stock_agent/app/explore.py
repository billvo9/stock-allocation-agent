"""
The Explore page "Fund history": a few broad funds' past, as facts.

Layout only. Every number comes from the metrics provider through
`view.build_view`, every caveat from `labels`, and every surface from the
shared components (`stock_agent.ui`). Each first-class state is shown in
words, never as an empty chart or a zero: loading, no fund selected, stale
data, an unavailable fund, data gaps, a period outside the coverage, no
common start, and a metric that is unavailable (until T5-A: pending approval
of OD8 group A).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import streamlit as st

from stock_agent.app import labels, view
from stock_agent.app.figures import FundStyle, calendar_year_figure, fund_styles, growth_figure
from stock_agent.app.history import FundHistory
from stock_agent.app.provider import GROWTH_START_VALUE, PENDING_REASON, DescriptiveMetrics
from stock_agent.app.view import MINUS, day, money, percent
from stock_agent.ui import components as ui

FUNDS_KEY = "explore_funds"
PERIOD_KEY = "explore_period"
YEARS_KEY = "explore_years"
SYNTHETIC_LIMITATION = "synthetic_data"
CARDS_PER_ROW = 3
GAPS_LISTED = 3


# Streamlit markdown reads a pair of dollar signs as LaTeX, so every
# markdown-rendered text goes through these wrappers, which escape them.
def _md(text: str) -> str:
    return text.replace("$", "\\$")


def _markdown(text: str, where=None) -> None:
    (where or st).markdown(_md(text))


def _caption(text: str, where=None) -> None:
    (where or st).caption(_md(text))


def _notice(level: str, text: str) -> None:
    ui.notice(level, _md(text))


def _section(title: str, caption: str | None = None) -> None:
    ui.section(_md(title), _md(caption) if caption else None)


def _sessions(count: int) -> str:
    return f"{count:,} session" + ("" if count == 1 else "s")


def loading_state() -> None:
    """Shown while fund histories load; replaced by the page once they are ready."""

    with ui.state_card("loading"):
        ui.pill("NOTICE", "Loading", live=True)
        _caption(labels.LOADING)


def render(
    load_histories: Callable[[], Sequence[FundHistory]], metrics: DescriptiveMetrics
) -> None:
    ui.page_header(labels.PAGE_TITLE, eyebrow=labels.PAGE_EYEBROW, purpose=_md(labels.PAGE_PURPOSE))
    loading = st.empty()
    with loading.container():
        loading_state()
    histories = tuple(load_histories())
    loading.empty()

    badges = st.empty()
    if any(SYNTHETIC_LIMITATION in history.limitations for history in histories):
        _notice("warning", labels.SYNTHETIC_NOTICE)
    funds_column, period_column = st.columns([3, 2], gap="large")
    selected = _pick_funds(histories, funds_column)
    usable = tuple(history for history in selected if history.coverage_start is not None)
    coverage = view.shared_coverage(usable)
    period, requested = _pick_period(coverage, period_column)
    with badges.container():
        _badges(histories, selected)

    if not selected:
        _notice("notice", labels.EMPTY)
        _about(selected)
        return
    _status_notices(selected)
    if not usable:
        _notice("notice", labels.NO_AVAILABLE_FUND)
    elif coverage is None:
        _notice("notice", labels.NO_SHARED_PERIOD)
    elif period is None:
        text = labels.OUTSIDE_COVERAGE.format(
            period=requested, start=day(coverage[0]), end=day(coverage[1])
        )
        _notice("notice", text)
    else:
        result = view.build_view(usable, period, metrics)
        styles = fund_styles(histories)
        _gap_notices(result)
        if _any_pending(result):
            _notice("notice", labels.PENDING_EXPLAINER)
        _growth(result, styles)
        _declines(result)
        _calendar(result, styles)
    _about(selected)


# --- selection and status ---


def _option(history: FundHistory) -> str:
    if history.coverage_start is None:
        return f"{history.name} · no data"
    return f"{history.name} · stale" if history.stale else history.name


def _pick_funds(histories: Sequence[FundHistory], where) -> tuple[FundHistory, ...]:
    by_id = {history.fund_id: history for history in histories}
    default = [h.fund_id for h in histories if h.coverage_start is not None and not h.stale][:3]
    chosen = where.multiselect(
        "Funds",
        list(by_id),
        default=default,
        format_func=lambda fund_id: _option(by_id[fund_id]),
        key=FUNDS_KEY,
        help="Listed in a fixed order. No fund is ranked or preferred.",
    )
    return tuple(history for history in histories if history.fund_id in chosen)


def _pick_period(
    coverage: tuple[pd.Timestamp, pd.Timestamp] | None, where
) -> tuple[view.Period | None, str]:
    if coverage is None:
        _caption("Choose a fund with data to set a period.", where)
        return None, ""
    preset = where.radio(
        "Period",
        list(view.PRESET_LABELS),
        format_func=view.PRESET_LABELS.get,
        key=PERIOD_KEY,
        horizontal=True,
        help="Calendar years, cut to the dates every selected fund covers.",
    )
    if preset == view.CUSTOM:
        first, last = coverage[0].year, coverage[1].year
        start_year, end_year = first, last
        if first < last:
            start_year, end_year = where.select_slider(
                "Years",
                options=list(range(first, last + 1)),
                value=(first, last),
                key=f"{YEARS_KEY}_{first}_{last}",
            )
        requested = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
        period = view.clamp_period(start_year, end_year, coverage, requested)
    else:
        requested = view.PRESET_LABELS[preset]
        period = view.preset_period(preset, coverage)
    if period is not None:
        limited = ", limited to the dates every selected fund covers" if period.clamped else ""
        _caption(f"Shown: {day(period.start)} to {day(period.end)}{limited}.", where)
    return period, requested


def _badges(histories: Sequence[FundHistory], selected: Sequence[FundHistory]) -> None:
    items = [("NOTICE", labels.EVIDENCE_BADGE)]
    if any(SYNTHETIC_LIMITATION in history.limitations for history in histories):
        items.append(("WARNING", labels.SYNTHETIC_BADGE))
    if any(history.stale for history in selected):
        items.append(("WARNING", labels.STALE_BADGE))
    ui.pills(items)


def _status_notices(selected: Sequence[FundHistory]) -> None:
    for history in selected:
        if history.stale:
            _notice("warning", f"{history.name} is stale. {history.stale_reason}")
        if history.coverage_start is None:
            reasons = ", ".join(view.unavailable_reasons(history)) or "no_records"
            _notice(
                "warning",
                f"{history.name} has no available data, so it is excluded, not set to zero "
                f"(reason: {reasons}).",
            )


def _gap_notices(result: view.ExploreView) -> None:
    for fund in result.funds:
        if not fund.gaps:
            continue
        listed = "; ".join(
            f"{day(gap.first)} to {day(gap.last)}, {_sessions(gap.sessions)} ({gap.reason})"
            for gap in fund.gaps[:GAPS_LISTED]
        )
        extra = len(fund.gaps) - GAPS_LISTED
        more = f"; and {extra:,} more" if extra > 0 else ""
        total = sum(gap.sessions for gap in fund.gaps)
        _notice(
            "notice",
            f"{fund.history.name} has {_sessions(total)} without data in this period: "
            f"{listed}{more}. Missing returns are never filled or set to zero.",
        )


def _any_pending(result: view.ExploreView) -> bool:
    values = [result.start, *(m for f in result.funds for m in (f.growth, f.drawdown, f.calendar))]
    return any(isinstance(v, view.Unavailable) and v.reason == PENDING_REASON for v in values)


def _unavailable(key: str, title: str, items: Sequence[tuple[str, view.Unavailable]]) -> None:
    """One dashed card per distinct reason, naming the funds it applies to."""

    groups: dict[tuple[str, str], list[str]] = {}
    for name, missing in items:
        groups.setdefault((missing.reason, missing.detail), []).append(name)
    for index, ((reason, detail), names) in enumerate(groups.items()):
        pending = reason == PENDING_REASON
        with ui.state_card(f"{key}-{index}"):
            ui.pill("FUTURE", labels.PENDING_BADGE if pending else labels.UNAVAILABLE_BADGE)
            _markdown(f"**{title}** is not shown for {', '.join(names)}.")
            _caption(f"Unavailable: {detail}.")


# --- sections ---


def _growth(result: view.ExploreView, styles: dict[str, FundStyle]) -> None:
    _section(labels.GROWTH_TITLE, " ".join(labels.HONEST_LABELS))
    drawn = [(f.history, f.growth) for f in result.funds if isinstance(f.growth, pd.Series)]
    missing = [
        (f.history.name, f.growth) for f in result.funds if isinstance(f.growth, view.Unavailable)
    ]
    if missing:
        _unavailable("growth", labels.GROWTH_TITLE, missing)
    if not drawn:
        return
    lasts = []
    for history, growth in drawn:
        last = view.last_value(growth)
        if last is not None:
            lasts.append(f"{history.name} {money(last[1])} on {day(last[0])}")
    _markdown(
        f"A hypothetical {money(GROWTH_START_VALUE)} placed in each fund on "
        f"{day(result.start)}. Last values: {'; '.join(lasts)}."
    )
    figure = growth_figure(drawn, styles, start_value=GROWTH_START_VALUE)
    ui.chart(figure, key="explore_growth")
    _caption(labels.LOG_SCALE)
    with st.expander("Growth as a table: values at each year end"):
        table = view.year_end_table([(history.name, growth) for history, growth in drawn])
        st.dataframe(table, hide_index=True)


def _decline_card(fund: view.FundView) -> None:
    _markdown(f"**{fund.history.name}**")
    summary = fund.drawdown
    if isinstance(summary, view.Unavailable):
        st.metric("Worst decline", "—")
        _caption(f"Unavailable: {summary.detail}.")
        return
    if summary.max_drawdown == 0:
        st.metric("Worst decline", "None")
        _caption("No decline from a previous high in this period.")
        return
    st.metric("Worst decline", f"{MINUS}{percent(summary.max_drawdown)}")
    _caption(
        f"{money(summary.dollar_decline)} below its peak, from a hypothetical "
        f"{money(GROWTH_START_VALUE)} start."
    )
    _caption(f"Peak {day(summary.peak_date)} · low {day(summary.trough_date)}.")
    span = f"{_sessions(summary.duration_sessions)} ({summary.duration_days:,} calendar days)"
    if summary.recovered:
        _caption(f"Recovered on {day(summary.recovery_date)}: {span} from peak to recovery.")
    else:
        _caption(f"Not recovered by {day(fund.growth.index[-1])}: {span} since the peak.")


def _declines(result: view.ExploreView) -> None:
    _section(labels.DECLINE_TITLE, f"{labels.DECLINE_SCOPE} {labels.DAILY_CLOSES}")
    funds = result.funds
    for first in range(0, len(funds), CARDS_PER_ROW):
        row = funds[first : first + CARDS_PER_ROW]
        for column, fund in zip(st.columns(CARDS_PER_ROW), row, strict=False):
            with column, ui.card(f"decline-{fund.history.fund_id}"):
                _decline_card(fund)


def _calendar(result: view.ExploreView, styles: dict[str, FundStyle]) -> None:
    _section(labels.CALENDAR_TITLE, labels.CALENDAR_SCOPE)
    tables = [(f.history, f.calendar) for f in result.funds if isinstance(f.calendar, pd.DataFrame)]
    missing = [
        (f.history.name, f.calendar)
        for f in result.funds
        if isinstance(f.calendar, view.Unavailable)
    ]
    if missing:
        _unavailable("calendar", labels.CALENDAR_TITLE, missing)
    if not tables:
        return
    _markdown("\n".join(f"- {view.losing_summary(h.name, table)}" for h, table in tables))
    ui.chart(calendar_year_figure(tables, styles), key="explore_calendar")
    with st.expander("Calendar-year returns as a table"):
        table = view.calendar_table([(history.name, frame) for history, frame in tables])
        st.dataframe(table, hide_index=True)


def _sources(selected: Sequence[FundHistory]) -> pd.DataFrame:
    rows = []
    for history in selected:
        start, end = history.coverage_start, history.coverage_end
        rows.append(
            {
                "Fund": history.name,
                "Role": labels.ROLE_TEXT[history.role],
                "Source": history.source_label,
                "Coverage": "No available data" if start is None else f"{day(start)} to {day(end)}",
                "Retrieved": history.retrieved_label,
                "Vintage": history.vintage_note,
                "Freshness": f"Stale: {history.stale_reason}" if history.stale else "Not stale",
                "Evidence": history.evidence_status,
            }
        )
    return pd.DataFrame(rows)


def _about(selected: Sequence[FundHistory]) -> None:
    _section(labels.ABOUT_TITLE)
    facts = (
        f"{labels.GROWTH_TITLE}: a computation from the records, not an account statement.",
        *labels.HONEST_LABELS,
        f"Evidence status: descriptive ({labels.EVIDENCE_BADGE.lower()}), not a forecast.",
    )
    _markdown("\n".join(f"- {fact}" for fact in facts))
    if selected:
        st.dataframe(_sources(selected), hide_index=True)
        codes = sorted({code for history in selected for code in history.limitations})
        if codes:
            limits = "\n".join(f"- {labels.LIMITATION_TEXT.get(code, code)}" for code in codes)
            _markdown(f"**Limitations**\n\n{limits}")
    _section(labels.CANNOT_TELL_TITLE)
    _markdown(labels.CANNOT_TELL)
