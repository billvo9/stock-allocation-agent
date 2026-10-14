"""
Pure status of the lab's data, for the Overview and Evidence pages: what is
loaded, what is synthetic, and what needs attention. Facts only, read from
the fund histories and their records (`view.gaps`, `view.breaks`); no
financial formula, no Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from stock_agent.app import labels, view
from stock_agent.app.history import FundHistory


@dataclass(frozen=True)
class Attention:
    """One thing that needs attention: a status level, its word, a title and a sentence."""

    level: str
    word: str
    title: str
    text: str


@dataclass(frozen=True, eq=False)
class DataStatus:
    funds: tuple[FundHistory, ...]
    with_data: tuple[FundHistory, ...]
    synthetic: tuple[FundHistory, ...]
    stale: tuple[FundHistory, ...]
    without_data: tuple[FundHistory, ...]
    gaps: tuple[
        tuple[FundHistory, tuple[view.Gap, ...]], ...
    ]  # funds with sessions lacking a close
    breaks: tuple[tuple[FundHistory, view.Break], ...]  # growth chains that cannot continue

    @property
    def all_synthetic(self) -> bool:
        return bool(self.funds) and len(self.synthetic) == len(self.funds)

    @property
    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """The earliest first close and the latest last close among funds with data."""

        if not self.with_data:
            return None
        return (
            min(history.coverage_start for history in self.with_data),
            max(history.coverage_end for history in self.with_data),
        )


def data_status(histories: Sequence[FundHistory]) -> DataStatus:
    with_data = tuple(h for h in histories if h.coverage_start is not None)
    gapped = tuple((h, view.gaps(h)) for h in with_data if view.gaps(h))
    broken = tuple((h, b) for h in with_data for b in view.breaks(h))
    return DataStatus(
        funds=tuple(histories),
        with_data=with_data,
        synthetic=tuple(h for h in histories if h.synthetic),
        stale=tuple(h for h in histories if h.stale),
        without_data=tuple(h for h in histories if h.coverage_start is None),
        gaps=gapped,
        breaks=broken,
    )


def _sessions(count: int) -> str:
    return f"{count:,} session" + ("" if count == 1 else "s")


def attention(status: DataStatus) -> tuple[Attention, ...]:
    """What needs attention, most serious first; every item is a fact about the records."""

    items = []
    if status.synthetic:
        items.append(
            Attention("WARNING", "Synthetic", labels.SYNTHETIC_BADGE, labels.SYNTHETIC_NOTICE)
        )
    for history in status.stale:
        items.append(Attention("WARNING", "Stale", history.name, history.stale_reason))
    for history in status.without_data:
        codes = view.unavailable_reasons(history) or ("no_records",)
        reasons = "; ".join(labels.reason_text(code) for code in codes)
        items.append(
            Attention(
                "WARNING",
                "No data",
                history.name,
                f"No available data ({reasons}), so it is left out everywhere, not set to zero.",
            )
        )
    for history, broken in status.breaks:
        text = labels.BREAK_NOTE.format(
            date=view.day(broken.date), reason=labels.reason_text(broken.reason)
        )
        items.append(Attention("WARNING", "Break", history.name, text))
    for history, gaps in status.gaps:
        total = sum(gap.sessions for gap in gaps)
        first = gaps[0]
        where = f"first {view.day(first.first)} to {view.day(first.last)}"
        items.append(
            Attention(
                "NOTICE",
                "Gap",
                history.name,
                f"{_sessions(total)} without a close ({where}). Missing values are never filled "
                "or set to zero.",
            )
        )
    return tuple(items)


def happening(status: DataStatus) -> tuple[str, ...]:
    """Plain sentences on what the lab holds now."""

    sentences = []
    count = len(status.funds)
    if status.all_synthetic:
        sentences.append(
            f"The lab is in synthetic demo mode: all {count} funds come from a seeded generator. "
            "No real fund data is loaded."
        )
    else:
        sentences.append(f"The lab holds {count} funds, {len(status.synthetic)} of them synthetic.")
    coverage = status.coverage
    if coverage is not None:
        sentences.append(
            f"{len(status.with_data)} of {count} funds have data, together covering "
            f"{view.day(coverage[0])} to {view.day(coverage[1])}."
        )
    sentences.append(
        "The approved descriptive calculations are running: hypothetical growth of $10,000, "
        "maximum drawdown and recovery, and calendar-year returns."
    )
    return tuple(sentences)


def measurements(status: DataStatus) -> pd.DataFrame:
    """Per fund, the exact counts behind the Overview's summary."""

    gaps = {h.fund_id: sum(g.sessions for g in found) for h, found in status.gaps}
    breaks: dict[str, list[str]] = {}
    for history, broken in status.breaks:
        breaks.setdefault(history.fund_id, []).append(view.day(broken.date))
    rows = []
    for history in status.funds:
        start, end = history.coverage_start, history.coverage_end
        has_data = start is not None
        rows.append(
            {
                "Fund": history.name,
                "Coverage": f"{view.day(start)} to {view.day(end)}" if has_data else "No data",
                "Sessions with a close": f"{len(history.available_sessions):,}",
                "Sessions without a close": (
                    f"{gaps.get(history.fund_id, 0):,}" if has_data else "—"
                ),
                "Growth cannot continue from": ", ".join(breaks.get(history.fund_id, [])) or "—",
                "Freshness": "Stale" if history.stale else "Not stale",
                "Data": "Synthetic" if history.synthetic else "Historical market data",
            }
        )
    return pd.DataFrame(rows)
