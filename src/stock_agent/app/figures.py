"""
Pure Plotly figure builders for the Explore page. They draw provider
results as given and compute nothing.

Visual language:
- each fund keeps one identity across the page: an Okabe-Ito colour (each
  at least 3:1 on the chart surface, WCAG 1.4.11), a marker symbol and a
  line dash, so colour is never the only carrier, and its name in the legend;
- gains and losses are neutral: they are read from the sign and the zero
  line, never from red or green;
- long growth histories use a logarithmic axis, labelled as such;
- gaps stay gaps (never connected across missing sessions).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from stock_agent.app import labels
from stock_agent.app.history import FundHistory
from stock_agent.app.view import money
from stock_agent.ui import style, theme

# Okabe-Ito blue, vermillion, bluish green, reddish purple and black: the
# members of the shared palette with at least 3:1 contrast on the chart surface.
FUND_COLOURS = tuple(style.OKABE_ITO[index] for index in (0, 5, 2, 3, 6))
FUND_MARKERS = ("circle", "square", "diamond", "triangle-up", "x")
FUND_DASHES = ("solid", "dash", "dot", "dashdot", "longdash")
MARKERS_SHOWN = 24  # markers per line: enough to identify it, few enough to stay calm


@dataclass(frozen=True)
class FundStyle:
    colour: str
    marker: str
    dash: str


def fund_styles(histories: Sequence[FundHistory]) -> dict[str, FundStyle]:
    """A stable identity per fund, by its position among all loaded funds."""

    return {
        history.fund_id: FundStyle(
            FUND_COLOURS[index % len(FUND_COLOURS)],
            FUND_MARKERS[index % len(FUND_MARKERS)],
            FUND_DASHES[index % len(FUND_DASHES)],
        )
        for index, history in enumerate(histories)
    }


def growth_figure(
    series: Sequence[tuple[FundHistory, pd.Series]],
    styles: dict[str, FundStyle],
    *,
    start_value: float,
) -> go.Figure:
    figure = go.Figure()
    for history, growth in series:
        identity = styles[history.fund_id]
        figure.add_trace(
            go.Scatter(
                x=growth.index,
                y=growth.to_numpy(dtype="float64"),
                name=history.name,
                mode="lines+markers",
                line={"color": identity.colour, "dash": identity.dash, "width": 2},
                marker={
                    "symbol": identity.marker,
                    "size": 7,
                    "color": identity.colour,
                    "maxdisplayed": MARKERS_SHOWN,
                },
                connectgaps=False,
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>",
            )
        )
    figure.add_hline(
        y=start_value,
        line={"color": style.REFERENCE_LINE, "dash": "dot", "width": 1},
        annotation_text=f"Start: {money(start_value)}",
        annotation_position="bottom right",
        annotation_font={"color": theme.COLOURS["text_muted"]},
    )
    layout = style.base_layout(
        labels.GROWTH_TITLE,
        x_title="Session date",
        y_title="Value in US dollars (log scale)",
        subtitle="Distributions reinvested on the ex-date · before taxes and transaction costs",
        height=440,
    )
    layout["yaxis"].update({"type": "log", "tickprefix": "$", "tickformat": ",.0f"})
    figure.update_layout(**layout)
    return figure


def calendar_year_figure(
    tables: Sequence[tuple[FundHistory, pd.DataFrame]], styles: dict[str, FundStyle]
) -> go.Figure:
    figure = go.Figure()
    for history, table in tables:
        identity = styles[history.fund_id]
        partial = table["partial"].astype(bool).tolist()
        losing = table["losing"].astype(bool).tolist()
        notes = [
            "; ".join(
                text
                for text, flag in (("lost money", lost), ("partial year, not annualized", part))
                if flag
            )
            for lost, part in zip(losing, partial, strict=True)
        ]
        figure.add_trace(
            go.Bar(
                x=[str(int(year)) for year in table["year"]],
                y=table["total_return"].to_numpy(dtype="float64"),
                name=history.name,
                marker={
                    "color": identity.colour,
                    "line": {"color": identity.colour, "width": 1},
                    "pattern": {
                        "shape": ["/" if flag else "" for flag in partial],
                        "fgcolor": theme.COLOURS["surface"],
                    },
                },
                customdata=notes,
                hovertemplate=(
                    "%{fullData.name}<br>%{x}: %{y:+.1%}<br>%{customdata}<extra></extra>"
                ),
            )
        )
    layout = style.base_layout(
        labels.CALENDAR_TITLE,
        x_title="Calendar year",
        y_title="Total return in the year",
        subtitle="Bars below zero lost money · hatched bars are partial years, not annualized",
        height=400,
    )
    layout["yaxis"].update(
        {"tickformat": ".0%", "zeroline": True, "zerolinecolor": style.REFERENCE_LINE}
    )
    layout["xaxis"].update({"type": "category"})
    figure.update_layout(**layout, barmode="group")
    return figure
