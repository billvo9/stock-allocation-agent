"""
Pure Plotly figure builders for the Stock Probability page. They draw the
view's frames as given and compute nothing.

Visual language: the three directions keep one identity everywhere (theme
`DIRECTION_COLOURS`, a colour-blind-safe blue / vermillion / grey, never red
and green), always paired with a marker shape so colour never carries the
category alone; gaps stay gaps; a synthetic stock says so in every chart's
subtitle (a chart can be exported on its own); every chart uses the Lab's
editorial layout, in the viewer's colour scheme ("light" or "dark") which
the page passes in, and has a table beside it on the page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from stock_agent.app import stock_wording as words
from stock_agent.stock_lab.contracts import CATEGORIES, CATEGORY_WORDS, StableBand
from stock_agent.ui import style, theme


def _subtitle(text: str, synthetic: bool) -> str:
    return f"{words.SYNTHETIC_CHART} · {text}" if synthetic else text


def price_figure(frame: pd.DataFrame, *, synthetic: bool, scheme: str = "light") -> go.Figure:
    """The split-consistent price on the last shown session's share basis (log axis)."""

    figure = go.Figure(
        go.Scatter(
            x=frame["session"],
            y=frame["display_level"],
            mode="lines",
            name=words.PRICE_SERIES,
            line={"color": theme.LAB_PALETTES[scheme]["text"], "width": 1.5},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.2f}<extra></extra>",
        )
    )
    layout = style.lab_layout(
        words.PRICE_TITLE,
        x_title="Session date",
        y_title=words.PRICE_AXIS,
        subtitle=_subtitle(words.PRICE_SUBTITLE, synthetic),
        height=360,
        scheme=scheme,
    )
    layout["yaxis"].update({"type": "log"})
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure


def recent_figure(
    recent: pd.DataFrame, band: StableBand, *, synthetic: bool, scheme: str = "light"
) -> go.Figure:
    """The last sessions' price returns, one marker per session, by direction."""

    tokens = theme.LAB_PALETTES[scheme]
    figure = go.Figure()
    for category in CATEGORIES:
        rows = recent[recent["category"].eq(category.value)]
        figure.add_trace(
            go.Scatter(
                x=rows["session"],
                y=rows["price_return"],
                mode="markers",
                name=CATEGORY_WORDS[category],
                marker={
                    "symbol": theme.DIRECTION_MARKERS[category.value],
                    "size": 9,
                    "color": theme.DIRECTION_PALETTES[scheme][category.value],
                },
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.2%}<extra>%{fullData.name}</extra>",
            )
        )
    for edge, label in ((band.tau, f"+{band.words()[1:]}"), (-band.tau, f"−{band.words()[1:]}")):
        figure.add_hline(
            y=edge,
            line={"color": tokens["reference_line"], "dash": "dot", "width": 1},
            annotation_text=label,
            annotation_position="right",
            annotation_font={"color": tokens["text_muted"]},
        )
    layout = style.lab_layout(
        words.RECENT_TITLE.format(count=len(recent)),
        x_title="Session date",
        y_title="Price return",
        subtitle=_subtitle(words.RECENT_SUBTITLE, synthetic),
        height=340,
        scheme=scheme,
    )
    layout["yaxis"].update(
        {"tickformat": ".2%", "zeroline": True, "zerolinecolor": tokens["reference_line"]}
    )
    figure.update_layout(**layout)
    return figure
