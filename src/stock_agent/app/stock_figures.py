"""
Pure Plotly figure builders for the Stock Probability page. They draw the
view's frames as given and compute nothing.

Visual language: the three directions keep one identity everywhere (theme
`DIRECTION_PALETTES`, a colour-blind-safe blue / vermillion / grey, never red
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


# --- the modelled views (PR 2) ---


def five_session_figure(
    frame: pd.DataFrame, category: str, *, synthetic: bool, scheme: str = "light"
) -> go.Figure:
    """
    One direction's probability for each of the next five sessions: a dot with
    its marker, a bar for the development-test range, and the stock's usual
    rate as a hollow marker on a dotted line. The axis is fixed at 0-100%.
    """

    tokens = theme.LAB_PALETTES[scheme]
    rows = frame[frame["category"] == category].sort_values("horizon")
    colour = theme.DIRECTION_PALETTES[scheme][category]
    has_range = rows["range_low"].notna()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rows["horizon"],
            y=rows["usual"],
            mode="lines+markers",
            name=words.USUAL_SERIES,
            line={"color": tokens["reference_line"], "dash": "dot", "width": 1},
            marker={"symbol": "circle-open", "size": 9, "color": tokens["reference_line"]},
            hovertemplate="Usual: %{y:.0%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["horizon"],
            y=rows["probability"],
            mode="markers",
            name=words.FORECAST_SERIES,
            marker={"symbol": theme.DIRECTION_MARKERS[category], "size": 12, "color": colour},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (rows["range_high"] - rows["probability"]).where(has_range, 0),
                "arrayminus": (rows["probability"] - rows["range_low"]).where(has_range, 0),
                "color": colour,
                "thickness": 1.5,
                "width": 4,
            },
            customdata=rows["session"],
            hovertemplate="%{customdata}: %{y:.0%}<extra></extra>",
        )
    )
    layout = style.lab_layout(
        CATEGORY_WORDS[next(c for c in CATEGORIES if c.value == category)],
        x_title="Sessions ahead",
        y_title="Probability",
        subtitle=_subtitle(words.FIVE_SUBTITLE, synthetic),
        height=300,
        scheme=scheme,
    )
    layout["yaxis"].update({"range": [0, 1], "tickformat": ".0%"})
    layout["xaxis"].update({"tickmode": "array", "tickvals": [1, 2, 3, 4, 5], "range": [0.5, 5.5]})
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure


def streak_figure(
    profile: pd.DataFrame,
    direction: str,
    current: int | None,
    *,
    synthetic: bool,
    scheme: str = "light",
) -> go.Figure:
    """How often the next session reversed after exactly k sessions in `direction`."""

    tokens = theme.LAB_PALETTES[scheme]
    reverse = "fell" if direction == "increase" else "rose"
    opposite = "decrease" if direction == "increase" else "increase"
    estimate = profile[reverse]
    low, high = profile[f"{reverse}_low"], profile[f"{reverse}_high"]
    thin = profile["status"] != "ok"
    figure = go.Figure(
        go.Bar(
            x=profile["length"],
            y=estimate,
            text=[f"{n:,} cases" for n in profile["events"]],
            textposition="outside",
            marker={
                "color": theme.DIRECTION_PALETTES[scheme][opposite],
                "pattern": {
                    "shape": ["/" if flag else "" for flag in thin],
                    "fgcolor": tokens["surface"],
                },
                "line": {
                    "color": tokens["text"],
                    "width": [2.5 if k == current else 0 for k in profile["length"]],
                },
            },
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (high - estimate).fillna(0),
                "arrayminus": (estimate - low).fillna(0),
                "color": tokens["text_muted"],
            },
            hovertemplate="After %{x} in a row: %{y:.0%}<extra></extra>",
        )
    )
    noun = "rising" if direction == "increase" else "falling"
    layout = style.lab_layout(
        words.STREAK_CHART_TITLE.format(noun=noun, reverse=reverse),
        x_title=f"{noun.capitalize()} sessions in a row",
        y_title=f"Next session {reverse}",
        subtitle=_subtitle(words.STREAK_CHART_SUBTITLE, synthetic),
        height=360,
        scheme=scheme,
    )
    layout["yaxis"].update({"range": [0, 1], "tickformat": ".0%"})
    layout["xaxis"].update({"tickmode": "array", "tickvals": list(profile["length"])})
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure


def reliability_figure(
    frame: pd.DataFrame,
    model: str,
    reference: str,
    category: int,
    *,
    synthetic: bool,
    scheme: str = "light",
) -> go.Figure:
    """Predicted probability against how often it happened, on square 0-1 axes."""

    tokens = theme.LAB_PALETTES[scheme]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name=words.DIAGONAL,
            line={"color": tokens["reference_line"], "dash": "dot", "width": 1},
            hoverinfo="skip",
        )
    )
    names = {code: c.value for c, code in zip(CATEGORIES, (2, 1, 0), strict=True)}
    colour = theme.DIRECTION_PALETTES[scheme][names[category]]
    for name, symbol, fill in ((reference, "circle-open", False), (model, "circle", True)):
        rows = frame[(frame["model"] == name) & (frame["category"] == category)]
        figure.add_trace(
            go.Scatter(
                x=rows["mean_predicted"],
                y=rows["observed"],
                mode="markers",
                name=name,
                marker={
                    "symbol": symbol,
                    "size": 10,
                    "color": colour if fill else tokens["text_muted"],
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": rows["ci_high"] - rows["observed"],
                    "arrayminus": rows["observed"] - rows["ci_low"],
                    "color": colour if fill else tokens["text_muted"],
                    "visible": fill,
                },
                customdata=rows["n"],
                hovertemplate="said %{x:.0%}, happened %{y:.0%} (%{customdata:,} forecasts)"
                "<extra>%{fullData.name}</extra>",
            )
        )
    layout = style.lab_layout(
        words.RELIABILITY_TITLE.format(
            category=CATEGORY_WORDS[next(c for c in CATEGORIES if c.value == names[category])]
        ),
        x_title="Forecast probability",
        y_title="How often it happened",
        subtitle=_subtitle(words.RELIABILITY_SUBTITLE, synthetic),
        height=380,
        scheme=scheme,
    )
    for axis in ("xaxis", "yaxis"):
        layout[axis].update({"range": [0, 1], "tickformat": ".0%", "constrain": "domain"})
    layout["yaxis"]["scaleanchor"] = "x"
    figure.update_layout(**layout)
    return figure


def forecast_count_figure(
    frame: pd.DataFrame, model: str, category: int, *, synthetic: bool, scheme: str = "light"
) -> go.Figure:
    """How many development-test forecasts fell in each probability range."""

    rows = frame[(frame["model"] == model) & (frame["category"] == category)]
    figure = go.Figure(
        go.Bar(
            x=[
                f"{low:.0%}–{high:.0%}"
                for low, high in zip(rows["bin_low"], rows["bin_high"], strict=True)
            ],
            y=rows["n"],
            marker={"color": theme.LAB_PALETTES[scheme]["text_muted"]},
            hovertemplate="%{x}: %{y:,} forecasts<extra></extra>",
        )
    )
    layout = style.lab_layout(
        words.COUNTS_TITLE,
        x_title="Forecast probability range",
        y_title="Forecasts",
        subtitle=_subtitle(words.COUNTS_SUBTITLE, synthetic),
        height=280,
        scheme=scheme,
    )
    layout["showlegend"] = False
    figure.update_layout(**layout)
    return figure
