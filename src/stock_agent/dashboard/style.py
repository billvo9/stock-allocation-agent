"""
Visual language shared by every chart.

- Colour means model identity only (Okabe-Ito, colour-blind safe), keyed
  by model name so a model keeps its colour across pages and runs.
- Role is encoded redundantly by marker shape and in legend text: null
  (circle), control (square), canary (x), unsafe canary reference
  (hatched, labelled). Hollow markers mean "corroborating method"; a
  stored warning status is shown as text, never by marker fill alone.
- Status colours are separate from the model palette and are used only for
  stored T2 statuses, never for p-values or rankings.
"""

from __future__ import annotations

import hashlib

OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#000000",  # black
)
# Non-model categories (symbols, fold segments) use a different palette (Paul
# Tol "muted"), so a colour never means two things.
CATEGORY_PALETTE = (
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
)
KNOWN_MODEL_COLOURS = {
    "zero": "#7F7F7F",
    "pooled_mean": "#000000",
    "random_noise": "#56B4E9",
    "per_symbol_mean": "#0072B2",
    "momentum_20d": "#E69F00",
    "memorizer": "#009E73",
    "canary_unsafe_reference": "#D55E00",
}
ROLE_MARKERS = {
    "null": ("circle", "dot"),
    "control": ("square", "dash"),
    "canary": ("x", "dashdot"),
    "canary_unsafe_reference": ("diamond-open", "longdash"),
    "candidate": ("circle", "solid"),
}
ROLE_LABELS = {
    "null": "null",
    "control": "control",
    "canary": "canary",
    "canary_unsafe_reference": "UNSAFE reference (leaky by design)",
    "candidate": "candidate",
}
STATUS_COLOURS = {
    "VALID": "#1B7837",
    "WARNING": "#B35806",
    "INVALID/BLOCKED": "#B2182B",
    "ok": "#4D4D4D",
    "warning": "#B35806",
    "unavailable": "#616161",
}
PRIMARY_OPACITY = 1.0
CORROBORATING_OPACITY = 0.9
NEUTRAL_SCALE = "Blues"  # sequential, for distances (never red/green)
FONT = {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 13}
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg"},
}


def category_colours(categories) -> dict[str, str]:
    """Stable colours for non-model categories (e.g. symbols), by sorted name."""

    return {
        name: CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]
        for index, name in enumerate(sorted(categories))
    }


def model_colour(name: str) -> str:
    """Stable colour for a model name (fixed for known T2 models)."""

    if name in KNOWN_MODEL_COLOURS:
        return KNOWN_MODEL_COLOURS[name]
    index = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % len(OKABE_ITO)
    return OKABE_ITO[index]


def role_marker(role: str) -> tuple[str, str]:
    return ROLE_MARKERS.get(role, ("circle", "solid"))


def legend_name(name: str, role: str) -> str:
    return f"{name} ({ROLE_LABELS.get(role, role)})"


def base_layout(title: str, *, x_title: str = "", y_title: str = "", height: int = 380) -> dict:
    """Common layout: explicit title and axis labels, white background, no dual axes."""

    return {
        "title": {
            "text": title,
            "font": {"size": 15},
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "yref": "container",
            "yanchor": "top",
        },
        "xaxis": {
            "title": {"text": x_title},
            "showgrid": True,
            "gridcolor": "#EEEEEE",
            "automargin": True,
        },
        "yaxis": {
            "title": {"text": y_title},
            "showgrid": True,
            "gridcolor": "#EEEEEE",
            "automargin": True,
        },
        "font": FONT,
        "height": height,
        "margin": {"l": 60, "r": 20, "t": 110, "b": 50},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "closest",
    }
