"""
Design tokens for the research dashboard: the single source of its look.

Three consumers read these tokens, so pages never style themselves:
- Streamlit's theme (`.streamlit/config.toml`, checked against
  `streamlit_theme()` by a test): palette, alert colours, type scale, radii;
- a thin CSS layer (`css()`, injected once by the app): cards, status pills,
  captions, focus rings, tabular numbers, motion;
- Plotly layouts (`style.base_layout`): fonts, surfaces, grid, fold lines.

Principles. Warm neutral surfaces and one restrained accent; colour carries
meaning only (status, or model identity in charts, see style.py) and never
alone; low chrome; numbers in tabular figures. Motion only marks a state
change and never touches a value: a chart fades in when it appears (opacity
only; Plotly never tweens data, which would draw values T2 never stored),
and hovering a clickable summary shades it. The dashboard's CSS motion is
off under prefers-reduced-motion.
"""

from __future__ import annotations

COLOURS = {
    "canvas": "#F7F5F0",  # page background (warm paper)
    "surface": "#FFFFFF",  # cards, charts
    "surface_muted": "#F1EDE6",  # sidebar, widget fills, table headers
    "border": "#E3DDD2",
    "border_strong": "#CBC3B5",
    "text": "#1F1D1A",
    "text_muted": "#57524A",
    "text_subtle": "#6E675C",
    "accent": "#2D5D6C",  # restrained slate teal: links, focus, selection
    "accent_soft": "#E3ECEE",
    "grid": "#ECE7DF",
    "fold_boundary": "#B3AA9B",  # darker than the grid and drawn dotted
    "reference_line": "#8E867A",
}

# Status treatments. Colour is never the only carrier: every status is also
# written out in words.
STATUS = {
    "VALID": {"fg": "#1E6633", "bg": "#E6F0E8", "border": "#B5D2BC"},
    "WARNING": {"fg": "#874A0E", "bg": "#FAEFDF", "border": "#E8C89A"},
    "INVALID/BLOCKED": {"fg": "#99202C", "bg": "#F7E5E6", "border": "#E2B2B7"},
    "NOTICE": {"fg": "#26505D", "bg": "#E5EDEF", "border": "#BBCED4"},
    "FUTURE": {"fg": "#57524A", "bg": "#F1EDE6", "border": "#CBC3B5"},
}

TYPE = {
    # The platform UI font (no web font is loaded, so nothing is fetched).
    "sans": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
    "mono": "SF Mono, JetBrains Mono, Menlo, Consolas, monospace",
    "base_px": 15,
    "heading_sizes": ["1.85rem", "1.35rem", "1.12rem", "1rem", "0.95rem", "0.9rem"],
    "heading_weights": [650, 600, 600, 600, 600, 600],
    "metric_size": "1.55rem",
    "metric_weight": 600,
    "chart_title_px": 15,
    "chart_px": 13,
}

SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
RADIUS = {"control": "8px", "card": "12px", "pill": "999px"}
MOTION = {"fast_ms": 120, "appear_ms": 200, "easing": "cubic-bezier(0.2, 0, 0, 1)"}
CONTENT_MAX_WIDTH = "1240px"


def streamlit_theme() -> dict[str, dict[str, object]]:
    """The [theme] and [theme.sidebar] tables of .streamlit/config.toml."""

    status = STATUS
    return {
        "theme": {
            "base": "light",
            "primaryColor": COLOURS["accent"],
            "backgroundColor": COLOURS["canvas"],
            "secondaryBackgroundColor": COLOURS["surface_muted"],
            "textColor": COLOURS["text"],
            "linkColor": COLOURS["accent"],
            "borderColor": COLOURS["border"],
            "dataframeBorderColor": COLOURS["border"],
            "dataframeHeaderBackgroundColor": COLOURS["surface_muted"],
            "showWidgetBorder": True,
            "baseRadius": RADIUS["control"],
            "buttonRadius": RADIUS["control"],
            "font": TYPE["sans"],
            "headingFont": TYPE["sans"],
            "codeFont": TYPE["mono"],
            "baseFontSize": TYPE["base_px"],
            "headingFontSizes": TYPE["heading_sizes"],
            "headingFontWeights": TYPE["heading_weights"],
            "metricValueFontSize": TYPE["metric_size"],
            "metricValueFontWeight": TYPE["metric_weight"],
            "greenColor": status["VALID"]["fg"],
            "greenBackgroundColor": status["VALID"]["bg"],
            "greenTextColor": status["VALID"]["fg"],
            "yellowColor": status["WARNING"]["fg"],
            "yellowBackgroundColor": status["WARNING"]["bg"],
            "yellowTextColor": status["WARNING"]["fg"],
            "orangeColor": status["WARNING"]["fg"],
            "orangeBackgroundColor": status["WARNING"]["bg"],
            "orangeTextColor": status["WARNING"]["fg"],
            "redColor": status["INVALID/BLOCKED"]["fg"],
            "redBackgroundColor": status["INVALID/BLOCKED"]["bg"],
            "redTextColor": status["INVALID/BLOCKED"]["fg"],
            "blueColor": status["NOTICE"]["fg"],
            "blueBackgroundColor": status["NOTICE"]["bg"],
            "blueTextColor": status["NOTICE"]["fg"],
            "grayColor": COLOURS["text_subtle"],
            "grayBackgroundColor": COLOURS["surface_muted"],
            "grayTextColor": COLOURS["text_muted"],
        },
        "theme.sidebar": {
            "backgroundColor": COLOURS["surface_muted"],
            "secondaryBackgroundColor": COLOURS["surface"],
            "borderColor": COLOURS["border"],
        },
    }


def _variables() -> str:
    pairs = {
        **{f"--sa-{name.replace('_', '-')}": value for name, value in COLOURS.items()},
        "--sa-radius-card": RADIUS["card"],
        "--sa-radius-control": RADIUS["control"],
        "--sa-motion-fast": f"{MOTION['fast_ms']}ms",
        "--sa-motion-appear": f"{MOTION['appear_ms']}ms",
        "--sa-ease": MOTION["easing"],
        "--sa-mono": TYPE["mono"],
    }
    for level, tones in STATUS.items():
        slug = level.split("/")[-1].lower()
        for part, value in tones.items():
            pairs[f"--sa-{slug}-{part}"] = value
    return "\n".join(f"  {name}: {value};" for name, value in pairs.items())


def status_slug(level: str) -> str:
    """CSS modifier for a status level: valid, warning, blocked, notice, future."""

    return level.split("/")[-1].lower()


def css() -> str:
    """The dashboard's CSS layer. Stable hooks only: data-testid and keyed containers."""

    pills = "\n".join(
        f".sa-pill--{status_slug(level)} {{ color: var(--sa-{status_slug(level)}-fg); "
        f"background: var(--sa-{status_slug(level)}-bg); "
        f"border-color: var(--sa-{status_slug(level)}-border); }}"
        for level in STATUS
    )
    space = {name: f"{value}px" for name, value in SPACE.items()}
    return f"""
:root {{
{_variables()}
}}
[data-testid="stMainBlockContainer"] {{ max-width: {CONTENT_MAX_WIDTH}; }}
h1, h2, h3 {{ letter-spacing: -0.012em; }}
.sa-eyebrow {{
  color: var(--sa-text-subtle);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: -0.35rem;
}}
/* Streamlit draws captions at 60% opacity (below 4.5:1 on these surfaces):
   render them opaque in the muted text token instead. */
[data-testid="stCaptionContainer"] {{ opacity: 1; color: var(--sa-text-muted); }}
/* A visible keyboard focus ring in the accent (6.7:1 on the canvas). */
a:focus-visible, button:focus-visible, summary:focus-visible,
[role="combobox"]:focus-visible, [role="slider"]:focus-visible, input:focus-visible {{
  outline: 2px solid var(--sa-accent);
  outline-offset: 2px;
}}
[data-testid="stMetric"] {{
  background: var(--sa-surface);
  border: 1px solid var(--sa-border);
  border-radius: var(--sa-radius-card);
  padding: {space["md"]} {space["lg"]};
}}
[data-testid="stMetricLabel"] {{ color: var(--sa-text-muted); }}
[class*="st-key-sa-card"] [data-testid="stMetric"] {{
  background: transparent;
  border: none;
  border-radius: 0;
  padding: {space["xs"]} 0;
}}
[data-testid="stMetricValue"], [data-testid="stTable"], .sa-num {{
  font-variant-numeric: tabular-nums;
}}
[class*="st-key-sa-card"] {{
  background: var(--sa-surface);
  border: 1px solid var(--sa-border);
  border-radius: var(--sa-radius-card);
  padding: {space["md"]} {space["lg"]};
}}
[class*="st-key-sa-future"] {{
  background: var(--sa-future-bg);
  border: 1px dashed var(--sa-future-border);
  border-radius: var(--sa-radius-card);
  padding: {space["md"]} {space["lg"]};
  color: var(--sa-future-fg);
}}
[data-testid="stPlotlyChart"] {{
  background: var(--sa-surface);
  border: 1px solid var(--sa-border);
  border-radius: var(--sa-radius-card);
  padding: {space["xs"]};
  animation: sa-appear var(--sa-motion-appear) var(--sa-ease);
}}
[data-testid="stExpander"] details {{
  background: var(--sa-surface);
  border-color: var(--sa-border);
  border-radius: var(--sa-radius-card);
}}
[data-testid="stExpander"] summary {{
  transition: background-color var(--sa-motion-fast) var(--sa-ease);
}}
[data-testid="stExpander"] summary:hover {{ background: var(--sa-surface-muted); }}
[data-testid="stAlert"] {{ border-radius: var(--sa-radius-card); }}
code {{ font-family: var(--sa-mono); }}
.sa-pill {{
  display: inline-flex;
  align-items: center;
  gap: {space["xs"]};
  padding: 0.18rem 0.65rem;
  border: 1px solid;
  border-radius: {RADIUS["pill"]};
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}}
{pills}
@keyframes sa-appear {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
/* Only the dashboard's own motion; Streamlit handles its own widgets. */
@media (prefers-reduced-motion: reduce) {{
  [data-testid="stPlotlyChart"], [data-testid="stExpander"] summary {{
    animation: none !important;
    transition: none !important;
  }}
}}
""".strip()


def _channel(value: int) -> float:
    scaled = value / 255
    return scaled / 12.92 if scaled <= 0.03928 else ((scaled + 0.055) / 1.055) ** 2.4


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio of two #RRGGBB colours."""

    def luminance(colour: str) -> float:
        red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
        return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)
