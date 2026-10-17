"""
Design tokens shared by the research dashboard and the product app: the
single source of their look.

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

Two variants share these rules. The research dashboard uses the tokens
below (`COLOURS`, `TYPE`, `css()`, `streamlit_theme()`). The Portfolio Lab
uses the editorial variant (`LAB_COLOURS`, `LAB_TYPE`, `lab_css()`,
`lab_streamlit_theme()`): warm ivory surfaces, charcoal text, one restrained
terracotta accent, Source Serif 4 headings and Source Sans 3 for text,
tables and controls. Its fonts are bundled OFL files served by the Lab's own
Streamlit static directory (`FONT_FACES`), so nothing is fetched from a CDN.

The Lab also has a dark colour scheme (`LAB_DARK_COLOURS`,
`LAB_DARK_STATUS`; `LAB_PALETTES` and `LAB_STATUS` by scheme) with the same
roles and the same contrast rules. Streamlit's Settings menu offers System,
Light and Dark; System follows the operating system. The research dashboard
stays light.
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
    # A measurement that could not be taken (Model Health): its own hue, so it is
    # never read as the amber of a measured warning.
    "UNAVAILABLE": {"fg": "#5B3A8C", "bg": "#EFEAF6", "border": "#CFC1E3"},
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

# Model identity for T4's linear models: a palette disjoint from the T2 model
# colours (Okabe-Ito) and the category palette, each >= 4.5:1 on white so a
# line reads on the chart surface. Fixed names first, then a hashed fallback.
CANDIDATE_COLOURS = {
    "ridge": "#6A3D9A",
    "lasso": "#1B7F79",
    "elastic_net": "#B15928",
    "ols": "#5E6B2B",
}
CANDIDATE_FALLBACK = ("#8C2D04", "#2B5C8A")

# The Stock Lab's direction categories: Okabe-Ito blue and vermillion (a
# colour-blind-safe pair, never red and green) and a warm grey, each at least
# 3:1 on the Lab's chart surface, and always paired with a marker or pattern
# so colour never carries the category alone.
DIRECTION_COLOURS = {"increase": "#0072B2", "stable": "#6E675C", "decrease": "#C04F00"}
# On the dark chart surface the blue and the grey fall below 3:1: Okabe-Ito sky
# blue, a light warm grey and the Okabe-Ito vermillion keep each hue's meaning.
DIRECTION_DARK_COLOURS = {"increase": "#56B4E9", "stable": "#A39A8C", "decrease": "#D55E00"}
DIRECTION_PALETTES = {"light": DIRECTION_COLOURS, "dark": DIRECTION_DARK_COLOURS}
DIRECTION_MARKERS = {"increase": "triangle-up", "stable": "circle", "decrease": "triangle-down"}
DIRECTION_PATTERNS = {"increase": "", "stable": ".", "decrease": "/"}

SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
RADIUS = {"control": "8px", "card": "12px", "pill": "999px"}
MOTION = {"fast_ms": 120, "appear_ms": 200, "easing": "cubic-bezier(0.2, 0, 0, 1)"}
CONTENT_MAX_WIDTH = "1240px"

# --- the Portfolio Lab's editorial variant ---
# Same token names as COLOURS, so every shared rule reads either set. Warm
# ivory and off-white surfaces, deep charcoal text and one terracotta accent
# (links, focus, selection, a hairline beside eyebrows), each tested for WCAG
# AA on every surface.
LAB_COLOURS = {
    "canvas": "#FAF7F1",  # warm ivory page
    "surface": "#FFFDF8",  # off-white cards and charts
    "surface_muted": "#F2EDE4",  # sidebar, widget fills, table headers
    "border": "#E5DDD0",
    "border_strong": "#CDBFAE",
    "text": "#1D1B18",  # deep charcoal
    "text_muted": "#544E46",
    "text_subtle": "#685F55",
    "accent": "#9A4325",  # restrained terracotta
    "accent_soft": "#F4E5DB",
    "grid": "#EEE7DC",
    "fold_boundary": "#B3A898",
    "reference_line": "#8A8174",
}
# The Lab's dark scheme: the same roles over warm near-black surfaces, cards
# and charts a step lighter than the canvas, ivory text and a lighter
# terracotta. Each pair keeps the light scheme's contrast rules (tested), and
# the chart lines keep their light contrast (reference 3.9:1, fold 2.5:1).
LAB_DARK_COLOURS = {
    "canvas": "#1A1815",
    "surface": "#23201C",
    "surface_muted": "#2B2823",
    "border": "#3B362F",
    "border_strong": "#5A5248",
    "text": "#F3EEE6",
    "text_muted": "#CFC6B8",
    "text_subtle": "#B3A898",
    "accent": "#DE8C66",
    "accent_soft": "#3E2A20",
    "grid": "#34302A",
    "fold_boundary": "#645C51",
    "reference_line": "#857C70",
}
# Status treatments on the dark scheme: light tones on deep tints of the same
# hues, still one hue per status and always written out in words.
LAB_DARK_STATUS = {
    "VALID": {"fg": "#8FD3A0", "bg": "#1D3024", "border": "#3A6447"},
    "WARNING": {"fg": "#F2BC72", "bg": "#35281A", "border": "#715430"},
    "INVALID/BLOCKED": {"fg": "#F4A7AE", "bg": "#3C2023", "border": "#77404A"},
    "NOTICE": {"fg": "#97CAD7", "bg": "#1C2D32", "border": "#3E606A"},
    "FUTURE": {"fg": "#CFC6B8", "bg": "#2B2823", "border": "#5A5248"},
    "UNAVAILABLE": {"fg": "#CDB8EE", "bg": "#2B2339", "border": "#574873"},
}
SCHEMES = ("light", "dark")
LAB_PALETTES = {"light": LAB_COLOURS, "dark": LAB_DARK_COLOURS}
LAB_STATUS = {"light": STATUS, "dark": LAB_DARK_STATUS}
# CSS family names for the bundled Source Serif 4 and Source Sans 3. Streamlit
# writes a font face's family unquoted, and an unquoted family cannot contain a
# token that starts with a digit ("Source Serif 4" would be dropped by the
# browser), so the aliases are single identifiers.
SERIF = "SourceSerif4"
SANS = "SourceSans3"
LAB_TYPE = {
    "serif": f"{SERIF}, 'Iowan Old Style', Georgia, serif",
    "sans": f"{SANS}, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif",
    "mono": TYPE["mono"],
    "base_px": 16,
    "heading_sizes": ["2.55rem", "1.7rem", "1.3rem", "1.05rem", "1rem", "0.95rem"],
    "heading_weights": [600, 600, 600, 600, 600, 600],
    "metric_size": "1.65rem",
    "metric_weight": 600,
    "chart_title_px": 16,
    "chart_px": 13,
}
LAB_RADIUS = {"control": "10px", "card": "14px", "pill": "999px"}
LAB_CONTENT_MAX_WIDTH = "1120px"
# Bundled SIL Open Font License fonts (variable, unmodified Adobe releases:
# Source Serif 4 4.005, Source Sans 3 3.052; licence in the font directory).
# Streamlit serves them from the Lab's static directory, src/stock_agent/app/
# static/, at STATIC_URL; nothing is fetched from another host.
STATIC_URL = "app/static"
FONT_DIRECTORY = "fonts"
FONT_FACES = (
    {"family": SERIF, "file": "SourceSerif4Variable-Roman.woff2", "weight": "200 900"},
    {"family": SANS, "file": "SourceSans3VF-Upright.woff2", "weight": "200 900"},
    {"family": SANS, "file": "SourceSans3VF-Italic.woff2", "weight": "200 900", "style": "italic"},
)


def _theme_colours(colours, status) -> tuple[dict[str, object], dict[str, object]]:
    """A palette's Streamlit colour options: the main table and its sidebar table."""

    main = {
        "primaryColor": colours["accent"],
        "backgroundColor": colours["canvas"],
        "secondaryBackgroundColor": colours["surface_muted"],
        "textColor": colours["text"],
        "linkColor": colours["accent"],
        "borderColor": colours["border"],
        "dataframeBorderColor": colours["border"],
        "dataframeHeaderBackgroundColor": colours["surface_muted"],
    }
    for hue, level in (
        ("green", "VALID"),
        ("yellow", "WARNING"),
        ("orange", "WARNING"),
        ("red", "INVALID/BLOCKED"),
        ("blue", "NOTICE"),
    ):
        main[f"{hue}Color"] = status[level]["fg"]
        main[f"{hue}BackgroundColor"] = status[level]["bg"]
        main[f"{hue}TextColor"] = status[level]["fg"]
    main["grayColor"] = colours["text_subtle"]
    main["grayBackgroundColor"] = colours["surface_muted"]
    main["grayTextColor"] = colours["text_muted"]
    sidebar = {
        "backgroundColor": colours["surface_muted"],
        "secondaryBackgroundColor": colours["surface"],
        "borderColor": colours["border"],
    }
    return main, sidebar


def _theme_type(type_, radius) -> dict[str, object]:
    """Streamlit's type and shape options, the same in every colour scheme."""

    return {
        "showWidgetBorder": True,
        "baseRadius": radius["control"],
        "buttonRadius": radius["control"],
        "font": type_["sans"],
        "headingFont": type_.get("serif", type_["sans"]),
        "codeFont": type_["mono"],
        "baseFontSize": type_["base_px"],
        "headingFontSizes": type_["heading_sizes"],
        "headingFontWeights": type_["heading_weights"],
        "metricValueFontSize": type_["metric_size"],
        "metricValueFontWeight": type_["metric_weight"],
    }


def _streamlit_theme(colours, type_, radius) -> dict[str, dict[str, object]]:
    main, sidebar = _theme_colours(colours, STATUS)
    return {
        "theme": {"base": "light", **main, **_theme_type(type_, radius)},
        "theme.sidebar": sidebar,
    }


def streamlit_theme() -> dict[str, dict[str, object]]:
    """The [theme] and [theme.sidebar] tables of .streamlit/config.toml."""

    return _streamlit_theme(COLOURS, TYPE, RADIUS)


def font_url(face: dict[str, str]) -> str:
    """Where Streamlit serves a bundled font file (relative to the app's root URL)."""

    return f"{STATIC_URL}/{FONT_DIRECTORY}/{face['file']}"


def lab_streamlit_theme() -> dict[str, object]:
    """
    The Portfolio Lab's script-level config tables (src/stock_agent/app/
    .streamlit/config.toml, checked against this by a test): [theme] with
    type and shape only, every colour in both [theme.light] and [theme.dark]
    (and their sidebar tables), and one [[theme.fontFaces]] table per bundled
    font file.

    With both variants, Streamlit offers System, Light and Dark in Settings,
    merging [theme] with the chosen variant and forcing its base, so [theme]
    sets no base. Every colour is in both variants: one missing from a
    variant would fall back to [theme], which the repository's config (the
    research theme) fills underneath the Lab's.
    """

    tables: dict[str, object] = {"theme": _theme_type(LAB_TYPE, LAB_RADIUS)}
    for scheme in SCHEMES:
        main, sidebar = _theme_colours(LAB_PALETTES[scheme], LAB_STATUS[scheme])
        tables[f"theme.{scheme}"] = main
        tables[f"theme.{scheme}.sidebar"] = sidebar
    tables["theme.fontFaces"] = [
        {
            "family": face["family"],
            "url": font_url(face),
            "weight": face["weight"],
            "style": face.get("style", "normal"),
        }
        for face in FONT_FACES
    ]
    return tables


def _variables(colours=COLOURS, radius=RADIUS, mono=TYPE["mono"], status=STATUS) -> str:
    pairs = {
        **{f"--sa-{name.replace('_', '-')}": value for name, value in colours.items()},
        "--sa-radius-card": radius["card"],
        "--sa-radius-control": radius["control"],
        "--sa-motion-fast": f"{MOTION['fast_ms']}ms",
        "--sa-motion-appear": f"{MOTION['appear_ms']}ms",
        "--sa-ease": MOTION["easing"],
        "--sa-mono": mono,
    }
    for level, tones in status.items():
        slug = level.split("/")[-1].lower()
        for part, value in tones.items():
            pairs[f"--sa-{slug}-{part}"] = value
    return "\n".join(f"  {name}: {value};" for name, value in pairs.items())


def status_slug(level: str) -> str:
    """CSS modifier for a status level: valid, warning, blocked, notice, future, unavailable."""

    return level.split("/")[-1].lower()


def css() -> str:
    """The dashboard's CSS layer. Stable hooks only: data-testid and keyed containers."""

    return _shared_css(COLOURS, RADIUS, CONTENT_MAX_WIDTH, TYPE["mono"])


def _shared_css(colours, radius, max_width: str, mono: str, status=STATUS) -> str:
    """The rules both variants share, over one token set."""

    pills = "\n".join(
        f".sa-pill--{status_slug(level)} {{ color: var(--sa-{status_slug(level)}-fg); "
        f"background: var(--sa-{status_slug(level)}-bg); "
        f"border-color: var(--sa-{status_slug(level)}-border); }}"
        for level in STATUS
    )
    space = {name: f"{value}px" for name, value in SPACE.items()}
    return f"""
:root {{
{_variables(colours, radius, mono, status)}
}}
[data-testid="stMainBlockContainer"] {{ max-width: {max_width}; }}
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
  border-radius: {radius["pill"]};
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}}
.sa-pills {{ display: flex; flex-wrap: wrap; gap: {space["sm"]}; }}
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


PHONE_MAX_WIDTH = "640px"


def _light_dark(light: str, dark: str) -> str:
    return f"light-dark({light}, {dark})"


def lab_css() -> str:
    """
    The Portfolio Lab's CSS layer: the shared rules over the editorial tokens,
    then the editorial type, hero, lists, links and navigation. Every custom
    motion is opacity only; under prefers-reduced-motion all motion stops,
    Streamlit's own widget transitions included.

    Every colour variable holds both schemes as `light-dark(light, dark)`.
    Streamlit sets `color-scheme` on its app and sidebar containers from the
    active theme, and light-dark() resolves against it where a variable is
    used, so the layer follows a Settings choice (System, Light or Dark) at
    once, with no rerun and never against Streamlit's own widgets.
    """

    colours = {name: _light_dark(LAB_COLOURS[name], LAB_DARK_COLOURS[name]) for name in LAB_COLOURS}
    status = {
        level: {
            part: _light_dark(tone, LAB_DARK_STATUS[level][part]) for part, tone in tones.items()
        }
        for level, tones in STATUS.items()
    }
    shared = _shared_css(colours, LAB_RADIUS, LAB_CONTENT_MAX_WIDTH, LAB_TYPE["mono"], status)
    space = {name: f"{value}px" for name, value in SPACE.items()}
    return f"""
{shared}
:root {{
  --sa-serif: {LAB_TYPE["serif"]};
  --sa-sans: {LAB_TYPE["sans"]};
}}
[data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; padding-bottom: 4rem; }}
h1, h2, h3 {{ font-family: var(--sa-serif); font-weight: 600; letter-spacing: -0.015em; }}
h1 {{ line-height: 1.12; }}
h4, h5, h6 {{ font-family: var(--sa-sans); }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  line-height: 1.6;
  max-width: 76ch;
}}
.sa-eyebrow {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: -0.1rem;
}}
.sa-eyebrow::before {{
  content: "";
  width: 1.5rem;
  height: 1px;
  background: var(--sa-accent);
}}
.sa-lede {{
  font-family: var(--sa-serif);
  font-size: 1.2rem;
  line-height: 1.5;
  color: var(--sa-text-muted);
  max-width: 62ch;
  margin: 0;
}}
[class*="st-key-sa-hero"] {{
  padding-bottom: {space["xl"]};
  margin-bottom: {space["sm"]};
  border-bottom: 1px solid var(--sa-border);
  animation: sa-appear var(--sa-motion-appear) var(--sa-ease);
}}
[class*="st-key-sa-card"], [class*="st-key-sa-future"] {{
  padding: {space["lg"]} {space["xl"]};
  animation: sa-appear var(--sa-motion-appear) var(--sa-ease);
}}
[class*="st-key-sa-card"] {{ box-shadow: 0 1px 2px rgba(29, 27, 24, 0.04); }}
[data-testid="stHtml"] .sa-list {{ list-style: none; margin: 0; padding: 0; }}
.sa-list__item {{
  display: grid;
  grid-template-columns: minmax(9.5rem, max-content) 1fr;
  gap: {space["xs"]} {space["lg"]};
  align-items: baseline;
  padding: {space["md"]} 0;
  border-top: 1px solid var(--sa-border);
}}
.sa-list__item:first-child {{ border-top: none; padding-top: {space["xs"]}; }}
.sa-list__item .sa-pill {{ justify-self: start; }}
.sa-list__title {{ font-weight: 600; color: var(--sa-text); margin: 0; }}
.sa-list__text {{ color: var(--sa-text-muted); margin: 0.15rem 0 0; line-height: 1.55; }}
[data-testid="stPageLink"] a {{
  border: 1px solid var(--sa-border-strong);
  border-radius: {LAB_RADIUS["pill"]};
  padding: 0.35rem 0.95rem;
  background: var(--sa-surface);
  transition: background-color var(--sa-motion-fast) var(--sa-ease),
    border-color var(--sa-motion-fast) var(--sa-ease);
}}
[data-testid="stPageLink"] a:hover {{
  background: var(--sa-accent-soft);
  border-color: var(--sa-accent);
}}
[data-testid="stPageLink"] a p {{ font-weight: 600; white-space: normal; }}
[data-testid="stPageLink"] a span {{ overflow: visible; max-width: none; }}
[data-testid="stMultiSelectTagsContainer"] [data-tag] {{
  background-color: var(--sa-accent-soft);
  color: var(--sa-accent);
  border: 1px solid var(--sa-border-strong);
}}
[data-testid="stMultiSelectTagsContainer"] [data-tag] svg {{ fill: var(--sa-accent); }}
[data-testid="stTable"] th {{ color: var(--sa-text-muted); font-weight: 600; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ vertical-align: top; }}
/* Streamlit's navigation links drop the outline on focus: restore a visible ring. */
[data-testid="stTopNavLink"][href]:focus-visible,
[data-testid="stSidebarNavLink"][href]:focus-visible,
[data-testid="stPageLink-NavLink"][href]:focus-visible {{
  outline: 2px solid var(--sa-accent);
  outline-offset: 2px;
}}
[data-testid="stHeader"] {{
  background: var(--sa-canvas);
  border-bottom: 1px solid var(--sa-border);
}}
[data-testid="stCode"] pre {{ border-radius: var(--sa-radius-control); }}
@media (max-width: {PHONE_MAX_WIDTH}) {{
  [data-testid="stMainBlockContainer"] {{
    padding-left: {space["lg"]};
    padding-right: {space["lg"]};
    padding-top: 3.75rem;
  }}
  [data-testid="stHeading"] h1 {{ font-size: 2rem; }}
  [data-testid="stHeading"] h2 {{ font-size: 1.45rem; }}
  .sa-lede {{ font-size: 1.06rem; }}
  [class*="st-key-sa-card"], [class*="st-key-sa-future"] {{ padding: {space["md"]} {space["lg"]}; }}
  .sa-list__item {{ grid-template-columns: 1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
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
