"""
The Portfolio Lab's editorial variant of the shared design system:
WCAG contrast of its light and dark tokens and chart identities, the bundled
OFL fonts and their licence, the script-level Streamlit config generated
from the tokens (matching light and dark variants, with no research value
leaking in), local-only font serving, CSS that follows the viewer's active
theme, charts drawn in the viewer's colour scheme with visible hatching,
unselected radio rings at 3:1 or more, motion that respects
prefers-reduced-motion, the phone layout rules, and backward compatibility
of the shared components the research dashboard and the Model Health Center
use.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

from stock_agent.app import figures, stock_figures, synthetic
from stock_agent.stock_lab.contracts import StableBand
from stock_agent.ui import components, style, theme

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "stock_agent" / "app"
LAB_CONFIG = APP / ".streamlit" / "config.toml"
RESEARCH_CONFIG = REPO / ".streamlit" / "config.toml"
FONTS = APP / "static" / theme.FONT_DIRECTORY
SURFACES = ("canvas", "surface", "surface_muted")


# --- tokens ---


def test_the_lab_tokens_name_the_same_roles_as_the_research_tokens():
    assert set(theme.LAB_COLOURS) == set(theme.COLOURS) == set(theme.LAB_DARK_COLOURS)
    assert set(theme.LAB_PALETTES) == set(theme.LAB_STATUS) == set(theme.SCHEMES)
    assert theme.LAB_PALETTES["light"] is theme.LAB_COLOURS
    assert theme.LAB_STATUS["light"] is theme.STATUS


def test_the_dark_status_treatments_mirror_the_light_ones():
    assert list(theme.LAB_DARK_STATUS) == list(theme.STATUS)
    for level, tones in theme.LAB_DARK_STATUS.items():
        assert list(tones) == list(theme.STATUS[level]), level
    # One hue per status in the dark too, so Unavailable never reads as Warning.
    assert len({tones["fg"] for tones in theme.LAB_DARK_STATUS.values()}) == len(theme.STATUS)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
@pytest.mark.parametrize("surface", SURFACES)
def test_lab_text_and_accent_meet_wcag_aa_on_every_surface(scheme, surface):
    colours = theme.LAB_PALETTES[scheme]
    for text in ("text", "text_muted", "text_subtle", "accent"):
        ratio = theme.contrast_ratio(colours[text], colours[surface])
        assert ratio >= 4.5, (scheme, text, surface, ratio)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_lab_muted_text_and_soft_accents_stay_legible(scheme):
    colours = theme.LAB_PALETTES[scheme]
    for tones in theme.LAB_STATUS[scheme].values():  # captions sit on alert backgrounds too
        assert theme.contrast_ratio(colours["text_muted"], tones["bg"]) >= 4.5
    assert theme.contrast_ratio(colours["accent"], colours["accent_soft"]) >= 4.5  # chosen tags
    assert theme.contrast_ratio(colours["accent"], colours["canvas"]) >= 3.0  # the focus ring


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_lab_status_treatments_are_legible_in_both_schemes(scheme):
    for level, tones in theme.LAB_STATUS[scheme].items():
        assert theme.contrast_ratio(tones["fg"], tones["bg"]) >= 4.5, (scheme, level)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_chart_identities_keep_3_to_1_contrast_on_the_lab_chart_surface(scheme):
    surface = theme.LAB_PALETTES[scheme]["surface"]
    colours = figures.FUND_PALETTES[scheme]
    assert len(set(colours)) == len(figures.FUND_COLOURS) == 5
    for colour in (*colours, theme.LAB_PALETTES[scheme]["reference_line"]):
        assert theme.contrast_ratio(colour, surface) >= 3.0, (scheme, colour)


def test_the_lab_chart_layout_uses_the_editorial_tokens_and_never_tweens():
    layout = go.Figure().update_layout(**style.lab_layout("t", subtitle="s")).layout
    assert layout.font.family == theme.LAB_TYPE["sans"]
    assert layout.plot_bgcolor == layout.paper_bgcolor == theme.LAB_COLOURS["surface"]
    assert layout.xaxis.gridcolor == theme.LAB_COLOURS["grid"]
    assert layout.transition.duration is None
    # The research layout is untouched by the Lab's.
    research = style.base_layout("t")
    assert research["font"]["family"] == theme.TYPE["sans"]
    assert research["plot_bgcolor"] == theme.COLOURS["surface"]


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_the_lab_chart_layout_draws_in_the_scheme_it_is_given(scheme):
    colours = theme.LAB_PALETTES[scheme]
    layout = go.Figure().update_layout(**style.lab_layout("t", subtitle="s", scheme=scheme)).layout
    assert layout.plot_bgcolor == layout.paper_bgcolor == colours["surface"]
    assert layout.font.color == colours["text"]
    assert layout.title.subtitle.font.color == colours["text_muted"]
    for axis in (layout.xaxis, layout.yaxis):
        assert axis.gridcolor == axis.zerolinecolor == colours["grid"]
        assert axis.linecolor == colours["border_strong"]
    assert style.lab_layout("t") == style.lab_layout("t", scheme="light")


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_the_explore_figures_draw_in_the_scheme_they_are_given(scheme):
    colours, identities = theme.LAB_PALETTES[scheme], figures.FUND_PALETTES[scheme]
    history = synthetic.demo_histories()[0]
    growth = pd.Series(
        [10_000.0, 10_200.0], index=pd.DatetimeIndex(["2021-01-04", "2021-01-05"], tz="UTC")
    )
    styles = figures.fund_styles([history], scheme)
    assert styles[history.fund_id].colour == identities[0]
    drawn = figures.growth_figure([(history, growth)], styles, start_value=1e4, scheme=scheme)
    assert drawn.layout.paper_bgcolor == colours["surface"]
    assert drawn.data[0].line.color == drawn.data[0].marker.color == identities[0]
    assert drawn.layout.shapes[0].line.color == colours["reference_line"]
    assert drawn.layout.annotations[0].font.color == colours["text_muted"]
    table = pd.DataFrame(
        {"year": [2021], "total_return": [0.02], "partial": [True], "losing": [False]}
    )
    calendar = figures.calendar_year_figure([(history, table)], styles, scheme=scheme)
    assert calendar.layout.paper_bgcolor == colours["surface"]
    assert calendar.data[0].marker.pattern.fgcolor == colours["surface"]  # hatching
    assert calendar.layout.yaxis.zerolinecolor == colours["reference_line"]
    # Light stays the default for any caller that names no scheme.
    assert figures.fund_styles([history]) == figures.fund_styles([history], "light")


def _hex_colours(value) -> set[str]:
    if isinstance(value, dict):
        return set().union(*map(_hex_colours, value.values()))
    if isinstance(value, list | tuple):
        return set().union(*map(_hex_colours, value))
    is_hex = isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value)
    return {value.upper()} if is_hex else set()


def _scheme_only(scheme: str) -> set[str]:
    """Colours of `scheme`'s palettes that the other scheme never uses."""

    def palette(name):
        return {
            *theme.LAB_PALETTES[name].values(),
            *theme.DIRECTION_PALETTES[name].values(),
            *figures.FUND_PALETTES[name],
        }

    other = "dark" if scheme == "light" else "light"
    research = set(theme.COLOURS.values()) if scheme == "light" else set()
    return {colour.upper() for colour in (palette(scheme) | research) - palette(other)}


def _lab_figures(scheme: str) -> dict[str, go.Figure]:
    """Every Lab figure builder, on small frames, in `scheme`."""

    history = synthetic.demo_histories()[0]
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], tz="UTC")
    styles = figures.fund_styles([history], scheme)
    calendar = pd.DataFrame(
        {"year": [2023, 2024], "total_return": [0.1, -0.02], "partial": [True, False]}
    ).assign(losing=lambda frame: frame["total_return"] < 0)
    five = pd.DataFrame(
        {
            "category": "increase",
            "horizon": [1, 2],
            "usual": [0.3, 0.3],
            "probability": [0.32, 0.31],
            "range_low": [0.25, None],
            "range_high": [0.4, None],
            "session": ["2025-01-02", "2025-01-03"],
        }
    )
    profile = pd.DataFrame(
        {
            "length": [1, 2],
            "fell": [0.4, 0.5],
            "fell_low": [0.3, 0.2],
            "fell_high": [0.5, 0.8],
            "events": [120, 12],
            "status": ["ok", "thin"],
        }
    )
    reliability = pd.DataFrame(
        {
            "model": ["M1", "B1"],
            "category": [2, 2],
            "mean_predicted": [0.3, 0.32],
            "observed": [0.31, 0.3],
            "ci_low": [0.25, 0.24],
            "ci_high": [0.37, 0.36],
            "n": [400, 400],
            "bin_low": [0.2, 0.2],
            "bin_high": [0.4, 0.4],
        }
    )
    recent = pd.DataFrame(
        {
            "session": sessions,
            "price_return": [0.01, -0.012, 0.001],
            "category": ["increase", "decrease", "stable"],
        }
    )
    price = pd.DataFrame({"session": sessions, "display_level": [10.0, 10.2, 10.1]})
    observed = SimpleNamespace(  # the shape of inference.recent.RecentPrices that it reads
        sessions=tuple(
            SimpleNamespace(session=f"2026-10-{day}", level=level)
            for day, level in ((13, 10.0), (14, None), (15, 10.1))
        ),
        forecast_as_of="2026-10-14",
        retrieved_at="2026-10-16T00:30:00.000000Z",
    )
    growth = pd.Series([1e4, 1.01e4, 1.02e4], index=sessions)
    kw = {"synthetic": True, "scheme": scheme}
    return {
        "growth": figures.growth_figure(
            [(history, growth)], styles, start_value=1e4, scheme=scheme
        ),
        "calendar": figures.calendar_year_figure([(history, calendar)], styles, scheme=scheme),
        "price": stock_figures.price_figure(price, **kw),
        "observed": stock_figures.observed_figure(observed, scheme=scheme),
        "recent": stock_figures.recent_figure(recent, StableBand(), **kw),
        "five": stock_figures.five_session_figure(five, "increase", **kw),
        "streak": stock_figures.streak_figure(profile, "increase", 1, **kw),
        "reliability": stock_figures.reliability_figure(reliability, "M1", "B1", 2, **kw),
        "counts": stock_figures.forecast_count_figure(reliability, "M1", 2, **kw),
    }


def _hatch_background(marker) -> str | None:
    """The colour Plotly fills behind a bar's hatch lines (None: transparent)."""

    pattern = marker.pattern
    if pattern.bgcolor is not None:
        return pattern.bgcolor
    # "overlay" draws the lines over the bar colour; Plotly's default, "replace",
    # draws them over a transparent background instead.
    return marker.color if pattern.fillmode == "overlay" else None


# The fixtures in _lab_figures: a partial year first, a thin sample second.
HATCHED = {"calendar": ["/", ""], "streak": ["", "/"]}


@pytest.mark.parametrize("scheme", theme.SCHEMES)
@pytest.mark.parametrize("name", sorted(HATCHED))
def test_hatched_bars_are_visibly_hatched_and_the_others_stay_solid(scheme, name):
    marker = _lab_figures(scheme)[name].data[0].marker
    assert list(marker.pattern.shape) == HATCHED[name]  # "" draws a solid bar
    # Surface-coloured lines on a transparent fill vanish into the chart, leaving
    # the hatched bar an empty outline (or nothing, without an outline).
    background = _hatch_background(marker)
    assert background == marker.color, (scheme, name)  # the bar keeps its colour
    assert marker.pattern.fgcolor == theme.LAB_PALETTES[scheme]["surface"]
    ratio = theme.contrast_ratio(marker.pattern.fgcolor, background)
    assert ratio >= 3.0, (scheme, name, ratio)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_every_lab_figure_draws_only_in_its_own_scheme(scheme):
    surface = theme.LAB_PALETTES[scheme]["surface"]
    other = _scheme_only("dark" if scheme == "light" else "light")
    for name, figure in _lab_figures(scheme).items():
        assert figure.layout.paper_bgcolor == figure.layout.plot_bgcolor == surface, name
        leaked = _hex_colours(figure.to_plotly_json()) & other
        assert not leaked, (scheme, name, leaked)


def test_every_lab_figure_builder_takes_the_colour_scheme():
    builders = [
        getattr(module, name)
        for module in (figures, stock_figures)
        for name in dir(module)
        if name.endswith("_figure") and callable(getattr(module, name))
    ]
    assert len(builders) == len(_lab_figures("light"))  # every builder is exercised above
    for builder in builders:
        parameter = inspect.signature(builder).parameters.get("scheme")
        assert parameter is not None and parameter.default == "light", builder.__name__


@pytest.mark.parametrize(
    ("reported", "scheme"), [("dark", "dark"), ("light", "light"), (None, "light")]
)
def test_the_colour_scheme_is_the_viewers_active_theme(monkeypatch, reported, scheme):
    # st.context.theme.type is None outside a browser session (and in AppTest).
    monkeypatch.setattr(
        components.st, "context", SimpleNamespace(theme=SimpleNamespace(type=reported))
    )
    assert components.colour_scheme() == scheme


# --- fonts, licence and local serving ---


def test_the_bundled_fonts_are_woff2_files_with_their_ofl_licence():
    for face in theme.FONT_FACES:
        path = FONTS / face["file"]
        assert path.is_file(), face
        assert path.read_bytes()[:4] == b"wOF2", face
    licence = (FONTS / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007" in licence
    assert licence.count("with Reserved Font Name ‘Source’") == 2
    for section in ("PREAMBLE", "DEFINITIONS", "PERMISSION & CONDITIONS", "TERMINATION"):
        assert section in licence
    assert "Source Serif 4, version 4.005" in licence
    assert "Source Sans 3,\n  version 3.052" in licence


def test_font_family_aliases_are_valid_unquoted_css_identifiers():
    # Streamlit writes a font face's family unquoted; "Source Serif 4" would be
    # dropped by the browser (a token may not start with a digit), and the page
    # would silently fall back to another font.
    for face in theme.FONT_FACES:
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", face["family"]), face["family"]
    assert theme.LAB_TYPE["serif"].startswith(f"{theme.SERIF},")
    assert theme.LAB_TYPE["sans"].startswith(f"{theme.SANS},")


def test_the_lab_config_is_generated_from_the_tokens_and_serves_fonts_locally():
    config = tomllib.loads(LAB_CONFIG.read_text(encoding="utf-8"))
    expected = theme.lab_streamlit_theme()
    assert config["theme"] == {
        **expected["theme"],
        **{
            scheme: {**expected[f"theme.{scheme}"], "sidebar": expected[f"theme.{scheme}.sidebar"]}
            for scheme in theme.SCHEMES
        },
        "fontFaces": expected["theme.fontFaces"],
    }
    assert config["server"]["enableStaticServing"] is True
    for face in config["theme"]["fontFaces"]:
        assert face["url"].startswith(f"{theme.STATIC_URL}/{theme.FONT_DIRECTORY}/")
        assert (APP / "static" / face["url"].removeprefix(f"{theme.STATIC_URL}/")).is_file()
    families = {face["family"] for face in config["theme"]["fontFaces"]}
    assert config["theme"]["headingFont"].split(",")[0] in families
    assert config["theme"]["font"].split(",")[0] in families


def test_the_lab_config_offers_matching_light_and_dark_variants():
    tables = theme.lab_streamlit_theme()
    light, dark = tables["theme.light"], tables["theme.dark"]
    # A key missing from one variant would silently fall back to [theme] there.
    assert list(light) == list(dark)
    assert list(tables["theme.light.sidebar"]) == list(tables["theme.dark.sidebar"])
    # [theme] holds type and shape only: no colour, and no base, which would
    # pin one variant (Streamlit forces each variant's base itself).
    assert "base" not in tables["theme"] and not set(tables["theme"]) & set(light)
    assert "theme.sidebar" not in tables
    for scheme, table in (("light", light), ("dark", dark)):
        colours = theme.LAB_PALETTES[scheme]
        assert table["backgroundColor"] == colours["canvas"]
        assert table["textColor"] == colours["text"]
        assert table["primaryColor"] == colours["accent"]
        assert table["redBackgroundColor"] == theme.LAB_STATUS[scheme]["INVALID/BLOCKED"]["bg"]
        sidebar = tables[f"theme.{scheme}.sidebar"]
        assert sidebar["backgroundColor"] == colours["surface_muted"]


def test_no_research_theme_value_reaches_the_lab():
    # Streamlit reads the repository's config (the research theme, which the
    # launcher's working directory picks up) and then the Lab's, option by
    # option. Every research option must be overridden in [theme] or in both
    # variants; the research base ("light") is overridden by each variant.
    research = tomllib.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))["theme"]
    lab = tomllib.loads(LAB_CONFIG.read_text(encoding="utf-8"))["theme"]
    for key in set(research) - {"base", "sidebar"}:
        assert key in lab or all(key in lab[scheme] for scheme in theme.SCHEMES), key
    for key in research["sidebar"]:
        assert all(key in lab[scheme]["sidebar"] for scheme in theme.SCHEMES), key


def test_the_lab_css_follows_the_viewers_active_theme():
    css = theme.lab_css()
    root = css.split(":root {")[1].split("}")[0]
    for name, light in theme.LAB_COLOURS.items():
        dark = theme.LAB_DARK_COLOURS[name]
        assert f"--sa-{name.replace('_', '-')}: light-dark({light}, {dark});" in root, name
    for level, tones in theme.STATUS.items():
        for part, light in tones.items():
            dark = theme.LAB_DARK_STATUS[level][part]
            variable = f"--sa-{theme.status_slug(level)}-{part}"
            assert f"{variable}: light-dark({light}, {dark});" in root, variable
    # Every colour has both values, so nothing stays light on a dark page.
    paired = re.sub(r"light-dark\(#[0-9A-F]{6}, #[0-9A-F]{6}\)", "", css)
    assert not re.search(r"#[0-9A-Fa-f]{3,8}\b", paired)
    # light-dark() reads the color-scheme Streamlit sets from the active theme;
    # the operating system's preference never overrides a Settings choice.
    assert "prefers-color-scheme" not in css and "color-scheme:" not in css
    # The research dashboard stays light.
    assert "light-dark(" not in theme.css()


# Streamlit 1.64's radio option: a label (data-selected / data-disabled when
# set) holding a visually hidden input and a row whose first child is the ring.
RADIO_RING = (
    '[data-testid="stRadioOption"]:not([data-selected]):not([data-disabled])'
    " > div > div:first-child"
)


def _radio_ring_role() -> str:
    css = " ".join(theme.lab_css().split())
    assert css.count(f"{RADIO_RING} {{") == 1
    rule = css.split(f"{RADIO_RING} {{")[1].split("}")[0]
    variable = re.fullmatch(r" background-color: var\(--sa-([a-z-]+)\); ", rule)
    assert variable, rule
    return variable.group(1).replace("-", "_")


@pytest.mark.parametrize("scheme", theme.SCHEMES)
@pytest.mark.parametrize("surface", SURFACES)
def test_unselected_radio_rings_meet_3_to_1_on_every_surface(scheme, surface):
    # WCAG 1.4.11: a control's boundary needs 3:1 against what is next to it.
    # Streamlit draws the ring in borderColor (1.5:1 at most); the ring sits on a
    # card, the canvas or the sidebar and around a disc in the page background.
    colours = theme.LAB_PALETTES[scheme]
    ratio = theme.contrast_ratio(colours[_radio_ring_role()], colours[surface])
    assert ratio >= 3.0, (scheme, surface, ratio)


def test_only_the_radio_rings_change_and_other_widget_borders_keep_their_colour():
    assert _radio_ring_role() != "border"
    tables = theme.lab_streamlit_theme()
    for scheme in theme.SCHEMES:
        border = theme.LAB_PALETTES[scheme]["border"]
        assert tables[f"theme.{scheme}"]["borderColor"] == border
        assert tables[f"theme.{scheme}.sidebar"]["borderColor"] == border


def test_no_style_reaches_another_host():
    sources = [theme.lab_css(), theme.css(), LAB_CONFIG.read_text(encoding="utf-8")]
    for source in sources:
        assert "@import" not in source and "googleapis" not in source
        assert not re.search(r"url\(\s*['\"]?https?:", source)
    for face in theme.lab_streamlit_theme()["theme.fontFaces"]:
        assert not face["url"].startswith(("http:", "https:", "//"))


def test_the_research_dashboard_keeps_its_own_theme():
    research = theme.css()
    assert theme.LAB_COLOURS["accent"] not in research
    assert "--sa-serif" not in research
    assert theme.streamlit_theme()["theme"]["headingFont"] == theme.TYPE["sans"]


# --- motion and layout ---


def test_lab_motion_is_opacity_only_and_stops_under_reduced_motion():
    css = theme.lab_css()
    assert "infinite" not in css
    keyframes = css.split("@keyframes sa-appear")[1].split("}}")[0]
    assert "opacity" in keyframes and "transform" not in keyframes
    reduced = css.split("@media (prefers-reduced-motion: reduce)")[-1]
    for rule in ("animation-duration: 0.01ms !important", "transition-duration: 0.01ms !important"):
        assert rule in reduced
    assert "*, *::before, *::after" in reduced  # Streamlit's own widget motion too


def test_the_lab_has_a_phone_layout_and_visible_focus():
    css = theme.lab_css()
    phone = css.split(f"@media (max-width: {theme.PHONE_MAX_WIDTH})")[1]
    assert ".sa-list__item { grid-template-columns: 1fr; }" in phone
    assert "padding-left: 16px" in phone
    assert ":focus-visible" in css and "outline: 2px solid var(--sa-accent)" in css
    for hook in ("stTopNavLink", "stSidebarNavLink", "stPageLink-NavLink"):
        assert f'[data-testid="{hook}"][href]:focus-visible' in css, hook


def test_the_lab_layer_is_applied_once_and_the_default_stays_research(monkeypatch):
    calls = []
    monkeypatch.setattr(components.st, "html", calls.append)
    components.apply_theme()
    components.apply_theme("lab")
    assert calls[0] == f"<style>{theme.css()}</style>"
    assert calls[1] == f"<style>{theme.lab_css()}</style>"


# --- components: backward compatible, escaped ---

# Signatures the research dashboard and the Model Health Center call today.
SIGNATURES = {
    "page_header": "(title: 'str', *, eyebrow: 'str | None' = None, purpose: 'str | None' = None)",
    "pill_html": "(level: 'str', text: 'str | None' = None, *, live: 'bool' = False)",
    "pill": "(level: 'str', text: 'str | None' = None, *, where=None, live: 'bool' = False)",
    "pills": "(items: 'Sequence[tuple[str, str]]', *, where=None)",
    "status_banner": "(level: 'str', reasons: 'Sequence[str]') -> 'None'",
    "stat_row": "(stats: 'Sequence[Stat]') -> 'None'",
    "card": "(key: 'str')",
    "state_card": "(key: 'str')",
    "section": "(title: 'str', caption: 'str | None' = None) -> 'None'",
    "notice": "(level: 'str', text: 'str', *, where=None) -> 'None'",
    "chart": "(figure, key: 'str') -> 'None'",
    "download": "(frame: 'pd.DataFrame', name: 'str', run_id: 'str') -> 'None'",
    "future_state": "(key: 'str', title: 'str', reason: 'str') -> 'None'",
}


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_shared_component_signatures_are_unchanged(name):
    signature = str(inspect.signature(getattr(components, name)))
    assert signature.startswith(SIGNATURES[name].split(" ->")[0]), signature


def test_estimate_card_and_base_layout_keep_their_parameters():
    estimate = inspect.signature(components.estimate_card).parameters
    assert list(estimate) == ["where", "label", "row", "fmt", "help", "method", "note"]
    layout = inspect.signature(style.base_layout).parameters
    assert list(layout) == ["title", "x_title", "y_title", "height", "subtitle"]
    assert list(inspect.signature(components.apply_theme).parameters) == ["variant"]


def test_status_lists_escape_their_text_and_spell_out_every_status(monkeypatch):
    calls = []
    monkeypatch.setattr(components.st, "html", calls.append)
    components.status_list([components.Item("WARNING", "Stale", "<b>Fund</b>", "a < b & c")])
    html = calls[0]
    assert "&lt;b&gt;Fund&lt;/b&gt;" in html and "a &lt; b &amp; c" in html
    assert 'class="sa-pill sa-pill--warning">Stale</span>' in html


def test_ledes_escape_their_text(monkeypatch):
    calls = []
    monkeypatch.setattr(components.st, "html", calls.append)
    components.lede("<script>x</script>")
    assert calls == ['<p class="sa-lede">&lt;script&gt;x&lt;/script&gt;</p>']


# --- launcher ---


def _launcher():
    path = REPO / "scripts" / "run_portfolio_lab.py"
    spec = importlib.util.spec_from_file_location("stock_agent_run_portfolio_lab_theme", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launcher_runs_this_checkouts_code(monkeypatch):
    module = _launcher()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/elsewhere", str(module.SOURCE)]))
    paths = module.environment()["PYTHONPATH"].split(os.pathsep)
    assert paths == [str(module.SOURCE), "/elsewhere"]  # this checkout first, no duplicate
    monkeypatch.delenv("PYTHONPATH")
    assert module.environment()["PYTHONPATH"] == str(module.SOURCE)
    assert module.APP == APP / "run.py"


def test_the_launcher_leaves_the_colour_scheme_to_the_viewer():
    # The theme comes only from the generated config; a --theme flag would override it.
    assert not [part for part in _launcher().command(8502) if part.startswith("--theme.")]
