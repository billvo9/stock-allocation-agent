"""
The Portfolio Lab's editorial variant of the shared design system:
WCAG contrast of its tokens and chart identities, the bundled OFL fonts and
their licence, the script-level Streamlit config generated from the tokens,
local-only font serving, motion that respects prefers-reduced-motion, the
phone layout rules, and backward compatibility of the shared components the
research dashboard and the Model Health Center use.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import tomllib
from pathlib import Path

import plotly.graph_objects as go
import pytest

from stock_agent.app import figures
from stock_agent.ui import components, style, theme

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "stock_agent" / "app"
LAB_CONFIG = APP / ".streamlit" / "config.toml"
FONTS = APP / "static" / theme.FONT_DIRECTORY
SURFACES = ("canvas", "surface", "surface_muted")


# --- tokens ---


def test_the_lab_tokens_name_the_same_roles_as_the_research_tokens():
    assert set(theme.LAB_COLOURS) == set(theme.COLOURS)


@pytest.mark.parametrize("surface", SURFACES)
def test_lab_text_and_accent_meet_wcag_aa_on_every_surface(surface):
    for text in ("text", "text_muted", "text_subtle", "accent"):
        ratio = theme.contrast_ratio(theme.LAB_COLOURS[text], theme.LAB_COLOURS[surface])
        assert ratio >= 4.5, (text, surface, ratio)


def test_lab_muted_text_and_soft_accents_stay_legible():
    colours = theme.LAB_COLOURS
    for tones in theme.STATUS.values():  # captions sit on alert backgrounds too
        assert theme.contrast_ratio(colours["text_muted"], tones["bg"]) >= 4.5
    assert theme.contrast_ratio(colours["accent"], colours["accent_soft"]) >= 4.5  # chosen tags
    assert theme.contrast_ratio(colours["accent"], colours["canvas"]) >= 3.0  # the focus ring


def test_chart_identities_keep_3_to_1_contrast_on_the_lab_chart_surface():
    for colour in figures.FUND_COLOURS:
        assert theme.contrast_ratio(colour, theme.LAB_COLOURS["surface"]) >= 3.0, colour


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
        "sidebar": expected["theme.sidebar"],
        "fontFaces": expected["theme.fontFaces"],
    }
    assert config["server"]["enableStaticServing"] is True
    for face in config["theme"]["fontFaces"]:
        assert face["url"].startswith(f"{theme.STATIC_URL}/{theme.FONT_DIRECTORY}/")
        assert (APP / "static" / face["url"].removeprefix(f"{theme.STATIC_URL}/")).is_file()
    families = {face["family"] for face in config["theme"]["fontFaces"]}
    assert config["theme"]["headingFont"].split(",")[0] in families
    assert config["theme"]["font"].split(",")[0] in families


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
