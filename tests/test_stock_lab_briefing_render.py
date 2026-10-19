"""
The briefing's plain-text and HTML renderings: synthetic data labelled at
the top, no number for a stock without a stored forecast, a self-contained
page (nothing fetched, no local dashboard link) in the Lab's palette with
legible contrast in light and dark, and no advice or trading wording in any
state. Synthetic fixtures.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from stock_lab_briefing_helpers import Lab, build, ny

from stock_agent.stock_lab.briefing.render import render_html, render_text
from stock_agent.ui import theme

ROOT = Path(__file__).resolve().parents[1]
MONDAY = ny("2026-10-19", "08:30")
THURSDAY, FRIDAY = "2026-10-15", "2026-10-16"
SIGNALS = {
    "current_run": {"direction": "decrease", "length": 2},
    "after_this_streak": {
        "definition": "2 falling sessions in a row",
        "events": 80,
        "outcomes": {
            "continuation": {"estimate": 0.35, "n_eff": 80.0, "status": "ok"},
            "stabilization": {"estimate": 0.3, "n_eff": 80.0, "status": "ok"},
            "reversal": {"estimate": 0.35, "n_eff": 80.0, "status": "ok"},
        },
    },
    "bull_trap": {
        "setup_now": False,
        "setup_session": None,
        "historical_failure_rate": None,
        "base_rate": None,
        "setups_resolved": 3,
    },
}
# The same scan as the app's destinations, plus trading and order words.
ADVICE_PATTERNS = (
    r"\bbuy",
    r"\bsell",
    r"recommend",
    r"should invest",
    r"should hold",
    r"outperform",
    r"\bconsider\b",
    r"\ballocate\b",
    r"\bbest\b",
    r"\btrad(e|es|ing)\b",
    r"\border(s)?\b",
    r"\bposition(s)?\b",
    r"\btarget price\b",
)


def _script():
    spec = importlib.util.spec_from_file_location(
        "stock_lab_briefing", ROOT / "scripts" / "stock_lab_briefing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


THEME = _script().briefing_theme()


@pytest.fixture
def states(tmp_path):
    """One briefing per state: fresh, stale, refused and missing synthetic; empty real."""

    lab = Lab(tmp_path)
    lab.forecast("DEMO_SWING", THURSDAY)  # becomes stale
    lab.forecast("DEMO_CALM", FRIDAY, p=(0.2, 0.5, 0.3), signals=SIGNALS)
    lab.forecast(
        "DEMO_TREND",
        FRIDAY,
        refusal={"reason": "insufficient_history", "detail": "Too little <history>."},
        created_at=ny(FRIDAY, "21:06"),
    )
    synthetic = build(lab, MONDAY, tickers=["DEMO_CALM", "DEMO_SWING", "DEMO_TREND", "DEMO_NONE"])
    real = build(lab, MONDAY, data_class="real", tickers=["AMD", "MU"])
    return {"synthetic": synthetic, "real": real}


def _visible(page: str) -> str:
    body = re.sub(r"<style>.*?</style>", " ", page, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", " ", body)


def test_synthetic_briefings_say_so_first(states):
    text = render_text(states["synthetic"])
    page = render_html(states["synthetic"], THEME)
    assert text.splitlines()[0].startswith("*** SYNTHETIC DEMO")
    assert '<p class="banner">Synthetic demo' in page
    assert "SYNTHETIC" not in render_text(states["real"])
    assert "banner" not in render_html(states["real"], THEME).split("</style>")[1]


def test_a_briefing_without_records_shows_no_probability_at_all(states):
    text = render_text(states["real"])
    page = _visible(render_html(states["real"], THEME))
    for rendering in (text, page):
        assert not re.search(r"\d\s*%", rendering)
        assert rendering.count("Unavailable") >= 2
        assert "decision A4" in rendering


def test_stale_refused_and_missing_stocks_carry_words_not_numbers(states):
    text = render_text(states["synthetic"])
    blocks = dict(
        re.findall(r"== (DEMO_\w+) · \w+ ==\n(.*?)(?=\n== |\nRELIABILITY)", text, re.DOTALL)
    )
    assert "Stale" in blocks["DEMO_SWING"] and "Next session 2026" not in blocks["DEMO_SWING"]
    assert "Too little <history>." in blocks["DEMO_TREND"]
    assert "No forecast is recorded for DEMO_NONE" in blocks["DEMO_NONE"]
    for symbol in ("DEMO_SWING", "DEMO_TREND", "DEMO_NONE"):
        assert "Increase " not in blocks[symbol].split("Forecast:")[0].split("Close:")[0]
    assert "Next session 2026-10-19" in blocks["DEMO_CALM"]
    assert "Increase   30%" in blocks["DEMO_CALM"]


def test_the_weekend_is_named_in_both_renderings(states):
    for rendering in (render_text(states["synthetic"]), render_html(states["synthetic"], THEME)):
        flat = " ".join(rendering.split())
        assert "Friday 16 October 2026" in flat and "no information" in flat


def test_the_page_is_self_contained_and_links_nowhere(states):
    for briefing in states.values():
        page = render_html(briefing, THEME).lower()
        for forbidden in ("<script", "<link", "<img", "<iframe", "src=", "href=", "url(",
                          "http://", "https://", "localhost", "127.0.0.1"):  # fmt: skip
            assert forbidden not in page, forbidden


def test_stored_text_is_escaped_in_the_page(states):
    page = render_html(states["synthetic"], THEME)
    assert "Too little &lt;history&gt;." in page and "<history>" not in page


def test_the_page_follows_the_lab_palette_in_light_and_dark(states):
    page = render_html(states["synthetic"], THEME)
    assert "@media (prefers-color-scheme: dark)" in page
    assert f"--canvas: {theme.LAB_COLOURS['canvas']};" in page
    assert f"--canvas: {theme.LAB_DARK_COLOURS['canvas']};" in page
    assert f"--dir-increase: {theme.DIRECTION_DARK_COLOURS['increase']};" in page


@pytest.mark.parametrize("scheme", ("light", "dark"))
def test_every_colour_pairing_the_page_uses_is_legible(scheme):
    colours = THEME[scheme]["colours"]
    surfaces = (colours["canvas"], colours["surface"], colours["surface_muted"])
    for key in ("text", "text_muted", "text_subtle", "accent"):
        for surface in surfaces:
            assert theme.contrast_ratio(colours[key], surface) >= 4.5, (key, surface)
    for name in ("VALID", "WARNING", "NOTICE", "UNAVAILABLE"):
        tone = THEME[scheme]["status"][name]
        assert theme.contrast_ratio(tone["fg"], tone["bg"]) >= 4.5, name
    for key, colour in THEME[scheme]["direction"].items():
        for surface in surfaces:
            assert theme.contrast_ratio(colour, surface) >= 3.0, (key, surface)


def test_every_category_is_written_in_words_beside_its_marker(states):
    page = render_html(states["synthetic"], THEME)
    for word, marker in (("Increase", "▲"), ("Stable", "●"), ("Decrease", "▼")):
        assert f">{marker}</span> {word}</span>" in page


@pytest.mark.parametrize("kind", ("synthetic", "real"))
def test_no_state_carries_advice_or_trading_wording(states, kind):
    for rendering in (render_text(states[kind]), _visible(render_html(states[kind], THEME))):
        lowered = rendering.lower()
        for pattern in ADVICE_PATTERNS:
            assert not re.search(pattern, lowered), pattern


def test_an_all_stale_briefing_says_so_in_its_header(tmp_path):
    lab = Lab(tmp_path)
    lab.forecast("DEMO_CALM", THURSDAY)
    lab.forecast("DEMO_SWING", THURSDAY, created_at=ny(THURSDAY, "21:06"))
    briefing = build(lab, MONDAY, tickers=["DEMO_CALM", "DEMO_SWING"])
    assert briefing.data_as_of is None
    for rendering in (render_text(briefing), _visible(render_html(briefing, THEME))):
        flat = " ".join(rendering.split())
        assert "no stock has a forecast as of that close" in flat
        assert (
            "2 of 2 stocks are stale" in flat and "No stock has a forecast for 2026-10-19" in flat
        )
        assert not re.search(r"\d\s*%", flat.split("STOCKS")[0].split("Stocks")[0])


def test_the_page_scrolls_wide_tables_and_scopes_every_header(states):
    page = render_html(states["synthetic"], THEME)
    tables = page.count("<table>")
    assert tables >= 2 and page.count('<div class="scroll"><table><caption>') == tables
    assert '<th scope="col"' in page and '<th scope="row">' in page and "<th>" not in page
    assert 'class="bar" aria-hidden="true"' in page and 'role="img"' not in page
