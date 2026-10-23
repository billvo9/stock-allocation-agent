"""
The Lab app's page, stylesheet and script (docs/product/lab_app_design.md):
nothing fetched from another host, nothing the strict content security policy
would block, no advice words, and every colour pair of both themes readable.
"""

from __future__ import annotations

import re

import pytest
from test_app_destinations import scan_for_advice

from stock_agent.app.web import server

STATIC = server.STATIC
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
SCRIPT = (STATIC / "app.js").read_text(encoding="utf-8")
STYLE = (STATIC / "app.css").read_text(encoding="utf-8")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
CODE = re.sub(r"^\s*//.*$", "", SCRIPT, flags=re.MULTILINE)  # the script without its comments
STRINGS = " ".join(a or b for a, b in re.findall(r'"([^"\n]*)"|`([^`\n]*)`', CODE))


def test_the_server_sends_exactly_these_files():
    files = server.static_files()
    assert set(files) == {
        "/",
        "/static/app.css",
        "/static/app.js",
        "/static/mark.svg",
        "/static/source-sans-3.woff2",
    }
    assert all(body for body, _ in files.values())
    assert files["/static/source-sans-3.woff2"][0][:4] == b"wOF2"
    assert files["/static/mark.svg"][0].lstrip().startswith(b"<?xml")
    assert {route: kind for route, (_, kind) in files.items()} == {
        "/": "text/html; charset=utf-8",
        "/static/app.css": "text/css; charset=utf-8",
        "/static/app.js": "text/javascript; charset=utf-8",
        "/static/mark.svg": "image/svg+xml",
        "/static/source-sans-3.woff2": "font/woff2",
    }


def test_nothing_comes_from_another_host():
    for name, text in (("index.html", PAGE), ("app.js", SCRIPT), ("app.css", STYLE)):
        urls = set(re.findall(r"https?://[^\s\"')`]+", text)) - {SVG_NAMESPACE}
        assert not urls, (name, urls)
    assert "@import" not in STYLE
    assert re.findall(r"url\(([^)]*)\)", STYLE) == ['"/static/source-sans-3.woff2"']


def test_the_page_keeps_to_its_content_security_policy():
    # No inline script or style, so "script-src 'self'; style-src 'self'" holds.
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", PAGE, flags=re.DOTALL)
    assert len(scripts) == 1 and PAGE.count("<script") == 1
    ((attributes, body),) = scripts
    assert 'src="/static/app.js"' in attributes and not body.strip()
    assert "<style" not in PAGE and not re.search(r"\sstyle=", PAGE)
    assert not re.search(r"\son[a-z]+=", PAGE)


def test_the_script_never_writes_markup_or_inline_styles():
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "cssText",
        '"style"',
        ".style.",
        "setTimeout(",
        "setInterval(",
    ):
        assert banned not in CODE, banned


def test_no_advice_words():
    scan_for_advice(re.sub(r"<[^>]+>", " ", PAGE))
    scan_for_advice(STRINGS)
    assert "Nothing can be bought or sold here." in STRINGS


# --- colour: every pair of both themes ---


def _tokens(selector: str) -> dict[str, str]:
    block = re.search(re.escape(selector) + r"\s*\{(.*?)\}", STYLE, flags=re.DOTALL).group(1)
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9A-Fa-f]{6})", block))


DARK = _tokens(":root")
LIGHT = {**DARK, **_tokens(':root[data-theme="light"]')}


def _luminance(colour: str) -> float:
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


SURFACES = ("bg", "surface", "raised")
# Text on the screen's own surfaces: 4.5:1 (WCAG AA for normal text).
TEXT_PAIRS = [
    *[(fg, bg) for fg in ("text", "muted", "subtle", "fc", "up", "down") for bg in SURFACES],
    ("text", "ok-bg"),
    ("text", "warn-bg"),
    ("text", "bad-bg"),
    ("muted", "warn-bg"),
    ("muted", "bad-bg"),
    ("demo", "demo-bg"),
]
# Marks that carry meaning beside their words: 3:1 (WCAG non-text contrast).
MARK_PAIRS = [
    *[(fg, bg) for fg in ("up", "down", "line-ink") for bg in SURFACES],
    ("ok", "surface"),
    ("warn", "warn-bg"),
    ("bad", "bad-bg"),
]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_text_pair_is_readable(theme):
    tokens = DARK if theme == "dark" else LIGHT
    weak = [
        (fg, bg, round(contrast(tokens[fg], tokens[bg]), 2))
        for fg, bg in TEXT_PAIRS
        if contrast(tokens[fg], tokens[bg]) < 4.5
    ]
    assert not weak


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_mark_is_visible(theme):
    tokens = DARK if theme == "dark" else LIGHT
    weak = [
        (fg, bg, round(contrast(tokens[fg], tokens[bg]), 2))
        for fg, bg in MARK_PAIRS
        if contrast(tokens[fg], tokens[bg]) < 3
    ]
    assert not weak


def test_outlooks_are_never_green_or_red():
    for tokens in (DARK, LIGHT):
        assert tokens["fc"] not in (tokens["up"], tokens["down"])
    rules = re.findall(r"([^{}]*\.fc[^{}]*)\{([^}]*)\}", STYLE)
    assert rules and all("--up" not in body and "--down" not in body for _, body in rules)
