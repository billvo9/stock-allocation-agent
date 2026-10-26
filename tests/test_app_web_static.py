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
    # The Health tab's badge: its mark in the page's background colour on its status colour.
    ("bg", "warn"),
    ("bg", "bad"),
]
# Marks that carry meaning beside their words: 3:1 (WCAG non-text contrast).
MARK_PAIRS = [
    *[(fg, bg) for fg in ("up", "down", "line-ink", "fc", "note") for bg in SURFACES],
    ("fc", "fc-flat"),  # the outlook bar's Up and Down segments beside its Flat one
    ("ok", "surface"),
    ("warn", "warn-bg"),
    ("bad", "bad-bg"),
    # Health: the ring on its tint, each check's mark on its card, the tab's badge.
    ("ok", "ok-bg"),
    ("warn", "surface"),
    ("bad", "surface"),
    *[(fg, "bg") for fg in ("ok", "warn", "bad")],
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
        assert {tokens["fc"], tokens["fc-flat"]}.isdisjoint({tokens["up"], tokens["down"]})
    rules = re.findall(r"([^{}]*\.fc[^{}]*)\{([^}]*)\}", STYLE)
    assert rules and all("--up" not in body and "--down" not in body for _, body in rules)
    # The detail's outlook bars, their legend, each day's row and the meter.
    outlook = re.findall(
        r"([^{}]*(?:\.seg|\.hatch|\.obar|\.olegend|\.five|\.meter)[^{}]*)\{([^}]*)\}", STYLE
    )
    assert len(outlook) >= 10
    assert all("--up" not in body and "--down" not in body for _, body in outlook)


def _blend(colour: str, behind: str, share: float) -> str:
    """`share` of `colour` over `behind` (what color-mix with transparent paints)."""

    mixed = [
        round(share * int(colour[i : i + 2], 16) + (1 - share) * int(behind[i : i + 2], 16))
        for i in (1, 3, 5)
    ]
    return "#" + "".join(f"{c:02X}" for c in mixed)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_move_reads_on_its_own_tinted_pill(theme):
    tokens = DARK if theme == "dark" else LIGHT
    scope = r":root\[data-theme=\"light\"\] " if theme == "light" else ""
    for direction in ("up", "down"):
        rule = re.search(
            rf"(?m)^{scope}\.chg\.{direction} \{{[^}}]*color-mix\(in srgb, var\(--{direction}\) "
            r"(\d+)%, transparent\)",
            STYLE,
        )
        share = int(rule.group(1)) / 100
        pill = _blend(tokens[direction], tokens["bg"], share)
        assert contrast(tokens[direction], pill) >= 4.5, (theme, direction, share)


def test_the_chart_and_the_bars_are_drawn_without_inline_style():
    # Shapes are SVG drawn to the measured width of their box, so the strict policy holds.
    assert "new ResizeObserver(" in CODE and "setAttribute(" in CODE
    assert "preserveAspectRatio" not in CODE


def test_the_detail_is_reached_only_through_the_checked_route():
    # The page asks for one stock by a ticker it read from the address, checked against the
    # server's own name rule, and never builds any other request from data.
    assert re.findall(r"fetch\(([^,]+),", CODE) == ["path"]
    assert re.findall(r"await ask\((.+?)\);", CODE) == [
        '"/api/home"',
        "`/api/stock/${encodeURIComponent(ticker)}`",
        "`/api/${name}`",
    ]
    # History and Health: `name` is one of two fixed screens, never anything read from data.
    assert 'const TABS = ["history", "health"];' in CODE
    assert re.findall(r"loadTab\((.+?)\)", CODE) == ["name", "state.tab", "state.tab"]
    assert CODE.count("if (TABS.includes(state.tab)) loadTab(state.tab);") == 2
    assert r"const TICKER = /^#stock\/([A-Z0-9][A-Z0-9_-]{0,19})$/;" in CODE
    assert server.NAME.pattern == "[A-Z0-9][A-Z0-9_-]{0,19}"


def test_the_launch_key_stays_in_this_tab_and_leaves_the_address_at_once():
    # The key arrives in the fragment, is kept in sessionStorage only (this tab, never
    # localStorage), is removed from the address before routing, and is sent in the
    # header the server checks.
    assert "sessionStorage.setItem(KEY_STORE" in CODE
    assert "localStorage.setItem(KEY_STORE" not in CODE
    take = CODE[CODE.index("function takeKey()") :]
    take = take[: take.index("\n  }\n")]
    assert "history.replaceState" in take
    start = CODE[CODE.index("function start()") :]
    assert start.index("takeKey();") < start.index("routed()")
    assert '"X-Lab-Token"' in CODE and server.KEY_HEADER == "X-Lab-Token".lower()


def test_the_watchlist_draws_every_line_on_one_shared_scale():
    # The page's drawing can't run here; these pin the wiring the shared scale needs.
    assert "const span = sharedSpan(data.stocks);" in CODE
    assert "stockRow(stock, span)" in CODE and "spark(p.line, p.direction, span)" in CODE
    body = CODE[CODE.index("function spark(") :]
    body = body[: body.index("\n  }\n")]
    assert "Math.max(span || 0, hi - lo)" in body and "Math.log(v / latest)" in body
    shared = CODE[CODE.index("function sharedSpan(") :]
    assert (
        "Math.log(Math.max(...values) / Math.min(...values))" in shared[: shared.index("\n  }\n")]
    )


def test_a_page_without_the_key_is_sent_back_to_the_app_with_no_retry_or_back():
    assert re.search(r"const LOCKED = \[[^\]]*, false, true\];", CODE)  # no Back, final
    assert "response.status === 401) state.problem = LOCKED" in CODE
    assert "response.status === 401) state.problems[name] = LOCKED" in CODE
    assert "state.detailProblem = [LOCKED[0], LOCKED[1], true, true]" in CODE
    change = CODE[CODE.index('window.addEventListener("hashchange"') :]
    change = change[: change.index("});")]
    assert change.index("takeKey();") < change.index("follow();")


def test_history_colours_only_the_move_that_happened():
    # What the model said is never green or red; what happened is marked by its direction.
    assert ".tl-item .res .mv.up { color: var(--up); }" in STYLE
    assert ".tl-item .res .mv.down { color: var(--down); }" in STYLE
    said = re.findall(r"([^{}]*\.said[^{}]*)\{([^}]*)\}", STYLE)
    assert said and all("--up" not in body and "--down" not in body for _, body in said)
    item = CODE[CODE.index("function historyItem(") :]
    item = item[: item.index("\n  }\n")]
    assert "`mv ${x.result.direction}`" in item and "x.said.up" in item


def test_the_health_tab_shows_the_badge_home_sends():
    # The server decides the badge (Home's status line, or a check Model Health flags; owner
    # decision of 2026-10-26), so the page draws only what Home sends: its level's colour,
    # its mark and its words, and nothing when there is none.
    assert 'id="health-badge"' in PAGE
    render = CODE[CODE.index("function render()") :]
    render = render[: render.index("\n  }\n")]
    assert "const shown = badgeNow();" in render
    assert "badge.hidden = !shown;" in render
    assert "`badge ${shown.level}`" in render  # its colour
    assert "badge.textContent = shown ? shown.mark" in render
    assert 'badge.setAttribute("aria-label", shown.words)' in render
    assert "status.level" not in render  # no rule of its own
    # Home's, or Health's when Health answered later (both send the same rule's badge), so the
    # tab and the Health screen show the same moment.
    newest = CODE[CODE.index("function badgeNow()") :]
    newest = newest[: newest.index("\n  }\n")]
    assert "Date.parse(health.generated_at) > Date.parse(home.generated_at)" in newest
    assert "return home ? home.badge : null;" in newest


def test_healths_flagged_line_carries_the_badges_mark():
    # The "!" on the tab points at this line: it carries the same mark, beside its words.
    screen = CODE[CODE.index("function healthScreen(") :]
    screen = screen[: screen.index("\n  }\n")]
    assert 'h("span", { class: "flag-mark", "aria-hidden": "true" }, "!")' in screen
    assert ".hero .flag-mark {" in STYLE
    assert re.search(r'data-tab="history".*data-tab="health".*data-tab="more"', PAGE, re.DOTALL)


def test_a_screen_that_answers_again_clears_its_problem():
    # An unchanged answer after a failure must still take the problem off the screen.
    assert "!state.problems[name] && USABLE[name](data) && same(state[name], data)) return;" in CODE
    assert (
        "!state.detailProblem && usableDetail(data, ticker) && same(state.detail, data)) return;"
        in (CODE)
    )


def test_lists_keep_their_role_in_safari():
    # Safari drops a list's role when its markers are hidden, so the page names it.
    for found in ('h("ul", { class: "list", role: "list" }', 'h("ul", { class: "tl-items", role: "list" }',
                  'h("ol", { class: "tl", role: "list"'):  # fmt: skip
        assert found in CODE, found
