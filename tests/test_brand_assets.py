"""
The Portfolio Lab's brand assets (assets/brand/): well-formed SVGs on a
1024-pixel canvas, full-bleed app icons (macOS masks them itself), one
small artwork shared by the small icon and the rounded mark, colours taken from
the Lab's dark scheme, and the browser-tab icon the app passes to Streamlit.
"""

from __future__ import annotations

import ast
import re
import struct
import zlib
from pathlib import Path
from xml.etree import ElementTree

import pytest

from stock_agent.ui.theme import LAB_COLOURS, LAB_DARK_COLOURS, LAB_DARK_STATUS

REPO = Path(__file__).resolve().parents[1]
BRAND = REPO / "assets" / "brand"
SVG = "{http://www.w3.org/2000/svg}"
ICON, SMALL, MARK = (
    "portfolio-lab-icon.svg",
    "portfolio-lab-icon-small.svg",
    "portfolio-lab-mark.svg",
)
APP = REPO / "src" / "stock_agent" / "app"
FAVICON = APP / "static" / "portfolio-lab-icon.png"


def root(name: str) -> ElementTree.Element:
    return ElementTree.parse(BRAND / name).getroot()


def drawing(element: ElementTree.Element) -> list[tuple[str, dict[str, str]]]:
    """The drawn elements in order (groups opened; title and defs left out)."""

    items = []
    for child in element:
        if child.tag in {f"{SVG}title", f"{SVG}defs"}:
            continue
        if child.tag == f"{SVG}g":
            items.extend(drawing(child))
        else:
            items.append((child.tag, dict(child.attrib)))
    return items


@pytest.mark.parametrize("name", [ICON, SMALL, MARK])
def test_each_svg_is_a_titled_1024_canvas(name):
    svg = root(name)
    assert svg.tag == f"{SVG}svg"
    assert svg.get("viewBox") == "0 0 1024 1024"
    assert (svg.get("width"), svg.get("height")) == ("1024", "1024")
    assert svg.find(f"{SVG}title").text == "Portfolio Lab"


@pytest.mark.parametrize("name", [ICON, SMALL])
def test_the_app_icons_are_full_bleed(name):
    svg = root(name)
    first = drawing(svg)[0]
    assert first == (f"{SVG}rect", {"width": "1024", "height": "1024", "fill": "url(#surface)"})
    assert svg.find(f".//{SVG}clipPath") is None


def test_the_mark_is_the_small_artwork_on_a_rounded_square():
    mark, small = root(MARK), root(SMALL)
    group = mark.find(f"{SVG}g")
    assert group.get("clip-path") == "url(#tile)"
    assert mark.find(f".//{SVG}clipPath[@id='tile']") is not None
    assert drawing(group) == drawing(small)


def test_the_icons_share_one_set_of_gradients():
    def gradients(name: str) -> list[tuple[str, dict[str, str], list[dict[str, str]]]]:
        found = root(name).iter()
        return [
            (item.tag, dict(item.attrib), [dict(stop.attrib) for stop in item])
            for item in found
            if item.tag.endswith("Gradient")
        ]

    assert gradients(ICON) == gradients(SMALL) == gradients(MARK)


def test_every_colour_is_a_theme_token_or_a_documented_brand_colour():
    allowed = {
        LAB_DARK_COLOURS["accent"],  # terracotta bands and glow
        LAB_DARK_COLOURS["text"],  # ivory grid and "now" line
        LAB_DARK_STATUS["WARNING"]["fg"],  # amber inner band
        LAB_COLOURS["surface"],  # the last price
        "#352D26",  # surface gradient, top
        "#13110F",  # surface gradient, bottom
        "#F6F1E9",  # price line
        "#FFF6EA",  # median
    }
    for name in (ICON, SMALL, MARK):
        text = (BRAND / name).read_text(encoding="utf-8")
        used = {colour.upper() for colour in re.findall(r"#[0-9A-Fa-f]{6}\b", text)}
        assert used <= allowed, (name, used - allowed)
    readme = (BRAND / "README.md").read_text(encoding="utf-8")
    assert all(f"`{colour}`" in readme for colour in allowed)


@pytest.mark.parametrize("name", [ICON, SMALL])
def test_the_app_icon_surface_is_opaque(name):
    surface = root(name).find(f".//{SVG}linearGradient[@id='surface']")
    assert [stop.get("stop-opacity") for stop in surface] == [None, None]


def test_the_browser_tab_icon_is_a_64_pixel_rgba_png_with_rounded_corners():
    data = FAVICON.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour_type = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, colour_type) == (64, 64, 8, 6)  # 8-bit RGBA
    # The first pixel of the first row survives every PNG filter unchanged.
    chunks, position = [], 8
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        if kind == b"IDAT":
            chunks.append(data[position + 8 : position + 8 + length])
        position += 12 + length
    first_row = zlib.decompress(b"".join(chunks))[: 1 + 64 * 4]
    assert first_row[1 + 3] == 0  # the top-left corner is transparent


def test_the_lab_passes_its_icon_to_the_browser_tab():
    tree = ast.parse((APP / "run.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "set_page_config"
    ]
    assert len(calls) == 1
    icon = {keyword.arg: keyword.value for keyword in calls[0].keywords}["page_icon"]
    assert isinstance(icon, ast.Name) and icon.id == "ICON"
    assigned = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and [target.id for target in node.targets] == ["ICON"]
    ]
    assert (
        ast.unparse(assigned[0]) == "Path(__file__).with_name('static') / 'portfolio-lab-icon.png'"
    )
    assert FAVICON == Path(APP / "run.py").with_name("static") / "portfolio-lab-icon.png"
    assert FAVICON.is_file()
