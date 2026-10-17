"""
The Stock Lab stays independent: it imports only the standard library, numpy,
pandas and itself (its Tiingo adapter also the audited key handling and
transport), and nothing in the main project imports it except the Lab's
page and its owner-run script.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stock_agent.ui import theme

SOURCE = Path(__file__).resolve().parents[1] / "src"
PROJECT = SOURCE.parent
PACKAGE = SOURCE / "stock_agent" / "stock_lab"
MODULES = sorted(PACKAGE.glob("*.py"))
ALLOWED_THIRD_PARTY = {"numpy", "pandas"}
TIINGO_EXTRA = {"stock_agent.data.vendors.tiingo"}
# The only project code that may import the Stock Lab.
CONSUMERS = ("stock_agent/app/", "stock_agent/stock_lab/")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            # `from package import module` names the module too.
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_stock_lab_modules_import_only_the_standard_library_numpy_pandas_and_itself(path):
    extra = TIINGO_EXTRA if path.stem == "tiingo" else set()
    for name in _imports(path):
        top = name.split(".")[0]
        assert (
            top in sys.stdlib_module_names
            or top in ALLOWED_THIRD_PARTY
            or name.startswith("stock_agent.stock_lab")
            or any(name == module or name.startswith(f"{module}.") for module in extra)
        ), (path.name, name)


def test_only_the_tiingo_adapter_may_reach_the_network():
    network = {"urllib", "http", "socket", "requests", "httpx", "urllib3", "aiohttp"}
    for path in MODULES:
        if path.stem == "tiingo":
            continue
        assert not {n.split(".")[0] for n in _imports(path)} & network, path.name


def test_nothing_else_in_the_project_imports_the_stock_lab():
    for path in sorted((SOURCE / "stock_agent").rglob("*.py")):
        relative = path.relative_to(SOURCE).as_posix()
        if relative.startswith(CONSUMERS):
            continue
        assert not [n for n in _imports(path) if n.startswith("stock_agent.stock_lab")], relative


def test_the_app_never_imports_the_tiingo_adapter():
    for path in sorted((SOURCE / "stock_agent" / "app").rglob("*.py")):
        assert "stock_agent.stock_lab.tiingo" not in _imports(path), path.name


def test_only_the_fetch_script_uses_the_adapter():
    users = [
        script.name
        for script in sorted((PROJECT / "scripts").glob("*.py"))
        if "stock_agent.stock_lab.tiingo" in _imports(script)
    ]
    assert users == ["stock_lab_fetch.py"]


def test_importing_the_page_side_loads_no_network_or_research_module():
    statement = (
        "import json, sys\n"
        "import stock_agent.stock_lab.view, stock_agent.stock_lab.store\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    for forbidden in ("urllib.request", "http.client", "sklearn", "stock_agent.data"):
        assert not [m for m in loaded if m == forbidden or m.startswith(forbidden + ".")]
    project = {m for m in loaded if m.startswith("stock_agent")}
    assert project <= {"stock_agent", "stock_agent.stock_lab"} | {
        m for m in project if m.startswith("stock_agent.stock_lab.")
    }


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_direction_colours_are_legible_and_never_carry_a_category_alone(scheme):
    colours = theme.DIRECTION_PALETTES[scheme]
    assert theme.DIRECTION_PALETTES["light"] is theme.DIRECTION_COLOURS
    assert set(colours) == set(theme.DIRECTION_MARKERS) == set(theme.DIRECTION_PATTERNS)
    assert len(set(colours.values())) == 3
    for colour in colours.values():
        for surface in ("surface", "canvas"):
            ratio = theme.contrast_ratio(colour, theme.LAB_PALETTES[scheme][surface])
            assert ratio >= 3.0, (scheme, colour, surface)
    assert len(set(theme.DIRECTION_MARKERS.values())) == 3
    assert len(set(theme.DIRECTION_PATTERNS.values())) == 3
