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
MODULES = sorted(PACKAGE.rglob("*.py"))  # the inference subpackage included
ALLOWED_THIRD_PARTY = {"numpy", "pandas"}
# The research-side modules and the one extra each may import. The Lab's page
# never imports any of them (tested below and in test_app_explore.py).
EXTRA = {
    "tiingo": {"stock_agent.data.vendors.tiingo"},
    "evaluation": {"stock_agent.model_validation.folds", "stock_agent.model_diagnostics.inference"},
    "size_simulation": {"stock_agent.model_diagnostics.inference"},
    "decision_tests": {"stock_agent.model_diagnostics.inference"},
    "logistic": {"sklearn"},
}
RESEARCH_SIDE = (
    "tiingo",
    "evaluation",
    "logistic",
    "predict",
    "registry",
    "frozen_forecast",
    "evening",
    "replay",
    "size_simulation",
    "decision_tests",
    "planned_simulation",
    "comparison_coverage",
)
# The only project code that may import the Stock Lab.
# The exploratory showcase (2026-10-26) reuses the Stock Lab's write-once store, symbol
# classification, audit and Tiingo adapter for its own separate private store.
CONSUMERS = ("stock_agent/app/", "stock_agent/stock_lab/", "stock_agent/showcase/")


def _key(path: Path) -> str:
    """The module's path inside the package without .py ("tiingo", "inference/store")."""

    return path.relative_to(PACKAGE).with_suffix("").as_posix()


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


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_stock_lab_modules_import_only_the_standard_library_numpy_pandas_and_itself(path):
    extra = EXTRA.get(_key(path), set())
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
        if _key(path) == "tiingo":
            continue
        assert not {n.split(".")[0] for n in _imports(path)} & network, path.name


def test_nothing_else_in_the_project_imports_the_stock_lab():
    for path in sorted((SOURCE / "stock_agent").rglob("*.py")):
        relative = path.relative_to(SOURCE).as_posix()
        if relative.startswith(CONSUMERS):
            continue
        assert not [n for n in _imports(path) if n.startswith("stock_agent.stock_lab")], relative


def test_the_app_never_imports_the_research_side():
    forbidden = {f"stock_agent.stock_lab.{name}" for name in RESEARCH_SIDE}
    for path in sorted((SOURCE / "stock_agent" / "app").rglob("*.py")):
        assert not _imports(path) & forbidden, path.name


def test_page_side_modules_never_import_the_research_side():
    forbidden = {f"stock_agent.stock_lab.{name}" for name in RESEARCH_SIDE}
    for path in MODULES:
        if _key(path) in RESEARCH_SIDE:
            continue
        assert not _imports(path) & forbidden, path.name


def test_only_the_fetch_scripts_use_the_adapter():
    # The research fetch (pre-2025 history into the research store), the
    # recent-window fetch (the inference window into the inference store) and the
    # showcase fetch (pre-2025 history into the separate showcase store).
    users = [
        script.name
        for script in sorted((PROJECT / "scripts").glob("*.py"))
        if "stock_agent.stock_lab.tiingo" in _imports(script)
    ]
    assert users == ["showcase_fetch.py", "stock_lab_fetch.py", "stock_lab_fetch_recent.py"]


def test_importing_the_page_side_loads_no_network_or_research_module():
    statement = (
        "import json, sys\n"
        "import stock_agent.stock_lab.view, stock_agent.stock_lab.store\n"
        "import stock_agent.stock_lab.reversal, stock_agent.stock_lab.traps\n"
        "import stock_agent.stock_lab.forecast, stock_agent.stock_lab.scores\n"
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
    for forbidden in (
        "urllib.request",
        "http.client",
        "sklearn",
        "scipy",
        "stock_agent.data",
        "stock_agent.model_validation",
        "stock_agent.model_diagnostics",
    ):
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
