"""
The neutral `ui` package never depends on the research dashboard (or on any
data, model or publication code), so the product app can share the design
system without loading the dashboard.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "stock_agent" / "ui"
MODULES = sorted(path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__")


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_the_package_holds_the_moved_design_system():
    assert MODULES == ["components", "style", "theme"]
    dashboard = SOURCE / "stock_agent" / "dashboard"
    for name in MODULES:
        assert not (dashboard / f"{name}.py").exists(), name


@pytest.mark.parametrize("name", ["__init__", *MODULES])
def test_ui_modules_import_only_ui_from_this_project(name):
    imported = _direct_imports(PACKAGE / f"{name}.py")
    project = {module for module in imported if module.startswith("stock_agent")}
    assert all(module.startswith("stock_agent.ui") for module in project), sorted(project)
    assert not any(module.startswith("stock_agent.dashboard") for module in imported)


def test_importing_ui_loads_no_dashboard_or_other_project_module():
    script = (
        "import json, sys\n"
        "import stock_agent.ui.components, stock_agent.ui.style, stock_agent.ui.theme\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    project = {name for name in loaded if name.startswith("stock_agent")}
    allowed = {"stock_agent", "stock_agent.ui", *(f"stock_agent.ui.{m}" for m in MODULES)}
    assert project <= allowed, sorted(project - allowed)
