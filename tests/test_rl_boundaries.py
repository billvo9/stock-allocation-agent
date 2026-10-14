"""
Import boundaries of the `rl` package.

- It never imports the economically uncorrected `stock_agent.environment`
  simulator (standing owner rule: no PPO on an uncorrected simulator).
- It reaches no network or stored data.
- Only the scenario generator touches the latent (true) returns; the
  environment, observations and policies see observed data only.
- PR 1 adds no dependency: no gymnasium, torch, stable-baselines3 or SciPy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "rl"
MODULES = sorted(path for path in PACKAGE.glob("*.py"))

ALLOWED_PROJECT_IMPORTS = ("stock_agent.rl", "stock_agent.model_diagnostics.seeds")
FORBIDDEN_TOP_LEVEL = {
    "requests",
    "urllib",
    "http",
    "socket",
    "yfinance",
    "edgar",
    "duckdb",
    "gymnasium",
    "torch",
    "stable_baselines3",
    "scipy",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_the_package_has_the_expected_modules():
    names = {path.stem for path in MODULES}
    assert {"env", "observation", "scenarios", "actions", "transition", "policies"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_project_imports_stay_inside_rl_and_the_seed_utility(path):
    project = {name for name in _imports(path) if name.startswith("stock_agent")}
    assert all(name.startswith(ALLOWED_PROJECT_IMPORTS) for name in project), sorted(project)
    assert not any(name.startswith("stock_agent.environment") for name in project)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_no_network_storage_or_new_dependency_imports(path):
    top_level = {name.split(".")[0] for name in _imports(path)}
    assert not top_level & FORBIDDEN_TOP_LEVEL, sorted(top_level & FORBIDDEN_TOP_LEVEL)


@pytest.mark.parametrize(
    "path", [p for p in MODULES if p.stem != "scenarios"], ids=lambda p: p.stem
)
def test_only_the_generator_touches_latent_returns(path):
    assert "latent_returns" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_no_module_reads_stored_data_or_reports(path):
    text = path.read_text(encoding="utf-8")
    assert '"data/' not in text and "'data/" not in text and "reports/" not in text
