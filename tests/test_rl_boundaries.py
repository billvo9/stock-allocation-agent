"""
Import boundaries of the `rl` package.

- It never imports the economically uncorrected `stock_agent.environment`
  simulator (standing owner rule: no PPO on an uncorrected simulator). Its
  project dependencies outside `rl` are the approved T5a transition and
  schedule, `stock_agent.portfolio.transition` and
  `stock_agent.portfolio.schedule` (D1-D8, approved 2026-10-15).
- It reaches no network or stored data.
- Only the scenario generator touches the latent (true) returns; the
  environment, observations and policies see observed data only.
- The optional learning libraries stay in two modules: gymnasium in
  `gym_env` and `ppo`, torch and stable-baselines3 in `ppo` only. Every
  other module, and importing the package, works without them. SciPy is
  not used at all.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "rl"
MODULES = sorted(path for path in PACKAGE.glob("*.py"))

ALLOWED_PROJECT_IMPORTS = (
    "stock_agent.rl",
    "stock_agent.model_diagnostics.seeds",
    "stock_agent.portfolio.transition",
    "stock_agent.portfolio.schedule",
)
FORBIDDEN_TOP_LEVEL = {
    "requests",
    "urllib",
    "http",
    "socket",
    "yfinance",
    "edgar",
    "duckdb",
    "scipy",
}
LEARNING_LIBRARIES = {
    "gym_env": {"gymnasium"},
    "ppo": {"gymnasium", "torch", "stable_baselines3"},
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
def test_no_network_storage_or_scipy_imports(path):
    top_level = {name.split(".")[0] for name in _imports(path)}
    assert not top_level & FORBIDDEN_TOP_LEVEL, sorted(top_level & FORBIDDEN_TOP_LEVEL)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_learning_libraries_stay_in_their_two_modules(path):
    top_level = {name.split(".")[0] for name in _imports(path)}
    learning = top_level & {"gymnasium", "torch", "stable_baselines3"}
    assert learning <= LEARNING_LIBRARIES.get(path.stem, set()), sorted(learning)


def test_importing_the_package_and_its_core_loads_no_learning_library():
    script = (
        "import json, sys\n"
        "import stock_agent.rl.env, stock_agent.rl.harness, stock_agent.rl.config\n"
        "import stock_agent.rl.learned, stock_agent.rl.health, stock_agent.rl.safety\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('gymnasium', 'torch', 'stable_baselines3'))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PACKAGE.parents[1])},
        timeout=120,
        check=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    "path", [p for p in MODULES if p.stem != "scenarios"], ids=lambda p: p.stem
)
def test_only_the_generator_touches_latent_returns(path):
    assert "latent_returns" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_no_module_reads_stored_data_or_reports(path):
    text = path.read_text(encoding="utf-8")
    assert '"data/' not in text and "'data/" not in text and "reports/" not in text
