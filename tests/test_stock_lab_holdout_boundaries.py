"""
The sealed holdout path is isolated (owner decision of 2026-10-26: one run,
then the period is spent; research code keeps refusing it for training,
tuning and selection):

- no module outside `stock_lab/holdout/` imports it or names its file,
  directory, environment variable or header id, and only the two owner-run
  holdout scripts import it, each limited to named imports;
- research, evaluation, training, tuning and backtest entry points and
  scripts load nothing from it;
- inside the package, only the store writes or opens the file, and the run
  itself (`evaluate`) never imports the store.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stock_agent.stock_lab.holdout import store

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "stock_agent"
PROJECT = SOURCE.parent
HOLDOUT = "stock_agent.stock_lab.holdout"
SIDE = "stock_lab/holdout/"
SCRIPT_CONSUMERS = {
    "scripts/stock_lab_holdout_fetch.py": {
        f"{HOLDOUT}.protocol": {"PROTOCOL", "HoldoutRefused", "Protocol"},
        f"{HOLDOUT}.store": {"HoldoutStore"},
    },
    "scripts/stock_lab_holdout_evaluate.py": {
        f"{HOLDOUT}": {"evaluate"},
        f"{HOLDOUT}.protocol": {"PROTOCOL", "HoldoutRefused", "Protocol"},
        f"{HOLDOUT}.report": {"lines"},
        f"{HOLDOUT}.store": {"HoldoutStore"},
    },
}
FORBIDDEN_TEXT = (
    "stock_lab.holdout",
    store.DB_NAME,
    store.DIRECTORY,
    store.HOME_ENV,
    str(store.APPLICATION_ID),
    "SLHO",
)
RESEARCH_ENTRY_POINTS = (
    "stock_agent.stock_lab.evaluation",
    "stock_agent.stock_lab.estimators",
    "stock_agent.stock_lab.logistic",
    "stock_agent.stock_lab.predict",
    "stock_agent.stock_lab.registry",
    "stock_agent.stock_lab.frozen_forecast",
    "stock_agent.stock_lab.evening",
    "stock_agent.stock_lab.replay",
    "stock_agent.stock_lab.scoring",
    "stock_agent.stock_lab.job",
    "stock_agent.stock_lab.size_simulation",
    "stock_agent.stock_lab.decision_tests",
    "stock_agent.stock_lab.planned_simulation",
    "stock_agent.stock_lab.comparison_coverage",
    "stock_agent.stock_lab.features",
    "stock_agent.stock_lab.store",
    "stock_agent.stock_lab.briefing.report",
    "stock_agent.model_validation.folds",
    "stock_agent.model_diagnostics.runner",
    "stock_agent.forecasting.experiment",
    "stock_agent.rl.ppo",
)
RESEARCH_SCRIPTS = (
    "scripts/stock_lab_evaluate.py",
    "scripts/stock_lab_a7_simulation.py",
    "scripts/stock_lab_comparison_coverage.py",
    "scripts/stock_lab_fetch.py",
    "scripts/stock_lab_fetch_recent.py",
    "scripts/stock_lab_daily.py",
    "scripts/stock_lab_schedule.py",
    "scripts/stock_lab_briefing.py",
)


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def _module(path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = list(path.relative_to(SOURCE).parent.parts) if path.is_relative_to(SOURCE) else []
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join([*base, *([node.module] if node.module else [])])


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            module = _module(path, node)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _reaches(name: str) -> bool:
    return name == HOLDOUT or name.startswith(HOLDOUT + ".")


def _code(path: Path) -> str:
    """The source without its module docstring (a docstring may name the store)."""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if ast.get_docstring(tree, clean=False) is not None:
        text = text.replace(ast.get_source_segment(text, tree.body[0]), "", 1)
    return text


def violations(path: Path, allowed: dict[str, set[str]]) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if _reaches(alias.name)]
        elif isinstance(node, ast.ImportFrom):
            module = _module(path, node)
            names = [f"{module}.{alias.name}" for alias in node.names]
            if not any(_reaches(name) for name in (module, *names)):
                continue
            permitted = allowed.get(module, set())
            found += [
                f"{module}.{alias.name}" for alias in node.names if alias.name not in permitted
            ]
    return found


OUTSIDE = sorted(path for path in PACKAGE.rglob("*.py") if not _relative(path).startswith(SIDE))


@pytest.mark.parametrize("path", OUTSIDE, ids=_relative)
def test_no_other_module_imports_or_names_the_holdout(path):
    assert not sorted(name for name in _imports(path) if _reaches(name)), path
    code = _code(path)
    assert not [needle for needle in FORBIDDEN_TEXT if needle in code], path


def test_only_the_two_holdout_scripts_import_it_with_named_imports():
    for script in sorted((PROJECT / "scripts").rglob("*.py")):
        relative = script.relative_to(PROJECT).as_posix()
        if relative in SCRIPT_CONSUMERS:
            assert violations(script, SCRIPT_CONSUMERS[relative]) == [], relative
            code = _code(script)
            assert "importlib" not in code and "__import__" not in code, relative
            assert not [needle for needle in FORBIDDEN_TEXT[1:] if needle in code], relative
            continue
        assert not sorted(name for name in _imports(script) if _reaches(name)), relative
        assert not [needle for needle in FORBIDDEN_TEXT if needle in _code(script)], relative
    for relative in RESEARCH_SCRIPTS:
        assert (PROJECT / relative).is_file() and relative not in SCRIPT_CONSUMERS


def test_the_package_holds_only_its_reviewed_modules_and_only_the_store_writes():
    directory = PACKAGE / "stock_lab" / "holdout"
    names = {path.name for path in directory.glob("*.py")}
    assert names == {"__init__.py", "protocol.py", "store.py", "evaluate.py", "report.py"}
    assert not [p for p in directory.iterdir() if p.is_dir() and p.name != "__pycache__"]
    writers = sorted(
        path.name
        for path in directory.glob("*.py")
        if any(
            call in _code(path)
            for call in ("INSERT INTO", "sqlite3", "CREATE TABLE", "application_id", "open(")
        )
    )
    assert writers == ["store.py"]
    for name in ("evaluate.py", "report.py", "protocol.py"):
        imported = _imports(directory / name)
        assert not [n for n in imported if n.startswith(f"{HOLDOUT}.store")], name
    assert not [n for n in _imports(directory / "report.py") if n.startswith("stock_agent")]


def _loaded(statement: str) -> set[str]:
    script = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_research_entry_points_and_scripts_load_nothing_from_the_holdout():
    statement = "\n".join(f"import {name}" for name in RESEARCH_ENTRY_POINTS)
    statement += "\nimport importlib.util\n" + "\n".join(
        f"spec = importlib.util.spec_from_file_location('script_{index}', {str(PROJECT / relative)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)"
        for index, relative in enumerate(RESEARCH_SCRIPTS)
    )
    loaded = _loaded(statement)
    assert set(RESEARCH_ENTRY_POINTS) <= loaded
    assert not sorted(name for name in loaded if _reaches(name))


def test_the_run_loads_neither_the_store_nor_the_network():
    loaded = _loaded(f"import {HOLDOUT}.evaluate, {HOLDOUT}.report")
    assert f"{HOLDOUT}.store" not in loaded
    for forbidden in ("urllib.request", "http.client", "stock_agent.data"):
        assert not [m for m in loaded if m == forbidden or m.startswith(forbidden + ".")]
