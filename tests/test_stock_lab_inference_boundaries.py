"""
The Stock Lab's inference store is isolated from research code (decision A2:
protected-period observations serve "only as inputs to a previously frozen
model", never for retrospective selection, training or backtests; decision
A6: an isolated descriptive reader on the private page).

- No module outside `stock_lab/inference/` imports it or names its file,
  environment variable or header id, directly or through importlib, except
  the reviewed consumers below, each limited to named imports.
- Research, evaluation, training, tuning and backtest entry points load
  nothing from it.
- The research store refuses its file; the readers never load the writer;
  only the writer writes.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stock_agent.stock_lab.inference import schema

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "stock_agent"
PROJECT = SOURCE.parent
INFERENCE = "stock_agent.stock_lab.inference"
STORE_SIDE = "stock_lab/inference/"
# Modules (relative to src/stock_agent) that may import the inference package, and
# the only names each may import, with `from <module> import <name>`. The Lab page
# reads recent observed prices; nothing else in the project reads the store. The
# frozen-model reader's consumer (the forecast job) is added here, after review, in
# the change that activates real forecasts (decision A4).
MODULE_CONSUMERS = {
    "app/stock_lab.py": {
        f"{INFERENCE}.recent": {"RecentPrices", "read_recent"},
    },
}
# Owner-run scripts (relative to the repository root) and their named imports.
SCRIPT_CONSUMERS = {
    "scripts/stock_lab_fetch_recent.py": {
        f"{INFERENCE}.schema": {
            "INFERENCE_SESSIONS",
            "PROTECTED_START",
            "InferenceRefused",
            "Refusal",
            "close_available_at",
        },
        f"{INFERENCE}.store": {"InferenceStore"},
        f"{INFERENCE}.window": {"load_inference_window"},
    },
}
# Text that reaches the store without an import statement: the module path (for
# importlib), the file and directory names, the environment variable, the header id.
FORBIDDEN_TEXT = (
    "stock_lab.inference",
    schema.DB_NAME,
    schema.DIRECTORY,
    schema.HOME_ENV,
    str(schema.APPLICATION_ID),
    "SLIW",
)
# Research, evaluation, training, tuning and backtest entry points of the Stock Lab
# and of the main project: importing them must load nothing from the store.
RESEARCH_ENTRY_POINTS = (
    "stock_agent.stock_lab.evaluation",
    "stock_agent.stock_lab.estimators",
    "stock_agent.stock_lab.logistic",
    "stock_agent.stock_lab.predict",
    "stock_agent.stock_lab.registry",
    "stock_agent.stock_lab.replay",
    "stock_agent.stock_lab.scoring",
    "stock_agent.stock_lab.job",
    "stock_agent.stock_lab.size_simulation",
    "stock_agent.stock_lab.decision_tests",
    "stock_agent.stock_lab.planned_simulation",
    "stock_agent.stock_lab.features",
    "stock_agent.stock_lab.view",
    "stock_agent.stock_lab.forecast_view",
    "stock_agent.stock_lab.audit",
    "stock_agent.stock_lab.tiingo",
    "stock_agent.model_validation.folds",
    "stock_agent.model_diagnostics.runner",
    "stock_agent.forecasting.experiment",
    "stock_agent.rl.ppo",
)
# The scripts that research, evaluate, simulate, forecast or schedule.
RESEARCH_SCRIPTS = (
    "scripts/stock_lab_evaluate.py",
    "scripts/stock_lab_a7_simulation.py",
    "scripts/stock_lab_fetch.py",
    "scripts/stock_lab_daily.py",
    "scripts/stock_lab_schedule.py",
)


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def _module(path: Path, node: ast.ImportFrom) -> str:
    """The absolute module of an `ImportFrom`, relative imports resolved."""

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
    return name == INFERENCE or name.startswith(INFERENCE + ".")


def _code(path: Path) -> str:
    """The source without its module docstring (a docstring may name the store)."""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if ast.get_docstring(tree, clean=False) is not None:
        text = text.replace(ast.get_source_segment(text, tree.body[0]), "", 1)
    return text


def violations(path: Path, allowed: dict[str, set[str]]) -> list[str]:
    """Imports of the inference package beyond `allowed`, as `module.name` strings."""

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


OUTSIDE = sorted(
    path
    for path in PACKAGE.rglob("*.py")
    if not _relative(path).startswith(STORE_SIDE) and _relative(path) not in MODULE_CONSUMERS
)


@pytest.mark.parametrize("path", OUTSIDE, ids=_relative)
def test_no_other_module_imports_or_names_the_inference_store(path):
    assert not sorted(name for name in _imports(path) if _reaches(name)), path
    code = _code(path)
    assert not [needle for needle in FORBIDDEN_TEXT if needle in code], path


def test_the_lab_page_imports_only_the_descriptive_reader():
    for relative, allowed in MODULE_CONSUMERS.items():
        path = PACKAGE / relative
        assert violations(path, allowed) == [], relative
        code = _code(path)
        assert "importlib" not in code and "__import__" not in code, relative
        assert not [needle for needle in FORBIDDEN_TEXT[1:] if needle in code], relative


def test_only_the_inference_fetch_script_imports_the_store():
    for script in sorted((PROJECT / "scripts").rglob("*.py")):
        relative = script.relative_to(PROJECT).as_posix()
        if relative in SCRIPT_CONSUMERS:
            assert violations(script, SCRIPT_CONSUMERS[relative]) == [], relative
            assert "importlib" not in _code(script), relative
            continue
        assert not sorted(name for name in _imports(script) if _reaches(name)), relative
        assert not [needle for needle in FORBIDDEN_TEXT if needle in _code(script)], relative
    for relative in RESEARCH_SCRIPTS:
        assert (PROJECT / relative).is_file() and relative not in SCRIPT_CONSUMERS


def test_the_consumer_rule_catches_whole_module_imports_and_other_names(tmp_path):
    sample = tmp_path / "page.py"
    sample.write_text(
        "from stock_agent.stock_lab.inference.recent import read_recent, select_vintage\n"
        "from stock_agent.stock_lab.inference import window\n"
        "from stock_agent.stock_lab.inference.schema import bars_of\n"
        "import stock_agent.stock_lab.inference.store\n",
        encoding="utf-8",
    )
    allowed = MODULE_CONSUMERS["app/stock_lab.py"]
    assert violations(sample, allowed) == [
        "stock_agent.stock_lab.inference.recent.select_vintage",
        "stock_agent.stock_lab.inference.window",
        "stock_agent.stock_lab.inference.schema.bars_of",
        "stock_agent.stock_lab.inference.store",
    ]


def test_relative_imports_are_resolved():
    module = PACKAGE / "stock_lab" / "evaluation.py"
    node = ast.parse("from .inference.window import load_inference_window").body[0]
    assert _module(module, node) == "stock_agent.stock_lab.inference.window"
    node = ast.parse("from . import inference").body[0]
    assert _module(module, node) == "stock_agent.stock_lab"
    assert _reaches(f"{_module(module, node)}.{node.names[0].name}")


def test_the_readers_never_import_the_writer_and_only_the_writer_writes():
    directory = PACKAGE / "stock_lab" / "inference"
    for name in ("window.py", "recent.py", "schema.py"):
        imported = _imports(directory / name)
        assert not [n for n in imported if n.startswith(f"{INFERENCE}.store")], name
    recent = _imports(directory / "recent.py")
    assert not [n for n in recent if n.startswith(f"{INFERENCE}.window")]
    writers = sorted(
        path.name
        for path in directory.glob("*.py")
        if any(
            call in _code(path)
            for call in ("INSERT INTO", "executescript", "CREATE TABLE", "application_id =")
        )
    )
    assert writers == ["store.py"]
    readers = sorted(path.name for path in directory.glob("*.py") if "mode=ro" in _code(path))
    assert readers == ["schema.py"]  # every read goes through schema.read_only


def test_the_package_holds_only_its_reviewed_modules():
    # A new module here would be exempt from every check above: add it deliberately.
    names = {path.name for path in (PACKAGE / "stock_lab" / "inference").glob("*.py")}
    assert names == {"__init__.py", "schema.py", "store.py", "window.py", "recent.py"}
    assert not [
        p
        for p in (PACKAGE / "stock_lab" / "inference").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    ]


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


def test_research_entry_points_load_nothing_from_the_inference_store():
    loaded = _loaded("\n".join(f"import {name}" for name in RESEARCH_ENTRY_POINTS))
    assert set(RESEARCH_ENTRY_POINTS) <= loaded
    assert not sorted(name for name in loaded if _reaches(name))


def test_research_scripts_load_nothing_from_the_inference_store():
    # Loading each script (its main is guarded) must not reach the store, even
    # through another module that imports it.
    statement = "import importlib.util\n" + "\n".join(
        f"spec = importlib.util.spec_from_file_location('script_{index}', {str(PROJECT / relative)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)"
        for index, relative in enumerate(RESEARCH_SCRIPTS)
    )
    loaded = _loaded(statement)
    assert "stock_agent.stock_lab.store" in loaded  # the scripts were really loaded
    assert not sorted(name for name in loaded if _reaches(name))


def test_the_descriptive_reader_loads_neither_the_writer_nor_the_frozen_model_reader():
    loaded = _loaded(f"import {INFERENCE}.recent")
    assert f"{INFERENCE}.store" not in loaded and f"{INFERENCE}.window" not in loaded
    loaded = _loaded(f"import {INFERENCE}.window")
    assert f"{INFERENCE}.store" not in loaded
    for forbidden in ("urllib.request", "http.client", "sklearn", "stock_agent.data"):
        assert not [m for m in loaded if m == forbidden or m.startswith(forbidden + ".")]
