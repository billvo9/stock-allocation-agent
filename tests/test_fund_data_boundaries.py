import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
FUNDS = SOURCE / "stock_agent" / "data" / "funds"
CALENDARS = SOURCE / "stock_agent" / "calendars"
MODULES = sorted(path for path in (*FUNDS.glob("*.py"), *CALENDARS.glob("*.py")))

# Everything the fund-data store and the calendar may load from this repository.
# audit.py imports features.training; nothing else is allowed.
ALLOWED_PROJECT_MODULES = {
    "stock_agent",
    "stock_agent.calendars",
    "stock_agent.calendars.nyse",
    "stock_agent.data",
    "stock_agent.data.funds",
    *(f"stock_agent.data.funds.{path.stem}" for path in FUNDS.glob("*.py")),
    "stock_agent.features",
    "stock_agent.features.training",
    "stock_agent.model_validation",
    "stock_agent.model_validation.audit",
    "stock_agent.model_validation.folds",
}
# Network clients and vendor SDKs. pandas itself loads urllib.parse (URL
# parsing, no I/O), so the runtime check names the request-making modules.
NETWORK_MODULES = ("requests", "urllib3", "yfinance", "edgar", "fredapi", "httpx", "aiohttp")
NETWORK_SUBMODULES = ("urllib.request", "http.client")
FORBIDDEN_IMPORT_ROOTS = {"requests", "urllib", "urllib3", "http", "socket", "yfinance"}
FORBIDDEN_IMPORT_ROOTS |= {"httpx", "aiohttp", "edgar", "fredapi"}


def _loaded_modules(statement: str) -> set[str]:
    script = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))"
    environment = {**os.environ, "PYTHONPATH": str(SOURCE)}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_the_store_loads_no_network_library_and_no_publication_module():
    loaded = _loaded_modules(
        "import stock_agent.data.funds.adapter, stock_agent.data.funds.ingest, "
        "stock_agent.data.funds.reader"
    )
    assert not [name for name in loaded if name.split(".")[0] in NETWORK_MODULES]
    assert not [name for name in loaded if name in NETWORK_SUBMODULES]
    assert not [name for name in loaded if name.startswith("stock_agent.publication")]
    project = {name for name in loaded if name.startswith("stock_agent")}
    assert project <= ALLOWED_PROJECT_MODULES, sorted(project - ALLOWED_PROJECT_MODULES)


def test_the_reader_never_loads_the_ingest_writer():
    loaded = _loaded_modules("import stock_agent.data.funds.reader")
    assert "stock_agent.data.funds.reader" in loaded
    assert "stock_agent.data.funds.ingest" not in loaded


@pytest.mark.parametrize("path", MODULES, ids=lambda path: f"{path.parent.name}.{path.stem}")
def test_no_module_imports_a_network_library_or_the_publication_writer(path):
    imported = _direct_imports(path)
    assert not {name for name in imported if name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS}
    assert not {name for name in imported if name.startswith("stock_agent.publication")}
    if path.stem == "reader":
        assert "stock_agent.data.funds.ingest" not in imported


def test_the_lockbox_is_imported_never_restated():
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        assert "2025-01-01" not in text and "2025, 1, 1" not in text, path
        tree = ast.parse(text)
        used = any(
            isinstance(node, ast.Name) and node.id == "MODEL_HOLDOUT_START"
            for node in ast.walk(tree)
        )
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "stock_agent.model_validation.folds"
            and "MODEL_HOLDOUT_START" in {alias.name for alias in node.names}
            for node in ast.walk(tree)
        )
        assert imported or not used, path
