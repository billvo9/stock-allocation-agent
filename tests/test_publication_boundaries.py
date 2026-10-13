import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.model_validation import folds
from stock_agent.publication import envelope, synthetic
from stock_agent.publication.envelope import CodeState, PublicationRefused, RefusalReason
from stock_agent.publication.reader import read_publication
from stock_agent.publication.writer import write_publication

SOURCE = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SOURCE / "stock_agent" / "publication"
MODULES = ("envelope", "policy", "synthetic", "writer", "reader")

# Everything importing the publication package may load from this repository.
# audit.py imports features.training; nothing else is allowed.
ALLOWED_PROJECT_MODULES = {
    "stock_agent",
    "stock_agent.features",
    "stock_agent.features.training",
    "stock_agent.model_validation",
    "stock_agent.model_validation.audit",
    "stock_agent.model_validation.folds",
    "stock_agent.publication",
    *(f"stock_agent.publication.{name}" for name in MODULES),
}
ALLOWED_DIRECT_IMPORTS = {
    "__future__",
    "collections.abc",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "errno",
    "hashlib",
    "json",
    "math",
    "numpy",
    "os",
    "pandas",
    "pathlib",
    "re",
    "tempfile",
    "types",
    "typing",
    "stock_agent.model_validation.audit",
    "stock_agent.model_validation.folds",
    "stock_agent.publication",
    *(f"stock_agent.publication.{name}" for name in MODULES),
}
HEAVY = ("streamlit", "plotly", "sklearn", "duckdb", "yfinance", "requests", "edgar")


def _loaded_modules(statement):
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


def test_the_reader_never_loads_the_writer():
    loaded = _loaded_modules("import stock_agent.publication.reader")
    assert "stock_agent.publication.reader" in loaded
    assert "stock_agent.publication.writer" not in loaded


def test_the_package_loads_only_allowed_project_modules_and_no_heavy_dependencies():
    loaded = _loaded_modules(
        "import stock_agent.publication.reader, stock_agent.publication.writer"
    )
    project = {name for name in loaded if name.startswith("stock_agent")}
    assert project <= ALLOWED_PROJECT_MODULES, sorted(project - ALLOWED_PROJECT_MODULES)
    assert not [name for name in loaded if name.split(".")[0] in HEAVY]


@pytest.mark.parametrize("name", MODULES)
def test_direct_imports_are_on_the_allowlist(name):
    tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= ALLOWED_DIRECT_IMPORTS, sorted(imported - ALLOWED_DIRECT_IMPORTS)
    if name == "reader":
        assert "stock_agent.publication.writer" not in imported


def test_sources_reference_no_data_or_report_directories():
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "data/" not in text and "reports/" not in text, path


def test_the_lockbox_has_one_source():
    assert envelope.MODEL_HOLDOUT_START is folds.MODEL_HOLDOUT_START


def _write_2021(root):
    spec = synthetic.SyntheticSpec(
        "syn_price_path",
        "1",
        {"n_assets": 2, "start": "2021-03-01", "n_sessions": 5, "seed": 3, "n_missing": 0},
    )
    root.mkdir(parents=True, exist_ok=True)
    return write_publication(
        root,
        synthetic.draft(spec),
        as_of=pd.Timestamp("2021-03-05T23:59:59.999999Z"),
        created_at=pd.Timestamp("2026-10-13T12:00:00Z"),
        code_state=CodeState("d" * 40, False),
    )


def test_writer_and_reader_obey_the_one_lockbox_binding(tmp_path, monkeypatch):
    path = _write_2021(tmp_path / "before")
    monkeypatch.setattr(envelope, "MODEL_HOLDOUT_START", pd.Timestamp("2020-01-01", tz="UTC"))
    for call, args in ((_write_2021, (tmp_path / "after",)), (read_publication, (path,))):
        with pytest.raises(PublicationRefused) as caught:
            call(*args)
        assert caught.value.reason is RefusalReason.LOCKBOX_VIOLATION
    assert list((tmp_path / "after").iterdir()) == []
