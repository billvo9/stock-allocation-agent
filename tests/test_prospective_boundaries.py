"""
The prospective store is isolated from research code (owner decision of
2026-10-14: "A test enforces that research code cannot read the prospective
store").

- Research packages never import the prospective store, the rules calendar
  or the Tiingo prospective mode, directly, dynamically or by loading another
  module that does.
- Every research reader refuses a prospective root and a prospective batch.
- Prospective data cannot be ingested into the research store.
- The store's reader (`inference`) and writer (`store`) are separate, and
  `inference` is the only module that reads stored prospective rows.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from prospective_helpers import FUNDS, LATER, batch, day, stored

from stock_agent.data.funds.adapter import RawBatch, SyntheticFundAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import read_batch as read_research_batch
from stock_agent.data.funds.reader import read_store as read_research_store
from stock_agent.data.funds.schema import FundDataRefused, Instrument, RefusalReason
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.data.prospective.validation import MARKER_FILE
from stock_agent.data.prospective.window import H_END, WINDOW_START, ProspectiveRefusal
from stock_agent.forecasting.artifacts import ForecastArtifactError, read_forecast_artifact
from stock_agent.health import collect
from stock_agent.model_diagnostics.artifacts import read_run

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "stock_agent"
PROJECT = SRC.parent

# Modules that may import the prospective side, by path relative to src/stock_agent.
# Everything else, the research Tiingo adapter included, is checked.
STORE_SIDE = (
    "data/prospective/",
    "data/vendors/tiingo_prospective.py",
    "data/vendors/anchors.py",
    "calendars/nyse_rules.py",
)
# The forecast job's modules, by path relative to src/stock_agent. Add a module here
# in the change that introduces it, after review. They may import only these names,
# with `from ... import name` (a whole-module import would reach `read_batch`).
INFERENCE_CONSUMERS: set[str] = set()
# The forecast job's scripts, by path relative to the repository root (reviewed in PR #74),
# and the private Model Health run, which passes the loader, the window constants and the
# rules calendar to research-side checks that never import them: the same rule, so a
# script reaches the store only through the loader.
SCRIPT_CONSUMERS = {
    "scripts/forecast_prospective.py",
    "scripts/health_check.py",
    # The Stock Lab's daily job: the rules calendar only (its expected session and
    # the sessions a stock is behind), passed to the Stock Lab, which never imports it.
    "scripts/stock_lab_daily.py",
    # The Stock Lab's recent-window fetch: the rules calendar only (the 252 sessions it
    # requests), passed to the Stock Lab's inference store, which never imports it.
    "scripts/stock_lab_fetch_recent.py",
    # The Stock Lab's premarket briefing: the rules calendar only (which session a briefing
    # is for, and the weekends and holidays before it), passed to the Stock Lab's briefing
    # modules, which never import it.
    "scripts/stock_lab_briefing.py",
    # The Stock Lab's morning check: the rules calendar only (whether this morning is a
    # session's, and which session's evening is still open), passed to the Stock Lab's
    # morning module, which never imports it.
    "scripts/stock_lab_morning_check.py",
}
CONSUMER_NAMES = {
    "stock_agent.data.prospective.inference": {
        "InferenceWindow",
        "ProspectiveRefused",
        "load_inference_window",
        "PRICE_COLUMNS",
        "DISTRIBUTION_COLUMNS",
        "SPLIT_COLUMNS",
        "MISSING_COLUMNS",
        "BATCH_COLUMNS",
        "REVISION_COLUMNS",
        "NO_BATCH",
        "MISSING_SESSION",
    },
    "stock_agent.data.prospective.window": {"H_END", "LOOKBACK_SESSIONS", "WINDOW_START"},
    # The rules calendar's session list and id, which consumers check every window against.
    "stock_agent.calendars.nyse_rules": {"sessions", "CALENDAR_ID"},
}
FORBIDDEN_MODULES = (
    "stock_agent.data.prospective",
    "stock_agent.data.vendors.tiingo_prospective",
    "stock_agent.calendars.nyse_rules",
)
FORBIDDEN_TEXT = ("data.prospective", "tiingo_prospective", "nyse_rules", MARKER_FILE)
# Research entry points: importing them must load nothing from the prospective side.
RESEARCH_ENTRY_POINTS = (
    "stock_agent.features.training",
    "stock_agent.features.build",
    "stock_agent.features.panel",
    "stock_agent.model_validation.folds",
    "stock_agent.model_diagnostics.runner",
    "stock_agent.forecasting.experiment",
    "stock_agent.forecasting.etf",
    "stock_agent.forecasting.etf_data",
    "stock_agent.forecasting.artifacts",
    "stock_agent.data.funds.reader",
    "stock_agent.data.funds.ingest",
    "stock_agent.health.collect",
    "stock_agent.portfolio.descriptive",
    "stock_agent.rl.ppo",
)
# pandas itself loads `socket` and `urllib.parse`, so the runtime check names the
# request-making modules; the static check bans every network import.
NETWORK_MODULES = ("urllib.request", "http.client", "requests", "urllib3", "httpx", "aiohttp")
NETWORK_IMPORT_ROOTS = {"requests", "urllib", "urllib3", "http", "socket", "httpx", "aiohttp"}


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def _package(path: Path) -> list[str]:
    """The dotted package a module's relative imports resolve against: its directory."""

    return list(path.relative_to(SRC).parent.parts)


def _module(path: Path, node: ast.ImportFrom) -> str:
    """The absolute module of an `ImportFrom`, relative imports resolved."""

    if not node.level:
        return node.module or ""
    package = _package(path) if path.is_relative_to(SRC) else []
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join([*base, *([node.module] if node.module else [])])


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _module(path, node)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _code(path: Path) -> str:
    """The source without its module docstring (a docstring may name a module)."""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if ast.get_docstring(tree, clean=False) is not None:
        text = text.replace(ast.get_source_segment(text, tree.body[0]), "", 1)
    return text


def _forbidden(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN_MODULES)


RESEARCH_MODULES = sorted(
    path
    for path in PACKAGE.rglob("*.py")
    if not _relative(path).startswith(STORE_SIDE) and _relative(path) not in INFERENCE_CONSUMERS
)


@pytest.mark.parametrize("path", RESEARCH_MODULES, ids=_relative)
def test_research_modules_never_reference_the_prospective_side(path):
    assert not sorted(name for name in _imports(path) if _forbidden(name)), path
    code = _code(path)
    assert not [needle for needle in FORBIDDEN_TEXT if needle in code], path  # importlib too


def consumer_violations(path: Path) -> list[str]:
    """Imports of the prospective side beyond `CONSUMER_NAMES`, as `module.name` strings."""

    violations = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            violations += [alias.name for alias in node.names if _forbidden(alias.name)]
        elif isinstance(node, ast.ImportFrom):
            module = _module(path, node)
            names = [f"{module}.{alias.name}" for alias in node.names]
            if not any(_forbidden(name) for name in (module, *names)):
                continue
            allowed = CONSUMER_NAMES.get(module, set())
            violations += [
                f"{module}.{alias.name}" for alias in node.names if alias.name not in allowed
            ]
    return violations


def test_inference_consumers_import_only_the_loader_and_the_window_constants():
    for relative in INFERENCE_CONSUMERS:
        assert consumer_violations(PACKAGE / relative) == [], relative
    for relative in SCRIPT_CONSUMERS:
        path = PROJECT / relative
        assert path.is_file(), relative
        assert consumer_violations(path) == [], relative
        code = _code(path)
        assert "importlib" not in code and "__import__" not in code, relative  # static only


def test_the_consumer_rule_catches_the_writer_and_whole_module_imports(tmp_path):
    sample = tmp_path / "job.py"
    sample.write_text(
        "from stock_agent.data.prospective.inference import load_inference_window, read_batch\n"
        "from stock_agent.data.prospective.window import H_END\n"
        "from stock_agent.data.prospective import store\n"
        "import stock_agent.data.prospective.inference\n",
        encoding="utf-8",
    )
    assert consumer_violations(sample) == [
        "stock_agent.data.prospective.inference.read_batch",
        "stock_agent.data.prospective.store",
        "stock_agent.data.prospective.inference",
    ]


def test_relative_imports_are_resolved(tmp_path):
    # A research module importing the reader relatively must still be caught.
    module = PACKAGE / "data" / "funds" / "adapter.py"
    node = ast.parse("from ..prospective.inference import load_inference_window").body[0]
    assert _module(module, node) == "stock_agent.data.prospective.inference"
    node = ast.parse("from . import schema").body[0]
    assert _module(module, node) == "stock_agent.data.funds"
    package = PACKAGE / "data" / "__init__.py"
    node = ast.parse("from .prospective import store").body[0]
    assert _module(package, node) == "stock_agent.data.prospective"


def test_the_store_side_modules_import_only_what_they_need():
    anchors = {n for n in _imports(PACKAGE / "data/vendors/anchors.py") if _forbidden(n)}
    assert anchors <= {
        "stock_agent.data.prospective.window",
        "stock_agent.data.prospective.window.WINDOW_START",
        "stock_agent.calendars.nyse_rules",  # the rules calendar checks ex-dates
    }
    collector = _imports(PACKAGE / "data/vendors/tiingo_prospective.py")
    assert not [n for n in collector if n.startswith("stock_agent.data.prospective.inference")]


def test_the_research_scripts_use_no_prospective_reader():
    for script in sorted((PROJECT / "scripts").glob("*.py")):
        if script.name in {"collect_prospective.py", "audit_fund_coverage.py"}:
            continue
        if f"scripts/{script.name}" in SCRIPT_CONSUMERS:
            continue  # held to the stricter consumer rule instead (loader and calendar only)
        imported = _imports(script)
        assert not sorted(name for name in imported if _forbidden(name)), script.name
    audit = _imports(PROJECT / "scripts" / "audit_fund_coverage.py")
    # The audit only needs the marker's name, to refuse a prospective root.
    assert {name for name in audit if _forbidden(name)} == {
        "stock_agent.data.prospective.validation",
        "stock_agent.data.prospective.validation.MARKER_FILE",
    }


def _loaded(statement: str) -> set[str]:
    script = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_research_entry_points_load_nothing_from_the_prospective_side():
    loaded = _loaded("\n".join(f"import {name}" for name in RESEARCH_ENTRY_POINTS))
    assert set(RESEARCH_ENTRY_POINTS) <= loaded
    assert not sorted(name for name in loaded if _forbidden(name))
    assert not sorted(name for name in loaded if name.startswith("stock_agent.data.vendors"))


def test_the_research_tiingo_adapter_loads_only_the_window_constants():
    # The research adapter takes anchors (`anchors.py`), whose scopes need WINDOW_START;
    # it loads no prospective writer, reader, contract or collector.
    loaded = _loaded("import stock_agent.data.vendors.tiingo")
    prospective = sorted(name for name in loaded if name.startswith("stock_agent.data.prospective"))
    assert prospective == ["stock_agent.data.prospective", "stock_agent.data.prospective.window"]
    assert "stock_agent.data.vendors.tiingo_prospective" not in loaded


def test_the_prospective_store_loads_no_network_or_vendor_module():
    loaded = _loaded(
        "import stock_agent.data.prospective.inference, stock_agent.data.prospective.store"
    )
    assert not [name for name in loaded if name in NETWORK_MODULES]
    assert not [name for name in loaded if name.startswith("stock_agent.data.vendors")]
    assert not [name for name in loaded if name.startswith("stock_agent.publication")]
    for path in (PACKAGE / "data" / "prospective").glob("*.py"):
        roots = {name.split(".")[0] for name in _imports(path)}
        assert not roots & NETWORK_IMPORT_ROOTS, path.name
        assert not [name for name in _imports(path) if name.startswith("stock_agent.data.vendors")]


def test_the_reader_never_loads_the_writer_and_is_the_only_reader():
    loaded = _loaded("import stock_agent.data.prospective.inference")
    assert "stock_agent.data.prospective.store" not in loaded
    assert "stock_agent.data.prospective.store" not in _imports(
        PACKAGE / "data" / "prospective" / "inference.py"
    )
    readers = sorted(
        _relative(path)
        for path in PACKAGE.rglob("*.py")
        if "read_parquet" in path.read_text(encoding="utf-8")
        and any(_forbidden(name) for name in _imports(path))
    )
    assert readers == ["data/prospective/inference.py"]
    writers = sorted(
        _relative(path)
        for path in (PACKAGE / "data" / "prospective").glob("*.py")
        if any(
            call in path.read_text(encoding="utf-8")
            for call in ("to_parquet", "write_bytes", "os.rename", "mkdtemp")
        )
    )
    assert writers == ["data/prospective/store.py"]


# --- runtime: research readers refuse the prospective store ----------------------------


@pytest.fixture
def prospective_root(tmp_path) -> Path:
    return stored(tmp_path / "prospective", batch())


def _batch_dir(root: Path) -> Path:
    return next(entry for entry in root.iterdir() if entry.name != MARKER_FILE)


def test_f1_readers_refuse_a_prospective_root_and_batch(prospective_root):
    with pytest.raises(FundDataRefused) as caught:
        read_research_store(prospective_root)
    assert caught.value.reason is RefusalReason.NOT_A_BATCH
    with pytest.raises(FundDataRefused) as caught:
        read_research_batch(_batch_dir(prospective_root))
    assert caught.value.reason is RefusalReason.NOT_A_BATCH


def test_the_model_health_fund_store_check_refuses_a_prospective_root(prospective_root):
    checks = collect._fund_store_checks(prospective_root, as_of=datetime(2026, 10, 15, tzinfo=UTC))
    assert checks and all("read_store refused" in check.detail for check in checks)


@pytest.mark.parametrize("reader", [read_run, read_forecast_artifact])
def test_diagnostics_and_forecast_readers_refuse_a_prospective_root(prospective_root, reader):
    for target in (prospective_root, _batch_dir(prospective_root)):
        with pytest.raises((ValueError, FileNotFoundError, ForecastArtifactError, KeyError)):
            reader(target)


# --- runtime: prospective data never enters the research store ------------------------


def _research_root(tmp_path) -> Path:
    root = tmp_path / "research"
    root.mkdir()
    return root


def test_a_prospective_batch_cannot_be_ingested_into_the_research_store(tmp_path):
    root = _research_root(tmp_path)
    with pytest.raises(FundDataRefused) as caught:
        ingest_batch(root, batch())
    assert caught.value.reason is RefusalReason.INVALID_FIELD
    item = batch()
    disguised = RawBatch(
        adapter_id=item.adapter_id,
        adapter_version=item.adapter_version,
        fund_ids=item.fund_ids,
        start=item.start,
        end_exclusive=item.end + pd.Timedelta(days=1),
        instruments=item.instruments,
        retrieved_at=item.retrieved_at,
        raw_payload=item.raw_payload,
        terms_note=item.terms_note,
        vintage_note=item.vintage_note,
        prices=item.prices,
        distributions=item.distributions,
        splits=item.splits,
    )
    with pytest.raises(FundDataRefused) as caught:
        ingest_batch(root, disguised)
    assert caught.value.reason is RefusalReason.REQUEST_AFTER_CAP
    assert list(root.iterdir()) == []


def test_a_research_batch_in_a_prospective_root_refuses_that_store(prospective_root):
    raw = SyntheticFundAdapter(retrieved_at=pd.Timestamp("2026-10-14T00:00:00Z")).fetch(
        ["SYN_EQ"], day("2024-12-02"), day("2025-01-01")
    )
    ingest_batch(prospective_root, raw)  # F1's writer does not know the marker
    with pytest.raises(FundDataRefused) as caught:
        load_inference_window(prospective_root, FUNDS, H_END, retrieved_by=LATER)
    assert caught.value.reason == ProspectiveRefusal.FOREIGN_ENTRY


def test_the_audit_refuses_a_prospective_root(prospective_root, tmp_path, capsys):
    script = PROJECT / "scripts" / "audit_fund_coverage.py"
    spec = importlib.util.spec_from_file_location("stock_agent_audit_boundary", script)
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)

    def never():
        raise AssertionError("the adapter must not be built")

    repo = tmp_path / "repo"
    (repo / "data" / "raw").mkdir(parents=True)
    before = sorted(prospective_root.rglob("*"))
    for root in (prospective_root, prospective_root / "inner"):
        argv = ["--authorized", "--store", "--root", str(root)]
        assert audit.main(argv, adapter_factory=never, project_root=repo) == 2
        assert "prospective store" in capsys.readouterr().err
    assert sorted(prospective_root.rglob("*")) == before


def test_the_window_constants_bound_every_stored_row(prospective_root):
    result = load_inference_window(prospective_root, FUNDS, H_END, retrieved_by=LATER)
    assert result.prices["session_date"].min() >= WINDOW_START
    assert result.prices["session_date"].max() <= H_END
    assert Instrument("AGG", "AGG", None)  # the F1 instrument type is shared, unchanged
