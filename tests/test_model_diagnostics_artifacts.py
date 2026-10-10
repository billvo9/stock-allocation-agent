"""
The saved-output boundary: contract checks, run directories, the ledger,
table fingerprints, and the package's dependency direction.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec
from stock_agent.model_diagnostics import artifacts, contract, runner
from stock_agent.model_diagnostics.record import canonical_json
from stock_agent.model_validation.audit import table_sha256
from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows

SRC = Path(__file__).resolve().parents[1] / "src" / "stock_agent"
SPEC = LabelSpec(horizon=5, entry_lag=1)


@pytest.fixture(scope="module")
def run(panel_factory):
    raw = panel_factory(symbols=("A", "B", "C", "D"), sessions=200, horizon=5, seed=8)
    raw = raw.drop(columns=["target_return", "target_start_date", "target_end_date"])
    lockbox = raw["date"].drop_duplicates().sort_values().iloc[180]
    labeled, facts = runner.prepare_development_frame(
        raw, symbols=["A", "B", "C", "D"], label_spec=SPEC, holdout_start=lockbox
    )
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(sessions, start=sessions.iloc[80], end=lockbox, block_sessions=30)
    folds = make_expanding_folds(labeled, windows, holdout_start=lockbox, lockbox_start=lockbox)
    models = [m for m in runner.default_models(score_feature="feature_a") if m.name != "memorizer"]
    config = runner.DiagnosticsConfig(
        hac_lag=3,
        block_length=10,
        bootstrap_reps=19,
        acf_max_lag=3,
        permutation_draws=1,
        stale_models=(),
        permutation_models=("feature_a",),
        calibration_reps=20,
        calibration_bootstrap_reps=9,
        canary_model="none",
    )
    return runner.run_diagnostics(
        labeled,
        folds,
        universe=["A", "B", "C", "D"],
        label_spec=SPEC,
        diagnostic_features=["feature_a", "feature_b"],
        provenance={
            "code": {"git_sha": "x", "git_dirty": False},
            "environment": {},
            "truncation": facts,
        },
        started_at=datetime(2026, 10, 9, tzinfo=UTC),
        models=models,
        config=config,
        lockbox_start=lockbox,
    )


# --- run directories ---


def test_write_then_read_round_trips_every_table_and_the_record(run, tmp_path):
    directory = artifacts.write_run(run, tmp_path, extra_record={"status": "completed"})
    assert directory == tmp_path / "runs" / run.run_id
    record, tables = artifacts.read_run(directory)
    assert record["run_id"] == run.run_id and record["status"] == "completed"
    assert record["outputs"]["results_sha256"] == run.record["outputs"]["results_sha256"]
    assert set(tables) == set(run.tables)
    for name, table in run.tables.items():
        assert table_sha256(tables[name]) == table_sha256(table), name
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["output_schema_version"] == contract.OUTPUT_SCHEMA_VERSION
    assert not list((tmp_path / "runs").glob(".*partial"))


def test_runs_are_never_overwritten(run, tmp_path):
    artifacts.write_run(run, tmp_path)
    with pytest.raises(FileExistsError):
        artifacts.write_run(run, tmp_path)


def test_reading_refuses_tampered_incomplete_or_newer_runs(run, tmp_path):
    directory = artifacts.write_run(run, tmp_path)
    metrics = directory / "metrics.parquet"
    table = pd.read_parquet(metrics)
    table.loc[0, "estimate"] = 123.0
    table.to_parquet(metrics, index=False)
    with pytest.raises(ValueError, match="manifest hash"):
        artifacts.read_run(directory)

    other = artifacts.write_run(run, tmp_path / "second")
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["output_schema_version"] = "2.0"
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="major version"):
        artifacts.read_run(other)
    (other / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        artifacts.read_run(other)


def test_reading_refuses_an_edited_record(run, tmp_path):
    directory = artifacts.write_run(run, tmp_path)
    record = json.loads((directory / "record.json").read_text())
    record["lockbox_evidence"]["holdout_values_used"] = True
    (directory / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest hash"):
        artifacts.read_run(directory)


def test_record_json_has_no_nan_tokens(run, tmp_path):
    directory = artifacts.write_run(run, tmp_path)
    text = (directory / "record.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    assert canonical_json({"x": float("nan")}) == '{"x":null}'


def test_ledger_appends_one_line_per_attempt(tmp_path):
    artifacts.append_ledger(tmp_path, {"run_id": "a", "status": "completed"})
    artifacts.append_ledger(tmp_path, {"status": "failed", "error": "boom"})
    assert artifacts.read_ledger(tmp_path) == [
        {"run_id": "a", "status": "completed"},
        {"error": "boom", "status": "failed"},
    ]


# --- contract ---


def test_contract_rejects_missing_extra_and_mistyped_columns(run):
    metrics = run.tables["metrics"]
    with pytest.raises(ValueError, match="missing columns"):
        contract.conform("metrics", metrics.drop(columns="estimate"))
    with pytest.raises(ValueError, match="unexpected columns"):
        contract.conform("metrics", metrics.assign(extra=1))
    with pytest.raises(ValueError, match="violates the contract"):
        contract.validate("metrics", metrics.assign(estimate=metrics["estimate"].astype(object)))
    with pytest.raises(ValueError, match="violates the contract"):
        contract.validate("predictions", run.tables["predictions"].assign(date=pd.NaT))
    contract.check_version("1.0")  # every minor up to the reader's own is readable
    contract.check_version(contract.OUTPUT_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="newer minor"):
        contract.check_version("1.7")  # cannot validate columns it does not know
    with pytest.raises(ValueError, match="major"):
        contract.check_version("2.0")


# --- fingerprints ---


def test_table_sha256_covers_values_names_and_order():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    base = table_sha256(frame)
    assert table_sha256(frame.copy()) == base
    assert table_sha256(frame.iloc[::-1]) != base
    assert table_sha256(frame.rename(columns={"b": "c"})) != base
    assert table_sha256(frame.assign(a=[1.0, 2.5])) != base
    assert table_sha256(frame.assign(a=[-0.0 + 1.0, 2.0])) == base


# --- dependency direction and I/O boundary ---


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_model_diagnostics_depends_only_on_validation_and_label_code():
    allowed = ("stock_agent.model_validation", "stock_agent.model_diagnostics")
    allowed_exact = {"stock_agent.features.training"}
    for path in (SRC / "model_diagnostics").glob("*.py"):
        for module in _imports(path):
            if module.startswith("stock_agent"):
                assert module.startswith(allowed) or module in allowed_exact, (path.name, module)
    for path in (SRC / "model_validation").glob("*.py"):
        assert not any("model_diagnostics" in m for m in _imports(path)), path.name


def test_only_the_artifacts_module_touches_files():
    io_calls = {
        "open",
        "to_parquet",
        "read_parquet",
        "to_csv",
        "to_json",
        "write_text",
        "read_text",
        "mkdir",
        "unlink",
        "remove",
        "save",
        "dump",
        "replace",
    }
    for path in (SRC / "model_diagnostics").glob("*.py"):
        if path.name == "artifacts.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not (called & io_calls - {"replace"}), (path.name, called & io_calls)
        assert "pathlib" not in _imports(path), path.name


def test_every_written_metric_has_a_scope_and_a_status(run):
    for name in ("metrics", "curves", "feature_stats"):
        table = run.tables[name]
        assert table["scope"].notna().all()
        assert table["status"].isin(["ok", "unavailable", "warning"]).all()
    unavailable = run.tables["metrics"].query("status == 'unavailable'")
    assert unavailable["reason"].notna().all()
    assert np.isnan(unavailable.loc[unavailable["inference_method"] != "point", "se"]).all()


def _rewrite_manifest(directory, change):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda m: m["files"].update({"../outside.parquet": {"file_sha256": "x"}}),
            "unexpected files",
        ),
        (lambda m: m["files"].pop("record.json"), "lacks"),
        (lambda m: m.update(run_id="20990101T000000Z-000000000000"), "!= directory"),
        (lambda m: m.pop("files"), "missing"),
    ],
    ids=["path_outside_the_run", "record_not_listed", "run_id_mismatch", "no_file_list"],
)
def test_reader_reads_only_the_files_a_run_may_contain(run, tmp_path, change, match):
    directory = artifacts.write_run(run, tmp_path)
    _rewrite_manifest(directory, change)
    with pytest.raises(ValueError, match=match):
        artifacts.read_run(directory)


def test_reader_refuses_symlinked_tables_and_directories(run, tmp_path):
    directory = artifacts.write_run(run, tmp_path / "a")
    outside = tmp_path / "outside.parquet"
    (directory / "checks.parquet").rename(outside)
    (directory / "checks.parquet").symlink_to(outside)
    with pytest.raises(ValueError, match="inside the run directory"):
        artifacts.read_run(directory)
    clean = artifacts.write_run(run, tmp_path / "b")
    link = tmp_path / "link" / run.run_id
    link.parent.mkdir()
    link.symlink_to(clean, target_is_directory=True)
    with pytest.raises(ValueError, match="not a run directory"):
        artifacts.read_run(link)


def test_the_reader_does_not_import_the_statistical_runner():
    import subprocess
    import sys

    code = (
        "import sys, stock_agent.model_diagnostics.artifacts as a;"
        "print(sorted(m for m in sys.modules if m.startswith('stock_agent')))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    for module in ("runner", "scoring", "inference", "harness", "controls"):
        assert f".{module}'" not in loaded, module


def test_a_run_without_a_canary_records_no_reference_predictor(run):
    assert run.record["spec"]["reference_predictors"] == []
    assert "canary_unsafe_reference" not in set(run.tables["predictions"]["model"])
