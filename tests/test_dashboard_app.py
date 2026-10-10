"""
App smoke tests with Streamlit's AppTest against compact fixture runs.

Every page renders from stored outputs; unavailable results show their
stored reason; the unsafe canary reference is confined to its panel;
refused runs and empty roots show a message instead of charts; and
rendering every page imports no model, statistics, or raw-data module.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from stock_agent.dashboard import loader, status
from stock_agent.model_diagnostics import artifacts, contract, runner

APP = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "dashboard" / "app.py"
PAGES = [
    "page_scripts/overview.py",
    "page_scripts/folds.py",
    "page_scripts/features.py",
    "page_scripts/nulls.py",
    "page_scripts/validation.py",
    "page_scripts/calibration.py",
    "page_scripts/uncertainty.py",
    "page_scripts/research_warnings.py",
    "page_scripts/glossary.py",
]
FORBIDDEN = (
    "stock_agent.model_diagnostics.runner",
    "stock_agent.model_diagnostics.scoring",
    "stock_agent.model_diagnostics.inference",
    "stock_agent.model_diagnostics.controls",
    "stock_agent.model_diagnostics.placebo",
    "stock_agent.model_diagnostics.canary",
    "stock_agent.model_diagnostics.feature_diagnostics",
    "stock_agent.model_validation.harness",
    "stock_agent.model_validation.preprocessing",
    "stock_agent.features.build",
    "stock_agent.data",
    "duckdb",
    "yfinance",
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(root, monkeypatch) -> AppTest:
    monkeypatch.setenv(loader.ROOT_ENVIRONMENT_VARIABLE, str(root))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    return app


def _with_checks(run, run_id, *, warnings_active: bool):
    """A copy of the fixture run whose stored checks all pass (one optional warning)."""

    checks = run.tables["checks"].copy()
    blocking = checks["severity"].isin(["leakage", "instrument"])
    checks.loc[blocking, "passed"] = True
    research = checks["severity"] == "research_warning"
    checks.loc[research, "passed"] = False
    if warnings_active:
        checks.loc[research[research].index[:1], "passed"] = True
    record = copy.deepcopy(run.record)
    record["run_id"] = run_id
    checks = contract.conform("checks", checks)
    record["checks_summary"] = runner.summarize_checks(checks)
    tables = {**run.tables, "checks": checks}
    return dataclasses.replace(run, run_id=run_id, record=record, tables=tables)


def test_every_page_renders_from_stored_outputs(dashboard_root, monkeypatch):
    root, run = dashboard_root
    app = _app(root, monkeypatch)
    assert not app.exception
    assert app.sidebar.selectbox[0].value == run.run_id
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])
        assert app.title, page


@pytest.mark.parametrize(
    ("warnings_active", "level", "element"),
    [(None, "INVALID/BLOCKED", "error"), (True, "WARNING", "warning"), (False, "VALID", "success")],
    ids=["fixture_blocked", "warning", "valid"],
)
def test_the_banner_shows_the_status_derived_from_stored_checks(
    dashboard_root, tmp_path, monkeypatch, warnings_active, level, element
):
    root, run = dashboard_root
    if warnings_active is not None:
        variant = _with_checks(
            run, "20991231T000000Z-cccccccccccc", warnings_active=warnings_active
        )
        artifacts.write_run(variant, tmp_path)
        root = tmp_path
    view = loader.load_run(loader.list_runs(root)[0].path)
    stored = status.run_status(
        view.tables["checks"], view.record["checks_summary"], record_issues=view.record_issues
    )
    assert stored.level == level
    app = _app(root, monkeypatch)
    banners = getattr(app, element)
    assert banners and banners[0].value.startswith(f"**{level}**")


def test_overview_shows_the_stored_provenance(dashboard_root, monkeypatch):
    root, run = dashboard_root
    app = _app(root, monkeypatch)
    markdown = " ".join(m.value for m in app.markdown)
    assert run.run_id in markdown
    assert run.record["spec"]["label"]["label_id"] in markdown
    assert "1 attempts" in markdown


def _rendered_text(app) -> str:
    """Rendered text and tables, except stored check rows (which may name the positive control)."""

    parts = [m.value for m in (*app.markdown, *app.caption, *app.info, *app.warning, *app.error)]
    parts = [part for part in parts if "canary_detects_unpurged_leakage" not in part]
    for frame in app.dataframe:
        table = frame.value
        if {"check", "severity"} <= set(table.columns):
            continue  # the stored checks table: names the check's subject, draws nothing
        parts.append(table.to_csv(index=False))
    parts += [str(table.value) for table in app.table]
    return " ".join(parts)


def test_unsafe_canary_reference_is_confined_to_its_panel(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    assert "canary_unsafe_reference" not in app.sidebar.multiselect[0].options
    for page in PAGES:
        app.switch_page(page).run()
        for chart in app.get("plotly_chart"):
            key = chart.proto.id.rsplit("-", 1)[-1]
            if key == "canary":
                continue
            assert "canary_unsafe_reference" not in chart.proto.spec, (page, key)
            assert "UNSAFE" not in chart.proto.spec, (page, key)
        assert "canary_unsafe_reference" not in _rendered_text(app), page
        for box in app.selectbox:
            assert "canary_unsafe_reference" not in box.options, (page, box.key)
    app.switch_page("page_scripts/nulls.py").run()
    canary = [c for c in app.get("plotly_chart") if c.proto.id.endswith("-canary")]
    assert len(canary) == 1 and "UNSAFE reference" in canary[0].proto.spec


def test_no_widget_can_choose_the_output_root(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.text_input and not app.sidebar.text_input, page
        assert not app.get("file_uploader"), page


def test_unavailable_metrics_show_their_stored_reason(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    app.switch_page("page_scripts/validation.py").run()
    app.selectbox(key="validation_model").set_value("zero").run()
    captions = " ".join(c.value for c in app.caption)
    assert "unavailable: constant_prediction" in captions


def test_clearing_the_model_selection_shows_no_models(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    app.sidebar.multiselect[0].set_value([]).run()
    app.switch_page("page_scripts/validation.py").run()
    assert any("Select at least one model" in i.value for i in app.info)


def test_refused_runs_and_empty_roots_show_messages_not_charts(
    dashboard_root, tmp_path, monkeypatch
):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    record["spec"]["lockbox"]["mode"] = "holdout"
    record["run_id"] = "20991231T000000Z-bbbbbbbbbbbb"
    artifacts.write_run(dataclasses.replace(run, run_id=record["run_id"], record=record), tmp_path)
    app = _app(tmp_path, monkeypatch)
    assert any("holdout_run_locked" in e.value for e in app.error)
    assert not app.get("plotly_chart")

    empty = _app(tmp_path / "empty", monkeypatch)
    assert any("No runs under" in i.value for i in empty.sidebar.info)


def test_rendering_every_page_imports_no_model_statistics_or_raw_data_module(dashboard_root):
    root, _ = dashboard_root
    script = f"""
import json, sys
from streamlit.testing.v1 import AppTest
app = AppTest.from_file({str(APP)!r}, default_timeout=180)
app.run()
assert not app.exception, [e.value for e in app.exception]
for page in {PAGES!r}:
    app.switch_page(page).run()
    assert not app.exception, (page, [e.value for e in app.exception])
print(json.dumps(sorted(sys.modules)))
"""
    environment = {**os.environ, loader.ROOT_ENVIRONMENT_VARIABLE: str(root)}
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    present = sorted(m for m in loaded if m.startswith(FORBIDDEN))
    assert not present, present
    assert "stock_agent.dashboard.views" in loaded


def test_the_launcher_binds_to_localhost_without_usage_statistics():
    import importlib.util

    path = APP.parents[3] / "scripts" / "run_dashboard.py"
    spec = importlib.util.spec_from_file_location("stock_agent_run_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.command(8501)
    pairs = dict(zip(command[5::2], command[6::2], strict=True))
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert pairs["--server.address"] == "127.0.0.1"
    assert pairs["--browser.gatherUsageStats"] == "false"
    assert pairs["--server.headless"] == "true"
    assert module.ROOT_ENVIRONMENT_VARIABLE == loader.ROOT_ENVIRONMENT_VARIABLE


def test_every_model_and_selection_renders_without_errors(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    names = app.sidebar.multiselect[0].options
    app.sidebar.multiselect[0].set_value(names).run()
    for page, key in (
        ("page_scripts/validation.py", "validation_model"),
        ("page_scripts/uncertainty.py", "uncertainty_model"),
        ("page_scripts/calibration.py", "calibration_model"),
    ):
        app.switch_page(page).run()
        assert any(box.key == key for box in app.selectbox), (page, key)
        for name in app.selectbox(key=key).options:
            app.selectbox(key=key).set_value(name).run()
            assert not app.exception, (page, name, [e.value for e in app.exception])
    app.switch_page("page_scripts/validation.py").run()
    for baseline in app.selectbox(key="baseline").options:
        app.selectbox(key="baseline").set_value(baseline).run()
        assert not app.exception
    app.switch_page("page_scripts/folds.py").run()
    for fold in app.selectbox(key="fold").options:
        app.selectbox(key="fold").set_value(fold).run()
        assert not app.exception


def test_a_run_changed_after_loading_is_verified_again(dashboard_root, tmp_path, monkeypatch):
    _, run = dashboard_root
    directory = artifacts.write_run(run, tmp_path)
    app = _app(tmp_path, monkeypatch)
    assert not any("integrity_check_failed" in e.value for e in app.error)
    assert app.get("plotly_chart") or app.title
    (directory / "checks.parquet").write_bytes(b"tampered")
    app.run()
    assert any("integrity_check_failed" in e.value for e in app.error)


def test_every_page_renders_a_schema_1_0_run_and_labels_its_gaps(schema_1_0_root, monkeypatch):
    root, _ = schema_1_0_root
    app = _app(root, monkeypatch)
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])
    app.switch_page("page_scripts/overview.py").run()
    markdown = " ".join(m.value for m in app.markdown)
    assert "output schema **1.0**" in markdown
    assert "derived from stored check names (legacy run)" in markdown
    info = " ".join(i.value for i in app.info)
    assert "curves.ci_method: not recorded (schema 1.0)" in info
    families = next(f.value for f in app.dataframe if "family" in f.value.columns)
    assert (families["passed"] + families["failed"] + families["undetermined"]).equals(
        families["total"]
    )


def test_the_overview_shows_the_recorded_contract_fields(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    app = _app(root, monkeypatch)
    markdown = " ".join(m.value for m in app.markdown)
    assert "output schema **1.1**" in markdown and "fold_block_t** (recorded)" in markdown
    assert not any("not recorded (schema" in i.value for i in app.info)
    models = next(f.value for f in app.dataframe if "eligible_as_candidate" in f.value.columns)
    assert models.loc[models["role"] != "candidate", "eligible_as_candidate"].eq(False).all()
