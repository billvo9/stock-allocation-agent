"""
The Model Health page under Streamlit's AppTest (missing, refused, healthy
and failing summaries; the standalone app entry), the launcher, and the
import boundaries of the health package: the view and the vocabulary load no
Streamlit or pandas, the page uses only the shared `stock_agent.ui`
components, and nothing in the package imports another track's code or a
network library.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from stock_agent.health import page, training, verification
from stock_agent.health.context import CodeState
from stock_agent.health.model import Area, Check, HealthState, NotApplicable, Scope
from stock_agent.health.summary import build_summary, write_summary

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"
PACKAGE = SOURCE / "stock_agent" / "health"
APP = PACKAGE / "app.py"
TIMEOUT = 120
NOW = datetime.now(UTC)
VERIFY_OK = """verify summary:
  PASS  pytest -q  (exit 0)
  PASS  python -m pytest -q  (exit 0)
  PASS  ruff format --check src tests scripts .claude/hooks  (exit 0)
  PASS  ruff check src tests scripts .claude/hooks  (exit 0)
"""
VERIFY_OK = "1958 passed in 73.68s\n1958 passed in 81.11s\n" + VERIFY_OK


def write(path: Path, *extra: Check, scope=Scope.SYNTHETIC, not_applicable=()) -> Path:
    checks = verification.verify_checks(
        VERIFY_OK, measured_at=NOW - timedelta(minutes=5), now=NOW, evidence="evidence/verify.log"
    )
    summary = build_summary(
        (*checks, *extra),
        generated_at=NOW,
        code=CodeState("6b33c1b215ef89acc600d72021dd013ba6fd56a2", False),
        scope=scope,
        run_context={"environment": "local"},
        not_applicable=not_applicable,
    )
    return write_summary(path, summary)


def failing_pins() -> Check:
    return Check(
        check_id="deps.pins",
        area=Area.DEPS,
        title="Pinned dependencies",
        state=HealthState.FAILED,
        reason="pin_mismatch",
        measured_at=NOW,
        value=5,
        unit="of 6 pins installed at the pinned version",
        threshold="every == pin installed",
        summary="1 pinned package costs $0 to fix but is not at its pin: ruff.",
        detail="ruff==0.16.2 (installed 0.16.3)",
        evidence="requirements.txt",
    )


def _page(sources):
    from stock_agent.health import page
    from stock_agent.ui.components import apply_theme

    apply_theme()
    page.render(sources)


def app_for(ci: Path | None, local: Path | None) -> AppTest:
    sources = page.sources_from_environment(
        {
            page.CI_SUMMARY_VARIABLE: str(ci) if ci else "",
            page.LOCAL_SUMMARY_VARIABLE: str(local) if local else "",
        }
    )
    app = AppTest.from_function(_page, default_timeout=TIMEOUT, kwargs={"sources": sources})
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def text(app) -> str:
    parts = [
        element.value
        for element in (
            *app.title,
            *app.subheader,
            *app.markdown,
            *app.caption,
            *app.info,
            *app.warning,
            *app.error,
            *app.success,
            *app.code,
        )
    ]
    parts += [html.proto.body for html in app.get("html") if "<style>" not in html.proto.body]
    parts += [f"{metric.label} {metric.value}" for metric in app.metric]
    joined = " ".join(parts).replace("**", "")
    return re.sub(r"\\(.)", r"\1", joined)  # markdown escapes read as the characters


def test_missing_summaries_show_how_to_create_them(tmp_path):
    app = app_for(None, tmp_path / "absent.json")
    rendered = text(app)
    assert [tab.label for tab in app.tabs] == ["Latest CI run", "Local monitor"]
    assert rendered.count("No summary yet") == 2
    assert "gh run download" in rendered
    assert "scripts/health_check.py local --run-verify" in rendered
    assert not app.metric  # nothing pretends to be a measurement


def test_a_healthy_ci_summary_leads_with_plain_words_and_real_counts(tmp_path):
    app = app_for(write(tmp_path / "ci.json"), None)
    rendered = text(app)
    assert "Everything measured is healthy" in rendered
    assert "Next action: No action needed." in rendered
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Tests passed"] == "1,958"
    assert metrics["Quality gates passed"] == "4 of 4"
    assert metrics["Checks healthy"] == "2 of 2"
    assert "Nothing on this page trades" in rendered
    assert "CI cannot see private data" in rendered
    assert [re.sub(r"\\(.)", r"\1", e.label) for e in app.expander] == [
        "Measurements: Quality gates (verify.py)",
        "Measurements: Test suite",
    ]
    assert "docs/operations/runbook.md#ci-tests" in rendered


def test_a_failing_summary_names_the_first_thing_to_fix(tmp_path):
    app = app_for(write(tmp_path / "ci.json", failing_pins()), None)
    rendered = text(app)
    assert "Something is broken" in rendered
    assert "Next action: Start with “Pinned dependencies”" in rendered
    assert "Failure: Pinned dependencies is failed" in rendered
    assert app.error[0].value.startswith("**Blocking.** Next action")


def test_a_refused_summary_shows_nothing_from_it(tmp_path):
    bad = tmp_path / "bad.json"
    document = json.loads(write(tmp_path / "ok.json").read_text(encoding="utf-8"))
    document["overall_state"] = "failed"  # the checks say healthy
    bad.write_text(json.dumps(document), encoding="utf-8")
    app = app_for(bad, None)
    rendered = text(app)
    assert "was refused (state_mismatch)" in rendered
    assert "Nothing from it is shown" in rendered
    assert not app.metric


def test_dollar_signs_are_escaped_exactly_once(tmp_path):
    app = app_for(write(tmp_path / "ci.json", failing_pins()), None)
    shown = [e.value for e in (*app.markdown, *app.error, *app.caption) if "$" in e.value]
    assert shown
    for value in shown:
        assert "\\\\$" not in value and re.search(r"(?<!\\)\$", value) is None, value


def test_a_private_summary_says_what_it_waits_for(tmp_path):
    private = write(
        tmp_path / "local.json",
        *training.training_checks(None),
        scope=Scope.PRIVATE,
    )
    app = app_for(None, private)
    rendered = text(app)
    assert "Keep this summary private" in rendered
    assert "Waiting for evidence: Training health" in rendered


def test_the_standalone_app_renders_from_environment_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(page.CI_SUMMARY_VARIABLE, str(write(tmp_path / "ci.json")))
    monkeypatch.delenv(page.LOCAL_SUMMARY_VARIABLE, raising=False)
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not app.exception, [error.value for error in app.exception]
    assert "Model health" in text(app)
    assert {metric.label: metric.value for metric in app.metric}["Tests passed"] == "1,958"


# --- launcher ------------------------------------------------------------------------


def _launcher():
    path = REPO / "scripts" / "run_model_health.py"
    spec = importlib.util.spec_from_file_location("run_model_health", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launcher_binds_to_localhost_without_usage_statistics():
    launcher = _launcher()
    command = launcher.command(8503)
    pairs = dict(zip(command[5::2], command[6::2], strict=True))
    assert command[1:4] == ["-m", "streamlit", "run"] and command[4] == str(APP)
    assert pairs["--server.address"] == "127.0.0.1"
    assert pairs["--browser.gatherUsageStats"] == "false"
    assert (launcher.CI_SUMMARY_VARIABLE, launcher.LOCAL_SUMMARY_VARIABLE) == (
        page.CI_SUMMARY_VARIABLE,
        page.LOCAL_SUMMARY_VARIABLE,
    )


def test_the_launcher_finds_the_newest_downloaded_ci_summary(tmp_path):
    launcher = _launcher()
    assert launcher.newest_ci_summary(tmp_path) is None
    old, new = tmp_path / "100" / "health_summary.json", tmp_path / "200" / "health_summary.json"
    for index, path in enumerate((old, new)):
        path.parent.mkdir()
        path.write_text("{}")
        os.utime(path, (1_000_000 + index, 1_000_000 + index))
    assert launcher.newest_ci_summary(tmp_path) == new
    env = launcher.environment(new, None)
    assert env[page.CI_SUMMARY_VARIABLE] == str(new.resolve())
    assert page.LOCAL_SUMMARY_VARIABLE not in env


# --- import boundaries ---------------------------------------------------------------------

MODULES = sorted(path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__")
OTHER_TRACKS = (
    "stock_agent.app",
    "stock_agent.dashboard",
    "stock_agent.rl",
    "stock_agent.environment",
    "stock_agent.model_diagnostics",
    "stock_agent.portfolio",
)
# The ML track's registry is read through its public API (and written only by
# the synthetic fixture builder); no other health module touches it.
REGISTRY_USERS = {"registry", "fixtures"}
NETWORK = ("requests", "urllib.request", "http.client", "socket", "yfinance", "edgar", "httpx")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


@pytest.mark.parametrize("name", ["__init__", *MODULES])
def test_no_health_module_imports_another_tracks_code_or_the_network(name):
    imported = _imports(PACKAGE / f"{name}.py")
    assert not [m for m in imported if m.startswith(OTHER_TRACKS)], name
    assert not [m for m in imported if m.startswith(NETWORK)], name
    if name not in REGISTRY_USERS:
        assert not [m for m in imported if m.startswith("stock_agent.registry")], name


@pytest.mark.parametrize(
    ("name", "allowed_project"),
    [
        ("model", set()),
        ("view", {"stock_agent.health.model"}),
        ("page", {"stock_agent.health.model", "stock_agent.health.view", "stock_agent.ui"}),
    ],
)
def test_the_page_side_imports_only_the_vocabulary_view_and_ui(name, allowed_project):
    imported = _imports(PACKAGE / f"{name}.py")
    project = {m for m in imported if m.startswith("stock_agent")}
    assert project <= allowed_project, sorted(project - allowed_project)
    third_party = {
        m.split(".")[0]
        for m in imported
        if not m.startswith("stock_agent") and m.split(".")[0] not in sys.stdlib_module_names
    }
    assert third_party <= ({"streamlit"} if name == "page" else set()), third_party


def test_the_page_builds_from_the_shared_components():
    source = (PACKAGE / "page.py").read_text(encoding="utf-8")
    for banned in ("st.title(", "unsafe_allow_html", "st.error(", "st.warning(", "st.html("):
        assert banned not in source, banned
    assert not re.search(r"#[0-9A-Fa-f]{6}\b", source)  # no colour literals
    assert source.count("def _esc(") == 1  # one escaping helper for summary text


def test_loading_the_view_loads_no_streamlit_pandas_or_collector():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys\nimport stock_agent.health.view\n"
                "print(json.dumps(sorted(sys.modules)))"
            ),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SOURCE)},
        timeout=TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert not {"streamlit", "pandas", "numpy"} & loaded
    project = sorted(m for m in loaded if m.startswith("stock_agent"))
    assert project == [
        "stock_agent",
        "stock_agent.health",
        "stock_agent.health.model",
        "stock_agent.health.view",
    ]


def test_not_applicable_checks_are_shown_without_a_state(tmp_path):
    item = NotApplicable(
        "forecast.coverage",
        Area.FORECAST,
        "Forecast interval coverage",
        "point_forecast",
        "All 3 evaluated forecasts are point forecasts.",
    )
    app = app_for(None, write(tmp_path / "local.json", scope=Scope.PRIVATE, not_applicable=[item]))
    rendered = text(app)
    assert "Not applicable" in rendered
    assert "Forecast interval coverage (Forecasts): All 3 evaluated forecasts" in rendered
    assert "Everything measured is healthy" in rendered


def test_markdown_in_producer_text_stays_literal(tmp_path):
    hostile = Check(
        check_id="training.health",
        area=Area.TRAINING,
        title="Training health",
        state=HealthState.FAILED,
        reason="training_check_failed",
        measured_at=NOW,
        value=0,
        unit="of 8 listed training checks pass",
        threshold="all pass",
        summary="Checks failed: ![x](https://example.com/p.png) and [click](https://example.com).",
        detail="detail",
        evidence="reports/health_inputs/rl_training_health/[run](https://example.com).json",
    )
    app = app_for(write(tmp_path / "ci.json", hostile), None)
    sources = [e.value for e in (*app.markdown, *app.caption, *app.error, *app.warning)]
    joined = " ".join(sources)
    assert "\\!\\[x\\]\\(https\\:\\/\\/example" in joined  # escaped, rendered as text
    for value in sources:
        assert re.search(r"(?<!\\)!(?<!\\)\[", value) is None, value
        assert re.search(r"(?<!\\)\]\((?!https://github\.com/)", value) is None, value
    assert "![x](https://example.com/p.png)" in text(app)  # read back as literal characters
