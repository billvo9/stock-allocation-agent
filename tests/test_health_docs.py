"""
The runbook covers every check the health center can emit, each with the
five required parts, and the daily workflow keeps its safety properties:
schedule and manual trigger only, read-only permissions, no secrets, no
private paths, the CI command and an always-uploaded summary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from stock_agent.health import (
    backtest,
    contracts,
    data_quality,
    forecast,
    ledger,
    policy,
    prospective,
    registry,
    stock_lab,
)
from stock_agent.health.model import runbook_for

REPO = Path(__file__).resolve().parents[1]
RUNBOOK = REPO / "docs" / "operations" / "runbook.md"
DESIGN = REPO / "docs" / "operations" / "model_health.md"
WORKFLOW = REPO / ".github" / "workflows" / "health.yml"
PARTS = ("**What it means.**", "**Likely causes.**", "**Diagnose.**", "**Safe retry.**")
FALLBACK = "**Fallback.**"

CHECK_IDS = [
    "ci.verify",
    "ci.tests",
    "ci.main_runs",
    "ci.daily_health",
    "deps.pins",
    "deps.pip_check",
    *contracts._CONTRACTS,
    "smoke.data_monitors",
    "smoke.input_readers",
    *data_quality.CHECK_IDS,
    *prospective.TITLES,
    "training.health",
    "training.ml_runs",
    *registry.CHECKS,
    *forecast.CHECKS,
    *ledger.TITLES,
    *backtest.MEASURES,
    *policy.TITLES,
    *stock_lab.TITLES,
]


def sections() -> dict[str, str]:
    text = RUNBOOK.read_text(encoding="utf-8")
    pieces = re.split(r'<a id="([a-z0-9-]+)"></a>\n', text)
    return dict(zip(pieces[1::2], pieces[2::2], strict=True))


def test_every_check_has_a_runbook_entry_and_no_entry_is_orphaned():
    anchors = sections()
    expected = {runbook_for(check_id).split("#")[1] for check_id in CHECK_IDS}
    assert len(CHECK_IDS) == len(set(CHECK_IDS)) == 41
    assert set(anchors) == expected


@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_each_entry_names_its_check_and_has_every_part(check_id):
    body = sections()[runbook_for(check_id).split("#")[1]]
    entry = body.split("\n<a id=")[0]
    assert f"(`{check_id}`)" in entry.splitlines()[0]
    for part in (*PARTS, FALLBACK):
        assert part in entry, (check_id, part)


def test_the_runbook_promises_no_automatic_action():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "Alerts are informational." in text
    assert "Never make a check pass by changing what it measures." in text


def test_the_runbook_says_where_to_look_how_to_reproduce_retry_and_fall_back():
    headings = re.findall(r"^## (.+)$", RUNBOOK.read_text(encoding="utf-8"), re.MULTILINE)
    for heading in (
        "Where to look",
        "Reproduce a failure",
        "When to retry",
        "When to disable a policy or rely on the fallback",
    ):
        assert heading in headings, heading


def test_every_check_a_local_run_emits_has_a_runbook_entry(tmp_path):
    from stock_agent.health import collect

    def runner(argv, **kwargs):
        import subprocess

        return subprocess.CompletedProcess(argv, 0, "", "")

    environment = collect.Environment(
        root=tmp_path, env={}, runner=runner, installed=lambda name: None, python="python"
    )
    summary, _ = collect.run_local(
        environment, tmp_path / "out", verify=None, inputs=collect.PrivateInputs()
    )
    assert {check.check_id for check in summary.checks} == set(CHECK_IDS)


def load_workflow() -> tuple[dict, str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def test_the_workflow_runs_daily_and_on_demand_only():
    workflow, _ = load_workflow()
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as true
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    (schedule,) = triggers["schedule"]
    assert re.fullmatch(r"\d+ \d+ \* \* \*", schedule["cron"])


def test_the_workflow_is_read_only_and_uses_no_secrets_or_private_paths():
    workflow, text = load_workflow()
    assert workflow["permissions"] == {"contents": "read"}
    (job,) = workflow["jobs"].values()
    assert "permissions" not in job
    assert "secrets." not in text and "GITHUB_TOKEN" not in text
    for private in ("data/", "reports/", "stock-agent-private", "~/", "/Users/"):
        assert private not in text, private
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["persist-credentials"] is False


def test_the_workflow_runs_the_ci_command_and_always_uploads_the_summary():
    workflow, _ = load_workflow()
    (job,) = workflow["jobs"].values()
    runs = [step.get("run", "") for step in job["steps"]]
    assert any("scripts/health_check.py ci --run-verify" in run for run in runs)
    assert not any("health_check.py local" in run for run in runs)
    upload = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/upload"))
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "model-health"


def test_the_design_doc_documents_every_check():
    text = DESIGN.read_text(encoding="utf-8")
    for check_id in CHECK_IDS:
        assert f"`{check_id}`" in text, check_id


def test_every_action_is_pinned_to_a_full_commit_sha():
    _, text = load_workflow()
    uses = re.findall(r"uses: (\S+)(.*)", text)
    assert len(uses) == 3
    for action, comment in uses:
        assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", action), action
        assert re.fullmatch(r" # v\d+\.\d+\.\d+", comment), comment


def test_the_workflow_bounds_its_run_time_and_concurrency():
    workflow, _ = load_workflow()
    (job,) = workflow["jobs"].values()
    assert 0 < job["timeout-minutes"] <= 60
    assert workflow["concurrency"]["group"] == "model-health"
