"""
The Model Health view model: plain-language headline and next action first,
facts taken from the summary (never invented, never a zero for a missing
value), areas worst-first, the age warning, provenance, and the caveat that
nothing acts automatically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stock_agent.health import view
from stock_agent.health.context import CodeState
from stock_agent.health.model import Area, Better, Check, HealthState, NotApplicable, Scope
from stock_agent.health.summary import build_summary

NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)


def check(check_id, state=HealthState.HEALTHY, value=1, unit="units", **fields) -> Check:
    base = {
        "area": Area(check_id.split(".")[0]),
        "title": check_id,
        "measured_at": NOW,
        "threshold": "t",
        "summary": f"{check_id} summary.",
        "detail": "technical",
        "evidence": "evidence/x",
        "reason": None if state is HealthState.HEALTHY else "why",
    }
    return Check(check_id=check_id, state=state, value=value, unit=unit, **{**base, **fields})


def summary(*checks, scope=Scope.SYNTHETIC, context=None, generated=NOW):
    return build_summary(
        checks,
        generated_at=generated,
        code=CodeState("a" * 40, False),
        scope=scope,
        run_context=context or {"environment": "local"},
    )


def build(*checks, **kwargs):
    return view.build_view(summary(*checks, **kwargs), now=NOW)


def test_a_healthy_summary_needs_no_action():
    result = build(check("ci.tests", value=1958, unit="tests passed", title="Test suite"))
    assert (result.state_word, result.level, result.notice) == ("Healthy", "VALID", "success")
    assert result.headline == "Everything measured is healthy"
    assert result.next_action == "No action needed."
    assert result.facts[0] == view.Fact(
        "Tests passed", "1,958", "Healthy; measured 2026-10-14 18:00 UTC"
    )


def test_the_worst_check_drives_the_next_action():
    result = build(
        check("data.coverage", HealthState.DEGRADED, title="Session coverage"),
        check("deps.pins", HealthState.FAILED, title="Pinned dependencies", summary="Pins broke."),
    )
    assert result.headline == "Something is broken"
    assert result.next_action.startswith("Start with “Pinned dependencies”: Pins broke.")
    assert "Nothing was changed automatically." in result.next_action
    assert [area.title for area in result.areas] == ["Dependencies", "Data quality"]


def test_waiting_for_evidence_is_said_plainly():
    waiting = [
        check(
            f"portfolio.m{index}",
            HealthState.INSUFFICIENT_EVIDENCE,
            None,
            measured_at=None,
            title=f"Measure {index}",
        )
        for index in range(6)
    ]
    result = build(check("ci.tests"), *waiting)
    assert result.state_word == "Insufficient evidence"
    assert result.next_action == (
        "No action needed for what was measured. Waiting for evidence: Measure 0, Measure 1, "
        "Measure 2, Measure 3 and 2 more."
    )
    assert result.summary == "7 checks: 6 insufficient evidence, 1 healthy."


def test_missing_values_are_never_shown_as_numbers():
    result = build(
        check(
            "ci.tests",
            HealthState.INSUFFICIENT_EVIDENCE,
            None,
            measured_at=None,
            title="Test suite",
            summary="No test result was recorded.",
        )
    )
    tests = result.facts[0]
    assert (tests.value, tests.caption) == (
        "—",
        "Insufficient evidence: No test result was recorded.",
    )
    shown = result.areas[0].checks[0]
    assert (shown.value, shown.measured) == ("not measured", "not measured")


@pytest.mark.parametrize(
    ("value", "unit", "text"),
    [
        (0.984375, "fraction of requested fund-sessions with a close", "98.44% (fraction"),
        (4, "of 4 gates passed", "4 of 4 gates passed"),
        (1958, "tests passed", "1,958 tests passed"),
        ("success", None, "success"),
        (0.0123, "largest gap", "0.0123 largest gap"),
    ],
)
def test_values_are_shown_as_read(value, unit, text):
    assert view.value_text(check("data.x", value=value, unit=unit)).startswith(text)


def test_gates_and_ci_facts_come_from_their_checks():
    result = build(
        check("ci.verify", value=4, unit="of 4 gates passed", better=Better.HIGHER),
        check("ci.main_runs", value="success", unit=None),
    )
    labels = {fact.label: fact.value for fact in result.facts}
    assert labels["Quality gates passed"] == "4 of 4"
    assert labels["Latest CI on main"] == "success"
    assert "Tests passed" not in labels  # absent from the summary: not shown at all


def test_old_summaries_warn_and_provenance_names_the_run():
    context = {
        "environment": "github_actions",
        "workflow": "Model health",
        "run_number": "42",
        "run_attempt": "1",
        "event": "schedule",
        "ref": "main",
        "run_url": "https://github.com/o/r/actions/runs/9",
    }
    result = build(check("ci.tests"), context=context, generated=NOW - timedelta(days=3))
    assert result.age_warning.startswith("This summary is 3 days old")
    assert result.provenance[0].startswith("Generated 2026-10-11 18:00 UTC; code aaaaaaaaaaaa")
    assert "run 42 (attempt 1), event schedule, ref main" in result.provenance[1]
    assert result.provenance[1].endswith("https://github.com/o/r/actions/runs/9.")


def test_scope_note_and_caveat_are_always_present():
    public = build(check("ci.tests"))
    private = build(check("ci.tests"), scope=Scope.PRIVATE)
    assert "CI cannot see private data" in public.scope_note
    assert "Keep this summary private" in private.scope_note
    assert "Nothing on this page trades" in public.caveat


def test_runbook_links_point_at_the_checks_anchor():
    shown = build(check("deps.pip_check")).areas[0].checks[0]
    assert shown.runbook_url == (
        "https://github.com/billvo9/stock-allocation-agent/blob/main/"
        "docs/operations/runbook.md#deps-pip-check"
    )


def test_an_empty_summary_is_not_healthy():
    result = view.build_view(summary(), now=NOW)
    assert result.state_word == "Insufficient evidence"
    assert result.summary == "This summary has no checks."


def test_not_applicable_entries_are_listed_apart_from_the_state():
    item = NotApplicable(
        "forecast.coverage", Area.FORECAST, "Forecast interval coverage", "point_forecast", "x."
    )
    result = view.build_view(
        build_summary(
            [check("ci.tests")],
            generated_at=NOW,
            code=CodeState("a" * 40, False),
            scope=Scope.PRIVATE,
            run_context={"environment": "local"},
            not_applicable=[item],
        ),
        now=NOW,
    )
    assert result.state_word == "Healthy"
    assert result.not_applicable == ("Forecast interval coverage (Forecasts): x.",)
