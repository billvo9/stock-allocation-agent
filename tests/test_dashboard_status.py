"""
Run status and research notices come from stored check rows only:
severity-aware reading of `passed`, no escalation of warnings or info rows,
and undetermined blocking checks never shown as valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.dashboard import loader, status


def _checks(rows):
    frame = pd.DataFrame(rows, columns=["check", "model", "severity", "passed"])
    frame["passed"] = frame["passed"].astype("boolean")
    return frame


def _summary(leakage=0, instrument=0, warnings=0):
    return {
        "leakage_checks_failed": leakage,
        "instrument_checks_failed": instrument,
        "research_warnings": warnings,
    }


def test_passed_is_read_by_severity():
    checks = _checks(
        [
            ("lockbox_closed", None, "leakage", True),
            ("canary", "m", "leakage", False),
            ("size", None, "instrument", pd.NA),
            ("control_apparent_skill", "c", "research_warning", True),
            ("control_apparent_skill", "d", "research_warning", False),
            ("fpr", None, "info", False),
        ]
    )
    assert status.check_outcomes(checks).tolist() == [
        "pass",
        "fail",
        "undetermined",
        "warning_active",
        "warning_inactive",
        "info",
    ]


@pytest.mark.parametrize(
    ("rows", "summary", "level"),
    [
        ([("a", None, "leakage", True), ("i", None, "info", False)], _summary(), status.VALID),
        (
            [("a", None, "leakage", True), ("w", "c", "research_warning", True)],
            _summary(warnings=1),
            status.WARNING,
        ),
        ([("a", None, "instrument", False)], _summary(instrument=1), status.BLOCKED),
        ([("a", None, "leakage", False)], _summary(leakage=1), status.BLOCKED),
        ([("a", None, "leakage", pd.NA)], _summary(), status.BLOCKED),
        ([("a", None, "leakage", True)], _summary(leakage=1), status.BLOCKED),
    ],
    ids=[
        "info_never_escalates",
        "active_research_warning",
        "instrument_failure",
        "leakage_failure",
        "undetermined_is_never_valid",
        "summary_disagrees_with_rows",
    ],
)
def test_run_status_mapping(rows, summary, level):
    assert status.run_status(_checks(rows), summary).level == level


def test_stored_summary_matches_the_dashboard_tallies(dashboard_root):
    root, run = dashboard_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    result = status.run_status(view.tables["checks"], view.record["checks_summary"])
    assert result.summary_matches
    assert result.counts["leakage_failed"] == run.record["checks_summary"]["leakage_checks_failed"]
    assert result.counts["research_warnings"] == run.record["checks_summary"]["research_warnings"]


def test_notices_quote_stored_facts_without_escalating(dashboard_root):
    root, _ = dashboard_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    notices = status.research_notices(view)
    categories = {notice.category for notice in notices}
    assert {"universe", "lockbox", "undefined"} <= categories
    outcomes = status.check_outcomes(view.tables["checks"])
    blocking_checks = int(outcomes.isin(["fail", "undetermined"]).sum())
    assert sum(n.level == "blocking" for n in notices) == (1 if blocking_checks else 0)
    drift = [n for n in notices if n.category == "drift"]
    assert all(
        n.level == "notice" and "never a reason to remove" in n.detail.lower() for n in drift
    )
    undefined = next(n for n in notices if n.category == "undefined")
    stored = view.tables["metrics"]
    assert undefined.rows["rows"].sum() == int((stored["status"] == "unavailable").sum())
    lockbox = next(n for n in notices if n.category == "lockbox")
    assert "holdout_values_used False" in lockbox.detail


def test_a_dirty_tree_is_surfaced_as_a_notice(dashboard_root):
    root, _ = dashboard_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    assert not any(n.category == "provenance" for n in status.research_notices(view))
    view.record["spec"]["code"]["git_dirty"] = True
    try:
        provenance = [n for n in status.research_notices(view) if n.category == "provenance"]
        assert len(provenance) == 1 and provenance[0].level == "notice"
    finally:
        view.record["spec"]["code"]["git_dirty"] = False


def test_status_rules_are_published():
    assert [rule[0] for rule in status.STATUS_RULES] == [
        status.BLOCKED,
        status.WARNING,
        status.VALID,
    ]
    assert np.all([isinstance(rule[1], str) and rule[1] for rule in status.STATUS_RULES])


def test_undetermined_research_warnings_are_noticed_but_never_escalate():
    checks = _checks(
        [
            ("lockbox_closed", None, "leakage", True),
            ("control_apparent_skill", "c", "research_warning", pd.NA),
        ]
    )
    assert status.run_status(checks, _summary()).level == status.VALID


def test_notices_never_quote_the_unsafe_canary_reference(dashboard_root):
    root, _ = dashboard_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    for notice in status.research_notices(view):
        if notice.source == "checks":
            continue  # stored check rows may name the positive control they test
        text = notice.title + notice.detail
        if notice.rows is not None:
            text += notice.rows.to_csv(index=False)
        assert "canary_unsafe_reference" not in text, notice.title
