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
from stock_agent.model_diagnostics import runner


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


# --- per-family summary cross-checks (fail closed) ---

FAMILY_ROWS = [
    ("lockbox_closed", None, "leakage", True),
    ("predictions_complete", None, "leakage", True),
    ("size", None, "instrument", True),
    ("control_apparent_skill", "c", "research_warning", False),
    ("control_apparent_skill", "d", "research_warning", pd.NA),
    ("fpr", None, "info", False),
]


def _recorded(rows):
    checks = _checks(rows)
    return checks, runner.summarize_checks(checks)


def test_a_recorded_summary_that_matches_the_rows_does_not_block():
    checks, summary = _recorded(FAMILY_ROWS)
    result = status.run_status(checks, summary)
    assert result.level == status.VALID
    assert result.summary_matches and result.summary_source == "recorded"
    assert result.families == {
        family: {key: entry[key] for key in status.FAMILY_KEYS}
        for family, entry in summary["families"].items()
    }


def test_the_summary_records_undetermined_blocking_checks():
    # The T3-found defect: an undetermined leakage check was absent from the
    # 1.0 summary. 1.1 counts it, the counts match the rows, and the run is
    # still blocked by the check itself.
    checks, summary = _recorded([*FAMILY_ROWS, ("canary", "m", "leakage", pd.NA)])
    assert summary["families"]["leakage"]["undetermined"] == 1
    result = status.run_status(checks, summary)
    assert result.summary_matches
    assert result.level == status.BLOCKED and result.reasons == ["canary: undetermined"]


def _bump(family, key, by=1):
    def change(summary):
        summary["families"][family][key] += by

    return change


def _balanced(family, source, target):
    def change(summary):  # identity still holds; only the split disagrees with the rows
        summary["families"][family][source] -= 1
        summary["families"][family][target] += 1

    return change


def _drop_family(summary):
    summary["families"].pop("instrument")


def _extra_family(summary):
    summary["families"]["custom"] = {"total": 0, "passed": 0, "failed": 0, "undetermined": 0}


def _families_not_a_mapping(summary):
    summary["families"] = [1, 2]


def _family_missing_a_count(summary):
    summary["families"]["leakage"].pop("undetermined")


def _legacy_key_disagrees(summary):
    summary["research_warnings"] = 3


SUMMARY_DEFECTS = [
    ("total_too_high", _bump("leakage", "total"), "passed + failed + undetermined != total"),
    ("passed_too_high", _bump("leakage", "passed"), "passed + failed + undetermined != total"),
    ("failed_too_high", _bump("info", "failed"), "passed + failed + undetermined != total"),
    (
        "undetermined_hidden",
        _bump("research_warning", "undetermined", -1),
        "passed + failed + undetermined != total",
    ),
    ("split_disagrees", _balanced("research_warning", "undetermined", "failed"), "but rows give"),
    ("family_missing", _drop_family, "covers"),
    ("family_extra", _extra_family, "covers"),
    ("not_a_mapping", _families_not_a_mapping, "not a mapping"),
    ("count_missing", _family_missing_a_count, "lacks"),
    ("legacy_key", _legacy_key_disagrees, "checks_summary.research_warnings"),
]


@pytest.mark.parametrize(
    ("change", "reason"), [d[1:] for d in SUMMARY_DEFECTS], ids=[d[0] for d in SUMMARY_DEFECTS]
)
def test_any_summary_row_disagreement_blocks_the_run(change, reason):
    checks, summary = _recorded(FAMILY_ROWS)
    change(summary)
    result = status.run_status(checks, summary)
    assert result.level == status.BLOCKED
    assert not result.summary_matches
    assert any(reason in item for item in result.reasons), result.reasons


def test_a_missing_summary_blocks_the_run():
    checks, _ = _recorded(FAMILY_ROWS)
    result = status.run_status(checks, None)
    assert result.level == status.BLOCKED and "checks_summary is missing" in result.reasons[0]


def test_a_legacy_summary_is_checked_on_its_own_keys_and_families_come_from_rows():
    checks, summary = _recorded(FAMILY_ROWS)
    summary.pop("families")
    result = status.run_status(checks, summary)
    assert result.level == status.VALID and result.summary_matches
    assert result.summary_source.startswith("legacy")
    assert result.families == status.family_counts(checks)


def test_record_issues_block_the_run():
    checks, summary = _recorded(FAMILY_ROWS)
    result = status.run_status(checks, summary, record_issues=("decision method conflict",))
    assert result.level == status.BLOCKED and result.reasons == ["decision method conflict"]


def test_the_fixture_run_summary_matches_its_rows_family_by_family(dashboard_root):
    root, run = dashboard_root
    view = loader.load_run(loader.list_runs(root)[0].path)
    result = status.run_status(
        view.tables["checks"], view.record["checks_summary"], record_issues=view.record_issues
    )
    assert result.summary_matches and result.summary_source == "recorded"
    for family, entry in run.record["checks_summary"]["families"].items():
        assert result.families[family] == {key: entry[key] for key in status.FAMILY_KEYS}
        assert entry["passed"] + entry["failed"] + entry["undetermined"] == entry["total"]
