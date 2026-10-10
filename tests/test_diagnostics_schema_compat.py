"""
Output schema 1.0 and 1.1 side by side.

A 1.0 run (written in the PR #43 layout by `write_schema_1_0_run`) must
stay readable, and the dashboard must label what 1.0 never recorded instead
of filling it with assumed values. A 1.1 run must carry every 1.1 field: a
1.1 record that omits or contradicts one is blocked or refused, never
silently derived.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pandas as pd
import pytest

from stock_agent.dashboard import figures, loader, status
from stock_agent.model_diagnostics import artifacts, contract

NEW_CURVE_COLUMNS = ("ci_method", "ci_level", "hac_lag")
CURVES_1_0 = [
    "model",
    "role",
    "scope",
    "curve",
    "group",
    "fold_id",
    "date",
    "x",
    "x_label",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "reason",
]
VARIANT_ID = "20991231T000000Z-bbbbbbbbbbbb"


def _view(root):
    (entry,) = loader.list_runs(root)
    return loader.load_run(entry.path)


def _status(view):
    return status.run_status(
        view.tables["checks"], view.record["checks_summary"], record_issues=view.record_issues
    )


# --- the versioned contract ---


def test_the_1_0_curves_layout_is_frozen():
    assert list(contract.schema("curves", "1.0")) == CURVES_1_0
    assert contract.added_after("curves", "1.0") == NEW_CURVE_COLUMNS
    assert contract.added_after("curves", "1.1") == ()
    assert list(contract.schema("curves", "1.1")) == [
        *CURVES_1_0[:12],
        *NEW_CURVE_COLUMNS,
        *CURVES_1_0[12:],
    ]
    for name in contract.SCHEMAS:
        if name != "curves":
            assert contract.schema(name, "1.0") == contract.schema(name, "1.1"), name


def test_each_version_validates_only_its_own_layout(dashboard_root):
    _, run = dashboard_root
    current = run.tables["curves"]
    legacy = contract.conform("curves", current.drop(columns=list(NEW_CURVE_COLUMNS)), "1.0")
    contract.validate("curves", current, "1.1")
    contract.validate("curves", legacy, "1.0")
    with pytest.raises(ValueError, match="schema 1.0"):
        contract.validate("curves", current, "1.0")
    with pytest.raises(ValueError, match="schema 1.1"):
        contract.validate("curves", legacy, "1.1")


# --- representative 1.0 and 1.1 artifacts ---


def test_both_versions_read_through_the_verified_reader(dashboard_root, schema_1_0_root):
    root_1_1, _ = dashboard_root
    root_1_0, _ = schema_1_0_root
    record_1_1, tables_1_1 = artifacts.read_run(loader.list_runs(root_1_1)[0].path)
    record_1_0, tables_1_0 = artifacts.read_run(loader.list_runs(root_1_0)[0].path)
    assert record_1_1["spec"]["output_schema_version"] == "1.1"
    assert record_1_0["spec"]["output_schema_version"] == "1.0"
    assert list(tables_1_0["curves"].columns) == CURVES_1_0
    assert "families" not in record_1_0["checks_summary"]
    assert "decision_method" not in record_1_0["spec"]
    for name in contract.TABLES:
        shared = list(tables_1_0[name].columns)
        pd.testing.assert_frame_equal(tables_1_0[name], tables_1_1[name][shared])


def test_a_1_1_run_loads_with_every_field_recorded(dashboard_root):
    root, _ = dashboard_root
    view = _view(root)
    assert view.schema_version == "1.1"
    assert view.legacy_gaps == () and view.record_issues == ()
    assert (view.decision_method, view.decision_method_source) == ("fold_block_t", "recorded")
    reference = view.models[view.models["role"] == contract.UNSAFE_REFERENCE_ROLE].iloc[0]
    assert not reference["eligible_as_candidate"]
    canary = view.models[view.models["role"] == "canary"].iloc[0]
    assert reference["output_kind"] == canary["output_kind"]
    result = _status(view)
    assert result.summary_source == "recorded" and result.summary_matches
    curves = view.tables["curves"]
    positions = curves[curves["curve"] == "rank_position_realized"]
    assert figures.interval_label(positions[positions["model"] == "feature_a"]) == (
        "bars: nominal 95% Newey-West (normal critical values) over dates, lag 3"
    )


def test_a_1_0_run_shows_what_it_never_recorded_without_inventing_it(
    dashboard_root, schema_1_0_root
):
    root_1_1, _ = dashboard_root
    root_1_0, _ = schema_1_0_root
    legacy, current = _view(root_1_0), _view(root_1_1)
    assert legacy.schema_version == "1.0"
    assert legacy.record_issues == ()

    # Curve intervals: stored bounds kept; the method, level and lag stay empty.
    curves = legacy.tables["curves"]
    contract.validate("curves", curves)  # upgraded to the current layout for the pages
    for column in NEW_CURVE_COLUMNS:
        assert curves[column].isna().all(), column
    pd.testing.assert_frame_equal(
        curves[CURVES_1_0], current.tables["curves"][CURVES_1_0], check_like=False
    )
    positions = curves[
        (curves["curve"] == "rank_position_realized") & (curves["model"] == "feature_a")
    ]
    assert figures.interval_label(positions) == "bars: interval method not recorded"

    # Decision method: derived from stored check names, and labelled so.
    assert legacy.decision_method == "fold_block_t"
    assert legacy.decision_method_source == "derived from stored check names (legacy run)"

    # Roles: eligibility derived from the stored role; the reference's output kind unknown.
    for model in legacy.models.itertuples(index=False):
        assert model.eligible_as_candidate == (model.role == contract.CANDIDATE_ROLE)
    reference = legacy.models[legacy.models["role"] == contract.UNSAFE_REFERENCE_ROLE]
    assert reference["output_kind"].tolist() == ["not recorded"]
    assert not reference["eligible_as_candidate"].any()

    # Every derivation or absence is listed.
    gaps = " | ".join(legacy.legacy_gaps)
    for expected in (
        "curves.ci_method: not recorded (schema 1.0)",
        "curves.ci_level: not recorded (schema 1.0)",
        "curves.hac_lag: not recorded (schema 1.0)",
        "decision_method: derived from stored check names (legacy run)",
        "models.eligible_as_candidate: derived from the stored role (legacy run)",
        "output kind is not recorded",
        "checks_summary.families: derived from the stored check rows",
    ):
        assert expected in gaps, expected

    # Status: same rows, same verdict; the family counts come from the rows.
    legacy_status, current_status = _status(legacy), _status(current)
    assert legacy_status.level == current_status.level
    assert legacy_status.families == current_status.families
    assert legacy_status.summary_source.startswith("legacy")
    legacy_notice = [n for n in status.research_notices(legacy) if n.category == "legacy"]
    assert len(legacy_notice) == 1 and "not recorded" in legacy_notice[0].detail


def test_a_1_0_run_without_identifiable_decision_checks_stays_unavailable(
    dashboard_root, schema_1_0_writer, tmp_path
):
    _, run = dashboard_root
    checks = run.tables["checks"].copy()
    checks["metric"] = checks["metric"].str.replace("_t_fold_block", "_t_unknown", regex=False)
    run = dataclasses.replace(run, tables={**run.tables, "checks": checks})
    schema_1_0_writer(run, tmp_path)
    view = _view(tmp_path)
    assert view.decision_method is None
    assert view.decision_method_source.startswith("unavailable")


# --- a 1.1 record must carry the 1.1 fields (fail closed) ---


def _variant(run, *, record=None, tables=None):
    record = copy.deepcopy(run.record if record is None else record)
    record["run_id"] = VARIANT_ID
    return dataclasses.replace(
        run, run_id=VARIANT_ID, record=record, tables=tables or dict(run.tables)
    )


def _edit(run, change):
    record = copy.deepcopy(run.record)
    change(record)
    return _variant(run, record=record)


def _drop_families(record):
    record["checks_summary"].pop("families")


def _drop_decision_method(record):
    record["spec"].pop("decision_method")


def _conflicting_decision_method(record):
    record["spec"]["decision_method"] = "newey_west"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_drop_families, "record lacks checks_summary.families"),
        (_drop_decision_method, "record lacks spec.decision_method"),
        (_conflicting_decision_method, "recorded decision_method newey_west disagrees"),
    ],
    ids=["no_families", "no_decision_method", "decision_method_conflict"],
)
def test_a_1_1_record_missing_or_contradicting_a_field_is_blocked(
    dashboard_root, tmp_path, change, reason
):
    _, run = dashboard_root
    artifacts.write_run(_edit(run, change), tmp_path)
    view = _view(tmp_path)
    result = _status(view)
    assert result.level == status.BLOCKED
    assert any(reason in item for item in result.reasons), result.reasons
    blocking = [n for n in status.research_notices(view) if n.category == "record"]
    assert view.record_issues and blocking and blocking[0].level == "blocking"


def _null_marked_eligible(record):
    model = next(m for m in record["spec"]["models"] if m["role"] == "null")
    model["eligible_as_candidate"] = True


def _reference_marked_eligible(record):
    record["spec"]["reference_predictors"][0]["eligible_as_candidate"] = True


def _reference_given_candidate_role(record):
    record["spec"]["reference_predictors"][0]["role"] = contract.CANDIDATE_ROLE


def _drop_eligibility(record):
    record["spec"]["models"][0].pop("eligible_as_candidate")


def _eligibility_not_boolean(record):
    record["spec"]["models"][0]["eligible_as_candidate"] = "no"


def _drop_reference_predictors(record):
    record["spec"].pop("reference_predictors")


def _empty_reference_predictors(record):
    record["spec"]["reference_predictors"] = []


def _unknown_decision_method(record):
    record["spec"]["decision_method"] = "t_test_of_my_choice"


@pytest.mark.parametrize(
    "change",
    [
        _null_marked_eligible,
        _reference_marked_eligible,
        _reference_given_candidate_role,
        _drop_eligibility,
        _eligibility_not_boolean,
        _drop_reference_predictors,
        _empty_reference_predictors,
        _unknown_decision_method,
    ],
)
def test_a_record_that_could_present_a_non_candidate_as_a_strategy_is_refused(
    dashboard_root, tmp_path, change
):
    _, run = dashboard_root
    directory = artifacts.write_run(_edit(run, change), tmp_path)
    artifacts.read_run(directory)  # the files themselves verify
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == "malformed_run"


def test_manifest_and_record_versions_must_agree(dashboard_root, tmp_path):
    _, run = dashboard_root
    directory = artifacts.write_run(run, tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_schema_version"] = "1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema 1.0"):  # 1.1 curves under a 1.0 manifest
        artifacts.read_run(directory)
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == "integrity_check_failed"


def test_a_record_declaring_another_version_than_its_manifest_is_refused(dashboard_root, tmp_path):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    record["spec"]["output_schema_version"] = "1.0"
    directory = artifacts.write_run(_variant(run, record=record), tmp_path)  # manifest says 1.1
    with pytest.raises(ValueError, match="declares schema 1.0 but the manifest says 1.1"):
        artifacts.read_run(directory)


def test_a_record_without_a_schema_version_is_refused(dashboard_root, tmp_path):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    record["spec"].pop("output_schema_version")
    directory = artifacts.write_run(_variant(run, record=record), tmp_path)
    with pytest.raises(ValueError, match="declares schema None"):
        artifacts.read_run(directory)
