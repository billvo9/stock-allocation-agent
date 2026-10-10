"""
The dashboard loader: verified, read-only access to saved T2 runs.

Refusal fixtures are rewritten with the real writer, so their hashes and
contracts are valid and each test exercises the intended guard (not a hash
mismatch). Reason codes are asserted.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pandas as pd
import pytest

from stock_agent.dashboard import loader
from stock_agent.model_diagnostics import artifacts, contract
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

VARIANT_ID = "20991231T000000Z-aaaaaaaaaaaa"


def _variant(run, run_id=VARIANT_ID, *, record=None, tables=None):
    record = copy.deepcopy(run.record if record is None else record)
    record["run_id"] = run_id
    return dataclasses.replace(run, run_id=run_id, record=record, tables=tables or dict(run.tables))


def _holdout(record) -> pd.Timestamp:
    return pd.Timestamp(record["spec"]["lockbox"]["holdout_start"])


def test_a_verified_development_run_loads_unchanged(dashboard_root):
    root, run = dashboard_root
    (entry,) = loader.list_runs(root)
    view = loader.load_run(entry.path)
    _, stored = artifacts.read_run(entry.path)
    assert view.run_id == run.run_id
    for name in run.tables:
        pd.testing.assert_frame_equal(view.tables[name], stored[name])
    assert view.decision_method == "fold_block_t"
    assert view.holdout_start == _holdout(run.record)
    assert view.folds["test_start"].dtype.tz is not None
    assert isinstance(view.folds["train_rows_by_symbol"].iloc[0], dict)
    roles = dict(zip(view.models["name"], view.models["role"], strict=True))
    assert roles["canary_unsafe_reference"] == "canary_unsafe_reference"


def test_the_ledger_is_summarised_live_and_tolerates_a_bad_line(dashboard_root, tmp_path):
    root, run = dashboard_root
    summary = loader.ledger_summary(root, run.record["spec_sha256"])
    assert summary == {
        "status": "ok",
        "attempts": 1,
        "failed": 0,
        "same_spec": 1,
        "distinct_variants": 0,
    }
    (tmp_path / "ledger.jsonl").write_text("{not json\n")
    assert loader.ledger_summary(tmp_path, None)["status"] == "unreadable"


def test_run_listing_ignores_partial_symlinked_and_misnamed_directories(dashboard_root, tmp_path):
    _, run = dashboard_root
    artifacts.write_run(run, tmp_path)
    runs = tmp_path / "runs"
    (runs / f".{run.run_id}.partial").mkdir()
    (runs / "notes").mkdir()
    (runs / "20991231T000000Z-ffffffffffff").symlink_to(runs / run.run_id)
    newer = "20991231T000001Z-000000000000"
    (runs / newer).mkdir()
    listed = loader.list_runs(tmp_path)
    assert [entry.run_id for entry in listed] == [newer, run.run_id]
    assert [entry.has_manifest for entry in listed] == [False, True]
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(runs / newer)
    assert refused.value.reason == "incomplete_run"


# --- lockbox guard (each case would expose holdout data if the guard were removed) ---


def _set_record(path, value):
    def change(record, tables):
        node = record
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value(record) if callable(value) else value
        return record, tables

    return change


def _set_table(name, column, value, row=0):
    def change(record, tables):
        table = tables[name].copy()
        label = table.index[row]  # positional: -1 is the last row, never a new one
        table.loc[label, column] = value(record) if callable(value) else value
        tables[name] = contract.conform(name, table)
        return record, tables

    return change


def _at_lockbox(record):
    return _holdout(record)


def _iso_at_lockbox(record):
    return _holdout(record).isoformat()


def _iso_after_lockbox(record):
    return (_holdout(record) + pd.Timedelta(days=1)).isoformat()


def _earliest_fold_date(record):
    return "2021-02-01T00:00:00+00:00"


FOLD_DATE_CASES = [
    _set_table("folds", column, _iso_at_lockbox, row=-1)
    for column in (
        "train_first_date",
        "train_last_date",
        "train_max_target_end_date",
        "knowledge_cutoff",
        "test_start",
        "test_first_date",
        "test_last_date",
        "score_available_at",
    )
] + [
    _set_table("folds", column, _iso_after_lockbox, row=-1)
    for column in ("holdout_start", "test_end_exclusive")
]

LOCKBOX_CASES = [
    (
        "record_mode_holdout",
        _set_record(("spec", "lockbox", "mode"), "holdout"),
        "holdout_run_locked",
    ),
    (
        "evidence_mode_holdout",
        _set_record(("lockbox_evidence", "mode"), "holdout"),
        "holdout_run_locked",
    ),
    ("fold_mode_holdout", _set_table("folds", "mode", "holdout"), "holdout_run_locked"),
    (
        "values_used_true",
        _set_record(("lockbox_evidence", "holdout_values_used"), True),
        "holdout_values_used",
    ),
    (
        "values_used_missing_value",
        _set_record(("lockbox_evidence", "holdout_values_used"), None),
        "holdout_values_used",
    ),
    (
        "prediction_label_end_in_lockbox",
        _set_table("predictions", "target_end_date", _at_lockbox),
        "lockbox_dates_present",
    ),
    ("curve_date_in_lockbox", _set_table("curves", "date", _at_lockbox), "lockbox_dates_present"),
    ("input_date_in_lockbox", _set_table("inputs", "date", _at_lockbox), "lockbox_dates_present"),
    (
        "fold_withholds_labels",
        _set_table("folds", "n_test_label_in_holdout", 5),
        "lockbox_dates_present",
    ),
    (
        "evidence_copy_earlier",
        _set_record(("lockbox_evidence", "holdout_start"), _earliest_fold_date),
        "lockbox_dates_present",
    ),
    *[
        (f"fold_date_{index}", case, "lockbox_dates_present")
        for index, case in enumerate(FOLD_DATE_CASES)
    ],
]


@pytest.mark.parametrize(
    ("change", "reason"), [c[1:] for c in LOCKBOX_CASES], ids=[c[0] for c in LOCKBOX_CASES]
)
def test_runs_that_could_expose_the_lockbox_are_refused(dashboard_root, tmp_path, change, reason):
    _, run = dashboard_root
    record, tables = change(copy.deepcopy(run.record), dict(run.tables))
    directory = artifacts.write_run(_variant(run, record=record, tables=tables), tmp_path)
    artifacts.read_run(directory)  # the files themselves verify
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == reason


def test_a_run_cannot_move_its_own_lockbox_past_the_code_constant(dashboard_root, tmp_path):
    _, run = dashboard_root
    record = copy.deepcopy(run.record)
    for section in (record["spec"]["lockbox"], record["lockbox_evidence"]):
        section["holdout_start"] = "2030-01-01T00:00:00+00:00"
    tables = dict(run.tables)
    inputs = tables["inputs"].copy()
    inputs.loc[0, "date"] = MODEL_HOLDOUT_START
    tables["inputs"] = contract.conform("inputs", inputs)
    directory = artifacts.write_run(_variant(run, record=record, tables=tables), tmp_path)
    with pytest.raises(loader.RunRefused, match="lockbox_dates_present"):
        loader.load_run(directory)


@pytest.mark.parametrize("kind", ["text", "naive"])
def test_loosely_typed_input_dates_cannot_slip_past_the_guard(dashboard_root, tmp_path, kind):
    _, run = dashboard_root
    tables = dict(run.tables)
    inputs = tables["inputs"].copy()
    late = pd.Timestamp("2030-01-04", tz="UTC")
    if kind == "text":
        inputs["date"] = [stamp.isoformat() for stamp in inputs["date"]]
        inputs.loc[0, "date"] = late.isoformat()
    else:
        inputs["date"] = inputs["date"].dt.tz_localize(None)
        inputs.loc[0, "date"] = late.tz_localize(None)
    tables["inputs"] = inputs
    directory = artifacts.write_run(_variant(run, tables=tables), tmp_path)
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == "lockbox_dates_present"


# --- malformed or tampered runs fail with a clear reason, never a crash ---


def _manifest_files_as_list(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = list(manifest["files"])
    path.write_text(json.dumps(manifest))


def _newer_major_version(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_schema_version"] = "2.0"
    path.write_text(json.dumps(manifest))


def _corrupt_table(directory):
    (directory / "checks.parquet").write_bytes(b"not parquet")


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (_manifest_files_as_list, "integrity_check_failed"),
        (_newer_major_version, "integrity_check_failed"),
        (_corrupt_table, "integrity_check_failed"),
    ],
    ids=["files_not_a_mapping", "newer_major_version", "corrupt_table"],
)
def test_tampered_runs_are_refused_with_a_reason(dashboard_root, tmp_path, damage, reason):
    _, run = dashboard_root
    directory = artifacts.write_run(run, tmp_path)
    damage(directory)
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == reason


@pytest.mark.parametrize(
    "change",
    [
        _set_record(("spec", "lockbox", "holdout_start"), "not a date"),
        lambda record, tables: (record["spec"].pop("models") and record, tables),
    ],
    ids=["unparsable_holdout_start", "no_registered_models"],
)
def test_malformed_records_are_refused_not_crashed(dashboard_root, tmp_path, change):
    _, run = dashboard_root
    record, tables = change(copy.deepcopy(run.record), dict(run.tables))
    directory = artifacts.write_run(_variant(run, record=record, tables=tables), tmp_path)
    with pytest.raises(loader.RunRefused) as refused:
        loader.load_run(directory)
    assert refused.value.reason == "malformed_run"


def test_only_files_inside_the_run_directory_are_read(dashboard_root, monkeypatch):
    root, _ = dashboard_root
    (entry,) = loader.list_runs(root)
    opened = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.pd, "read_parquet", spy)
    loader.load_run(entry.path)
    assert opened and all(path.startswith(str(entry.path)) for path in opened)
    assert {p.rsplit("/", 1)[-1] for p in opened} == {f"{t}.parquet" for t in contract.TABLES}


def test_the_root_comes_from_the_launcher_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(loader.ROOT_ENVIRONMENT_VARIABLE, str(tmp_path))
    assert loader.default_root() == tmp_path
    monkeypatch.delenv(loader.ROOT_ENVIRONMENT_VARIABLE)
    assert loader.default_root() == loader.PROJECT_ROOT / "reports"
    assert loader.list_runs(tmp_path / "missing") == []


def test_decision_method_is_read_from_stored_check_names():
    checks = pd.DataFrame({"metric": ["mean_rank_ic_abs_t_fold_block", "hac_lag", None]})
    assert loader.decision_method(checks) == "fold_block_t"
    mixed = pd.DataFrame({"metric": ["x_t_fold_block", "y_t_newey_west"]})
    assert loader.decision_method(mixed) is None
