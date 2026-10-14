"""The registry's health export (agreed input format v1.0 of the Model Health Center)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest
from test_registry import Clock, passing_record, register, synthetic_record

from stock_agent.model_diagnostics.record import canonical_json
from stock_agent.registry import Registry, RegistryError
from stock_agent.registry import export as rx

GENERATED = datetime(2026, 10, 14, 20, 0, 0, 123456, tzinfo=UTC)
SHA = "a" * 40

# The format example agreed with the Model Health Center (its fixture), so a
# document that its reader accepts must pass this validator too.
AGREED_EXAMPLE = {
    "artifact": "model_registry",
    "schema_version": "1.0",
    "generated_at": "2024-06-30T21:00:00.000000Z",
    "code_sha": None,
    "synthetic": True,
    "run_id": "synthetic-registry-export",
    "models": [
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "stage": "engineering_only",
            "stage_basis": ["20240628T210000Z-run0000002"],
            "registered_at": "2024-06-29T21:00:00.000000Z",
        }
    ],
    "selection_events": [],
    "experiments": {"variants_tried": 2, "runs": 2},
    "forecast_evaluation": [
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "horizon": 20,
            "forecast_kind": "interval",
            "interval_level": 0.9,
            "observed_coverage": 0.88,
            "n": 240,
            "calibration_status": "pass",
            "calibration_detail": None,
        },
        {
            "model_id": "synthetic.ridge",
            "version": 2,
            "horizon": 5,
            "forecast_kind": "point",
            "interval_level": None,
            "observed_coverage": None,
            "n": 240,
            "calibration_status": "not_run",
            "calibration_detail": None,
        },
    ],
}


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry", tmp_path, clock=Clock())


def build(registry):
    return rx.build_export(registry, generated_at=GENERATED, code_sha=SHA)


def test_the_agreed_example_passes_the_validator():
    rx.validate_export(AGREED_EXAMPLE)


def test_an_empty_registry_exports_a_valid_empty_document(registry):
    document = build(registry)
    assert document["models"] == [] and document["forecast_evaluation"] == []
    assert document["experiments"] == {"variants_tried": 0, "runs": 0, "distinct_variants": 0}
    assert document["synthetic"] is True
    assert document["generated_at"] == "2026-10-14T20:00:00.123456Z"


def test_models_carry_computed_stages_and_their_evidence_basis(registry, tmp_path):
    first = register(registry, synthetic_record("r1"))
    second = register(registry, passing_record("r2"))
    document = build(registry)
    (model,) = document["models"]  # the latest version of the one model id
    assert (model["model_id"], model["version"]) == (second.model_id, second.version)
    assert model["stage"] == "development_evaluated"
    assert model["stage_basis"] == [
        f"entry:{second.ref}:{second.entry_sha256[:16]}",
        f"artifact:experiment_run:{second.artifacts[0].sha256[:16]}",
        "run:r2",
    ]
    assert model["registered_at"].endswith("Z") and model["blockers"] == ["rule_not_approved"]
    assert document["synthetic"] is False  # a research-panel entry is not synthetic
    (tmp_path / "runs" / "r2" / "record.json").write_text("{}")
    tampered = build(registry)["models"][0]
    assert tampered["stage"] == "engineering_only"
    assert tampered["stage_basis"] == [f"entry:{second.ref}:{second.entry_sha256[:16]}"]
    assert first.ref != second.ref


def test_active_versions_are_exported_beside_the_latest(registry):
    first = register(registry, synthetic_record("r1"))
    register(registry, synthetic_record("r2"))
    registry.activate(first.ref, reason="reference")
    versions = [(m["model_id"], m["version"]) for m in build(registry)["models"]]
    assert versions == [("equity_xs.ridge", 1), ("equity_xs.ridge", 2)]


def test_the_latest_version_of_each_scope_is_exported(registry):
    register(registry, synthetic_record("r1"))  # v1 in scope A
    register(registry, synthetic_record("r2"), research_scope="other:scope")  # v2 in scope B
    register(registry, synthetic_record("r3"))  # v3 in scope A
    versions = [(m["version"], m["research_scope"]) for m in build(registry)["models"]]
    assert versions == [(2, "other:scope"), (3, "equity_xs:synthetic:h20")]


def test_activations_and_rollbacks_are_selection_events(registry):
    first = register(registry, synthetic_record("r1"))
    registry.activate(first.ref, reason="baseline")
    second = register(registry, synthetic_record("r2"))
    registry.activate(second.ref, reason="new version")
    registry.rollback(first.research_scope, first.ref, reason="regression")
    events = build(registry)["selection_events"]
    assert [e["event"] for e in events] == ["activate", "activate", "rollback"]
    assert [e["selected"] for e in events] == [str(first.ref), str(second.ref), str(first.ref)]
    assert [e["candidates"] for e in events] == [1, 2, 2]
    assert len({e["event_id"] for e in events}) == 3


def test_experiment_counts_every_registration_as_a_trial(registry):
    for run in ("r1", "r2", "r3"):
        register(registry, synthetic_record(run))
    experiments = build(registry)["experiments"]
    assert experiments == {"variants_tried": 3, "runs": 3, "distinct_variants": 1}


@pytest.mark.parametrize(
    ("observed", "nominal", "p_value", "status"),
    [
        (0.80, 0.80, 0.9, "pass"),
        (0.78, 0.80, 0.04, "warn"),  # significant but small
        (0.72, 0.80, 0.20, "warn"),  # large but not significant
        (0.72, 0.80, 0.004, "fail"),
        (0.78, 0.80, 0.004, "warn"),  # highly significant, but within the 0.05 band
        (0.82, 0.80, float("nan"), "pass"),
    ],
)
def test_the_calibration_status_rule(observed, nominal, p_value, status):
    assert rx.calibration_status(observed, nominal, p_value) == status


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(generated_at="2026-10-14T20:00:00Z"),
        lambda d: d.update(code_sha="abc"),
        lambda d: d["models"][0].update(stage="validated"),
        lambda d: d["models"][0].update(version=0),
        lambda d: d["forecast_evaluation"][1].update(calibration_status="pass"),
        lambda d: d["forecast_evaluation"][1].update(interval_level=0.8),
        lambda d: d["forecast_evaluation"][0].update(observed_coverage=float("nan")),
        lambda d: d["forecast_evaluation"][0].update(forecast_kind="density"),
        lambda d: d["experiments"].pop("runs"),
    ],
)
def test_malformed_documents_are_refused(change):
    document = copy.deepcopy(AGREED_EXAMPLE)
    change(document)
    with pytest.raises(RegistryError):
        rx.validate_export(document)


def test_exports_are_strict_json_written_once_under_their_run_id(registry, tmp_path):
    register(registry, synthetic_record("r1"))
    document = build(registry)
    path = rx.write_export(document, tmp_path / "health_inputs")
    assert path.parent.name == "model_registry" and path.stem == document["run_id"]

    def refuse(token):
        raise ValueError(token)

    assert json.loads(path.read_text(), parse_constant=refuse) == document
    with pytest.raises(FileExistsError):
        rx.write_export(document, tmp_path / "health_inputs")
    assert json.loads(canonical_json(document)) == document
