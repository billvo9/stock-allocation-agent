"""
Readers of other tracks' outputs in the agreed formats v1.0: training health,
the registry export's forecast evaluations and backtest summaries (each the
newest `<inputs>/<artifact>/<run_id>.json` by `generated_at`), and the ML
track's model registry, built here through its own write API and read through
its read API. Absent inputs are insufficient_evidence (never healthy),
unreadable ones unavailable, an unapproved transition is degraded, point
forecasts are not applicable, and each rule is exercised on the fixtures with
one field changed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stock_agent.health import backtest, fixtures, forecast, registry, training
from stock_agent.health.model import CONSTRAINT_BREACHED, HealthState, NotApplicable
from stock_agent.registry import Registry, derive_evidence

NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)
APPROVED = "approved"


def inputs(tmp_path: Path, *documents: dict) -> Path:
    for document in documents:
        fixtures.write_input(tmp_path, document)
    return tmp_path


def by_id(checks) -> dict:
    return {check.check_id: check for check in checks}


def approved(document: dict) -> dict:
    document["transition_status"] = APPROVED
    return real(document)


def real(document: dict) -> dict:
    """The fixture as a real (non-synthetic) input: synthetic ones are capped in private scope."""

    document["synthetic"] = False
    return document


# --- the newest file and the common envelope ----------------------------------------


@pytest.mark.parametrize(
    ("reader", "artifact"),
    [
        (training.training_checks, training.ARTIFACT),
        (backtest.backtest_checks, backtest.ARTIFACT),
        (forecast.forecast_checks, forecast.ARTIFACT),
    ],
)
def test_no_directory_or_no_file_is_insufficient_evidence(tmp_path, reader, artifact):
    (tmp_path / artifact).mkdir()
    for path, reason in (
        (None, "input_not_configured"),
        (tmp_path / "absent", "input_missing"),
        (tmp_path / artifact, "input_missing"),
    ):
        for check in reader(path):
            assert (check.state, check.reason, check.value, check.measured_at) == (
                HealthState.INSUFFICIENT_EVIDENCE,
                reason,
                None,
                None,
            )


def test_the_newest_file_by_generated_at_is_read(tmp_path):
    older, newer = approved(fixtures.training_health()), approved(fixtures.training_health())
    older["run_id"], older["status"] = "run-b", "failed"
    newer["run_id"], newer["generated_at"] = "run-a", "2025-01-01T00:00:00.000000Z"
    (check,) = training.training_checks(inputs(tmp_path, older, newer) / training.ARTIFACT)
    assert check.state is HealthState.HEALTHY  # run-a is newer although it sorts first
    assert "run run-a" in check.detail


def test_unreadable_files_beside_a_readable_one_are_degraded(tmp_path):
    root = inputs(tmp_path, approved(fixtures.training_health()))
    (root / training.ARTIFACT / "broken.json").write_text("{not json", encoding="utf-8")
    (check,) = training.training_checks(root / training.ARTIFACT)
    assert (check.state, check.reason) == (HealthState.DEGRADED, "unreadable_inputs")
    assert "unreadable files ignored: broken.json" in check.detail


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(artifact="backtest_summary"),
        lambda d: d.update(schema_version="2.0"),
        lambda d: d.update(generated_at="2024-12-31"),
        lambda d: d.update(code_sha="xyz"),
        lambda d: d.pop("synthetic"),
        lambda d: d.update(status="exploded"),
        lambda d: d.update(seeds=0),
        lambda d: d.pop("scenario_version"),
        lambda d: d.update(transition_status="pending"),
        lambda d: d["checks"][0].update(status="ok"),
        lambda d: d["checks"][0].update(value="high"),
        lambda d: d["checks"].append(dict(d["checks"][0])),
        lambda d: d.pop("checks"),
    ],
)
def test_a_malformed_training_file_is_unavailable(tmp_path, change):
    document = fixtures.training_health()
    change(document)
    directory = tmp_path / training.ARTIFACT
    directory.mkdir()
    (directory / f"{document['run_id']}.json").write_text(json.dumps(document), encoding="utf-8")
    (check,) = training.training_checks(directory)
    assert (check.state, check.reason) == (HealthState.UNAVAILABLE, "unrecognized_format")


def test_a_file_named_differently_from_its_run_id_is_unreadable(tmp_path):
    directory = tmp_path / training.ARTIFACT
    directory.mkdir()
    (directory / "other.json").write_text(json.dumps(fixtures.training_health()), encoding="utf-8")
    (check,) = training.training_checks(directory)
    assert check.state is HealthState.UNAVAILABLE
    assert "differs from run_id" in check.detail


def test_non_json_and_nan_are_unavailable(tmp_path):
    directory = tmp_path / training.ARTIFACT
    directory.mkdir()
    for name, text in (("a.json", "not json"), ("b.json", '{"artifact": NaN}')):
        (directory / name).write_text(text, encoding="utf-8")
    assert training.training_checks(directory)[0].state is HealthState.UNAVAILABLE


def test_extra_producer_fields_are_ignored(tmp_path):
    document = approved(fixtures.training_health())
    document["producer_notes"] = {"anything": [1, 2]}
    (check,) = training.training_checks(inputs(tmp_path, document) / training.ARTIFACT)
    assert check.state is HealthState.HEALTHY


# --- training ----------------------------------------------------------------------


def training_check(tmp_path, change=None, *, transition=APPROVED):
    document = real(fixtures.training_health())
    document["transition_status"] = transition
    if change:
        change(document)
    (check,) = training.training_checks(inputs(tmp_path, document) / training.ARTIFACT)
    return check


def test_a_completed_run_with_every_listed_check_passing_is_healthy(tmp_path):
    check = training_check(tmp_path)
    assert (check.state, check.value, check.unit) == (
        HealthState.HEALTHY,
        8,
        "of 8 listed training checks pass",
    )
    assert "ppo, status completed, 4 seeds, scenario synthetic-v1" in check.detail
    assert "real input" in check.detail
    assert check.measured_at.isoformat() == "2024-12-31T22:00:00+00:00"


def test_an_unapproved_transition_is_a_visible_limitation_not_a_failure(tmp_path):
    check = training_check(tmp_path, transition="engineering_only_unapproved_transition")
    assert (check.state, check.reason) == (HealthState.DEGRADED, "unapproved_transition")
    assert "not yet approved" in check.summary


def _drop(check_id):
    return lambda d: d.update(checks=[c for c in d["checks"] if c["check_id"] != check_id])


@pytest.mark.parametrize(
    ("change", "state", "reason"),
    [
        (lambda d: d.update(status="failed"), HealthState.FAILED, "training_failed"),
        (
            lambda d: d["checks"][1].update(status="fail"),
            HealthState.FAILED,
            "training_check_failed",
        ),
        (
            lambda d: d["checks"][1].update(status="warn"),
            HealthState.DEGRADED,
            "training_check_warning",
        ),
        (
            lambda d: d.update(status="running"),
            HealthState.INSUFFICIENT_EVIDENCE,
            "training_in_progress",
        ),
        (_drop("seed_spread"), HealthState.INSUFFICIENT_EVIDENCE, "training_checks_not_run"),
        (
            lambda d: d["checks"][0].update(status="not_run"),
            HealthState.INSUFFICIENT_EVIDENCE,
            "training_checks_not_run",
        ),
        (
            lambda d: d.update(checks=[]),
            HealthState.INSUFFICIENT_EVIDENCE,
            "training_checks_not_run",
        ),
    ],
)
def test_training_states(tmp_path, change, state, reason):
    check = training_check(tmp_path, change)
    assert (check.state, check.reason) == (state, reason)


def test_a_failure_outranks_an_unapproved_transition(tmp_path):
    check = training_check(
        tmp_path,
        lambda d: d["checks"][0].update(status="fail"),
        transition="engineering_only_unapproved_transition",
    )
    assert (check.state, check.reason) == (HealthState.FAILED, "training_check_failed")


def test_unknown_check_ids_are_shown_and_absent_listed_ones_named(tmp_path):
    def change(document):
        _drop("reward_trend")(document)
        document["checks"].append(
            {
                "check_id": "custom_probe",
                "status": "pass",
                "value": None,
                "unit": None,
                "threshold": None,
                "detail": None,
            }
        )

    check = training_check(tmp_path, change)
    assert "reward_trend: not_run (absent)" in check.detail
    assert "other checks: custom_probe: pass" in check.detail
    assert check.value == 7


# --- forecasts (the registry export) ------------------------------------------------------


def forecast_items(tmp_path, change=None) -> dict:
    document = real(fixtures.registry_export())
    if change:
        change(document)
    items = forecast.forecast_checks(inputs(tmp_path, document) / forecast.ARTIFACT)
    return {item.check_id: item for item in items}


def test_interval_forecasts_are_judged_and_point_forecasts_named(tmp_path):
    items = forecast_items(tmp_path)
    coverage, calibration = items["forecast.coverage"], items["forecast.calibration"]
    assert (coverage.state, coverage.value) == (HealthState.HEALTHY, pytest.approx(0.02))
    assert (calibration.state, calibration.value) == (HealthState.HEALTHY, 1)
    assert "1 point forecasts: not applicable: point forecast" in coverage.detail


def test_a_point_only_export_is_not_applicable_and_has_no_state(tmp_path):
    items = forecast_items(
        tmp_path, lambda d: d.update(forecast_evaluation=d["forecast_evaluation"][1:])
    )
    assert all(isinstance(item, NotApplicable) for item in items.values())
    assert {item.reason for item in items.values()} == {"point_forecast"}


@pytest.mark.parametrize(
    ("change", "check_id", "state", "reason"),
    [
        (
            lambda d: d["forecast_evaluation"][0].update(observed_coverage=0.82),
            "forecast.coverage",
            HealthState.DEGRADED,
            "coverage_off",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(observed_coverage=0.75),
            "forecast.coverage",
            HealthState.FAILED,
            "coverage_off",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(n=49),
            "forecast.coverage",
            HealthState.INSUFFICIENT_EVIDENCE,
            "too_few_observations",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(observed_coverage=None),
            "forecast.coverage",
            HealthState.INSUFFICIENT_EVIDENCE,
            "too_few_observations",
        ),
        (
            lambda d: d.update(forecast_evaluation=[]),
            "forecast.coverage",
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_forecast_evaluation",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(calibration_status="fail"),
            "forecast.calibration",
            HealthState.FAILED,
            "calibration_failed",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(calibration_status="warn"),
            "forecast.calibration",
            HealthState.DEGRADED,
            "calibration_warning",
        ),
        (
            lambda d: d["forecast_evaluation"][0].update(calibration_status="not_run"),
            "forecast.calibration",
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_calibration_result",
        ),
    ],
)
def test_forecast_rules(tmp_path, change, check_id, state, reason):
    item = forecast_items(tmp_path, change)[check_id]
    assert (item.state, item.reason) == (state, reason)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["forecast_evaluation"][1].update(observed_coverage=0.5),  # point with coverage
        lambda d: d["forecast_evaluation"][0].update(forecast_kind="quantile"),
        lambda d: d["forecast_evaluation"][0].update(version=0),
        lambda d: d.pop("experiments"),
    ],
)
def test_a_malformed_export_is_unavailable(tmp_path, change):
    items = forecast_items(tmp_path, change)
    assert {item.state for item in items.values()} == {HealthState.UNAVAILABLE}


# --- registry (the ML track's registry, through its read API) -----------------------


def registry_states(root, artifacts_root=None, *, allow_synthetic=True) -> dict:
    checks = registry.registry_checks(
        root, artifacts_root=artifacts_root, now=lambda: NOW, allow_synthetic=allow_synthetic
    )
    return by_id(checks)


def test_a_registry_of_synthetic_versions_only_is_never_healthy_in_private_scope(tmp_path):
    root = fixtures.build_registry(tmp_path)
    capped = registry_states(root, allow_synthetic=False)
    assert {(c.state, c.reason) for c in capped.values()} == {
        (HealthState.INSUFFICIENT_EVIDENCE, "synthetic_input")
    }
    record = max((root.parent / "runs").iterdir()) / "record.json"
    record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    broken = registry_states(root, allow_synthetic=False)
    assert broken["registry.entries"].state is HealthState.FAILED  # a failure still shows


def test_a_clean_registry_is_healthy(tmp_path):
    checks = registry_states(fixtures.build_registry(tmp_path))
    assert {key: (c.state, c.value) for key, c in checks.items()} == {
        "registry.entries": (HealthState.HEALTHY, 2),
        "registry.stages": (HealthState.HEALTHY, 0),
        "registry.selection_history": (HealthState.HEALTHY, 1),
        "registry.experiments": (HealthState.HEALTHY, 2),
    }
    assert "engineering_only 2" in checks["registry.stages"].detail
    assert "data_class_engineering_only 2" in checks["registry.stages"].detail
    assert checks["registry.entries"].measured_at == NOW


def test_absent_or_empty_registries_are_never_healthy(tmp_path):
    for root, reason in ((None, "input_not_configured"), (tmp_path / "absent", "input_missing")):
        for check in registry_states(root).values():
            assert (check.state, check.reason) == (HealthState.INSUFFICIENT_EVIDENCE, reason)
    empty = tmp_path / "empty"
    empty.mkdir()
    states = {key: c.reason for key, c in registry_states(empty).items()}
    assert states["registry.entries"] == "no_entries"
    assert states["registry.selection_history"] == "no_selection_events"


def test_an_edited_run_record_fails_entries_and_the_active_version(tmp_path):
    root = fixtures.build_registry(tmp_path)
    record = max((root.parent / "runs").iterdir()) / "record.json"
    record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    checks = registry_states(root)
    assert (checks["registry.entries"].state, checks["registry.entries"].reason) == (
        HealthState.FAILED,
        "artifacts_unverified",
    )
    history = checks["registry.selection_history"]
    assert (history.state, history.reason) == (HealthState.FAILED, "active_unverified")
    assert "synthetic.ridge@v2" in history.detail


def test_an_edited_entry_makes_the_reader_refuse_the_registry(tmp_path):
    root = fixtures.build_registry(tmp_path)
    entry = root / "entries" / fixtures.REGISTRY_MODEL / "v0001.json"
    entry.write_text(entry.read_text(encoding="utf-8").replace("0.01", "0.99"), encoding="utf-8")
    checks = registry_states(root)
    assert (checks["registry.entries"].state, checks["registry.entries"].reason) == (
        HealthState.FAILED,
        "registry_refused",
    )
    assert checks["registry.stages"].state is HealthState.UNAVAILABLE


def test_a_registry_without_activations_has_no_history(tmp_path):
    checks = registry_states(fixtures.build_registry(tmp_path, activate=False))
    history = checks["registry.selection_history"]
    assert (history.state, history.reason) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "no_selection_events",
    )


def test_a_prior_variant_count_below_the_registry_is_degraded(tmp_path):
    checks = registry_states(fixtures.build_registry(tmp_path, prior_variants=(0, 0)))
    experiments = checks["registry.experiments"]
    assert (experiments.state, experiments.reason) == (
        HealthState.DEGRADED,
        "experiment_count_below_registry",
    )
    assert "synthetic.ridge@v2 records 0, 1 registered" in experiments.detail


def test_a_run_without_candidates_fails_the_experiment_count(tmp_path):
    artifacts_root = tmp_path / "ml"
    store = Registry(artifacts_root / "registry", artifacts_root, clock=lambda: NOW)
    record = fixtures.synthetic_run_record("20240628T210000Z-norun0001", prior_variants=0)
    record["multiple_testing"]["candidates_in_run"] = 0
    store.register(
        model_id="synthetic.ridge",
        research_scope=fixtures.REGISTRY_SCOPE,
        config={"model": "ridge"},
        training_data={"data_class": "synthetic"},
        metrics={},
        artifacts=[fixtures.write_run_artifact(artifacts_root, record)],
        evidence=derive_evidence(record, "ridge"),
        code={"git_sha": "c" * 40, "git_dirty": False},
    )
    experiments = registry_states(store.root)["registry.experiments"]
    assert (experiments.state, experiments.reason) == (
        HealthState.FAILED,
        "experiment_count_missing",
    )


def test_the_registry_reader_writes_nothing(tmp_path):
    root = fixtures.build_registry(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    registry_states(root)
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


# --- backtest ----------------------------------------------------------------------


def backtest_checks(tmp_path, change=None, *, transition=APPROVED) -> dict:
    document = real(fixtures.backtest_summary())
    document["transition_status"] = transition
    if change:
        change(document)
    return by_id(backtest.backtest_checks(inputs(tmp_path, document) / backtest.ARTIFACT))


def test_measures_within_their_limits_are_healthy_and_labelled_simulated(tmp_path):
    checks = backtest_checks(tmp_path)
    assert {key: check.state for key, check in checks.items()} == {
        key: HealthState.HEALTHY for key in backtest.MEASURES
    }
    concentration = checks["portfolio.concentration"]
    assert (concentration.value, concentration.unit) == (0.34, "fraction of portfolio value")
    assert "policy equal_weight_monthly" in concentration.detail
    assert "Simulated backtest, not a track record." in concentration.detail


def test_an_unapproved_transition_shows_on_every_portfolio_check(tmp_path):
    checks = backtest_checks(tmp_path, transition="engineering_only_unapproved_transition")
    assert {(c.state, c.reason) for c in checks.values()} == {
        (HealthState.DEGRADED, "unapproved_transition")
    }


def test_a_breached_limit_fails_with_constraint_breached(tmp_path):
    checks = backtest_checks(
        tmp_path,
        lambda d: d["measures"].update(costs_bps=75.0),
        transition="engineering_only_unapproved_transition",
    )
    costs = checks["portfolio.costs"]
    assert (costs.state, costs.reason, costs.value) == (
        HealthState.FAILED,
        CONSTRAINT_BREACHED,
        75.0,
    )
    assert checks["portfolio.turnover"].reason == "unapproved_transition"


def test_a_measure_at_its_limit_is_within_it(tmp_path):
    checks = backtest_checks(tmp_path, lambda d: d["measures"].update(max_weight=0.4))
    assert checks["portfolio.concentration"].state is HealthState.HEALTHY


def test_unreported_measures_and_undeclared_limits_are_not_judged(tmp_path):
    def change(document):
        document["measures"]["max_drawdown"] = None
        document["constraints"]["max_turnover"] = None

    checks = backtest_checks(tmp_path, change)
    assert (checks["portfolio.drawdown"].state, checks["portfolio.drawdown"].reason) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "measure_not_reported",
    )
    turnover = checks["portfolio.turnover"]
    assert (turnover.state, turnover.reason, turnover.value) == (
        HealthState.INSUFFICIENT_EVIDENCE,
        "no_limit_declared",
        0.8,
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["measures"].pop("turnover"),
        lambda d: d.pop("policy"),
        lambda d: d.pop("transition_status"),
    ],
)
def test_a_malformed_backtest_is_unavailable(tmp_path, change):
    checks = backtest_checks(tmp_path, change)
    assert {check.state for check in checks.values()} == {HealthState.UNAVAILABLE}


# --- synthetic inputs in the private scope -------------------------------------------


@pytest.mark.parametrize(
    ("reader", "document"),
    [
        (training.training_checks, approved(fixtures.training_health())),
        (forecast.forecast_checks, real(fixtures.registry_export())),
        (backtest.backtest_checks, approved(fixtures.backtest_summary())),
    ],
)
def test_a_synthetic_input_is_capped_unless_synthetic_is_allowed(tmp_path, reader, document):
    document["synthetic"] = True
    directory = inputs(tmp_path, document) / document["artifact"]
    capped = [item for item in reader(directory) if not isinstance(item, NotApplicable)]
    assert {(c.state, c.reason) for c in capped} == {
        (HealthState.INSUFFICIENT_EVIDENCE, "synthetic_input")
    }
    allowed = reader(directory, allow_synthetic=True)
    assert {c.state for c in allowed if not isinstance(c, NotApplicable)} == {HealthState.HEALTHY}


def test_a_synthetic_failure_still_shows_as_failed(tmp_path):
    document = approved(fixtures.backtest_summary())
    document["synthetic"] = True
    document["measures"]["max_weight"] = 0.9
    checks = by_id(backtest.backtest_checks(inputs(tmp_path, document) / backtest.ARTIFACT))
    assert checks["portfolio.concentration"].state is HealthState.FAILED
    assert checks["portfolio.turnover"].reason == "synthetic_input"


def test_a_negative_drawdown_is_refused_not_judged(tmp_path):
    checks = backtest_checks(tmp_path, lambda d: d["measures"].update(max_drawdown=-0.45))
    assert {(c.state, c.reason) for c in checks.values()} == {
        (HealthState.UNAVAILABLE, "unrecognized_format")
    }
