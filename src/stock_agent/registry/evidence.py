"""
Evidence: the facts a stage rule may read, derived only from a run record.

`derive_evidence(record, model)` is the single way evidence enters the
registry. It reads the experiment-run record (written by the experiment
runner, verified by its manifest) and copies the facts the stage rules need:
data class, lockbox evidence, check outcomes per family, the decision method,
the nulls the model was compared with, the multiple-testing count and the
universe's point-in-time status. The store re-derives it from the run
artifact whenever an entry is assessed, so evidence that does not follow from
a verified run never counts.

Run-record keys read here (the experiment runner's contract):

    run_id
    data.data_class                      one of entries.DATA_CLASSES
    lockbox_evidence.{mode, holdout_start, max_date, max_target_end_date,
                      holdout_values_used}
    checks_summary.families.<severity>.{total, passed, failed, undetermined}
    decision_method
    nulls.{models, permutation_draws}
    multiple_testing.{candidates_in_run, grid_points_in_run,
                      prior_variants_in_scope}
    universe.{point_in_time, survivorship_free}
    models.<name>.{role, variant_id}
"""

from __future__ import annotations

from collections.abc import Mapping

from stock_agent.registry.entries import DATA_CLASSES, RegistryError

EVIDENCE_SCHEMA_VERSION = "1.0"
CHECK_FAMILIES = ("leakage", "instrument")
_COUNTS = ("total", "passed", "failed", "undetermined")


def _get(record: Mapping[str, object], path: str) -> object:
    value: object = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise RegistryError(f"The run record has no {path!r}; evidence cannot be derived.")
        value = value[key]
    return value


def _int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryError(f"Run record {path!r} must be an integer; got {value!r}.")
    return value


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise RegistryError(f"Run record {path!r} must be true or false; got {value!r}.")
    return value


def derive_evidence(record: Mapping[str, object], model: str) -> dict[str, object]:
    """The evidence block of `model`'s registry entry, from its run record."""

    models = _get(record, "models")
    if not isinstance(models, Mapping) or model not in models:
        raise RegistryError(f"Model {model!r} is not part of run {record.get('run_id')!r}.")
    data_class = _get(record, "data.data_class")
    if data_class not in DATA_CLASSES:
        raise RegistryError(f"Run data class {data_class!r} is not one of {DATA_CLASSES}.")
    families = _get(record, "checks_summary.families")
    checks = {}
    for family in CHECK_FAMILIES:
        counts = families.get(family) if isinstance(families, Mapping) else None
        checks[family] = (
            None
            if counts is None
            else {
                key: _int(counts.get(key), f"checks_summary.families.{family}.{key}")
                for key in _COUNTS
            }
        )
    null_models = _get(record, "nulls.models")
    if not isinstance(null_models, (list, tuple)) or not all(
        isinstance(name, str) for name in null_models
    ):
        raise RegistryError("Run record nulls.models must be a list of model names.")
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": str(_get(record, "run_id")),
        "model": model,
        "role": str(_get(models[model], "role")),  # model names may contain dots
        "variant_id": str(_get(models[model], "variant_id")),
        "data_class": data_class,
        "mode": str(_get(record, "lockbox_evidence.mode")),
        "lockbox": {
            "holdout_start": str(_get(record, "lockbox_evidence.holdout_start")),
            "max_date": str(_get(record, "lockbox_evidence.max_date")),
            "max_target_end_date": str(_get(record, "lockbox_evidence.max_target_end_date")),
            "holdout_values_used": _bool(
                _get(record, "lockbox_evidence.holdout_values_used"),
                "lockbox_evidence.holdout_values_used",
            ),
        },
        "checks": checks,
        "decision_method": str(_get(record, "decision_method")),
        "nulls": {
            "models": sorted(null_models),
            "permutation_draws": _int(
                _get(record, "nulls.permutation_draws"), "nulls.permutation_draws"
            ),
        },
        "multiple_testing": {
            key: _int(_get(record, f"multiple_testing.{key}"), f"multiple_testing.{key}")
            for key in ("candidates_in_run", "grid_points_in_run", "prior_variants_in_scope")
        },
        "universe": {
            key: _bool(_get(record, f"universe.{key}"), f"universe.{key}")
            for key in ("point_in_time", "survivorship_free")
        },
    }
