"""
The frozen, registered model's record, `stock_lab_frozen_model` (schema 1.0),
and its checks (standard library only, so the store and any reader can use
it without loading the research side; `registry` builds and loads it).

A frozen model is everything a forecast needs, fixed before any real
forecast is made and never changed afterwards: a different estimator,
half-life, parameter, feature version, band, state rule or environment is a
new model with a new id, never an edit of the old one. Each record carries:

    model_id            sha256 of its canonical content (without the id); a
                        forecast's `model_version` is its first 16 characters
    created_at, data_class
    estimator           the frozen forecaster (B1 unless a challenger was
                        selected under the owner-confirmed decision rule, A7)
    benchmark           "B1": the comparator recorded beside every forecast
    selection           what the evaluation selected, under which rule, and
                        why this estimator was frozen
    state_rule          how the model's running state uses the inference
                        window (registry.STATE_RULES; owner decision A2, item 4)
    window_sessions     the inference window it reads: 252 sessions (decision A2)
    window_tail_weight  the share of B1's recency weight that falls beyond that
                        window over a long history, lambda^252 with lambda =
                        2^(-1 / half-life): what the window leaves out
    state_check         development only, reported, never used to select: B1
                        from trailing windows against B1 from the full history
                        on the last development sessions (registry.state_check)
    parameters_sha256   sha256 of what a forecast is computed from (estimator,
                        state rule, window, settings, half-life and
                        parameters): two freezes of the same model share it,
                        though their ids differ
    evaluation_run_id, evaluation_sha256
                        the stored development evaluation it comes from
    config, half_life, feature_version, outcome_end
    stocks              the stocks it covers; any other is refused
    long_run            B0, the long-run mix of each stock (daily and
                        five-session categories), at the end of development
    tilt, errors, m1    B2's run-length tilt, B3's error distributions and
                        M1's coefficients, when the estimator needs them
                        (None otherwise)
    reversal            per stock, what followed rising and falling runs of
                        1 to 7 sessions in development history (descriptive)
    drift_reference     decile edges and shares of a few inputs in
                        development, for Model Health's drift check
    data                the evaluation panel's hash and each stock's history
                        (sessions, rows, content hash, and the stored batches
                        when the caller names them)
    environment         Python's minor version and the numpy, pandas and
                        scikit-learn versions; loading refuses any other
    code                the commit that froze it and whether the checkout had
                        uncommitted changes to tracked files, when Git can say
    notes               plain sentences

Checks (`check`), so the store refuses a record that `registry.freeze` would
never write, even one whose id was recomputed after an edit:
    - the fields, the schema and the id against the content;
    - the estimator and the B1 benchmark; the selection names the frozen
      estimator, and a challenger only under a confirmed decision rule
      (CONFIRMED_DECISION_RULES: owner decision A7, confirmed 2026-10-19);
    - the data class: a real model covers real tickers only, uses the
      owner's state rule (REAL_STATE_RULES), its outcomes end on 2024-12-31
      and no history goes past it; a synthetic one covers demo identifiers
      only;
    - the state rule, the 252-session window and the parameters' hash;
    - created_at in the stored form (UTC, microseconds, Z).
Strict JSON: no NaN, no infinity (a missing number is null).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

SCHEMA = "stock_lab_frozen_model"
SCHEMA_VERSION = "1.0"
ESTIMATORS = ("B0", "B1", "B2", "B3", "M1")
BENCHMARK = "B1"
FROZEN_LONG_RUN = "frozen_long_run"
WINDOW_ONLY = "window_only"
STATE_RULES = (FROZEN_LONG_RUN, WINDOW_ONLY)
# The owner's decision A2, item 4 (2026-10-19, docs/PROJECT_STATE.md): a real model
# uses frozen_long_run. window_only stays for synthetic checks of the machinery.
REAL_STATE_RULES = (FROZEN_LONG_RUN,)
WINDOW_SESSIONS = 252  # decision A2's trailing window (equal to inference.schema's; tested)
DEVELOPMENT_END = "2024-12-31"  # contracts.DEVELOPMENT_END (equality tested)
# Decision A7, confirmed by the owner on 2026-10-19 (docs/PROJECT_STATE.md): the
# fold-block t on the yearly fold means decides whether a challenger beats B1, one-sided,
# with Holm across the challengers at a 2.5% family level and simplest-first steps at
# 2.5%. Evaluations record it (evaluation.py, schema 1.3); a model may register a
# challenger only under a rule named here.
A7_DECISION_RULE = "fold_block_t_holm_one_sided_2.5pct_v1"
CONFIRMED_DECISION_RULES: frozenset[str] = frozenset({A7_DECISION_RULE})
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
PARAMETER_FIELDS = (
    "estimator",
    "benchmark",
    "state_rule",
    "window_sessions",
    "config",
    "half_life",
    "feature_version",
    "outcome_end",
    "stocks",
    "long_run",
    "tilt",
    "errors",
    "m1",
)
FIELDS = (
    "schema",
    "schema_version",
    "model_id",
    "created_at",
    "data_class",
    "estimator",
    "benchmark",
    "selection",
    "state_rule",
    "window_sessions",
    "window_tail_weight",
    "state_check",
    "parameters_sha256",
    "evaluation_run_id",
    "evaluation_sha256",
    "config",
    "half_life",
    "feature_version",
    "outcome_end",
    "stocks",
    "long_run",
    "tilt",
    "errors",
    "m1",
    "reversal",
    "drift_reference",
    "data",
    "environment",
    "code",
    "notes",
)


class RegistryRefused(ValueError):
    """A refusal with a machine-readable `reason` and plain words."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def plain(value):
    """JSON-ready values: a non-finite number becomes null, numpy scalars and tuples plain."""

    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        value = value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def canonical(value: object) -> str:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(record: Mapping[str, object]) -> str:
    """sha256 of a record's strict canonical JSON (an evaluation's stored content hash)."""

    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def model_id(record: Mapping[str, object]) -> str:
    return content_sha256({key: value for key, value in record.items() if key != "model_id"})


def parameters_sha256(record: Mapping[str, object]) -> str:
    """sha256 of what a forecast is computed from (PARAMETER_FIELDS)."""

    return content_sha256({key: record[key] for key in PARAMETER_FIELDS})


def _refuse(reason: str, detail: str) -> None:
    raise RegistryRefused(reason, detail)


def _check_selection(record: Mapping[str, object]) -> None:
    selection = record["selection"]
    if not isinstance(selection, Mapping) or selection.get("frozen") != record["estimator"]:
        _refuse("invalid_record", "the selection does not name the frozen estimator")
    if record["estimator"] == BENCHMARK:
        return
    rule = selection.get("decision_rule")
    if rule is None or rule not in CONFIRMED_DECISION_RULES:
        _refuse("a7_unresolved", "a challenger is registered only under a confirmed decision rule")
    if selection.get("evaluation_selected") != record["estimator"]:
        _refuse("not_the_selection", "a challenger must be the evaluation's selection")


def _check_data(record: Mapping[str, object]) -> None:
    stocks = record["stocks"]
    if not isinstance(stocks, list) or not stocks or stocks != sorted(set(stocks)):
        _refuse("invalid_record", "the stocks must be a sorted list without repeats")
    demo = ["_" in str(stock) for stock in stocks]
    histories = (
        (record["data"] or {}).get("histories") if isinstance(record["data"], Mapping) else None
    )
    if not isinstance(histories, Mapping) or sorted(histories) != stocks:
        _refuse("invalid_record", "each stock's history must be described")
    if record["data_class"] == "synthetic":
        if not all(demo):
            _refuse("data_class_mismatch", "a synthetic model covers demo identifiers only")
        return
    if any(demo):
        _refuse("data_class_mismatch", "a real model never covers a demo identifier")
    if record["state_rule"] not in REAL_STATE_RULES:
        _refuse("state_rule_not_approved", f"a real model uses {', '.join(REAL_STATE_RULES)}")
    if record["outcome_end"] != DEVELOPMENT_END:
        _refuse("outcome_cutoff", f"a real model's outcomes end on {DEVELOPMENT_END}")
    late = [s for s in stocks if not str(histories[s].get("last_session", "9")) <= DEVELOPMENT_END]
    if late:
        _refuse("protected_period", f"{', '.join(late)}: a history goes past {DEVELOPMENT_END}")


def check(record: Mapping[str, object]) -> Mapping[str, object]:
    """The record, after the checks in the module docstring (RegistryRefused if not)."""

    if not isinstance(record, Mapping) or tuple(sorted(record)) != tuple(sorted(FIELDS)):
        _refuse("invalid_record", "unexpected or missing fields")
    if record["schema"] != SCHEMA or record["schema_version"] != SCHEMA_VERSION:
        _refuse("invalid_record", "unknown schema")
    if record["model_id"] != model_id(record):
        _refuse("model_id_mismatch", "the model id does not match the content")
    if record["estimator"] not in ESTIMATORS or record["benchmark"] != BENCHMARK:
        _refuse("invalid_record", "unknown estimator or benchmark")
    if record["data_class"] not in ("real", "synthetic"):
        _refuse("invalid_record", "data_class must be real or synthetic")
    if record["state_rule"] not in STATE_RULES:
        _refuse("unknown_state_rule", f"{record['state_rule']!r}")
    if record["window_sessions"] != WINDOW_SESSIONS:
        _refuse("invalid_record", f"the inference window is {WINDOW_SESSIONS} sessions")
    if record["parameters_sha256"] != parameters_sha256(record):
        _refuse("invalid_record", "the parameters' hash does not match them")
    if not isinstance(record["created_at"], str) or not STAMP.fullmatch(record["created_at"]):
        _refuse("invalid_record", "created_at must be UTC with microseconds and a Z")
    _check_selection(record)
    _check_data(record)
    return record


def version(record: Mapping[str, object]) -> str:
    """The id's first 16 characters: the `model_version` a forecast of this model carries."""

    return str(record["model_id"])[:16]
