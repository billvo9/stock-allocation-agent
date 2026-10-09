"""
Run record: what was run, on what, and whether the lockbox stayed closed.

Pure: callers inject everything that needs the outside world (git SHA and
dirty state, file hashes, package versions, clock), so identical inputs
give an identical record and tests never shell out.

Identity:
- spec_sha256: SHA-256 of the canonical JSON of the pre-execution spec
  (code, data, label, lockbox, universe, folds, models, inference and
  placebo settings, environment). Same spec -> same hash.
- run_id: "<started_at UTC %Y%m%dT%H%M%SZ>-<spec_sha256[:12]>". Every
  execution, including a repeat or a failure, gets its own id and ledger
  line, so the number of trials is never undercounted. Two runs with the
  same spec but different results_sha256 reveal nondeterminism.
- variant_id: hash of the model-defining fields only (name, role,
  estimator, params, features, output kind), for multiple-testing counts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import pandas as pd

RECORD_SCHEMA_VERSION = "1.0"


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_json(value: object) -> str:
    """Sorted keys, no whitespace, non-finite numbers as null, no NaN tokens."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def run_id(spec_sha256: str, started_at: datetime) -> str:
    stamp = pd.Timestamp(started_at)
    if stamp.tzinfo is None:
        raise ValueError("started_at must be timezone-aware.")
    return f"{stamp.tz_convert(UTC).strftime('%Y%m%dT%H%M%SZ')}-{spec_sha256[:12]}"


def variant_id(model: Mapping[str, object]) -> str:
    fields = ("name", "role", "estimator", "params", "feature_columns", "output_kind")
    return sha256_json({key: model[key] for key in fields})[:16]
