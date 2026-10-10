"""
The saved-output contract read by later tickets (T3 dashboard) without
re-running models or reading raw data.

One run directory holds:

    record.json         run record (provenance, spec, lockbox evidence)
    manifest.json       output_schema_version, per-table row counts and
                        content hashes; written last (completion marker)
    <table>.parquet     one file per table below

Tables are tidy and typed. A new diagnostic adds rows (a new `metric`,
`curve` or `statistic` value), not columns. Column sets and kinds are
checked in memory before writing and again on reading, against the schema
of the version the run was written with. A reader accepts every minor
version up to its own and refuses newer minors and other majors. Pure: no
file I/O here (see artifacts.py).

Versions:
    1.0  T2 (PR #43).
    1.1  curves record their interval method, level and HAC lag
         (ci_method, ci_level, hac_lag). The run record adds an explicit
         decision_method, machine-readable reference predictors (the
         unpurged canary reference, never eligible as a candidate),
         eligible_as_candidate on every model, and per-family check
         counts. All additions are additive; 1.0 runs remain readable.

Scopes: "evaluation" = out-of-sample test rows (metrics, curves, drift);
"training" = one fold's training rows (feature diagnostics). Every row of
metrics and curves is evaluation scope; nothing in a run feeds an
evaluation table back into fitting.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_SCHEMA_VERSION = "1.1"

# Roles. Only "candidate" models may ever be read as strategies.
CANDIDATE_ROLE = "candidate"
UNSAFE_REFERENCE_ROLE = "canary_unsafe_reference"
NON_CANDIDATE_ROLES = ("null", "control", "canary", UNSAFE_REFERENCE_ROLE)

STRING = "string"
NULLABLE_STRING = "nullable_string"
INT = "int"
NULLABLE_INT = "nullable_int"
FLOAT = "float"
NULLABLE_BOOL = "nullable_bool"
DATETIME = "datetime_utc"
NULLABLE_DATETIME = "nullable_datetime_utc"

_PANDAS_DTYPE = {
    STRING: "string",
    NULLABLE_STRING: "string",
    INT: "int64",
    NULLABLE_INT: "Int64",
    FLOAT: "float64",
    NULLABLE_BOOL: "boolean",
}

SCHEMAS: dict[str, dict[str, str]] = {
    "predictions": {
        "model": STRING,
        "role": STRING,
        "fold_id": INT,
        "date": DATETIME,
        "symbol": STRING,
        "prediction": FLOAT,
        "target_start_date": DATETIME,
        "target_end_date": DATETIME,
        "target_return": FLOAT,
    },
    "metrics": {
        "model": STRING,
        "role": STRING,
        "scope": STRING,
        "fold_id": NULLABLE_INT,
        "metric": STRING,
        "inference_method": STRING,
        "estimate": FLOAT,
        "se": FLOAT,
        "ci_low": FLOAT,
        "ci_high": FLOAT,
        "ci_level": FLOAT,
        "ci_method": NULLABLE_STRING,
        "p_value": FLOAT,
        "null_value": FLOAT,
        "n": NULLABLE_INT,
        "n_eff": FLOAT,
        "hac_lag": NULLABLE_INT,
        "block_length": NULLABLE_INT,
        "bootstrap_reps": NULLABLE_INT,
        "seed": NULLABLE_INT,
        "df": FLOAT,
        "status": STRING,
        "reason": NULLABLE_STRING,
    },
    "curves": {
        "model": STRING,
        "role": STRING,
        "scope": STRING,
        "curve": STRING,
        "group": NULLABLE_STRING,
        "fold_id": NULLABLE_INT,
        "date": NULLABLE_DATETIME,
        "x": FLOAT,
        "x_label": NULLABLE_STRING,
        "value": FLOAT,
        "ci_low": FLOAT,
        "ci_high": FLOAT,
        "ci_method": NULLABLE_STRING,
        "ci_level": FLOAT,
        "hac_lag": NULLABLE_INT,
        "n": NULLABLE_INT,
        "status": STRING,
        "reason": NULLABLE_STRING,
    },
    "null_draws": {
        "model": STRING,
        "null_kind": STRING,
        "draw": INT,
        "seed": NULLABLE_INT,
        "detail": NULLABLE_STRING,
        "metric": STRING,
        "value": FLOAT,
    },
    "feature_stats": {
        "scope": STRING,
        "fold_id": INT,
        "feature": STRING,
        "statistic": STRING,
        "basis": STRING,
        "value": FLOAT,
        "status": STRING,
        "reason": NULLABLE_STRING,
    },
    "feature_pairs": {
        "scope": STRING,
        "fold_id": INT,
        "feature_a": STRING,
        "feature_b": STRING,
        "statistic": STRING,
        "value": FLOAT,
        "n": INT,
        "status": STRING,
        "reason": NULLABLE_STRING,
    },
    "checks": {
        "check": STRING,
        "model": NULLABLE_STRING,
        "severity": STRING,
        "metric": NULLABLE_STRING,
        "observed": FLOAT,
        "threshold": FLOAT,
        "comparison": STRING,
        "passed": NULLABLE_BOOL,
        "detail": NULLABLE_STRING,
    },
}

# Tables whose columns depend on the run (feature lists, availability columns).
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "folds": ("fold_id", "mode", "test_start", "test_end_exclusive", "n_train", "n_test"),
    "inputs": ("date", "symbol", "target_start_date", "target_end_date", "target_return"),
}

TABLES = (*SCHEMAS, *REQUIRED_COLUMNS)

# Columns added after 1.0: table -> column -> version that added it.
ADDED_IN = {"curves": {"ci_method": "1.1", "ci_level": "1.1", "hac_lag": "1.1"}}


def _minor(version: str) -> tuple[int, int]:
    major, _, minor = str(version).partition(".")
    return int(major), int(minor or 0)


def schema(name: str, version: str = OUTPUT_SCHEMA_VERSION) -> dict[str, str]:
    """Columns and kinds of table `name` as written under `version`."""

    added = ADDED_IN.get(name, {})
    return {
        column: kind
        for column, kind in SCHEMAS[name].items()
        if _minor(added.get(column, "1.0")) <= _minor(version)
    }


def predates(version: str, other: str) -> bool:
    """True when `version` is an earlier minor of the same major than `other`."""

    return _minor(version) < _minor(other)


def added_after(name: str, version: str) -> tuple[str, ...]:
    """Columns of `name` that did not exist under `version` (legacy gaps)."""

    return tuple(column for column in SCHEMAS.get(name, {}) if column not in schema(name, version))


def _utc(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, utc=True)
    return converted.dt.as_unit("ns")


def conform(name: str, frame: pd.DataFrame, version: str = OUTPUT_SCHEMA_VERSION) -> pd.DataFrame:
    """Copy of `frame` with the column order and dtypes of `version`'s contract."""

    if name in REQUIRED_COLUMNS:
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in frame.columns]
        if missing:
            raise ValueError(f"Table {name!r} is missing columns: {missing}")
        result = frame.reset_index(drop=True).copy()
        for column in result.columns:
            if isinstance(result[column].dtype, pd.DatetimeTZDtype):
                result[column] = _utc(result[column])
        return result
    columns = schema(name, version)
    missing = [c for c in columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    if missing or extra:
        raise ValueError(f"Table {name!r}: missing columns {missing}, unexpected columns {extra}")
    result = frame.loc[:, list(columns)].reset_index(drop=True).copy()
    for column, kind in columns.items():
        if kind in (DATETIME, NULLABLE_DATETIME):
            result[column] = _utc(result[column])
        else:
            result[column] = result[column].astype(_PANDAS_DTYPE[kind])
    validate(name, result, version)
    return result


def validate(name: str, frame: pd.DataFrame, version: str = OUTPUT_SCHEMA_VERSION) -> None:
    """Raise unless `frame` matches table `name` exactly as written under `version`."""

    if name in REQUIRED_COLUMNS:
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in frame.columns]
        if missing:
            raise ValueError(f"Table {name!r} is missing columns: {missing}")
        return
    if name not in SCHEMAS:
        raise ValueError(f"Unknown table {name!r}.")
    columns = schema(name, version)
    if list(frame.columns) != list(columns):
        raise ValueError(
            f"Table {name!r} (schema {version}) columns {list(frame.columns)} != {list(columns)}"
        )
    problems = []
    for column, kind in columns.items():
        values = frame[column]
        dtype = values.dtype
        if kind in (DATETIME, NULLABLE_DATETIME):
            ok = isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"
        elif kind in (STRING, NULLABLE_STRING):
            ok = pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype)
        elif kind == INT:
            ok = str(dtype) == "int64"
        elif kind == NULLABLE_INT:
            ok = str(dtype) == "Int64"
        elif kind == FLOAT:
            ok = str(dtype) == "float64"
        else:
            ok = str(dtype) == "boolean"
        if ok and kind in (STRING, DATETIME) and values.isna().any():
            ok = False
        if not ok:
            problems.append(f"{column}: {dtype} is not {kind}")
    if problems:
        raise ValueError(f"Table {name!r} violates the contract: " + "; ".join(problems))


def check_version(version: str) -> None:
    """Refuse another major version, or a newer minor this contract does not know."""

    try:
        written = _minor(version)
    except ValueError:
        raise ValueError(f"Output schema version {version!r} is not a version.") from None
    current = _minor(OUTPUT_SCHEMA_VERSION)
    if written[0] != current[0]:
        raise ValueError(
            f"Output schema version {version} is not readable by contract "
            f"{OUTPUT_SCHEMA_VERSION} (major version differs)."
        )
    if written[1] > current[1]:
        raise ValueError(
            f"Output schema version {version} is newer than contract {OUTPUT_SCHEMA_VERSION} "
            "(written by a newer minor version); update the reader."
        )
