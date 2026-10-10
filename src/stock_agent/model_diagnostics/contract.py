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
checked in memory before writing and again on reading; a reader refuses an
unknown MAJOR version. Pure: no file I/O here (see artifacts.py).

Scopes: "evaluation" = out-of-sample test rows (metrics, curves, drift);
"training" = one fold's training rows (feature diagnostics). Every row of
metrics and curves is evaluation scope; nothing in a run feeds an
evaluation table back into fitting.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_SCHEMA_VERSION = "1.0"

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


def _utc(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, utc=True)
    return converted.dt.as_unit("ns")


def conform(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of `frame` with the contract's column order and dtypes."""

    if name in REQUIRED_COLUMNS:
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in frame.columns]
        if missing:
            raise ValueError(f"Table {name!r} is missing columns: {missing}")
        result = frame.reset_index(drop=True).copy()
        for column in result.columns:
            if isinstance(result[column].dtype, pd.DatetimeTZDtype):
                result[column] = _utc(result[column])
        return result
    schema = SCHEMAS[name]
    missing = [c for c in schema if c not in frame.columns]
    extra = [c for c in frame.columns if c not in schema]
    if missing or extra:
        raise ValueError(f"Table {name!r}: missing columns {missing}, unexpected columns {extra}")
    result = frame.loc[:, list(schema)].reset_index(drop=True).copy()
    for column, kind in schema.items():
        if kind in (DATETIME, NULLABLE_DATETIME):
            result[column] = _utc(result[column])
        else:
            result[column] = result[column].astype(_PANDAS_DTYPE[kind])
    validate(name, result)
    return result


def validate(name: str, frame: pd.DataFrame) -> None:
    """Raise unless `frame` matches the contract for table `name` exactly."""

    if name in REQUIRED_COLUMNS:
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in frame.columns]
        if missing:
            raise ValueError(f"Table {name!r} is missing columns: {missing}")
        return
    if name not in SCHEMAS:
        raise ValueError(f"Unknown table {name!r}.")
    schema = SCHEMAS[name]
    if list(frame.columns) != list(schema):
        raise ValueError(f"Table {name!r} columns {list(frame.columns)} != {list(schema)}")
    problems = []
    for column, kind in schema.items():
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
    """Refuse outputs written under an unknown major schema version."""

    major = str(version).split(".")[0]
    if major != OUTPUT_SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"Output schema version {version} is not readable by contract "
            f"{OUTPUT_SCHEMA_VERSION} (major version differs)."
        )
