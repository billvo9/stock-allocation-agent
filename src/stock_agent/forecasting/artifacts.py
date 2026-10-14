"""
Versioned out-of-fold forecast artifacts (docs/research/ml_forecast_artifacts.md).

One artifact holds one registered model version's out-of-fold forecasts of
one target at one horizon: exactly one row per (date, asset). Each row says
which outer fold produced it, the training window that fold saw, the model
version, and `available_at`, the earliest instant a consumer may use it.

Out-of-fold means the row's date lies in its fold's test window and the
fold's training labels all ended before that window opened:

    train_start <= train_end < test_start <= date < test_end_exclusive
    train_label_end_max < test_start

Availability. A forecast is derived from its row's features (known at the
row date's knowledge time) and from a model fitted on labels that ended
before test_start, so it is available no earlier than the knowledge time of
its date. The only knowledge rule today is synthetic: an owner decision of
2026-10-13 approved "a synthetic date d becomes known at d 23:59:59.999999Z"
for synthetic fixtures only. Real-data artifacts are refused until a
production availability rule (OD9's signal cutoff) is approved.

Consumers select forecasts with `forecasts_asof(frame, decision_time)`: per
asset, the latest row with available_at <= decision_time, never the latest
row overall.

Layout of an artifact directory (write-once, like T2 run directories):

    forecasts.parquet   the table below
    metadata.json       artifact-level metadata (ForecastMetadata)
    manifest.json       schema version and every file's SHA-256; written last

The registry references an artifact by the SHA-256 of its manifest.json.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_END_COLUMN
from stock_agent.model_diagnostics.record import canonical_json
from stock_agent.model_validation.audit import file_sha256, table_sha256
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, WalkForwardFold

FORECAST_SCHEMA_VERSION = "1.0"
DATA_FILE = "forecasts.parquet"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"

FORECAST_KINDS = ("point", "interval")
TRACKS = ("equity_cross_sectional", "etf_return_risk")

# Knowledge rules: how a row date maps to the instant its inputs are known.
SYNTHETIC_END_OF_UTC_DATE = "synthetic_end_of_utc_date"
KNOWLEDGE_RULES = {
    SYNTHETIC_END_OF_UTC_DATE: (
        "a synthetic session date d becomes known at d 23:59:59.999999Z (owner decision "
        "2026-10-13, synthetic fixtures only; not a production availability rule)"
    ),
}
_KNOWLEDGE_OFFSET = {SYNTHETIC_END_OF_UTC_DATE: pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)}
# Data classes each rule may describe. Real data has no approved rule yet.
_RULE_DATA_CLASSES = {SYNTHETIC_END_OF_UTC_DATE: ("synthetic",)}

DATETIME = "datetime_utc"
STRING = "string"
INT = "int"
FLOAT = "float"
COLUMNS: dict[str, str] = {
    "date": DATETIME,
    "asset": STRING,
    "target": STRING,
    "horizon": INT,
    "prediction": FLOAT,
    "interval_low": FLOAT,
    "interval_high": FLOAT,
    "interval_level": FLOAT,
    "fold_id": INT,
    "train_start": DATETIME,
    "train_end": DATETIME,
    "train_label_end_max": DATETIME,
    "test_start": DATETIME,
    "test_end_exclusive": DATETIME,
    "model_id": STRING,
    "model_version": INT,
    "available_at": DATETIME,
}
_PANDAS = {STRING: "string", INT: "int64", FLOAT: "float64"}


class ForecastArtifactError(ValueError):
    """A forecast table or artifact violates the forecast artifact contract."""


@dataclass(frozen=True)
class ForecastMetadata:
    """Artifact-level facts; every row of the table must agree with them."""

    model_id: str
    model_version: int
    variant_id: str
    run_id: str
    track: str
    target: str
    horizon: int
    label_id: str
    forecast_kind: str
    data_class: str
    knowledge_rule: str
    config_sha256: str
    code: Mapping[str, object] = field(default_factory=dict)
    lockbox_start: str = MODEL_HOLDOUT_START.isoformat()
    forecast_schema_version: str = FORECAST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ForecastArtifactError(f"track must be one of {TRACKS}; got {self.track!r}.")
        if self.forecast_kind not in FORECAST_KINDS:
            raise ForecastArtifactError(f"forecast_kind must be one of {FORECAST_KINDS}.")
        if self.knowledge_rule not in KNOWLEDGE_RULES:
            raise ForecastArtifactError(
                f"knowledge_rule {self.knowledge_rule!r} is not an approved rule; "
                f"known: {sorted(KNOWLEDGE_RULES)}."
            )
        if self.data_class not in _RULE_DATA_CLASSES[self.knowledge_rule]:
            raise ForecastArtifactError(
                f"Knowledge rule {self.knowledge_rule!r} covers only "
                f"{_RULE_DATA_CLASSES[self.knowledge_rule]} data; no production availability "
                f"rule is approved for {self.data_class!r} forecasts (OD9 pending)."
            )
        if self.forecast_schema_version != FORECAST_SCHEMA_VERSION:
            raise ForecastArtifactError(
                f"Forecast schema {self.forecast_schema_version!r} is not readable by contract "
                f"{FORECAST_SCHEMA_VERSION}."
            )
        if pd.Timestamp(self.lockbox_start) > MODEL_HOLDOUT_START:
            raise ForecastArtifactError("lockbox_start may not be later than the owner's lockbox.")

    def to_json(self) -> dict[str, object]:
        return json.loads(canonical_json(asdict(self)))


@dataclass(frozen=True, eq=False)
class ForecastArtifact:
    metadata: ForecastMetadata
    frame: pd.DataFrame
    manifest_sha256: str


def knowledge_time(dates: pd.Series, rule: str) -> pd.Series:
    """The instant each row date's inputs are known under `rule`."""

    if rule not in _KNOWLEDGE_OFFSET:
        raise ForecastArtifactError(f"Unknown knowledge rule {rule!r}.")
    return pd.to_datetime(dates, utc=True) + _KNOWLEDGE_OFFSET[rule]


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.as_unit("ns")


def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with the contract's column order and dtypes (raises on extra/missing columns)."""

    missing = [c for c in COLUMNS if c not in frame.columns]
    extra = [c for c in frame.columns if c not in COLUMNS]
    if missing or extra:
        raise ForecastArtifactError(f"Forecast table: missing {missing}, unexpected {extra}.")
    result = frame.loc[:, list(COLUMNS)].reset_index(drop=True).copy()
    for column, kind in COLUMNS.items():
        result[column] = (
            _utc(result[column]) if kind == DATETIME else result[column].astype(_PANDAS[kind])
        )
    return result.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _check_dtypes(frame: pd.DataFrame) -> list[str]:
    problems = []
    if list(frame.columns) != list(COLUMNS):
        return [f"columns {list(frame.columns)} != {list(COLUMNS)}"]
    for column, kind in COLUMNS.items():
        dtype = frame[column].dtype
        if kind == DATETIME:
            ok = isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"
        elif kind == STRING:
            ok = pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype)
        else:
            ok = str(dtype) == _PANDAS[kind]
        if ok and kind in (DATETIME, STRING) and frame[column].isna().any():
            ok = False
        if not ok:
            problems.append(f"{column}: {dtype} is not {kind} (or has missing values)")
    return problems


def validate_forecasts(frame: pd.DataFrame, metadata: ForecastMetadata) -> None:
    """Raise ForecastArtifactError unless `frame` satisfies every contract invariant."""

    problems = _check_dtypes(frame)
    if problems:
        raise ForecastArtifactError("Forecast table violates the contract: " + "; ".join(problems))
    if frame.empty:
        raise ForecastArtifactError("A forecast artifact holds at least one row.")
    if frame.duplicated(["date", "asset"]).any():
        raise ForecastArtifactError("A forecast artifact holds one row per (date, asset).")
    for column, expected in (
        ("model_id", metadata.model_id),
        ("model_version", metadata.model_version),
        ("target", metadata.target),
        ("horizon", metadata.horizon),
    ):
        if not (frame[column] == expected).all():
            raise ForecastArtifactError(f"Every row's {column} must equal the metadata's.")

    dates = frame["date"]
    if (dates != dates.dt.normalize()).any():
        raise ForecastArtifactError("Forecast dates must be session dates at 00:00 UTC.")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=np.float64)).all():
        raise ForecastArtifactError("Predictions must be finite.")

    if not (frame["train_start"] <= frame["train_end"]).all():
        raise ForecastArtifactError("train_start must not be after train_end.")
    if not (frame["train_end"] < frame["test_start"]).all():
        raise ForecastArtifactError("A fold's training rows must precede its test window.")
    if not (frame["train_label_end_max"] < frame["test_start"]).all():
        raise ForecastArtifactError(
            "Not out-of-fold: a training label ends at or after its fold's test start."
        )
    if not ((frame["test_start"] <= dates) & (dates < frame["test_end_exclusive"])).all():
        raise ForecastArtifactError("Every row's date must lie inside its fold's test window.")
    windows = frame.groupby("fold_id")[["test_start", "test_end_exclusive"]].nunique()
    if (windows > 1).any().any():
        raise ForecastArtifactError("Every row of one fold must carry the same test window.")
    per_fold = frame.groupby("fold_id")[["test_start", "test_end_exclusive"]].first().sort_index()
    if (
        per_fold["test_start"].iloc[1:].to_numpy()
        < per_fold["test_end_exclusive"].iloc[:-1].to_numpy()
    ).any():
        raise ForecastArtifactError("Fold test windows must be sorted by fold_id and disjoint.")

    lockbox = pd.Timestamp(metadata.lockbox_start)
    if (dates >= lockbox).any() or (frame["available_at"] >= lockbox).any():
        raise ForecastArtifactError("Forecast dates and availability must precede the lockbox.")
    known = knowledge_time(dates, metadata.knowledge_rule)
    if (frame["available_at"] < known).any():
        raise ForecastArtifactError(
            "A forecast cannot be available before its inputs are known "
            f"({KNOWLEDGE_RULES[metadata.knowledge_rule]})."
        )

    interval = frame[["interval_low", "interval_high", "interval_level"]].to_numpy(np.float64)
    if metadata.forecast_kind == "interval":
        level = frame["interval_level"]
        if (
            not np.isfinite(interval).all()
            or not (frame["interval_low"] <= frame["interval_high"]).all()
            or not ((level > 0.0) & (level < 1.0)).all()
        ):
            raise ForecastArtifactError(
                "Interval forecasts need finite bounds with low <= high and a level in (0, 1)."
            )
    elif not np.isnan(interval).all():
        raise ForecastArtifactError("Point forecasts must leave the interval columns empty (NaN).")


def forecast_frame(
    predictions: pd.DataFrame,
    labeled: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    metadata: ForecastMetadata,
    intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    The contract table from harness predictions (fold_id, date, symbol,
    prediction) of one model on the folds built from `labeled`.

    Training windows are read from each fold's training rows. `intervals`
    (optional, interval kind only) holds fold_id, date, symbol,
    interval_low, interval_high and interval_level for the same rows.
    """

    by_fold = {fold.fold_id: fold for fold in folds}
    unknown = sorted(set(predictions["fold_id"]) - set(by_fold))
    if unknown:
        raise ForecastArtifactError(f"Predictions name folds {unknown} that were not supplied.")
    windows = []
    for fold_id in sorted(set(predictions["fold_id"])):
        fold = by_fold[fold_id]
        train = labeled.iloc[fold.train_rows]
        windows.append(
            {
                "fold_id": fold_id,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "train_label_end_max": train[TARGET_END_COLUMN].max(),
                "test_start": fold.test_start,
                "test_end_exclusive": fold.test_end,
            }
        )
    table = predictions.loc[:, ["fold_id", "date", "symbol", "prediction"]].merge(
        pd.DataFrame(windows), on="fold_id", how="left", validate="many_to_one"
    )
    if intervals is not None:
        keys = ["fold_id", "date", "symbol"]
        table = table.merge(
            intervals.loc[:, [*keys, "interval_low", "interval_high", "interval_level"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    else:
        for column in ("interval_low", "interval_high", "interval_level"):
            table[column] = math.nan
    table = table.rename(columns={"symbol": "asset"})
    table["target"] = metadata.target
    table["horizon"] = metadata.horizon
    table["model_id"] = metadata.model_id
    table["model_version"] = metadata.model_version
    table["available_at"] = knowledge_time(table["date"], metadata.knowledge_rule)
    result = conform(table)
    validate_forecasts(result, metadata)
    return result


def forecasts_asof(frame: pd.DataFrame, decision_time: object) -> pd.DataFrame:
    """
    The forecasts a decision at `decision_time` may use: per asset, the
    latest row with available_at <= decision_time (an as-of selection, never
    the latest row overall). Assets with no available row are absent.
    """

    stamp = pd.Timestamp(decision_time)
    if stamp.tzinfo is None:
        raise ForecastArtifactError("decision_time must be timezone-aware.")
    usable = frame.loc[frame["available_at"] <= stamp.tz_convert("UTC")]
    latest = usable.sort_values(["asset", "available_at", "date"], kind="mergesort")
    return latest.groupby("asset", sort=True).tail(1).reset_index(drop=True)


def artifact_id(metadata: ForecastMetadata, content_sha256: str) -> str:
    return f"{metadata.model_id}-v{metadata.model_version:04d}-{content_sha256[:12]}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_forecast_artifact(
    frame: pd.DataFrame, metadata: ForecastMetadata, root: str | Path
) -> tuple[Path, str]:
    """
    Write a verified artifact under root/ (write-once) and return its
    directory and the SHA-256 of its manifest (the registry's reference).
    """

    table = conform(frame)
    validate_forecasts(table, metadata)
    content = table_sha256(table)
    directory = Path(root) / artifact_id(metadata, content)
    if directory.exists():
        raise FileExistsError(f"{directory} already exists; artifacts are never overwritten.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.partial"
    staging.mkdir()
    written: list[Path] = []
    try:
        written.append(staging / DATA_FILE)
        table.to_parquet(staging / DATA_FILE, index=False)
        written.append(staging / METADATA_FILE)
        _write_json(staging / METADATA_FILE, metadata.to_json())
        files = {
            DATA_FILE: {
                "rows": len(table),
                "content_sha256": content,
                "file_sha256": file_sha256(staging / DATA_FILE),
            },
            METADATA_FILE: {"file_sha256": file_sha256(staging / METADATA_FILE)},
        }
        written.append(staging / MANIFEST_FILE)
        _write_json(
            staging / MANIFEST_FILE,
            {"forecast_schema_version": FORECAST_SCHEMA_VERSION, "files": files},
        )
        manifest_sha = file_sha256(staging / MANIFEST_FILE)
        os.replace(staging, directory)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return directory, manifest_sha


def read_forecast_artifact(directory: str | Path) -> ForecastArtifact:
    """Read and verify an artifact: manifest hashes, metadata, contract invariants."""

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ForecastArtifactError(f"{directory} is not a forecast artifact directory.")
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ForecastArtifactError(f"{directory} has no manifest; the artifact is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("forecast_schema_version") != FORECAST_SCHEMA_VERSION:
        raise ForecastArtifactError(
            f"Forecast schema {manifest.get('forecast_schema_version')!r} is not readable by "
            f"contract {FORECAST_SCHEMA_VERSION}."
        )
    files = manifest.get("files", {})
    if set(files) != {DATA_FILE, METADATA_FILE}:
        raise ForecastArtifactError(f"{manifest_path} must list exactly the data and metadata.")
    for name, entry in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != entry["file_sha256"]:
            raise ForecastArtifactError(f"{path} does not match its manifest hash.")
    metadata = ForecastMetadata(
        **json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    )
    frame = pd.read_parquet(directory / DATA_FILE)
    validate_forecasts(frame, metadata)
    if table_sha256(frame) != files[DATA_FILE]["content_sha256"]:
        raise ForecastArtifactError(f"{directory / DATA_FILE} content does not match its hash.")
    if directory.name != artifact_id(metadata, files[DATA_FILE]["content_sha256"]):
        raise ForecastArtifactError(f"{directory.name} is not this artifact's id.")
    return ForecastArtifact(metadata, frame, file_sha256(manifest_path))
