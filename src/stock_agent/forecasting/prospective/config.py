"""
The prospective track's registered configuration: strict YAML and its hash.

One config file (configs/prospective/etf_prospective_v1.yaml) pre-registers every
choice before the first forecast: assets, horizons, targets, features, the
training window, the fold geometry, the ridge grid, the band method and
level, the prospective timing conventions and the fixed labels. Unknown keys
are errors. The config hash is the SHA-256 of the canonical JSON of the
parsed config, so formatting and key order do not change it. Changing any
value is a new variant: it registers new model versions, which the registry
counts (forecasting.metrics.prior_trials).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import yaml

from stock_agent.forecasting.config import NAME_PATTERN, ConfigError
from stock_agent.forecasting.etf_data import CONTRACTS_BY_NAME
from stock_agent.model_diagnostics.nested import RIDGE_GRID
from stock_agent.model_diagnostics.record import sha256_json
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

SCHEMA_VERSION = "1.0"
TRACK = "etf_prospective"
SOURCES = ("f1_research_store",)
TARGETS = ("total_log_return", "log_realized_volatility")
TARGET_SHORT = {"total_log_return": "ret", "log_realized_volatility": "vol"}
BAND_METHODS = ("split_conformal_signed_v1",)
NULL_MODELS = ("historical_mean",)
CANDIDATES = ("ridge",)
SKILL_LABEL = "unvalidated, no demonstrated skill"


def _section(cls, raw: object, name: str):
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping.")
    known = {f.name for f in fields(cls)}
    unknown, missing = sorted(set(raw) - known), sorted(known - set(raw))
    if unknown or missing:
        raise ConfigError(f"{name}: unknown keys {unknown}, missing keys {missing}.")
    try:
        return cls(**raw)
    except TypeError as error:
        raise ConfigError(f"{name}: {error}") from None


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return value


def _date(value: object, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(str(value), tz="UTC")
    except (ValueError, TypeError):
        raise ConfigError(f"{name} {value!r} is not a date.") from None
    if stamp != stamp.normalize():
        raise ConfigError(f"{name} must be a date.")
    return stamp


def _names(values: object, name: str, allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ConfigError(f"{name} must be a non-empty list.")
    items = tuple(values)
    if len(set(items)) != len(items) or not all(isinstance(item, str) for item in items):
        raise ConfigError(f"{name} must list distinct strings.")
    if allowed is not None and set(items) - set(allowed):
        raise ConfigError(f"{name} must be a subset of {allowed}; got {list(items)}.")
    return items


@dataclass(frozen=True)
class DataSection:
    source: str
    training_end: str
    min_history_sessions: int

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ConfigError(f"data.source must be one of {SOURCES}.")
        if _date(self.training_end, "data.training_end") >= MODEL_HOLDOUT_START:
            raise ConfigError("data.training_end must be before the 2025-01-01 lockbox.")
        _int(self.min_history_sessions, "data.min_history_sessions", 61)


@dataclass(frozen=True)
class FoldSection:
    start: str
    block_sessions: int
    inner_block_sessions: int
    inner_windows: int
    inner_min_train_sessions: int

    def __post_init__(self) -> None:
        if _date(self.start, "folds.start") >= MODEL_HOLDOUT_START:
            raise ConfigError("folds.start must be before the lockbox.")
        _int(self.block_sessions, "folds.block_sessions", 5)
        _int(self.inner_block_sessions, "folds.inner_block_sessions", 5)
        _int(self.inner_windows, "folds.inner_windows", 2)
        _int(self.inner_min_train_sessions, "folds.inner_min_train_sessions", 20)


@dataclass(frozen=True)
class ModelSection:
    candidate: str
    null_model: str
    ridge_grid: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES:
            raise ConfigError(f"models.candidate must be one of {CANDIDATES}.")
        if self.null_model not in NULL_MODELS:
            raise ConfigError(f"models.null_model must be one of {NULL_MODELS}.")
        grid = tuple(float(c) for c in self.ridge_grid)
        object.__setattr__(self, "ridge_grid", grid)
        if grid != tuple(float(p["penalty"]) for p in RIDGE_GRID):
            raise ConfigError(
                "models.ridge_grid must be T4's pre-registered ridge grid "
                f"{[p['penalty'] for p in RIDGE_GRID]}; another grid is another model."
            )


@dataclass(frozen=True)
class BandSection:
    method: str
    level: float
    lower_quantile: float
    upper_quantile: float
    min_calibration: int

    def __post_init__(self) -> None:
        if self.method not in BAND_METHODS:
            raise ConfigError(f"band.method must be one of {BAND_METHODS}.")
        level, lower, upper = (
            float(self.level),
            float(self.lower_quantile),
            float(self.upper_quantile),
        )
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)
        if not 0.0 < lower < upper < 1.0 or abs((upper - lower) - level) > 1e-12:
            raise ConfigError(
                "band quantiles must satisfy 0 < lower < upper < 1, upper - lower = level."
            )
        _int(self.min_calibration, "band.min_calibration", 20)


@dataclass(frozen=True)
class EvaluationSection:
    """Development inference settings (fold-block t decides; the bootstrap is reported)."""

    bootstrap_reps: int
    seed: int

    def __post_init__(self) -> None:
        _int(self.bootstrap_reps, "evaluation.bootstrap_reps", 9)
        _int(self.seed, "evaluation.seed", 0)


@dataclass(frozen=True)
class ProspectiveSection:
    h_end: str
    lookback_sessions: int
    close_available_local: str
    record_deadline_local: str
    timezone: str

    def __post_init__(self) -> None:
        if _date(self.h_end, "prospective.h_end") < MODEL_HOLDOUT_START:
            raise ConfigError("prospective.h_end must be on or after the lockbox start.")
        _int(self.lookback_sessions, "prospective.lookback_sessions", 61)
        for name in ("close_available_local", "record_deadline_local"):
            value = getattr(self, name)
            try:
                pd.Timestamp(f"2000-01-01 {value}")
            except (ValueError, TypeError):
                raise ConfigError(f"prospective.{name} {value!r} is not a time of day.") from None
        if self.timezone != "America/New_York":
            raise ConfigError("prospective.timezone must be America/New_York (the NYSE's).")


@dataclass(frozen=True)
class ProspectiveConfig:
    schema_version: str
    experiment: str
    track: str
    description: str
    universe_id: str
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    entry_lag: int
    targets: tuple[str, ...]
    features: tuple[str, ...]
    data: DataSection
    folds: FoldSection
    models: ModelSection
    band: BandSection
    evaluation: EvaluationSection
    prospective: ProspectiveSection
    labels: tuple[str, ...]
    registry_prefix: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_json())

    @property
    def training_end(self) -> pd.Timestamp:
        return _date(self.data.training_end, "data.training_end")

    @property
    def h_end(self) -> pd.Timestamp:
        return _date(self.prospective.h_end, "prospective.h_end")


def parse_config(raw: Mapping[str, object]) -> ProspectiveConfig:
    """Validate a config mapping (as loaded from YAML)."""

    if not isinstance(raw, Mapping):
        raise ConfigError("A prospective config must be a mapping.")
    expected = {f.name for f in fields(ProspectiveConfig)}
    unknown, missing = sorted(set(raw) - expected), sorted(expected - set(raw))
    if unknown or missing:
        raise ConfigError(f"Config keys: unknown {unknown}, missing {missing}.")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION!r}.")
    if raw["track"] != TRACK:
        raise ConfigError(f"track must be {TRACK!r}.")
    for name in ("experiment", "universe_id", "registry_prefix"):
        if not isinstance(raw[name], str) or not NAME_PATTERN.fullmatch(raw[name]):
            raise ConfigError(f"{name} {raw[name]!r} must match {NAME_PATTERN.pattern}.")
    assets = _names(raw["assets"], "assets")
    horizons = raw["horizons"]
    if not isinstance(horizons, (list, tuple)) or not horizons:
        raise ConfigError("horizons must be a non-empty list.")
    horizons = tuple(_int(h, "horizons[]", 1) for h in horizons)
    if len(set(horizons)) != len(horizons) or list(horizons) != sorted(horizons):
        raise ConfigError("horizons must be distinct and increasing.")
    labels = _names(raw["labels"], "labels")
    if SKILL_LABEL not in labels:
        raise ConfigError(f"labels must include {SKILL_LABEL!r}.")
    features = _names(raw["features"], "features", tuple(CONTRACTS_BY_NAME))
    return ProspectiveConfig(
        schema_version=SCHEMA_VERSION,
        experiment=raw["experiment"],
        track=TRACK,
        description=str(raw["description"]),
        universe_id=raw["universe_id"],
        assets=assets,
        horizons=horizons,
        entry_lag=_int(raw["entry_lag"], "entry_lag", 1),  # same-close labels are refused
        targets=_names(raw["targets"], "targets", TARGETS),
        features=features,
        data=_section(DataSection, raw["data"], "data"),
        folds=_section(FoldSection, raw["folds"], "folds"),
        models=_section(ModelSection, raw["models"], "models"),
        band=_section(BandSection, raw["band"], "band"),
        evaluation=_section(EvaluationSection, raw["evaluation"], "evaluation"),
        prospective=_section(ProspectiveSection, raw["prospective"], "prospective"),
        labels=labels,
        registry_prefix=raw["registry_prefix"],
    )


def load_config(path: str | Path) -> ProspectiveConfig:
    """Parse a YAML prospective config (safe loader only)."""

    with Path(path).open(encoding="utf-8") as handle:
        return parse_config(yaml.safe_load(handle))
