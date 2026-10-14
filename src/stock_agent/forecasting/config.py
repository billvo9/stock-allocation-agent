"""
Experiment configuration: strict YAML parsing and the config hash.

A config file (configs/ml/*.yaml) fully defines an experiment: data source
and its seed, universe rule, feature families, label, fold geometry,
models, evaluation settings and registry naming. Unknown keys are errors,
so a typo cannot silently fall back to a default. The config hash is the
SHA-256 of the canonical JSON of the parsed, normalized config, so key
order and YAML formatting do not change it.

Only synthetic data sources exist. Real-data experiments need a registered
research panel (RES-1), a production availability rule (OD9) and the G1
record of every real-panel scoring (OD12); a config naming any other
source is refused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import pandas as pd
import yaml

from stock_agent.model_diagnostics.record import sha256_json

CONFIG_SCHEMA_VERSION = "1.0"
TRACKS = ("equity_cross_sectional", "etf_return_risk")
DATA_SOURCES = {
    "synthetic_equity_panel": "equity_cross_sectional",
    "synthetic_etf_panel": "etf_return_risk",
}
LINEAR_CANDIDATES = ("ridge", "lasso", "elastic_net")
CHALLENGERS = ("hist_gradient_boosting",)
BENCHMARKS = (None, "equal_weight")
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,59}")


class ConfigError(ValueError):
    """An experiment config is malformed or names something not allowed."""


def _section(cls, raw: object, name: str):
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping.")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{name}: unknown keys {unknown}; allowed {sorted(known)}.")
    try:
        return cls(**raw)
    except TypeError as error:
        raise ConfigError(f"{name}: {error}") from None


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ConfigError(f"{name} {value!r} must match {NAME_PATTERN.pattern}.")
    return value


def _int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return value


@dataclass(frozen=True)
class DataConfig:
    source: str = ""
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in DATA_SOURCES:
            raise ConfigError(
                f"data.source {self.source!r} is not available. Only synthetic sources exist "
                f"({sorted(DATA_SOURCES)}); real-data experiments need a registered research "
                "panel (RES-1), an approved availability rule (OD9) and the G1 scoring record "
                "(OD12)."
            )
        if not isinstance(self.params, Mapping):
            raise ConfigError("data.params must be a mapping.")


@dataclass(frozen=True)
class UniverseConfig:
    id: str = ""
    min_history_sessions: int = 60

    def __post_init__(self) -> None:
        _name(self.id, "universe.id")
        _int(self.min_history_sessions, "universe.min_history_sessions", 1)


@dataclass(frozen=True)
class FeatureConfig:
    families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", tuple(self.families))
        if not self.families or len(set(self.families)) != len(self.families):
            raise ConfigError("features.families must be a non-empty list without duplicates.")


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 20
    entry_lag: int = 1
    benchmark: str | None = None

    def __post_init__(self) -> None:
        _int(self.horizon, "label.horizon", 1)
        _int(self.entry_lag, "label.entry_lag", 1)  # same-close labels are refused
        if self.benchmark not in BENCHMARKS:
            raise ConfigError(f"label.benchmark must be one of {BENCHMARKS}.")


@dataclass(frozen=True)
class FoldConfig:
    start: str = ""
    block_sessions: int = 63
    inner_block_sessions: int = 63
    inner_windows: int = 4
    inner_min_train_sessions: int = 252

    def __post_init__(self) -> None:
        try:
            stamp = pd.Timestamp(self.start)
        except (ValueError, TypeError):
            raise ConfigError(f"folds.start {self.start!r} is not a date.") from None
        if stamp != stamp.normalize():
            raise ConfigError("folds.start must be a date (sessions are dates).")
        _int(self.block_sessions, "folds.block_sessions", 5)
        _int(self.inner_block_sessions, "folds.inner_block_sessions", 5)
        _int(self.inner_windows, "folds.inner_windows", 2)
        _int(self.inner_min_train_sessions, "folds.inner_min_train_sessions", 20)


@dataclass(frozen=True)
class ModelConfig:
    candidates: tuple[str, ...] = ("ridge",)
    include_ols: bool = True
    challengers: tuple[str, ...] = ()
    increment_reference: str = "ridge"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "challengers", tuple(self.challengers))
        if not self.candidates or set(self.candidates) - set(LINEAR_CANDIDATES):
            raise ConfigError(
                f"models.candidates must be a non-empty subset of {LINEAR_CANDIDATES}."
            )
        if set(self.challengers) - set(CHALLENGERS):
            raise ConfigError(f"models.challengers must be a subset of {CHALLENGERS}.")
        if self.increment_reference not in self.candidates:
            raise ConfigError("models.increment_reference must be one of the linear candidates.")


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 0
    permutation_draws: int = 19
    permutation_models: tuple[str, ...] = ()
    bootstrap_reps: int = 199
    calibration_reps: int = 2000
    calibration_bootstrap_reps: int = 199
    stale_lags: tuple[int, ...] = ()
    interval_levels: tuple[float, ...] = (0.5, 0.8, 0.9)

    def __post_init__(self) -> None:
        _int(self.seed, "evaluation.seed")
        _int(self.permutation_draws, "evaluation.permutation_draws")
        _int(self.bootstrap_reps, "evaluation.bootstrap_reps", 9)
        _int(self.calibration_reps, "evaluation.calibration_reps", 10)
        _int(self.calibration_bootstrap_reps, "evaluation.calibration_bootstrap_reps", 9)
        object.__setattr__(self, "permutation_models", tuple(self.permutation_models))
        object.__setattr__(self, "stale_lags", tuple(self.stale_lags))
        object.__setattr__(self, "interval_levels", tuple(float(x) for x in self.interval_levels))
        if any(not 0.0 < level < 1.0 for level in self.interval_levels):
            raise ConfigError("evaluation.interval_levels must lie strictly between 0 and 1.")


@dataclass(frozen=True)
class RegistryConfig:
    """Every run registers its candidates, so every variant tried is counted."""

    model_prefix: str = ""

    def __post_init__(self) -> None:
        _name(self.model_prefix, "registry.model_prefix")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    experiment: str
    track: str
    description: str
    data: DataConfig
    universe: UniverseConfig
    features: FeatureConfig
    label: LabelConfig
    folds: FoldConfig
    models: ModelConfig
    evaluation: EvaluationConfig
    registry: RegistryConfig

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_json())

    @property
    def data_class(self) -> str:
        return "synthetic"  # every available source is synthetic (DATA_SOURCES)


TOP_LEVEL = {
    "schema_version",
    "experiment",
    "track",
    "description",
    "data",
    "universe",
    "features",
    "label",
    "folds",
    "models",
    "evaluation",
    "registry",
}


def parse_config(raw: Mapping[str, object]) -> ExperimentConfig:
    """Validate a config mapping (as loaded from YAML)."""

    if not isinstance(raw, Mapping):
        raise ConfigError("An experiment config must be a mapping.")
    unknown, missing = sorted(set(raw) - TOP_LEVEL), sorted(TOP_LEVEL - set(raw))
    if unknown or missing:
        raise ConfigError(f"Config keys: unknown {unknown}, missing {missing}.")
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}.")
    experiment = _name(raw["experiment"], "experiment")
    if raw["track"] not in TRACKS:
        raise ConfigError(f"track must be one of {TRACKS}.")
    config = ExperimentConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        experiment=experiment,
        track=str(raw["track"]),
        description=str(raw["description"]),
        data=_section(DataConfig, raw["data"], "data"),
        universe=_section(UniverseConfig, raw["universe"], "universe"),
        features=_section(FeatureConfig, raw["features"], "features"),
        label=_section(LabelConfig, raw["label"], "label"),
        folds=_section(FoldConfig, raw["folds"], "folds"),
        models=_section(ModelConfig, raw["models"], "models"),
        evaluation=_section(EvaluationConfig, raw["evaluation"], "evaluation"),
        registry=_section(RegistryConfig, raw["registry"], "registry"),
    )
    if DATA_SOURCES[config.data.source] != config.track:
        raise ConfigError(
            f"data.source {config.data.source!r} does not serve track {config.track!r}."
        )
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config (safe loader only)."""

    with Path(path).open(encoding="utf-8") as handle:
        return parse_config(yaml.safe_load(handle))
