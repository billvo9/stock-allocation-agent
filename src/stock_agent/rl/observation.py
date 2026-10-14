"""
Point-in-time observations for allocation policies.

At decision session d (the close of d), a policy may know:
- each fund's returns up to the latest session its data feed has delivered
  (d - obs_lag[d, i]); never r_{d+1}, r_{d+2}, ... (those are earned after
  the decision executes at the close of d+1);
- the portfolio's own weights after the close of d and its drawdown so far
  (broker records, not the market-data feed);
- the cost regime at d, when the scenario declares it observable.

`snapshot_at` is the only bridge from a scenario to a policy. It copies rows
0..d, blanks what the feed has not delivered, and `MarketSnapshot` refuses
any row after d, so no feature function can index the future.

Market features are z-scored with a `FeatureNormalizer` fitted on training
fold sessions only (`fit_normalizer` refuses dev and test folds), frozen,
and saved with every checkpoint. Weights and drawdown are already bounded
and are not normalized.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from stock_agent.rl.scenarios import ScenarioData, generate_scenario
from stock_agent.rl.splits import FoldSpec, LeakageError

FEATURE_SPEC_VERSION = "1.0.0"
DEFAULT_CLIP = 10.0


@dataclass(frozen=True)
class FeatureSpec:
    return_windows: tuple[int, ...] = (5, 20, 60)
    vol_windows: tuple[int, ...] = (20, 60)
    include_cost_regime: bool = True
    min_fraction: float = 0.5
    version: str = FEATURE_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.return_windows and not self.vol_windows:
            raise ValueError("A feature spec needs at least one window.")
        if any(window < 1 for window in self.return_windows):
            raise ValueError("Return windows must be at least one session.")
        if any(window < 2 for window in self.vol_windows):
            raise ValueError("Volatility windows must be at least two sessions.")
        if not 0.0 < self.min_fraction <= 1.0:
            raise ValueError("min_fraction must be in (0, 1].")

    @property
    def lookback(self) -> int:
        return max((*self.return_windows, *self.vol_windows))

    def market_names(self, asset_names: tuple[str, ...], cash_index: int) -> tuple[str, ...]:
        names: list[str] = []
        for index, asset in enumerate(asset_names):
            if index == cash_index:
                continue
            names += [f"{asset}:ret_{window}" for window in self.return_windows]
            names += [f"{asset}:vol_{window}" for window in self.vol_windows]
        if self.include_cost_regime:
            names.append("cost_multiplier")
        return tuple(names)

    def to_dict(self) -> dict[str, object]:
        return {
            "return_windows": list(self.return_windows),
            "vol_windows": list(self.vol_windows),
            "include_cost_regime": self.include_cost_regime,
            "min_fraction": self.min_fraction,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, record: dict) -> FeatureSpec:
        return cls(
            return_windows=tuple(record["return_windows"]),
            vol_windows=tuple(record["vol_windows"]),
            include_cost_regime=bool(record["include_cost_regime"]),
            min_fraction=float(record["min_fraction"]),
            version=record["version"],
        )


@dataclass(frozen=True)
class MarketSnapshot:
    """Market data visible at the close of `decision_session`: rows 0..d only."""

    decision_session: int
    returns: np.ndarray
    obs_age: np.ndarray
    cost_multiplier: float | None

    def __post_init__(self) -> None:
        rows = self.returns.shape[0]
        if rows > self.decision_session + 1:
            raise LeakageError(
                f"A snapshot for session {self.decision_session} holds {rows} rows, "
                "including rows after the decision session."
            )
        if rows != self.decision_session + 1:
            raise ValueError("A snapshot must hold every row from session 0 to d.")
        for name in ("returns", "obs_age"):
            array = np.array(getattr(self, name))
            array.flags.writeable = False
            object.__setattr__(self, name, array)


def snapshot_at(data: ScenarioData, decision_session: int) -> MarketSnapshot:
    d = int(decision_session)
    if not 0 <= d < data.sessions:
        raise ValueError(f"Session {d} is outside the scenario.")
    rows = np.arange(d + 1)[:, None]
    lag = np.asarray(data.obs_lag[d])
    visible = data.has_close[: d + 1] & (rows <= d - lag[None, :])
    returns = np.where(visible, data.returns[: d + 1], np.nan)
    cost = float(data.cost_multiplier[d]) if data.cost_regime_observable else None
    return MarketSnapshot(
        decision_session=d, returns=returns, obs_age=lag.astype(np.int64), cost_multiplier=cost
    )


def _window(snapshot: MarketSnapshot, asset: int, length: int) -> np.ndarray | None:
    first = snapshot.decision_session - length + 1
    if first < 0:
        return None
    values = snapshot.returns[first:, asset]
    return values[np.isfinite(values)]


def market_features(snapshot: MarketSnapshot, spec: FeatureSpec, cash_index: int) -> np.ndarray:
    """Raw (unnormalized) market features in `spec.market_names` order."""

    values: list[float] = []
    for asset in range(snapshot.returns.shape[1]):
        if asset == cash_index:
            continue
        for length in spec.return_windows:
            window = _window(snapshot, asset, length)
            needed = max(1, math.ceil(spec.min_fraction * length))
            if window is None or window.size < needed:
                values.append(math.nan)
            else:
                values.append(float(np.prod(1.0 + window) - 1.0))
        for length in spec.vol_windows:
            window = _window(snapshot, asset, length)
            needed = max(2, math.ceil(spec.min_fraction * length))
            if window is None or window.size < needed:
                values.append(math.nan)
            else:
                values.append(float(window.std(ddof=1)))
    if spec.include_cost_regime:
        if snapshot.cost_multiplier is None:
            raise ValueError("The feature spec uses the cost regime, which is not observable.")
        values.append(float(snapshot.cost_multiplier))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Frozen z-score parameters for market features, fitted on a training fold."""

    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float
    fitted_on: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not len(self.names) == len(self.mean) == len(self.scale):
            raise ValueError("names, mean and scale must have equal lengths.")
        if any(s <= 0 or not math.isfinite(s) for s in self.scale):
            raise ValueError("Every scale must be positive and finite.")
        if self.clip <= 0:
            raise ValueError("clip must be positive.")

    @classmethod
    def identity(cls, names: tuple[str, ...], clip: float = DEFAULT_CLIP) -> FeatureNormalizer:
        """No scaling. For tests and rule-only runs; never for a learned policy."""

        return cls(
            names=tuple(names),
            mean=(0.0,) * len(names),
            scale=(1.0,) * len(names),
            clip=clip,
            fitted_on={"role": "identity", "note": "unfitted; tests and rule-only runs"},
        )

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(clipped z, unclipped z). Never updates the frozen statistics."""

        z = (np.asarray(raw, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.clip(z, -self.clip, self.clip), z

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "clip": self.clip,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, record: dict) -> FeatureNormalizer:
        return cls(
            names=tuple(record["names"]),
            mean=tuple(float(v) for v in record["mean"]),
            scale=tuple(float(v) for v in record["scale"]),
            clip=float(record["clip"]),
            fitted_on=dict(record["fitted_on"]),
        )


ScenarioLoader = Callable[[str, int], ScenarioData]


def fit_normalizer(
    fold: FoldSpec,
    spec: FeatureSpec,
    *,
    loader: ScenarioLoader = generate_scenario,
    clip: float = DEFAULT_CLIP,
) -> FeatureNormalizer:
    """
    Mean and standard deviation of each market feature over every complete
    decision session in the training fold's window. Market features do not
    depend on the policy, so the statistics are fixed before any training.
    """

    if fold.role != "train":
        raise LeakageError(
            f"Normalization may be fitted on the training fold only, not the {fold.role} fold."
        )
    if fold.start < spec.lookback:
        raise ValueError(
            f"The fold starts at {fold.start}, before the {spec.lookback}-session lookback."
        )
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    versions: set[str] = set()
    for scenario_id, seed in fold.episodes():
        data = loader(scenario_id, seed)
        versions.add(data.generator_version)
        names = spec.market_names(data.asset_names, data.cash_index)
        for d in data.complete_sessions(fold.start, fold.stop):
            rows.append(market_features(snapshot_at(data, int(d)), spec, data.cash_index))
    if not rows or names is None:
        raise ValueError("The training fold has no decision sessions.")
    matrix = np.vstack(rows)
    if np.any(np.all(~np.isfinite(matrix), axis=0)):
        raise ValueError("A feature has no finite value in the training fold.")
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    constant = [name for name, s in zip(names, std, strict=True) if not s > 1e-12]
    scale = np.where(std > 1e-12, std, 1.0)
    return FeatureNormalizer(
        names=names,
        mean=tuple(float(v) for v in mean),
        scale=tuple(float(v) for v in scale),
        clip=clip,
        fitted_on={
            "role": fold.role,
            "scenario_ids": list(fold.scenario_ids),
            "seeds": list(fold.seeds),
            "window": [fold.start, fold.stop],
            "generator_versions": sorted(versions),
            "feature_spec": spec.to_dict(),
            "decision_sessions": int(matrix.shape[0]),
            "constant_features": constant,
        },
    )


@dataclass(frozen=True)
class Observation:
    decision_session: int
    snapshot: MarketSnapshot
    raw_market: np.ndarray
    z_market: np.ndarray
    weights: np.ndarray
    drawdown: float
    vector: np.ndarray
    max_obs_age: int

    @property
    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.vector)))

    @property
    def max_abs_z(self) -> float:
        finite = self.z_market[np.isfinite(self.z_market)]
        return float(np.max(np.abs(finite))) if finite.size else math.nan


class ObservationBuilder:
    def __init__(
        self,
        spec: FeatureSpec,
        normalizer: FeatureNormalizer,
        asset_names: tuple[str, ...],
        cash_index: int,
    ) -> None:
        market = spec.market_names(asset_names, cash_index)
        if normalizer.names != market:
            raise ValueError("The normalizer was fitted for different features than this spec.")
        self.spec = spec
        self.normalizer = normalizer
        self.cash_index = cash_index
        self.asset_names = tuple(asset_names)
        self.names = (*market, *(f"weight:{asset}" for asset in asset_names), "drawdown")

    @property
    def dimension(self) -> int:
        return len(self.names)

    def build(self, snapshot: MarketSnapshot, weights: np.ndarray, drawdown: float) -> Observation:
        raw = market_features(snapshot, self.spec, self.cash_index)
        clipped, z = self.normalizer.transform(raw)
        weights = np.asarray(weights, dtype=np.float64).copy()
        weights.flags.writeable = False
        vector = np.concatenate([clipped, weights, [float(drawdown)]])
        vector.flags.writeable = False
        return Observation(
            decision_session=snapshot.decision_session,
            snapshot=snapshot,
            raw_market=raw,
            z_market=z,
            weights=weights,
            drawdown=float(drawdown),
            vector=vector,
            max_obs_age=int(np.max(snapshot.obs_age)) if snapshot.obs_age.size else 0,
        )
