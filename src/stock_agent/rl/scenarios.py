"""
A deterministic, versioned, seeded library of synthetic market scenarios.

Each scenario is an ETF-like universe (an equity fund, a bond fund, a
short-Treasury fund) plus a cash account, simulated session by session from
a cyclic schedule of regimes:

    calm, crash, high_correlation, extreme_volatility,
    stale_feed (stale observations), gappy (missing closes), high_costs

`generate_scenario(scenario_id, seed, generator_version)` replays exactly:
every random draw comes from a legacy `RandomState` stream (frozen across
NumPy versions) seeded by SHA-256 of (seed, scenario, version, purpose,
segment), so one segment's draws never shift another's.

Arrays (T sessions, n assets, cash last):

    latent_returns  true per-session total returns; for tests and audit only.
                    The environment and observations never read them.
    returns         the return from an asset's previous close to its close at
                    t; NaN where the asset has no close (a gap), and the
                    return at the next close covers the whole gap.
    has_close       False on gap sessions (cash always has a close).
    obs_lag         data-feed staleness: at decision session t the latest
                    observation of asset i is from session t - obs_lag[t, i].
                    Execution still happens at true closes; only the
                    information channel is stale.
    cost_multiplier per-session multiplier on the base cost rates.

Assumptions (synthetic, illustrative, never calibrated to real funds):
annual log drifts and volatilities per regime, constant within a segment;
Gaussian or unit-variance Student-t shocks with a fixed correlation;
uninvested cash earns 0 (decision D4 (a), approved 2026-10-15 for synthetic
engineering only); base cost rates are the D2 provisional all-in rates
(5, 5, 2 and 0 bp, approved 2026-10-15 as provisional assumptions), read
from `stock_agent.portfolio.transition.PROVISIONAL_RATES`.

Generator versions. The version enters every seed, so a new version changes
every draw, and an old version is refused rather than approximated.
- 1.1.0 (2026-10-14): the crash regime is reliably severe. Over 1,000
  40-session crash segments the equity fund's return had a median of about
  -30%, a 5-95% range of about -44% to -12%, and under 1% of segments were
  non-negative (1.0.0: median -22%, 8.9% non-negative).
- 1.0.0: first version; no longer generated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from stock_agent.model_diagnostics.seeds import derive_seed
from stock_agent.portfolio.transition import BROAD_ETF, CASH_ACCOUNT, PROVISIONAL_RATES, TBILL_FUND

GENERATOR_VERSION = "1.1.0"
SUPPORTED_GENERATOR_VERSIONS = ("1.1.0",)

ASSET_NAMES = ("equity_etf", "bond_etf", "short_treasury_fund", "cash")
CASH_INDEX = 3
N_RISKY = 3
ASSET_CLASSES = (BROAD_ETF, BROAD_ETF, TBILL_FUND, CASH_ACCOUNT)
BASE_COST_RATES = tuple(PROVISIONAL_RATES[klass] for klass in ASSET_CLASSES)
SESSIONS_PER_YEAR = 252
DEFAULT_TOTAL_SESSIONS = 1500


def _correlation_matrix(pairs: tuple[float, float, float]) -> np.ndarray:
    eq_bond, eq_stf, bond_stf = pairs
    return np.array(
        [[1.0, eq_bond, eq_stf], [eq_bond, 1.0, bond_stf], [eq_stf, bond_stf, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RegimeSpec:
    """
    One market regime for the three risky funds.

    correlation is (equity-bond, equity-short Treasury, bond-short Treasury).
    tail_df is None for Gaussian shocks, else the Student-t degrees of
    freedom (scaled to unit variance; must exceed 2).
    """

    name: str
    annual_drift: tuple[float, float, float]
    annual_vol: tuple[float, float, float]
    correlation: tuple[float, float, float]
    tail_df: float | None = None
    gap_probability: float = 0.0
    stale_probability: float = 0.0
    stale_max_run: int = 0
    cost_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if len(self.annual_drift) != N_RISKY or len(self.annual_vol) != N_RISKY:
            raise ValueError("A regime needs one drift and one volatility per risky fund.")
        if any(v < 0 for v in self.annual_vol):
            raise ValueError("Volatilities must be non-negative.")
        try:
            np.linalg.cholesky(_correlation_matrix(self.correlation))
        except np.linalg.LinAlgError as error:
            raise ValueError(f"Regime {self.name!r}: correlation is not positive definite.") from (
                error
            )
        for name in ("gap_probability", "stale_probability"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"Regime {self.name!r}: {name} must be a probability in [0, 1).")
        if self.stale_probability > 0 and self.stale_max_run < 1:
            raise ValueError("A stale feed needs stale_max_run >= 1.")
        if self.tail_df is not None and self.tail_df <= 2:
            raise ValueError("tail_df must exceed 2 for a finite variance.")
        if self.cost_multiplier < 0:
            raise ValueError("cost_multiplier must be non-negative.")

    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(_correlation_matrix(self.correlation))


_CALM = {
    "annual_drift": (0.06, 0.025, 0.02),
    "annual_vol": (0.15, 0.05, 0.006),
    "correlation": (-0.2, 0.0, 0.3),
}

REGIMES: dict[str, RegimeSpec] = {
    "calm": RegimeSpec(name="calm", **_CALM),
    "crash": RegimeSpec(
        name="crash",
        annual_drift=(-2.25, 0.08, 0.02),
        annual_vol=(0.35, 0.09, 0.01),
        correlation=(-0.4, 0.0, 0.3),
        tail_df=4.0,
    ),
    "high_correlation": RegimeSpec(
        name="high_correlation",
        annual_drift=(0.03, 0.01, 0.02),
        annual_vol=(0.20, 0.08, 0.008),
        correlation=(0.9, 0.5, 0.5),
    ),
    "extreme_volatility": RegimeSpec(
        name="extreme_volatility",
        annual_drift=(0.0, 0.0, 0.02),
        annual_vol=(0.80, 0.25, 0.02),
        correlation=(-0.2, 0.0, 0.3),
        tail_df=3.0,
    ),
    "stale_feed": RegimeSpec(name="stale_feed", stale_probability=0.04, stale_max_run=5, **_CALM),
    "gappy": RegimeSpec(name="gappy", gap_probability=0.03, **_CALM),
    "high_costs": RegimeSpec(name="high_costs", cost_multiplier=5.0, **_CALM),
}


@dataclass(frozen=True)
class ScenarioSpec:
    """A cyclic regime schedule: segments repeat until total_sessions."""

    scenario_id: str
    description: str
    schedule: tuple[tuple[str, int], ...]
    total_sessions: int = DEFAULT_TOTAL_SESSIONS
    cash_return_per_session: float = 0.0
    cost_regime_observable: bool = True

    def __post_init__(self) -> None:
        if not self.schedule or any(length < 1 for _, length in self.schedule):
            raise ValueError("A schedule needs segments of at least one session.")
        unknown = {name for name, _ in self.schedule} - set(REGIMES)
        if unknown:
            raise ValueError(f"Unknown regimes in schedule: {sorted(unknown)}")
        if self.total_sessions < 3:
            raise ValueError("A scenario needs at least three sessions.")

    def segments(self) -> list[tuple[str, int, int]]:
        """(regime, start, stop) segments covering [0, total_sessions)."""

        out: list[tuple[str, int, int]] = []
        start = 0
        while start < self.total_sessions:
            for name, length in self.schedule:
                stop = min(start + length, self.total_sessions)
                out.append((name, start, stop))
                start = stop
                if start >= self.total_sessions:
                    break
        return out


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.scenario_id: spec
    for spec in (
        ScenarioSpec("calm", "Calm markets throughout.", (("calm", 300),)),
        ScenarioSpec(
            "crash",
            "A 40-session equity crash with heavy tails every 300 sessions.",
            (("calm", 260), ("crash", 40)),
        ),
        ScenarioSpec(
            "high_correlation",
            "100-session spells where equity and bonds move together.",
            (("calm", 200), ("high_correlation", 100)),
        ),
        ScenarioSpec(
            "extreme_volatility",
            "80-session spells of very high, heavy-tailed volatility.",
            (("calm", 220), ("extreme_volatility", 80)),
        ),
        ScenarioSpec(
            "stale_observations",
            "150-session spells where the data feed lags by up to five sessions.",
            (("calm", 150), ("stale_feed", 150)),
        ),
        ScenarioSpec(
            "missing_closes",
            "150-session spells where funds randomly miss closes.",
            (("calm", 150), ("gappy", 150)),
        ),
        ScenarioSpec(
            "high_costs",
            "150-session spells with five times the base trading costs.",
            (("calm", 150), ("high_costs", 150)),
        ),
        ScenarioSpec(
            "mixed",
            "Every regime in turn; the default training scenario.",
            (
                ("calm", 120),
                ("crash", 30),
                ("calm", 60),
                ("high_correlation", 60),
                ("extreme_volatility", 30),
                ("stale_feed", 30),
                ("gappy", 30),
                ("high_costs", 60),
            ),
        ),
    )
}


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class ScenarioData:
    scenario_id: str
    generator_version: str
    seed: int
    returns: np.ndarray
    has_close: np.ndarray
    obs_lag: np.ndarray
    cost_multiplier: np.ndarray
    regime: np.ndarray
    latent_returns: np.ndarray
    cost_regime_observable: bool = True
    asset_names: tuple[str, ...] = ASSET_NAMES
    cash_index: int = CASH_INDEX
    base_cost_rates: tuple[float, ...] = BASE_COST_RATES
    _complete: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.asset_names)
        sessions = self.returns.shape[0]
        for name in ("returns", "has_close", "obs_lag", "latent_returns"):
            if getattr(self, name).shape != (sessions, n):
                raise ValueError(f"{name} must have shape ({sessions}, {n}).")
        if self.cost_multiplier.shape != (sessions,) or self.regime.shape != (sessions,):
            raise ValueError("cost_multiplier and regime need one entry per session.")
        for name in ("returns", "has_close", "obs_lag", "cost_multiplier", "regime"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "latent_returns", _readonly(self.latent_returns))
        object.__setattr__(self, "_complete", _readonly(self.has_close.all(axis=1)))

    @property
    def sessions(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)

    def identity(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        header = {
            "scenario_id": self.scenario_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "asset_names": list(self.asset_names),
            "base_cost_rates": list(self.base_cost_rates),
            "cost_regime_observable": self.cost_regime_observable,
        }
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        for array in (self.returns, self.has_close, self.obs_lag, self.cost_multiplier):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def is_complete(self, session: int) -> bool:
        return bool(self._complete[session])

    def complete_sessions(self, start: int, stop: int) -> np.ndarray:
        """Sessions in [start, stop) where every asset has a close."""

        start, stop = max(0, start), min(self.sessions, stop)
        return np.flatnonzero(self._complete[start:stop]) + start

    def span_returns(self, start: int, stop: int) -> np.ndarray:
        """
        Each asset's total return from its close at `start` to its close at
        `stop` (both complete sessions): the product of its closes' returns
        in (start, stop]. A gap's return is already inside the next close's.
        """

        if stop <= start:
            raise ValueError("The span must end after it starts.")
        if not (self.is_complete(start) and self.is_complete(stop)):
            raise ValueError("Span endpoints must be complete sessions.")
        window = self.returns[start + 1 : stop + 1]
        growth = np.where(self.has_close[start + 1 : stop + 1], 1.0 + window, 1.0)
        return np.prod(growth, axis=0) - 1.0

    def cost_rates_at(self, session: int) -> np.ndarray:
        rates = np.asarray(self.base_cost_rates, dtype=np.float64) * float(
            self.cost_multiplier[session]
        )
        rates[self.cash_index] = 0.0
        return rates


def _segment_shocks(regime: RegimeSpec, length: int, rng: np.random.RandomState) -> np.ndarray:
    shocks = rng.standard_normal((length, N_RISKY))
    if regime.tail_df is not None:
        mixing = rng.chisquare(regime.tail_df, size=length)
        shocks = shocks * np.sqrt((regime.tail_df - 2.0) / mixing)[:, None]
    return shocks @ regime.cholesky().T


def generate_scenario(
    scenario_id: str,
    seed: int,
    generator_version: str = GENERATOR_VERSION,
    library: dict[str, ScenarioSpec] | None = None,
) -> ScenarioData:
    if generator_version not in SUPPORTED_GENERATOR_VERSIONS:
        raise ValueError(
            f"Unsupported generator version {generator_version!r}; this code generates "
            f"{SUPPORTED_GENERATOR_VERSIONS}. Replays never approximate another version."
        )
    spec = (library or SCENARIOS)[scenario_id]
    sessions = spec.total_sessions
    n = len(ASSET_NAMES)

    def stream(*purpose: object) -> np.random.RandomState:
        return np.random.RandomState(
            derive_seed(seed, "rl_scenario", scenario_id, generator_version, *purpose)
        )

    segments = spec.segments()
    regime_names = np.empty(sessions, dtype=object)
    log_returns = np.zeros((sessions, n), dtype=np.float64)
    gap_p = np.zeros(sessions)
    stale_p = np.zeros(sessions)
    stale_run = np.zeros(sessions, dtype=np.int64)
    cost_multiplier = np.ones(sessions)
    for index, (name, start, stop) in enumerate(segments):
        regime = REGIMES[name]
        regime_names[start:stop] = name
        drift = np.asarray(regime.annual_drift) / SESSIONS_PER_YEAR
        vol = np.asarray(regime.annual_vol) / np.sqrt(SESSIONS_PER_YEAR)
        shocks = _segment_shocks(regime, stop - start, stream("returns", index))
        log_returns[start:stop, :N_RISKY] = drift + shocks * vol
        gap_p[start:stop] = regime.gap_probability
        stale_p[start:stop] = regime.stale_probability
        stale_run[start:stop] = regime.stale_max_run
        cost_multiplier[start:stop] = regime.cost_multiplier
    log_returns[:, CASH_INDEX] = np.log1p(spec.cash_return_per_session)
    latent = np.expm1(log_returns)

    has_close = np.ones((sessions, n), dtype=bool)
    has_close[:, :N_RISKY] = stream("gaps").random_sample((sessions, N_RISKY)) >= gap_p[:, None]
    has_close[0, :] = True

    returns = np.full((sessions, n), np.nan)
    for asset in range(n):
        accumulated = 0.0
        for t in range(sessions):
            accumulated += log_returns[t, asset]
            if has_close[t, asset]:
                returns[t, asset] = np.expm1(accumulated)
                accumulated = 0.0
    returns[:, CASH_INDEX] = latent[:, CASH_INDEX]

    stale_rng = stream("stale")
    starts = stale_rng.random_sample((sessions, N_RISKY)) < stale_p[:, None]
    lengths = 1 + np.floor(
        stale_rng.random_sample((sessions, N_RISKY)) * stale_run[:, None]
    ).astype(np.int64)
    obs_lag = np.zeros((sessions, n), dtype=np.int64)
    for t, asset in zip(*np.nonzero(starts), strict=True):
        for s in range(t, min(sessions, t + int(lengths[t, asset]))):
            obs_lag[s, asset] = max(obs_lag[s, asset], min(s - t + 1, s))

    return ScenarioData(
        scenario_id=scenario_id,
        generator_version=generator_version,
        seed=seed,
        returns=returns,
        has_close=has_close,
        obs_lag=obs_lag,
        cost_multiplier=cost_multiplier,
        regime=regime_names,
        latent_returns=latent,
        cost_regime_observable=spec.cost_regime_observable,
    )
