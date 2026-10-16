"""
Seeded synthetic demo stocks for the Stock Probability page and tests.

They are not real stocks and never stand in for one: their identifiers
(DEMO_CALM, DEMO_SWING, DEMO_TREND) contain an underscore, which no US
ticker can, so a demo stock can never be mistaken for, or stored as, a
watchlist ticker. Every view of them is labelled synthetic.

Each history is a deterministic function of its seed (NumPy's legacy
RandomState stream, stable across NumPy versions). Sessions are synthetic
weekdays ending 2024-12-31; they are not NYSE sessions.

Model (per demo stock): a two-state volatility regime (a Markov chain that
stays put with probability `stay`), returns

    r_t = drift + phi * r_(t-1) + sigma_(regime_t) * e_t,   e_t ~ Student t(5), unit variance

then raw bars: a split whose raw close drops by the factor, quarterly cash
dividends whose ex-date price return includes the mechanical drop, an open
with a small gap, a high and low around them, and a volume that rises with
|r_t|. DEMO_TREND also has staged failed breakouts (a decline, a rebound
above the prior high, then a fall back below it), so the reversal and trap
views have something to show.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

LAST_SESSION = "2024-12-31"
SESSIONS = 1500


@dataclass(frozen=True)
class DemoSpec:
    stock_id: str
    name: str
    description: str
    seed: int
    drift: float
    phi: float
    sigma_low: float
    sigma_high: float
    stay: float
    split_at: int | None = None
    split_factor: float = 1.0
    dividend_yield: float = 0.0
    staged_traps: int = 0


DEMOS = (
    DemoSpec(
        "DEMO_CALM",
        "Calm demo stock (synthetic)",
        "Low volatility with mild mean reversion: many Stable sessions at ±0.5%.",
        seed=11,
        drift=0.0002,
        phi=-0.04,
        sigma_low=0.006,
        sigma_high=0.010,
        stay=0.99,
        dividend_yield=0.006,
    ),
    DemoSpec(
        "DEMO_SWING",
        "Swing demo stock (synthetic)",
        "Volatility that switches between calm and turbulent spells, with a 3-for-1 split.",
        seed=23,
        drift=0.0003,
        phi=0.0,
        sigma_low=0.010,
        sigma_high=0.030,
        stay=0.985,
        split_at=900,
        split_factor=3.0,
        dividend_yield=0.003,
    ),
    DemoSpec(
        "DEMO_TREND",
        "Trend demo stock (synthetic)",
        "Moderate volatility, slight momentum, and several staged failed breakouts.",
        seed=37,
        drift=0.0004,
        phi=0.05,
        sigma_low=0.013,
        sigma_high=0.022,
        stay=0.99,
        staged_traps=4,
    ),
)
DEMO_IDS = tuple(spec.stock_id for spec in DEMOS)
_BY_ID = {spec.stock_id: spec for spec in DEMOS}


def is_demo(stock_id: str) -> bool:
    return stock_id in _BY_ID


def demo_spec(stock_id: str) -> DemoSpec:
    return _BY_ID[stock_id]


def _student_t(rng: np.random.RandomState, size: int, dof: float = 5.0) -> np.ndarray:
    draws = rng.standard_t(dof, size=size)
    return draws / np.sqrt(dof / (dof - 2.0))


def _returns(spec: DemoSpec, rng: np.random.RandomState, n: int) -> np.ndarray:
    shocks = _student_t(rng, n)
    switches = rng.rand(n)
    high = False
    r = np.zeros(n)
    for t in range(n):
        if switches[t] > spec.stay:
            high = not high
        sigma = spec.sigma_high if high else spec.sigma_low
        previous = r[t - 1] if t else 0.0
        r[t] = spec.drift + spec.phi * previous + sigma * shocks[t]
    return np.clip(r, -0.35, 0.35)


def _stage_traps(r: np.ndarray, spec: DemoSpec, rng: np.random.RandomState) -> None:
    """Overwrite stretches with: a decline, a rebound to a new 20-session high, a failure."""

    if not spec.staged_traps:
        return
    starts = np.linspace(300, len(r) - 120, spec.staged_traps).astype(int)
    for start in starts:
        pattern = np.concatenate(
            [
                np.full(12, -0.012),  # a decline of about 13%
                np.full(10, 0.016),  # a rebound of about 17%, through the prior high
                np.full(3, 0.006),
                np.full(8, -0.014),  # a failure back below the breakout level
            ]
        )
        noise = rng.normal(0.0, 0.002, size=len(pattern))
        r[start : start + len(pattern)] = pattern + noise


def demo_bars(stock_id: str) -> pd.DataFrame:
    """The raw daily bars of one demo stock (prices.BAR_COLUMNS)."""

    spec = demo_spec(stock_id)
    rng = np.random.RandomState(spec.seed)
    n = SESSIONS
    sessions = pd.bdate_range(end=LAST_SESSION, periods=n, tz="UTC").as_unit("us")
    r = _returns(spec, rng, n)
    _stage_traps(r, spec, rng)

    split = np.ones(n)
    if spec.split_at is not None:
        split[spec.split_at] = spec.split_factor
    level = 40.0 * np.cumprod(1.0 + r)
    raw_close = level / np.cumprod(split)

    dividends = np.zeros(n)
    if spec.dividend_yield:
        for t in range(63, n, 63):
            amount = round(spec.dividend_yield * raw_close[t - 1], 4)
            dividends[t] = amount
            # The ex-date price return includes the mechanical drop.
            raw_close[t:] -= amount * raw_close[t:] / raw_close[t]
    raw_close = np.round(raw_close, 4)

    sigma = np.maximum(np.abs(r), 0.004)
    previous = np.concatenate([[raw_close[0]], raw_close[:-1] / split[1:]])
    previous[0] = raw_close[0]
    gaps = rng.normal(0.0, 0.35, size=n) * sigma
    opens = np.round(previous * np.exp(gaps), 4)
    spread = np.abs(rng.normal(0.0, 0.4, size=n)) * sigma
    highs = np.round(np.maximum(opens, raw_close) * (1.0 + spread), 4)
    lows = np.round(np.minimum(opens, raw_close) * (1.0 - spread), 4)
    volume = 2.0e6 * np.exp(rng.normal(0.0, 0.3, size=n)) * (1.0 + 8.0 * np.abs(r))
    volume = np.round(volume * np.cumprod(split))
    return pd.DataFrame(
        {
            "session": sessions,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": raw_close,
            "volume": volume,
            "div_cash": dividends,
            "split_factor": split,
        }
    )


def demo_histories() -> Mapping[str, pd.DataFrame]:
    """Every demo stock's bars, by identifier, in DEMO_IDS order."""

    return {stock_id: demo_bars(stock_id) for stock_id in DEMO_IDS}
