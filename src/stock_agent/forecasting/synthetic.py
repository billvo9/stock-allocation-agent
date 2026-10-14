"""
Seeded synthetic market data with planted, known signals.

Synthetic data verifies the pipeline: a planted signal must be recovered,
and null data must show nothing. It is never evidence about markets.

Equity panel (`synthetic_equity_panel`)
    Daily log returns  r_it = beta_i m_t + mu_it + sigma_i e_it
    m_t ~ N(market_drift, market_vol), e_it ~ N(0, 1), beta_i ~ U(0.8, 1.2),
    sigma_i ~ U(idio_vol range); close = 100 exp(cumulative r) from listing.

    A quarterly quality characteristic q_ik (AR(1), phi = 0.9) is reported
    in filings: gross profit / total assets = 0.30 + 0.10 q. Each filing
    becomes available `filing_lag` days after its quarter end at 21:30 UTC
    (after the close, so usable from the next session under the 00:00 UTC
    rule); some quarters get a later amendment with a revised value.

    The planted drift mu_it depends only on what is observable on day t:
    z_it is the within-date standardized gross profitability of the latest
    period known on t (as-of: latest period, then its latest version), and

        none       mu = 0
        linear     mu = s z
        nonlinear  mu = s (|z| - mean over the date of |z|)

    so a 20-session label has an expected excess return of about 20 s z
    (linear). The nonlinear drift is symmetric in z: a linear model cannot
    use it, a tree ensemble can. Leverage and volume are unrelated to
    returns (noise features). `truth` holds mu_it.

Every random draw uses numpy's legacy RandomState (stream frozen across
numpy versions). Calendars are business days and must end before the
2025-01-01 lockbox.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stock_agent.features.asof import asof_fundamentals
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

SIGNALS = ("none", "linear", "nonlinear")
FILING_HOUR_UTC = pd.Timedelta(hours=21, minutes=30)


@dataclass(frozen=True)
class SyntheticEquitySpec:
    n_assets: int = 30
    n_sessions: int = 1500
    start: str = "2012-01-03"
    signal: str = "linear"
    signal_strength: float = 4e-4
    late_listings: int = 3
    delistings: int = 0
    market_drift: float = 3e-4
    market_vol: float = 0.01
    idio_vol_low: float = 0.012
    idio_vol_high: float = 0.025
    filing_lag_min_days: int = 30
    filing_lag_max_days: int = 55
    amendment_share: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}; got {self.signal!r}.")
        if self.n_assets < 4 or self.n_sessions < 300:
            raise ValueError("A synthetic panel needs at least 4 assets and 300 sessions.")
        if self.late_listings + self.delistings > self.n_assets // 2:
            raise ValueError("At most half the assets may list late or delist.")

    def describe(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class SyntheticEquityPanel:
    prices: pd.DataFrame  # date, symbol, close, volume
    fundamentals: pd.DataFrame  # symbol, period_end, available_at, gross_profit, total_assets,
    #                             total_liabilities, amendment
    truth: pd.DataFrame  # date, symbol, drift (the planted daily expected excess return)
    spec: SyntheticEquitySpec
    # Every asset that ever existed is in the panel, delisted ones included.
    survivorship_free: bool = True


def _calendar(start: str, periods: int) -> pd.DatetimeIndex:
    calendar = pd.bdate_range(start, periods=periods, tz="UTC")
    if calendar[-1] >= MODEL_HOLDOUT_START:
        raise ValueError(
            f"The synthetic calendar ends {calendar[-1].date()}; synthetic data must end "
            "before the 2025-01-01 lockbox."
        )
    return calendar


def _filings(
    rng: np.random.RandomState, spec: SyntheticEquitySpec, symbols, calendar
) -> pd.DataFrame:
    quarter_ends = pd.date_range(
        calendar[0] - pd.offsets.QuarterEnd(2), calendar[-1], freq="QE", tz="UTC"
    )
    rows = []
    for symbol in symbols:
        q = rng.randn()
        assets = float(np.exp(rng.normal(9.0, 1.0)))
        for period_end in quarter_ends:
            q = 0.9 * q + np.sqrt(1 - 0.9**2) * rng.randn()
            lag = rng.randint(spec.filing_lag_min_days, spec.filing_lag_max_days + 1)
            available = period_end.normalize() + pd.Timedelta(days=int(lag)) + FILING_HOUR_UTC
            leverage = float(np.clip(0.5 + 0.1 * rng.randn(), 0.05, 0.95))
            rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end.normalize(),
                    "available_at": available,
                    "gross_profit": (0.30 + 0.10 * q) * assets,
                    "total_assets": assets,
                    "total_liabilities": leverage * assets,
                    "amendment": False,
                }
            )
            if rng.rand() < spec.amendment_share:
                revised = q + 0.3 * rng.randn()
                rows.append(
                    {
                        **rows[-1],
                        "available_at": available + pd.Timedelta(days=int(rng.randint(20, 80))),
                        "gross_profit": (0.30 + 0.10 * revised) * assets,
                        "amendment": True,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["available_at"] = frame["available_at"].dt.as_unit("ns")
    frame["period_end"] = frame["period_end"].dt.as_unit("ns")
    return frame.sort_values(["symbol", "available_at", "period_end"]).reset_index(drop=True)


def _row_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean of the finite values of each row (NaN for a row without any)."""

    finite = np.isfinite(matrix)
    counts = finite.sum(axis=1)
    sums = np.where(finite, matrix, 0.0).sum(axis=1)
    return np.divide(sums, counts, out=np.full(len(matrix), np.nan), where=counts > 0)


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    """Within-row z-scores over finite values (population std); NaN elsewhere."""

    mean = _row_mean(matrix)
    centred = matrix - mean[:, None]
    variance = _row_mean(centred**2)
    scale = np.sqrt(variance)
    usable = np.isfinite(scale) & (scale > 0)
    return np.where(usable[:, None], centred / np.where(usable, scale, 1.0)[:, None], np.nan)


def synthetic_equity_panel(spec: SyntheticEquitySpec) -> SyntheticEquityPanel:
    rng = np.random.RandomState(spec.seed)
    calendar = _calendar(spec.start, spec.n_sessions)
    n, k = spec.n_sessions, spec.n_assets
    symbols = [f"S{index:03d}" for index in range(k)]

    first = np.zeros(k, dtype=np.int64)
    last = np.full(k, n, dtype=np.int64)
    late = rng.choice(k, size=spec.late_listings + spec.delistings, replace=False)
    for index in late[: spec.late_listings]:
        first[index] = rng.randint(n // 5, n // 2)
    for index in late[spec.late_listings :]:
        last[index] = rng.randint(n // 2, n - n // 5)

    beta = rng.uniform(0.8, 1.2, size=k)
    sigma = rng.uniform(spec.idio_vol_low, spec.idio_vol_high, size=k)
    market = rng.normal(spec.market_drift, spec.market_vol, size=n)
    shocks = rng.randn(n, k)
    log_volume_level = rng.normal(13.0, 1.0, size=k)
    volume_noise = rng.randn(n, k)
    fundamentals = _filings(rng, spec, symbols, calendar)

    listed = (np.arange(n)[:, None] >= first[None, :]) & (np.arange(n)[:, None] < last[None, :])
    keys = pd.DataFrame(
        {"date": np.repeat(calendar, k), "symbol": np.tile(symbols, n)}
    )  # date-major, symbol-minor: row t * k + i
    observed = asof_fundamentals(keys, fundamentals)
    profitability = (observed["gross_profit"] / observed["total_assets"]).to_numpy().reshape(n, k)
    profitability = np.where(listed, profitability, np.nan)
    z = _standardize_rows(profitability)
    known = np.isfinite(z)
    if spec.signal == "linear":
        drift = spec.signal_strength * np.where(known, z, 0.0)
    elif spec.signal == "nonlinear":
        magnitude = np.where(known, np.abs(z), np.nan)
        centre = _row_mean(magnitude)
        drift = spec.signal_strength * np.where(known, magnitude - centre[:, None], 0.0)
    else:
        drift = np.zeros((n, k))
    drift = np.where(listed, drift, 0.0)

    returns = beta[None, :] * market[:, None] + drift + sigma[None, :] * shocks
    returns = np.where(listed, returns, 0.0)
    cumulative = np.cumsum(returns, axis=0)
    close = 100.0 * np.exp(cumulative - cumulative[first, np.arange(k)][None, :])
    volume = np.exp(log_volume_level[None, :] + 0.3 * volume_noise)

    prices = pd.DataFrame(
        {
            "date": np.repeat(calendar, k),
            "symbol": np.tile(symbols, n),
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
            "drift": drift.reshape(-1),
            "_listed": listed.reshape(-1),
        }
    )
    prices = prices.loc[prices["_listed"]].drop(columns="_listed").reset_index(drop=True)
    truth = prices.loc[:, ["date", "symbol", "drift"]].copy()
    prices = prices.drop(columns="drift")
    prices["date"] = prices["date"].dt.as_unit("ns")
    truth["date"] = truth["date"].dt.as_unit("ns")
    in_panel = fundamentals["symbol"].isin(prices["symbol"].unique())
    return SyntheticEquityPanel(
        prices=prices,
        fundamentals=fundamentals.loc[in_panel].reset_index(drop=True),
        truth=truth,
        spec=spec,
    )
