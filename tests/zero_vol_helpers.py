"""
Synthetic F1 research batches for the zero-volatility tests: the prospective
fixture's VTI and AGG, with SHV replaced by a tick-discretized series whose
closes sit on the $0.01 grid. Every value is a fixture value, never market data.

SHV's latent price follows P_t = P_{t-1} exp(mu + sigma eps_t); its close is
P_t rounded to the cent, so a session moves only when the latent price
crosses a half-cent boundary. A low-volatility regime (a T-bill fund at zero
rates) makes whole outcome windows flat: measured zero realized volatility.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from forecast_prospective_helpers import ASSETS

from stock_agent.calendars import nyse
from stock_agent.forecasting.prospective import synthetic

START = pd.Timestamp("2019-01-02", tz="UTC")
END_EXCLUSIVE = pd.Timestamp("2025-01-01", tz="UTC")
RETRIEVED = pd.Timestamp("2026-10-14T12:00:00Z")
SHV_START = 110.30
# (first session, daily latent sigma in bp, daily drift in bp): moving, then flat, then accruing.
# REGIMES makes about 10% of SHV's 5-session development rows measured zeros; under B2 the
# band widens and over-covers the non-zero rows by more than Z2 allows (a failing case).
REGIMES = (("2019-01-02", 1.5, 0.0), ("2021-01-04", 0.25, 0.0), ("2022-07-01", 1.5, 1.6))
# MILD_REGIMES: a shorter flat stretch, about 2% zeros; every criterion passes (a passing case).
MILD_REGIMES = (("2019-01-02", 1.5, 0.0), ("2021-07-01", 0.3, 0.0), ("2022-01-03", 1.5, 1.6))


def tick_closes(
    sessions: pd.DatetimeIndex, seed: int = 3, regimes=REGIMES, start: float = SHV_START
) -> np.ndarray:
    """SHV-like closes on the $0.01 grid from a latent geometric random walk."""

    rng = np.random.RandomState(seed)
    sigma = np.zeros(len(sessions))
    drift = np.zeros(len(sessions))
    for first, sigma_bp, drift_bp in regimes:
        later = sessions >= pd.Timestamp(first, tz="UTC")
        sigma[later] = sigma_bp * 1e-4
        drift[later] = drift_bp * 1e-4
    steps = drift + sigma * rng.randn(len(sessions))
    steps[0] = 0.0
    latent = start * np.exp(np.cumsum(steps))
    return np.rint(latent * 100.0) / 100.0


def tick_batch(
    *,
    seed: int = 3,
    regimes=REGIMES,
    distributions: list[tuple[str, float]] | None = None,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
):
    """
    The prospective fixture batch with SHV's closes replaced by `tick_closes`
    and SHV's distributions replaced by `distributions` ((ex-date, amount as
    paid) pairs; none by default).
    """

    batch = synthetic.fixture_batch(
        ASSETS, start=START, end_exclusive=end_exclusive, retrieved_at=RETRIEVED
    )
    sessions = nyse.sessions(START, end_exclusive - pd.Timedelta(days=1))
    prices = batch.prices.copy()
    shv = prices["fund_id"] == "SHV"
    order = pd.DatetimeIndex(prices.loc[shv, "session_date"])
    closes = pd.Series(tick_closes(sessions, seed, regimes), index=sessions)
    prices.loc[shv, "close"] = closes.reindex(order).to_numpy()
    paid = batch.distributions.loc[batch.distributions["fund_id"] != "SHV"]
    if distributions:
        template = batch.distributions.iloc[[0]]
        extra = pd.concat(
            [
                template.assign(
                    fund_id="SHV",
                    ex_date=pd.Timestamp(day, tz="UTC").as_unit("us"),
                    amount=float(amount),
                )
                for day, amount in distributions
            ],
            ignore_index=True,
        )
        paid = pd.concat([paid, extra], ignore_index=True)
    paid = paid.reset_index(drop=True)
    return replace(batch, prices=prices.reset_index(drop=True), distributions=paid)
