"""
Synthetic fixtures for the prospective track: tests and demonstrations only,
never evidence about funds.

- `fixture_batch`: an F1 `RawBatch` of GARCH-like unadjusted closes with cash
  distributions (and VTI's 2008 2-for-1 split), under `FIXTURE_ADAPTER_ID`,
  whose data class is "synthetic". It is ingested with the ordinary F1
  writer, so training reads it through the ordinary fail-closed reader.
- `fixture_store` and `load_fixture_window`: sessions after 2024 on a
  weekday calendar, from the same generator, and a loader with the
  signature of the real prospective-store loader
  (`load_inference_window(store_root, assets, forecast_close, lookback=63)`);
  here `store_root` is the `FixtureStore` itself.

Prices follow P_t = (P_{t-1} (1 + r_t) - D_t) / f_t, so the approved F5.1
total return (f_t P_t + D_t) / P_{t-1} - 1 recovers r_t exactly; r_t is
GARCH(1,1) as in forecasting.etf_data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.adapter import RawBatch
from stock_agent.data.funds.schema import (
    DISTRIBUTIONS,
    PRICES,
    SPLITS,
    Instrument,
    dumps_canonical,
)

FIXTURE_ADAPTER_ID = "synthetic_prospective_fixture"
SOURCE_ID = "synthetic_prospective_fixture"
# daily vol, GARCH alpha and beta, daily drift, distribution every k months, yield per payment
PROFILES = {
    "VTI": (0.011, 0.08, 0.90, 3e-4, 3, 0.004),
    "AGG": (0.003, 0.04, 0.94, 1e-4, 1, 0.0025),
    "SHV": (0.0006, 0.03, 0.95, 1.5e-4, 1, 0.003),
}
SPLITS_BY_FUND = {"VTI": (pd.Timestamp("2008-06-18", tz="UTC"), 2.0)}
TRAINING_FIRST = pd.Timestamp("2003-01-02", tz="UTC")


def _garch(rng: np.random.RandomState, n: int, profile: tuple) -> np.ndarray:
    vol, alpha, beta, drift = profile[:4]
    shocks = rng.randn(n)
    omega = vol**2 * (1.0 - alpha - beta)
    variance, returns = vol**2, np.zeros(n)
    for t in range(n):
        if t:
            variance = omega + alpha * (returns[t - 1] - drift) ** 2 + beta * variance
        returns[t] = drift + np.sqrt(variance) * shocks[t]
    return returns


def _series(fund_id: str, sessions: pd.DatetimeIndex, seed: int, start_price: float = 100.0):
    """Closes, distributions and splits of one fund on `sessions`."""

    profile = PROFILES[fund_id]
    rng = np.random.RandomState(seed + sorted(PROFILES).index(fund_id))
    returns = _garch(rng, len(sessions), profile)
    every, cash_yield = profile[4], profile[5]
    first_of_month = ~pd.Series(sessions.year * 12 + sessions.month).duplicated().to_numpy()
    month_number = sessions.year * 12 + sessions.month
    paying = first_of_month & (month_number % every == 0)
    paying[0] = False
    split = SPLITS_BY_FUND.get(fund_id)
    closes, cash, ratios = np.zeros(len(sessions)), [], []
    previous = start_price
    for t, session in enumerate(sessions):
        factor = split[1] if split is not None and session == split[0] else 1.0
        amount = round(previous * cash_yield, 6) if paying[t] and factor == 1.0 else 0.0
        value = previous if t == 0 else (previous * (1.0 + returns[t]) - amount) / factor
        closes[t] = value
        if amount:
            cash.append((session, amount))
        if factor != 1.0:
            ratios.append((session, factor))
        previous = value
    return closes, cash, ratios


def _tables(fund_ids: Sequence[str], sessions: pd.DatetimeIndex, seed: int):
    prices, distributions, splits = [], [], []
    for fund_id in fund_ids:
        closes, cash, ratios = _series(fund_id, sessions, seed)
        prices.append(
            pd.DataFrame(
                {
                    "fund_id": fund_id,
                    "session_date": sessions,
                    "close": closes,
                    "adjustment_basis": "unadjusted",
                    "volume": pd.array([None] * len(sessions), dtype="Int64"),
                    "source_id": SOURCE_ID,
                }
            )
        )
        distributions += [
            {
                "fund_id": fund_id,
                "ex_date": day,
                "declaration_date": pd.NaT,
                "record_date": pd.NaT,
                "pay_date": pd.NaT,
                "amount": amount,
                "amount_basis": "per_share_as_paid",
                "distribution_type": "income_dividend",
                "source_sequence": 0,
                "source_id": SOURCE_ID,
            }
            for day, amount in cash
        ]
        splits += [
            {
                "fund_id": fund_id,
                "effective_date": day,
                "ratio_new_per_old": ratio,
                "source_id": SOURCE_ID,
            }
            for day, ratio in ratios
        ]
    names = {
        "prices": [n for n in PRICES.names if n != "batch_id"],
        "distributions": [n for n in DISTRIBUTIONS.names if n != "batch_id"],
        "splits": [n for n in SPLITS.names if n != "batch_id"],
    }
    price_frame = pd.concat(prices, ignore_index=True).loc[:, names["prices"]]
    dist_frame = pd.DataFrame(distributions, columns=names["distributions"])
    split_frame = pd.DataFrame(splits, columns=names["splits"])
    for frame, columns in (
        (price_frame, ("session_date",)),
        (dist_frame, ("ex_date", "declaration_date", "record_date", "pay_date")),
        (split_frame, ("effective_date",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
    dist_frame["source_sequence"] = dist_frame["source_sequence"].astype("int64")
    dist_frame["amount"] = dist_frame["amount"].astype("float64")
    split_frame["ratio_new_per_old"] = split_frame["ratio_new_per_old"].astype("float64")
    return price_frame, dist_frame, split_frame


def fixture_batch(
    fund_ids: Sequence[str],
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    retrieved_at: pd.Timestamp,
    seed: int = 0,
) -> RawBatch:
    """A synthetic F1 batch for [start, end_exclusive) on NYSE sessions (before 2025)."""

    ids = tuple(sorted(fund_ids))
    sessions = nyse.sessions(start, end_exclusive - pd.Timedelta(days=1))
    prices, distributions, splits = _tables(ids, sessions, seed)
    payload = {
        "fixture": FIXTURE_ADAPTER_ID,
        "seed": seed,
        "start": str(start.date()),
        "end_exclusive": str(end_exclusive.date()),
        "fund_ids": list(ids),
    }
    return RawBatch(
        adapter_id=FIXTURE_ADAPTER_ID,
        adapter_version="1",
        fund_ids=ids,
        start=start,
        end_exclusive=end_exclusive,
        instruments=tuple(Instrument(fund_id, fund_id, None) for fund_id in ids),
        retrieved_at=retrieved_at,
        raw_payload=dumps_canonical(payload).encode("utf-8"),
        terms_note="Synthetic fixture generated in this repository; no vendor terms apply.",
        vintage_note="Deterministic GARCH fixture v1: unadjusted closes, cash as paid.",
        prices=prices,
        distributions=distributions,
        splits=splits,
    )


@dataclass(frozen=True, eq=False)
class FixtureStore:
    """Post-2024 synthetic sessions (weekdays) standing in for the prospective store."""

    sessions: pd.DatetimeIndex
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame


def fixture_store(fund_ids: Sequence[str], *, first: str, last: str, seed: int = 7) -> FixtureStore:
    sessions = pd.bdate_range(first, last, tz="UTC").as_unit("us")
    prices, distributions, splits = _tables(tuple(sorted(fund_ids)), sessions, seed)
    return FixtureStore(sessions, prices, distributions, splits)


def truncated(store: FixtureStore, last: object) -> FixtureStore:
    """The same store as known on `last` (no session after it)."""

    stamp = (
        pd.Timestamp(last, tz="UTC") if pd.Timestamp(last).tzinfo is None else pd.Timestamp(last)
    )
    return FixtureStore(
        store.sessions[store.sessions <= stamp],
        store.prices.loc[store.prices["session_date"] <= stamp].reset_index(drop=True),
        store.distributions.loc[store.distributions["ex_date"] <= stamp].reset_index(drop=True),
        store.splits.loc[store.splits["effective_date"] <= stamp].reset_index(drop=True),
    )


def weekday_sessions(start: object, end: object) -> pd.DatetimeIndex:
    """The fixture store's exchange calendar: every weekday (no holidays)."""

    return pd.bdate_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), tz="UTC")


weekday_sessions.calendar_id = "synthetic_weekdays_v1"


def load_fixture_window(
    store_root: FixtureStore, assets: Sequence[str], forecast_close: object, lookback: int = 63
) -> dict[str, object]:
    """The fixture loader: the `lookback` sessions ending at `forecast_close`."""

    close = pd.Timestamp(forecast_close)
    close = close.tz_localize("UTC") if close.tzinfo is None else close.tz_convert("UTC")
    sessions = store_root.sessions[store_root.sessions <= close]
    if len(sessions) < lookback or sessions[-1] != close:
        raise ValueError(f"The fixture store has no {lookback}-session window ending {close}.")
    window = sessions[-lookback:]
    first = window[0]

    def inside(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        keep = frame["fund_id"].isin(assets) & (frame[column] >= first) & (frame[column] <= close)
        return frame.loc[keep].reset_index(drop=True)

    return {
        "sessions": window,
        "prices": inside(store_root.prices, "session_date"),
        "distributions": inside(store_root.distributions, "ex_date"),
        "splits": inside(store_root.splits, "effective_date"),
    }


__all__ = [
    "FIXTURE_ADAPTER_ID",
    "TRAINING_FIRST",
    "FixtureStore",
    "fixture_batch",
    "fixture_store",
    "load_fixture_window",
    "truncated",
    "weekday_sessions",
]
