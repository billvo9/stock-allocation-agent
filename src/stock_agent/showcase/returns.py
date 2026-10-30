"""
Stored bars to the return table the T5a engine reads, with the approved F5.1
total return (`portfolio.descriptive.total_returns`; OD8 group A, approved
2026-10-14):

    r_t = (f_t * P_t + D_t) / P_s - 1

on raw closes P, the vendor's split factor f (new shares per old share on the
split's ex-date) and the vendor-reported cash dividend D on its ex-date, s
being the previous session with a close. No other formula is used.

Labels the report carries with these returns:

- dividends are vendor-reported (Tiingo `divCash`), not checked against the
  issuers' records: a missing or wrong dividend changes the return, and a
  non-cash distribution (a spin-off) is not in the data at all;
- a dividend and a split on the same session have no recorded share basis,
  so that return is unavailable, never guessed (the engine then stops a
  portfolio holding the stock: `distribution_in_gap`);
- sessions the vendor did not serve stay gaps: the next close carries a
  labelled multi-session return (R5a), and wealth is unavailable across the
  gap while a held asset has no close.

The table's last column is the engine's funding leg (0.0 here; with a registered
cash account, `showcase.run.universe_data` sets it to the account's return, C4).
Every registered target holds 0% of it, so its entry never reaches a wealth path;
`tests/test_showcase_run.py` checks that the results do not change when it does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.portfolio.descriptive import (
    NO_CLOSE,
    NO_PRIOR_CLOSE,
    PER_SHARE_AS_PAID,
    UNADJUSTED,
    UNAVAILABLE,
    total_returns,
)

FUNDING_LEG = "funding_leg"
BENIGN = frozenset({NO_CLOSE, NO_PRIOR_CLOSE})  # gaps and first closes, not data defects
RETURN_LABEL = (
    "Total return from raw closes with vendor-reported cash dividends (Tiingo divCash, not "
    "issuer-verified) reinvested on the ex-date and vendor split factors (F5.1)"
)


@dataclass(frozen=True, eq=False)
class ReturnTable:
    """
    - `sessions`: the window's NYSE sessions (row t of the arrays).
    - `assets`: the tickers, then `FUNDING_LEG` (index `cash_index`).
    - `returns`, `has_close`: the engine's inputs (`portfolio.engine.simulate`).
    - `frames`: each ticker's `total_returns` rows (status and reason per session).
    - `unavailable`: per ticker, sessions with a close but no available return,
      by reason (gaps and first closes excluded).
    - `missing_sessions`: per ticker, window sessions without a close.
    """

    sessions: pd.DatetimeIndex
    assets: tuple[str, ...]
    returns: np.ndarray
    has_close: np.ndarray
    cash_index: int
    frames: Mapping[str, pd.DataFrame]
    unavailable: Mapping[str, dict[str, int]]
    missing_sessions: Mapping[str, int]


def window_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """The NYSE sessions of the window (the frozen calendar refuses dates after 2024)."""

    return nyse.sessions(start, end)


def ticker_returns(sessions: pd.DatetimeIndex, bars: pd.DataFrame) -> pd.DataFrame:
    """One ticker's F5.1 total returns on `sessions` from its stored bars."""

    frame = bars.copy()
    frame["session"] = pd.DatetimeIndex(frame["session"])
    outside = ~frame["session"].isin(sessions)
    if outside.any():
        raise ValueError(f"a bar on {frame.loc[outside, 'session'].iloc[0]} is not a session")
    closes = frame.loc[np.isfinite(frame["close"]), ["session", "close"]]
    distributions = frame.loc[frame["div_cash"] > 0, ["session", "div_cash"]].rename(
        columns={"session": "ex_date", "div_cash": "amount"}
    )
    splits = frame.loc[frame["split_factor"] != 1.0, ["session", "split_factor"]].rename(
        columns={"session": "effective_date", "split_factor": "ratio"}
    )
    return total_returns(
        sessions,
        closes.set_index("session")["close"],
        distributions,
        splits,
        close_basis=UNADJUSTED,
        amount_basis=PER_SHARE_AS_PAID,
    )


def return_table(sessions: pd.DatetimeIndex, bars: Mapping[str, pd.DataFrame]) -> ReturnTable:
    """The engine's return table for `bars` (ticker to stored bars), in their order."""

    tickers: Sequence[str] = tuple(bars)
    if len(tickers) < 2:
        raise ValueError("a universe needs at least two tickers")
    n = len(tickers)
    returns = np.zeros((len(sessions), n + 1))
    has_close = np.ones((len(sessions), n + 1), dtype=bool)
    frames, unavailable, missing = {}, {}, {}
    for column, ticker in enumerate(tickers):
        frame = ticker_returns(sessions, bars[ticker])
        frames[ticker] = frame
        returns[:, column] = frame["total_return"].to_numpy(dtype="float64")
        has_close[:, column] = frame["close_observed"].to_numpy(dtype=bool)
        defective = (
            frame["close_observed"]
            & frame["status"].eq(UNAVAILABLE)
            & ~frame["reason"].isin(list(BENIGN))
        )
        reasons = Counter(str(reason) for reason in frame.loc[defective, "reason"])
        unavailable[ticker] = dict(sorted(reasons.items()))
        missing[ticker] = int((~frame["close_observed"]).sum())
    return ReturnTable(
        sessions=sessions,
        assets=(*tickers, FUNDING_LEG),
        returns=returns,
        has_close=has_close,
        cash_index=n,
        frames=frames,
        unavailable=unavailable,
        missing_sessions=missing,
    )
