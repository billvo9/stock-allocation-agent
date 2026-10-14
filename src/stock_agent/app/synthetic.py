"""
Synthetic demo fund histories for the Explore page, until publications for
real funds exist.

Synthetic demo data, not real fund performance. The generator is pure: it
reads no file, network or clock, and the same seed always gives the same
records (a legacy `numpy.random.RandomState` stream, stable across numpy
versions).

Each fund is built the way real records will be: a raw closing-price path
with dated cash distributions (and, for the equity fund, a 2-for-1 split),
turned into total returns by the approved group A calculation
(`stock_agent.portfolio.descriptive.total_returns`). Sessions are weekdays,
2007-01-01 to 2024-12-31, dated 00:00 UTC.

Funds:
- "Synthetic equity fund": quarterly distributions, a 2-for-1 split on
  2014-06-09, and five sessions without a close (2015-08-17 to 2015-08-21),
  so the next return is a labelled six-session return;
- "Synthetic bond fund" and "Synthetic cash fund": monthly distributions;
- "Synthetic bond fund, stale copy": the bond fund cut at 2023-06-30;
- "Synthetic bond fund, gap over a distribution": the bond fund without
  closes from 2019-03-01 to 2019-03-07, across an ex-date, so its growth
  cannot be continued past the gap;
- "Synthetic international equity fund": no close at all.

Stress regimes in 2007-2009 and 2022 exist only so the period presets have
something to show; their sizes are invented.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from stock_agent.app.history import FundHistory
from stock_agent.portfolio.descriptive import PER_SHARE_AS_PAID, UNADJUSTED
from stock_agent.portfolio.descriptive import total_returns as _total_returns

SEED = 20261015  # chosen so the demo paths look typical; they carry no evidence
GENERATOR_VERSION = "2"
CALENDAR_START = "2007-01-01"
CALENDAR_END = "2024-12-31"
GAP_START = "2015-08-17"
GAP_END = "2015-08-21"
DISTRIBUTION_GAP_START = "2019-03-01"
DISTRIBUTION_GAP_END = "2019-03-07"
SPLIT_DATE = "2014-06-09"
SPLIT_RATIO = 2.0
STALE_END = "2023-06-30"
SOURCE_LABEL = "Synthetic demo data, not real fund performance"
LIMITATIONS = ("synthetic_data", "synthetic_calendar")
START_PRICE = 100.0

# Per role: (daily mean, daily standard deviation), then stress regimes
# (first date, last date, daily mean, daily standard deviation).
_BASE = {
    "equity": (0.00035, 0.011),
    "bonds": (0.00014, 0.0028),
    "cash proxy": (0.00006, 0.000005),
}
_STRESS = {
    "equity": (
        ("2007-10-10", "2009-03-09", -0.0012, 0.022),
        ("2022-01-04", "2022-10-12", -0.0011, 0.016),
    ),
    "bonds": (("2022-01-03", "2022-10-24", -0.0006, 0.005),),
    "cash proxy": (),
}
# Distribution months and yield per distribution (a fraction of the prior close).
_DISTRIBUTIONS = {
    "equity": ((3, 6, 9, 12), 0.0045),
    "bonds": (tuple(range(1, 13)), 0.0025),
    "cash proxy": (tuple(range(1, 13)), 0.0012),
}


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def sessions() -> pd.DatetimeIndex:
    return pd.bdate_range(CALENDAR_START, CALENDAR_END, tz="UTC")


def _total_return_draws(
    rng: np.random.RandomState, dates: pd.DatetimeIndex, role: str
) -> np.ndarray:
    mean, sd = _BASE[role]
    means = np.full(len(dates), mean)
    sds = np.full(len(dates), sd)
    for first, last, stress_mean, stress_sd in _STRESS[role]:
        inside = (dates >= utc(first)) & (dates <= utc(last))
        means[inside] = stress_mean
        sds[inside] = stress_sd
    return means + sds * rng.randn(len(dates))


def _ex_dates(dates: pd.DatetimeIndex, months: tuple[int, ...]) -> pd.DatetimeIndex:
    """The first session of each distribution month."""

    frame = pd.Series(dates, index=dates)
    firsts = frame.groupby([dates.year, dates.month]).min()
    return pd.DatetimeIndex([d for d in firsts if d.month in months and d != dates[0]])


def _path(
    dates: pd.DatetimeIndex, draws: np.ndarray, role: str, split: bool
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Raw closes, distributions and splits whose total returns are `draws`."""

    months, yield_ = _DISTRIBUTIONS[role]
    ex_dates = set(_ex_dates(dates, months))
    split_date = utc(SPLIT_DATE) if split else None
    closes = np.empty(len(dates))
    closes[0] = START_PRICE
    paid: list[tuple[pd.Timestamp, float]] = []
    for position in range(1, len(dates)):
        previous = closes[position - 1]
        cash = round(yield_ * previous, 4) if dates[position] in ex_dates else 0.0
        value = previous * (1.0 + draws[position]) - cash
        if dates[position] == split_date:
            value /= SPLIT_RATIO
        closes[position] = value
        if cash:
            paid.append((dates[position], cash))
    distributions = pd.DataFrame(paid, columns=["ex_date", "amount"])
    splits = pd.DataFrame(
        [(split_date, SPLIT_RATIO)] if split else [], columns=["effective_date", "ratio"]
    )
    return pd.Series(closes, index=dates), distributions, splits


def total_returns(sessions, closes, distributions=None, splits=None):
    """The approved calculation, stating the synthetic inputs' basis: raw closes, as-paid amounts."""

    return _total_returns(
        sessions,
        closes,
        distributions,
        splits,
        close_basis=UNADJUSTED,
        amount_basis=PER_SHARE_AS_PAID,
    )


def _fund(fund_id: str, name: str, role: str, records: pd.DataFrame, **stale) -> FundHistory:
    return FundHistory(
        fund_id=fund_id,
        name=name,
        role=role,
        records=records,
        calendar=sessions(),
        source_label=SOURCE_LABEL,
        retrieved_label="Not retrieved: generated in memory by the demo generator",
        vintage_note=f"Synthetic generator version {GENERATOR_VERSION}, seed {SEED}",
        limitations=LIMITATIONS,
        **stale,
    )


def _without(closes: pd.Series, first: str, last: str) -> pd.Series:
    return closes[(closes.index < utc(first)) | (closes.index > utc(last))]


@lru_cache(maxsize=2)
def demo_histories(seed: int = SEED) -> tuple[FundHistory, ...]:
    """The six synthetic demo funds, in display order."""

    rng = np.random.RandomState(seed)
    dates = sessions()
    equity = _path(dates, _total_return_draws(rng, dates, "equity"), "equity", split=True)
    bonds = _path(dates, _total_return_draws(rng, dates, "bonds"), "bonds", split=False)
    cash = _path(dates, _total_return_draws(rng, dates, "cash proxy"), "cash proxy", split=False)
    kept = dates[dates <= utc(STALE_END)]
    stale_reason = (
        f"Not refreshed: its records end {STALE_END}, while the other demo funds run to "
        f"{CALENDAR_END}."
    )
    equity_closes, equity_distributions, equity_splits = equity
    bond_closes, bond_distributions, _ = bonds
    cash_closes, cash_distributions, _ = cash
    return (
        _fund(
            "synthetic-equity",
            "Synthetic equity fund",
            "equity",
            total_returns(
                dates,
                _without(equity_closes, GAP_START, GAP_END),
                equity_distributions,
                equity_splits,
            ),
        ),
        _fund(
            "synthetic-bond",
            "Synthetic bond fund",
            "bonds",
            total_returns(dates, bond_closes, bond_distributions),
        ),
        _fund(
            "synthetic-cash",
            "Synthetic cash fund",
            "cash proxy",
            total_returns(dates, cash_closes, cash_distributions),
        ),
        _fund(
            "synthetic-bond-stale",
            "Synthetic bond fund, stale copy",
            "bonds",
            total_returns(
                kept,
                bond_closes[bond_closes.index <= utc(STALE_END)],
                bond_distributions[bond_distributions["ex_date"] <= utc(STALE_END)],
            ),
            stale=True,
            stale_reason=stale_reason,
        ),
        _fund(
            "synthetic-bond-distribution-gap",
            "Synthetic bond fund, gap over a distribution",
            "bonds",
            total_returns(
                dates,
                _without(bond_closes, DISTRIBUTION_GAP_START, DISTRIBUTION_GAP_END),
                bond_distributions,
            ),
        ),
        _fund(
            "synthetic-international-equity",
            "Synthetic international equity fund",
            "equity",
            total_returns(dates, pd.Series(dtype="float64")),
        ),
    )
