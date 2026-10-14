"""
Synthetic demo fund histories for the Explore page, until publications for
real funds exist.

Synthetic demo data, not real fund performance. The generator is pure: it
reads no file, network or clock, and the same seed always gives the same
records (a legacy `numpy.random.RandomState` stream, stable across numpy
versions).

Funds (weekday sessions, 2007-01-01 to 2024-12-31, dated 00:00 UTC):
- "Synthetic equity fund", with a five-session data gap (2015-08-17 to
  2015-08-21; status `unavailable`, reason `missing_input`);
- "Synthetic bond fund";
- "Synthetic cash fund";
- "Synthetic bond fund, stale copy": the bond fund's records cut at
  2023-06-30 and marked stale;
- "Synthetic international equity fund": every record unavailable.

Daily returns are normal draws. Stress regimes in 2007-2009 and 2022 exist
only so the period presets have something to show; their sizes are invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.app.history import OK, UNAVAILABLE, FundHistory

SEED = 20261013
GENERATOR_VERSION = "1"
CALENDAR_START = "2007-01-01"
CALENDAR_END = "2024-12-31"
GAP_START = "2015-08-17"
GAP_END = "2015-08-21"
STALE_END = "2023-06-30"
GAP_REASON = "missing_input"
SOURCE_LABEL = "Synthetic demo data, not real fund performance"
LIMITATIONS = ("synthetic_data", "synthetic_calendar")

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


def _sessions() -> pd.DatetimeIndex:
    return pd.bdate_range(CALENDAR_START, CALENDAR_END, tz="UTC")


def _returns(rng: np.random.RandomState, dates: pd.DatetimeIndex, role: str) -> np.ndarray:
    mean, sd = _BASE[role]
    means = np.full(len(dates), mean)
    sds = np.full(len(dates), sd)
    for first, last, stress_mean, stress_sd in _STRESS[role]:
        inside = (dates >= pd.Timestamp(first, tz="UTC")) & (dates <= pd.Timestamp(last, tz="UTC"))
        means[inside] = stress_mean
        sds[inside] = stress_sd
    return means + sds * rng.randn(len(dates))


def _records(dates: pd.DatetimeIndex, returns: np.ndarray, missing: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "daily_total_return": np.where(missing, np.nan, returns),
            "status": np.where(missing, UNAVAILABLE, OK),
            "reason": np.where(missing, GAP_REASON, None),
        }
    )


def _fund(
    fund_id: str, name: str, role: str, records: pd.DataFrame, seed: int, **stale
) -> FundHistory:
    return FundHistory(
        fund_id=fund_id,
        name=name,
        role=role,
        records=records,
        source_label=SOURCE_LABEL,
        retrieved_label="Not retrieved: generated in memory by the demo generator",
        vintage_note=f"Synthetic generator version {GENERATOR_VERSION}, seed {seed}",
        limitations=LIMITATIONS,
        **stale,
    )


def demo_histories(seed: int = SEED) -> tuple[FundHistory, ...]:
    """The five synthetic demo funds, in display order."""

    rng = np.random.RandomState(seed)
    dates = _sessions()
    equity = _returns(rng, dates, "equity")
    bonds = _returns(rng, dates, "bonds")
    cash = _returns(rng, dates, "cash proxy")
    none = np.zeros(len(dates), dtype=bool)
    gap = (dates >= pd.Timestamp(GAP_START, tz="UTC")) & (dates <= pd.Timestamp(GAP_END, tz="UTC"))
    kept = dates <= pd.Timestamp(STALE_END, tz="UTC")
    stale_reason = (
        f"Not refreshed: its records end {STALE_END}, while the other demo funds run to "
        f"{CALENDAR_END}."
    )
    return (
        _fund(
            "synthetic-equity",
            "Synthetic equity fund",
            "equity",
            _records(dates, equity, gap),
            seed,
        ),
        _fund("synthetic-bond", "Synthetic bond fund", "bonds", _records(dates, bonds, none), seed),
        _fund(
            "synthetic-cash", "Synthetic cash fund", "cash proxy", _records(dates, cash, none), seed
        ),
        _fund(
            "synthetic-bond-stale",
            "Synthetic bond fund, stale copy",
            "bonds",
            _records(dates[kept], bonds[kept], none[kept]),
            seed,
            stale=True,
            stale_reason=stale_reason,
        ),
        _fund(
            "synthetic-international-equity",
            "Synthetic international equity fund",
            "equity",
            _records(dates, np.full(len(dates), np.nan), ~none),
            seed,
        ),
    )
