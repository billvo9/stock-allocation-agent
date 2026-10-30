"""
The cash account's return on real data (OD8 group C: C1, with C9's booking rule; C3 makes
it the risk-free rate; C4 makes it the engine's cash column for real data). Approved by
the owner on 2026-10-26: "Approve all C1–C9 (Recommended)".

C1 (option a). Over each span between consecutive closes (t-1, t], the account earns
the DGS3MO investment-basis yield for the span's start date, accrued simply on actual
calendar days, split at each January 1 (365, or 366 in a leap year):

    r_cash,t = y_{s(t-1)} x sum_j days_j / B_j

computed by the existing `transform._simple_interest_accrual` (no new arithmetic). The
yield y_{s(t-1)} is the first release (`vintages.first_release`) of the latest DGS3MO
date on or before t-1, so a later revision never rewrites booked history.

C9 (option a), an explicit, scoped exception to the derived-value rule: booked wealth
uses y_{t-1} at t, although as a decision input (C2) that value counts only later. It
covers booked account state only; a field carrying the rate itself stays under C2 and
is read with `vintages.as_of`.

Carry and staleness (C1's edge cases): a span start without a published value (a
bond-market holiday on an NYSE session, such as Columbus Day, or any other day without
one) uses the latest earlier value, with reason `bond_market_holiday_carry`; a value
more than `STALE_AFTER` weekdays old (holidays counted, so it flags sooner) is flagged
`stale_rate`. A rate from ALFRED's first vintage for a date before 2005-06-28 is
flagged `pre_alfred` (it may include revisions made before then; C2.3). No value on or
before a span's start refuses the series.

Labels (C1 / F3.4, and FRED's terms): `CASH_LABEL`, `FRED_NOTICE`, `FRED_CITATION`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.data.rates.transform import _simple_interest_accrual
from stock_agent.data.rates.vintages import PRE_ALFRED, first_release

CASH_LABEL = "Cash: 3-month Treasury bill rate approximation (FRED DGS3MO), before fees"
FRED_NOTICE = (
    "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve "
    "Bank of St. Louis."
)
FRED_CITATION = (
    "Board of Governors of the Federal Reserve System (US), 3-Month Treasury Constant Maturity "
    "Rate [DGS3MO], retrieved from FRED, Federal Reserve Bank of St. Louis."
)
CARRY = "bond_market_holiday_carry"
STALE = "stale_rate"
STALE_AFTER = 5  # business days


class CashUnavailable(ValueError):
    """The cash series cannot be built for these sessions (no rate on or before a span)."""


def cash_returns(vintages: pd.DataFrame, sessions: pd.DatetimeIndex, rate_id: str) -> pd.DataFrame:
    """
    One row per span (sessions[t-1], sessions[t]]: `session` (t), `accrual` (r_cash,t),
    `annual_yield`, `rate_date` (the first-release date used), `reason` (None or
    `bond_market_holiday_carry`) and `stale` (more than STALE_AFTER business days old).
    """

    if len(sessions) < 2:
        raise CashUnavailable("at least two sessions are needed")
    days = pd.DatetimeIndex(sessions).tz_localize(None).normalize().as_unit("ns")
    if days.has_duplicates or not days.is_monotonic_increasing:
        raise CashUnavailable("sessions must be strictly increasing")
    first = first_release(vintages)
    first = first.loc[first["rate_id"] == rate_id]
    if first.empty:
        raise CashUnavailable(f"no {rate_id} vintages")
    values = first.loc[
        first["annual_yield"].notna(), ["date", "annual_yield", "availability_rule"]
    ].copy()
    values["date"] = pd.DatetimeIndex(values["date"]).tz_localize(None).normalize().as_unit("ns")
    values = values.sort_values("date").rename(columns={"date": "rate_date"})
    spans = pd.DataFrame({"start": days[:-1], "session": days[1:]})
    aligned = pd.merge_asof(
        spans, values, left_on="start", right_on="rate_date", direction="backward"
    )
    if aligned["annual_yield"].isna().any():
        missing = aligned.loc[aligned["annual_yield"].isna(), "start"].iloc[0]
        raise CashUnavailable(f"no {rate_id} value on or before {missing.date()}")
    carried = aligned["rate_date"] < aligned["start"]
    age = np.busday_count(
        aligned["rate_date"].to_numpy(dtype="datetime64[D]"),
        aligned["start"].to_numpy(dtype="datetime64[D]"),
    )
    accrual = [
        _simple_interest_accrual(float(y), start, end)
        for y, start, end in zip(
            aligned["annual_yield"], aligned["start"], aligned["session"], strict=True
        )
    ]
    return pd.DataFrame(
        {
            "session": pd.DatetimeIndex(sessions[1:]),
            "accrual": np.array(accrual, dtype="float64"),
            "annual_yield": aligned["annual_yield"].to_numpy(dtype="float64"),
            "rate_date": aligned["rate_date"].to_numpy(),
            "reason": pd.Series([CARRY if c else None for c in carried], dtype=object),
            "stale": age > STALE_AFTER,
            # C2.3: a rate dated before ALFRED's first vintage may include revisions made
            # before 2005-06-28; flagged, as the packet requires for booked accrual.
            "pre_alfred": (aligned["availability_rule"] == PRE_ALFRED).to_numpy(),
        }
    )
