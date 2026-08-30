from __future__ import annotations

import calendar
import math

import pandas as pd

from stock_agent.data.rates.schema import (
    validate_rate_frame,
)


def percent_yield_to_decimal(
    yields: pd.Series,
) -> pd.Series:
    """Convert provider percentage yields to decimal yields."""

    return yields.astype(float) / 100.0


def _simple_interest_accrual(
    annual_yield: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Convert an annual yield into a holding-period simple return.

    Accrual uses actual calendar days and accounts for leap years.
    A period crossing a calendar-year boundary is split so each
    segment uses the correct 365- or 366-day denominator.
    """

    if not math.isfinite(annual_yield):
        raise ValueError("Annual yield must be finite.")

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if end <= start:
        raise ValueError("Holding-period end date must be after start date.")

    accrual = 0.0
    current = start

    while current < end:
        next_year = pd.Timestamp(
            year=current.year + 1,
            month=1,
            day=1,
        )

        segment_end = min(
            end,
            next_year,
        )

        elapsed_days = (segment_end - current).days

        days_in_year = 366.0 if calendar.isleap(current.year) else 365.0

        accrual += annual_yield * elapsed_days / days_in_year

        current = segment_end

    return float(accrual)


def prepare_risk_free_returns(
    rates: pd.DataFrame,
    portfolio_dates: pd.DatetimeIndex,
    rate_id: str,
) -> pd.Series:
    """Build holding-period risk-free returns for a backtest timeline.

    `portfolio_dates` must contain the complete backtest timeline,
    including the initial portfolio date.

    For each holding period:
        portfolio_dates[t - 1] -> portfolio_dates[t]

    the latest rate observation available on or before the period
    start is used.

    The returned Series is indexed by portfolio_dates[1:], matching
    the dates of realized portfolio returns.
    """

    if not isinstance(
        portfolio_dates,
        pd.DatetimeIndex,
    ):
        raise TypeError("portfolio_dates must be a pandas DatetimeIndex.")

    if len(portfolio_dates) < 2:
        raise ValueError("At least two portfolio dates are required.")

    if not rate_id:
        raise ValueError("rate_id cannot be empty.")

    validate_rate_frame(rates)

    dates = portfolio_dates.normalize()

    if dates.has_duplicates:
        raise ValueError("Portfolio dates contain duplicates.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Portfolio dates must be strictly increasing.")

    selected_rates = rates[rates["rate_id"] == rate_id].copy()

    if selected_rates.empty:
        raise ValueError(f"No rate data found for rate_id={rate_id!r}.")

    quote_conventions = set(selected_rates["quote_convention"].dropna())

    if quote_conventions != {"investment_basis"}:
        raise ValueError(
            "prepare_risk_free_returns currently supports only investment_basis rates."
        )

    selected_rates["date"] = pd.to_datetime(
        selected_rates["date"],
        errors="raise",
    ).dt.normalize()

    selected_rates = (
        selected_rates[
            [
                "date",
                "annual_yield",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    period_starts = dates[:-1]
    period_ends = dates[1:]

    periods = pd.DataFrame(
        {
            "period_start": period_starts,
            "period_end": period_ends,
        }
    )

    aligned = pd.merge_asof(
        periods,
        selected_rates,
        left_on="period_start",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )

    if aligned["annual_yield"].isna().any():
        first_missing = aligned.loc[
            aligned["annual_yield"].isna(),
            "period_start",
        ].iloc[0]

        raise ValueError(
            f"No risk-free rate available on or before portfolio start {first_missing.date()}."
        )

    holding_period_returns = [
        _simple_interest_accrual(
            annual_yield=float(annual_yield),
            start=period_start,
            end=period_end,
        )
        for (
            annual_yield,
            period_start,
            period_end,
        ) in zip(
            aligned["annual_yield"],
            aligned["period_start"],
            aligned["period_end"],
            strict=True,
        )
    ]

    risk_free_returns = pd.Series(
        data=holding_period_returns,
        index=period_ends,
        name="risk_free_return",
        dtype=float,
    )

    expected_index = dates[1:]

    if not risk_free_returns.index.equals(expected_index):
        raise RuntimeError("Risk-free return dates do not match portfolio return dates.")

    return risk_free_returns
