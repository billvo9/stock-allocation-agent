from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _validate_returns(
    returns: pd.Series,
    periods_per_year: int,
) -> None:
    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    if returns.isna().any():
        raise ValueError("Returns cannot contain missing values.")

    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns must contain only finite values.")


def calculate_cagr(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    _validate_returns(
        returns,
        periods_per_year,
    )

    cumulative_growth = float((1.0 + returns).prod())

    return cumulative_growth ** (periods_per_year / len(returns)) - 1


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    _validate_returns(
        returns,
        periods_per_year,
    )

    if len(returns) < 2:
        return float("nan")

    periodic_volatility = float(returns.std(ddof=1))

    return periodic_volatility * math.sqrt(periods_per_year)


def _annual_to_periodic_rate(
    annual_rate: float,
    periods_per_year: int,
) -> float:
    if annual_rate <= -1.0:
        raise ValueError("Annual rate must be greater than -100%.")

    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def calculate_sharpe_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    _validate_returns(
        returns,
        periods_per_year,
    )

    if len(returns) < 2:
        return float("nan")

    periodic_rf = _annual_to_periodic_rate(
        annual_risk_free_rate,
        periods_per_year,
    )

    excess_returns = returns - periodic_rf

    excess_volatility = float(excess_returns.std(ddof=1))

    if excess_volatility == 0.0:
        return np.nan

    return math.sqrt(periods_per_year) * (excess_returns.mean()) / excess_volatility


def calculate_sortino_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    _validate_returns(
        returns,
        periods_per_year,
    )

    periodic_rf = _annual_to_periodic_rate(
        annual_risk_free_rate,
        periods_per_year,
    )

    excess_returns = returns - periodic_rf

    downside_returns = np.minimum(
        excess_returns.to_numpy(dtype=float),
        0.0,
    )

    periodic_downside_deviation = float(np.sqrt(np.mean(downside_returns**2)))

    if np.isclose(
        periodic_downside_deviation,
        0.0,
    ):
        return float("nan")

    annualized_downside_deviation = periodic_downside_deviation * math.sqrt(periods_per_year)

    annualized_excess_return = float(excess_returns.mean()) * periods_per_year

    return annualized_excess_return / annualized_downside_deviation


def _calculate_max_drawdown_from_returns(
    returns: pd.Series,
) -> float:
    wealth_index = (1.0 + returns).cumprod()

    running_peak = wealth_index.cummax()

    drawdowns = wealth_index / running_peak - 1.0

    return abs(float(drawdowns.min()))


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    _validate_returns(
        returns,
        periods_per_year,
    )

    cagr = calculate_cagr(
        returns,
        periods_per_year,
    )

    max_drawdown = _calculate_max_drawdown_from_returns(returns)

    if max_drawdown == 0.0:
        return np.nan

    return cagr / max_drawdown
