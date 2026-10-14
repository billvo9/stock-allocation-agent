"""
Synthetic ETF-like series, their features and their two targets.

Assets differ on purpose (no single model or target fits them all):

    asset        daily vol   GARCH (alpha, beta)   drift     trend term
    EQ_US        1.1%        0.08, 0.90            3e-4      yes
    EQ_INTL      1.2%        0.07, 0.91            2e-4      yes
    BOND_AGG     0.3%        0.04, 0.94            1e-4      no
    BOND_SHORT   0.1%        0.03, 0.95            8e-5      no
    COMMODITY    1.4%        0.06, 0.92            0         yes
    REIT         1.4%        0.09, 0.89            2e-4      no

Daily log return r_t = mu_t + sqrt(h_t) e_t, e_t ~ N(0, 1):

    volatility = "garch"     h_t = omega + alpha (r_{t-1} - mu_{t-1})^2 + beta h_{t-1},
                             omega = sigma^2 (1 - alpha - beta) (unconditional sigma^2)
    volatility = "constant"  h_t = sigma^2
    return_signal = "trend"  mu_t = drift + s * sigma * tanh(m_{t-1}) for trend assets,
                             m_{t-1} = the 60-session log return to t-1 over
                             (sigma sqrt 60), known before day t
    return_signal = "none"   mu_t = drift

Features (session_close; every value uses the asset's own returns through
the close of the row date):

    tsmom_20d, tsmom_60d     trailing log return / (sigma_60 sqrt(k)): trend
                             measured against the asset's own risk
                             (relative_technical, not ranked across assets)
    log_rv_5d, _20d, _60d    log realized volatility, sqrt(mean r^2) over k
                             sessions (risk_liquidity; HAR-style inputs)

Targets (entry `entry_lag` sessions after the row date, exit `horizon`
sessions after entry; known after the exit close, like every label):

    forward_return           close(exit) / close(entry) - 1
    log_realized_volatility  log sqrt(mean of r_k^2 for entry < k <= exit)

Synthetic only: verifies the pipeline; never evidence about funds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stock_agent.features.contracts import SESSION_CLOSE, FeatureContract, contract_index
from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    add_forward_return_target,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

ASSETS = {
    "EQ_US": {"vol": 0.011, "alpha": 0.08, "beta": 0.90, "drift": 3e-4, "trend": True},
    "EQ_INTL": {"vol": 0.012, "alpha": 0.07, "beta": 0.91, "drift": 2e-4, "trend": True},
    "BOND_AGG": {"vol": 0.003, "alpha": 0.04, "beta": 0.94, "drift": 1e-4, "trend": False},
    "BOND_SHORT": {"vol": 0.001, "alpha": 0.03, "beta": 0.95, "drift": 8e-5, "trend": False},
    "COMMODITY": {"vol": 0.014, "alpha": 0.06, "beta": 0.92, "drift": 0.0, "trend": True},
    "REIT": {"vol": 0.014, "alpha": 0.09, "beta": 0.89, "drift": 2e-4, "trend": False},
}
RETURN_SIGNALS = ("none", "trend")
VOLATILITY = ("garch", "constant")
TARGETS = ("forward_return", "log_realized_volatility")
TARGET_SHORT = {"forward_return": "ret", "log_realized_volatility": "vol"}


@dataclass(frozen=True)
class SyntheticEtfSpec:
    n_sessions: int = 2000
    start: str = "2010-01-04"
    return_signal: str = "trend"
    signal_strength: float = 0.05
    volatility: str = "garch"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.return_signal not in RETURN_SIGNALS:
            raise ValueError(f"return_signal must be one of {RETURN_SIGNALS}.")
        if self.volatility not in VOLATILITY:
            raise ValueError(f"volatility must be one of {VOLATILITY}.")
        if self.n_sessions < 400:
            raise ValueError("A synthetic ETF panel needs at least 400 sessions.")

    def describe(self) -> dict[str, object]:
        return {**asdict(self), "assets": ASSETS}


@dataclass(frozen=True, eq=False)
class SyntheticEtfPanel:
    prices: pd.DataFrame  # date, symbol, close
    truth: pd.DataFrame  # date, symbol, drift, conditional_vol
    spec: SyntheticEtfSpec
    survivorship_free: bool = True  # a fixed synthetic list; nothing is dropped


def synthetic_etf_panel(spec: SyntheticEtfSpec) -> SyntheticEtfPanel:
    calendar = pd.bdate_range(spec.start, periods=spec.n_sessions, tz="UTC")
    if calendar[-1] >= MODEL_HOLDOUT_START:
        raise ValueError("Synthetic data must end before the 2025-01-01 lockbox.")
    rng = np.random.RandomState(spec.seed)
    n = spec.n_sessions
    frames, truths = [], []
    for symbol, profile in ASSETS.items():
        shocks = rng.randn(n)
        sigma2 = profile["vol"] ** 2
        alpha, beta = (profile["alpha"], profile["beta"])
        if spec.volatility == "constant":
            alpha, beta = 0.0, 0.0
        omega = sigma2 * (1.0 - alpha - beta)
        trend = spec.return_signal == "trend" and profile["trend"]
        returns = np.zeros(n)
        variance = np.full(n, sigma2)
        drift = np.full(n, profile["drift"])
        for t in range(n):
            if t > 0:
                surprise = returns[t - 1] - drift[t - 1]
                variance[t] = omega + alpha * surprise**2 + beta * variance[t - 1]
            if trend and t > 60:
                signal = returns[t - 60 : t].sum() / (profile["vol"] * np.sqrt(60.0))
                drift[t] = profile["drift"] + spec.signal_strength * profile["vol"] * np.tanh(
                    signal
                )
            returns[t] = drift[t] + np.sqrt(variance[t]) * shocks[t]
        close = 100.0 * np.exp(np.cumsum(returns))
        frames.append(pd.DataFrame({"date": calendar, "symbol": symbol, "close": close}))
        truths.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "symbol": symbol,
                    "drift": drift,
                    "conditional_vol": np.sqrt(variance),
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    truth = pd.concat(truths, ignore_index=True)
    for frame in (prices, truth):
        frame["date"] = frame["date"].dt.as_unit("ns")
    order = ["date", "symbol"]
    return SyntheticEtfPanel(
        prices=prices.sort_values(order).reset_index(drop=True),
        truth=truth.sort_values(order).reset_index(drop=True),
        spec=spec,
    )


CONTRACTS = (
    FeatureContract(
        "tsmom_20d",
        "relative_technical",
        SESSION_CLOSE,
        60,
        "impute_training_median",
        ("close",),
        "20-session log return over (60-session daily vol x sqrt 20): trend against own risk",
    ),
    FeatureContract(
        "tsmom_60d",
        "relative_technical",
        SESSION_CLOSE,
        60,
        "impute_training_median",
        ("close",),
        "60-session log return over (60-session daily vol x sqrt 60)",
    ),
    FeatureContract(
        "log_rv_5d",
        "risk_liquidity",
        SESSION_CLOSE,
        5,
        "impute_training_median",
        ("close",),
        "log sqrt(mean squared daily log return), last 5 sessions",
    ),
    FeatureContract(
        "log_rv_20d",
        "risk_liquidity",
        SESSION_CLOSE,
        20,
        "impute_training_median",
        ("close",),
        "log sqrt(mean squared daily log return), last 20 sessions",
    ),
    FeatureContract(
        "log_rv_60d",
        "risk_liquidity",
        SESSION_CLOSE,
        60,
        "impute_training_median",
        ("close",),
        "log sqrt(mean squared daily log return), last 60 sessions",
    ),
)
CONTRACTS_BY_NAME = contract_index(CONTRACTS)
PERSISTENCE_FEATURE = "log_rv_20d"  # the random-walk volatility forecast


def etf_features(prices: pd.DataFrame) -> pd.DataFrame:
    """date, symbol, history_sessions and every ETF feature (per asset, backward-looking)."""

    frame = prices.loc[:, ["date", "symbol", "close"]].copy()
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    by_symbol = frame.groupby("symbol", sort=False)
    frame["history_sessions"] = by_symbol.cumcount() + 1
    log_return = np.log(frame["close"] / by_symbol["close"].shift(1))
    grouped = log_return.groupby(frame["symbol"], sort=False)
    squared = (log_return**2).groupby(frame["symbol"], sort=False)
    vol_60 = np.sqrt(squared.rolling(60, min_periods=60).mean().to_numpy())
    for k in (5, 20, 60):
        mean_square = squared.rolling(k, min_periods=k).mean().to_numpy()
        with np.errstate(divide="ignore"):
            frame[f"log_rv_{k}d"] = np.where(mean_square > 0, 0.5 * np.log(mean_square), np.nan)
    for k in (20, 60):
        trailing = grouped.rolling(k, min_periods=k).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            frame[f"tsmom_{k}d"] = np.where(vol_60 > 0, trailing / (vol_60 * np.sqrt(k)), np.nan)
    columns = ["date", "symbol", "history_sessions", *CONTRACTS_BY_NAME]
    return frame.loc[:, columns].sort_values(["date", "symbol"]).reset_index(drop=True)


def etf_labels(prices: pd.DataFrame, *, horizon: int, entry_lag: int, target: str) -> pd.DataFrame:
    """date, symbol and the label columns for one target (raw, per asset)."""

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}.")
    labeled = add_forward_return_target(
        prices.loc[:, ["date", "symbol", "close"]],
        horizon=horizon,
        price_column="close",
        entry_lag=entry_lag,
    )
    if target == "log_realized_volatility":
        by_symbol = labeled.groupby("symbol", sort=False)
        log_return = np.log(labeled["close"] / by_symbol["close"].shift(1))
        squared = (log_return**2).groupby(labeled["symbol"], sort=False)
        # mean of r_k^2 over the `horizon` returns ending at the exit row
        window = squared.rolling(horizon, min_periods=horizon).mean().to_numpy()
        at_exit = (
            pd.Series(window, index=labeled.index)
            .groupby(labeled["symbol"])
            .shift(-(entry_lag + horizon))
        )
        with np.errstate(divide="ignore"):
            value = np.where(at_exit > 0, 0.5 * np.log(at_exit), np.nan)
        known = labeled[TARGET_RETURN_COLUMN].notna().to_numpy()
        labeled[TARGET_RETURN_COLUMN] = np.where(known, value, np.nan)
    labeled["date"] = labeled["date"].dt.as_unit("ns")
    columns = ["date", "symbol", TARGET_RETURN_COLUMN, TARGET_START_COLUMN, TARGET_END_COLUMN]
    return labeled.loc[:, columns].sort_values(["date", "symbol"]).reset_index(drop=True)
