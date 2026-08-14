from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_return_matrix(
        frame: pd.DataFrame,
        symbols: list[str],
) -> pd.DataFrame:
    """Convert long-format feature data into date x symbol returns."""

    filtered = frame[
        frame["symbol"].isin(symbols)
    ][["date", "symbol", "daily_return"]]

    returns = (
        filtered.pivot(
            index="index",
            columns="symbol",
            values="daily_return",
        )
        .sort_index()
    )

    missing_symbols = set(symbols) - set(returns.colums)

    if missing_symbols:
        raise ValueError(
            f"missing required symbols: {sorted(missing_symbols)}"
        )

    returns = returns[symbols]

    if returns.isna().any().any():
        raise ValueError(
            "Return matrix contains missing values."
        )

    return returns


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    """
    Update portfolio weights after asset returns occur.
    """

    new_weights_returns = weights * (asset_returns + 1)

    normalized_new_weights_returns = new_weights_returns / np.sum(new_weights_returns)

    return normalized_new_weights_returns