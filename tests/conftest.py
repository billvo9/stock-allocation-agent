from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec


def build_labeled_panel(
    *,
    symbols: tuple[str, ...] = ("A", "B", "C"),
    sessions: int = 120,
    horizon: int = 5,
    entry_lag: int = 1,
    seed: int = 0,
    start: str = "2021-01-04",
    drop_fraction: float = 0.0,
    listing: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Seeded synthetic labeled panel: independent log random walks per symbol,
    two noise features, optional listing windows and randomly dropped
    sessions (gaps). Dates are session dates at 00:00 UTC; the result has a
    default RangeIndex sorted by (date, symbol).
    """

    rng = np.random.RandomState(seed)  # legacy stream: stable across numpy versions
    calendar = pd.bdate_range(start, periods=sessions, tz="UTC")
    rows = []
    for symbol in symbols:
        prices = 100.0 * np.exp(np.cumsum(0.01 * rng.randn(sessions)))
        first, last = (listing or {}).get(symbol, (0, sessions))
        for index in range(first, last):
            if drop_fraction and rng.rand() < drop_fraction:
                continue
            rows.append(
                {
                    "date": calendar[index],
                    "symbol": symbol,
                    "adjusted_close": prices[index],
                    "feature_a": rng.randn(),
                    "feature_b": rng.randn(),
                }
            )
    labeled = LabelSpec(horizon=horizon, entry_lag=entry_lag).apply(pd.DataFrame(rows))
    return labeled.sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture
def labeled_panel() -> Callable[..., pd.DataFrame]:
    return build_labeled_panel


@pytest.fixture(scope="session")
def panel_factory() -> Callable[..., pd.DataFrame]:
    """build_labeled_panel for module-scoped fixtures (expensive shared runs)."""

    return build_labeled_panel
