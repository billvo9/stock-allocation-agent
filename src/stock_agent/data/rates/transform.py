from __future__ import annotations

import pandas as pd


def percent_yield_to_decimal(
    yields: pd.Series,
) -> pd.Series:
    """Convert provider percentage yields to decimal yields."""

    return yields.astype(float) / 100.0
