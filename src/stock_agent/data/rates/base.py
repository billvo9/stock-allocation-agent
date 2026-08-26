from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from stock_agent.data.rates.schema import (
    RateSeriesSpec,
)


class RateDataSource(ABC):
    @abstractmethod
    def get_rates(
        self,
        spec: RateSeriesSpec,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Return rates using the canonical rate schema."""
