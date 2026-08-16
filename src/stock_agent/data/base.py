from abc import ABC, abstractmethod

import pandas as pd


class MarketDataSource(ABC):
    """Interface implemented by all market-data providers."""

    @abstractmethod
    def get_prices(
        self,
        symbol: str,
        vendor_symbol: str,
        start: str,
        end: str,
        currency: str = "USD",
    ) -> pd.DataFrame:
        """Return standardized historical price data."""
        raise NotImplementedError
