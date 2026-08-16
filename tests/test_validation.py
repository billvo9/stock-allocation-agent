import pandas as pd
import pytest

from stock_agent.data.yfinance_source import YFinanceDataSource
from stock_agent.exceptions import MarketDataValidationError


def test_duplicate_dates_raise_validation_error():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "symbol": ["MU", "MU"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "adjusted_close": [103.0, 104.0],
            "volume": [1_000_000, 1_100_000],
            "currency": ["USD", "USD"],
            "source": ["yfinance", "yfinance"],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="duplicate trading dates",
    ):
        YFinanceDataSource._validate(frame, "MU")
