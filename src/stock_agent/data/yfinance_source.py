from __future__ import annotations

import pandas as pd
import yfinance as yf

from stock_agent.data.base import MarketDataSource
from stock_agent.data.schema import CANONICAL_PRICE_COLUMNS
from stock_agent.exceptions import (
    MarketDataDownloadError,
    MarketDataValidationError,
)


class YFinanceDataSource(MarketDataSource):
    """Market-data adapter for Yahoo Finance via yfinance."""

    SOURCE_NAME = "yfinance"

    def get_prices(
        self,
        symbol: str,
        vendor_symbol: str,
        start: str,
        end: str,
        currency: str = "USD",
    ) -> pd.DataFrame:
        try:
            frame = yf.download(
                vendor_symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise MarketDataDownloadError(
                f"Unable to download market data for {symbol} "
                f"({vendor_symbol}) from {start} to {end}. "
                f"Original error: {exc}"
            ) from exc

        if frame.empty:
            raise MarketDataDownloadError(
                f"No market data returned for {symbol} ({vendor_symbol}) from {start} to {end}."
            )

        frame = self._flatten_columns(frame)

        # YOUR PART
        frame = standardize_yfinance_columns(frame)

        frame = frame.reset_index()

        # yfinance may call the index Date or Datetime.
        first_column = frame.columns[0]
        frame = frame.rename(columns={first_column: "date"})

        frame["symbol"] = symbol
        frame["currency"] = currency
        frame["source"] = self.SOURCE_NAME

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

        self._validate(frame, symbol)

        return frame[CANONICAL_PRICE_COLUMNS].sort_values("date").reset_index(drop=True)

    @staticmethod
    def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Flatten a yfinance MultiIndex when one is returned."""

        result = frame.copy()

        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)

        return result

    @staticmethod
    def _validate(frame: pd.DataFrame, symbol: str) -> None:
        missing = [column for column in CANONICAL_PRICE_COLUMNS if column not in frame.columns]

        if missing:
            raise MarketDataValidationError(
                f"{symbol}: standardized market data are missing required columns: {missing}"
            )

        if frame["date"].isna().any():
            raise MarketDataValidationError(f"{symbol}: one or more dates could not be parsed.")

        if frame["date"].duplicated().any():
            raise MarketDataValidationError(f"{symbol}: duplicate trading dates were detected.")


def standardize_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance price columns into the project's canonical schema.

    TODO_STUDENT
    """

    result = frame.copy()

    column_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    result = result.rename(columns=column_map)

    # raise NotImplementedError("TODO_STUDENT: implement column standardization.")
    return result
