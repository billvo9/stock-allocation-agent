from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import yfinance as yf

from stock_agent.data.fundamentals.base import (
    FundamentalDataBundle,
    FundamentalDataSource,
)
from stock_agent.data.fundamentals.schema import (
    FUNDAMENTAL_SNAPSHOT_COLUMNS,
    METADATA_COLUMNS,
    NUMERIC_SNAPSHOT_COLUMNS,
    validate_fundamental_snapshot_frame,
    validate_metadata_frame,
)

TickerFactory = Callable[
    [str],
    Any,
]

Clock = Callable[
    [],
    pd.Timestamp,
]


class YFinanceFundamentalDataSource(FundamentalDataSource):
    """Fundamental-data adapter for Yahoo Finance."""

    SOURCE_NAME = "yfinance"

    def __init__(
        self,
        ticker_factory: TickerFactory | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._ticker_factory = ticker_factory or yf.Ticker

        self._clock = clock or self._utc_now

    @staticmethod
    def _utc_now() -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    @staticmethod
    def _normalize_timestamp(
        value: pd.Timestamp,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")

        return timestamp.tz_convert("UTC")

    def _load_info(
        self,
        provider_symbol: str,
    ) -> dict[str, Any]:
        try:
            ticker = self._ticker_factory(provider_symbol)

            info = ticker.get_info()

        except Exception as exc:
            raise RuntimeError(
                f"Unable to download fundamental data for {provider_symbol}."
            ) from exc

        if not isinstance(
            info,
            dict,
        ):
            raise TypeError(
                f"Yahoo Finance returned invalid fundamental data for {provider_symbol}."
            )

        return info

    def get_company_data(
        self,
        symbol: str,
        provider_symbol: str,
    ) -> FundamentalDataBundle:
        """Return canonical Yahoo metadata and snapshot data."""

        if not symbol.strip():
            raise ValueError("Symbol cannot be empty.")

        if not provider_symbol.strip():
            raise ValueError("Provider symbol cannot be empty.")

        info = self._load_info(provider_symbol)

        retrieved_at = self._normalize_timestamp(self._clock())

        metadata = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "provider_symbol": (provider_symbol),
                    "sector": info.get("sector"),
                    "sector_key": info.get("sectorKey"),
                    "industry": info.get("industry"),
                    "industry_key": info.get("industryKey"),
                    "country": info.get("country"),
                    "exchange": info.get("exchange"),
                    "quote_type": info.get("quoteType"),
                    "currency": info.get("currency"),
                    "source": self.SOURCE_NAME,
                    "retrieved_at": (retrieved_at),
                }
            ],
            columns=METADATA_COLUMNS,
        )

        snapshot = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "provider_symbol": (provider_symbol),
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": (info.get("enterpriseValue")),
                    "forward_pe": info.get("forwardPE"),
                    "trailing_pe": info.get("trailingPE"),
                    "price_to_book": info.get("priceToBook"),
                    "forward_eps": info.get("forwardEps"),
                    "trailing_eps": info.get("trailingEps"),
                    "beta": info.get("beta"),
                    "dividend_yield": (info.get("dividendYield")),
                    "shares_outstanding": (info.get("sharesOutstanding")),
                    "source": self.SOURCE_NAME,
                    "retrieved_at": (retrieved_at),
                }
            ],
            columns=(FUNDAMENTAL_SNAPSHOT_COLUMNS),
        )

        for column in NUMERIC_SNAPSHOT_COLUMNS:
            snapshot[column] = pd.to_numeric(
                snapshot[column],
                errors="coerce",
            )

        validate_metadata_frame(metadata)

        validate_fundamental_snapshot_frame(snapshot)

        return FundamentalDataBundle(
            metadata=metadata,
            snapshot=snapshot,
        )
