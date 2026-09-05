from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import yfinance as yf

from stock_agent.data.fundamentals.base import (
    FundamentalDataBundle,
    FundamentalDataSource,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
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

    @staticmethod
    def _statement_value(
        frame: pd.DataFrame,
        row_name: str,
        period_end: pd.Timestamp,
    ) -> float:
        if frame.empty:
            return float("nan")

        if row_name not in frame.index:
            return float("nan")

        matching_columns = [
            column for column in frame.columns if pd.Timestamp(column).date() == period_end.date()
        ]

        if not matching_columns:
            return float("nan")

        value = frame.at[
            row_name,
            matching_columns[0],
        ]

        if pd.isna(value):
            return float("nan")

        return float(value)

    @staticmethod
    def _find_sec_filing(
        period_end: pd.Timestamp,
        filings: list[dict] | None,
    ) -> dict | None:
        if not filings:
            return None

        candidates: list[tuple[pd.Timestamp, dict]] = []

        maximum_date = period_end + pd.Timedelta(days=120)

        for filing in filings:
            if not isinstance(
                filing,
                dict,
            ):
                continue

            form_type = str(
                filing.get(
                    "type",
                    "",
                )
            ).upper()

            if form_type not in {
                "10-Q",
                "10-K",
            }:
                continue

            raw_date = filing.get("date")

            if raw_date is None:
                continue

            filing_date = pd.Timestamp(raw_date)

            if filing_date.tzinfo is None:
                filing_date = filing_date.tz_localize("UTC")
            else:
                filing_date = filing_date.tz_convert("UTC")

            if period_end <= filing_date <= maximum_date:
                candidates.append(
                    (
                        filing_date,
                        filing,
                    )
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])

        return candidates[0][1]

    def get_quarterly_fundamentals(
        self,
        symbol: str,
        provider_symbol: str,
    ) -> pd.DataFrame:
        """Return canonical quarterly fundamentals."""

        if not symbol.strip():
            raise ValueError("Symbol cannot be empty.")

        if not provider_symbol.strip():
            raise ValueError("Provider symbol cannot be empty.")

        try:
            ticker = self._ticker_factory(provider_symbol)

            income = ticker.get_income_stmt(freq="quarterly")

            balance = ticker.get_balance_sheet(freq="quarterly")

            cashflow = ticker.get_cash_flow(freq="quarterly")

            filings = ticker.get_sec_filings()

        except Exception as exc:
            raise RuntimeError(
                f"Unable to download quarterly fundamentals for {symbol} ({provider_symbol})."
            ) from exc

        retrieved_at = self._normalize_timestamp(self._clock())

        period_dates = {
            pd.Timestamp(column).date()
            for frame in [
                income,
                balance,
                cashflow,
            ]
            for column in frame.columns
        }

        rows: list[dict] = []

        for period_date in sorted(period_dates):
            period_end = pd.Timestamp(
                period_date,
                tz="UTC",
            )

            filing = self._find_sec_filing(
                period_end,
                filings,
            )

            filing_date = pd.NaT
            available_at = pd.NaT
            sec_form_type = None
            availability_source = None

            if filing is not None:
                filing_date = pd.Timestamp(filing["date"])

                if filing_date.tzinfo is None:
                    filing_date = filing_date.tz_localize("UTC")
                else:
                    filing_date = filing_date.tz_convert("UTC")

                available_at = filing_date + pd.Timedelta(days=1)

                sec_form_type = filing.get("type")

                availability_source = "sec_filing_date_plus_1d"

            rows.append(
                {
                    "symbol": symbol,
                    "provider_symbol": (provider_symbol),
                    "period_end": period_end,
                    "available_at": available_at,
                    "retrieved_at": retrieved_at,
                    "revenue": self._statement_value(
                        income,
                        "TotalRevenue",
                        period_end,
                    ),
                    "gross_profit": (
                        self._statement_value(
                            income,
                            "GrossProfit",
                            period_end,
                        )
                    ),
                    "operating_income": (
                        self._statement_value(
                            income,
                            "OperatingIncome",
                            period_end,
                        )
                    ),
                    "net_income": (
                        self._statement_value(
                            income,
                            "NetIncome",
                            period_end,
                        )
                    ),
                    "diluted_eps": (
                        self._statement_value(
                            income,
                            "DilutedEPS",
                            period_end,
                        )
                    ),
                    "diluted_average_shares": (
                        self._statement_value(
                            income,
                            "DilutedAverageShares",
                            period_end,
                        )
                    ),
                    "total_assets": (
                        self._statement_value(
                            balance,
                            "TotalAssets",
                            period_end,
                        )
                    ),
                    "total_debt": (
                        self._statement_value(
                            balance,
                            "TotalDebt",
                            period_end,
                        )
                    ),
                    "cash_and_cash_equivalents": (
                        self._statement_value(
                            balance,
                            "CashAndCashEquivalents",
                            period_end,
                        )
                    ),
                    "inventory": (
                        self._statement_value(
                            balance,
                            "Inventory",
                            period_end,
                        )
                    ),
                    "stockholders_equity": (
                        self._statement_value(
                            balance,
                            "StockholdersEquity",
                            period_end,
                        )
                    ),
                    "operating_cash_flow": (
                        self._statement_value(
                            cashflow,
                            "OperatingCashFlow",
                            period_end,
                        )
                    ),
                    "capital_expenditure": (
                        self._statement_value(
                            cashflow,
                            "CapitalExpenditure",
                            period_end,
                        )
                    ),
                    "free_cash_flow": (
                        self._statement_value(
                            cashflow,
                            "FreeCashFlow",
                            period_end,
                        )
                    ),
                    "filing_date": filing_date,
                    "sec_form_type": sec_form_type,
                    "sec_accession_number": None,
                    "availability_source": (availability_source),
                    "currency": None,
                    "source": self.SOURCE_NAME,
                }
            )

        frame = pd.DataFrame(
            rows,
            columns=(QUARTERLY_FUNDAMENTAL_COLUMNS),
        )

        validate_quarterly_fundamental_frame(frame)

        return frame.sort_values("period_end").reset_index(drop=True)
