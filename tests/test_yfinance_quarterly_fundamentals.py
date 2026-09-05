from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_agent.data.fundamentals.yfinance_source import (
    YFinanceFundamentalDataSource,
)

PERIOD = pd.Timestamp("2026-05-31")


def _income_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            PERIOD: {
                "TotalRevenue": 1000.0,
                "GrossProfit": 400.0,
                "OperatingIncome": 250.0,
                "NetIncome": 200.0,
                "DilutedEPS": 2.0,
                "DilutedAverageShares": 100.0,
            }
        }
    )


def _balance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            PERIOD: {
                "TotalAssets": 5000.0,
                "TotalDebt": 1000.0,
                "CashAndCashEquivalents": 500.0,
                "Inventory": 300.0,
                "StockholdersEquity": 3000.0,
            }
        }
    )


def _cashflow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            PERIOD: {
                "OperatingCashFlow": 350.0,
                "CapitalExpenditure": -150.0,
                "FreeCashFlow": 200.0,
            }
        }
    )


class FakeQuarterlyTicker:
    def __init__(
        self,
        filings: list[dict] | None = None,
    ) -> None:
        self._filings = filings or []

    def get_income_stmt(
        self,
        freq: str,
    ) -> pd.DataFrame:
        assert freq == "quarterly"
        return _income_frame()

    def get_balance_sheet(
        self,
        freq: str,
    ) -> pd.DataFrame:
        assert freq == "quarterly"
        return _balance_frame()

    def get_cash_flow(
        self,
        freq: str,
    ) -> pd.DataFrame:
        assert freq == "quarterly"
        return _cashflow_frame()

    def get_sec_filings(
        self,
    ) -> list[dict]:
        return self._filings


def _fixed_clock() -> pd.Timestamp:
    return pd.Timestamp("2026-09-04T12:00:00Z")


def test_quarterly_source_maps_statement_fields():
    ticker = FakeQuarterlyTicker(
        filings=[
            {
                "date": date(
                    2026,
                    6,
                    26,
                ),
                "type": "10-Q",
            }
        ]
    )

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: ticker),
        clock=_fixed_clock,
    )

    result = source.get_quarterly_fundamentals(
        symbol="MU",
        provider_symbol="MU",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["revenue"] == pytest.approx(1000.0)

    assert row["gross_profit"] == pytest.approx(400.0)

    assert row["operating_income"] == pytest.approx(250.0)

    assert row["total_assets"] == pytest.approx(5000.0)

    assert row["inventory"] == pytest.approx(300.0)

    assert row["operating_cash_flow"] == pytest.approx(350.0)

    assert row["capital_expenditure"] == pytest.approx(-150.0)

    assert row["free_cash_flow"] == pytest.approx(200.0)


def test_quarterly_source_uses_conservative_filing_availability():
    ticker = FakeQuarterlyTicker(
        filings=[
            {
                "date": date(
                    2026,
                    6,
                    26,
                ),
                "type": "10-Q",
            }
        ]
    )

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: ticker),
        clock=_fixed_clock,
    )

    result = source.get_quarterly_fundamentals(
        symbol="MU",
        provider_symbol="MU",
    )

    row = result.iloc[0]

    assert row["filing_date"] == pd.Timestamp("2026-06-26T00:00:00Z")

    assert row["available_at"] == pd.Timestamp("2026-06-27T00:00:00Z")

    assert row["sec_form_type"] == "10-Q"

    assert row["availability_source"] == "sec_filing_date_plus_1d"


def test_quarterly_source_allows_missing_statement_field():
    balance = _balance_frame().drop(index="Inventory")

    class MissingInventoryTicker(FakeQuarterlyTicker):
        def get_balance_sheet(
            self,
            freq: str,
        ) -> pd.DataFrame:
            assert freq == "quarterly"
            return balance

    ticker = MissingInventoryTicker()

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: ticker),
        clock=_fixed_clock,
    )

    result = source.get_quarterly_fundamentals(
        symbol="MU",
        provider_symbol="MU",
    )

    assert pd.isna(
        result.loc[
            0,
            "inventory",
        ]
    )


def test_quarterly_source_allows_unknown_availability():
    ticker = FakeQuarterlyTicker()

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: ticker),
        clock=_fixed_clock,
    )

    result = source.get_quarterly_fundamentals(
        symbol="MU",
        provider_symbol="MU",
    )

    assert pd.isna(
        result.loc[
            0,
            "available_at",
        ]
    )

    assert pd.isna(
        result.loc[
            0,
            "filing_date",
        ]
    )


def test_quarterly_source_wraps_provider_failure():
    class FailingTicker:
        def get_income_stmt(
            self,
            freq: str,
        ) -> pd.DataFrame:
            raise RuntimeError("provider unavailable")

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: FailingTicker()),
        clock=_fixed_clock,
    )

    with pytest.raises(
        RuntimeError,
        match="quarterly fundamentals",
    ):
        source.get_quarterly_fundamentals(
            symbol="MU",
            provider_symbol="MU",
        )
