from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from stock_agent.data.fundamentals.yfinance_source import (
    YFinanceFundamentalDataSource,
)


class FakeTicker:
    def __init__(
        self,
        info: dict[str, Any],
    ) -> None:
        self._info = info
        self.get_info_calls = 0

    def get_info(self) -> dict[str, Any]:
        self.get_info_calls += 1
        return self._info


class FailingTicker:
    def get_info(self) -> dict[str, Any]:
        raise RuntimeError("provider unavailable")


def _fixed_clock() -> pd.Timestamp:
    return pd.Timestamp("2026-09-03T12:00:00Z")


def _make_info() -> dict[str, Any]:
    return {
        "sector": "Technology",
        "sectorKey": "technology",
        "industry": "Semiconductors",
        "industryKey": "semiconductors",
        "country": "United States",
        "exchange": "NMS",
        "quoteType": "EQUITY",
        "currency": "USD",
        "marketCap": 150_000_000_000,
        "enterpriseValue": 145_000_000_000,
        "forwardPE": 18.5,
        "trailingPE": 22.0,
        "priceToBook": 3.2,
        "forwardEps": 7.5,
        "trailingEps": 6.8,
        "beta": 1.25,
        "dividendYield": 0.004,
        "sharesOutstanding": 1_100_000_000,
    }


def test_yfinance_source_maps_metadata_and_snapshot():
    fake_ticker = FakeTicker(_make_info())

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: fake_ticker),
        clock=_fixed_clock,
    )

    result = source.get_company_data(
        symbol="MU",
        provider_symbol="MU",
    )

    metadata = result.metadata
    snapshot = result.snapshot

    assert (
        metadata.loc[
            0,
            "symbol",
        ]
        == "MU"
    )

    assert (
        metadata.loc[
            0,
            "provider_symbol",
        ]
        == "MU"
    )

    assert (
        metadata.loc[
            0,
            "sector",
        ]
        == "Technology"
    )

    assert (
        metadata.loc[
            0,
            "industry",
        ]
        == "Semiconductors"
    )

    assert (
        metadata.loc[
            0,
            "country",
        ]
        == "United States"
    )

    assert snapshot.loc[
        0,
        "market_cap",
    ] == pytest.approx(150_000_000_000)

    assert snapshot.loc[
        0,
        "forward_pe",
    ] == pytest.approx(18.5)

    assert snapshot.loc[
        0,
        "beta",
    ] == pytest.approx(1.25)

    assert fake_ticker.get_info_calls == 1


def test_yfinance_source_uses_same_retrieval_timestamp():
    fake_ticker = FakeTicker(_make_info())

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: fake_ticker),
        clock=_fixed_clock,
    )

    result = source.get_company_data(
        symbol="MU",
        provider_symbol="MU",
    )

    metadata_time = result.metadata.loc[
        0,
        "retrieved_at",
    ]

    snapshot_time = result.snapshot.loc[
        0,
        "retrieved_at",
    ]

    assert metadata_time == snapshot_time

    assert metadata_time == pd.Timestamp("2026-09-03T12:00:00Z")


def test_yfinance_source_allows_missing_optional_fields():
    fake_ticker = FakeTicker(
        {
            "sector": "Technology",
            "currency": "USD",
        }
    )

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: fake_ticker),
        clock=_fixed_clock,
    )

    result = source.get_company_data(
        symbol="TEST",
        provider_symbol="TEST",
    )

    assert pd.isna(
        result.metadata.loc[
            0,
            "industry",
        ]
    )

    assert pd.isna(
        result.snapshot.loc[
            0,
            "forward_pe",
        ]
    )

    assert pd.isna(
        result.snapshot.loc[
            0,
            "market_cap",
        ]
    )


def test_yfinance_source_rejects_empty_symbol():
    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: FakeTicker(_make_info())),
        clock=_fixed_clock,
    )

    with pytest.raises(
        ValueError,
        match="Symbol",
    ):
        source.get_company_data(
            symbol="",
            provider_symbol="MU",
        )


def test_yfinance_source_wraps_provider_failure():
    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: FailingTicker()),
        clock=_fixed_clock,
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to download",
    ):
        source.get_company_data(
            symbol="MU",
            provider_symbol="MU",
        )


def test_yfinance_source_rejects_invalid_provider_response():
    class InvalidTicker:
        def get_info(self) -> list[str]:
            return [
                "invalid",
                "response",
            ]

    source = YFinanceFundamentalDataSource(
        ticker_factory=(lambda symbol: InvalidTicker()),
        clock=_fixed_clock,
    )

    with pytest.raises(
        TypeError,
        match="invalid fundamental data",
    ):
        source.get_company_data(
            symbol="MU",
            provider_symbol="MU",
        )
