from __future__ import annotations

import pandas as pd

from stock_agent.data.fundamentals.schema import (
    FUNDAMENTAL_SNAPSHOT_COLUMNS,
    METADATA_COLUMNS,
)
from stock_agent.data.fundamentals.storage import (
    merge_metadata,
    merge_snapshot_history,
)


def _metadata_row(
    symbol: str,
    retrieved_at: str,
    sector: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "provider_symbol": symbol,
                "sector": sector,
                "sector_key": (sector.lower()),
                "industry": "Test Industry",
                "industry_key": "test-industry",
                "country": "United States",
                "exchange": "NMS",
                "quote_type": "EQUITY",
                "currency": "USD",
                "source": "yfinance",
                "retrieved_at": pd.Timestamp(retrieved_at),
            }
        ],
        columns=METADATA_COLUMNS,
    )


def _snapshot_row(
    symbol: str,
    retrieved_at: str,
    forward_pe: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "provider_symbol": symbol,
                "market_cap": 100_000_000,
                "enterprise_value": (90_000_000),
                "forward_pe": forward_pe,
                "trailing_pe": 20.0,
                "price_to_book": 3.0,
                "forward_eps": 5.0,
                "trailing_eps": 4.0,
                "beta": 1.2,
                "dividend_yield": 0.01,
                "shares_outstanding": (10_000_000),
                "source": "yfinance",
                "retrieved_at": pd.Timestamp(retrieved_at),
            }
        ],
        columns=(FUNDAMENTAL_SNAPSHOT_COLUMNS),
    )


def test_merge_metadata_adds_new_symbol():
    existing = _metadata_row(
        symbol="MU",
        retrieved_at=("2026-09-03T12:00:00Z"),
        sector="Technology",
    )

    incoming = _metadata_row(
        symbol="JNJ",
        retrieved_at=("2026-09-04T12:00:00Z"),
        sector="Healthcare",
    )

    result = merge_metadata(
        existing=existing,
        incoming=incoming,
    )

    assert list(result["symbol"]) == [
        "JNJ",
        "MU",
    ]


def test_merge_metadata_keeps_newest_row():
    existing = _metadata_row(
        symbol="MU",
        retrieved_at=("2026-09-03T12:00:00Z"),
        sector="Old Sector",
    )

    incoming = _metadata_row(
        symbol="MU",
        retrieved_at=("2026-09-04T12:00:00Z"),
        sector="Technology",
    )

    result = merge_metadata(
        existing=existing,
        incoming=incoming,
    )

    assert len(result) == 1

    assert result.loc[0, "sector"] == "Technology"

    assert result.loc[
        0,
        "retrieved_at",
    ] == pd.Timestamp("2026-09-04T12:00:00Z")


def test_merge_snapshot_history_preserves_history():
    existing = _snapshot_row(
        symbol="MU",
        retrieved_at=("2026-09-03T12:00:00Z"),
        forward_pe=6.4,
    )

    incoming = _snapshot_row(
        symbol="MU",
        retrieved_at=("2026-09-04T12:00:00Z"),
        forward_pe=6.2,
    )

    result = merge_snapshot_history(
        existing=existing,
        incoming=incoming,
    )

    assert len(result) == 2

    assert list(result["forward_pe"]) == [
        6.4,
        6.2,
    ]


def test_merge_snapshot_history_is_idempotent():
    existing = _snapshot_row(
        symbol="MU",
        retrieved_at=("2026-09-04T12:00:00Z"),
        forward_pe=6.2,
    )

    incoming = _snapshot_row(
        symbol="MU",
        retrieved_at=("2026-09-04T12:00:00Z"),
        forward_pe=6.2,
    )

    result = merge_snapshot_history(
        existing=existing,
        incoming=incoming,
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "forward_pe",
        ]
        == 6.2
    )
