from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals import (
    FUNDAMENTAL_SNAPSHOT_COLUMNS,
    METADATA_COLUMNS,
    validate_fundamental_snapshot_frame,
    validate_metadata_frame,
)


def _make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "MU",
                "provider_symbol": "MU",
                "sector": "Technology",
                "sector_key": "technology",
                "industry": "Semiconductors",
                "industry_key": "semiconductors",
                "country": "United States",
                "exchange": "NMS",
                "quote_type": "EQUITY",
                "currency": "USD",
                "source": "yfinance",
                "retrieved_at": pd.Timestamp("2026-09-03T12:00:00Z"),
            }
        ],
        columns=METADATA_COLUMNS,
    )


def _make_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "MU",
                "provider_symbol": "MU",
                "market_cap": 150_000_000_000,
                "enterprise_value": 145_000_000_000,
                "forward_pe": 18.5,
                "trailing_pe": 22.0,
                "price_to_book": 3.2,
                "forward_eps": 7.5,
                "trailing_eps": 6.8,
                "beta": 1.25,
                "dividend_yield": 0.004,
                "shares_outstanding": 1_100_000_000,
                "source": "yfinance",
                "retrieved_at": pd.Timestamp("2026-09-03T12:00:00Z"),
            }
        ],
        columns=FUNDAMENTAL_SNAPSHOT_COLUMNS,
    )


def test_valid_metadata_frame_passes():
    frame = _make_metadata()

    validate_metadata_frame(frame)


def test_metadata_allows_optional_missing_values():
    frame = _make_metadata()

    frame.loc[0, "industry"] = None
    frame.loc[0, "country"] = None

    validate_metadata_frame(frame)


def test_metadata_rejects_missing_required_identifier():
    frame = _make_metadata()

    frame.loc[0, "symbol"] = None

    with pytest.raises(
        ValueError,
        match="symbol",
    ):
        validate_metadata_frame(frame)


def test_valid_fundamental_snapshot_passes():
    frame = _make_snapshot()

    validate_fundamental_snapshot_frame(frame)


def test_snapshot_allows_optional_missing_numeric_values():
    frame = _make_snapshot()

    frame.loc[0, "forward_pe"] = None
    frame.loc[0, "beta"] = None

    validate_fundamental_snapshot_frame(frame)


def test_snapshot_rejects_invalid_numeric_value():
    frame = _make_snapshot()

    frame["forward_pe"] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="forward_pe",
    ):
        validate_fundamental_snapshot_frame(frame)
