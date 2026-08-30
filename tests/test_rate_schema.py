import pandas as pd
import pytest

from stock_agent.data.rates.schema import (
    validate_rate_frame,
)


def test_validate_rate_frame_accepts_valid_data():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-20",
                    "2026-08-21",
                ]
            ),
            "rate_id": [
                "USD_TREASURY_3M",
                "USD_TREASURY_3M",
            ],
            "provider_series_id": [
                "DGS3MO",
                "DGS3MO",
            ],
            "currency": [
                "USD",
                "USD",
            ],
            "tenor": [
                "3M",
                "3M",
            ],
            "annual_yield": [
                0.0387,
                0.0388,
            ],
            "quote_convention": [
                "investment_basis",
                "investment_basis",
            ],
            "source": [
                "FRED",
                "FRED",
            ],
        }
    )

    validate_rate_frame(frame)


def test_missing_columns_rate_frame():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-20",
                    "2026-08-21",
                ]
            ),
            "rate_id": [
                "USD_TREASURY_3M",
                "USD_TREASURY_3M",
            ],
            "provider_series_id": [
                "DGS3MO",
                "DGS3MO",
            ],
            "currency": [
                "USD",
                "USD",
            ],
            "tenor": [
                "3M",
                "3M",
            ],
            "quote_convention": [
                "investment_basis",
                "investment_basis",
            ],
            "source": [
                "FRED",
                "FRED",
            ],
        }
    )
    with pytest.raises(ValueError, match="annual_yield"):
        validate_rate_frame(frame)
