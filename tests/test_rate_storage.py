import pandas as pd
import pytest

from stock_agent.data.rates.storage import (
    merge_rate_history,
)


def _make_rate_frame(
    dates: list[str],
    yields: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "rate_id": "USD_TREASURY_3M",
            "provider_series_id": "DGS3MO",
            "currency": "USD",
            "tenor": "3M",
            "annual_yield": yields,
            "quote_convention": ("investment_basis"),
            "source": "FRED",
        }
    )


def test_merge_rate_history_incoming_data_wins():
    existing = _make_rate_frame(
        dates=[
            "2026-01-02",
            "2026-01-03",
        ],
        yields=[
            0.0400,
            0.0410,
        ],
    )

    incoming = _make_rate_frame(
        dates=[
            "2026-01-03",
            "2026-01-04",
        ],
        yields=[
            0.0420,
            0.0430,
        ],
    )

    result = merge_rate_history(
        existing=existing,
        incoming=incoming,
    )

    assert len(result) == 3

    assert list(result["date"]) == list(
        pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ]
        )
    )

    assert list(result["annual_yield"]) == pytest.approx(
        [
            0.0400,
            0.0420,
            0.0430,
        ]
    )


def test_merge_rate_history_is_idempotent():
    existing = _make_rate_frame(
        dates=["2026-01-02"],
        yields=[0.0400],
    )

    incoming = existing.copy()

    result = merge_rate_history(
        existing=existing,
        incoming=incoming,
    )

    assert len(result) == 1
