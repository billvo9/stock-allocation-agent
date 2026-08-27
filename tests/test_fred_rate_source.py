from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.rates.fred_source import (
    FredRateDataSource,
)
from stock_agent.data.rates.schema import (
    RATE_COLUMNS,
    RateSeriesSpec,
)


class FakeFredDataSource:
    def __init__(
        self,
        observations: pd.DataFrame,
    ) -> None:
        self._observations = observations

    def get_observations(
        self,
        release_id: int,
        series_id: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        return self._observations.copy()


def test_fred_rate_source_builds_canonical_rate_data():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-20",
                    "2026-08-21",
                    "2026-08-22",
                ]
            ),
            "value": [
                "3.87",
                ".",
                "3.88",
            ],
            "provider_series_id": [
                "DGS3MO",
                "DGS3MO",
                "DGS3MO",
            ],
            "provider_release_id": [
                18,
                18,
                18,
            ],
            "source": [
                "FRED",
                "FRED",
                "FRED",
            ],
        }
    )

    fake_source = FakeFredDataSource(observations)

    source = FredRateDataSource(fred_source=fake_source)

    spec = RateSeriesSpec(
        rate_id="USD_TREASURY_3M",
        provider_series_id="DGS3MO",
        provider_release_id=18,
        currency="USD",
        tenor="3M",
        quote_convention=("investment_basis"),
    )

    result = source.get_rates(
        spec=spec,
        start="2026-08-20",
        end="2026-08-22",
    )

    assert list(result.columns) == (RATE_COLUMNS)

    assert len(result) == 2

    assert result["annual_yield"].iloc[0] == pytest.approx(0.0387)

    assert result["annual_yield"].iloc[1] == pytest.approx(0.0388)

    assert (result["rate_id"] == "USD_TREASURY_3M").all()

    assert (result["currency"] == "USD").all()

    assert (result["tenor"] == "3M").all()

    assert (result["quote_convention"] == "investment_basis").all()


def test_fred_rate_source_requires_release_id():
    observations = pd.DataFrame()

    fake_source = FakeFredDataSource(observations)

    source = FredRateDataSource(fred_source=fake_source)

    spec = RateSeriesSpec(
        rate_id="USD_TREASURY_3M",
        provider_series_id="DGS3MO",
        provider_release_id=None,
        currency="USD",
        tenor="3M",
        quote_convention=("investment_basis"),
    )

    with pytest.raises(
        ValueError,
        match="release",
    ):
        source.get_rates(
            spec=spec,
            start="2026-08-20",
            end="2026-08-21",
        )
