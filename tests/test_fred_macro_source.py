from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.macro.fred_source import FredMacroDataSource
from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
    MacroSeriesSpec,
)


class FakeFredDataSource:
    def __init__(
        self,
        observations: pd.DataFrame,
    ) -> None:
        self.observations = observations
        self.calls: list[dict[str, object]] = []

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        vintage_dates: list[str] | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "vintage_dates": vintage_dates,
            }
        )

        return self.observations.copy()


def _spec() -> MacroSeriesSpec:
    return MacroSeriesSpec(
        series_id="us_unemployment_rate",
        provider_series_id="UNRATE",
        frequency="monthly",
        units="percent",
    )


def _raw_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "value": "3.7",
                "realtime_start": "2024-02-02",
                "realtime_end": "2024-03-07",
                "provider_series_id": "UNRATE",
                "source": "FRED",
            },
            {
                "date": "2024-01-01",
                "value": "3.8",
                "realtime_start": "2024-03-08",
                "realtime_end": "9999-12-31",
                "provider_series_id": "UNRATE",
                "source": "FRED",
            },
        ]
    )


def test_fred_macro_source_returns_canonical_schema():
    fake_fred = FakeFredDataSource(_raw_observations())

    source = FredMacroDataSource(
        fred_source=fake_fred,
    )

    result = source.get_observations(
        spec=_spec(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert list(result.columns) == MACRO_COLUMNS
    assert len(result) == 2


def test_fred_macro_source_maps_canonical_series_id():
    fake_fred = FakeFredDataSource(_raw_observations())

    source = FredMacroDataSource(
        fred_source=fake_fred,
    )

    result = source.get_observations(
        spec=_spec(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert (result["series_id"] == "us_unemployment_rate").all()

    assert (result["provider_series_id"] == "UNRATE").all()


def test_fred_macro_source_converts_values_to_numeric():
    fake_fred = FakeFredDataSource(_raw_observations())

    source = FredMacroDataSource(
        fred_source=fake_fred,
    )

    result = source.get_observations(
        spec=_spec(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert result["value"].tolist() == [
        pytest.approx(3.7),
        pytest.approx(3.8),
    ]


def test_fred_macro_source_preserves_revisions():
    fake_fred = FakeFredDataSource(_raw_observations())

    source = FredMacroDataSource(
        fred_source=fake_fred,
    )

    result = source.get_observations(
        spec=_spec(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert len(result) == 2

    assert result["observation_date"].nunique() == 1

    assert result["available_at"].nunique() == 2

    assert result["value"].tolist() == [
        pytest.approx(3.7),
        pytest.approx(3.8),
    ]


def test_fred_macro_source_drops_missing_provider_values():
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "value": ".",
                "realtime_start": "2024-02-02",
                "realtime_end": "9999-12-31",
                "provider_series_id": "UNRATE",
                "source": "FRED",
            },
            {
                "date": "2024-02-01",
                "value": "3.9",
                "realtime_start": "2024-03-08",
                "realtime_end": "9999-12-31",
                "provider_series_id": "UNRATE",
                "source": "FRED",
            },
        ]
    )

    source = FredMacroDataSource(
        fred_source=FakeFredDataSource(raw),
    )

    result = source.get_observations(
        spec=_spec(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(3.9)


def test_fred_macro_source_requests_complete_realtime_history():
    fake_fred = FakeFredDataSource(_raw_observations())

    source = FredMacroDataSource(
        fred_source=fake_fred,
    )

    source.get_observations(
        spec=_spec(),
        start="2016-01-01",
        end="2026-12-31",
    )

    call = fake_fred.calls[0]

    assert call["series_id"] == "UNRATE"
    assert call["observation_start"] == "2016-01-01"
    assert call["observation_end"] == "2026-12-31"

    assert call["realtime_start"] == "1776-07-04"

    assert call["realtime_end"] == "9999-12-31"


def test_fred_macro_source_raises_when_no_data_returned():
    source = FredMacroDataSource(
        fred_source=FakeFredDataSource(pd.DataFrame()),
    )

    with pytest.raises(
        ValueError,
        match="no macro observations",
    ):
        source.get_observations(
            spec=_spec(),
            start="2024-01-01",
            end="2024-12-31",
        )


def test_fred_macro_source_raises_when_all_values_missing():
    raw = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "value": ".",
                "realtime_start": "2024-02-02",
                "realtime_end": "9999-12-31",
                "provider_series_id": "UNRATE",
                "source": "FRED",
            }
        ]
    )

    source = FredMacroDataSource(
        fred_source=FakeFredDataSource(raw),
    )

    with pytest.raises(
        ValueError,
        match="no usable macro observations",
    ):
        source.get_observations(
            spec=_spec(),
            start="2024-01-01",
            end="2024-12-31",
        )
