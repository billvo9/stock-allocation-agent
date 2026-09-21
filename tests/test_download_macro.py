from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_agent.data.macro.config import (
    MacroSeriesConfig,
)
from stock_agent.data.macro.ingestion import (
    build_macro_summary,
    download_macro_data,
    save_macro_data,
)
from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
)


class FakeMacroDataSource:
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
    ) -> None:
        self.frames = frames
        self.calls: list[dict[str, object]] = []

    def get_observations(
        self,
        spec,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "series_id": spec.series_id,
                "start": start,
                "end": end,
            }
        )

        return self.frames[spec.series_id].copy()


def _config(
    macro_id: str,
    provider_series_id: str,
    provider: str = "FRED",
) -> MacroSeriesConfig:
    return MacroSeriesConfig(
        macro_id=macro_id,
        provider=provider,
        provider_series_id=(provider_series_id),
        country="US",
        category="test",
        expected_frequency="monthly",
        units="percent",
    )


def _macro_frame(
    series_id: str,
    provider_series_id: str,
    rows: list[tuple[str, str, float]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "observation_date": (observation_date),
                "available_at": available_at,
                "vintage_date": available_at,
                "series_id": series_id,
                "provider_series_id": (provider_series_id),
                "value": value,
                "frequency": "monthly",
                "units": "percent",
                "source": "FRED",
            }
            for (
                observation_date,
                available_at,
                value,
            ) in rows
        ],
        columns=MACRO_COLUMNS,
    )

    for column in [
        "observation_date",
        "available_at",
        "vintage_date",
    ]:
        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
        )

    return frame


def test_download_macro_data_combines_series():
    unemployment = _macro_frame(
        "us_unemployment_rate",
        "UNRATE",
        [
            (
                "2024-01-01",
                "2024-02-02",
                3.7,
            ),
        ],
    )

    inflation = _macro_frame(
        "us_consumer_price_index",
        "CPIAUCSL",
        [
            (
                "2024-01-01",
                "2024-02-13",
                308.4,
            ),
        ],
    )

    provider = FakeMacroDataSource(
        {
            "us_unemployment_rate": (unemployment),
            "us_consumer_price_index": (inflation),
        }
    )

    configs = [
        _config(
            "us_unemployment_rate",
            "UNRATE",
        ),
        _config(
            "us_consumer_price_index",
            "CPIAUCSL",
        ),
    ]

    result = download_macro_data(
        configs=configs,
        start="2024-01-01",
        end="2024-12-31",
        providers={
            "FRED": provider,
        },
    )

    assert len(result) == 2

    assert set(result["series_id"]) == {
        "us_unemployment_rate",
        "us_consumer_price_index",
    }


def test_download_macro_data_preserves_revisions():
    revisions = _macro_frame(
        "us_unemployment_rate",
        "UNRATE",
        [
            (
                "2024-01-01",
                "2024-02-02",
                3.7,
            ),
            (
                "2024-01-01",
                "2024-03-08",
                3.8,
            ),
        ],
    )

    provider = FakeMacroDataSource(
        {
            "us_unemployment_rate": (revisions),
        }
    )

    result = download_macro_data(
        configs=[
            _config(
                "us_unemployment_rate",
                "UNRATE",
            )
        ],
        start="2024-01-01",
        end="2024-12-31",
        providers={
            "FRED": provider,
        },
    )

    assert len(result) == 2

    assert result["observation_date"].nunique() == 1

    assert result["available_at"].nunique() == 2


def test_download_macro_data_passes_date_range():
    frame = _macro_frame(
        "us_unemployment_rate",
        "UNRATE",
        [
            (
                "2024-01-01",
                "2024-02-02",
                3.7,
            )
        ],
    )

    provider = FakeMacroDataSource(
        {
            "us_unemployment_rate": frame,
        }
    )

    download_macro_data(
        configs=[
            _config(
                "us_unemployment_rate",
                "UNRATE",
            )
        ],
        start="2015-01-01",
        end="2026-09-21",
        providers={
            "FRED": provider,
        },
    )

    assert provider.calls == [
        {
            "series_id": ("us_unemployment_rate"),
            "start": "2015-01-01",
            "end": "2026-09-21",
        }
    ]


def test_unknown_provider_raises():
    config = _config(
        "us_unemployment_rate",
        "UNRATE",
        provider="UNKNOWN",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported macro provider",
    ):
        download_macro_data(
            configs=[config],
            start="2024-01-01",
            end="2024-12-31",
            providers={},
        )


def test_invalid_date_range_raises():
    with pytest.raises(
        ValueError,
        match="start must not be after end",
    ):
        download_macro_data(
            configs=[
                _config(
                    "us_unemployment_rate",
                    "UNRATE",
                )
            ],
            start="2025-01-01",
            end="2024-01-01",
            providers={},
        )


def test_empty_config_raises():
    with pytest.raises(
        ValueError,
        match="At least one macro series",
    ):
        download_macro_data(
            configs=[],
            start="2024-01-01",
            end="2024-12-31",
            providers={},
        )


def test_save_macro_data_round_trip(
    tmp_path: Path,
):
    frame = _macro_frame(
        "us_unemployment_rate",
        "UNRATE",
        [
            (
                "2024-01-01",
                "2024-02-02",
                3.7,
            ),
        ],
    )

    output = tmp_path / "raw" / "macro_vintages.parquet"

    result_path = save_macro_data(
        frame,
        output,
    )

    loaded = pd.read_parquet(result_path)

    pd.testing.assert_frame_equal(
        loaded,
        frame,
    )


def test_macro_summary_counts_observations_and_revisions():
    frame = _macro_frame(
        "us_unemployment_rate",
        "UNRATE",
        [
            (
                "2024-01-01",
                "2024-02-02",
                3.7,
            ),
            (
                "2024-01-01",
                "2024-03-08",
                3.8,
            ),
            (
                "2024-02-01",
                "2024-03-08",
                3.9,
            ),
        ],
    )

    summary = build_macro_summary(frame)

    row = summary.loc["us_unemployment_rate"]

    assert row["rows"] == 3
    assert row["observations"] == 2
