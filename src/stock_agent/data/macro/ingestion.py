from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from stock_agent.data.macro.base import MacroDataSource
from stock_agent.data.macro.config import MacroSeriesConfig
from stock_agent.data.macro.fred_source import FredMacroDataSource
from stock_agent.data.macro.schema import validate_macro_frame


def build_default_provider_registry() -> dict[str, MacroDataSource]:
    """Construct the default macro-data provider registry."""

    return {
        "FRED": FredMacroDataSource(),
    }


def download_macro_data(
    configs: list[MacroSeriesConfig],
    start: str,
    end: str,
    providers: Mapping[str, MacroDataSource] | None = None,
) -> pd.DataFrame:
    """
    Download configured macroeconomic series.

    Historical revisions are preserved at ingestion time.
    """

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)

    if start_date > end_date:
        raise ValueError("start must not be after end.")

    if not configs:
        raise ValueError("At least one macro series is required.")

    provider_registry = (
        dict(providers) if providers is not None else build_default_provider_registry()
    )

    frames: list[pd.DataFrame] = []

    for config in configs:
        provider_name = config.provider.upper()

        provider = provider_registry.get(provider_name)

        if provider is None:
            raise ValueError(f"Unsupported macro provider: {provider_name}")

        frame = provider.get_observations(
            spec=config.to_spec(),
            start=start,
            end=end,
        )

        if frame.empty:
            raise ValueError(f"Macro provider returned no data for {config.macro_id}.")

        frames.append(frame)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "series_id",
            "observation_date",
            "available_at",
        ]
    ).reset_index(drop=True)

    validate_macro_frame(result)

    return result


def save_macro_data(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist canonical macro vintage history."""

    validate_macro_frame(frame)

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_parquet(
        output,
        index=False,
    )

    return output


def build_macro_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact ingestion-quality summary."""

    validate_macro_frame(frame)

    return (
        frame.groupby("series_id")
        .agg(
            rows=(
                "observation_date",
                "size",
            ),
            observations=(
                "observation_date",
                "nunique",
            ),
            first_observation=(
                "observation_date",
                "min",
            ),
            last_observation=(
                "observation_date",
                "max",
            ),
            first_available_at=(
                "available_at",
                "min",
            ),
            last_available_at=(
                "available_at",
                "max",
            ),
        )
        .sort_index()
    )
