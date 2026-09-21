from __future__ import annotations

import pandas as pd

from stock_agent.data.fred_source import FredDataSource
from stock_agent.data.macro.base import MacroDataSource
from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
    MacroSeriesSpec,
    validate_macro_frame,
)


class FredMacroDataSource(MacroDataSource):
    """
    FRED adapter for canonical macroeconomic observations.

    The generic FredDataSource handles HTTP, authentication,
    pagination, and FRED-specific response structure.

    This adapter maps raw provider observations into the
    project's canonical macro schema.
    """

    SOURCE_NAME = "FRED"

    FULL_REALTIME_START = "1776-07-04"
    FULL_REALTIME_END = "9999-12-31"

    def __init__(
        self,
        fred_source: FredDataSource | None = None,
    ) -> None:
        self._fred_source = fred_source or FredDataSource()

    def get_observations(
        self,
        spec: MacroSeriesSpec,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return point-in-time historical macro observations.

        Multiple revisions of the same observation are preserved.
        The FRED realtime_start becomes both:

        available_at:
            when this value became usable historically.

        vintage_date:
            provider vintage associated with that value.
        """

        raw = self._fred_source.get_series_observations(
            series_id=spec.provider_series_id,
            observation_start=start,
            observation_end=end,
            realtime_start=self.FULL_REALTIME_START,
            realtime_end=self.FULL_REALTIME_END,
        )

        if raw.empty:
            raise ValueError(f"FRED returned no macro observations for {spec.provider_series_id}.")

        usable = raw[raw["value"].notna() & raw["value"].ne(".")].copy()

        if usable.empty:
            raise ValueError(
                f"FRED returned no usable macro observations for {spec.provider_series_id}."
            )

        values = pd.to_numeric(
            usable["value"],
            errors="raise",
        )

        observation_dates = pd.to_datetime(
            usable["date"],
            errors="raise",
            utc=True,
        )

        available_dates = pd.to_datetime(
            usable["realtime_start"],
            errors="raise",
            utc=True,
        )

        frame = pd.DataFrame(
            {
                "observation_date": observation_dates,
                "available_at": available_dates,
                "vintage_date": available_dates,
                "series_id": spec.series_id,
                "provider_series_id": spec.provider_series_id,
                "value": values,
                "frequency": spec.frequency,
                "units": spec.units,
                "source": self.SOURCE_NAME,
            },
            columns=MACRO_COLUMNS,
        )

        frame = frame.sort_values(
            [
                "series_id",
                "observation_date",
                "available_at",
            ]
        ).reset_index(drop=True)

        validate_macro_frame(frame)

        return frame
