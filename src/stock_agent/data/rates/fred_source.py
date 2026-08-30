from __future__ import annotations

import pandas as pd
import requests

from stock_agent.data.fred_source import (
    FredDataSource,
)
from stock_agent.data.rates.base import (
    RateDataSource,
)
from stock_agent.data.rates.schema import (
    RATE_COLUMNS,
    RateSeriesSpec,
    validate_rate_frame,
)
from stock_agent.data.rates.transform import (
    percent_yield_to_decimal,
)


class FredRateDataSource(RateDataSource):
    """Adapt generic FRED observations into canonical rate data."""

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        fred_source: FredDataSource | None = None,
    ) -> None:
        self._fred_source = fred_source or FredDataSource(
            api_key=api_key,
            session=session,
        )

    def get_rates(
        self,
        spec: RateSeriesSpec,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        if spec.provider_release_id is None:
            raise ValueError("FRED V2 requires provider_release_id.")

        observations = self._fred_source.get_observations(
            release_id=(spec.provider_release_id),
            series_id=(spec.provider_series_id),
            start=start,
            end=end,
        )

        if observations.empty:
            raise ValueError("No FRED rate observations were found.")

        usable = observations[observations["value"].notna() & observations["value"].ne(".")].copy()

        if usable.empty:
            raise ValueError("FRED rate observations contain no usable values.")

        provider_yields = pd.to_numeric(
            usable["value"],
            errors="raise",
        )

        annual_yields = percent_yield_to_decimal(provider_yields)

        frame = pd.DataFrame(
            {
                "date": usable["date"].to_numpy(),
                "rate_id": spec.rate_id,
                "provider_series_id": (spec.provider_series_id),
                "currency": spec.currency,
                "tenor": spec.tenor,
                "annual_yield": (annual_yields.to_numpy()),
                "quote_convention": (spec.quote_convention),
                "source": "FRED",
            },
            columns=RATE_COLUMNS,
        )

        frame = frame.sort_values(["date", "rate_id"]).reset_index(drop=True)

        validate_rate_frame(frame)

        return frame
