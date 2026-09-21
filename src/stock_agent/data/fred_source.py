from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

FRED_OBSERVATION_COLUMNS = [
    "date",
    "value",
    "provider_series_id",
    "provider_release_id",
    "source",
]

FRED_SERIES_OBSERVATION_COLUMNS = [
    "date",
    "value",
    "realtime_start",
    "realtime_end",
    "provider_series_id",
    "source",
]


class FredDataSource:
    """
    Generic transport layer for FRED.

    Existing FRED API Version 2 release-level retrieval is preserved
    for current consumers such as the rates pipeline.

    Series-level Version 1 endpoints are also supported so that
    point-in-time and vintage-aware macroeconomic data can be retrieved.
    """

    # Existing Version 2 endpoint.
    BASE_URL = "https://api.stlouisfed.org/fred/v2/release/observations"

    # Version 1 series-level endpoints.
    SERIES_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
    SERIES_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"

    REQUEST_TIMEOUT_SECONDS = 15

    MAX_OBSERVATIONS_PER_REQUEST = 500_000

    # Version 1 documented limits.
    MAX_SERIES_OBSERVATIONS_PER_REQUEST = 100_000
    MAX_VINTAGE_DATES_PER_REQUEST = 10_000

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("FRED_API_KEY")

        if not self._api_key:
            raise ValueError("FRED_API_KEY is not configured.")

        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Existing Version 2 release-level API
    # ------------------------------------------------------------------

    def _request_page(
        self,
        release_id: int,
        next_cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {
            "release_id": release_id,
            "format": "json",
            "limit": self.MAX_OBSERVATIONS_PER_REQUEST,
        }

        if next_cursor is not None:
            params["next_cursor"] = next_cursor

        try:
            response = self._session.get(
                self.BASE_URL,
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            payload = response.json()

        except requests.RequestException as exc:
            raise TypeError("FRED API request failed.") from exc

        if not isinstance(payload, dict):
            raise TypeError("Unexpected FRED API response.")

        series_collection = payload.get("series")

        if not isinstance(series_collection, list):
            raise TypeError("FRED response is missing a valid series collection.")

        return payload

    @staticmethod
    def _has_more(
        payload: dict[str, Any],
    ) -> bool:
        value = payload.get(
            "has_more",
            False,
        )

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.lower() == "true"

        raise TypeError("Invalid FRED pagination state.")

    def _collect_series_observations(
        self,
        release_id: int,
        series_id: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        next_cursor: str | None = None
        series_found = False

        while True:
            payload = self._request_page(
                release_id=release_id,
                next_cursor=next_cursor,
            )

            for series in payload["series"]:
                if series.get("series_id") != series_id:
                    continue

                series_found = True

                observations = series.get(
                    "observations",
                    [],
                )

                if not isinstance(observations, list):
                    raise TypeError("Invalid FRED observations.")

                for observation in observations:
                    if not isinstance(observation, dict):
                        raise TypeError("Invalid FRED observation.")

                    rows.append(
                        {
                            "date": observation.get("date"),
                            "value": observation.get("value"),
                            "provider_series_id": series_id,
                            "provider_release_id": release_id,
                            "source": "FRED",
                        }
                    )

            if not self._has_more(payload):
                break

            cursor = payload.get("next_cursor")

            if not isinstance(cursor, str) or not cursor:
                raise RuntimeError("FRED indicates more data but returned no cursor.")

            next_cursor = cursor

        if not series_found:
            raise ValueError(f"FRED series not found: {series_id}")

        return rows

    def get_observations(
        self,
        release_id: int,
        series_id: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Retrieve observations from a Version 2 FRED release.

        This method is kept for backwards compatibility with the
        existing rate-data pipeline.
        """

        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)

        if start_date > end_date:
            raise ValueError("Start date must not be after end date.")

        rows = self._collect_series_observations(
            release_id=release_id,
            series_id=series_id,
        )

        frame = pd.DataFrame(
            rows,
            columns=FRED_OBSERVATION_COLUMNS,
        )

        if frame.empty:
            return frame

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        frame = frame[
            frame["date"].between(
                start_date,
                end_date,
            )
        ].copy()

        return frame.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Version 1 series-level API
    # ------------------------------------------------------------------

    def _request_series_observations_page(
        self,
        series_id: str,
        observation_start: str | None,
        observation_end: str | None,
        realtime_start: str | None,
        realtime_end: str | None,
        vintage_dates: list[str] | None,
        offset: int,
    ) -> dict[str, Any]:
        """
        Request one page from fred/series/observations.

        Raw provider values are deliberately preserved here.
        Domain-specific interpretation belongs in downstream adapters.
        """

        if vintage_dates and (realtime_start is not None or realtime_end is not None):
            raise ValueError(
                "vintage_dates cannot be combined with realtime_start or realtime_end."
            )

        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "output_type": 1,
            "limit": self.MAX_SERIES_OBSERVATIONS_PER_REQUEST,
            "offset": offset,
            "sort_order": "asc",
        }

        if observation_start is not None:
            params["observation_start"] = observation_start

        if observation_end is not None:
            params["observation_end"] = observation_end

        if realtime_start is not None:
            params["realtime_start"] = realtime_start

        if realtime_end is not None:
            params["realtime_end"] = realtime_end

        if vintage_dates:
            params["vintage_dates"] = ",".join(vintage_dates)

        try:
            response = self._session.get(
                self.SERIES_OBSERVATIONS_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            payload = response.json()

        except requests.RequestException as exc:
            raise RuntimeError("FRED series observations request failed.") from exc

        if not isinstance(payload, dict):
            raise TypeError("Unexpected FRED series observations response.")

        observations = payload.get("observations")

        if not isinstance(observations, list):
            raise TypeError("FRED series response is missing observations.")

        return payload

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        vintage_dates: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve generic FRED series observations.

        The returned values remain in provider form. In particular,
        missing FRED observations such as "." are not interpreted here.

        realtime_start and realtime_end describe when a value was
        considered current by FRED and therefore allow downstream
        consumers to perform point-in-time alignment.
        """

        if (
            observation_start is not None
            and observation_end is not None
            and pd.Timestamp(observation_start) > pd.Timestamp(observation_end)
        ):
            raise ValueError("observation_start must not be after observation_end.")

        rows: list[dict[str, object]] = []
        offset = 0

        while True:
            payload = self._request_series_observations_page(
                series_id=series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                realtime_start=realtime_start,
                realtime_end=realtime_end,
                vintage_dates=vintage_dates,
                offset=offset,
            )

            observations = payload["observations"]

            for observation in observations:
                if not isinstance(observation, dict):
                    raise TypeError("Invalid FRED series observation.")

                rows.append(
                    {
                        "date": observation.get("date"),
                        "value": observation.get("value"),
                        "realtime_start": observation.get("realtime_start"),
                        "realtime_end": observation.get("realtime_end"),
                        "provider_series_id": series_id,
                        "source": "FRED",
                    }
                )

            count = payload.get("count")
            current_offset = payload.get(
                "offset",
                offset,
            )
            limit = payload.get(
                "limit",
                self.MAX_SERIES_OBSERVATIONS_PER_REQUEST,
            )

            if not isinstance(count, int):
                raise TypeError("FRED series response has invalid count.")

            if not isinstance(current_offset, int):
                raise TypeError("FRED series response has invalid offset.")

            if not isinstance(limit, int):
                raise TypeError("FRED series response has invalid limit.")

            next_offset = current_offset + len(observations)

            if not observations or next_offset >= count:
                break

            offset = next_offset

        frame = pd.DataFrame(
            rows,
            columns=FRED_SERIES_OBSERVATION_COLUMNS,
        )

        if frame.empty:
            return frame

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        )

        return frame.sort_values(
            [
                "date",
                "realtime_start",
            ]
        ).reset_index(drop=True)

    def _request_vintage_dates_page(
        self,
        series_id: str,
        offset: int,
    ) -> dict[str, Any]:
        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": self.MAX_VINTAGE_DATES_PER_REQUEST,
            "offset": offset,
            "sort_order": "asc",
        }

        try:
            response = self._session.get(
                self.SERIES_VINTAGE_DATES_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            payload = response.json()

        except requests.RequestException as exc:
            raise RuntimeError("FRED vintage-date request failed.") from exc

        if not isinstance(payload, dict):
            raise TypeError("Unexpected FRED vintage-date response.")

        vintage_dates = payload.get("vintage_dates")

        if not isinstance(vintage_dates, list):
            raise TypeError("FRED vintage-date response is missing vintage_dates.")

        return payload

    def get_series_vintage_dates(
        self,
        series_id: str,
    ) -> list[str]:
        """
        Return dates when a FRED series received new or revised data.
        """

        vintage_dates: list[str] = []
        offset = 0

        while True:
            payload = self._request_vintage_dates_page(
                series_id=series_id,
                offset=offset,
            )

            page_dates = payload["vintage_dates"]

            for vintage_date in page_dates:
                if not isinstance(vintage_date, str):
                    raise TypeError("Invalid FRED vintage date.")

                vintage_dates.append(vintage_date)

            count = payload.get("count")
            current_offset = payload.get(
                "offset",
                offset,
            )

            if not isinstance(count, int):
                raise TypeError("FRED vintage response has invalid count.")

            if not isinstance(current_offset, int):
                raise TypeError("FRED vintage response has invalid offset.")

            next_offset = current_offset + len(page_dates)

            if not page_dates or next_offset >= count:
                break

            offset = next_offset

        return vintage_dates
