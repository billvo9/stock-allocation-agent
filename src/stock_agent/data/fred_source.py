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


class FredDataSource:
    """Generic transport layer for FRED API Version 2."""

    BASE_URL = "https://api.stlouisfed.org/fred/v2/release/observations"

    REQUEST_TIMEOUT_SECONDS = 15
    MAX_OBSERVATIONS_PER_REQUEST = 500_000

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("FRED_API_KEY")

        if not self._api_key:
            raise ValueError("FRED_API_KEY is not configured.")

        self._session = session or requests.Session()

    def _request_page(
        self,
        release_id: int,
        next_cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {
            "release_id": release_id,
            "format": "json",
            "limit": (self.MAX_OBSERVATIONS_PER_REQUEST),
        }

        if next_cursor is not None:
            params["next_cursor"] = next_cursor

        try:
            response = self._session.get(
                self.BASE_URL,
                params=params,
                headers={"Authorization": (f"Bearer {self._api_key}")},
                timeout=(self.REQUEST_TIMEOUT_SECONDS),
            )

            response.raise_for_status()

            payload = response.json()

        except requests.RequestException as exc:
            raise TypeError("FRED API request failed.") from exc

        if not isinstance(payload, dict):
            raise TypeError("Unexpected FRED API response.")

        series_collection = payload.get("series")

        if not isinstance(
            series_collection,
            list,
        ):
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

                if not isinstance(
                    observations,
                    list,
                ):
                    raise TypeError("Invalid FRED observations.")

                for observation in observations:
                    if not isinstance(
                        observation,
                        dict,
                    ):
                        raise TypeError("Invalid FRED observation.")

                    rows.append(
                        {
                            "date": observation.get("date"),
                            "value": observation.get("value"),
                            "provider_series_id": (series_id),
                            "provider_release_id": (release_id),
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
