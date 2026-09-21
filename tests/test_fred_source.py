from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from stock_agent.data.fred_source import (
    FRED_OBSERVATION_COLUMNS,
    FredDataSource,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
    ) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
    ) -> None:
        self._payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )

        payload = self._payloads[len(self.calls) - 1]

        return FakeResponse(payload)


def test_fred_source_returns_generic_observations():
    payload = {
        "has_more": "false",
        "series": [
            {
                "series_id": "DGS3MO",
                "observations": [
                    {
                        "date": "2026-08-20",
                        "value": "3.87",
                    },
                    {
                        "date": "2026-08-21",
                        "value": ".",
                    },
                    {
                        "date": "2026-08-22",
                        "value": "3.88",
                    },
                ],
            },
            {
                "series_id": "OTHER_SERIES",
                "observations": [
                    {
                        "date": "2026-08-20",
                        "value": "999",
                    }
                ],
            },
        ],
    }

    session = FakeSession([payload])

    source = FredDataSource(
        api_key="fake-test-key",
        session=session,
    )

    result = source.get_observations(
        release_id=18,
        series_id="DGS3MO",
        start="2026-08-20",
        end="2026-08-21",
    )

    assert list(result.columns) == (FRED_OBSERVATION_COLUMNS)

    assert len(result) == 2

    assert list(result["value"]) == [
        "3.87",
        ".",
    ]

    assert list(result["provider_series_id"]) == [
        "DGS3MO",
        "DGS3MO",
    ]

    assert list(result["provider_release_id"]) == [
        18,
        18,
    ]

    assert result["date"].iloc[0] == pd.Timestamp("2026-08-20")


def test_fred_source_uses_bearer_authentication():
    payload = {
        "has_more": "false",
        "series": [
            {
                "series_id": "DGS3MO",
                "observations": [],
            }
        ],
    }

    session = FakeSession([payload])

    source = FredDataSource(
        api_key="fake-test-key",
        session=session,
    )

    source.get_observations(
        release_id=18,
        series_id="DGS3MO",
        start="2026-08-20",
        end="2026-08-21",
    )

    call = session.calls[0]

    assert call["headers"] == {"Authorization": ("Bearer fake-test-key")}

    assert "api_key" not in call["params"]

    assert call["params"]["release_id"] == 18


def test_fred_source_collects_paginated_observations():
    first_page = {
        "has_more": "true",
        "next_cursor": ("DGS3MO,2026-08-21"),
        "series": [
            {
                "series_id": "DGS3MO",
                "observations": [
                    {
                        "date": "2026-08-20",
                        "value": "3.87",
                    }
                ],
            }
        ],
    }

    second_page = {
        "has_more": "false",
        "series": [
            {
                "series_id": "DGS3MO",
                "observations": [
                    {
                        "date": "2026-08-21",
                        "value": "3.88",
                    }
                ],
            }
        ],
    }

    session = FakeSession(
        [
            first_page,
            second_page,
        ]
    )

    source = FredDataSource(
        api_key="fake-test-key",
        session=session,
    )

    result = source.get_observations(
        release_id=18,
        series_id="DGS3MO",
        start="2026-08-20",
        end="2026-08-21",
    )

    assert list(result["value"]) == [
        "3.87",
        "3.88",
    ]

    assert len(session.calls) == 2

    second_call = session.calls[1]

    assert second_call["params"]["next_cursor"] == "DGS3MO,2026-08-21"


def test_fred_source_requires_cursor_when_more_data_exists():
    payload = {
        "has_more": "true",
        "series": [
            {
                "series_id": "DGS3MO",
                "observations": [],
            }
        ],
    }

    session = FakeSession([payload])

    source = FredDataSource(
        api_key="fake-test-key",
        session=session,
    )

    with pytest.raises(
        RuntimeError,
        match="cursor",
    ):
        source.get_observations(
            release_id=18,
            series_id="DGS3MO",
            start="2026-08-20",
            end="2026-08-21",
        )


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def get(
        self,
        url: str,
        *,
        params: dict,
        headers: dict | None = None,
        timeout: int | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )

        return _FakeResponse(self._payloads.pop(0))


def test_series_observations_preserve_vintage_metadata():
    session = _FakeSession(
        [
            {
                "count": 2,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": "2024-02-02",
                        "realtime_end": "2024-03-01",
                        "date": "2024-01-01",
                        "value": "3.7",
                    },
                    {
                        "realtime_start": "2024-03-02",
                        "realtime_end": "9999-12-31",
                        "date": "2024-02-01",
                        "value": "3.9",
                    },
                ],
            }
        ]
    )

    source = FredDataSource(
        api_key="test-key",
        session=session,
    )

    result = source.get_series_observations(
        series_id="UNRATE",
    )

    assert len(result) == 2

    assert result["provider_series_id"].tolist() == [
        "UNRATE",
        "UNRATE",
    ]

    assert result["value"].tolist() == [
        "3.7",
        "3.9",
    ]

    assert result["realtime_start"].tolist() == [
        "2024-02-02",
        "2024-03-02",
    ]

    assert result["source"].tolist() == [
        "FRED",
        "FRED",
    ]


def test_series_observations_preserve_missing_provider_value():
    session = _FakeSession(
        [
            {
                "count": 1,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": "2024-02-01",
                        "realtime_end": "9999-12-31",
                        "date": "2024-01-01",
                        "value": ".",
                    }
                ],
            }
        ]
    )

    source = FredDataSource(
        api_key="test-key",
        session=session,
    )

    result = source.get_series_observations(
        series_id="UNRATE",
    )

    assert result.loc[0, "value"] == "."


def test_series_observations_follow_offset_pagination():
    session = _FakeSession(
        [
            {
                "count": 3,
                "offset": 0,
                "limit": 2,
                "observations": [
                    {
                        "realtime_start": "2024-02-01",
                        "realtime_end": "9999-12-31",
                        "date": "2024-01-01",
                        "value": "3.7",
                    },
                    {
                        "realtime_start": "2024-03-01",
                        "realtime_end": "9999-12-31",
                        "date": "2024-02-01",
                        "value": "3.8",
                    },
                ],
            },
            {
                "count": 3,
                "offset": 2,
                "limit": 2,
                "observations": [
                    {
                        "realtime_start": "2024-04-01",
                        "realtime_end": "9999-12-31",
                        "date": "2024-03-01",
                        "value": "3.9",
                    }
                ],
            },
        ]
    )

    source = FredDataSource(
        api_key="test-key",
        session=session,
    )

    result = source.get_series_observations(
        series_id="UNRATE",
    )

    assert len(result) == 3
    assert len(session.calls) == 2

    assert session.calls[0]["params"]["offset"] == 0
    assert session.calls[1]["params"]["offset"] == 2


def test_vintage_dates_follow_offset_pagination():
    session = _FakeSession(
        [
            {
                "count": 3,
                "offset": 0,
                "limit": 2,
                "vintage_dates": [
                    "2024-01-10",
                    "2024-02-10",
                ],
            },
            {
                "count": 3,
                "offset": 2,
                "limit": 2,
                "vintage_dates": [
                    "2024-03-10",
                ],
            },
        ]
    )

    source = FredDataSource(
        api_key="test-key",
        session=session,
    )

    result = source.get_series_vintage_dates(
        series_id="UNRATE",
    )

    assert result == [
        "2024-01-10",
        "2024-02-10",
        "2024-03-10",
    ]

    assert len(session.calls) == 2


def test_vintage_dates_cannot_be_combined_with_realtime_range():
    source = FredDataSource(
        api_key="test-key",
        session=_FakeSession([]),
    )

    with pytest.raises(
        ValueError,
        match="vintage_dates cannot be combined",
    ):
        source.get_series_observations(
            series_id="UNRATE",
            realtime_start="2024-01-01",
            vintage_dates=["2024-02-01"],
        )
