from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.macro.schema import (
    MACRO_COLUMNS,
    MacroSeriesSpec,
    normalize_macro_frame,
    validate_macro_frame,
)


def _macro_row(
    observation_date: str = "2025-01-01",
    available_at: str | None = "2025-01-10",
    vintage_date: str | None = "2025-01-10",
    series_id: str = "US_UNEMPLOYMENT_RATE",
    provider_series_id: str = "UNRATE",
    value: float | str = 4.1,
    frequency: str = "monthly",
    units: str = "percent",
    source: str = "FRED",
) -> dict:
    """Create one canonical macroeconomic observation."""

    return {
        "observation_date": observation_date,
        "available_at": available_at,
        "vintage_date": vintage_date,
        "series_id": series_id,
        "provider_series_id": provider_series_id,
        "value": value,
        "frequency": frequency,
        "units": units,
        "source": source,
    }


def test_valid_macro_frame_passes_validation():
    frame = pd.DataFrame(
        [
            _macro_row(),
        ]
    )

    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"],
        utc=True,
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"],
        utc=True,
    )
    frame["vintage_date"] = pd.to_datetime(
        frame["vintage_date"],
        utc=True,
    )

    validate_macro_frame(frame)


def test_missing_required_column_raises():
    frame = pd.DataFrame(
        [
            _macro_row(),
        ]
    ).drop(
        columns=["available_at"],
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_macro_frame(frame)


def test_missing_available_at_raises():
    frame = pd.DataFrame(
        [
            _macro_row(
                available_at=None,
            ),
        ]
    )

    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"],
        utc=True,
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"],
        utc=True,
    )
    frame["vintage_date"] = pd.to_datetime(
        frame["vintage_date"],
        utc=True,
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        validate_macro_frame(frame)


def test_duplicate_macro_vintage_raises():
    row = _macro_row()

    frame = pd.DataFrame(
        [
            row,
            row.copy(),
        ]
    )

    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"],
        utc=True,
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"],
        utc=True,
    )
    frame["vintage_date"] = pd.to_datetime(
        frame["vintage_date"],
        utc=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_macro_frame(frame)


def test_invalid_observation_date_raises():
    frame = pd.DataFrame(
        [
            _macro_row(
                observation_date="not-a-date",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        normalize_macro_frame(frame)


def test_numeric_string_value_is_converted_to_float():
    frame = pd.DataFrame(
        [
            _macro_row(
                value="4.1",
            ),
        ]
    )

    result = normalize_macro_frame(frame)

    assert result.loc[0, "value"] == pytest.approx(4.1)
    assert pd.api.types.is_float_dtype(
        result["value"],
    )


def test_vintage_date_may_be_missing():
    frame = pd.DataFrame(
        [
            _macro_row(
                vintage_date=None,
            ),
        ]
    )

    result = normalize_macro_frame(frame)

    assert pd.isna(
        result.loc[0, "vintage_date"],
    )

    assert result.loc[0, "available_at"] == pd.Timestamp(
        "2025-01-10",
        tz="UTC",
    )


def test_normalized_frame_uses_canonical_column_order():
    frame = pd.DataFrame(
        [
            _macro_row(),
        ]
    )

    frame = frame[list(reversed(frame.columns))]

    result = normalize_macro_frame(frame)

    assert list(result.columns) == MACRO_COLUMNS


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("series_id", ""),
        ("provider_series_id", ""),
        ("frequency", ""),
        ("units", ""),
        ("series_id", "   "),
    ],
)
def test_macro_series_spec_requires_non_empty_strings(
    field_name: str,
    invalid_value: str,
):
    values = {
        "series_id": "US_UNEMPLOYMENT_RATE",
        "provider_series_id": "UNRATE",
        "frequency": "monthly",
        "units": "percent",
    }

    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        MacroSeriesSpec(**values)
