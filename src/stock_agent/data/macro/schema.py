from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MACRO_COLUMNS = [
    "observation_date",
    "available_at",
    "vintage_date",
    "series_id",
    "provider_series_id",
    "value",
    "frequency",
    "units",
    "source",
]


@dataclass(frozen=True)
class MacroSeriesSpec:
    """
    Provider-neutral description of a macroeconomic series.

    series_id:
        Internal semantic identifier used by our system.

        Example:
            US_UNEMPLOYMENT_RATE

    provider_series_id:
        Identifier used by the external data provider.

        Example for FRED:
            UNRATE

    frequency:
        Native observation frequency.

        Examples:
            daily
            weekly
            monthly
            quarterly

    units:
        Economic meaning of the stored value.

        Examples:
            percent
            index
            billions_usd
            persons
    """

    series_id: str
    provider_series_id: str
    frequency: str
    units: str

    def __post_init__(self) -> None:
        fields = {
            "series_id": self.series_id,
            "provider_series_id": self.provider_series_id,
            "frequency": self.frequency,
            "units": self.units,
        }

        for field_name, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")


def normalize_macro_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize a macroeconomic frame into the canonical schema.

    Dates are converted to UTC.

    Values are converted to numeric values.

    The function does not fill missing economic observations
    or infer release dates. Those decisions belong to the
    provider-specific ingestion layer.
    """

    missing_columns = [column for column in MACRO_COLUMNS if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Macro data are missing required columns: {missing_columns}")

    result = frame.copy()

    result["observation_date"] = pd.to_datetime(
        result["observation_date"],
        errors="coerce",
        utc=True,
    )

    result["available_at"] = pd.to_datetime(
        result["available_at"],
        errors="coerce",
        utc=True,
    )

    result["vintage_date"] = pd.to_datetime(
        result["vintage_date"],
        errors="coerce",
        utc=True,
    )

    result["value"] = pd.to_numeric(
        result["value"],
        errors="coerce",
    )

    result = result[MACRO_COLUMNS]

    validate_macro_frame(result)

    return result.sort_values(
        [
            "series_id",
            "observation_date",
            "vintage_date",
        ],
        na_position="last",
    ).reset_index(drop=True)


def validate_macro_frame(
    frame: pd.DataFrame,
) -> None:
    """
    Validate canonical macroeconomic observations.

    available_at is mandatory because point-in-time safety
    depends on knowing when an observation became available.

    vintage_date may be missing for providers that do not
    expose revision history.
    """

    missing_columns = [column for column in MACRO_COLUMNS if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Macro data are missing required columns: {missing_columns}")

    if frame.empty:
        raise ValueError("Macro data cannot be empty.")

    required_non_null_columns = [
        "observation_date",
        "available_at",
        "series_id",
        "provider_series_id",
        "value",
        "frequency",
        "units",
        "source",
    ]

    columns_with_missing_values = [
        column for column in required_non_null_columns if frame[column].isna().any()
    ]

    if columns_with_missing_values:
        raise ValueError(
            f"Macro data contain missing values in required columns: {columns_with_missing_values}"
        )

    string_columns = [
        "series_id",
        "provider_series_id",
        "frequency",
        "units",
        "source",
    ]

    for column in string_columns:
        empty_strings = frame[column].astype(str).str.strip().eq("")

        if empty_strings.any():
            raise ValueError(f"Macro data contain empty values in {column}.")

    duplicate_columns = [
        "series_id",
        "observation_date",
        "vintage_date",
        "source",
    ]

    if frame.duplicated(
        subset=duplicate_columns,
    ).any():
        raise ValueError(
            "Macro data contain duplicate "
            "(series_id, observation_date, "
            "vintage_date, source) observations."
        )
