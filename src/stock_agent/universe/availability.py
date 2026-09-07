from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

MISSING_TEXT_TOKENS = frozenset(
    {
        "",
        ".",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
    }
)

SUMMARY_COLUMNS = [
    "available_feature_count",
    "available_feature_fraction",
    "any_requested_feature_available",
    "all_requested_features_available",
]


def _feature_is_available(
    series: pd.Series,
) -> pd.Series:
    """
    Return whether each feature value is usable.

    Native missing values such as None, NaN, and NaT
    are unavailable.

    Common provider text placeholders such as
    '.', 'N/A', and blank strings are also treated
    as unavailable.
    """

    available = series.notna()

    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        normalized = series.astype("string").str.strip().str.casefold()

        available &= ~normalized.isin(MISSING_TEXT_TOKENS)

    return available.astype(bool)


def build_feature_availability(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Add row-level feature availability indicators.

    This function does not determine whether an asset
    is investable. It only describes which requested
    information is available for each asset-date row.

    The input frame is expected to have already been
    point-in-time aligned.
    """

    requested_features = list(feature_columns)

    if not requested_features:
        raise ValueError("feature_columns cannot be empty.")

    if len(requested_features) != len(set(requested_features)):
        raise ValueError("feature_columns contains duplicates.")

    missing_columns = [feature for feature in requested_features if feature not in frame.columns]

    if missing_columns:
        raise ValueError(f"Availability frame is missing requested features: {missing_columns}")

    availability_columns = [f"{feature}_available" for feature in requested_features]

    generated_columns = availability_columns + SUMMARY_COLUMNS

    collisions = [column for column in generated_columns if column in frame.columns]

    if collisions:
        raise ValueError(f"Availability output would overwrite existing columns: {collisions}")

    result = frame.copy()

    for feature, availability_column in zip(
        requested_features,
        availability_columns,
        strict=True,
    ):
        result[availability_column] = _feature_is_available(result[feature])

    result["available_feature_count"] = result[availability_columns].sum(axis=1).astype(int)

    feature_count = len(requested_features)

    result["available_feature_fraction"] = result["available_feature_count"] / feature_count

    result["any_requested_feature_available"] = result["available_feature_count"] > 0

    result["all_requested_features_available"] = result["available_feature_count"] == feature_count

    return result
