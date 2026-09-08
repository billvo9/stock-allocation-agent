from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

PANEL_KEY_COLUMNS = [
    "date",
    "symbol",
]


def _prepare_asset_date_frame(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    """
    Validate and normalize one asset-date frame.

    Every row must represent one unique symbol-date
    observation.
    """

    missing_columns = [column for column in PANEL_KEY_COLUMNS if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")

    if frame.empty:
        raise ValueError(f"{frame_name} cannot be empty.")

    result = frame.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    )

    if result["date"].isna().any():
        raise ValueError(f"{frame_name} contains invalid dates.")

    if result["symbol"].isna().any():
        raise ValueError(f"{frame_name} contains missing symbols.")

    result["symbol"] = result["symbol"].astype(str).str.strip()

    if result["symbol"].eq("").any():
        raise ValueError(f"{frame_name} contains blank symbols.")

    if result.duplicated(subset=PANEL_KEY_COLUMNS).any():
        raise ValueError(f"{frame_name} contains duplicate (date, symbol) rows.")

    return result


def _validate_point_in_time_columns(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    """
    Validate point-in-time availability when present.

    If a frame contains `available_at`, the information
    must not become available after the asset-date row.
    """

    if "available_at" not in frame.columns:
        return frame

    result = frame.copy()

    original_available_at = result["available_at"].copy()

    result["available_at"] = pd.to_datetime(
        result["available_at"],
        errors="coerce",
        utc=True,
    )

    invalid_available_at = original_available_at.notna() & result["available_at"].isna()

    if invalid_available_at.any():
        raise ValueError(f"{frame_name} contains invalid available_at values.")

    future_information = result["available_at"].notna() & (result["available_at"] > result["date"])

    if future_information.any():
        raise ValueError(f"{frame_name} contains future information.")

    return result


def build_dynamic_feature_panel(
    market_features: pd.DataFrame,
    feature_frames: (Mapping[str, pd.DataFrame] | None) = None,
) -> pd.DataFrame:
    """
    Build a model-ready unbalanced asset-date panel.

    The market feature frame defines the master set of
    asset-date observations.

    Supplemental feature frames are left-joined onto
    those observations.

    Assets are never required to share a common
    historical start date. Therefore adding a newly
    listed asset does not truncate the histories of
    older assets.
    """

    result = _prepare_asset_date_frame(
        market_features,
        frame_name="market_features",
    )

    result = _validate_point_in_time_columns(
        result,
        frame_name="market_features",
    )

    if feature_frames is None:
        feature_frames = {}

    for (
        frame_name,
        feature_frame,
    ) in feature_frames.items():
        prepared = _prepare_asset_date_frame(
            feature_frame,
            frame_name=frame_name,
        )

        prepared = _validate_point_in_time_columns(
            prepared,
            frame_name=frame_name,
        )

        overlapping_columns = [
            column
            for column in prepared.columns
            if (column not in PANEL_KEY_COLUMNS and column in result.columns)
        ]

        if overlapping_columns:
            raise ValueError(
                f"{frame_name} would overwrite existing panel columns: {overlapping_columns}"
            )

        row_count_before = len(result)

        result = result.merge(
            prepared,
            on=PANEL_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if len(result) != row_count_before:
            raise RuntimeError("Feature merge changed the number of market observations.")

    return result.sort_values(
        PANEL_KEY_COLUMNS[::-1],
        kind="stable",
    ).reset_index(drop=True)
