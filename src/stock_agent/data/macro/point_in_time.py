from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from stock_agent.data.macro.schema import validate_macro_frame

MACRO_STATE_COLUMNS = [
    "date",
    "series_id",
    "provider_series_id",
    "value",
    "observation_date",
    "available_at",
    "vintage_date",
    "frequency",
    "units",
    "source",
    "macro_available",
]

_STATIC_METADATA_COLUMNS = [
    "provider_series_id",
    "frequency",
    "units",
    "source",
]


def _normalize_decision_dates(
    decision_dates: Iterable[object],
) -> pd.DatetimeIndex:
    """
    Convert requested decision dates to unique UTC calendar dates.

    Duplicate dates are intentionally collapsed because macro state
    is identical for every asset evaluated on the same date.
    """

    values = list(decision_dates)

    if not values:
        raise ValueError("At least one decision date is required.")

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed).any():
        raise ValueError("Decision dates contain invalid values.")

    normalized = pd.DatetimeIndex(parsed).normalize()

    return normalized.unique().sort_values()


def _extract_series_metadata(
    frame: pd.DataFrame,
    series_id: str,
) -> dict[str, object]:
    """
    Extract static metadata for one macro series.

    Provider identity, frequency, units, and source should not vary
    inside the same canonical series.
    """

    metadata: dict[str, object] = {}

    for column in _STATIC_METADATA_COLUMNS:
        unique_values = frame[column].drop_duplicates()

        if len(unique_values) != 1:
            raise ValueError(f"Macro series {series_id!r} has inconsistent {column!r} values.")

        metadata[column] = unique_values.iloc[0]

    return metadata


def _unavailable_state_row(
    decision_date: pd.Timestamp,
    series_id: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Create a state row before any observation is available."""

    return {
        "date": decision_date,
        "series_id": series_id,
        "provider_series_id": (metadata["provider_series_id"]),
        "value": float("nan"),
        "observation_date": pd.NaT,
        "available_at": pd.NaT,
        "vintage_date": pd.NaT,
        "frequency": metadata["frequency"],
        "units": metadata["units"],
        "source": metadata["source"],
        "macro_available": False,
    }


def _available_state_row(
    decision_date: pd.Timestamp,
    row: dict[str, object],
) -> dict[str, object]:
    """Convert one historical vintage into an as-of state row."""

    return {
        "date": decision_date,
        "series_id": row["series_id"],
        "provider_series_id": (row["provider_series_id"]),
        "value": row["value"],
        "observation_date": (row["observation_date"]),
        "available_at": row["available_at"],
        "vintage_date": row["vintage_date"],
        "frequency": row["frequency"],
        "units": row["units"],
        "source": row["source"],
        "macro_available": True,
    }


def build_macro_state_asof(
    decision_dates: Iterable[object],
    macro_vintages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe macroeconomic state for each decision date.

    For each macro series and decision date:

    1. Only vintages with ``available_at <= decision_date`` may
       be considered.
    2. Among observations known at that time, use the latest
       economic observation.
    3. If that observation has multiple revisions already known,
       use the latest available revision.
    4. Preserve dates before the first release with
       ``macro_available=False`` rather than dropping them.

    Decision dates use an end-of-day convention. A macro observation
    whose ``available_at`` date equals the decision date is therefore
    considered available on that date.

    The result remains long-form. Feature engineering and wide-panel
    construction happen downstream.
    """

    validate_macro_frame(macro_vintages)

    dates = _normalize_decision_dates(decision_dates)

    frame = macro_vintages.copy()

    for column in [
        "observation_date",
        "available_at",
        "vintage_date",
    ]:
        frame[column] = pd.to_datetime(
            frame[column],
            errors="raise",
            utc=True,
        ).dt.normalize()

    rows: list[dict[str, object]] = []

    for series_id, series_frame in frame.groupby(
        "series_id",
        sort=True,
    ):
        metadata = _extract_series_metadata(
            series_frame,
            series_id,
        )

        events = (
            series_frame.sort_values(
                [
                    "available_at",
                    "vintage_date",
                    "observation_date",
                ]
            )
            .reset_index(drop=True)
            .to_dict("records")
        )

        latest_vintage_by_observation: dict[
            pd.Timestamp,
            dict[str, object],
        ] = {}

        latest_observation_date: pd.Timestamp | None = None

        event_index = 0

        for decision_date in dates:
            while (
                event_index < len(events) and events[event_index]["available_at"] <= decision_date
            ):
                event = events[event_index]

                observation_date = pd.Timestamp(event["observation_date"])

                # Because events are ordered by availability/vintage,
                # assigning here replaces an older revision with the
                # newest revision known so far.
                latest_vintage_by_observation[observation_date] = event

                if latest_observation_date is None or observation_date > latest_observation_date:
                    latest_observation_date = observation_date

                event_index += 1

            if latest_observation_date is None:
                rows.append(
                    _unavailable_state_row(
                        decision_date=decision_date,
                        series_id=series_id,
                        metadata=metadata,
                    )
                )
                continue

            current = latest_vintage_by_observation[latest_observation_date]

            rows.append(
                _available_state_row(
                    decision_date=decision_date,
                    row=current,
                )
            )

    result = pd.DataFrame(
        rows,
        columns=MACRO_STATE_COLUMNS,
    )

    if result.empty:
        return result

    result = result.sort_values(
        [
            "date",
            "series_id",
        ]
    ).reset_index(drop=True)

    return result
