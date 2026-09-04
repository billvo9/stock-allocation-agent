from __future__ import annotations

import pandas as pd

from stock_agent.data.fundamentals.schema import (
    FUNDAMENTAL_SNAPSHOT_COLUMNS,
    METADATA_COLUMNS,
    validate_fundamental_snapshot_frame,
    validate_metadata_frame,
)


def merge_metadata(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge instrument metadata.

    Keeps only the most recently retrieved row
    for each symbol.
    """

    if incoming.empty:
        if existing.empty:
            return pd.DataFrame(columns=METADATA_COLUMNS)

        validate_metadata_frame(existing)

        return existing[METADATA_COLUMNS].copy().sort_values("symbol").reset_index(drop=True)

    validate_metadata_frame(incoming)

    if existing.empty:
        combined = incoming.copy()
    else:
        validate_metadata_frame(existing)

        combined = pd.concat(
            [
                existing,
                incoming,
            ],
            ignore_index=True,
        )

    combined["retrieved_at"] = pd.to_datetime(
        combined["retrieved_at"],
        utc=True,
    )

    combined = (
        combined.sort_values(
            [
                "symbol",
                "retrieved_at",
            ]
        )
        .drop_duplicates(
            subset=["symbol"],
            keep="last",
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    combined = combined[METADATA_COLUMNS]

    validate_metadata_frame(combined)

    return combined


def merge_snapshot_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge point-in-time fundamental snapshots.

    Preserves historical snapshots and removes
    duplicate (symbol, retrieved_at) observations.
    """

    if incoming.empty:
        if existing.empty:
            return pd.DataFrame(columns=(FUNDAMENTAL_SNAPSHOT_COLUMNS))

        validate_fundamental_snapshot_frame(existing)

        return (
            existing[FUNDAMENTAL_SNAPSHOT_COLUMNS]
            .copy()
            .sort_values(
                [
                    "symbol",
                    "retrieved_at",
                ]
            )
            .reset_index(drop=True)
        )

    validate_fundamental_snapshot_frame(incoming)

    if existing.empty:
        combined = incoming.copy()
    else:
        validate_fundamental_snapshot_frame(existing)

        combined = pd.concat(
            [
                existing,
                incoming,
            ],
            ignore_index=True,
        )

    combined["retrieved_at"] = pd.to_datetime(
        combined["retrieved_at"],
        utc=True,
    )

    combined = (
        combined.sort_values(
            [
                "symbol",
                "retrieved_at",
            ]
        )
        .drop_duplicates(
            subset=[
                "symbol",
                "retrieved_at",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    combined = combined[FUNDAMENTAL_SNAPSHOT_COLUMNS]

    validate_fundamental_snapshot_frame(combined)

    return combined
