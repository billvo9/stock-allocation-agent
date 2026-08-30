import pandas as pd

from stock_agent.data.rates.schema import (
    validate_rate_frame,
)


def merge_rate_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            existing,
            incoming,
        ],
        ignore_index=True,
    )

    combined = (
        combined.drop_duplicates(
            subset=[
                "date",
                "rate_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "date",
                "rate_id",
            ]
        )
        .reset_index(drop=True)
    )

    validate_rate_frame(combined)

    return combined
