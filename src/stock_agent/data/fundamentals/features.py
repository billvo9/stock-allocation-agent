from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.data.fundamentals.quarterly_schema import (
    validate_quarterly_fundamental_frame,
)

QUARTERLY_FEATURE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "debt_to_assets",
    "revenue_growth_yoy",
    "eps_growth_yoy",
]

YOY_PERIOD_MATCH_TOLERANCE_DAYS = 16


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide while treating zero denominators as unavailable."""

    numerator_values = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator_values = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    valid = numerator_values.notna() & denominator_values.notna() & denominator_values.ne(0)

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=float,
    )

    result.loc[valid] = numerator_values.loc[valid] / denominator_values.loc[valid]

    return result


def _calculate_yoy_growth(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.Series:
    working = frame[
        [
            "symbol",
            "period_end",
            value_column,
        ]
    ].copy()

    working["period_end"] = pd.to_datetime(
        working["period_end"],
        utc=True,
    )

    # Preserve the exact row position so the result
    # can be returned in the same order as `frame`.
    working["_row_id"] = range(len(working))

    current = working.copy()

    current["comparison_date"] = current["period_end"] - pd.DateOffset(years=1)

    prior = working.rename(
        columns={
            "period_end": "prior_period_end",
            value_column: "prior_value",
        }
    )

    matched_frames: list[pd.DataFrame] = []

    for symbol, group in current.groupby(
        "symbol",
        sort=False,
    ):
        current_symbol = group.sort_values("comparison_date").copy()

        prior_symbol = prior[prior["symbol"] == symbol].sort_values("prior_period_end").copy()

        matched = pd.merge_asof(
            current_symbol,
            prior_symbol[
                [
                    "prior_period_end",
                    "prior_value",
                ]
            ],
            left_on="comparison_date",
            right_on="prior_period_end",
            direction="nearest",
            tolerance=pd.Timedelta(days=YOY_PERIOD_MATCH_TOLERANCE_DAYS),
        )

        matched_frames.append(matched)

    matched = (
        pd.concat(
            matched_frames,
            ignore_index=True,
        )
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    growth = _safe_ratio(
        matched[value_column] - matched["prior_value"],
        matched["prior_value"].abs(),
    )

    # Guarantee the returned Series has exactly
    # the same index as the caller's DataFrame.
    return pd.Series(
        growth.to_numpy(),
        index=frame.index,
        dtype=float,
    )


def build_quarterly_fundamental_features(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add model-ready features to canonical quarterly fundamentals.

    Features are calculated before daily point-in-time alignment so
    quarterly lags retain their accounting-period meaning.
    """

    validate_quarterly_fundamental_frame(fundamentals)

    result = fundamentals.copy()

    result["period_end"] = pd.to_datetime(
        result["period_end"],
        utc=True,
    )

    result = result.sort_values(
        [
            "symbol",
            "period_end",
            "retrieved_at",
        ]
    ).reset_index(drop=True)

    result["gross_margin"] = _safe_ratio(
        result["gross_profit"],
        result["revenue"],
    )

    result["operating_margin"] = _safe_ratio(
        result["operating_income"],
        result["revenue"],
    )

    result["fcf_margin"] = _safe_ratio(
        result["free_cash_flow"],
        result["revenue"],
    )

    result["debt_to_assets"] = _safe_ratio(
        result["total_debt"],
        result["total_assets"],
    )

    result["revenue_growth_yoy"] = _calculate_yoy_growth(
        result,
        "revenue",
    )

    result["eps_growth_yoy"] = _calculate_yoy_growth(
        result,
        "diluted_eps",
    )

    return result
