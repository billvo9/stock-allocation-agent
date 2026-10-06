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
    """
    Year-over-year growth using only prior-year rows available at the time.

    Point-in-time rule (prior-year eligibility):

    - A prior-year candidate is eligible for a current row only when it has
      the same symbol, both rows have a known ``available_at``, and
      ``candidate.available_at <= current.available_at``. Unknown (NaT)
      availability can never be proven earlier, so it is never eligible, and
      a current row with unknown availability gets NaN growth.
    - Among eligible candidates, period matching is unchanged: the
      ``period_end`` nearest to ``current.period_end - 1 year`` within
      ``YOY_PERIOD_MATCH_TOLERANCE_DAYS`` (inclusive); an exact distance tie
      resolves to the earlier ``period_end``, as ``merge_asof`` did.
    - If several eligible candidates share the matched ``period_end``, the
      one with the latest ``available_at`` is used. If several share that
      latest ``available_at`` with different values, growth is NaN (no
      guessing); equal values are used.

    Because the prior-year row used is available no later than the current
    row, the current row's ``available_at`` remains the correct availability
    of the derived growth value. The result does not depend on row order.

    Known limitation: a prior-year amendment published after the current row
    does not re-emit growth for that current row; only later versions of the
    current period pick it up.
    """

    working = pd.DataFrame(
        {
            "symbol": frame["symbol"].to_numpy(),
            "period_end": pd.to_datetime(frame["period_end"], utc=True).array,
            "available_at": pd.to_datetime(frame["available_at"], utc=True).array,
            "value": pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float),
            # Positional id so the result maps back to `frame` exactly.
            "_row_id": np.arange(len(frame)),
        }
    )

    current = working[working["available_at"].notna()].copy()
    current["comparison_date"] = current["period_end"] - pd.DateOffset(years=1)

    prior = working[working["available_at"].notna()].rename(
        columns={
            "period_end": "prior_period_end",
            "available_at": "prior_available_at",
            "value": "prior_value",
        }
    )[["symbol", "prior_period_end", "prior_available_at", "prior_value"]]

    pairs = current.merge(prior, on="symbol", how="inner")

    pairs["distance"] = (pairs["prior_period_end"] - pairs["comparison_date"]).abs()

    pairs = pairs[
        (pairs["prior_available_at"] <= pairs["available_at"])
        & (pairs["distance"] <= pd.Timedelta(days=YOY_PERIOD_MATCH_TOLERANCE_DAYS))
    ]

    prior_value = pd.Series(np.nan, index=working["_row_id"], dtype=float)

    if not pairs.empty:
        # Nearest period_end; exact distance ties go to the earlier period_end.
        pairs = pairs.sort_values(["_row_id", "distance", "prior_period_end"])

        matched_period = pairs.groupby("_row_id")["prior_period_end"].transform("first")

        pairs = pairs[pairs["prior_period_end"] == matched_period]

        latest_available = pairs.groupby("_row_id")["prior_available_at"].transform("max")

        pairs = pairs[pairs["prior_available_at"] == latest_available]

        # Same-time versions: one distinct value (NaN counts as a value) is
        # used; conflicting values stay NaN.
        resolved = pairs.groupby("_row_id")["prior_value"].agg(
            lambda values: values.iloc[0] if values.nunique(dropna=False) == 1 else np.nan
        )

        prior_value.loc[resolved.index] = resolved.to_numpy(dtype=float)

    current_value = pd.Series(working["value"].to_numpy(), index=working["_row_id"])

    growth = _safe_ratio(
        current_value - prior_value,
        prior_value.abs(),
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
