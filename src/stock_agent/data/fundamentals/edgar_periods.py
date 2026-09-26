from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_PERIOD_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?: \((?P<label>Q[1-4]|YTD|FY)\))?$"
)


@dataclass(frozen=True)
class EdgarPeriodColumn:
    column: str
    period_end: pd.Timestamp
    period_label: str | None


def parse_edgar_period_column(
    column: object,
) -> EdgarPeriodColumn | None:
    """
    Parse an EdgarTools statement-value column.

    Examples:
        2026-05-28 (Q3)
        2026-05-28 (YTD)
        2026-05-28

    Non-period metadata columns return None.
    """

    if not isinstance(column, str):
        return None

    match = _PERIOD_PATTERN.fullmatch(column)

    if match is None:
        return None

    period_end = pd.Timestamp(
        match.group("date"),
        tz="UTC",
    )

    return EdgarPeriodColumn(
        column=column,
        period_end=period_end,
        period_label=match.group("label"),
    )


def find_period_columns(
    frame: pd.DataFrame,
) -> list[EdgarPeriodColumn]:
    """Return parsed financial-period columns."""

    periods: list[EdgarPeriodColumn] = []

    for column in frame.columns:
        parsed = parse_edgar_period_column(column)

        if parsed is not None:
            periods.append(parsed)

    return periods


def get_direct_quarter_value(
    row: pd.Series,
    period_end: str | pd.Timestamp,
    fiscal_quarter: int,
) -> float | None:
    """
    Return a directly reported standalone quarter.

    Example:
        2026-05-28 (Q3)

    This function deliberately does not use YTD values.
    """

    if fiscal_quarter not in {1, 2, 3, 4}:
        raise ValueError("Fiscal quarter must be between 1 and 4.")

    target_date = pd.Timestamp(period_end)

    if target_date.tzinfo is None:
        target_date = target_date.tz_localize("UTC")
    else:
        target_date = target_date.tz_convert("UTC")

    expected_label = f"Q{fiscal_quarter}"

    for column in row.index:
        parsed = parse_edgar_period_column(column)

        if parsed is None:
            continue

        if parsed.period_end == target_date and parsed.period_label == expected_label:
            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                return None

            return float(value)

    return None


def get_ytd_value(
    row: pd.Series,
    period_end: str | pd.Timestamp,
) -> float | None:
    """Return the YTD value for one period end."""

    target_date = pd.Timestamp(period_end)

    if target_date.tzinfo is None:
        target_date = target_date.tz_localize("UTC")
    else:
        target_date = target_date.tz_convert("UTC")

    for column in row.index:
        parsed = parse_edgar_period_column(column)

        if parsed is None:
            continue

        if parsed.period_end == target_date and parsed.period_label == "YTD":
            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                return None

            return float(value)

    return None


def get_instant_value(
    row: pd.Series,
    period_end: str | pd.Timestamp,
) -> float | None:
    """
    Return a point-in-time balance-sheet value.

    Instant columns normally have a date without
    Q/YTD/FY suffix.
    """

    target_date = pd.Timestamp(period_end)

    if target_date.tzinfo is None:
        target_date = target_date.tz_localize("UTC")
    else:
        target_date = target_date.tz_convert("UTC")

    for column in row.index:
        parsed = parse_edgar_period_column(column)

        if parsed is None:
            continue

        if parsed.period_end == target_date and parsed.period_label is None:
            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                return None

            return float(value)

    return None


def derive_quarter_from_ytd(
    current_ytd: float,
    previous_ytd: float | None,
    fiscal_quarter: int,
) -> float:
    """
    Convert cumulative YTD flow data into a
    standalone quarterly value.

    Q1:
        quarter = current YTD

    Q2/Q3:
        quarter = current YTD - previous YTD
    """

    if fiscal_quarter not in {1, 2, 3}:
        raise ValueError("YTD quarter derivation currently supports Q1 through Q3 only.")

    if fiscal_quarter == 1:
        return float(current_ytd)

    if previous_ytd is None:
        raise ValueError("Previous YTD value is required for Q2/Q3.")

    return float(current_ytd - previous_ytd)
