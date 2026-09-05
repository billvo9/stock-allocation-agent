from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)


def _make_quarterly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "MU",
                "provider_symbol": "MU",
                "period_end": pd.Timestamp("2026-05-31T00:00:00Z"),
                "available_at": pd.Timestamp("2026-06-27T00:00:00Z"),
                "retrieved_at": pd.Timestamp("2026-09-04T12:00:00Z"),
                "revenue": 9_300_000_000.0,
                "gross_profit": 3_700_000_000.0,
                "operating_income": 2_400_000_000.0,
                "net_income": 2_000_000_000.0,
                "diluted_eps": 1.80,
                "diluted_average_shares": (1_110_000_000.0),
                "total_assets": 70_000_000_000.0,
                "total_debt": 14_000_000_000.0,
                "cash_and_cash_equivalents": (8_000_000_000.0),
                "inventory": 7_000_000_000.0,
                "stockholders_equity": (50_000_000_000.0),
                "operating_cash_flow": (4_000_000_000.0),
                "capital_expenditure": (-3_000_000_000.0),
                "free_cash_flow": 1_000_000_000.0,
                "filing_date": pd.Timestamp("2026-06-26T00:00:00Z"),
                "sec_form_type": "10-Q",
                "sec_accession_number": None,
                "availability_source": ("sec_filing_date_plus_1d"),
                "currency": "USD",
                "source": "yfinance",
            }
        ],
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


def test_valid_quarterly_frame_passes():
    frame = _make_quarterly_frame()

    validate_quarterly_fundamental_frame(frame)


def test_optional_accounting_values_may_be_missing():
    frame = _make_quarterly_frame()

    frame.loc[0, "inventory"] = None
    frame.loc[0, "free_cash_flow"] = None

    validate_quarterly_fundamental_frame(frame)


def test_available_at_may_be_missing():
    frame = _make_quarterly_frame()

    frame.loc[0, "available_at"] = pd.NaT
    frame.loc[
        0,
        "availability_source",
    ] = None

    validate_quarterly_fundamental_frame(frame)


def test_missing_period_end_fails():
    frame = _make_quarterly_frame()

    frame.loc[0, "period_end"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="period_end",
    ):
        validate_quarterly_fundamental_frame(frame)


def test_available_at_cannot_precede_period_end():
    frame = _make_quarterly_frame()

    frame.loc[
        0,
        "available_at",
    ] = pd.Timestamp("2026-05-30T00:00:00Z")

    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        validate_quarterly_fundamental_frame(frame)


def test_available_at_cannot_follow_retrieved_at():
    frame = _make_quarterly_frame()

    frame.loc[
        0,
        "available_at",
    ] = pd.Timestamp("2026-10-01T00:00:00Z")

    with pytest.raises(
        ValueError,
        match="later than",
    ):
        validate_quarterly_fundamental_frame(frame)


def test_invalid_numeric_value_fails():
    frame = _make_quarterly_frame()

    frame["revenue"] = frame["revenue"].astype(object)

    frame.loc[
        0,
        "revenue",
    ] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="revenue",
    ):
        validate_quarterly_fundamental_frame(frame)


def test_duplicate_quarterly_vintage_fails():
    frame = _make_quarterly_frame()

    duplicate = pd.concat(
        [
            frame,
            frame,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_quarterly_fundamental_frame(duplicate)
