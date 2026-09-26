from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_periods import (
    derive_quarter_from_ytd,
    find_period_columns,
    get_direct_quarter_value,
    get_instant_value,
    get_ytd_value,
    parse_edgar_period_column,
)


def test_parse_quarter_period_column():
    result = parse_edgar_period_column("2026-05-28 (Q3)")

    assert result is not None
    assert result.period_end == pd.Timestamp(
        "2026-05-28",
        tz="UTC",
    )
    assert result.period_label == "Q3"


def test_parse_ytd_period_column():
    result = parse_edgar_period_column("2026-05-28 (YTD)")

    assert result is not None
    assert result.period_label == "YTD"


def test_parse_instant_period_column():
    result = parse_edgar_period_column("2026-05-28")

    assert result is not None
    assert result.period_label is None


def test_non_period_column_returns_none():
    assert parse_edgar_period_column("standard_concept") is None


def test_find_period_columns_ignores_metadata():
    frame = pd.DataFrame(
        columns=[
            "concept",
            "label",
            "2026-05-28 (Q3)",
            "2026-05-28 (YTD)",
        ]
    )

    result = find_period_columns(frame)

    assert len(result) == 2


def test_get_direct_quarter_value():
    row = pd.Series(
        {
            "concept": "example",
            "2026-05-28 (Q3)": 41.456e9,
            "2026-05-28 (YTD)": 78.959e9,
        }
    )

    result = get_direct_quarter_value(
        row=row,
        period_end="2026-05-28",
        fiscal_quarter=3,
    )

    assert result == pytest.approx(41.456e9)


def test_get_direct_quarter_does_not_use_ytd():
    row = pd.Series(
        {
            "2026-05-28 (YTD)": 78.959e9,
        }
    )

    result = get_direct_quarter_value(
        row=row,
        period_end="2026-05-28",
        fiscal_quarter=3,
    )

    assert result is None


def test_get_ytd_value():
    row = pd.Series(
        {
            "2026-05-28 (Q3)": 41.456e9,
            "2026-05-28 (YTD)": 78.959e9,
        }
    )

    result = get_ytd_value(
        row=row,
        period_end="2026-05-28",
    )

    assert result == pytest.approx(78.959e9)


def test_get_instant_value():
    row = pd.Series(
        {
            "2026-05-28": 100.0,
            "2025-08-28": 90.0,
        }
    )

    result = get_instant_value(
        row=row,
        period_end="2026-05-28",
    )

    assert result == pytest.approx(100.0)


def test_q1_ytd_is_same_as_quarter():
    result = derive_quarter_from_ytd(
        current_ytd=10.0,
        previous_ytd=None,
        fiscal_quarter=1,
    )

    assert result == pytest.approx(10.0)


def test_q2_is_current_ytd_minus_q1_ytd():
    result = derive_quarter_from_ytd(
        current_ytd=23.0,
        previous_ytd=10.0,
        fiscal_quarter=2,
    )

    assert result == pytest.approx(13.0)


def test_q3_is_current_ytd_minus_q2_ytd():
    result = derive_quarter_from_ytd(
        current_ytd=40.0,
        previous_ytd=23.0,
        fiscal_quarter=3,
    )

    assert result == pytest.approx(17.0)


def test_q2_requires_previous_ytd():
    with pytest.raises(
        ValueError,
        match="Previous YTD value",
    ):
        derive_quarter_from_ytd(
            current_ytd=23.0,
            previous_ytd=None,
            fiscal_quarter=2,
        )


def test_q4_is_not_generically_derived_from_ytd():
    with pytest.raises(
        ValueError,
        match="Q1 through Q3",
    ):
        derive_quarter_from_ytd(
            current_ytd=100.0,
            previous_ytd=75.0,
            fiscal_quarter=4,
        )
