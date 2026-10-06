from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.features import (
    build_quarterly_fundamental_features,
)
from stock_agent.data.fundamentals.point_in_time import (
    align_quarterly_fundamentals_asof,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)


def _fundamental_row(
    symbol: str,
    period_end: str,
    available_at: str,
    revenue: float,
    gross_profit: float = 400.0,
    retrieved_at: str = "2026-09-04T12:00:00Z",
) -> pd.DataFrame:
    available_timestamp = pd.Timestamp(available_at)

    filing_date = available_timestamp - pd.Timedelta(days=1)

    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "provider_symbol": symbol,
                "period_end": pd.Timestamp(period_end),
                "available_at": (available_timestamp),
                "retrieved_at": pd.Timestamp(retrieved_at),
                "revenue": revenue,
                "gross_profit": gross_profit,
                "operating_income": 250.0,
                "net_income": 200.0,
                "diluted_eps": 2.0,
                "diluted_average_shares": 100.0,
                "total_assets": 5000.0,
                "total_debt": 1000.0,
                "cash_and_cash_equivalents": 500.0,
                "inventory": 300.0,
                "stockholders_equity": 3000.0,
                "operating_cash_flow": 350.0,
                "capital_expenditure": -150.0,
                "free_cash_flow": 200.0,
                "filing_date": filing_date,
                "sec_form_type": "10-Q",
                "sec_accession_number": None,
                "availability_source": ("sec_filing_date_plus_1d"),
                "currency": "USD",
                "source": "yfinance",
            }
        ],
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


def _market_frame(
    dates: list[str],
    symbol: str = "MU",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates,
                utc=True,
            ),
            "symbol": symbol,
        }
    )


def test_future_fundamentals_are_not_used():
    market = _market_frame(
        [
            "2026-06-26",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == False
    )

    assert pd.isna(
        result.loc[
            0,
            "revenue",
        ]
    )


def test_report_is_available_exactly_on_available_at():
    market = _market_frame(
        [
            "2026-06-27",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == True
    )

    assert result.loc[
        0,
        "revenue",
    ] == pytest.approx(1000.0)


def test_latest_prior_report_is_carried_forward():
    market = _market_frame(
        [
            "2026-06-26",
            "2026-07-15",
        ]
    )

    q1 = _fundamental_row(
        symbol="MU",
        period_end="2026-02-28",
        available_at="2026-03-20",
        revenue=800.0,
    )

    q2 = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    fundamentals = pd.concat(
        [
            q1,
            q2,
        ],
        ignore_index=True,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert result.loc[
        0,
        "revenue",
    ] == pytest.approx(800.0)

    assert result.loc[
        1,
        "revenue",
    ] == pytest.approx(1000.0)


def test_asset_with_no_prior_report_does_not_crash():
    market = _market_frame(
        [
            "2026-01-15",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-02-28",
        available_at="2026-03-20",
        revenue=800.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "fundamental_available",
        ]
        == False
    )

    assert pd.isna(
        result.loc[
            0,
            "fundamental_age_days",
        ]
    )


def test_multiple_symbols_align_independently():
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-06-27",
                    "2026-06-27",
                ],
                utc=True,
            ),
            "symbol": [
                "MU",
                "NVDA",
            ],
        }
    )

    mu = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    nvda = _fundamental_row(
        symbol="NVDA",
        period_end="2026-04-30",
        available_at="2026-05-22",
        revenue=2000.0,
    )

    fundamentals = pd.concat(
        [
            mu,
            nvda,
        ],
        ignore_index=True,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    mu_result = result[result["symbol"] == "MU"].iloc[0]

    nvda_result = result[result["symbol"] == "NVDA"].iloc[0]

    assert mu_result["revenue"] == pytest.approx(1000.0)

    assert nvda_result["revenue"] == pytest.approx(2000.0)


def test_duplicate_symbol_availability_raises():
    # Re-keyed to (symbol, period_end, available_at): two versions of the
    # SAME fiscal period at the same instant cannot be told apart.
    fundamentals = pd.concat(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2026-05-31",
                available_at="2026-06-27",
                revenue=100.0,
                retrieved_at="2026-09-04T12:00:00Z",
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-05-31",
                available_at="2026-06-27",
                revenue=120.0,
                retrieved_at="2026-09-05T12:00:00Z",
            ),
        ],
        ignore_index=True,
    )

    market = _market_frame(
        [
            "2026-06-27",
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"\(symbol, period_end, available_at\)",
    ):
        align_quarterly_fundamentals_asof(
            market_frame=market,
            fundamentals=fundamentals,
        )


def test_aligned_fundamentals_never_come_from_future():
    market = _market_frame(
        [
            "2026-06-26",
            "2026-06-27",
            "2026-07-15",
        ]
    )

    fundamentals = _fundamental_row(
        symbol="MU",
        period_end="2026-05-31",
        available_at="2026-06-27",
        revenue=1000.0,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=fundamentals,
    )

    assert (result["available_at"].isna() | (result["available_at"] <= result["date"])).all()


def test_derived_features_follow_point_in_time_availability():
    fundamentals = pd.concat(
        [
            _fundamental_row(
                symbol="MU",
                period_end="2025-05-31",
                available_at="2025-06-27",
                revenue=100.0,
                gross_profit=40.0,
            ),
            _fundamental_row(
                symbol="MU",
                period_end="2026-05-31",
                available_at="2026-06-27",
                revenue=120.0,
                gross_profit=60.0,
            ),
        ],
        ignore_index=True,
    )

    featured = build_quarterly_fundamental_features(fundamentals)

    market = _market_frame(
        [
            "2026-06-26",
            "2026-06-27",
        ]
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=featured,
    )

    before = result.iloc[0]
    after = result.iloc[1]

    assert before["gross_margin"] == pytest.approx(0.40)

    assert after["gross_margin"] == pytest.approx(0.50)

    assert after["revenue_growth_yoy"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Current-state change log (versioned EDGAR history)
# ---------------------------------------------------------------------------


def _versions(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.concat(
        [
            _fundamental_row(
                symbol="MU",
                period_end=period_end,
                available_at=available_at,
                revenue=revenue,
            )
            for period_end, available_at, revenue in rows
        ],
        ignore_index=True,
    )


def _versioned_fy25() -> pd.DataFrame:
    # Mirrors the versioned MU history: Q2 (period 2025-02-27), Q3
    # (2025-05-29) with a late Q2/A and the Q3 re-version at the same
    # instant, then Q3/A.
    return _versions(
        ("2025-02-27", "2025-03-26T20:00Z", 7300.0),
        ("2025-05-29", "2025-06-27T20:00Z", 9000.0),
        ("2025-02-27", "2025-07-15T20:00Z", 7100.0),
        ("2025-05-29", "2025-07-15T20:00Z", 9200.0),
        ("2025-05-29", "2025-08-01T20:00Z", 9700.0),
    )


_ASOF_DATES = [
    "2025-03-26T19:59Z",
    "2025-03-26T20:00Z",
    "2025-06-27T20:00Z",
    "2025-07-15T19:59Z",
    "2025-07-15T20:00Z",
    "2025-07-31T00:00Z",
    "2025-08-01T20:00Z",
    "2025-09-01T00:00Z",
]


def test_same_instant_versions_of_different_periods_are_allowed_latest_period_wins():
    result = align_quarterly_fundamentals_asof(
        market_frame=_market_frame(_ASOF_DATES),
        fundamentals=_versioned_fy25(),
    )

    assert result["revenue"].tolist()[1:] == pytest.approx(
        [7300.0, 9000.0, 9000.0, 9200.0, 9200.0, 9700.0, 9700.0]
    )
    assert pd.isna(result["revenue"].iloc[0])
    assert not result["fundamental_available"].iloc[0]


def test_late_prior_period_version_never_becomes_current():
    # A Q2/A alone after Q3 (no same-instant Q3 re-version) must not
    # replace the newer Q3 period as the current row.
    fundamentals = _versions(
        ("2025-02-27", "2025-03-26T20:00Z", 7300.0),
        ("2025-05-29", "2025-06-27T20:00Z", 9000.0),
        ("2025-02-27", "2025-07-15T20:00Z", 7100.0),
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=_market_frame(["2025-07-15T20:00Z", "2025-12-31T00:00Z"]),
        fundamentals=fundamentals,
    )

    assert result["revenue"].tolist() == pytest.approx([9000.0, 9000.0])
    assert result["period_end"].dt.strftime("%Y-%m-%d").tolist() == ["2025-05-29"] * 2
    assert (result["available_at"] == pd.Timestamp("2025-06-27T20:00Z")).all()


def test_symbol_without_fundamentals_gets_missing_values():
    market = pd.concat(
        [_market_frame(["2025-08-01T20:00Z"]), _market_frame(["2025-08-01T20:00Z"], "NVDA")],
        ignore_index=True,
    )

    result = align_quarterly_fundamentals_asof(
        market_frame=market,
        fundamentals=_versioned_fy25(),
    )

    assert result["symbol"].tolist() == ["MU", "NVDA"]
    assert result["revenue"].iloc[0] == pytest.approx(9700.0)
    assert pd.isna(result["revenue"].iloc[1])
    assert result["fundamental_available"].tolist() == [True, False]
    assert pd.isna(result["fundamental_age_days"].iloc[1])


def test_asof_output_is_independent_of_input_row_order():
    nvda = _fundamental_row(
        symbol="NVDA",
        period_end="2025-04-27",
        available_at="2025-05-28T20:00Z",
        revenue=44000.0,
    )
    fundamentals = pd.concat([_versioned_fy25(), nvda], ignore_index=True)

    market = pd.concat(
        [_market_frame(_ASOF_DATES), _market_frame(_ASOF_DATES, "NVDA")],
        ignore_index=True,
    )

    baseline = align_quarterly_fundamentals_asof(market_frame=market, fundamentals=fundamentals)

    for seed in (0, 1, 2, 3):
        shuffled_market = market.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        shuffled_fundamentals = fundamentals.sample(frac=1.0, random_state=seed + 10)

        result = align_quarterly_fundamentals_asof(
            market_frame=shuffled_market,
            fundamentals=shuffled_fundamentals.reset_index(drop=True),
        )

        # Output follows the market row order; compare in a canonical order.
        restored = result.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
        expected = baseline.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

        pd.testing.assert_frame_equal(restored, expected)
        assert result[["date", "symbol"]].equals(shuffled_market[["date", "symbol"]])
